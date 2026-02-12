# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Domain randomization utilities for sim-to-real transfer.

Provides vectorised, device-aware helpers for:
- **ActionBuffer** – per-env action delay queue with optional packet loss and additive noise.
- **ObservationBuffer** – per-env observation latency buffer with additive Gaussian noise.
- **ActuatorRandomizer** – per-env randomisation of joint stiffness, damping & friction
  (written to the physics simulation via the Articulation API).
- **DomainRandomizationCfg** – dataclass grouping all tuneable parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch

from isaaclab.assets import Articulation


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DomainRandomizationCfg:
    """All domain-randomisation knobs in one place."""

    enabled: bool = True
    """Master switch – when ``False`` every helper is a transparent pass-through."""

    # -- Action buffer ---------------------------------------------------------
    action_delay_range: tuple[int, int] = (0, 3)
    """Uniform int range [lo, hi] for per-env action delay (in *decimated* steps)."""
    action_noise_std: float = 0.01
    """Std-dev of additive Gaussian noise on the (normalised) actions."""
    packet_loss_prob: float = 0.01
    """Probability that an action packet is dropped (replaced by the previous one)."""

    # -- Observation buffer ----------------------------------------------------
    obs_delay_range: tuple[int, int] = (0, 2)
    """Uniform int range [lo, hi] for per-env observation delay (in *decimated* steps)."""
    obs_noise_std_pos: float = 0.002
    """Observation noise on EE position (m)."""
    obs_noise_std_quat: float = 0.01
    """Observation noise on EE orientation quaternion."""
    obs_noise_std_joint_pos: float = 0.005
    """Observation noise on joint positions (rad)."""
    obs_noise_std_joint_vel: float = 0.01
    """Observation noise on joint velocities (rad/s)."""

    # -- Actuator randomisation ------------------------------------------------
    stiffness_scale_range: tuple[float, float] = (0.9, 1.1)
    """Multiplicative scale range applied to nominal joint stiffness."""
    damping_scale_range: tuple[float, float] = (0.9, 1.1)
    """Multiplicative scale range applied to nominal joint damping."""
    friction_variation: float = 0.2
    """Fractional variation (±) around default per-joint friction coefficients."""
    default_joint_friction: tuple[float, ...] = (8.24, 10.51, 7.9, 1.43, 1.05, 1.63)
    """Nominal static friction for each joint (shoulder_pan → wrist_3)."""


# ---------------------------------------------------------------------------
# ActionBuffer
# ---------------------------------------------------------------------------

class ActionBuffer:
    """Per-environment FIFO action queue with delay, packet-loss & additive noise.

    *Drop policy*: when a packet is lost the **previous** command is held.

    All internal tensors live on ``device``.
    """

    def __init__(
        self,
        num_envs: int,
        action_dim: int,
        cfg: DomainRandomizationCfg,
        device: torch.device | str,
    ) -> None:
        self.num_envs = num_envs
        self.action_dim = action_dim
        self.cfg = cfg
        self.device = torch.device(device)

        max_delay = cfg.action_delay_range[1]
        # buffer_len = max_delay + 1  (index 0 = "no delay")
        self.buffer_len = max_delay + 1

        # Ring-buffer: (num_envs, buffer_len, action_dim)
        self.buffer = torch.zeros(
            num_envs, self.buffer_len, action_dim, device=self.device
        )
        # Per-env write cursor
        self.cursor = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        # Per-env delay (re-sampled at reset)
        self.delay = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        # Last action actually applied (for hold-on-drop)
        self.last_action = torch.zeros(
            num_envs, action_dim, device=self.device
        )

        self._resample_delay()

    # -- public API ----------------------------------------------------------

    def push(self, actions: torch.Tensor) -> torch.Tensor:
        """Write *actions* into the ring-buffer and return the delayed+noisy command.

        Args:
            actions: ``(num_envs, action_dim)`` – clipped normalised actions.

        Returns:
            ``(num_envs, action_dim)`` – actions to actually execute.
        """
        if not self.cfg.enabled:
            return actions
        # print("Original actions: ", actions)
        # --- additive noise ---
        if self.cfg.action_noise_std > 0.0:
            actions = actions + torch.randn_like(actions) * self.cfg.action_noise_std

        # --- write into ring-buffer ---
        idx = self.cursor % self.buffer_len  # (num_envs,)
        # Advanced indexing: each env writes at its own cursor position
        self.buffer[torch.arange(self.num_envs, device=self.device), idx] = actions
        self.cursor += 1

        # --- read delayed action ---
        read_idx = (self.cursor - 1 - self.delay) % self.buffer_len
        delayed = self.buffer[
            torch.arange(self.num_envs, device=self.device), read_idx
        ]

        # --- packet loss (drop → hold previous) ---
        if self.cfg.packet_loss_prob > 0.0:
            drop_mask = (
                torch.rand(self.num_envs, device=self.device) < self.cfg.packet_loss_prob
            )
            delayed = torch.where(
                drop_mask.unsqueeze(-1), self.last_action, delayed
            )

        self.last_action = delayed.clone()
        # print("Delayed noisy actions: ", delayed)
        return delayed

    def reset(self, env_ids: torch.Tensor) -> None:
        """Clear buffer and re-sample delays for the given environments."""
        env_ids = env_ids.to(self.device, dtype=torch.long)
        self.buffer[env_ids] = 0.0
        self.cursor[env_ids] = 0
        self.last_action[env_ids] = 0.0
        self._resample_delay(env_ids)

    # -- internals -----------------------------------------------------------

    def _resample_delay(self, env_ids: torch.Tensor | None = None) -> None:
        lo, hi = self.cfg.action_delay_range
        if env_ids is None:
            self.delay = torch.randint(lo, hi + 1, (self.num_envs,), device=self.device)
        else:
            self.delay[env_ids] = torch.randint(
                lo, hi + 1, (len(env_ids),), device=self.device
            )


# ---------------------------------------------------------------------------
# ObservationBuffer
# ---------------------------------------------------------------------------

class ObservationBuffer:
    """Per-environment observation delay buffer with structured Gaussian noise.

    The noise standard-deviations are applied **per-component** according to the
    observation layout:

        [ee_pos(3), ee_quat(4), ee_lin_vel(3), ee_ang_vel(3),
         joint_pos(N), joint_vel(N), goal_pos(3), goal_quat(4)]

    Goal components are **not** corrupted (the agent knows its own goal exactly).
    """

    def __init__(
        self,
        num_envs: int,
        obs_dim: int,
        num_joints: int,
        cfg: DomainRandomizationCfg,
        device: torch.device | str,
    ) -> None:
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.num_joints = num_joints
        self.cfg = cfg
        self.device = torch.device(device)

        max_delay = cfg.obs_delay_range[1]
        self.buffer_len = max_delay + 1

        # Lazy init: buffer and noise vector are created on first append_and_get
        # so the actual obs tensor shape is used (obs_dim from config may differ).
        self.buffer: torch.Tensor | None = None
        self.cursor = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.delay = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._noise_std: torch.Tensor | None = None

        self._resample_delay()

    # -- public API ----------------------------------------------------------

    def append_and_get(self, obs: torch.Tensor) -> torch.Tensor:
        """Push *obs* and return the delayed, noisy version.

        Args:
            obs: ``(num_envs, obs_dim)``

        Returns:
            ``(num_envs, obs_dim)`` – observation to feed the policy.
        """
        if not self.cfg.enabled:
            return obs

        # Lazy init on first call so we match the real obs shape
        if self.buffer is None:
            actual_dim = obs.shape[1]
            self.obs_dim = actual_dim
            self.buffer = torch.zeros(
                self.num_envs, self.buffer_len, actual_dim, device=self.device
            )
            self._noise_std = self._build_noise_vector()

        idx = self.cursor % self.buffer_len
        self.buffer[torch.arange(self.num_envs, device=self.device), idx] = obs
        self.cursor += 1

        read_idx = (self.cursor - 1 - self.delay) % self.buffer_len
        delayed = self.buffer[
            torch.arange(self.num_envs, device=self.device), read_idx
        ].clone()

        # additive Gaussian noise (goal components left untouched via 0-std)
        noise = torch.randn_like(delayed) * self._noise_std.unsqueeze(0)
        delayed = delayed + noise

        return delayed

    def reset(self, env_ids: torch.Tensor) -> None:
        env_ids = env_ids.to(self.device, dtype=torch.long)
        if self.buffer is not None:
            self.buffer[env_ids] = 0.0
        self.cursor[env_ids] = 0
        self._resample_delay(env_ids)

    # -- internals -----------------------------------------------------------

    def _resample_delay(self, env_ids: torch.Tensor | None = None) -> None:
        lo, hi = self.cfg.obs_delay_range
        if env_ids is None:
            self.delay = torch.randint(lo, hi + 1, (self.num_envs,), device=self.device)
        else:
            self.delay[env_ids] = torch.randint(
                lo, hi + 1, (len(env_ids),), device=self.device
            )

    def _build_noise_vector(self) -> torch.Tensor:
        """Construct a 1-D noise-std tensor aligned with the observation layout.

        Layout (sizes depend on ``num_joints``):
            ee_pos(3) | ee_quat(4) | ee_lin_vel(3) | ee_ang_vel(3) |
            joint_pos(J) | joint_vel(J) | goal_pos(3) | goal_quat(4)
        """
        nj = self.num_joints
        cfg = self.cfg
        parts: list[torch.Tensor] = [
            torch.full((3,), cfg.obs_noise_std_pos, device=self.device),       # ee_pos
            torch.full((4,), cfg.obs_noise_std_quat, device=self.device),      # ee_quat
            torch.full((3,), cfg.obs_noise_std_pos, device=self.device),       # ee_lin_vel (same as pos)
            torch.full((3,), cfg.obs_noise_std_quat, device=self.device),      # ee_ang_vel (same as quat)
            torch.full((nj,), cfg.obs_noise_std_joint_pos, device=self.device),  # joint_pos
            torch.full((nj,), cfg.obs_noise_std_joint_vel, device=self.device),  # joint_vel
            torch.zeros(3, device=self.device),                                # goal_pos  (no noise)
            torch.zeros(4, device=self.device),                                # goal_quat (no noise)
        ]
        vec = torch.cat(parts)
        # Pad or truncate to match obs_dim (safety)
        if vec.shape[0] < self.obs_dim:
            vec = torch.cat([vec, torch.zeros(self.obs_dim - vec.shape[0], device=self.device)])
        return vec[: self.obs_dim]


# ---------------------------------------------------------------------------
# ActuatorRandomizer
# ---------------------------------------------------------------------------

class ActuatorRandomizer:
    """Per-environment randomisation of joint stiffness, damping and friction.

    On ``sample_and_apply`` the randomiser:
    1. Samples multiplicative scales from the configured ranges.
    2. Multiplies the **nominal** values stored at init time.
    3. Writes the result into the physics simulation via the Articulation API.

    Effort limits are intentionally **not** randomised.
    """

    def __init__(
        self,
        robot: Articulation,
        cfg: DomainRandomizationCfg,
        device: torch.device | str,
    ) -> None:
        self.robot = robot
        self.cfg = cfg
        self.device = torch.device(device)
        self.num_envs = robot.num_instances
        self.num_joints = robot.num_joints

        # Snapshot nominal values from the simulation (after scene.clone_environments)
        # Shape: (num_envs, num_joints)
        self.nominal_stiffness = robot.data.default_joint_stiffness.clone().to(self.device)
        self.nominal_damping = robot.data.default_joint_damping.clone().to(self.device)

        # Build nominal friction tensor from user-provided per-joint defaults
        # Expand to (num_envs, num_joints) – pad/truncate if joint count differs
        fric = list(cfg.default_joint_friction)
        if len(fric) < self.num_joints:
            fric.extend([fric[-1]] * (self.num_joints - len(fric)))
        fric_t = torch.tensor(fric[: self.num_joints], dtype=torch.float32, device=self.device)
        self.nominal_friction = fric_t.unsqueeze(0).expand(self.num_envs, -1).clone()

    # -- public API ----------------------------------------------------------

    def sample_and_apply(self, env_ids: torch.Tensor) -> None:
        """Sample new actuator parameters and write them to the physics sim.

        Should be called inside ``_reset_idx`` **after** ``scene.clone_environments``.

        Args:
            env_ids: 1-D long tensor of environment indices to randomise.
        """
        if not self.cfg.enabled:
            return

        env_ids = env_ids.to(self.device, dtype=torch.long)
        n = len(env_ids)
        nj = self.num_joints

        # --- stiffness ---
        s_lo, s_hi = self.cfg.stiffness_scale_range
        stiff_scale = torch.empty(n, nj, device=self.device).uniform_(s_lo, s_hi)
        new_stiffness = self.nominal_stiffness[env_ids] * stiff_scale
        self.robot.write_joint_stiffness_to_sim(new_stiffness, env_ids=env_ids)

        # --- damping ---
        d_lo, d_hi = self.cfg.damping_scale_range
        damp_scale = torch.empty(n, nj, device=self.device).uniform_(d_lo, d_hi)
        new_damping = self.nominal_damping[env_ids] * damp_scale
        self.robot.write_joint_damping_to_sim(new_damping, env_ids=env_ids)

        # --- friction (±variation around defaults) ---
        var = self.cfg.friction_variation
        fric_scale = torch.empty(n, nj, device=self.device).uniform_(1.0 - var, 1.0 + var)
        new_friction = self.nominal_friction[env_ids] * fric_scale
        self.robot.write_joint_friction_coefficient_to_sim(
            new_friction, env_ids=env_ids
        )
