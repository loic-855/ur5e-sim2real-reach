# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Unified domain randomization utilities for sim-to-real transfer (V4).

Consolidates all previous DR versions (v1–v4) into a single module with three
independent toggles in the configuration:

- ``enable_physical_rand`` – stiffness / damping / friction / mass / CoM randomisation.
- ``enable_noise``         – additive Gaussian noise on actions and observations.
- ``enable_delay``         – per-env FIFO delay (+ packet loss) on actions and observations.

**Action flexibility**: ``ActionBuffer`` auto-detects the action layout from
``action_dim`` and ``num_joints``:
- ``action_dim == num_joints``      → position-only mode (uniform noise ``action_noise_std_pos``).
- ``action_dim == 2 * num_joints``  → position + velocity feedforward mode
  (``action_noise_std_pos`` for dims ``[0..nj-1]``, ``action_noise_std_vel`` for
  dims ``[nj..2*nj-1]``).

**Observation layout** (24-dim for 6 arm joints, as used in V4):

    pos_error (3), ori_error (3), joint_pos (6), joint_vel (6),
    tcp_linear_vel (3), tcp_angular_vel (3)

All components are normalised before being fed to the policy; observation noise
stds are specified in physical units and converted to normalised space internally.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from isaaclab.assets import Articulation


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# === UR5e Robot Constants ===
MAX_REACH = 0.85  # UR5e reach ~850mm
MAX_JOINT_VEL = 3.14  # ~180°/s
TCP_MAX_SPEED = 2.0  # m/s, used for TCP linear velocity normalisation
JOINT_LIMITS = [np

@dataclass
class DomainRandomizationV4Cfg:
    """All domain-randomisation knobs for the V4 sim-to-real pipeline.

    Four independent toggles control which randomisation categories are active:
    - ``enable_actuator_rand``   – stiffness / damping / friction randomisation.
    - ``enable_mass_com_rand``   – link mass / center of mass randomisation.
    - ``enable_noise``           – additive Gaussian noise on actions and observations.
    - ``enable_delay``           – per-env FIFO delay (+ packet loss) on actions and observations.
    
    There is no global master switch – disable categories individually as needed.
    All disabled toggles provide zero-overhead pass-through.
    """

    # -------------------------------------------------------------------------
    # Toggle: actuator characteristics (stiffness, damping, friction)
    # -------------------------------------------------------------------------
    enable_actuator_rand: bool = False
    """Enable per-env randomisation of actuator PD gains (stiffness/damping)
    and joint friction coefficients."""

    # -------------------------------------------------------------------------
    # Toggle: link physical properties (masses, centers of mass)
    # -------------------------------------------------------------------------
    enable_mass_com_rand: bool = False
    """Enable per-env randomisation of link masses and centers of mass."""

    # -------------------------------------------------------------------------
    # Toggle: additive Gaussian noise on actions and observations
    # -------------------------------------------------------------------------
    enable_noise: bool = False
    """Enable additive Gaussian noise on actions and on observations."""

    # -------------------------------------------------------------------------
    # Toggle: per-env FIFO delay and packet loss on actions / observations
    # -------------------------------------------------------------------------
    enable_delay: bool = False
    """Enable per-env action delay, action packet-loss and observation delay."""

    # -- Action buffer ---------------------------------------------------------
    action_delay_range: tuple[int, int] = (1, 2)
    """Uniform int range [lo, hi] for per-env action delay (in *decimated* steps).
    Only active when ``enable_delay=True``."""

    action_noise_std_pos: float = 0.01
    """Std-dev of additive Gaussian noise on position-increment action dims.
    Only active when ``enable_noise=True``."""

    action_noise_std_vel: float = 0.01
    """Std-dev of additive Gaussian noise on velocity-feedforward action dims.
    Only active when ``enable_noise=True`` and action layout is pos+vel."""

    packet_loss_prob: float = 0.03
    """Probability that an action packet is dropped (previous command held).
    Only active when ``enable_delay=True``."""

    # -- Observation buffer ----------------------------------------------------
    obs_delay_range: tuple[int, int] = (0, 1)
    """Uniform int range [lo, hi] for per-env observation delay (in *decimated* steps).
    Only active when ``enable_delay=True``."""

    obs_noise_std_pos: float = 0.005
    """Observation noise std on position-error components (metres)."""

    obs_noise_std_ori: float = 0.01
    """Observation noise std on orientation-error components (radians)."""

    obs_noise_std_joint_pos: float = 0.01
    """Observation noise std on joint position components (radians)."""

    obs_noise_std_joint_vel: float = 0.02
    """Observation noise std on joint velocity components (rad/s)."""

    obs_noise_std_tcp_lin_vel: float = 0.01
    """Observation noise std on TCP linear velocity components (m/s)."""

    obs_noise_std_tcp_ang_vel: float = 0.02
    """Observation noise std on TCP angular velocity components (rad/s)."""

    # -- Actuator randomisation ------------------------------------------------
    stiffness_scale_range: tuple[float, float] = (0.8, 1.2)
    """Multiplicative scale range applied to nominal joint stiffness.
    Only active when ``enable_actuator_rand=True``."""

    damping_scale_range: tuple[float, float] = (0.8, 1.2)
    """Multiplicative scale range applied to nominal joint damping.
    Only active when ``enable_actuator_rand=True``."""

    friction_variation: float = 0.35
    """Fractional variation (±) around default per-joint friction coefficients.
    Only active when ``enable_actuator_rand=True``."""

    default_joint_friction: tuple[float, ...] = (8.24, 10.51, 7.9, 1.43, 1.05, 1.63)
    """Nominal static friction for each joint (shoulder_pan → wrist_3)."""

    # -- Mass / CoM randomisation ----------------------------------------------
    mass_scale_range: tuple[float, float] = (0.85, 1.15)
    """Multiplicative scale range applied to default link masses (±15%).
    Only active when ``enable_mass_com_rand=True``."""

    recompute_inertia: bool = True
    """Whether to recompute inertia tensors after changing mass (assumes uniform-density bodies)."""

    com_offset_range: tuple[float, float] = (-0.01, 0.01)
    """Additive uniform offset range (metres) applied independently to x, y, z of each
    link's center of mass. Only active when ``enable_mass_com_rand=True``."""


# ---------------------------------------------------------------------------
# ActionBuffer
# ---------------------------------------------------------------------------


class ActionBuffer:
    """Per-environment FIFO action queue with optional delay, packet-loss and noise.

    **Action layout detection** (via ``action_dim`` vs ``num_joints``):

    - ``action_dim == num_joints``     → *position-only*: uniform noise ``action_noise_std_pos``.
    - ``action_dim == 2 * num_joints`` → *pos + vel feedforward*: split noise
      (``action_noise_std_pos`` for first half, ``action_noise_std_vel`` for second half).

    *Drop policy*: when a packet is lost the **previous** command is held.
    """

    def __init__(
        self,
        num_envs: int,
        action_dim: int,
        num_joints: int,
        cfg: DomainRandomizationV4Cfg,
        device: torch.device | str,
    ) -> None:
        self.num_envs = num_envs
        self.action_dim = action_dim
        self.num_joints = num_joints
        self.cfg = cfg
        self.device = torch.device(device)

        # -- Detect layout and build noise vector --
        if action_dim == num_joints:
            self._mode = "pos"
        elif action_dim == 2 * num_joints:
            self._mode = "pos_vel"
        else:
            raise ValueError(
                f"ActionBuffer: action_dim={action_dim} is neither num_joints={num_joints} "
                f"(pos-only) nor 2*num_joints={2 * num_joints} (pos+vel). "
                "Cannot determine action layout."
            )
        self._noise_std = self._build_noise_vector()

        max_delay = cfg.action_delay_range[1]
        self.buffer_len = max(max_delay + 1, 1)

        # Ring-buffer: (num_envs, buffer_len, action_dim)
        self.buffer = torch.zeros(num_envs, self.buffer_len, action_dim, device=self.device)
        self.cursor = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.delay = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.last_action = torch.zeros(num_envs, action_dim, device=self.device)

        # -- Cached index vector (avoids re-allocating torch.arange every step) --
        self._env_idx = torch.arange(num_envs, device=self.device)

        # -- Pre-allocated noise buffer for in-place randn (avoids allocation each step) --
        self._noise_buf = torch.zeros(num_envs, action_dim, device=self.device)

        # -- Fast-path flags resolved once at construction time --
        # Skip everything: no noise, no delay → transparent pass-through
        self._skip_all = not cfg.enable_noise and not cfg.enable_delay
        # Skip ring-buffer mechanics (delay disabled)
        self._skip_buffer = not cfg.enable_delay

        self._resample_delay()

    # -- public API ----------------------------------------------------------

    def push(self, actions: torch.Tensor) -> torch.Tensor:
        """Write *actions* into the ring-buffer and return the processed command.

        Processing order: noise → delay → packet-loss.

        Args:
            actions: ``(num_envs, action_dim)`` – clipped normalised actions.

        Returns:
            ``(num_envs, action_dim)`` – actions to actually execute.
        """
        # -- Fast-path: both toggles off → transparent pass-through (zero overhead) --
        if self._skip_all:
            return actions

        # -- Fast-path: delay disabled → just apply noise and return (skip ring buffer) --
        if self._skip_buffer:
            if self.cfg.enable_noise:
                torch.randn(self._noise_buf.shape, device=self._noise_buf.device, dtype=self._noise_buf.dtype, out=self._noise_buf)
                actions = actions.clone()
                actions.add_(self._noise_buf * self._noise_std)
            return actions

        # -- Full path: delay enabled --

        # Additive noise (in-place random fill avoids allocation)
        if self.cfg.enable_noise:
            torch.randn(self._noise_buf.shape, device=self._noise_buf.device, dtype=self._noise_buf.dtype, out=self._noise_buf)
            actions = actions.clone()
            actions.add_(self._noise_buf * self._noise_std)

        # Write into ring-buffer
        idx = self.cursor % self.buffer_len
        self.buffer[self._env_idx, idx] = actions
        self.cursor += 1

        # Read delayed action
        read_idx = (self.cursor - 1 - self.delay) % self.buffer_len
        delayed = self.buffer[self._env_idx, read_idx]

        # Packet loss (drop → hold previous)
        if self.cfg.packet_loss_prob > 0.0:
            drop_mask = torch.rand(self.num_envs, device=self.device) < self.cfg.packet_loss_prob
            delayed = torch.where(drop_mask.unsqueeze(-1), self.last_action, delayed)

        # In-place copy avoids allocating a new tensor
        self.last_action.copy_(delayed)
        return delayed

    def reset(self, env_ids: torch.Tensor) -> None:
        """Clear buffer and re-sample delays for the given environments."""
        if self._skip_all:
            return
        env_ids = env_ids.to(self.device, dtype=torch.long)
        if not self._skip_buffer:
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
            self.delay[env_ids] = torch.randint(lo, hi + 1, (len(env_ids),), device=self.device)

    def _build_noise_vector(self) -> torch.Tensor:
        """Build a per-dim noise std vector according to the detected layout."""
        nj = self.num_joints
        cfg = self.cfg
        if self._mode == "pos":
            return torch.full((nj,), cfg.action_noise_std_pos, device=self.device)
        else:  # pos_vel
            return torch.cat([
                torch.full((nj,), cfg.action_noise_std_pos, device=self.device),
                torch.full((nj,), cfg.action_noise_std_vel, device=self.device),
            ])


# ---------------------------------------------------------------------------
# ObservationBuffer
# ---------------------------------------------------------------------------


class ObservationBuffer:
    """Per-environment observation delay buffer with structured Gaussian noise.

    Expected observation layout (24-dim for 6 arm joints):

        pos_error_norm (3), ori_error_norm (3),
        joint_pos_norm (6), joint_vel_norm (6),
        tcp_linear_vel_norm (3), tcp_angular_vel_norm (3)

    The buffer is lazily initialised on the first ``append_and_get`` call so that
    the actual runtime observation dimension is used.
    """

    def __init__(
        self,
        num_envs: int,
        obs_dim: int,
        num_joints: int,
        cfg: DomainRandomizationV4Cfg,
        device: torch.device | str,
    ) -> None:
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.num_joints = num_joints
        self.cfg = cfg
        self.device = torch.device(device)

        max_delay = cfg.obs_delay_range[1]
        self.buffer_len = max(max_delay + 1, 1)

        # Lazy init on first append_and_get call
        self.buffer: torch.Tensor | None = None
        self.cursor = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.delay = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._noise_std: torch.Tensor | None = None
        # Pre-allocated noise buffer (lazily sized to actual obs_dim at first call)
        self._noise_buf: torch.Tensor | None = None

        # -- Cached index vector --
        self._env_idx = torch.arange(num_envs, device=self.device)

        # -- Fast-path flags --
        self._skip_all = not cfg.enable_noise and not cfg.enable_delay
        self._skip_buffer = not cfg.enable_delay

        self._resample_delay()

    # -- public API ----------------------------------------------------------

    def append_and_get(self, obs: torch.Tensor) -> torch.Tensor:
        """Push *obs* into the buffer and return the (optionally delayed + noisy) observation.

        Args:
            obs: ``(num_envs, obs_dim)``

        Returns:
            ``(num_envs, obs_dim)`` – observation to feed the policy.
        """
        # -- Fast-path: both toggles off → transparent pass-through --
        if self._skip_all:
            return obs

        # Lazy init on first call so we match the real obs shape
        if self.buffer is None:
            actual_dim = obs.shape[1]
            self.obs_dim = actual_dim
            self.buffer = torch.zeros(
                self.num_envs, self.buffer_len, actual_dim, device=self.device
            )
            self._noise_std = self._build_noise_vector()
            self._noise_buf = torch.zeros(self.num_envs, actual_dim, device=self.device)

        # -- Fast-path: delay disabled → just apply noise and return (skip ring buffer) --
        if self._skip_buffer:
            # enable_noise must be True here (skip_all already handled above)
            torch.randn(self._noise_buf.shape, device=self._noise_buf.device, dtype=self._noise_buf.dtype, out=self._noise_buf)
            return obs + self._noise_buf * self._noise_std

        # -- Full path: delay enabled --
        idx = self.cursor % self.buffer_len
        self.buffer[self._env_idx, idx] = obs
        self.cursor += 1

        read_idx = (self.cursor - 1 - self.delay) % self.buffer_len
        # clone() is required so in-place noise addition doesn't corrupt the ring buffer
        delayed = self.buffer[self._env_idx, read_idx].clone()

        # Additive Gaussian noise (in-place to avoid extra allocation)
        if self.cfg.enable_noise:
            torch.randn(self._noise_buf.shape, device=self._noise_buf.device, dtype=self._noise_buf.dtype, out=self._noise_buf)
            delayed.add_(self._noise_buf * self._noise_std)

        return delayed

    def reset(self, env_ids: torch.Tensor) -> None:
        if self._skip_all:
            return
        env_ids = env_ids.to(self.device, dtype=torch.long)
        if self.buffer is not None and not self._skip_buffer:
            self.buffer[env_ids] = 0.0
        self.cursor[env_ids] = 0
        self._resample_delay(env_ids)

    # -- internals -----------------------------------------------------------

    def _resample_delay(self, env_ids: torch.Tensor | None = None) -> None:
        lo, hi = self.cfg.obs_delay_range
        if env_ids is None:
            self.delay = torch.randint(lo, hi + 1, (self.num_envs,), device=self.device)
        else:
            self.delay[env_ids] = torch.randint(lo, hi + 1, (len(env_ids),), device=self.device)

    def _build_noise_vector(self) -> torch.Tensor:
        """Construct a 1-D noise-std tensor aligned with the V4 observation layout (24-dim).

        Config values are in physical units; they are divided here by the same
        normalisation constants used in the observation vector to convert them
        into normalised-observation space.
        """
        nj = self.num_joints
        cfg = self.cfg
        expected_dim = 3 + 3 + nj + nj + 3 + 3  # 24 for 6 joints
        if self.obs_dim != expected_dim:
            raise ValueError(
                f"ObservationBuffer: expected obs_dim={expected_dim} for the V4 layout "
                f"(3+3+{nj}+{nj}+3+3), got obs_dim={self.obs_dim}. "
                "Update the environment observation_space or this noise builder."
            )
        import math
        # Joint position ranges: ±2π for all joints except elbow (joint 2) which is ±π.
        # Observation normalization is 2*(pos-lower)/(upper-lower)-1, so the
        # effective range is (upper-lower)/2 = half-range per joint.
        # noise_normalised = noise_physical / half_range
        joint_half_ranges = torch.tensor(
            [2 * math.pi, 2 * math.pi, math.pi, 2 * math.pi, 2 * math.pi, 2 * math.pi],
            device=self.device,
        )[:nj]

        parts: list[torch.Tensor] = [
            torch.full((3,),  cfg.obs_noise_std_pos         / MAX_REACH,      device=self.device),  # pos_error: m  → /MAX_REACH
            torch.full((3,),  cfg.obs_noise_std_ori         / math.pi,        device=self.device),  # ori_error: rad → /π
            torch.full((1,), cfg.obs_noise_std_joint_pos, device=self.device).expand(nj) / joint_half_ranges,  # joint_pos: rad → /half_range
            torch.full((nj,), cfg.obs_noise_std_joint_vel   / MAX_JOINT_VEL,  device=self.device),  # joint_vel: rad/s → /MAX_JOINT_VEL
            torch.full((3,),  cfg.obs_noise_std_tcp_lin_vel / TCP_MAX_SPEED,  device=self.device),  # tcp_lin_vel: m/s → /TCP_MAX_SPEED
            torch.full((3,),  cfg.obs_noise_std_tcp_ang_vel / math.pi,        device=self.device),  # tcp_ang_vel: rad/s → /π
        ]
        return torch.cat(parts)


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
    Only active when ``cfg.enable_actuator_rand=True`` (otherwise zero-overhead pass-through).
    """

    def __init__(
        self,
        robot: Articulation,
        cfg: DomainRandomizationV4Cfg,
        device: torch.device | str,
    ) -> None:
        self.robot = robot
        self.cfg = cfg
        self.device = torch.device(device)
        self.num_envs = robot.num_instances
        self.num_joints = robot.num_joints

        # -- Fast-path flag resolved once at construction time --
        self._skip_all = not cfg.enable_actuator_rand

        # Snapshot nominal values from the simulation (after scene.clone_environments)
        # Shape: (num_envs, num_joints)
        # Only allocate if randomization is enabled
        if not self._skip_all:
            self.nominal_stiffness = robot.data.default_joint_stiffness.clone().to(self.device)
            self.nominal_damping = robot.data.default_joint_damping.clone().to(self.device)

            # Build nominal friction tensor – pad/truncate if joint count differs
            fric = list(cfg.default_joint_friction)
            if len(fric) < self.num_joints:
                fric.extend([fric[-1]] * (self.num_joints - len(fric)))
            fric_t = torch.tensor(fric[: self.num_joints], dtype=torch.float32, device=self.device)
            self.nominal_friction = fric_t.unsqueeze(0).expand(self.num_envs, -1).clone()
        else:
            self.nominal_stiffness = None
            self.nominal_damping = None
            self.nominal_friction = None

    # -- public API ----------------------------------------------------------

    def sample_and_apply(self, env_ids: torch.Tensor) -> None:
        """Sample new actuator parameters and write them to the physics sim.

        Should be called inside ``_reset_idx`` **after** ``scene.clone_environments``.
        Zero-overhead pass-through if ``cfg.enable_actuator_rand=False``.

        Args:
            env_ids: 1-D long tensor of environment indices to randomise.
        """
        if self._skip_all:
            return

        env_ids = env_ids.to(self.device, dtype=torch.long)
        n = len(env_ids)
        nj = self.num_joints

        # --- stiffness ---
        s_lo, s_hi = self.cfg.stiffness_scale_range
        stiff_scale = torch.empty(n, nj, device=self.device).uniform_(s_lo, s_hi)
        self.robot.write_joint_stiffness_to_sim(
            self.nominal_stiffness[env_ids] * stiff_scale, env_ids=env_ids
        )

        # --- damping ---
        d_lo, d_hi = self.cfg.damping_scale_range
        damp_scale = torch.empty(n, nj, device=self.device).uniform_(d_lo, d_hi)
        self.robot.write_joint_damping_to_sim(
            self.nominal_damping[env_ids] * damp_scale, env_ids=env_ids
        )

        # --- friction (±variation around defaults) ---
        var = self.cfg.friction_variation
        fric_scale = torch.empty(n, nj, device=self.device).uniform_(1.0 - var, 1.0 + var)
        self.robot.write_joint_friction_coefficient_to_sim(
            self.nominal_friction[env_ids] * fric_scale, env_ids=env_ids
        )


# ---------------------------------------------------------------------------
# MassComRandomizer
# ---------------------------------------------------------------------------


class MassComRandomizer:
    """Per-environment randomisation of link masses and centers of mass.

    On ``sample_and_apply`` the randomiser:
    1. Resets masses to defaults, samples multiplicative scales from
       ``mass_scale_range``, applies them, and writes to the physics simulation.
    2. Optionally recomputes inertia tensors (linear scaling with mass).
    3. Resets CoMs to defaults, samples additive offsets from ``com_offset_range``,
       and writes to the physics simulation.

    Uses the ``root_physx_view`` API (CPU tensors only, same as Isaac Lab events).
    Only active when ``cfg.enable_mass_com_rand=True`` (otherwise zero-overhead pass-through).
    """

    def __init__(
        self,
        robot: Articulation,
        cfg: DomainRandomizationV4Cfg,
        device: torch.device | str,
    ) -> None:
        self.robot = robot
        self.cfg = cfg
        self.device = torch.device(device)
        self.num_envs = robot.num_instances
        self.num_bodies = robot.num_bodies

        # -- Fast-path flag resolved once at construction time --
        self._skip_all = not cfg.enable_mass_com_rand

        # PhysX view operates on CPU – keep snapshots on CPU
        # Only allocate if randomization is enabled
        if not self._skip_all:
            # Shape: (num_envs, num_bodies)
            self.default_mass = robot.data.default_mass.clone().cpu()
            # Shape: (num_envs, num_bodies, 7) – first 3: position, last 4: orientation
            self.default_coms = robot.root_physx_view.get_coms().clone()  # already CPU
        else:
            self.default_mass = None
            self.default_coms = None

    # -- public API ----------------------------------------------------------

    def sample_and_apply(self, env_ids: torch.Tensor) -> None:
        """Sample new mass/CoM parameters and write them to the physics sim.

        Should be called inside ``_reset_idx``.
        Zero-overhead pass-through if ``cfg.enable_mass_com_rand=False``.

        .. note::
            The PhysX tensor API works exclusively on CPU.

        Args:
            env_ids: 1-D long tensor of environment indices to randomise.
        """
        if self._skip_all:
            return

        env_ids_cpu = env_ids.to("cpu", dtype=torch.long)
        n = len(env_ids_cpu)
        nb = self.num_bodies

        # ---- Mass randomisation ----
        m_lo, m_hi = self.cfg.mass_scale_range
        mass_scale = torch.empty(n, nb, device="cpu").uniform_(m_lo, m_hi)

        masses = self.robot.root_physx_view.get_masses()
        masses[env_ids_cpu] = self.default_mass[env_ids_cpu].clone()
        masses[env_ids_cpu] *= mass_scale
        masses = torch.clamp(masses, min=1e-6)
        self.robot.root_physx_view.set_masses(masses, env_ids_cpu)

        # Recompute inertia tensors if requested (assumes uniform density)
        if self.cfg.recompute_inertia:
            inertias = self.robot.root_physx_view.get_inertias()
            ratios = masses[env_ids_cpu] / self.default_mass[env_ids_cpu].clamp(min=1e-6)
            # inertias shape: (num_envs, num_bodies, 9) – 3×3 flattened
            inertias[env_ids_cpu] *= ratios.unsqueeze(-1)
            self.robot.root_physx_view.set_inertias(inertias, env_ids_cpu)

        # ---- CoM randomisation ----
        c_lo, c_hi = self.cfg.com_offset_range
        com_offsets = torch.empty(n, nb, 3, device="cpu").uniform_(c_lo, c_hi)

        coms = self.robot.root_physx_view.get_coms().clone()
        coms[env_ids_cpu] = self.default_coms[env_ids_cpu].clone()
        coms[env_ids_cpu, :, :3] += com_offsets  # position part only (first 3 of 7)
        self.robot.root_physx_view.set_coms(coms, env_ids_cpu)
