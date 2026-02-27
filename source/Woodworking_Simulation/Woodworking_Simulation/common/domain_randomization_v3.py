# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Domain randomization utilities for sim-to-real transfer (V3).

Extension of V2 for the 12-dim action space (6 position increments + 6 velocity
targets).  Key differences:

- **ActionBuffer** applies *separate* noise standard-deviations to the position
  and velocity halves of the action vector.
- **DomainRandomizationV3Cfg** adds ``action_noise_std_pos`` and
  ``action_noise_std_vel`` fields (the old ``action_noise_std`` is ignored).
- **ObservationBuffer** and **ActuatorRandomizer** are unchanged and re-exported
  from V2 for convenience.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from isaaclab.assets import Articulation

# Re-export unchanged helpers from V2
from Woodworking_Simulation.common.domain_randomization_v2 import (  # noqa: F401
    ActuatorRandomizer,
    DomainRandomizationCfg,
    ObservationBuffer,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DomainRandomizationV3Cfg(DomainRandomizationCfg):
    """All domain-randomisation knobs – V3 variant with split action noise.

    Inherits from V2 config for compatibility with ObservationBuffer and
    ActuatorRandomizer.  The old ``action_noise_std`` field is still present
    but ignored by ActionBufferV3 in favour of the split fields below.
    """

    # -- Action buffer (overrides) ---------------------------------------------
    action_noise_std_pos: float = 0.025
    """Std-dev of additive Gaussian noise on the position-increment actions (dims 0-5)."""
    action_noise_std_vel: float = 0.01
    """Std-dev of additive Gaussian noise on the velocity-target actions (dims 6-11)."""


# ---------------------------------------------------------------------------
# ActionBufferV3
# ---------------------------------------------------------------------------

class ActionBufferV3:
    """Per-environment FIFO action queue with delay, packet-loss & **split** additive noise.

    Unlike V2 which applies a single noise std across all action dims, V3 uses
    ``action_noise_std_pos`` for dims ``[0 .. num_joints-1]`` (position increments) and
    ``action_noise_std_vel`` for dims ``[num_joints .. 2*num_joints-1]`` (velocity targets).

    *Drop policy*: when a packet is lost the **previous** command is held.
    """

    def __init__(
        self,
        num_envs: int,
        action_dim: int,
        num_joints: int,
        cfg: DomainRandomizationV3Cfg,
        device: torch.device | str,
    ) -> None:
        self.num_envs = num_envs
        self.action_dim = action_dim
        self.num_joints = num_joints
        self.cfg = cfg
        self.device = torch.device(device)

        # Build per-dim noise std vector
        self._noise_std = self._build_action_noise_vector()

        max_delay = cfg.action_delay_range[1]
        self.buffer_len = max_delay + 1

        # Ring-buffer: (num_envs, buffer_len, action_dim)
        self.buffer = torch.zeros(
            num_envs, self.buffer_len, action_dim, device=self.device
        )
        self.cursor = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.delay = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.last_action = torch.zeros(num_envs, action_dim, device=self.device)

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

        # --- structured additive noise ---
        noise = torch.randn_like(actions) * self._noise_std.unsqueeze(0)
        actions = actions + noise

        # --- write into ring-buffer ---
        idx = self.cursor % self.buffer_len
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

    def _build_action_noise_vector(self) -> torch.Tensor:
        """Construct a 1-D noise-std tensor: [pos_noise]*nj + [vel_noise]*nj."""
        nj = self.num_joints
        expected_dim = 2 * nj
        if self.action_dim != expected_dim:
            raise ValueError(
                f"ActionBufferV3 expects action_dim={expected_dim} (2×{nj} joints), "
                f"got action_dim={self.action_dim}."
            )
        parts = [
            torch.full((nj,), self.cfg.action_noise_std_pos, device=self.device),
            torch.full((nj,), self.cfg.action_noise_std_vel, device=self.device),
        ]
        return torch.cat(parts)
