# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Domain randomization utilities for sim-to-real transfer (V4).

Extension of V3 with additional physical randomization:

- **MassComRandomizer** – per-env randomisation of link masses (±15% scale)
  and center-of-mass offsets (small additive perturbation) using the
  ``root_physx_view`` API (same approach as ``isaaclab.envs.mdp.events``).
- **DomainRandomizationV4Cfg** – adds mass/CoM randomisation knobs.
- All V3 helpers (ActionBufferV3, ObservationBuffer, ActuatorRandomizer) are
  re-exported unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from isaaclab.assets import Articulation

# Re-export unchanged helpers from V3
from Woodworking_Simulation.common.domain_randomization_v3 import (  # noqa: F401
    ActionBufferV3,
    ActuatorRandomizer,
    DomainRandomizationV3Cfg,
    ObservationBuffer,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DomainRandomizationV4Cfg(DomainRandomizationV3Cfg):
    """All domain-randomisation knobs – V4 variant with mass & CoM randomisation.

    Inherits from V3 config for full backward compatibility.
    """

    # -- Mass randomisation ----------------------------------------------------
    mass_scale_range: tuple[float, float] = (0.85, 1.15)
    """Multiplicative scale range applied to the default link masses.
    (0.85, 1.15) corresponds to ±15%."""

    recompute_inertia: bool = True
    """Whether to recompute inertia tensors after changing mass.
    Assumes uniform-density bodies (same as Isaac Lab events)."""

    # -- Center-of-mass randomisation ------------------------------------------
    com_offset_range: tuple[float, float] = (-0.01, 0.01)
    """Additive uniform offset range (metres) applied independently to the
    x, y, z components of each link's center of mass."""


# ---------------------------------------------------------------------------
# MassComRandomizer
# ---------------------------------------------------------------------------

class MassComRandomizer:
    """Per-environment randomisation of link masses and centers of mass.

    On ``sample_and_apply`` the randomiser:
    1. Resets masses to their default values, samples multiplicative scales from
       ``mass_scale_range``, applies them, and writes to the physics simulation.
    2. Resets CoMs to their default values, samples additive offsets from
       ``com_offset_range``, and writes to the physics simulation.

    Uses the same ``root_physx_view`` API as ``isaaclab.envs.mdp.events``.
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

        # PhysX view operates on CPU – keep snapshots on CPU to avoid device mismatches
        # Shape: (num_envs, num_bodies)
        self.default_mass = robot.data.default_mass.clone().cpu()

        # Shape: (num_envs, num_bodies, 7) – first 3 are position, last 4 are orientation
        self.default_coms = robot.root_physx_view.get_coms().clone()  # already CPU

    # -- public API ----------------------------------------------------------

    def sample_and_apply(self, env_ids: torch.Tensor) -> None:
        """Sample new mass/CoM parameters and write them to the physics sim.

        Should be called inside ``_reset_idx``.

        .. note::
            The PhysX tensor API (``root_physx_view``) works exclusively on CPU.
            All operations here are performed on CPU tensors.

        Args:
            env_ids: 1-D long tensor of environment indices to randomise.
        """
        if not self.cfg.enabled:
            return

        env_ids_cpu = env_ids.to("cpu", dtype=torch.long)
        n = len(env_ids_cpu)
        nb = self.num_bodies

        # ---- Mass randomisation ----
        m_lo, m_hi = self.cfg.mass_scale_range
        mass_scale = torch.empty(n, nb, device="cpu").uniform_(m_lo, m_hi)

        # Get current masses (CPU) and reset to defaults for these envs
        masses = self.robot.root_physx_view.get_masses()
        masses[env_ids_cpu] = self.default_mass[env_ids_cpu].clone()
        # Apply scale
        masses[env_ids_cpu] *= mass_scale
        # Ensure positive
        masses = torch.clamp(masses, min=1e-6)
        # Write back
        self.robot.root_physx_view.set_masses(masses, env_ids_cpu)

        # Recompute inertia tensors if requested (assumes uniform density)
        if self.cfg.recompute_inertia:
            inertias = self.robot.root_physx_view.get_inertias()
            # default_mass never has zeros but guard anyway
            ratios = masses[env_ids_cpu] / self.default_mass[env_ids_cpu].clamp(min=1e-6)
            # Inertia tensor scales linearly with mass for uniform-density bodies
            # inertias shape: (num_envs, num_bodies, 9) – 3x3 flattened
            inertias[env_ids_cpu] *= ratios.unsqueeze(-1)
            self.robot.root_physx_view.set_inertias(inertias, env_ids_cpu)

        # ---- CoM randomisation ----
        c_lo, c_hi = self.cfg.com_offset_range
        com_offsets = torch.empty(n, nb, 3, device="cpu").uniform_(c_lo, c_hi)

        coms = self.robot.root_physx_view.get_coms().clone()
        # Reset to defaults first
        coms[env_ids_cpu] = self.default_coms[env_ids_cpu].clone()
        # Apply additive offsets to position part only (first 3 of 7)
        coms[env_ids_cpu, :, :3] += com_offsets
        # Write back
        self.robot.root_physx_view.set_coms(coms, env_ids_cpu)
