#!/usr/bin/env python3
"""
policy_controller.py
--------------------
Base class for loading TorchScript policies and computing actions.
Subclasses must implement `_compute_observation()` and `forward()`.
"""

import io
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from utils.config_loader import parse_env_config, get_physics_properties, get_robot_joint_properties

class PolicyController:
    """Base controller that loads a TorchScript policy and computes actions."""

    def __init__(self) -> None:
        self.policy: Optional[torch.jit.ScriptModule] = None
        self.policy_env_params: Optional[dict] = None
        self._decimation: int = 1
        self._dt: float = 0.01
        self.dof_names: list = []
        self.default_pos: list = []
        self.default_vel: list = []

    def load_policy(self, policy_path: Path, env_path: Path) -> None:
        """Load TorchScript policy and environment config."""
        print(f"Loading policy: {policy_path}")
        print(f"Loading env config: {env_path}")

        with open(policy_path, "rb") as f:
            self.policy = torch.jit.load(io.BytesIO(f.read()))
            print(self.policy)
        
        self.policy_env_params = parse_env_config(str(env_path))
        self._decimation, self._dt, _ = get_physics_properties(self.policy_env_params)
        _, _, _, _, self.default_pos, self.default_vel = get_robot_joint_properties(
            self.policy_env_params, self.dof_names
        )
        
        print(f"Policy loaded: dt={self._dt}, decimation={self._decimation}, policy_dt={self.policy_dt}")

    def _compute_action(self, obs: np.ndarray) -> np.ndarray:
        """Run policy inference and return clipped action [-1, 1]."""
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).view(1, -1).float()
            action = self.policy(obs_t).detach().view(-1).numpy()
        return np.clip(action, -1.0, 1.0)

    def _compute_observation(self, *args, **kwargs) -> Optional[np.ndarray]:
        """Build observation vector. Must be overridden."""
        raise NotImplementedError

    def forward(self, *args, **kwargs) -> Any:
        """Compute next joint targets. Must be overridden."""
        raise NotImplementedError

    @property
    def policy_dt(self) -> float:
        """Policy control timestep = sim_dt * decimation."""
        return self._dt * self._decimation