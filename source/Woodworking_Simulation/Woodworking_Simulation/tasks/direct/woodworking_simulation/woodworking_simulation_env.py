# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from .woodworking_simulation_env_cfg import WoodworkingSimulationEnvCfg


class WoodworkingSimulationEnv(DirectRLEnv):
    cfg: WoodworkingSimulationEnvCfg

    def __init__(self, cfg: WoodworkingSimulationEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.gripper_robot = self.scene.articulations["gripper_robot"]
        self.screwdriver_robot = self.scene.articulations["screwdriver_robot"]
        self.robots = (self.gripper_robot, self.screwdriver_robot)

        self.num_gripper_dofs = self.gripper_robot.data.joint_pos.shape[-1]
        self.num_screwdriver_dofs = self.screwdriver_robot.data.joint_pos.shape[-1]
        self.num_actions = self.num_gripper_dofs + self.num_screwdriver_dofs

        self.cfg.action_space = self.num_actions
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.num_actions,), dtype=np.float32)

        self.num_observations = 2 * (self.num_gripper_dofs + self.num_screwdriver_dofs)
        self.cfg.observation_space = self.num_observations
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_observations,), dtype=np.float32
        )

        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        screwdriver_robot = Articulation(self.cfg.screwdriver_robot_cfg)

        table_cfg = sim_utils.UsdFileCfg(usd_path=self.cfg.table_usd_path)
        table_cfg.func(
            "/World/envs/env_0/WoodworkingTable",
            table_cfg,
            translation=self.cfg.table_translation,
            orientation=self.cfg.table_orientation,
        )

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        self.scene.articulations["gripper_robot"] = self.robot
        self.scene.articulations["screwdriver_robot"] = screwdriver_robot

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if actions.ndim == 1:
            actions = actions.unsqueeze(0)
        if actions.shape[0] == 1 and self.num_envs > 1:
            actions = actions.expand(self.num_envs, -1)
        if actions.shape[0] != self.num_envs:
            repeat_factor = int(np.ceil(self.num_envs / actions.shape[0]))
            actions = actions.repeat(repeat_factor, 1)[: self.num_envs]
        if actions.shape[-1] < self.num_actions:
            pad_width = self.num_actions - actions.shape[-1]
            pad = torch.zeros((self.num_envs, pad_width), device=self.device, dtype=actions.dtype)
            actions = torch.cat((actions, pad), dim=-1)
        self.actions = actions[:, : self.num_actions]

    def _apply_action(self) -> None:
        gripper_actions = self.actions[:, : self.num_gripper_dofs]
        screwdriver_actions = self.actions[:, self.num_gripper_dofs :]

        self.gripper_robot.set_joint_effort_target(gripper_actions * self.cfg.action_scale)
        self.screwdriver_robot.set_joint_effort_target(screwdriver_actions * self.cfg.action_scale)

    def _get_observations(self) -> dict:
        gripper_pos = self.gripper_robot.data.joint_pos
        gripper_vel = self.gripper_robot.data.joint_vel
        screwdriver_pos = self.screwdriver_robot.data.joint_pos
        screwdriver_vel = self.screwdriver_robot.data.joint_vel
        obs = torch.cat((gripper_pos, gripper_vel, screwdriver_pos, screwdriver_vel), dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = torch.zeros_like(time_out)
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)

        if env_ids is None:
            env_ids = self.gripper_robot._ALL_INDICES

        for robot in self.robots:
            joint_pos = robot.data.default_joint_pos[env_ids]
            joint_vel = robot.data.default_joint_vel[env_ids]
            default_root_state = robot.data.default_root_state[env_ids]
            default_root_state[:, :3] += self.scene.env_origins[env_ids]

            robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
            robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
            robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)