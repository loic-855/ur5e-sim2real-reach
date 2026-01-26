# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

# Tuned for pose/orientation task (19-dim obs, 6-dim action)
# Focus: deterministic behavior, minimal oscillations
@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32  # Longer rollouts for smoother trajectories
    max_iterations = 800
    save_interval = 50
    experiment_name = "pose_orientation_no_gripper"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.3,  # Lower noise = more deterministic from start
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[128, 64],  # Smaller network for 19 obs / 6 actions
        critic_hidden_dims=[128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.1,  # Smaller clip = more conservative updates
        entropy_coef=0.002,  # Low entropy = less exploration = more deterministic
        num_learning_epochs=10,  # More epochs per update
        num_mini_batches=8,
        learning_rate=3.0e-4,  # Slightly lower LR for stability
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.005,  # Smaller KL = smoother policy changes
        max_grad_norm=0.5,  # Tighter gradient clipping
    )
