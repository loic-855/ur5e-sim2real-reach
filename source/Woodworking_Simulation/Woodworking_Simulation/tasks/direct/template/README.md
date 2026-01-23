# Template Configuration Files

This folder contains agent configuration files for training frameworks other than RSL_RL (PPO):

- **rl_games_ppo_cfg.yaml**: RL-Games PPO configuration
- **sb3_ppo_cfg.yaml**: Stable Baselines3 PPO configuration  
- **skrl_*.yaml**: SKRL (Skeletons for Reinforcement Learning) configurations
  - skrl_amp_cfg.yaml: Adversarial Motion Prediction
  - skrl_ippo_cfg.yaml: Independent PPO
  - skrl_mappo_cfg.yaml: Multi-Agent PPO
  - skrl_ppo_cfg.yaml: Standard PPO

These are kept in a shared template folder since they are not actively used in the current RSL_RL-focused training pipeline, but they may be useful for future experimentation.
