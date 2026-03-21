import gymnasium as gym


gym.register(
    id="Genesis-HVelocity-G1-v0",
    entry_point="genesislab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.flat_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)
