# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Dexsuite Franka Panda + RoHand environments.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="Isaac-Dexsuite-Panda-RoHand-Reorient-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexsuite_panda_rohand_env_cfg:DexsuitePandaRoHandReorientEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexsuitePandaRoHandPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Dexsuite-Panda-RoHand-Reorient-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexsuite_panda_rohand_env_cfg:DexsuitePandaRoHandReorientEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexsuitePandaRoHandPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Dexsuite-Panda-RoHand-Lift-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexsuite_panda_rohand_env_cfg:DexsuitePandaRoHandLiftEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexsuitePandaRoHandPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Dexsuite-Panda-RoHand-Lift-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexsuite_panda_rohand_env_cfg:DexsuitePandaRoHandLiftEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexsuitePandaRoHandPPORunnerCfg",
    },
)

