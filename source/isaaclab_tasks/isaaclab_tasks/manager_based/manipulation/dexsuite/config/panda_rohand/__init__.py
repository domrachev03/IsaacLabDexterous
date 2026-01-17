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

# State Observation - Reorient
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

# State Observation - Lift
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

# Visible point-cloud observation variants - Reorient
gym.register(
    id="Isaac-Dexsuite-Panda-RoHand-Reorient-Visible-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexsuite_panda_rohand_env_cfg:DexsuitePandaRoHandReorientVisibleEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_visible_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexsuitePandaRoHandVisiblePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Dexsuite-Panda-RoHand-Reorient-Visible-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexsuite_panda_rohand_env_cfg:DexsuitePandaRoHandReorientVisibleEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_visible_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexsuitePandaRoHandVisiblePPORunnerCfg",
    },
)

# Visible point-cloud observation variants - Lift
gym.register(
    id="Isaac-Dexsuite-Panda-RoHand-Lift-Visible-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexsuite_panda_rohand_env_cfg:DexsuitePandaRoHandLiftVisibleEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_visible_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexsuitePandaRoHandVisiblePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Dexsuite-Panda-RoHand-Lift-Visible-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexsuite_panda_rohand_env_cfg:DexsuitePandaRoHandLiftVisibleEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_visible_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexsuitePandaRoHandVisiblePPORunnerCfg",
    },
)

