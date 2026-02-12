# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Tessolo hand environment.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

inhand_task_entry = "isaaclab_tasks.direct.inhand_manipulation"

gym.register(
    id="Isaac-Repose-Cube-Tessolo-Direct-v0",
    entry_point=f"{inhand_task_entry}.inhand_manipulation_env:InHandManipulationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tessolo_hand_env_cfg:TessoloHandEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TessoloHandPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Repose-Cube-Tessolo-OpenAI-FF-Direct-v0",
    entry_point=f"{inhand_task_entry}.inhand_manipulation_env:InHandManipulationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tessolo_hand_env_cfg:TessoloHandOpenAIEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_ff_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TessoloHandAsymFFPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Repose-Cube-Tessolo-OpenAI-LSTM-Direct-v0",
    entry_point=f"{inhand_task_entry}.inhand_manipulation_env:InHandManipulationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tessolo_hand_env_cfg:TessoloHandOpenAIEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_lstm_cfg.yaml",
    },
)

### Vision

gym.register(
    id="Isaac-Repose-Cube-Tessolo-Vision-Direct-v0",
    entry_point=f"{__name__}.tessolo_hand_vision_env:TessoloHandVisionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tessolo_hand_vision_env:TessoloHandVisionEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TessoloHandVisionFFPPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_vision_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Repose-Cube-Tessolo-Vision-Direct-Play-v0",
    entry_point=f"{__name__}.tessolo_hand_vision_env:TessoloHandVisionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tessolo_hand_vision_env:TessoloHandVisionEnvPlayCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TessoloHandVisionFFPPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_vision_cfg.yaml",
    },
)
