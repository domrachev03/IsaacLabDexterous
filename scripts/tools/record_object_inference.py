#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause
"""
Run inference for a single object and record a video.

This script is modeled after `eval_object_success.py` but wraps the environment with
`gym.wrappers.RecordVideo` to save a video of the agent running on a single object type.

Usage:
    # Record video for object 0
    python scripts/tools/record_object_inference.py \
        --task Isaac-Dexsuite-Panda-RoHand-Lift-Play-v0 \
        --checkpoint logs/panda_rohand_reorient_pbt_agi.pth \
        --object-index 0 \
        --episodes 1 \
        --num-envs 1 \
        --video-folder out/videos \
        --video-length 800 \
        --headless
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import sys

import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Record one-object inference video (RL-Games).")
parser.add_argument("--task", type=str, default="Isaac-Dexsuite-Panda-RoHand-Lift-Play-v0", help="Gym task id.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to RL-Games checkpoint (.pth).")
parser.add_argument("--episodes", type=int, default=1, help="Episodes to run (videos to record).")
parser.add_argument("--num-envs", type=int, default=1, help="Parallel envs to run.")
parser.add_argument("--object-index", type=int, default=None, help="Index of object to record (0-based).")
parser.add_argument("--video-folder", type=str, default="videos", help="Folder to save recorded videos.")
parser.add_argument("--video-length", type=int, default=800, help="Max video length in steps.")
parser.add_argument(
    "--agent", type=str, default="rl_games_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# If recording we need cameras enabled
if args_cli.video_folder:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym

from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config


def format_asset_name(asset_cfg) -> str:
    """Human-readable asset label."""
    cfg_type = type(asset_cfg).__name__
    if hasattr(asset_cfg, "size"):
        size = getattr(asset_cfg, "size")
        return f"{cfg_type}[{', '.join(f'{v:.3f}' for v in size)}]"
    if hasattr(asset_cfg, "radius"):
        radius = getattr(asset_cfg, "radius")
        height = getattr(asset_cfg, "height", None)
        if height is not None:
            return f"{cfg_type}[r={radius:.3f}, h={height:.3f}]"
        return f"{cfg_type}[r={radius:.3f}]"
    return cfg_type


def format_asset_name_for_filename(asset_cfg, index: int) -> str:
    """Generate a clean filename from asset config."""
    cfg_type = type(asset_cfg).__name__
    
    # Remove 'Cfg' suffix if present
    if cfg_type.endswith("Cfg"):
        cfg_type = cfg_type[:-3]
    
    if hasattr(asset_cfg, "size"):
        size = getattr(asset_cfg, "size")
        size_str = "_".join(f"{v:.4f}" for v in size)
        return f"{cfg_type}_{size_str}_{index:02d}"
    if hasattr(asset_cfg, "radius"):
        radius = getattr(asset_cfg, "radius")
        height = getattr(asset_cfg, "height", None)
        if height is not None:
            return f"{cfg_type}_r{radius:.4f}_h{height:.4f}_{index:02d}"
        return f"{cfg_type}_r{radius:.4f}_{index:02d}"
    return f"{cfg_type}_{index:02d}"


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Record inference video for a single object."""
    
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Get the list of assets
    original_assets_cfg = copy.deepcopy(env_cfg.scene.object.spawn.assets_cfg)
    
    # Validate arguments
    if args_cli.checkpoint is None:
        print("ERROR: --checkpoint is required")
        return
        
    if args_cli.object_index is None:
        print("ERROR: --object-index is required")
        return
    
    if args_cli.object_index < 0 or args_cli.object_index >= len(original_assets_cfg):
        print(f"ERROR: object-index {args_cli.object_index} out of range [0, {len(original_assets_cfg)-1}]")
        return
    
    # Get the selected asset
    asset_cfg = original_assets_cfg[args_cli.object_index]
    asset_name = format_asset_name(asset_cfg)
    asset_filename = format_asset_name_for_filename(asset_cfg, args_cli.object_index)
    print(f"\n[INFO] Recording object {args_cli.object_index}: {asset_name}")

    # Lock to a single asset type
    env_cfg.scene.object.spawn.random_choice = False
    env_cfg.scene.object.spawn.assets_cfg = [copy.deepcopy(asset_cfg)]

    # Setup agent config
    if not isinstance(agent_cfg, dict):
        agent_cfg = agent_cfg.to_dict()

    agent_cfg["params"]["config"]["device"] = args_cli.device
    agent_cfg["params"]["config"]["device_name"] = args_cli.device
    agent_cfg["params"]["config"]["num_actors"] = args_cli.num_envs

    # Set checkpoint
    resume_path = retrieve_file_path(args_cli.checkpoint)
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # wrap around environment for rl-games
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

    # create isaac environment with render_mode for video recording
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording BEFORE rl-games wrapper
    # Use custom name_prefix to create videos like "BoxCfg_0.mp4"
    log_dir = os.path.abspath(args_cli.video_folder)
    os.makedirs(log_dir, exist_ok=True)
    video_kwargs = {
        "video_folder": log_dir,
        "step_trigger": lambda step: step == 0,
        "video_length": args_cli.video_length,
        "disable_logger": True,
        "name_prefix": asset_filename,
    }
    print(f"[INFO] Recording video to: {log_dir}/{asset_filename}.mp4")
    env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rl-games
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)

    # register the environment to rl-games registry
    vecenv.register(
        "IsaacRlgWrapper",
        lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs

    # create runner from rl-games
    runner = Runner()
    runner.load(agent_cfg)

    # obtain the agent from the runner
    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()

    # reset environment
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]

    # required: enables the flag for batched observations
    _ = agent.get_batch_size(obs, 1)

    # initialize RNN states if used
    if agent.is_rnn:
        agent.init_rnn()

    # Run inference and record video
    episodes_recorded = 0
    step_count = 0
    max_steps = args_cli.episodes * args_cli.video_length * 2  # Safety limit
    
    print(f"[INFO] Running up to {args_cli.episodes} episodes (recording)...")

    # Run evaluation loop
    while episodes_recorded < args_cli.episodes and step_count < max_steps and simulation_app.is_running():
        step_count += 1

        # run everything in inference mode
        with torch.inference_mode():
            # convert obs to agent format
            obs = agent.obs_to_torch(obs)
            # agent stepping
            actions = agent.get_action(obs, is_deterministic=True)
            # env stepping
            obs, _, dones, _ = env.step(actions)

            # Check for completed episodes
            done_mask = env.unwrapped.reset_buf
            if done_mask.any():
                episodes_recorded += int(done_mask.sum().item())
                
                # Reset RNN states for terminated episodes
                if agent.is_rnn and agent.states is not None:
                    for s in agent.states:
                        s[:, done_mask, :] = 0.0

            if isinstance(obs, dict):
                obs = obs["obs"]

    # Close environment
    env.close()
    
    print(f"\n[INFO] Recording complete. {episodes_recorded} episodes recorded.")
    print(f"[INFO] Videos saved to: {log_dir}")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
