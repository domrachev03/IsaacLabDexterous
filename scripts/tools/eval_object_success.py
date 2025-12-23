# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause
"""
Evaluate per-object success rate for Dexsuite Panda RoHand lift policy.

This script runs the RL-Games player loop for a single object type and reports success rate.
Use --object-index to specify which object to evaluate (0-based index).
Use --list-objects to see all available object types.

Usage:
    # List all objects
    python scripts/tools/eval_object_success.py --task Isaac-Dexsuite-Panda-RoHand-Lift-Play-v0 --list-objects
    
    # Evaluate a specific object (index 0)
    python scripts/tools/eval_object_success.py \
        --task Isaac-Dexsuite-Panda-RoHand-Lift-Play-v0 \
        --checkpoint logs/panda_rohand_reorient_pbt_agi.pth \
        --object-index 0 \
        --episodes 20 \
        --num-envs 8 \
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
parser = argparse.ArgumentParser(description="Evaluate per-object success rate for Dexsuite Panda RoHand Lift.")
parser.add_argument("--task", type=str, default="Isaac-Dexsuite-Panda-RoHand-Lift-Play-v0", help="Gym task id.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to RL-Games checkpoint (.pth).")
parser.add_argument("--episodes", type=int, default=10, help="Episodes to run.")
parser.add_argument("--num-envs", type=int, default=8, help="Parallel envs to run.")
parser.add_argument("--object-index", type=int, default=None, help="Index of object to evaluate (0-based).")
parser.add_argument("--list-objects", action="store_true", help="List all available object types and exit.")
parser.add_argument(
    "--agent", type=str, default="rl_games_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

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


def check_success(env, command_name: str, position_only: bool) -> torch.Tensor:
    """Check success based on command manager metrics.
    
    Success criteria (matching pose_commands.py visualization logic):
    - position_error < 0.05 m (5 cm)
    - orientation_error < 0.5 rad (if not position_only)
    """
    command_term = env.command_manager.get_term(command_name)
    pos_error = command_term.metrics["position_error"]
    
    success = pos_error < 0.05
    if not position_only:
        rot_error = command_term.metrics["orientation_error"]
        success = success & (rot_error < 0.5)
    
    return success


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Evaluate per-object success rates."""
    
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Get the list of assets
    original_assets_cfg = copy.deepcopy(env_cfg.scene.object.spawn.assets_cfg)
    
    # Handle --list-objects
    if args_cli.list_objects:
        print("\nAvailable object types:")
        print("-" * 50)
        for i, asset_cfg in enumerate(original_assets_cfg):
            print(f"  {i}: {format_asset_name(asset_cfg)}")
        print("-" * 50)
        print(f"Total: {len(original_assets_cfg)} objects")
        return
    
    # Validate arguments
    if args_cli.checkpoint is None:
        print("ERROR: --checkpoint is required when evaluating")
        return
        
    if args_cli.object_index is None:
        print("ERROR: --object-index is required when evaluating")
        print("Use --list-objects to see available objects")
        return
    
    if args_cli.object_index < 0 or args_cli.object_index >= len(original_assets_cfg):
        print(f"ERROR: object-index {args_cli.object_index} out of range [0, {len(original_assets_cfg)-1}]")
        return
    
    # Get the selected asset
    asset_cfg = original_assets_cfg[args_cli.object_index]
    asset_name = format_asset_name(asset_cfg)
    print(f"\n[INFO] Evaluating object {args_cli.object_index}: {asset_name}")

    # Lock to a single asset type
    env_cfg.scene.object.spawn.random_choice = False
    env_cfg.scene.object.spawn.assets_cfg = [copy.deepcopy(asset_cfg)]

    # Check if position_only mode (for success evaluation)
    position_only = getattr(env_cfg.commands.object_pose, "position_only", True)

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

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

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

    # Tracking
    successes = 0
    total_episodes = 0
    target_episodes = args_cli.episodes
    step_count = 0
    max_steps = target_episodes * env.unwrapped.max_episode_length * 2  # Safety limit
    
    print(f"[INFO] Running {target_episodes} episodes...")

    # Run evaluation loop
    while total_episodes < target_episodes and step_count < max_steps and simulation_app.is_running():
        step_count += 1

        # run everything in inference mode
        with torch.inference_mode():
            # Check success BEFORE step (current state of the episode)
            # Success = position_error < 0.05m AND (if not position_only) orientation_error < 0.5rad
            current_success = check_success(env.unwrapped, "object_pose", position_only)
            
            # convert obs to agent format
            obs = agent.obs_to_torch(obs)
            # agent stepping
            actions = agent.get_action(obs, is_deterministic=True)
            # env stepping
            obs, _, dones, _ = env.step(actions)

            # Check for completed episodes (timeout or out-of-bounds)
            done_mask = env.unwrapped.reset_buf
            if done_mask.any():
                # Count successes for episodes that just ended
                success_mask = current_success & done_mask
                successes += int(success_mask.sum().item())
                total_episodes += int(done_mask.sum().item())
                
                # Reset episode success tracker for envs that just reset
                current_success = current_success & ~done_mask

                # Reset RNN states for terminated episodes
                if agent.is_rnn and agent.states is not None:
                    for s in agent.states:
                        s[:, done_mask, :] = 0.0

            if isinstance(obs, dict):
                obs = obs["obs"]

    # Close environment
    env.close()

    # Print results
    rate = successes / total_episodes if total_episodes > 0 else 0.0
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Object:        {asset_name}")
    print(f"Object Index:  {args_cli.object_index}")
    print(f"Successes:     {successes}")
    print(f"Total:         {total_episodes}")
    print(f"Success Rate:  {rate:.2%}")
    print("=" * 60)
    
    # Output in parseable format for shell scripts
    print(f"\nPARSEABLE_RESULT:{args_cli.object_index}:{asset_name}:{successes}:{total_episodes}:{rate:.4f}")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
