# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Frozen success-rate evaluator for RL-Games checkpoints on Dexsuite tasks.

This is intentionally a SINGLE, checkpoint-agnostic script (not one script per task/checkpoint).
It loads one checkpoint, runs deterministic inference (no exploration noise) for a fixed number
of episodes over explicitly-provided seeds (so held-out seeds are possible), and reports a single,
frozen success definition alongside a Wilson 95% confidence interval and a mutually-exclusive
outcome breakdown.

Success definition (frozen; see mdp/commands/pose_commands.py::ObjectUniformPoseCommand._update_metrics):
    success := position_error < POSITION_SUCCESS_THRESHOLD
                and (position_only or orientation_error < ORIENTATION_SUCCESS_THRESHOLD)
    evaluated as an OR across every simulation step of the episode (sticky - "did the object ever
    reach the goal pose"), matching mdp.recorders.DexsuiteSuccessRecorder.

    IMPORTANT: this script computes position_error/orientation_error ITSELF from the robot/object
    state and the command term's command buffer (see `_sample_pose_error` below) instead of reading
    `command_term.metrics` after `env.step()` returns. `ManagerBasedRLEnv.step()` auto-resets and
    re-samples the command for any env whose episode just ended *before* returning, and recomputes
    `command_term.metrics` for those envs against the fresh post-reset state - so metrics read after
    step() describe the wrong episode for exactly the envs we care about (see the comment at the read
    site below for the full explanation). `mdp.recorders.DexsuiteSuccessRecorder` avoids this by
    hooking `record_post_step()`, which IsaacLab runs before the reset; there is no equivalent public
    hook available to this external evaluator, so it replicates the recorder's own pose-error math
    (`combine_frame_transforms` + `compute_pose_error`) and samples it before calling `env.step()`
    each iteration instead.

This threshold pair is intentionally NOT read from mdp.DifficultyScheduler's pos_tol/rot_tol
(mdp/curriculums.py), because those are a curriculum promotion/demotion control signal derived
from RewardsCfg.success.params (pos_std/2, rot_std/2) that can drift independently of the command
term's own thresholds if the reward-shaping params get retuned. See the R1 task report for the
full reconciliation between the two definitions.

Conventions (CLI args, AppLauncher setup, hydra task config, checkpoint resolution, rl_games
wiring) are copied from scripts/reinforcement_learning/rl_games/play.py.

Runtime constraints this script is written for:
    - Exactly ONE gym.make() call per process (safe under `timeout -s KILL` wrapping; the kit app
      ignores SIGTERM so do not rely on graceful shutdown across multiple env instances).
    - No GPU is required to read this script, but running it requires Isaac Sim / IsaacLab.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Frozen success-rate evaluation of an RL-Games checkpoint.")
parser.add_argument("--task", type=str, required=True, help="Name of the task (a *-Play-v0 variant is recommended).")
parser.add_argument(
    "--agent", type=str, default="rl_games_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint to evaluate.")
parser.add_argument(
    "--seeds",
    type=int,
    nargs="+",
    required=True,
    help=(
        "One or more explicit RNG seeds to evaluate on (e.g. held-out seeds not used during training)."
        " Episodes are split as evenly as possible across the given seeds."
    ),
)
parser.add_argument(
    "--min_episodes",
    type=int,
    default=1024,
    help="Minimum number of completed episodes to evaluate (hard floor of 1024, per spec).",
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel environments to simulate.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument(
    "--out",
    type=str,
    default=None,
    help="Path to write the JSON summary to. Defaults to '<checkpoint_dir>/<checkpoint_stem>.eval_summary.json'.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

if args_cli.min_episodes < 1024:
    print(
        f"[EVAL] WARNING: --min_episodes={args_cli.min_episodes} is below the required floor of 1024;"
        " clamping to 1024."
    )
    args_cli.min_episodes = 1024
if len(args_cli.seeds) == 0:
    raise ValueError("--seeds must contain at least one seed.")

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import hashlib
import json
import math
import os
import time

import torch

from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from isaaclab.envs import DirectMARLEnv, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

# ---------------------------------------------------------------------------
# Frozen success definition constants.
#
# These MUST mirror the literals in
# source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/dexsuite/mdp/commands/pose_commands.py
# (ObjectUniformPoseCommand._update_metrics). They are duplicated here (rather than imported) because
# the command term computes its threshold check as a local variable, not a stored/public attribute -
# there is nothing to import. If those literals ever change, this evaluator's frozen definition drifts
# from the live visualizer's definition and this comment (and the one at the top of this file) must be
# updated together.
# ---------------------------------------------------------------------------
POSITION_SUCCESS_THRESHOLD = 0.05  # meters
ORIENTATION_SUCCESS_THRESHOLD = 0.5  # radians

# Termination term names from dexsuite_env_cfg.py::TerminationsCfg. Any completed episode whose
# ending step does not match "time_out" or "object_out_of_bound" (and was not a sticky success)
# falls into "other" - this keeps the four buckets exhaustive and mutually exclusive by construction.
TIME_OUT_TERM = "time_out"
OUT_OF_BOUNDS_TERM = "object_out_of_bound"

Z_95 = 1.959963984540054  # two-sided 95% normal quantile


def wilson_interval(successes: int, total: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion."""
    if total == 0:
        return (0.0, 0.0)
    phat = successes / total
    denom = 1.0 + z * z / total
    center = phat + z * z / (2 * total)
    adj = z * math.sqrt(phat * (1.0 - phat) / total + z * z / (4 * total * total))
    lower = (center - adj) / denom
    upper = (center + adj) / denom
    return (max(0.0, lower), min(1.0, upper))


def _sample_pose_error(command_term, robot_asset, object_asset) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute (position_error, orientation_error) between the object and the active command, right now.

    This mirrors mdp.commands.pose_commands.py::ObjectUniformPoseCommand._update_metrics and
    mdp.recorders.py::DexsuiteSuccessRecorder.record_post_step() bit-for-bit (same
    combine_frame_transforms + compute_pose_error math over the same buffers), so the frozen success
    definition stays numerically identical to both the live command term and the training-time
    recorder. See the module docstring and the call site for why the evaluator computes this itself
    instead of reading `command_term.metrics`.
    """
    des_pos_w, des_quat_w = combine_frame_transforms(
        robot_asset.data.root_pos_w,
        robot_asset.data.root_quat_w,
        command_term.command[:, :3],
        command_term.command[:, 3:7],
    )
    pos_err, rot_err = compute_pose_error(
        des_pos_w, des_quat_w, object_asset.data.root_pos_w, object_asset.data.root_quat_w
    )
    return torch.norm(pos_err, dim=-1), torch.norm(rot_err, dim=-1)


def sha256_of_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    """SHA-256 of a file's contents, following symlinks to the real target."""
    real_path = os.path.realpath(path)
    digest = hashlib.sha256()
    with open(real_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: dict):
    """Evaluate one checkpoint deterministically and report a frozen success rate."""
    # -- resolve and hash the checkpoint (before touching sim state) --
    resume_path = retrieve_file_path(args_cli.checkpoint)
    checkpoint_sha256 = sha256_of_file(resume_path)
    print(f"[EVAL] Checkpoint: {resume_path}")
    print(f"[EVAL] Checkpoint SHA-256: {checkpoint_sha256}")

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # seed used only for env construction; every seed block below reseeds explicitly before its episodes
    env_cfg.seed = args_cli.seeds[0]
    agent_cfg["params"]["seed"] = args_cli.seeds[0]

    log_dir = os.path.dirname(os.path.dirname(resume_path))
    env_cfg.log_dir = log_dir

    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

    # -- single gym.make() call for the whole process --
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)

    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs

    runner = Runner()
    runner.load(agent_cfg)
    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()

    # -- resolve the frozen success definition against this task's own command term --
    command_term = env.unwrapped.command_manager.get_term("object_pose")
    position_only: bool = bool(command_term.cfg.position_only)
    # Resolved once and reused every iteration by `_sample_pose_error` (see its docstring and the read
    # site in the eval loop below for why we compute pose error ourselves rather than reading
    # `command_term.metrics` after `env.step()`).
    robot_asset = env.unwrapped.scene["robot"]
    object_asset = env.unwrapped.scene["object"]
    print(
        f"[EVAL] Success definition: position_error < {POSITION_SUCCESS_THRESHOLD} m"
        + ("" if position_only else f" AND orientation_error < {ORIENTATION_SUCCESS_THRESHOLD} rad")
        + " (sticky across the episode)"
    )

    termination_manager = env.unwrapped.termination_manager
    missing_terms = {TIME_OUT_TERM, OUT_OF_BOUNDS_TERM} - set(termination_manager.active_terms)
    if missing_terms:
        raise RuntimeError(
            f"Expected termination terms {TIME_OUT_TERM!r} and {OUT_OF_BOUNDS_TERM!r} on task {args_cli.task!r};"
            f" found {termination_manager.active_terms!r}. The outcome breakdown below is frozen against these"
            " exact term names and must be updated if the task's TerminationsCfg changes."
        )

    sim_device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs
    sticky_success = torch.zeros(num_envs, dtype=torch.bool, device=sim_device)

    outcome_counts = {"success": 0, "timeout": 0, "out_of_bounds": 0, "other": 0}
    per_seed_counts: dict[int, dict[str, int]] = {}

    target_total = args_cli.min_episodes
    num_seeds = len(args_cli.seeds)
    per_seed_target = math.ceil(target_total / num_seeds)

    print(
        f"[EVAL] Evaluating {target_total}+ episodes across {num_seeds} seed(s)"
        f" (~{per_seed_target} episodes/seed), num_envs={num_envs}, deterministic inference."
    )

    start_time = time.time()
    total_completed = 0
    batch_size_initialized = False

    for seed in args_cli.seeds:
        if total_completed >= target_total:
            break
        # explicit, reproducible reseed for this block (held-out seeds are just any int here)
        env.seed(seed)
        obs = env.reset()
        if isinstance(obs, dict):
            obs = obs["obs"]
        if not batch_size_initialized:
            # required once (mirrors play.py): enables the flag for batched observations
            _ = agent.get_batch_size(obs, 1)
            batch_size_initialized = True
        agent.reset()
        if agent.is_rnn:
            agent.init_rnn()
        sticky_success[:] = False
        seed_completed = 0
        seed_outcome_counts = {"success": 0, "timeout": 0, "out_of_bounds": 0, "other": 0}

        while seed_completed < per_seed_target and total_completed < target_total:
            with torch.inference_mode():
                # Sample pose error BEFORE stepping, and OR it into the sticky per-episode success flag.
                #
                # ORDERING HAZARD - do not move this read to after env.step(): ManagerBasedRLEnv.step()
                # (manager_based_rl_env.py) auto-resets and re-samples the command for any env whose
                # episode ends this call *before returning* - in order it does
                # termination_manager.compute() -> reward/obs -> recorder_manager.record_post_step()
                # [pre-reset, the correct point; this is what mdp.recorders.DexsuiteSuccessRecorder
                # hooks] -> record_pre_reset() + _reset_idx() [object re-placed, command re-sampled] ->
                # command_manager.compute() for ALL envs, including the ones just reset. So
                # `command_term.metrics` (and the live robot/object state) read AFTER env.step() returns
                # describe the freshly-reset state for exactly the envs whose episode just ended, not the
                # terminal state of the episode that ended - silently undercounting late successes and
                # spuriously granting others. There is no public hook equivalent to
                # record_post_step() available to this external evaluator, so instead we replicate its
                # pose-error math ourselves (`_sample_pose_error`) and read it here, before step(): this
                # state has not been touched by any reset/resample yet this iteration, so it's always
                # valid. For continuing envs this is exactly the previous step's (uncorrupted) result;
                # for the iteration in which an episode actually ends it is the state one physics step
                # before termination rather than the true terminal one (the true post-physics-pre-reset
                # state is only observable from inside step()) - a bounded, at-most-one-step staleness,
                # not the unbounded corruption of reading a randomly re-placed object against a freshly
                # resampled goal.
                pos_dist, rot_dist = _sample_pose_error(command_term, robot_asset, object_asset)
                step_success = pos_dist < POSITION_SUCCESS_THRESHOLD
                if not position_only:
                    step_success &= rot_dist < ORIENTATION_SUCCESS_THRESHOLD
                sticky_success |= step_success

                obs_t = agent.obs_to_torch(obs)
                # deterministic inference: no exploration noise, regardless of the agent's own config
                actions = agent.get_action(obs_t, is_deterministic=True)
                obs, _rewards, dones, _infos = env.step(actions)
                # `dones` lives on rl_device (see RlGamesVecEnvWrapper.step); keep it separate from the
                # sim-device mask below since agent.states (RNN) live on rl_device too.
                dones_rl = dones

                done_mask = dones_rl.to(sim_device)
                if done_mask.any():
                    done_ids = done_mask.nonzero(as_tuple=True)[0]
                    is_success = sticky_success[done_ids]
                    is_timeout = termination_manager.get_term(TIME_OUT_TERM)[done_ids]
                    is_oob = termination_manager.get_term(OUT_OF_BOUNDS_TERM)[done_ids]

                    for succ, tout, oob in zip(is_success.tolist(), is_timeout.tolist(), is_oob.tolist()):
                        if succ:
                            bucket = "success"
                        elif tout:
                            bucket = "timeout"
                        elif oob:
                            bucket = "out_of_bounds"
                        else:
                            bucket = "other"
                        outcome_counts[bucket] += 1
                        seed_outcome_counts[bucket] += 1

                    n_done = int(done_mask.sum().item())
                    seed_completed += n_done
                    total_completed += n_done

                    # this episode has ended for these envs; the env auto-resets them internally
                    sticky_success[done_ids] = False

                if agent.is_rnn and agent.states is not None:
                    for s in agent.states:
                        s[:, dones_rl, :] = 0.0

        per_seed_counts[seed] = seed_outcome_counts
        print(
            f"[EVAL] seed={seed}: {seed_completed} episodes"
            f" (success={seed_outcome_counts['success']}, timeout={seed_outcome_counts['timeout']},"
            f" out_of_bounds={seed_outcome_counts['out_of_bounds']}, other={seed_outcome_counts['other']})"
        )

    elapsed = time.time() - start_time

    denominator = sum(outcome_counts.values())
    numerator = outcome_counts["success"]
    success_rate = numerator / denominator if denominator > 0 else 0.0
    ci_lo, ci_hi = wilson_interval(numerator, denominator)

    print("\n" + "=" * 70)
    print("[EVAL] Frozen success-rate evaluation")
    print("=" * 70)
    print(f"  Task:                 {args_cli.task}")
    print(f"  Checkpoint:           {resume_path}")
    print(f"  Checkpoint SHA-256:   {checkpoint_sha256}")
    print(f"  Seeds:                {args_cli.seeds}")
    print(f"  Elapsed:              {elapsed:.1f} s")
    print("-" * 70)
    print(f"  Numerator (success):  {numerator}")
    print(f"  Denominator (total):  {denominator}")
    print(f"  Success rate:         {success_rate:.4f}")
    print(f"  Wilson 95% CI:        [{ci_lo:.4f}, {ci_hi:.4f}]")
    print("-" * 70)
    print("  Outcome breakdown (mutually exclusive, sums to denominator):")
    for key in ("success", "timeout", "out_of_bounds", "other"):
        frac = outcome_counts[key] / denominator if denominator > 0 else 0.0
        print(f"    {key:<13} {outcome_counts[key]:>6}  ({frac:.2%})")
    assert sum(outcome_counts.values()) == denominator  # exhaustiveness sanity check
    print("=" * 70)

    summary = {
        "task": args_cli.task,
        "checkpoint_path": resume_path,
        "checkpoint_sha256": checkpoint_sha256,
        "seeds": args_cli.seeds,
        "num_envs": num_envs,
        "deterministic": True,
        "success_definition": {
            "position_threshold_m": POSITION_SUCCESS_THRESHOLD,
            "orientation_threshold_rad": None if position_only else ORIENTATION_SUCCESS_THRESHOLD,
            "position_only": position_only,
            "sticky_across_episode": True,
            "source": "mdp/commands/pose_commands.py::ObjectUniformPoseCommand._update_metrics",
        },
        "numerator": numerator,
        "denominator": denominator,
        "success_rate": success_rate,
        "wilson_95ci": {"lower": ci_lo, "upper": ci_hi},
        "outcome_breakdown": outcome_counts,
        "per_seed_outcome_breakdown": per_seed_counts,
        "elapsed_seconds": elapsed,
        "timestamp_unix": time.time(),
    }

    out_path = args_cli.out
    if out_path is None:
        checkpoint_stem = os.path.splitext(os.path.basename(resume_path))[0]
        out_path = os.path.join(os.path.dirname(resume_path), f"{checkpoint_stem}.eval_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[EVAL] Wrote JSON summary to: {out_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
