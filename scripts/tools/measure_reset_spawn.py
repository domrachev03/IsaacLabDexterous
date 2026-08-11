# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause
"""Measure where the object actually spawns relative to the hand AT RESET.

Bead: UWLab-ctk.5

WHY THIS EXISTS
----------------
Two opposite failure modes have to be told apart, and until now nobody measured either:

  * TOO CLOSE: ``EventCfg.reset_object`` (see ``dexsuite_env_cfg.py``) samples the object pose
    uniformly around its default position with **no anti-overlap check whatsoever**, while
    ``reset_robot_joints`` / ``reset_robot_wrist_joint`` independently perturb every arm joint.
    An object that lands interpenetrating the hand at reset is a free grasp that never had to be
    learned.
  * TOO FAR: a previous campaign (different codebase) measured the nearest fingertip starting
    0.409 m from the object, and after a partial fix, still 0.170 m -- with zero initial contact
    in both cases. That gap is what stops contact ever being explored by the policy.

This script builds the env from its REGISTERED config exactly as training would (only
``num_envs``/``device``/``use_fabric`` are ever overridden, matching the convention used by
``scripts/environments/zero_agent.py``), resets it, steps it ONCE so contact sensors are
populated, and reports:

  * Per-env min distance from the object to each of the 5 fingertip bodies
    (``rl_dg_1_tip`` .. ``rl_dg_5_tip``) and to the palm/mount body (``rl_dg_mount``), reported
    TWO ways, clearly labeled:
      - "center_to_center": ``||body_pos_w - object_root_pos_w||``. This is NOT a surface
        distance -- it ignores object size entirely and is included only for reference.
      - "shape_aware_surface": distance from the body position to the surface of the object's
        oriented bounding box (OBB), estimated from the object's authored USD geometry
        (Cube/Sphere/Capsule/Cone/Mesh extents) plus its per-env rigid-body scale. This is a
        best-effort estimate (axis-aligned box vs. true collision mesh) -- see
        ``shape_aware_available_envs`` in the JSON output for how many envs it succeeded on. If
        it fails for an env, that env's shape-aware numbers are NaN and it is excluded from the
        shape-aware summary stats (falls back visibly, never silently, to center-to-center only).
      A shape-aware distance of exactly 0 means the body position lies ON/INSIDE the object's
      OBB, i.e. overlap/penetration -- this is the number that actually answers "TOO CLOSE".
  * Summary stats (min/p1/p5/median/p95/p99/max/mean) + a text histogram for "distance to
    nearest fingertip", for both distance flavors above.
  * Fraction of envs below {0, 1, 2, 5} cm and above {10, 20, 40} cm (shape-aware primary,
    center-to-center for comparison).
  * Count/fraction of envs in actual CONTACT at reset, read from the per-fingertip
    ``ContactSensor`` s the ur10_tessolo config declares (``rl_dg_*_tip_object_s``), via
    ``data.force_matrix_w`` (filtered specifically to the Object prim), AFTER one ``env.step()``
    so contacts are populated. This is explicitly a SENSOR-based number, not a geometry number.
  * The same style of stats for the object's distance to the table surface (naive center-z and
    shape-aware lowest-OBB-corner), so "floating in the air" is visible.
  * A JSON summary written to ``--out_json``.
  * A clear verdict line: overlap tail / too-far tail / both / neither, against the thresholds
    above.

Usage:
    python scripts/tools/measure_reset_spawn.py \
        --task Isaac-Dexsuite-UR10-Tessolo-Lift-v0 \
        --num_envs 4096 \
        --out_json reports/reset_spawn_ur10_tessolo_lift.json \
        --headless
"""

from __future__ import annotations

import argparse
import itertools
import json
import os

from isaaclab.app import AppLauncher

# ---------------------------------------------------------------------------
# CLI args (Isaac Sim must be launched before importing anything torch/omni).
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Measure object-vs-hand spawn distances at reset.")
parser.add_argument(
    "--task", type=str, default="Isaac-Dexsuite-UR10-Tessolo-Lift-v0", help="Name of the registered gym task."
)
parser.add_argument("--num_envs", type=int, default=4096, help="Number of parallel envs to reset and measure.")
parser.add_argument("--seed", type=int, default=None, help="Seed for the environment (reproducible reset sampling).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument(
    "--contact_force_threshold",
    type=float,
    default=0.2,
    help=(
        "Contact-force magnitude (N) above which a fingertip-object contact is counted as"
        " 'meaningful'. Defaults to 0.2 N, matching the threshold used by this env's"
        " good_finger_contact/any_finger_contact reward terms."
    ),
)
parser.add_argument(
    "--out_json",
    type=str,
    default="reports/measure_reset_spawn.json",
    help="Path to write the JSON summary to.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import torch

import gymnasium as gym

from isaaclab.utils.math import quat_apply, quat_apply_inverse

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

# names declared by dexsuite_ur10_tessolo_env_cfg.py (UR10TessoloMixinCfg.__post_init__)
FINGERTIP_BODY_NAMES = ["rl_dg_1_tip", "rl_dg_2_tip", "rl_dg_3_tip", "rl_dg_4_tip", "rl_dg_5_tip"]
MOUNT_BODY_NAME = "rl_dg_mount"
ALL_BODY_NAMES = FINGERTIP_BODY_NAMES + [MOUNT_BODY_NAME]

# 8 sign combinations for enumerating the corners of a box from its half-extents.
_CORNER_SIGNS = torch.tensor(list(itertools.product([-1.0, 1.0], repeat=3)), dtype=torch.float32)  # (8, 3)


# ---------------------------------------------------------------------------
# Shape-aware object geometry (best effort; USD-derived, computed once).
# ---------------------------------------------------------------------------
def compute_object_obb_local(object_asset, device: str) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Best-effort per-env object oriented-bounding-box, expressed in the object's own local frame.

    The object's geometry (which primitive was picked by ``MultiAssetSpawnerCfg``) and its
    per-env rigid-body scale (``EventCfg.randomize_object_scale``, mode="prestartup") are both
    fixed for the lifetime of the environment -- they are set once before the first reset and
    never change again. So this is computed once and reused across all reset samples.

    Mirrors the prim-traversal / relative-transform pattern already used by
    ``dexsuite.mdp.utils.sample_object_point_cloud`` (same repo) for the same object asset, so
    that the (non-obvious) handling of the object root's own ``xformOp:scale`` -- which the
    prim-to-root relative transform does NOT include, since it cancels out against the root's own
    world transform -- is applied consistently with that already-shipped code path.

    Args:
        object_asset: The ``RigidObject`` scene entity for "object".
        device: Torch device to place the output tensors on.

    Returns:
        half_extents: ``(num_envs, 3)`` box half-size along the object's own local axes. NaN rows
            mark envs where the computation failed for that env (see ``n_failed``).
        center_offset_local: ``(num_envs, 3)`` offset of the box center from the object's rigid-body
            origin (``root_pos_w``), expressed in the object's local frame. NaN where failed.
        n_ok: number of envs the computation succeeded for.
        n_failed: number of envs it failed for (best-effort fallback -- those envs get NaN and
            are excluded from shape-aware summary stats downstream).
    """
    import isaacsim.core.utils.prims as prim_utils
    from pxr import Usd, UsdGeom

    from isaaclab.sim.utils import get_all_matching_child_prims

    prim_paths = list(object_asset.root_physx_view.prim_paths)
    num_envs = len(prim_paths)
    half_extents = torch.full((num_envs, 3), float("nan"), dtype=torch.float32)
    center_offset = torch.full((num_envs, 3), float("nan"), dtype=torch.float32)

    xform_cache = UsdGeom.XformCache()
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide],
        False,  # useExtentsHint: force a fresh geometric compute, don't trust stale hints
    )

    n_ok = 0
    n_failed = 0
    for i, obj_path in enumerate(prim_paths):
        try:
            object_prim = prim_utils.get_prim_at_path(obj_path)
            world_root = xform_cache.GetLocalToWorldTransform(object_prim)
            geo_prims = get_all_matching_child_prims(
                obj_path,
                predicate=lambda p: p.GetTypeName() in ("Mesh", "Cube", "Sphere", "Cylinder", "Capsule", "Cone"),
            )
            if not geo_prims:
                n_failed += 1
                continue

            corners_root = []
            for prim in geo_prims:
                local_bbox = bbox_cache.ComputeUntransformedBound(prim)
                rng = local_bbox.ComputeAlignedRange()
                lo, hi = rng.GetMin(), rng.GetMax()
                # prim-local -> object-root-local (excludes the root's OWN transform, see note
                # above on xformOp:scale compensation below).
                rel = xform_cache.GetLocalToWorldTransform(prim) * world_root.GetInverse()
                for x in (lo[0], hi[0]):
                    for y in (lo[1], hi[1]):
                        for z in (lo[2], hi[2]):
                            p = rel.Transform((x, y, z))
                            corners_root.append([p[0], p[1], p[2]])

            pts = torch.tensor(corners_root, dtype=torch.float32)
            mn = pts.min(dim=0).values
            mx = pts.max(dim=0).values

            # root's own scale is cancelled out by `rel` above (it is common to both
            # GetLocalToWorldTransform(prim) and world_root when prim==root, and orthogonal to it
            # otherwise) -- so it has to be re-applied explicitly, exactly like
            # sample_object_point_cloud does for the same object asset.
            scale_attr = object_prim.GetAttribute("xformOp:scale")
            scale_val = scale_attr.Get() if (scale_attr and scale_attr.IsValid()) else None
            base_scale = (
                torch.tensor(scale_val, dtype=torch.float32) if scale_val is not None else torch.ones(3)
            )

            half_extents[i] = (mx - mn) / 2.0 * base_scale
            center_offset[i] = (mx + mn) / 2.0 * base_scale
            n_ok += 1
        except Exception as exc:  # best-effort: never let a single env's USD quirk kill the run
            print(f"[WARN] shape-aware OBB failed for env {i} ({obj_path}): {exc}")
            n_failed += 1
            continue

    return half_extents.to(device), center_offset.to(device), n_ok, n_failed


def point_to_obb_surface_distance(
    points_w: torch.Tensor, obb_center_w: torch.Tensor, obb_quat_w: torch.Tensor, half_extents: torch.Tensor
) -> torch.Tensor:
    """Euclidean distance from world-space point(s) to the surface of an oriented box.

    Returns exactly 0 when the point is ON or INSIDE the box (overlap/penetration).

    All args must be broadcastable to the same leading (batch) shape, with the last dim being 3
    for points/center/half_extents and 4 (wxyz) for the quaternion.
    """
    rel_w = points_w - obb_center_w
    rel_local = quat_apply_inverse(obb_quat_w, rel_w)
    outside = torch.clamp(rel_local.abs() - half_extents, min=0.0)
    return torch.linalg.norm(outside, dim=-1)


def obb_corners_w(
    obb_center_w: torch.Tensor, obb_quat_w: torch.Tensor, half_extents: torch.Tensor
) -> torch.Tensor:
    """World-space positions of the 8 corners of an oriented box. Returns ``(N, 8, 3)``."""
    num = obb_center_w.shape[0]
    signs = _CORNER_SIGNS.to(obb_center_w.device)  # (8, 3)
    corners_local = half_extents.unsqueeze(1) * signs.unsqueeze(0)  # (N, 8, 3)
    quat_exp = obb_quat_w.unsqueeze(1).expand(num, 8, 4)
    corners_w = quat_apply(quat_exp, corners_local) + obb_center_w.unsqueeze(1)
    return corners_w


# ---------------------------------------------------------------------------
# Stats / printing helpers
# ---------------------------------------------------------------------------
def summarize(x: np.ndarray) -> dict:
    """min/p1/p5/median/p95/p99/max/mean over the finite entries of ``x``."""
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return {k: None for k in ("min", "p1", "p5", "median", "p95", "p99", "max", "mean", "n")}
    return {
        "min": float(np.min(finite)),
        "p1": float(np.percentile(finite, 1)),
        "p5": float(np.percentile(finite, 5)),
        "median": float(np.percentile(finite, 50)),
        "p95": float(np.percentile(finite, 95)),
        "p99": float(np.percentile(finite, 99)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "n": int(finite.size),
    }


def threshold_fractions(x: np.ndarray) -> dict:
    """Fraction of (finite) entries below {0,1,2,5} cm and above {10,20,40} cm."""
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return {}
    n = finite.size
    out = {}
    for cm in (0.0, 1.0, 2.0, 5.0):
        out[f"frac_le_{cm:g}cm"] = float(np.sum(finite <= cm / 100.0) / n)
    for cm in (10.0, 20.0, 40.0):
        out[f"frac_ge_{cm:g}cm"] = float(np.sum(finite >= cm / 100.0) / n)
    return out


def print_ascii_histogram(x: np.ndarray, title: str, bins: int = 20, width: int = 50, unit_scale: float = 100.0):
    """Print a text histogram. ``unit_scale`` converts x (meters) to the printed unit (100 -> cm)."""
    finite = x[np.isfinite(x)]
    print(f"\n{title}")
    if finite.size == 0:
        print("  (no finite samples)")
        return
    counts, edges = np.histogram(finite, bins=bins)
    max_count = counts.max() if counts.max() > 0 else 1
    for c, lo, hi in zip(counts, edges[:-1], edges[1:]):
        bar = "#" * max(0, int(round(width * c / max_count)))
        print(f"  [{lo * unit_scale:7.2f}, {hi * unit_scale:7.2f}) cm | {bar} {c}")


def fmt_stats(stats: dict, unit_scale: float = 100.0, unit: str = "cm") -> str:
    if stats.get("n", 0) in (0, None):
        return "n=0 (no finite samples)"
    parts = []
    for key in ("min", "p1", "p5", "median", "p95", "p99", "max", "mean"):
        v = stats[key]
        parts.append(f"{key}={v * unit_scale:.2f}{unit}")
    return f"n={stats['n']} " + " ".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # construct the env from its REGISTERED cfg exactly as training would -- only num_envs /
    # device / use_fabric are ever touched here, matching scripts/environments/zero_agent.py.
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed

    env = gym.make(args_cli.task, cfg=env_cfg)
    env_u = env.unwrapped
    device = env_u.device
    num_envs = env_u.num_envs

    print(f"[INFO] Task: {args_cli.task} | num_envs: {num_envs} | device: {device}")

    # reset -- this is the exact sampler under test (reset_object / reset_robot_joints / ...).
    env.reset(seed=args_cli.seed)

    # step ONCE with zero action so contact sensors (which require a physics step to populate)
    # reflect the post-reset state, not stale/zeroed buffers.
    #
    # GUARD: ManagerBasedRLEnv.step() synchronously calls _reset_idx() for any env whose termination
    # fires on THIS step (see manager_based_rl_env.py's step(), which re-runs reset_object /
    # reset_robot_joints / reset_robot_wrist_joint before returning). If that happened here, the
    # positions read below would be from a SECOND reset, not the one under test, silently
    # contaminating every stat in this report. Assert it never does instead of assuming it won't.
    zero_actions = torch.zeros(env.action_space.shape, device=device)
    with torch.inference_mode():
        _, _, terminated, truncated, _ = env.step(zero_actions)
    n_reset_during_measurement = int((terminated | truncated).sum().item())
    assert n_reset_during_measurement == 0, (
        f"{n_reset_during_measurement}/{num_envs} envs reset DURING the measurement step (termination"
        " fired on the very first post-reset step), so their reported spawn positions are from a"
        " SECOND reset, not the reset under test -- this measurement is contaminated. This was dormant"
        " when this script was written (spawn ranges sit well inside object_out_of_bound), but is not"
        " guaranteed to stay that way. Investigate the termination/spawn ranges before trusting any"
        " numbers below."
    )

    robot = env_u.scene["robot"]
    object_asset = env_u.scene["object"]
    table_asset = env_u.scene["table"]

    # -- body lookups --
    body_ids, found_names = robot.find_bodies(ALL_BODY_NAMES, preserve_order=True)
    assert found_names == ALL_BODY_NAMES, (
        f"Expected robot bodies {ALL_BODY_NAMES}, articulation resolved {found_names}. The"
        " ur10_tessolo body-name convention (rl_dg_*_tip / rl_dg_mount) may have changed."
    )
    body_pos_w = robot.data.body_pos_w[:, body_ids, :]  # (N, 6, 3): 5 tips + mount

    object_pos_w = object_asset.data.root_pos_w  # (N, 3)
    object_quat_w = object_asset.data.root_quat_w  # (N, 4) wxyz

    # ---------------- center-to-center distances (NOT a surface distance) ----------------
    dist_center = torch.linalg.norm(body_pos_w - object_pos_w.unsqueeze(1), dim=-1)  # (N, 6)
    dist_center_tip = dist_center[:, :5]
    dist_center_mount = dist_center[:, 5]
    nearest_center_tip, _ = dist_center_tip.min(dim=1)

    # ---------------- shape-aware surface distances (best effort) ----------------
    half_extents_local, center_offset_local, n_ok, n_failed = compute_object_obb_local(object_asset, device)
    shape_aware_ok = torch.isfinite(half_extents_local).all(dim=-1)  # (N,) bool

    obb_center_w = object_pos_w + quat_apply(object_quat_w, center_offset_local)
    quat_exp6 = object_quat_w.unsqueeze(1).expand(num_envs, 6, 4)
    obb_center_exp6 = obb_center_w.unsqueeze(1).expand(num_envs, 6, 3)
    half_extents_exp6 = half_extents_local.unsqueeze(1).expand(num_envs, 6, 3)
    dist_shape = point_to_obb_surface_distance(body_pos_w, obb_center_exp6, quat_exp6, half_extents_exp6)  # (N,6)
    # mark failed envs as NaN across all bodies
    dist_shape[~shape_aware_ok] = float("nan")
    dist_shape_tip = dist_shape[:, :5]
    dist_shape_mount = dist_shape[:, 5]
    nearest_shape_tip, _ = torch.where(
        torch.isfinite(dist_shape_tip), dist_shape_tip, torch.full_like(dist_shape_tip, float("inf"))
    ).min(dim=1)
    nearest_shape_tip[~shape_aware_ok] = float("nan")

    print(
        f"[INFO] Shape-aware object OBB computed for {n_ok}/{num_envs} envs"
        f" ({n_failed} failed -> those envs fall back to center-to-center only, marked NaN"
        " in shape-aware fields)."
    )

    # ---------------- object-to-table clearance ----------------
    table_half_z = float(env_cfg.scene.table.spawn.size[2]) / 2.0  # table never rotates in this cfg
    table_top_z = table_asset.data.root_pos_w[:, 2] + table_half_z  # (N,)
    clearance_naive = object_pos_w[:, 2] - table_top_z  # center-z above table top; can go negative

    corners_w = obb_corners_w(obb_center_w, object_quat_w, half_extents_local)  # (N, 8, 3)
    lowest_corner_z = corners_w[..., 2].min(dim=1).values
    lowest_corner_z = torch.where(shape_aware_ok, lowest_corner_z, torch.full_like(lowest_corner_z, float("nan")))
    clearance_shape = lowest_corner_z - table_top_z

    # ---------------- contact sensors (SENSOR-based, not geometry) ----------------
    sensor_names = [f"{name}_object_s" for name in FINGERTIP_BODY_NAMES]
    force_mags = []
    for sname in sensor_names:
        sensor = env_u.scene.sensors[sname]
        force_w = sensor.data.force_matrix_w.view(num_envs, 3)  # filtered to Object prim only
        force_mags.append(torch.linalg.norm(force_w, dim=-1))
    force_mags = torch.stack(force_mags, dim=1)  # (N, 5)

    eps = 1e-4
    any_force_eps = (force_mags > eps).any(dim=1)
    any_force_thresh = (force_mags > args_cli.contact_force_threshold).any(dim=1)

    n_contact_eps = int(any_force_eps.sum().item())
    n_contact_thresh = int(any_force_thresh.sum().item())

    # =====================================================================
    # Reporting
    # =====================================================================
    center_np = nearest_center_tip.detach().cpu().numpy()
    shape_np = nearest_shape_tip.detach().cpu().numpy()
    mount_center_np = dist_center_mount.detach().cpu().numpy()
    mount_shape_np = dist_shape_mount.detach().cpu().numpy()
    clearance_naive_np = clearance_naive.detach().cpu().numpy()
    clearance_shape_np = clearance_shape.detach().cpu().numpy()

    print("\n" + "=" * 78)
    print("RESET SPAWN MEASUREMENT")
    print("=" * 78)
    print(f"Task: {args_cli.task} | envs: {num_envs} | seed: {args_cli.seed}")
    print(f"Bodies measured: {ALL_BODY_NAMES}")

    print("\n--- distance to NEAREST fingertip (of the 5 rl_dg_*_tip bodies) ---")
    print("[center-to-center; root-to-root, NOT a surface distance -- reference only]")
    print(" ", fmt_stats(summarize(center_np)))
    print("[shape-aware OBB-surface distance; 0 == overlap/penetration with the object]")
    print(" ", fmt_stats(summarize(shape_np)))

    print("\n--- distance to palm/mount body (rl_dg_mount) ---")
    print("[center-to-center]")
    print(" ", fmt_stats(summarize(mount_center_np)))
    print("[shape-aware OBB-surface]")
    print(" ", fmt_stats(summarize(mount_shape_np)))

    print_ascii_histogram(shape_np, "Histogram: nearest-fingertip SHAPE-AWARE surface distance (cm)")
    print_ascii_histogram(center_np, "Histogram: nearest-fingertip CENTER-TO-CENTER distance (cm), reference only")

    thr_shape = threshold_fractions(shape_np)
    thr_center = threshold_fractions(center_np)
    print("\n--- threshold fractions (shape-aware, primary) ---")
    for k, v in thr_shape.items():
        print(f"  {k}: {v:.4f}")
    print("--- threshold fractions (center-to-center, reference only) ---")
    for k, v in thr_center.items():
        print(f"  {k}: {v:.4f}")

    print("\n--- object <-> TABLE surface clearance ---")
    print("[naive: object root z - table top z; ignores object size]")
    print(" ", fmt_stats(summarize(clearance_naive_np)))
    print("[shape-aware: lowest OBB corner z - table top z]")
    print(" ", fmt_stats(summarize(clearance_shape_np)))
    print_ascii_histogram(clearance_shape_np, "Histogram: object-to-table clearance, shape-aware (cm)")

    print(
        "\n--- CONTACT at reset (source: ContactSensor.data.force_matrix_w, filtered to Object,"
        " after 1 env.step()) ---"
    )
    print(
        f"  envs with ANY measurable force (> {eps} N):        {n_contact_eps}/{num_envs}"
        f" ({n_contact_eps / num_envs:.4f})"
    )
    print(
        f"  envs with force > threshold ({args_cli.contact_force_threshold} N):"
        f"        {n_contact_thresh}/{num_envs} ({n_contact_thresh / num_envs:.4f})"
    )

    # ---------------- verdict ----------------
    frac_overlap = thr_shape.get("frac_le_0cm")
    frac_close_1cm = thr_shape.get("frac_le_1cm")
    frac_far_10cm = thr_shape.get("frac_ge_10cm")
    frac_far_20cm = thr_shape.get("frac_ge_20cm")

    has_overlap_tail = bool(frac_overlap is not None and frac_overlap > 0.0)
    has_too_far_tail = bool(frac_far_20cm is not None and frac_far_20cm > 0.05)

    if n_ok == 0:
        verdict = (
            "VERDICT: INCONCLUSIVE -- shape-aware OBB computation failed for every env; only"
            " center-to-center distances are available, and those cannot distinguish overlap"
            " from a merely-close-but-not-touching spawn. Re-run with the fix applied, or"
            " inspect the [WARN] lines above."
        )
    elif has_overlap_tail and has_too_far_tail:
        verdict = (
            f"VERDICT: BOTH tails present -- {frac_overlap:.2%} of envs spawn the object"
            f" OVERLAPPING a fingertip's OBB (shape-aware distance == 0, i.e. TOO CLOSE), AND"
            f" {frac_far_20cm:.2%} of envs spawn the nearest fingertip >= 20 cm away (TOO FAR)."
            f" (>=10 cm: {frac_far_10cm:.2%}, <=1 cm: {frac_close_1cm:.2%}.)"
        )
    elif has_overlap_tail:
        verdict = (
            f"VERDICT: OVERLAP tail present, no meaningful too-far tail -- {frac_overlap:.2%} of"
            f" envs spawn the object OVERLAPPING a fingertip's OBB at reset (shape-aware distance"
            f" == 0, TOO CLOSE / free grasp). Only {frac_far_20cm:.2%} of envs have the nearest"
            " fingertip >= 20 cm away."
        )
    elif has_too_far_tail:
        verdict = (
            f"VERDICT: TOO-FAR tail present, no overlap tail -- {frac_far_20cm:.2%} of envs spawn"
            f" the nearest fingertip >= 20 cm from the object (>=10cm: {frac_far_10cm:.2%}), with"
            " zero envs showing shape-aware overlap at reset."
        )
    else:
        verdict = (
            "VERDICT: NEITHER tail present by the checked thresholds -- 0% shape-aware overlap"
            f" (TOO CLOSE) and only {frac_far_20cm:.2%} of envs with nearest fingertip >= 20 cm"
            f" (TOO FAR). <=1cm fraction: {frac_close_1cm:.2%}; >=10cm fraction: {frac_far_10cm:.2%}."
        )
    print("\n" + "=" * 78)
    print(verdict)
    print("=" * 78)

    # =====================================================================
    # JSON summary
    # =====================================================================
    summary = {
        "task": args_cli.task,
        "num_envs": num_envs,
        "seed": args_cli.seed,
        "body_names_measured": ALL_BODY_NAMES,
        "shape_aware_available_envs": n_ok,
        "shape_aware_failed_envs": n_failed,
        "distance_nearest_fingertip": {
            "center_to_center": {**summarize(center_np), **threshold_fractions(center_np)},
            "shape_aware_surface": {**summarize(shape_np), **threshold_fractions(shape_np)},
            "note": "center_to_center is root-to-root and NOT a surface distance; shape_aware_surface is a"
            " best-effort OBB-surface estimate where 0 means overlap/penetration.",
        },
        "distance_mount": {
            "center_to_center": summarize(mount_center_np),
            "shape_aware_surface": summarize(mount_shape_np),
        },
        "table_clearance": {
            "naive_center_z_minus_table_top": summarize(clearance_naive_np),
            "shape_aware_lowest_corner_minus_table_top": summarize(clearance_shape_np),
            "note": "positive == above table surface (floating); negative == penetrating the table.",
        },
        "contact_at_reset": {
            "source": "ContactSensor.data.force_matrix_w (filtered to Object prim), read after one env.step()",
            "eps_N": eps,
            "count_any_force_gt_eps": n_contact_eps,
            "fraction_any_force_gt_eps": n_contact_eps / num_envs,
            "threshold_N": args_cli.contact_force_threshold,
            "count_gt_threshold": n_contact_thresh,
            "fraction_gt_threshold": n_contact_thresh / num_envs,
        },
        "verdict": verdict,
    }

    out_path = args_cli.out_json
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[INFO] Wrote JSON summary to: {os.path.abspath(out_path)}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
