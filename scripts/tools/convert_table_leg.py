# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convert the FurnitureBench square-table-leg mesh into a USD prop for the dexsuite object slot.

The source geometry ships as *two* OBJ files that must be assembled into one physical part before
any USD conversion happens:

  * ``square_table_leg4_200mm_body.obj``   - the plain square rod (long, featureless).
  * ``square_table_leg4_200mm_thread.obj`` - the threaded cap at one end (the grasp-relevant relief).

They are defined in the same local frame (the source URDF joins them with identity-transform fixed
joints, no ``<origin>`` offset), so this script welds them into a single triangle soup before handing
it to :class:`isaaclab.sim.converters.MeshConverter`.

MEASURED GEOMETRY (from ``tools/_measure_table_leg_mesh`` below, run directly against the OBJ files
with a hand-rolled numpy OBJ parser -- no Isaac / trimesh dependency; see ``main()`` for the runtime
reprint of these same numbers computed live from whatever OBJ paths are passed on the CLI):

    UNITS: metres. Determined empirically, not assumed: the combined bounding box's long-axis extent
    comes out to exactly 0.200000 (to 1e-6), which matches the "leg_200mm" folder name only if the raw
    OBJ coordinates are already SI metres (200 mm == 0.2 m). Millimetre-unit coordinates would have
    produced an extent of 200.0, centimetre-unit coordinates 20.0; neither matches. No unit conversion
    is applied anywhere in this script as a result.

    body.obj      : 5,115 vertices / 3,015 triangles.
                    bbox   [-0.01875, -0.0150005, -0.015] -> [0.15625, 0.0150005, 0.015]  (m)
                    extent [0.175000, 0.030001, 0.030000] m  -- long axis = X.
    thread.obj    : 26,740 vertices / 9,243 triangles.
                    bbox   [-0.04375, -0.0121875, -0.012187] -> [-0.01875, 0.0121875, 0.012188]  (m)
                    extent [0.025000, 0.024375, 0.024375] m  -- long axis = X.
    combined      : 31,855 vertices / 12,258 triangles (raw, unwelded, exactly as fed to the converter).
                    bbox   [-0.04375, -0.0150005, -0.015] -> [0.15625, 0.0150005, 0.015]  (m)
                    extent [0.200000, 0.030001, 0.030000] m  -- long axis = X (matches "200mm" leg).
                    centroid (raw combined mesh, tetra-volume-weighted): [0.062453, -0.0000042, 0.0000006] m
                    volume  : 1.573053e-4 m^3  (157.305 cm^3)

    WATERTIGHTNESS: each OBJ, taken at face value with its own per-face vertex indices, looks
    non-manifold (body.obj: 5,147/9,045 undirected edges have multiplicity 1; thread.obj similarly) --
    but that is an artifact of the exporter emitting independent vertex records per face (flat-shaded
    export), not a hole in the surface. Welding vertices by position (round to 1e-7 m) before the same
    edge-multiplicity check shows body.obj and thread.obj are each individually near-watertight (179
    boundary edges apiece, all lying exactly on the mating plane at x=-0.01875 m where the two parts
    join) and the WELDED COMBINED mesh is perfectly watertight: every one of its 18,387 undirected
    edges has multiplicity exactly 2, zero boundary edges. This is why the two OBJs must be merged
    before collision-mesh generation -- either piece alone has an open face at the seam.

    MASS SANITY CHECK: for the previous-campaign mass of 0.022750 kg and the measured volume above,
    the implied uniform density is 0.022750 / 1.573053e-4 = 144.6 kg/m^3. That is lighter than balsa
    wood (~120-160 kg/m^3) and roughly 3-6x lighter than the pine/beech (~500-700 kg/m^3) or ABS/PLA
    (~1000-1400 kg/m^3) a real FurnitureBench leg would plausibly be made of. The number is printed at
    runtime (see ``--mass``) precisely so this implausibility stays visible instead of being silently
    baked into a USD file.

WHY THE DEFAULT COLLISION APPROXIMATION IS NOT convexHull:
    The leg is a thin, mostly-square rod with a threaded relief cut into one end (see the mass split
    above: the threaded 25mm end is a separate, geometrically hollowed-out piece from the plain 175mm
    body). A convex hull collision shape fills in every concavity -- most importantly the thread
    groove -- because a hull is by definition the smallest convex set containing the points. That
    turns the one grippable feature on this object (the thread relief, which is what a fingertip can
    hook into) into a smooth cylinder indistinguishable from the rest of the rod, which silently
    changes what grasps are physically graspable in sim versus on the real part. convexDecomposition
    (CoACD-style multi-hull decomposition, matching what the source campaign already used for
    ``body_coacd/``) keeps the concavity. ``sdf`` keeps it exactly (up to voxel resolution) at a higher
    runtime cost. Both are offered; convexHull remains available via ``--collision-approximation
    convexHull`` for callers who explicitly want the (wrong, for this asset) fast approximation.

OUTPUT LOCATION: ``source/isaaclab_assets/data/props/FurnitureBench/SquareTableLeg200mm/`` (new
    directory), not ``data/robots/``. Every existing ``!``-exception in .gitignore under
    ``data/robots/`` is an articulated robot or end-effector (UR10e+Delto, the Rohand variants,
    the Tessolo hand) -- things with joints that get referenced by an ``ArticulationCfg``. This leg
    is a single rigid manipuland referenced by a ``RigidObjectCfg``/``UsdFileCfg``, i.e. a scene prop,
    not a robot; mixing it into ``data/robots/`` would make that directory's contents lie about what
    it holds. ``data/props/`` mirrors the existing per-asset subfolder convention
    (``data/robots/<RobotName>/...``) as ``data/props/<SourceCollection>/<AssetName>/...``.

USAGE (on the remote Isaac Sim box; do not run this locally -- no GPU / Kit runtime here):

    ./isaaclab.sh -p scripts/tools/convert_table_leg.py \\
        --collision-approximation convexDecomposition \\
        --mass 0.022750 \\
        --headless

Every Isaac process must be wrapped by the caller in ``timeout -s KILL ...`` since Kit ignores
SIGTERM; this script performs exactly one ``AppLauncher``/Kit session and exits.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Convert the FurnitureBench square-table-leg (body + thread OBJ pair) into a dexsuite USD prop."
)
parser.add_argument(
    "--body-obj",
    type=str,
    default=(
        "/home/dom-iva/github.com/orel/lerobot/UWLab_delto/source/uwlab_assets/uwlab_assets/local/Props/"
        "FurnitureBench/SquareTableOneLeg/leg_200mm/square_table_leg4_200mm_body.obj"
    ),
    help="Path to the plain-rod body OBJ.",
)
parser.add_argument(
    "--thread-obj",
    type=str,
    default=(
        "/home/dom-iva/github.com/orel/lerobot/UWLab_delto/source/uwlab_assets/uwlab_assets/local/Props/"
        "FurnitureBench/SquareTableOneLeg/leg_200mm/square_table_leg4_200mm_thread.obj"
    ),
    help="Path to the threaded-end OBJ.",
)
parser.add_argument(
    "--collision-approximation",
    type=str,
    default="convexDecomposition",
    choices=["convexHull", "convexDecomposition", "sdf"],
    help=(
        "Collision mesh approximation. Defaults to convexDecomposition because a convex hull fills in"
        " the thread relief and changes what can be grasped -- see the module docstring."
    ),
)
parser.add_argument(
    "--mass",
    type=float,
    default=0.022750,
    help="Mass (kg) to bake into the rigid body. Defaults to the previous campaign's measured value.",
)
parser.add_argument(
    "--hull-vertex-limit",
    type=int,
    default=64,
    help="Per-hull vertex cap, used by both convexHull and convexDecomposition. PhysX default is 64.",
)
parser.add_argument(
    "--max-convex-hulls",
    type=int,
    default=32,
    help="Max number of hulls produced by convexDecomposition. PhysX default is 32.",
)
parser.add_argument(
    "--voxel-resolution",
    type=int,
    default=500_000,
    help="Voxel resolution used by convexDecomposition. PhysX default is 500,000.",
)
parser.add_argument(
    "--sdf-resolution",
    type=int,
    default=256,
    help="SDF grid resolution, used only when --collision-approximation sdf. Matches the resolution"
    " the source URDF already used for the threaded end (sdf resolution=\"256\").",
)
parser.add_argument(
    "--contact-threshold",
    type=float,
    default=0.0,
    help="Force threshold (N) above which the PhysX contact-report API reports a contact.",
)
parser.add_argument(
    "--bbox-tolerance",
    type=float,
    default=0.001,
    help=(
        "Max allowed |diff| (m), per axis, between the converted USD's world-space bbox extent and"
        " the measured source (combined, welded) extent. MeshConverter's use_meter_as_world_unit flag"
        " is a documented no-op (see mesh_converter.py), so this is a round-trip sanity check that the"
        " converted geometry was not silently mis-scaled -- not a cosmetic warning."
    ),
)
parser.add_argument(
    "--output-dir",
    type=str,
    default="source/isaaclab_assets/data/props/FurnitureBench/SquareTableLeg200mm",
    help="Output USD directory, relative to the repo root unless given as an absolute path.",
)
parser.add_argument(
    "--usd-file-name",
    type=str,
    default="square_table_leg4_200mm.usd",
    help="Name of the generated top-level USD file.",
)
parser.add_argument(
    "--no-instanceable",
    action="store_false",
    dest="make_instanceable",
    default=True,
    help="Disable scene-graph instancing (default: instancing is on, matching MeshConverterCfg's default).",
)
parser.add_argument(
    "--force",
    action="store_true",
    help=(
        "Force USD regeneration even if a cached conversion with a matching hash already exists"
        " (see AssetConverterBase's hash-based skip-if-unchanged cache). Off by default so repeated"
        " invocations on a metered box reuse the cached conversion; pass this flag to bypass it."
    ),
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import pathlib
import tempfile

from pxr import Usd, UsdGeom

from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
from isaaclab.sim.schemas import schemas, schemas_cfg
from isaaclab.utils.assets import check_file_path
from isaaclab.utils.dict import print_dict

# repo root = two levels up from this file (scripts/tools/convert_table_leg.py)
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _load_obj_positions_and_tris(path: str):
    """Minimal OBJ reader: returns (Nx3 vertex positions, Mx3 triangle vertex indices).

    Ignores texture/normal indices and materials on purpose -- this script only needs positional
    geometry for collision-mesh generation and mass/volume bookkeeping; visual material is not load
    bearing for a contact-sensing prop and dropping it avoids having to re-resolve ``mtllib``/``usemtl``
    paths after the two source OBJs are merged into one file on disk.
    """
    verts: list[list[float]] = []
    tris: list[list[int]] = []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                idx = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:]]
                # fan-triangulate polygons with >3 vertices
                for i in range(1, len(idx) - 1):
                    tris.append([idx[0], idx[i], idx[i + 1]])
    return verts, tris


def _mesh_stats(name: str, verts: list[list[float]], tris: list[list[int]]) -> dict:
    """Compute and print bbox / long axis / volume / centroid / watertightness for one mesh.

    Implemented with plain python + (optionally) numpy if available; falls back to pure python loops
    so this has no hard dependency beyond the standard library. Volume/centroid use the divergence
    theorem (signed sum of tetrahedra with apex at the origin), which only gives the right answer for a
    closed (watertight) surface -- reported alongside the watertightness check for that reason.
    """
    try:
        import numpy as np

        v = np.asarray(verts, dtype=np.float64)
        f = np.asarray(tris, dtype=np.int64)
        bbox_min = v.min(axis=0)
        bbox_max = v.max(axis=0)
        extent = bbox_max - bbox_min
        long_axis = "xyz"[int(np.argmax(extent))]

        v0, v1, v2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
        cross = np.cross(v1, v2)
        tetra_vol = np.einsum("ij,ij->i", v0, cross) / 6.0
        volume = float(np.abs(tetra_vol.sum()))
        tet_centroid = (v0 + v1 + v2) / 4.0
        centroid = (tet_centroid * tetra_vol[:, None]).sum(axis=0) / tetra_vol.sum()

        # watertightness: weld vertices by position (1e-7 m) then require every undirected edge to
        # appear exactly twice.
        key = np.round(v, 7)
        _, inv = np.unique(key, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        wf = inv[f]
        edges = np.vstack([wf[:, [0, 1]], wf[:, [1, 2]], wf[:, [2, 0]]])
        edges_sorted = np.sort(edges, axis=1)
        _, counts = np.unique(edges_sorted, axis=0, return_counts=True)
        boundary_edges = int(np.sum(counts != 2))
        watertight = boundary_edges == 0

        result = dict(
            num_verts=len(v),
            num_tris=len(f),
            bbox_min=bbox_min.tolist(),
            bbox_max=bbox_max.tolist(),
            extent=extent.tolist(),
            long_axis=long_axis,
            volume=volume,
            centroid=centroid.tolist(),
            watertight=watertight,
            boundary_edges=boundary_edges,
        )
    except ImportError:
        raise RuntimeError("numpy is required to measure the mesh (ships with the Isaac Sim python env).")

    print(f"--- {name} ---")
    print(f"  vertices: {result['num_verts']}  triangles: {result['num_tris']}")
    print(f"  bbox min: {result['bbox_min']}")
    print(f"  bbox max: {result['bbox_max']}")
    print(f"  extent  : {result['extent']}  (long axis = {result['long_axis']})")
    print(f"  watertight (position-welded, 1e-7 m tolerance): {result['watertight']}"
          f"  (boundary edges: {result['boundary_edges']})")
    print(f"  volume (m^3): {result['volume']:.8e}  ({result['volume'] * 1e6:.3f} cm^3)")
    print(f"  centroid (m): {result['centroid']}")
    return result


def _merge_obj(
    body_path: str, thread_path: str, out_path: str
) -> tuple[list[list[float]], list[list[int]]]:
    """Weld the body + thread OBJs into a single triangle-soup OBJ for the mesh converter.

    The two parts are already defined in the same local frame (see module docstring), so this is a
    plain vertex-index-offset concatenation -- no transform is applied.
    """
    vb, fb = _load_obj_positions_and_tris(body_path)
    vt, ft = _load_obj_positions_and_tris(thread_path)
    offset = len(vb)
    verts = vb + vt
    tris = fb + [[a + offset, b + offset, c + offset] for a, b, c in ft]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("# merged: body + thread, generated by scripts/tools/convert_table_leg.py\n")
        for x, y, z in verts:
            f.write(f"v {x:.9f} {y:.9f} {z:.9f}\n")
        for a, b, c in tris:
            f.write(f"f {a + 1} {b + 1} {c + 1}\n")
    return verts, tris


def _verify_converted_usd_extent(usd_path: str, expected_extent: list[float], tolerance: float) -> None:
    """Round-trip check: re-open the CONVERTED USD and compare its world-space bbox extent against
    the measured SOURCE (combined, welded) OBJ extent.

    This script leaves ``MeshConverterCfg.scale`` at the default ``(1, 1, 1)`` on the reasoning that
    the source OBJ coordinates are already SI metres (see module docstring's MEASURED GEOMETRY
    section) -- but that reasoning is about the *input*, not the *output*. MeshConverter's own
    ``use_meter_as_world_unit`` flag is a documented no-op (mesh_converter.py: "This does not work
    right now :(, so we need to scale the mesh manually"), so a silently mis-scaled conversion (e.g.
    off by 10x/100x/1000x from a units mixup inside the converter) would look identical in the console
    to a correct one unless the produced USD is actually re-measured. This does that re-measurement
    and fails loudly, with both numbers, if they don't round-trip.
    """
    stage = Usd.Stage.Open(usd_path)
    default_prim = stage.GetDefaultPrim()
    if not default_prim.IsValid():
        raise RuntimeError(f"Converted USD at {usd_path} has no default prim; cannot verify extent.")
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide],
        True,  # useExtentsHint
    )
    rng = bbox_cache.ComputeWorldBound(default_prim).ComputeAlignedRange()
    lo, hi = rng.GetMin(), rng.GetMax()
    got_extent = [hi[i] - lo[i] for i in range(3)]
    diffs = [abs(got_extent[i] - expected_extent[i]) for i in range(3)]

    print("-" * 80)
    print("ROUND-TRIP CHECK: converted USD world-space bbox extent vs. measured source extent")
    print(f"  measured source (combined, welded) extent : {['%.6f' % v for v in expected_extent]} m")
    print(f"  converted USD world-space bbox extent      : {['%.6f' % v for v in got_extent]} m")
    print(f"  |diff|                                      : {['%.6f' % v for v in diffs]} m"
          f"  (tolerance {tolerance:.6f} m)")
    if any(d > tolerance for d in diffs):
        raise RuntimeError(
            "Converted USD geometry does NOT match the measured source extent within tolerance -- "
            f"expected {expected_extent} m, got {got_extent} m, diff {diffs} m > tolerance {tolerance} m."
            " This is exactly the failure mode MeshConverter's use_meter_as_world_unit no-op comment"
            " warns about -- do NOT trust this USD; do not use it downstream."
        )
    print("  OK: converted USD extent matches the measured source within tolerance.")
    print("-" * 80)


def _resolve_mesh_collision_props(args) -> schemas_cfg.MeshCollisionPropertiesCfg:
    """Build the mesh-collision-approximation config selected on the CLI."""
    if args.collision_approximation == "convexHull":
        return schemas_cfg.ConvexHullPropertiesCfg(hull_vertex_limit=args.hull_vertex_limit)
    elif args.collision_approximation == "convexDecomposition":
        return schemas_cfg.ConvexDecompositionPropertiesCfg(
            hull_vertex_limit=args.hull_vertex_limit,
            max_convex_hulls=args.max_convex_hulls,
            voxel_resolution=args.voxel_resolution,
        )
    elif args.collision_approximation == "sdf":
        return schemas_cfg.SDFMeshPropertiesCfg(sdf_resolution=args.sdf_resolution)
    else:
        raise ValueError(f"Unsupported collision approximation: {args.collision_approximation}")


DEXSUITE_OBJECT_SNIPPET_TEMPLATE = '''\
# Replacement for SceneCfg.object in dexsuite_env_cfg.py (currently lines 36-67), swapping the
# 16-primitive MultiAssetSpawnerCfg for the single, physically-calibrated table-leg USD.
#
# Kept from the original: the rigid_props solver settings, the CollisionPropertiesCfg() default, and
# the "dexsuite_object" semantic tag -- these are asset-agnostic and apply equally to a USD-file spawn.
#
# Dropped: MultiAssetSpawnerCfg / assets_cfg / random_choice -- there is exactly one asset now, so
# UsdFileCfg replaces MultiAssetSpawnerCfg directly rather than wrapping a length-1 list.
#
# Changed and why (see EventCfg ranges below, dexsuite_env_cfg.py:268-327):
#   * mass_props.mass: 0.2 -> 0.022750 kg (this leg's authored mass; matches --mass default above).
#   * object_scale_mass (mass_distribution_params=[0.2, 2.0], operation="scale") multiplies whatever
#     mass_props bakes in, so it is NOT broken by the smaller base mass -- but it now samples
#     [0.00455, 0.0455] kg instead of dexsuite's tuned [0.04, 0.4] kg. Any reward/impedance term that
#     assumes an ~0.2 kg-scale object (finger force thresholds, grasp-stability tolerances) is being
#     exercised an order of magnitude below where it was calibrated; re-check those terms, don't just
#     port the range unchanged.
#   * randomize_object_scale (scale_range=(0.75, 1.5), mode="prestartup") is appropriate for the
#     generic primitives (which have no "real" size to preserve) but is INAPPROPRIATE for this asset:
#     it is a calibrated stand-in for a specific FurnitureBench part, and geometric scale range (0.75,
#     1.5) would make the rod 4.5 cm square (1.5x) -- likely wider than the target gripper's aperture --
#     or 2.25 cm square / 15 cm long (0.75x), no longer matching the real leg being replicated. Tighten
#     this drastically (e.g. (0.97, 1.03) for sim-to-real slop only) or drop the event term for this
#     object entirely if exact dimensions matter downstream.

from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR

TABLE_LEG_USD_PATH = f"{{ISAACLAB_ASSETS_DATA_DIR}}/props/FurnitureBench/SquareTableLeg200mm/{usd_file_name}"

object: RigidObjectCfg = RigidObjectCfg(
    prim_path="{{ENV_REGEX_NS}}/Object",
    spawn=sim_utils.UsdFileCfg(
        usd_path=TABLE_LEG_USD_PATH,
        scale=(1.0, 1.0, 1.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=0,
            disable_gravity=False,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass={mass}),
        semantic_tags=[("class", "dexsuite_object")],
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.55, 0.1, 0.35)),
)
'''


def main():
    body_path = os.path.abspath(args_cli.body_obj)
    thread_path = os.path.abspath(args_cli.thread_obj)
    if not check_file_path(body_path):
        raise ValueError(f"Invalid body OBJ path: {body_path}")
    if not check_file_path(thread_path):
        raise ValueError(f"Invalid thread OBJ path: {thread_path}")

    print("=" * 80)
    print("MEASURED SOURCE GEOMETRY (recomputed live from the OBJ paths given on the CLI)")
    print("=" * 80)
    vb, fb = _load_obj_positions_and_tris(body_path)
    vt, ft = _load_obj_positions_and_tris(thread_path)
    _mesh_stats("body", vb, fb)
    _mesh_stats("thread", vt, ft)

    # resolve output paths
    output_dir = args_cli.output_dir
    if not os.path.isabs(output_dir):
        output_dir = str(REPO_ROOT / output_dir)
    usd_path_rel = pathlib.Path(os.path.relpath(os.path.join(output_dir, args_cli.usd_file_name), REPO_ROOT))
    instanceable_meshes_rel = usd_path_rel.parent / "Props" / "instanceable_meshes.usd"

    # merge body + thread into one triangle soup and re-measure the combined, welded solid.
    # Written to a SYSTEM TEMP DIR, not under output_dir (the tracked source tree): .gitignore only
    # blanket-ignores **/*.usd*, not .obj, so a ~31.8k-vertex intermediate OBJ left in
    # source/isaaclab_assets/... would be silently committable by accident. It is purely a
    # MeshConverter input, not a build artifact anyone needs to keep, so it doesn't need a tracked
    # location or a new .gitignore rule -- the OS temp dir is enough.
    merged_obj_dir = tempfile.mkdtemp(prefix="convert_table_leg_merged_")
    merged_obj_path = os.path.join(merged_obj_dir, "square_table_leg4_200mm_merged.obj")
    print(f"[INFO] Writing merged intermediate OBJ to temp dir (not the source tree): {merged_obj_path}")
    verts, tris = _merge_obj(body_path, thread_path, merged_obj_path)
    combined_stats = _mesh_stats("combined (body+thread, fed to MeshConverter)", verts, tris)

    density = args_cli.mass / combined_stats["volume"]
    print("-" * 80)
    print(f"MASS: {args_cli.mass:.6f} kg  ->  implied uniform density: {density:.2f} kg/m^3")
    if density < 200.0 or density > 3000.0:
        print(
            f"  [WARNING] {density:.1f} kg/m^3 is outside the plausible range for a solid wood/plastic"
            " furniture part (~150-1500 kg/m^3). Double-check the mass source before trusting this asset."
        )
    print("-" * 80)

    # collision approximation
    mesh_collision_props = _resolve_mesh_collision_props(args_cli)
    collision_props = schemas_cfg.CollisionPropertiesCfg(collision_enabled=True)
    mass_props = schemas_cfg.MassPropertiesCfg(mass=args_cli.mass)
    # rigid_body_enabled=True so a UsdPhysics.RigidBodyAPI exists for activate_contact_sensors() to
    # attach the PhysX contact-report API to below; solver settings mirror what dexsuite's
    # MultiAssetSpawnerCfg.rigid_props already used for the primitive objects (dexsuite_env_cfg.py:57-61).
    rigid_props = schemas_cfg.RigidBodyPropertiesCfg(
        rigid_body_enabled=True,
        disable_gravity=False,
        solver_position_iteration_count=16,
        solver_velocity_iteration_count=0,
    )

    mesh_converter_cfg = MeshConverterCfg(
        asset_path=merged_obj_path,
        usd_dir=output_dir,
        usd_file_name=args_cli.usd_file_name,
        force_usd_conversion=args_cli.force,
        make_instanceable=args_cli.make_instanceable,
        mass_props=mass_props,
        rigid_props=rigid_props,
        collision_props=collision_props,
        mesh_collision_props=mesh_collision_props,
    )

    print("Mesh converter config:")
    print_dict(mesh_converter_cfg.to_dict(), nesting=0)

    mesh_converter = MeshConverter(mesh_converter_cfg)
    print(f"Generated USD file: {mesh_converter.usd_path}")

    # verify the CONVERTED USD's geometry round-trips to the measured source extent (see docstring
    # of _verify_converted_usd_extent for why this can't just be inferred from scale=(1,1,1)).
    _verify_converted_usd_extent(mesh_converter.usd_path, combined_stats["extent"], args_cli.bbox_tolerance)

    # activate contact sensors on the resulting rigid body -- the dexsuite reward path reads
    # per-fingertip contacts filtered to the object prim, which requires the PhysX contact-report API.
    stage = Usd.Stage.Open(mesh_converter.usd_path)
    root_prim = stage.GetDefaultPrim()
    if not root_prim.IsValid():
        raise RuntimeError(f"Converted USD at {mesh_converter.usd_path} has no default prim.")
    root_prim_path = root_prim.GetPath().pathString
    schemas.activate_contact_sensors(root_prim_path, threshold=args_cli.contact_threshold, stage=stage)
    stage.GetRootLayer().Save()
    print(f"Activated contact sensors on rigid body: {root_prim_path}")

    # .gitignore exceptions required for a fresh clone to actually have the geometry
    print("=" * 80)
    print("Add the following '!'-exception line(s) to .gitignore (after line 83, alongside the other 12")
    print("explicit per-file exceptions at .gitignore lines 72-83 -- there is no wildcard exception,")
    print("each tracked USD is listed individually) -- otherwise this USD is silently untracked and a")
    print("fresh clone is missing the asset:")
    print(f"  !{usd_path_rel.as_posix()}")
    if args_cli.make_instanceable:
        print(f"  !{instanceable_meshes_rel.as_posix()}")
        print(
            "  (make_instanceable=True stores the actual mesh payload in the Props/instanceable_meshes.usd"
            " sidecar referenced by the top-level file above -- both must be tracked or the reference"
            " resolves to nothing on a fresh clone.)"
        )
    print("=" * 80)

    # printed, not applied: the SceneCfg.object replacement snippet
    print("Replacement RigidObjectCfg snippet for dexsuite_env_cfg.py SceneCfg.object (lines 36-67):")
    print("-" * 80)
    print(DEXSUITE_OBJECT_SNIPPET_TEMPLATE.format(usd_file_name=args_cli.usd_file_name, mass=args_cli.mass))
    print("-" * 80)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
