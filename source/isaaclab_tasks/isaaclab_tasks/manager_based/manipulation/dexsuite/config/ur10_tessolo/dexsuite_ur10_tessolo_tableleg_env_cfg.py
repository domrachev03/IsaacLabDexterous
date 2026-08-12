# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""UR10-Tessolo Lift task variant that spawns a single real FurnitureBench table-leg part
instead of the base task's 16-primitive MultiAssetSpawnerCfg.

Everything not called out below is inherited byte-identical from the primitives task
(:mod:`dexsuite_ur10_tessolo_env_cfg`) so this variant is a clean apples-to-apples comparison
against the running ``Isaac-Dexsuite-UR10-Tessolo-Lift-v0`` baseline. See the module-level
docstrings on :class:`TableLegSceneCfg` and :class:`TableLegEventCfg` for exactly what differs
and why.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR

from ... import dexsuite_env_cfg as dexsuite
from ... import mdp
from . import dexsuite_ur10_tessolo_env_cfg as ur10_tessolo

# Path to the pre-built, pre-committed table-leg USD (see source/isaaclab_assets/data/props/...).
# Measured: extent 0.200 x 0.030 x 0.030 m (long axis local X), volume 157.305 cm^3, watertight,
# convexDecomposition collision, contact sensors enabled. Round-trip bbox check confirmed the USD
# matches the source geometry.
TABLE_LEG_USD_PATH = (
    f"{ISAACLAB_ASSETS_DATA_DIR}/props/FurnitureBench/SquareTableLeg200mm/square_table_leg4_200mm.usd"
)

# Authored mass of the table-leg USD, in kg. Written explicitly (rather than left implicit in the
# USD) so it is a real dataclass field and therefore reachable from the CLI as
# `env.scene.object.spawn.mass_props.mass=<value>` -- IsaacLab's update_class_from_dict rejects any
# hydra key that is not already present in the dataclass tree (see UR10TessoloMixinCfg's
# `rewards.*.params["threshold"]` comment in dexsuite_ur10_tessolo_env_cfg.py for the same trap).
#
# Corrected from the original 0.022750 kg. The mesh volume is 157.3 cm^3, so 22.75 g implied a
# density of ~144.6 kg/m^3 -- styrofoam, not the printed PLA or wood a real FurnitureBench leg is
# made from (500-1250 kg/m^3). 0.12 kg implies ~763 kg/m^3, in the hardwood range.
#
# The reward gate is the operative reason for the correction, not just the density mismatch.
# `position_tracking` and `success` are both gated on 1.0 N of fingertip contact force (see
# UR10TessoloMixinCfg's `rewards.*.params["threshold"]` in dexsuite_ur10_tessolo_env_cfg.py) --
# a threshold inherited from the stock primitives task, whose objects weigh 40-400 g. At the old
# 22.75 g the object weighed only 0.223 N, so the gate demanded ~4.5x the object's own weight
# before paying any tracking reward; pressing that hard on a free light object ejects it rather
# than holding it. With `object_scale_mass` randomizing [0.5, 1.5] below, 0.12 kg yields 60-180 g,
# centered inside the 40-400 g envelope the gate was calibrated against.
#
# The fix belongs on the asset, not the gate: lowering the 1.0 N threshold instead would change
# the task definition and would still leave the object's mass wrong for sim-to-real transfer.
TABLE_LEG_MASS_KG = 0.12


@configclass
class TableLegSceneCfg(dexsuite.SceneCfg):
    """Dexsuite scene for the single-object table-leg variant.

    Only the ``object`` entity differs from :class:`dexsuite.SceneCfg`: a single
    :class:`~isaaclab.sim.UsdFileCfg`-spawned ``RigidObjectCfg`` replaces the
    16-primitive ``MultiAssetSpawnerCfg``. The robot, table, plane, and lights are all
    inherited untouched. The rigid-body/collision properties, the gravity setting, the
    "dexsuite_object" semantic tag, and the ``init_state`` position are preserved exactly
    as they were on the primitives (only the spawner type and its ``usd_path``/``mass_props``
    are new).
    """

    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=TABLE_LEG_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=0,
                disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=TABLE_LEG_MASS_KG),
            semantic_tags=[("class", "dexsuite_object")],
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.55, 0.1, 0.35)),
    )


@configclass
class TableLegEventCfg(ur10_tessolo.UR10TessoloEventCfg):
    """UR10-Tessolo event terms with the two primitive-tuned randomization ranges corrected for a
    real, dimensionally-fixed 200 mm part. Everything else (robot/object physics-material
    randomization, actuator gains, joint friction, resets, the gravity curriculum, and the
    UR10-Tessolo elbow/finger-root resets added by ``UR10TessoloEventCfg``) is inherited unchanged.
    """

    # Disabled outright rather than pinned to a degenerate (scale_range=(1.0, 1.0)) no-op.
    # Reason 1 (dimensional): this is a manufactured, dimensionally-calibrated 200 mm part, not a
    # procedurally-scalable primitive -- the base (0.75, 1.5) geometric range would train on legs
    # between 15 cm and 30 cm, which is nonsensical for a fixed real object.
    # Reason 2 (replicate_physics): randomize_rigid_body_scale runs in "prestartup" mode and writes
    # a per-env "xformOp:scale" USD attribute directly; its own docstring says this requires
    # scene.replicate_physics=False so the physics parser picks up the per-env override correctly.
    # A pinned-but-still-running term would keep that requirement alive even though the sampled
    # value never changes, which conflicts with replicate_physics=True below (see TableLeg mixin
    # cfg). Disabling the term removes the conflict entirely, at zero behavioral cost since every
    # env would have gotten scale=1.0 anyway.
    randomize_object_scale = None

    # Real leg mass is 120 g (TABLE_LEG_MASS_KG = 0.12 kg). The inherited [0.2, 2.0] SCALE range
    # spans 24-240 g. dexsuite's [0.2, 2.0] range is tuned for the 200 g primitives (40-400 g
    # absolute); the low end (24 g) already sits a comfortable ~24x above the ~1 g "prop silently
    # left at ~1 g, flung by contact, produced false-positive grasps" failure mode already hit on
    # this project. Narrowed to [0.5, 1.5] -> 60-180 g: keeps well over an order of magnitude of
    # headroom above the known ~1 g failure floor while still giving +/-50% mass domain
    # randomization around the authored value.
    object_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": [0.5, 1.5],
            "operation": "scale",
        },
    )


@configclass
class UR10TessoloTableLegMixinCfg(ur10_tessolo.UR10TessoloMixinCfg):
    """Same UR10-Tessolo robot/actions/rewards/reset-event wiring as the primitives task's
    ``UR10TessoloMixinCfg``, with the scene and randomization events swapped for the table-leg
    variants above.

    ``replicate_physics=True`` (vs. the base task's ``False``): the base ``SceneCfg`` sets it False
    because ``MultiAssetSpawnerCfg`` puts a different primitive in every env, which the PhysX
    cloner's ``has_multi_assets`` check (see isaaclab/scene/interactive_scene.py -- the carb flag
    "/isaaclab/spawn/multi_assets" is only ever set by the multi-asset spawn wrapper in
    isaaclab/sim/spawners/wrappers/wrappers.py) explicitly warns against replicating. This variant
    spawns exactly one asset (the leg) via UsdFileCfg, which never touches that flag, and
    randomize_object_scale (the other prestartup per-env USD writer) is disabled above, so there is
    no per-env heterogeneity left for the cloner to mishandle. Enabling replication here is
    expected to avoid the ~14 min CPU-only cloning cost measured at 4096 envs on every run.
    """

    scene: TableLegSceneCfg = TableLegSceneCfg(num_envs=4096, env_spacing=3, replicate_physics=True)
    events: TableLegEventCfg = TableLegEventCfg()


@configclass
class DexsuiteUR10TessoloTableLegLiftEnvCfg(UR10TessoloTableLegMixinCfg, dexsuite.DexsuiteLiftEnvCfg):
    pass


@configclass
class DexsuiteUR10TessoloTableLegLiftEnvCfg_PLAY(UR10TessoloTableLegMixinCfg, dexsuite.DexsuiteLiftEnvCfg_PLAY):
    pass
