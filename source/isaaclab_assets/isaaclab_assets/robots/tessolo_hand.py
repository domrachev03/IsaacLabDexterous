# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the standalone Tessolo Delto 5-finger hand.

The following configurations are available:

* :obj:`TESSOLO_HAND_CFG`: Tessolo Delto hand with implicit actuator model.

The hand has 20 revolute joints (5 fingers x 4 joints each) and 5 fingertips.
Joint naming: rj_dg_{finger}_{joint} where finger in {1..5}, joint in {1..4}.
Fingertip bodies: rl_dg_{1..5}_tip.
Fingers 1 and 5 are thumbs (lateral), fingers 2-4 are the central trio.

USD root prim is /ur10 (legacy naming); the hand is anchored to the world
via a FixedJoint at /ur10/FixedJoint.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR

# -----------------------------------------------------------------------------
# Tessolo Delto standalone hand configuration
# -----------------------------------------------------------------------------

TESSOLO_HAND_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/robots/URTessoloAlik/tessolo_hand_limited_jnts_self_collision.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=True,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1000.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
        activate_contact_sensors=True,
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            # Finger 1 (thumb-like, lateral)
            "rj_dg_1_1": 0.0,
            "rj_dg_1_2": -1.5707,
            r"rj_dg_1_(3|4)": 0.0,
            # Finger 2
            "rj_dg_2_1": -0.348,
            "rj_dg_2_2": 0.0,
            r"rj_dg_2_(3|4)": 0.5236,
            # Finger 3
            "rj_dg_3_1": 0.0,
            "rj_dg_3_2": 0.0,
            r"rj_dg_3_(3|4)": 0.5236,
            # Finger 4
            "rj_dg_4_1": 0.261,
            "rj_dg_4_2": 0.0,
            r"rj_dg_4_(3|4)": 0.5236,
            # Finger 5 (thumb-like, lateral)
            "rj_dg_5_2": 1.0472,
            r"rj_dg_5_(1|3|4)": 0.5236,
        },
    ),
    actuators={
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=[r"rj_dg_(1|2|3|4|5)_(1|2|3|4)"],
            effort_limit_sim={
                r"rj_dg_(1|2|3|4|5)_(1|2|3|4)": 30.0,
            },
            velocity_limit_sim={
                r"rj_dg_(1|2|3|4|5)_(1|2|3|4)": 10000.0,
            },
            stiffness={
                r"rj_dg_(1|2|3|4|5)_(1|2|3|4)": 3.0,
            },
            damping={
                r"rj_dg_(1|2|3|4|5)_(1|2|3|4)": 0.1,
            },
            friction={
                r"rj_dg_(1|2|3|4|5)_(1|2|3|4)": 0.01,
            },
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of standalone Tessolo Delto 5-finger hand."""
