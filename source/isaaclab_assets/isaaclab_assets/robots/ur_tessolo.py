# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
Configuration for the UR10 arm with Tessolo Delto 5-finger hand (appolo).

UR10 parameters (spawn/usd path, initial arm q, actuator stiffness/damping/effort)
are copied from the existing UR10 config for parity. Hand joints follow the
provided naming scheme: rj_dg_{finger}_{joint} where finger ∈ {1..5}, joint ∈ {1..4}.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR

# -----------------------------------------------------------------------------
# UR10 + Tessolo Delto configuration
# -----------------------------------------------------------------------------

UR10_TESSOLO_DELTO_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/robots/URTessoloAlik/ur10e_delto_optimized_separate_tips.usd",
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
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=1,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
        activate_contact_sensors=True,      # copied from UR10_CFG
        # articulation_props can be added/tuned if needed; left minimal to mirror UR10_CFG
        # joint_drive_props defaults to force control in KUKA+Allegro; leaving default here
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Copied UR10 initial arm posture (UR10_CFG)
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            # UR10 arm:
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.712,
            "elbow_joint": 1.712,
            "wrist_1_joint": 0.0,
            "wrist_2_joint": 0.0,
            "wrist_3_joint": 0.0,
            # Tessolo Delto hand (open pose; adjust as needed):
            "rj_dg_1_1": 0.0,
            "rj_dg_1_2": 0.0,
            r"rj_dg_1_(3|4)": 0.0,
            "rj_dg_2_1": -0.1745,
            "rj_dg_4_1": 0.1745,
            r"rj_dg_(3|5)_1": 0.0,
            r"rj_dg_(2|3|4|5)_(2|3|4)": 0.5237,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_pan_joint",
                                "shoulder_lift_joint",
                                "elbow_joint",
                                "wrist_1_joint",
                                "wrist_2_joint",
                                "wrist_3_joint"],
            velocity_limit=1000.0,
            effort_limit=870.0,
            stiffness=800.0,
            damping=40.0,
        ),
        # "tessolo_hand": ImplicitActuatorCfg(
        #     joint_names_expr=[r"rj_dg_(1|2|3|4|5)_(1|2|3|4)"],
        # ),

        "tessolo_hand": ImplicitActuatorCfg(
            joint_names_expr=["rj_dg_.*"],
            effort_limit={
                "rj_dg_.*":30
            },
            stiffness={
                "rj_dg_1_1": 0.8294,
                "rj_dg_1_2": 0.62859, 
                "rj_dg_1_3": 0.45859, 
                "rj_dg_1_4": 0.16977, 

                "rj_dg_2_1": 2.43808, 
                "rj_dg_2_2": 0.82762, 
                "rj_dg_2_3": 0.41117, 
                "rj_dg_2_4": 0.09574, 

                "rj_dg_3_1": 2.45355, 
                "rj_dg_3_2": 0.82643,
                "rj_dg_3_3": 0.41118, 
                "rj_dg_3_4": 0.09575, 

                "rj_dg_4_1": 2.3093,
                "rj_dg_4_2": 0.82729,
                "rj_dg_4_3": 0.41094,
                "rj_dg_4_4": 0.09573,

                "rj_dg_5_1": 1.73423,
                "rj_dg_5_2": 1.10747,
                "rj_dg_5_3": 0.45515,
                "rj_dg_5_4": 0.16936
            },
            damping={
         
                "rj_dg_1_1": 0.05314332,
                "rj_dg_1_2": 0.02990215, 
                "rj_dg_1_3": 0.01657904, 
                "rj_dg_1_4": 0.00270917, 

                "rj_dg_2_1": 0.26783944, 
                "rj_dg_2_2": 0.0451749, 
                "rj_dg_2_3": 0.0120614, 
                "rj_dg_2_4": 0.00088871, 

                "rj_dg_3_1": 0.2703927,
                "rj_dg_3_2": 0.04507756,
                "rj_dg_3_3": 0.0125066, 
                "rj_dg_3_4": 0.00088885, 

                "rj_dg_4_1": 0.24690116,
                "rj_dg_4_2": 0.04514794,
                "rj_dg_4_3": 0.01249565,
                "rj_dg_4_4": 0.00088857,

                "rj_dg_5_1": 0.16547778,
                "rj_dg_5_2": 0.0699276,
                "rj_dg_5_3": 0.01639284,
                "rj_dg_5_4": 0.0026993,
            },
        ),
    },
    # Keep default soft limit factor; raise if you need headroom for exploration
    soft_joint_pos_limit_factor=1.0,
)