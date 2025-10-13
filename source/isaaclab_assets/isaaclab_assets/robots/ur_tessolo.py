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
            joint_names_expr=[
                "rj_dg_(1|2|3|4|5)_(1|2|3|4)",
            ],
            # USD: float drive:angular:physics:maxForce
            effort_limit_sim={
                "rj_dg_(1|2|3|4|5)_(1|2|3|4)": 30.0,
            },
            velocity_limit_sim={
                "rj_dg_(1|2|3|4|5)_(1|2|3|4)": 10000.0,
            },
            # USD: float drive:angular:physics:stiffness
            # stiffness={
            #     "rj_dg_1_1": 0.8294013,
            #     "rj_dg_1_2": 0.6285933,
            #     "rj_dg_1_3": 0.4585949,
            #     "rj_dg_1_4": 0.1697713,
            #     "rj_dg_2_1": 2.4380815,
            #     "rj_dg_2_2": 0.8276208,
            #     "rj_dg_2_3": 0.4111692,
            #     "rj_dg_2_4": 0.0957437,
            #     "rj_dg_3_1": 2.4535544,
            #     "rj_dg_3_2": 0.8264341,
            #     "rj_dg_3_3": 0.4111796,
            #     "rj_dg_3_4": 0.0957482,
            #     "rj_dg_4_1": 2.3093047,
            #     "rj_dg_4_2": 0.8272905,
            #     "rj_dg_4_3": 0.4109422,
            #     "rj_dg_4_4": 0.0957327,
            #     "rj_dg_5_1": 1.7342279,
            #     "rj_dg_5_2": 1.1074699,
            #     "rj_dg_5_3": 0.4551452,
            #     "rj_dg_5_4": 0.1693594,
            # },
            # # USD: float drive:angular:physics:damping
            # damping={
            #     "rj_dg_1_1": 0.00033176053,
            #     "rj_dg_1_2": 0.00025143730,
            #     "rj_dg_1_3": 0.00018343795,
            #     "rj_dg_1_4": 0.00006790854,
            #     "rj_dg_2_1": 0.00097523263,
            #     "rj_dg_2_2": 0.00033104833,
            #     "rj_dg_2_3": 0.00016446768,
            #     "rj_dg_2_4": 0.00003829747,
            #     "rj_dg_3_1": 0.00098142180,
            #     "rj_dg_3_2": 0.00033057365,
            #     "rj_dg_3_3": 0.00016447184,
            #     "rj_dg_3_4": 0.00003829928,
            #     "rj_dg_4_1": 0.00092372190,
            #     "rj_dg_4_2": 0.00033091620,
            #     "rj_dg_4_3": 0.00016437689,
            #     "rj_dg_4_4": 0.00003829308,
            #     "rj_dg_5_1": 0.00069369114,
            #     "rj_dg_5_2": 0.00044298798,
            #     "rj_dg_5_3": 0.00018205808,
            #     "rj_dg_5_4": 0.00006774375,
            # },
            stiffness= {
                "rj_dg_(1|2|3|4|5)_(1|2|3|4)": 3.0,
            },
            damping={
                "rj_dg_(1|2|3|4|5)_(1|2|3|4)": 0.1,
            },
            friction={
                "rj_dg_(1|2|3|4|5)_(1|2|3|4)": 0.01,
            },
        ),
    },
    # Keep default soft limit factor; raise if you need headroom for exploration
    soft_joint_pos_limit_factor=1.0,
)