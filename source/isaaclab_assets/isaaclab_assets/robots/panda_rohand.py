# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Franka Emika Panda arm equipped with the RoHand gripper.

The following configurations are available:

* :obj:`PANDA_ROHAND_CFG`: Franka Panda arm with six independently actuated RoHand fingers.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR


PANDA_ROHAND_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/robots/panda_rohand.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
        # joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos={
            # Franka arm joints (mirrors FRANKA_PANDA_CFG default)
            "panda_joint1": 0.0,
            "panda_joint2": -0.569,
            "panda_joint3": 0.0,
            "panda_joint4": -2.210,
            "panda_joint5": 0.0,
            "panda_joint6": 3.037,
            "panda_joint7": 0.741,
            # RoHand actuated joints (fully open span)
            "th_root_link": 0.0,
            "th_slider_link": 0.0,
            "if_slider_link": 0.0,
            "mf_slider_link": 0.0,
            "rf_slider_link": 0.0,
            "lf_slider_link": 0.0,
        },
    ),
    actuators={
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit_sim=87.0,
            stiffness=400,
            damping=80,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit_sim=12.0,
            stiffness=400,
            damping=80,
        ),
        "rohand": ImplicitActuatorCfg(
            joint_names_expr=[
                "th_root_link",
                "(th|if|mf|rf|lf)_slider_link",
            ],
            effort_limit_sim={
                "th_root_link": 40.0,
                "(th|if|mf|rf|lf)_slider_link": 40.0,
            },
            stiffness={
                "th_root_link": 0.06814,
                "(th|if|mf|rf|lf)_slider_link": 800.0,
            },
            damping={
                "th_root_link": 0.00003,
                "(th|if|mf|rf|lf)_slider_link": 60.0,
            },
            velocity_limit_sim={
                "th_root_link": 100000.0,
                "(th|if|mf|rf|lf)_slider_link": 100000.0,
            },
            
        #     friction={
        #         "th_root_link": 0.02,
        #         "(th|if|mf|rf|lf)_slider_link": 0.01,
        #     },
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Franka Panda arm with RoHand gripper."""

