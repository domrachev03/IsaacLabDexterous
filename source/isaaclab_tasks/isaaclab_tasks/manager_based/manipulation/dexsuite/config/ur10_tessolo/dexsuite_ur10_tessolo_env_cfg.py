# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets.robots import UR10_TESSOLO_DELTO_CFG

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from ... import dexsuite_env_cfg as dexsuite
from ... import mdp


@configclass
class UR10TessoloRelJointPosActionCfg:
    # Scale is higher6.
    action = mdp.RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale={
            "shoulder_pan_joint": 0.1,  # rad
            "shoulder_lift_joint": 0.1,  # rad
            "elbow_joint": 0.1,  # rad
            "wrist_1_joint": 0.1,  # rad
            "wrist_2_joint": 0.1,  # rad
            "wrist_3_joint": 0.1,  # rad
            r"rj_dg_(1|2|3|4|5)_(1|2|3|4)": 0.1,  # deg
        },
    )


@configclass
class UR10TessoloReorientRewardCfg(dexsuite.RewardsCfg):
    # bool awarding term if 2 finger tips are in contact with object, one of the contacting fingers has to be thumb.
    good_finger_contact = RewTerm(
        func=mdp.any_contact,
        weight=2.0,
        params={
            "threshold": 1.0,
            "contact_names": ("rl_dg_1_4", "rl_dg_2_4", "rl_dg_3_4", "rl_dg_4_4", "rl_dg_5_4"),
        },
    )


@configclass
class UR10TessoloMixinCfg:
    rewards: UR10TessoloReorientRewardCfg = UR10TessoloReorientRewardCfg()
    actions: UR10TessoloRelJointPosActionCfg = UR10TessoloRelJointPosActionCfg()

    def __post_init__(self: dexsuite.DexsuiteReorientEnvCfg):
        super().__post_init__()
        self.commands.object_pose.body_name = "rl_dg_mount"
        self.scene.robot = UR10_TESSOLO_DELTO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # Replace initial position of UR robot by rotating base around z by 180deg
        self.scene.robot.init_state.rot = (0.0, 0.0, 0.0, 1.0)
        finger_tip_body_list = ["rl_dg_1_4", "rl_dg_2_4", "rl_dg_3_4", "rl_dg_4_4", "rl_dg_5_4"]
        for link_name in finger_tip_body_list:
            setattr(
                self.scene,
                f"{link_name}_object_s",
                ContactSensorCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/dg5f_my/" + link_name,
                    filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
                ),
            )
        self.observations.proprio.contact = ObsTerm(
            func=mdp.fingers_contact_force_b,
            params={"contact_sensor_names": [f"{link}_object_s" for link in finger_tip_body_list]},
            clip=(-20.0, 20.0),  # contact force in finger tips is under 20N normally
        )
        self.observations.proprio.hand_tips_state_b.params["body_asset_cfg"].body_names = [
            "rl_dg_mount",
            r"rl_dg_(1|2|3|4|5)_4",
        ]
        self.rewards.fingers_to_object.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=["rl_dg_mount", r"rl_dg_(1|2|3|4|5)_4"]
        )
        self.events.reset_robot_wrist_joint.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=["wrist_3_joint"])

        self.rewards.position_tracking.params["thumb_contact_name"] = "rl_dg_1_4"
        self.rewards.position_tracking.params["tip_contact_names"] = (
            "rl_dg_2_4",
            "rl_dg_3_4",
            "rl_dg_4_4",
            "rl_dg_5_4",
        )

        self.rewards.position_tracking.params["thumb_contact_name"] = "rl_dg_1_4"
        self.rewards.position_tracking.params["tip_contact_names"] = (
            "rl_dg_2_4",
            "rl_dg_3_4",
            "rl_dg_4_4",
            "rl_dg_5_4",
        )
        if self.rewards.orientation_tracking:
            self.rewards.orientation_tracking.params["thumb_contact_name"] = "rl_dg_1_4"
            self.rewards.orientation_tracking.params["tip_contact_names"] = (
                "rl_dg_2_4",
                "rl_dg_3_4",
                "rl_dg_4_4",
                "rl_dg_5_4",
            )
        self.rewards.success.params["thumb_contact_name"] = "rl_dg_1_4"
        self.rewards.success.params["tip_contact_names"] = (
            "rl_dg_2_4",
            "rl_dg_3_4",
            "rl_dg_4_4",
            "rl_dg_5_4",
        )


@configclass
class DexsuiteUR10TessoloReorientEnvCfg(UR10TessoloMixinCfg, dexsuite.DexsuiteReorientEnvCfg):
    pass


@configclass
class DexsuiteUR10TessoloReorientEnvCfg_PLAY(UR10TessoloMixinCfg, dexsuite.DexsuiteReorientEnvCfg_PLAY):
    pass


@configclass
class DexsuiteUR10TessoloLiftEnvCfg(UR10TessoloMixinCfg, dexsuite.DexsuiteLiftEnvCfg):
    pass


@configclass
class DexsuiteUR10TessoloLiftEnvCfg_PLAY(UR10TessoloMixinCfg, dexsuite.DexsuiteLiftEnvCfg_PLAY):
    pass
