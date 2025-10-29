# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets.robots import UR10_TESSOLO_DELTO_CFG

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.managers import EventTermCfg as EventTerm
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
            "shoulder_pan_joint": 0.1,
            "shoulder_lift_joint": 0.1,
            "elbow_joint": 0.1,
            "wrist_1_joint": 0.1,
            "wrist_2_joint": 0.1,
            "wrist_3_joint": 0.1,
            r"rj_dg_(1|2|3|4|5)_(1|2|3|4)": 0.1,
        },
    )


@configclass
class UR10TessoloReorientRewardCfg(dexsuite.RewardsCfg):
    # bool awarding term if 2 finger tips are in contact with object, one of the contacting fingers has to be thumb.
    good_finger_contact = RewTerm(
        func=mdp.contacts,
        weight=2.0,
        params={
            "threshold": 0.2,
        },
    )
    any_finger_contact = RewTerm(
        func=mdp.any_contact,
        weight=1.0,
        params={
            "threshold": 0.2,
        },
    )

    table_contact_penalty = RewTerm(
        func=mdp.table_contact_penalty,
        weight=-0.5,
        params={
            "table_contact_name": "table_s",
            "threshold": 0.2,
        },
    )

    object_upward_motion = RewTerm(
        func=mdp.object_upward_velocity_bonus,
        weight=0.5,
        params={
            "std": 0.2,
            "threshold": 0.2,
        },
    )

@configclass
class UR10TessoloEventCfg(dexsuite.EventCfg):
    reset_robot_elbow_joint = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="elbow_joint"),
            "position_range": [-0.2, 0.2],
            "velocity_range": [0.0, 0.0],
        },
    )


@configclass
class UR10TessoloMixinCfg:
    rewards: UR10TessoloReorientRewardCfg = UR10TessoloReorientRewardCfg()
    actions: UR10TessoloRelJointPosActionCfg = UR10TessoloRelJointPosActionCfg()
    events: UR10TessoloEventCfg = UR10TessoloEventCfg()

    def __post_init__(self: dexsuite.DexsuiteReorientEnvCfg):
        super().__post_init__()
        self.commands.object_pose.body_name = "rl_dg_mount"
        self.scene.robot = UR10_TESSOLO_DELTO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Replace initial position of UR robot by rotating base around z by 180deg
        self.scene.robot.init_state.rot = (0.0, 0.0, 0.0, 1.0)
        # Rotate command frame to align with robot base frame
        self.commands.object_pose.ranges.pos_x = (0.3, 0.7)
        # Enable contact with table for table_contact_penalty
        self.scene.table.spawn.activate_contact_sensors = True

        self.thumb_contact_name = ("rl_dg_1_tip", "rl_dg_5_tip")
        self.tip_contact_names = ("rl_dg_2_tip", "rl_dg_3_tip", "rl_dg_4_tip")

        finger_tip_body_list = ["rl_dg_1_tip", "rl_dg_2_tip", "rl_dg_3_tip", "rl_dg_4_tip", "rl_dg_5_tip"]
        for link_name in finger_tip_body_list:
            setattr(
                self.scene,
                f"{link_name}_object_s",
                ContactSensorCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/dg5f_my/" + link_name,
                    filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
                ),
            )
        self.scene.table_s = ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/table",
                filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
            )
        

        self.observations.proprio.contact = ObsTerm(
            func=mdp.fingers_contact_force_b,
            params={"contact_sensor_names": [f"{link}_object_s" for link in finger_tip_body_list]},
            clip=(-20.0, 20.0),  # contact force in finger tips is under 20N normally
        )
        self.observations.proprio.hand_tips_state_b.params["body_asset_cfg"].body_names = [
            "rl_dg_mount",
            r"rl_dg_(1|2|3|4|5)_tip",
        ]
        self.rewards.fingers_to_object.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=[r"rl_dg_(1|2|3|4|5)_tip"]
        )
        self.events.reset_robot_wrist_joint.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=["wrist_3_joint"]
        )

        self.rewards.position_tracking.params["thumb_contact_name"] = self.thumb_contact_name
        self.rewards.position_tracking.params["tip_contact_names"] = self.tip_contact_names

        if self.rewards.orientation_tracking:
            self.rewards.orientation_tracking.params["thumb_contact_name"] = self.thumb_contact_name
            self.rewards.orientation_tracking.params["tip_contact_names"] = self.tip_contact_names

        self.rewards.success.params["thumb_contact_name"] = self.thumb_contact_name
        self.rewards.success.params["tip_contact_names"] = self.tip_contact_names

        self.rewards.good_finger_contact.params["thumb_contact_name"] = self.thumb_contact_name
        self.rewards.good_finger_contact.params["tip_contact_names"] = self.tip_contact_names

        self.rewards.object_upward_motion.params["thumb_contact_name"] = self.thumb_contact_name
        self.rewards.object_upward_motion.params["tip_contact_names"] = self.tip_contact_names

        self.rewards.table_contact_penalty.params["thumb_asset_cfg"] = list(self.thumb_contact_name)
        self.rewards.table_contact_penalty.params["tip_asset_cfg"] = list(self.tip_contact_names)

        self.rewards.any_finger_contact.params["contact_names"] = list(self.thumb_contact_name) + list(self.tip_contact_names)

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
