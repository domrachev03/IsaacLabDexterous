# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets.robots import PANDA_ROHAND_CFG

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from ... import dexsuite_env_cfg as dexsuite
from ... import mdp


@configclass
class PandaRoHandRelJointPosActionCfg:
    action = mdp.RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint(1-7)", "th_root_link"],
        scale={
            "panda_joint1": 0.1,
            "panda_joint2": 0.1,
            "panda_joint3": 0.1,
            "panda_joint4": 0.1,
            "panda_joint5": 0.1,
            "panda_joint6": 0.1,
            "panda_joint7": 0.1,
            "th_root_link": 0.1,
        },
    )
    sliders = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["(th|if|mf|rf|lf)_slider_link"],
        scale=0.0005,
        use_default_offset=True,
        preserve_order=True,
    )
@configclass
class PandaRoHandReorientRewardCfg(dexsuite.RewardsCfg):
    good_finger_contact = RewTerm(
        func=mdp.contacts,
        weight=1.0,
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
    palm_contact = RewTerm(
        func=mdp.palm_contact,
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
    fix_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint2", "(th|if|mf|rf|lf)_slider_link"]),
            "position_range": [0.0, 0.0],
            "velocity_range": [0.0, 0.0],
        },
    )
    fix_thumb_root = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["th_root_link"]),
            "position_range": [1.56, 1.56],
            "velocity_range": [0.0, 0.0],
        },
    )
    
@configclass
class PandaRoHandMixinCfg:
    rewards: PandaRoHandReorientRewardCfg = PandaRoHandReorientRewardCfg()
    actions: PandaRoHandRelJointPosActionCfg = PandaRoHandRelJointPosActionCfg()
    events: UR10TessoloEventCfg = UR10TessoloEventCfg()

    def __post_init__(self: dexsuite.DexsuiteReorientEnvCfg):
        super().__post_init__()
        self.scene.robot = PANDA_ROHAND_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.commands.object_pose.body_name = "base_link"

        # The RoHand palm attaches under panda_link7, so align the command sampling to the table frame.
        self.scene.table.spawn.activate_contact_sensors = True

        thumb_contact_name = "th_fingertip"
        tip_contact_names = ["if_fingertip", "mf_fingertip", "rf_fingertip", "lf_fingertip"]
        palm_contact_name = "palm_ft"

        finger_body_names = tip_contact_names + [thumb_contact_name]
        for link_name in finger_body_names:
            setattr(
                self.scene,
                f"{link_name}_object_s",
                ContactSensorCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/franka/rohand_left_flattened/" + link_name,
                    filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
                ),
            )

        # Add palm contact sensor
        setattr(
            self.scene,
            f"{palm_contact_name}_object_s",
            ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Robot/franka/rohand_left_flattened/" + palm_contact_name,
                filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
            ),
        )

        self.scene.table_s = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/table",
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        )

        # Add palm contact to observations
        all_contact_bodies = finger_body_names + [palm_contact_name]
        self.observations.proprio.contact = ObsTerm(
            func=mdp.fingers_contact_force_b,
            params={"contact_sensor_names": [f"{link}_object_s" for link in all_contact_bodies]},
            clip=(-20.0, 20.0),
        )
        self.observations.proprio.hand_tips_state_b.params["body_asset_cfg"].body_names = all_contact_bodies

        self.rewards.fingers_to_object.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=all_contact_bodies
        )
        self.events.reset_robot_wrist_joint.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=["panda_joint7"]
        )

        self.rewards.position_tracking.params["thumb_contact_name"] = thumb_contact_name
        self.rewards.position_tracking.params["tip_contact_names"] = tip_contact_names

        if self.rewards.orientation_tracking:
            self.rewards.orientation_tracking.params["thumb_contact_name"] = thumb_contact_name
            self.rewards.orientation_tracking.params["tip_contact_names"] = tip_contact_names

        self.rewards.success.params["thumb_contact_name"] = thumb_contact_name
        self.rewards.success.params["tip_contact_names"] = tip_contact_names

        self.rewards.good_finger_contact.params["thumb_contact_name"] = thumb_contact_name
        self.rewards.good_finger_contact.params["tip_contact_names"] = tip_contact_names

        self.rewards.object_upward_motion.params["thumb_contact_name"] = thumb_contact_name
        self.rewards.object_upward_motion.params["tip_contact_names"] = tip_contact_names

        self.rewards.table_contact_penalty.params["thumb_asset_cfg"] = thumb_contact_name
        self.rewards.table_contact_penalty.params["tip_asset_cfg"] = tip_contact_names

        self.rewards.any_finger_contact.params["contact_names"] = finger_body_names
        self.rewards.palm_contact.params["palm_contact_name"] = palm_contact_name


@configclass
class DexsuitePandaRoHandReorientEnvCfg(PandaRoHandMixinCfg, dexsuite.DexsuiteReorientEnvCfg):
    pass


@configclass
class DexsuitePandaRoHandReorientEnvCfg_PLAY(PandaRoHandMixinCfg, dexsuite.DexsuiteReorientEnvCfg_PLAY):
    pass


@configclass
class DexsuitePandaRoHandLiftEnvCfg(PandaRoHandMixinCfg, dexsuite.DexsuiteLiftEnvCfg):
    pass


@configclass
class DexsuitePandaRoHandLiftEnvCfg_PLAY(PandaRoHandMixinCfg, dexsuite.DexsuiteLiftEnvCfg_PLAY):
    pass
