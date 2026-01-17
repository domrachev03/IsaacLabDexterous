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
        joint_names=[".*"],
        scale={
            "panda_joint1": 0.1,
            "panda_joint2": 0.1,
            "panda_joint3": 0.1,
            "panda_joint4": 0.1,
            "panda_joint5": 0.1,
            "panda_joint6": 0.1,
            "panda_joint7": 0.1,
            "th_root_link": 0.1,
            "(th|if|mf|rf|lf)_proximal_link": 0.1,
        },
    )
@configclass
class PandaRoHandReorientRewardCfg(dexsuite.RewardsCfg):
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
class PandaRoHandEventCfg(dexsuite.EventCfg):
    randomize_object_scale = EventTerm(
        func=mdp.randomize_rigid_body_scale,
        mode="prestartup",
        params={"scale_range": (0.75, 1.0), "asset_cfg": SceneEntityCfg("object")}, # Limit scaling 
    )

    # Setting absolute friction
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": [1.0, 1.0],                        
            "dynamic_friction_range": [1.0, 1.0],
            "restitution_range": [0.0, 0.0],
            "num_buckets": 250,
        },
    )

    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object", body_names=".*"),
            "static_friction_range": [1.0, 1.0],
            "dynamic_friction_range": [1.0, 1.0],
            "restitution_range": [0.0, 0.0],
            "num_buckets": 250,
        },
    )
    fix_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint2", "(th|if|mf|rf|lf)_proximal_link"]),
            "position_range": [0.0, 0.0],
            "velocity_range": [0.0, 0.0],
        },
    )
    fix_thumb_root = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["th_root_link"]),
            "position_range": [0.0, 1.56],
            "velocity_range": [0.0, 0.0],
        },
    )
    
@configclass
class PandaRoHandMixinCfg:
    rewards: PandaRoHandReorientRewardCfg = PandaRoHandReorientRewardCfg()
    actions: PandaRoHandRelJointPosActionCfg = PandaRoHandRelJointPosActionCfg()
    events: PandaRoHandEventCfg = PandaRoHandEventCfg()

    def __post_init__(self: dexsuite.DexsuiteReorientEnvCfg):
        super().__post_init__()
        self.scene.robot = PANDA_ROHAND_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.commands.object_pose.body_name = "base_link"

        # The RoHand palm attaches under panda_link7, so align the command sampling to the table frame.
        self.scene.table.spawn.activate_contact_sensors = True
        self.commands.object_pose.ranges.pos_x = (0.3, 0.7)

        thumb_contact_name = "th_fingertip"
        tip_contact_names = ["if_fingertip", "mf_fingertip", "rf_fingertip", "lf_fingertip"]

        finger_body_names = tip_contact_names + [thumb_contact_name]
        all_body_names = finger_body_names + ["palm_ft"]
        for link_name in finger_body_names:
            setattr(
                self.scene,
                f"{link_name}_object_s",
                ContactSensorCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/franka/rohand_left_flattened/" + link_name,
                    filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
                ),
            )

        self.scene.table_s = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/table",
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        )

        # Add palm contact to observations
        self.observations.proprio.contact = ObsTerm(
            func=mdp.fingers_contact_force_b,
            params={"contact_sensor_names": [f"{link}_object_s" for link in finger_body_names]},
            clip=(-20.0, 20.0),
        )
        self.observations.proprio.hand_tips_state_b.params["body_asset_cfg"].body_names = all_body_names

        self.rewards.fingers_to_object.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=all_body_names
        )
        self.events.reset_robot_wrist_joint.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=["panda_joint7"]
        )

        self.rewards.position_tracking.params["thumb_contact_name"] = thumb_contact_name
        self.rewards.position_tracking.params["tip_contact_names"] = tip_contact_names
        self.rewards.position_tracking.params["threshold"] = 0.2

        if self.rewards.orientation_tracking:
            self.rewards.orientation_tracking.params["thumb_contact_name"] = thumb_contact_name
            self.rewards.orientation_tracking.params["tip_contact_names"] = tip_contact_names

        self.rewards.success.params["thumb_contact_name"] = thumb_contact_name
        self.rewards.success.params["tip_contact_names"] = tip_contact_names
        self.rewards.success.params["threshold"] = 0.2

        self.rewards.good_finger_contact.params["thumb_contact_name"] = thumb_contact_name
        self.rewards.good_finger_contact.params["tip_contact_names"] = tip_contact_names

        self.rewards.object_upward_motion.params["thumb_contact_name"] = thumb_contact_name
        self.rewards.object_upward_motion.params["tip_contact_names"] = tip_contact_names

        self.rewards.table_contact_penalty.params["thumb_asset_cfg"] = thumb_contact_name
        self.rewards.table_contact_penalty.params["tip_asset_cfg"] = tip_contact_names

        self.rewards.any_finger_contact.params["contact_names"] = finger_body_names


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


# Vision-based (Visible) environments with point cloud observations
@configclass
class DexsuitePandaRoHandReorientVisibleEnvCfg(PandaRoHandMixinCfg, dexsuite.DexsuiteReorientVisibleEnvCfg):
    pass


@configclass
class DexsuitePandaRoHandReorientVisibleEnvCfg_PLAY(PandaRoHandMixinCfg, dexsuite.DexsuiteReorientVisibleEnvCfg_PLAY):
    pass


@configclass
class DexsuitePandaRoHandLiftVisibleEnvCfg(PandaRoHandMixinCfg, dexsuite.DexsuiteLiftVisibleEnvCfg):
    pass


@configclass
class DexsuitePandaRoHandLiftVisibleEnvCfg_PLAY(PandaRoHandMixinCfg, dexsuite.DexsuiteLiftVisibleEnvCfg_PLAY):
    pass
