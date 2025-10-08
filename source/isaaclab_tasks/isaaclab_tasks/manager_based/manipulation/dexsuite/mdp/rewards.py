# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import math as math_utils
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def action_rate_l2_clamped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1).clamp(
        -1000, 1000
    )


def action_l2_clamped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action), dim=1).clamp(-1000, 1000)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward reaching the object using a tanh-kernel on end-effector distance.

    The reward is close to 1 when the maximum distance between the object and any end-effector body is small.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    asset_pos = asset.data.body_pos_w[:, asset_cfg.body_ids]
    object_pos = object.data.root_pos_w
    object_ee_distance = torch.norm(asset_pos - object_pos[:, None, :], dim=-1).max(dim=-1).values
    return 1 - torch.tanh(object_ee_distance / std)


def any_contact(
    env: ManagerBasedRLEnv,
    threshold: float,
    contact_names: tuple[str, ...] = ("thumb_finger_tip", "index_finger_tip", "middle_finger_tip", "ring_finger_tip"),
) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""

    # FIXME: generalize to different robot arms
    tip_contact: list[ContactSensor] = [
        env.scene.sensors[f"{link}_object_s"].data.force_matrix_w.view(env.num_envs, 3) for link in contact_names
    ]
    # check if contact force is above threshold

    contact_mags = [torch.norm(contact, dim=-1) for contact in tip_contact]
    good_contact_cond1 = torch.stack([mag > threshold for mag in contact_mags], dim=-1).any(dim=-1)

    return good_contact_cond1


def contacts(
    env: ManagerBasedRLEnv,
    threshold: float,
    thumb_contact_name: str | list[str] = "thumb_finger_tip",
    tip_contact_names: tuple[str, ...] = ("index_finger_tip", "middle_finger_tip", "ring_finger_tip"),
) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    thumb_contact_name = thumb_contact_name if not isinstance(thumb_contact_name, str) else [thumb_contact_name]
    # FIXME: generalize to different robot arms
    thumb_contact: list[ContactSensor] = [
        env.scene.sensors[f"{link}_object_s"].data.force_matrix_w.view(env.num_envs, 3) for link in thumb_contact_name
    ]
    tip_contact: list[ContactSensor] = [
        env.scene.sensors[f"{link}_object_s"].data.force_matrix_w.view(env.num_envs, 3) for link in tip_contact_names
    ]
    # check if contact force is above threshold

    thumb_contact_mag = [torch.norm(contact, dim=-1) for contact in thumb_contact]
    contact_mags = [torch.norm(contact, dim=-1) for contact in tip_contact]
    good_contact_cond1 = torch.stack([mag > threshold for mag in thumb_contact_mag], dim=-1).any(dim=-1) & (
        torch.stack([mag > threshold for mag in contact_mags], dim=-1).any(dim=-1)
    )

    return good_contact_cond1


def success_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    align_asset_cfg: SceneEntityCfg,
    pos_std: float,
    rot_std: float | None = None,
    thumb_contact_name: str | list[str] = "thumb_finger_tip",
    tip_contact_names: tuple[str, ...] = ("index_finger_tip", "middle_finger_tip", "ring_finger_tip"),
) -> torch.Tensor:
    """Reward success by comparing commanded pose to the object pose using tanh kernels on error."""

    asset: RigidObject = env.scene[asset_cfg.name]
    object: RigidObject = env.scene[align_asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_w, des_quat_w = combine_frame_transforms(
        asset.data.root_pos_w, asset.data.root_quat_w, command[:, :3], command[:, 3:7]
    )
    pos_err, rot_err = compute_pose_error(des_pos_w, des_quat_w, object.data.root_pos_w, object.data.root_quat_w)
    pos_dist = torch.norm(pos_err, dim=1)
    if not rot_std:
        # square is not necessary but this help to keep the final value between having rot_std or not roughly the same
        return (1 - torch.tanh(pos_dist / pos_std)) ** 2
    rot_dist = torch.norm(rot_err, dim=1)
    return (
        (1 - torch.tanh(pos_dist / pos_std))
        * (1 - torch.tanh(rot_dist / rot_std))
        * contacts(env, 1.0, thumb_contact_name, tip_contact_names).float()
    )


def position_command_error_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    align_asset_cfg: SceneEntityCfg,
    thumb_contact_name: str | list[str] = "thumb_finger_tip",
    tip_contact_names: tuple[str, ...] = ("index_finger_tip", "middle_finger_tip", "ring_finger_tip"),
) -> torch.Tensor:
    """Reward tracking of commanded position using tanh kernel, gated by contact presence."""

    asset: RigidObject = env.scene[asset_cfg.name]
    object: RigidObject = env.scene[align_asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    distance = torch.norm(object.data.root_pos_w - des_pos_w, dim=1)
    return (1 - torch.tanh(distance / std)) * contacts(env, 1.0, thumb_contact_name, tip_contact_names).float()


def orientation_command_error_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    align_asset_cfg: SceneEntityCfg,
    thumb_contact_name: str | list[str] = "thumb_finger_tip",
    tip_contact_names: tuple[str, ...] = ("index_finger_tip", "middle_finger_tip", "ring_finger_tip"),
) -> torch.Tensor:
    """Reward tracking of commanded orientation using tanh kernel, gated by contact presence."""

    asset: RigidObject = env.scene[asset_cfg.name]
    object: RigidObject = env.scene[align_asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current orientations
    des_quat_b = command[:, 3:7]
    des_quat_w = math_utils.quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)
    quat_distance = math_utils.quat_error_magnitude(object.data.root_quat_w, des_quat_w)

    return (1 - torch.tanh(quat_distance / std)) * contacts(env, 1.0, thumb_contact_name, tip_contact_names).float()


def penalize_close_fingers(
    env: ManagerBasedRLEnv,
    min_distance: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=["rj_dg_1_tip", "rj_dg_2_tip", "rj_dg_3_tip", "rj_dg_4_tip", "rj_dg_5_tip"]
    ),
) -> torch.Tensor:
    """Penalize the fingers being too close to each other using tanh kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]
    finger_tips_pos = asset.data.body_pos_w[:, asset_cfg.body_ids]  # (num_envs, num_fingers, 3)
    num_fingers = finger_tips_pos.shape[1]
    if num_fingers < 2:
        return torch.zeros(env.num_envs, device=env.device)

    # Compute pairwise distances between finger tips
    dists = torch.cdist(finger_tips_pos, finger_tips_pos, p=2)  # (num_envs, num_fingers, num_fingers)

    # Create a mask to ignore self-distances (diagonal elements)
    mask = torch.eye(num_fingers, device=env.device).bool().unsqueeze(0)  # (1, num_fingers, num_fingers)
    dists = dists.masked_fill(mask, float("inf")).reshape(dists.shape[0], -1)  # Set self-distances to infinity

    # Get the minimum distance between any two fingers for each environment
    min_dists = dists.min(dim=1).values  # (num_envs,)

    # Penalize if the minimum distance is below the threshold
    rew = torch.clamp(min_distance - min_dists, min=0.0) / min_distance
    return rew
