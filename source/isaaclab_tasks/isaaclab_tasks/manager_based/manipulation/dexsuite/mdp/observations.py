# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_inv, quat_mul, subtract_frame_transforms

from .utils import sample_object_point_cloud

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_pos_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
):
    """Object position in the robot's root frame.

    Args:
        env: The environment.
        robot_cfg: Scene entity for the robot (reference frame). Defaults to ``SceneEntityCfg("robot")``.
        object_cfg: Scene entity for the object. Defaults to ``SceneEntityCfg("object")``.

    Returns:
        Tensor of shape ``(num_envs, 3)``: object position [x, y, z] expressed in the robot root frame.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    return quat_apply_inverse(robot.data.root_quat_w, object.data.root_pos_w - robot.data.root_pos_w)


def object_quat_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Object orientation in the robot's root frame.

    Args:
        env: The environment.
        robot_cfg: Scene entity for the robot (reference frame). Defaults to ``SceneEntityCfg("robot")``.
        object_cfg: Scene entity for the object. Defaults to ``SceneEntityCfg("object")``.

    Returns:
        Tensor of shape ``(num_envs, 4)``: object quaternion ``(w, x, y, z)`` in the robot root frame.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    return quat_mul(quat_inv(robot.data.root_quat_w), object.data.root_quat_w)


def body_state_b(
    env: ManagerBasedRLEnv,
    body_asset_cfg: SceneEntityCfg,
    base_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Body state (pos, quat, lin vel, ang vel) in the base asset's root frame.

    The state for each body is stacked horizontally as
    ``[position(3), quaternion(4)(wxyz), linvel(3), angvel(3)]`` and then concatenated over bodies.

    Args:
        env: The environment.
        body_asset_cfg: Scene entity for the articulated body whose links are observed.
        base_asset_cfg: Scene entity providing the reference (root) frame.

    Returns:
        Tensor of shape ``(num_envs, num_bodies * 13)`` with per-body states expressed in the base root frame.
    """
    body_asset: Articulation = env.scene[body_asset_cfg.name]
    base_asset: Articulation = env.scene[base_asset_cfg.name]
    # get world pose of bodies
    body_pos_w = body_asset.data.body_pos_w[:, body_asset_cfg.body_ids].view(-1, 3)
    body_quat_w = body_asset.data.body_quat_w[:, body_asset_cfg.body_ids].view(-1, 4)
    body_lin_vel_w = body_asset.data.body_lin_vel_w[:, body_asset_cfg.body_ids].view(-1, 3)
    body_ang_vel_w = body_asset.data.body_ang_vel_w[:, body_asset_cfg.body_ids].view(-1, 3)
    num_bodies = int(body_pos_w.shape[0] / env.num_envs)
    # get world pose of base frame
    root_pos_w = base_asset.data.root_link_pos_w.unsqueeze(1).repeat_interleave(num_bodies, dim=1).view(-1, 3)
    root_quat_w = base_asset.data.root_link_quat_w.unsqueeze(1).repeat_interleave(num_bodies, dim=1).view(-1, 4)
    # transform from world body pose to local body pose
    body_pos_b, body_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, body_pos_w, body_quat_w)
    body_lin_vel_b = quat_apply_inverse(root_quat_w, body_lin_vel_w)
    body_ang_vel_b = quat_apply_inverse(root_quat_w, body_ang_vel_w)
    # concate and return
    out = torch.cat((body_pos_b, body_quat_b, body_lin_vel_b, body_ang_vel_b), dim=1)
    return out.view(env.num_envs, -1)


class object_point_cloud_b(ManagerTermBase):
    """Object surface point cloud expressed in a reference asset's root frame.

    Points are pre-sampled on the object's surface in its local frame and transformed to world,
    then into the reference (e.g., robot) root frame. Optionally visualizes the points.

    Args (from ``cfg.params``):
        object_cfg: Scene entity for the object to sample. Defaults to ``SceneEntityCfg("object")``.
        ref_asset_cfg: Scene entity providing the reference frame. Defaults to ``SceneEntityCfg("robot")``.
        num_points: Number of points to sample on the object surface. Defaults to ``10``.
        visualize: Whether to draw markers for the points. Defaults to ``True``.
        static: If ``True``, cache world-space points on reset and reuse them (no per-step resampling).

    Returns (from ``__call__``):
        If ``flatten=False``: tensor of shape ``(num_envs, num_points, 3)``.
        If ``flatten=True``: tensor of shape ``(num_envs, 3 * num_points)``.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.object_cfg: SceneEntityCfg = cfg.params.get("object_cfg", SceneEntityCfg("object"))
        self.ref_asset_cfg: SceneEntityCfg = cfg.params.get("ref_asset_cfg", SceneEntityCfg("robot"))
        num_points: int = cfg.params.get("num_points", 10)
        self.object: RigidObject = env.scene[self.object_cfg.name]
        self.ref_asset: Articulation = env.scene[self.ref_asset_cfg.name]
        # lazy initialize visualizer and point cloud
        if cfg.params.get("visualize", True):
            from isaaclab.markers import VisualizationMarkers
            from isaaclab.markers.config import RAY_CASTER_MARKER_CFG

            ray_cfg = RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/ObservationPointCloud")
            ray_cfg.markers["hit"].radius = 0.0025
            self.visualizer = VisualizationMarkers(ray_cfg)
        self.points_local = sample_object_point_cloud(
            env.num_envs, num_points, self.object.cfg.prim_path, device=env.device
        )
        self.points_w = torch.zeros_like(self.points_local)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        ref_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        num_points: int = 10,
        flatten: bool = False,
        visualize: bool = True,
    ):
        """Compute the object point cloud in the reference asset's root frame.

        Note:
            Points are pre-sampled at initialization using ``self.num_points``; the ``num_points`` argument is
            kept for API symmetry and does not change the sampled set at runtime.

        Args:
            env: The environment.
            ref_asset_cfg: Reference frame provider (root). Defaults to ``SceneEntityCfg("robot")``.
            object_cfg: Object to sample. Defaults to ``SceneEntityCfg("object")``.
            num_points: Unused at runtime; see note above.
            flatten: If ``True``, return a flattened tensor ``(num_envs, 3 * num_points)``.
            visualize: If ``True``, draw markers for the points.

        Returns:
            Tensor of shape ``(num_envs, num_points, 3)`` or flattened if requested.
        """
        ref_pos_w = self.ref_asset.data.root_pos_w.unsqueeze(1).repeat(1, num_points, 1)
        ref_quat_w = self.ref_asset.data.root_quat_w.unsqueeze(1).repeat(1, num_points, 1)

        object_pos_w = self.object.data.root_pos_w.unsqueeze(1).repeat(1, num_points, 1)
        object_quat_w = self.object.data.root_quat_w.unsqueeze(1).repeat(1, num_points, 1)
        # apply rotation + translation
        self.points_w = quat_apply(object_quat_w, self.points_local) + object_pos_w
        if visualize:
            self.visualizer.visualize(translations=self.points_w.view(-1, 3))
        object_point_cloud_pos_b, _ = subtract_frame_transforms(ref_pos_w, ref_quat_w, self.points_w, None)

        return object_point_cloud_pos_b.view(env.num_envs, -1) if flatten else object_point_cloud_pos_b


class visible_object_point_cloud_b(ManagerTermBase):
    """Object surface point cloud built from the RGB-D camera-visible surface.

    The class reuses the pre-sampled object surface points, projects them into the camera frame,
    and filters them using both instance segmentation and depth consistency so that only
    camera-visible surface points remain. The result is expressed in the reference asset frame,
    matching the interface of :class:`object_point_cloud_b`.

    Args (from ``cfg.params``):
        object_cfg: Scene entity for the target object. Defaults to ``SceneEntityCfg("object")``.
        ref_asset_cfg: Reference frame provider. Defaults to ``SceneEntityCfg("robot")``.
        camera_cfg: Scene entity for the RGB-D camera. Defaults to ``SceneEntityCfg("rgbd_camera")``.
        num_points: Number of visible points to output. Defaults to ``16``.
        candidate_points: Number of pre-sampled surface points (>= num_points). Defaults to ``64``.
        depth_key: Camera data output key for depth. Defaults to ``"depth"``.
        segmentation_key: Camera data output key for instance IDs. Defaults to ``"instance_id_segmentation_fast"``.
        depth_tolerance: Allowed absolute difference (m) between projected sample depth and camera depth.
        visualize: Whether to draw the filtered points.

    Returns (from ``__call__``):
        Tensor of shape ``(num_envs, num_points, 3)`` (or flattened when requested).
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self._device = env.device
        self.object_cfg: SceneEntityCfg = cfg.params.get("object_cfg", SceneEntityCfg("object"))
        self.ref_asset_cfg: SceneEntityCfg = cfg.params.get("ref_asset_cfg", SceneEntityCfg("robot"))
        self.camera_cfg: SceneEntityCfg = cfg.params.get("camera_cfg", SceneEntityCfg("rgbd_camera"))
        self.num_points: int = cfg.params.get("num_points", 16)
        candidate_points: int = cfg.params.get("candidate_points", max(4 * self.num_points, self.num_points))
        self.depth_key: str = cfg.params.get("depth_key", "depth")
        self.segmentation_key: str = cfg.params.get("segmentation_key", "instance_id_segmentation_fast")
        self.depth_tolerance: float = cfg.params.get("depth_tolerance", 0.01)
        self.object: RigidObject = env.scene[self.object_cfg.name]
        self.ref_asset: Articulation = env.scene[self.ref_asset_cfg.name]
        self.camera = env.scene.sensors[self.camera_cfg.name]
        # visualizer (optional)
        if cfg.params.get("visualize", True):
            from isaaclab.markers import VisualizationMarkers
            from isaaclab.markers.config import RAY_CASTER_MARKER_CFG

            ray_cfg = RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/ObservationVisiblePointCloud")
            ray_cfg.markers["hit"].radius = 0.0025
            self.visualizer = VisualizationMarkers(ray_cfg)
        else:
            self.visualizer = None
        # sample object surface points once and reuse
        candidate_points = max(candidate_points, self.num_points)
        self.points_local = sample_object_point_cloud(
            env.num_envs, candidate_points, self.object.cfg.prim_path, device=self._device
        )
        self.num_candidates = self.points_local.shape[1]
        self.points_w = torch.zeros_like(self.points_local)
        self._visible_points_w = torch.zeros((env.num_envs, self.num_points, 3), device=self._device)
        self._object_instance_ids = torch.full((env.num_envs,), -1, dtype=torch.int64, device=self._device)
        self._object_prim_paths = tuple(self.object.root_physx_view.prim_paths)
        self._env_id_matrix = torch.arange(env.num_envs, device=self._device).unsqueeze(1).repeat(1, self.num_candidates)
        self._fallback_indices = torch.arange(self.num_candidates, device=self._device)
        self._selected_indices = torch.full((env.num_envs, self.num_points), -1, dtype=torch.int64, device=self._device)
        # self._height = self.camera.data.image_shape[0]
        # self._width = self.camera.data.image_shape[1]
        self._height = getattr(self.camera.cfg, "height", 1)
        self._width = getattr(self.camera.cfg, "width", 1)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        ref_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        camera_cfg: SceneEntityCfg = SceneEntityCfg("rgbd_camera"),
        num_points: int | None = None,
        candidate_points: int | None = None,
        depth_key: str | None = None,
        segmentation_key: str | None = None,
        depth_tolerance: float | None = None,
        flatten: bool = False,
        visualize: bool = True,
    ):
        # keep interface-compatible kwargs but ignore overrides at runtime
        _ = (
            env,
            ref_asset_cfg,
            object_cfg,
            camera_cfg,
            num_points,
            candidate_points,
            depth_key,
            segmentation_key,
            depth_tolerance,
        )

        self._update_world_points()
        if not self._camera_data_ready():
            selected_points = self.points_w[:, : self.num_points]
            return self._format_output(selected_points, flatten, visualize)

        self._ensure_instance_ids()
        visible_mask = self._compute_visibility_mask()
        selected_points = self._select_visible_points(visible_mask)
        return self._format_output(selected_points, flatten, visualize)

    def _update_world_points(self):
        object_pos_w = self.object.data.root_pos_w.unsqueeze(1).repeat(1, self.num_candidates, 1)
        object_quat_w = self.object.data.root_quat_w.unsqueeze(1).repeat(1, self.num_candidates, 1)
        self.points_w = quat_apply(object_quat_w, self.points_local) + object_pos_w

    def _camera_data_ready(self) -> bool:
        # ensure the camera buffers exist before attempting to read them
        outputs = self.camera.data.output
        return self.depth_key in outputs and self.segmentation_key in outputs

    def _compute_visibility_mask(self) -> torch.Tensor:
        """Returns a boolean mask (num_envs, num_candidates) for camera-visible points."""
        cam_pos = self.camera.data.pos_w.unsqueeze(1)
        cam_quat = self.camera.data.quat_w_ros.unsqueeze(1).repeat(1, self.num_candidates, 1)
        points_cam = quat_apply_inverse(cam_quat, self.points_w - cam_pos)
        z = points_cam[..., 2]
        x = points_cam[..., 0]
        y = points_cam[..., 1]

        fx = self.camera.data.intrinsic_matrices[:, 0, 0].unsqueeze(-1)
        fy = self.camera.data.intrinsic_matrices[:, 1, 1].unsqueeze(-1)
        cx = self.camera.data.intrinsic_matrices[:, 0, 2].unsqueeze(-1)
        cy = self.camera.data.intrinsic_matrices[:, 1, 2].unsqueeze(-1)

        eps = 1e-6
        inv_z = 1.0 / torch.clamp(z, min=eps)
        u = fx * (x * inv_z) + cx
        v = fy * (y * inv_z) + cy

        u_idx = torch.round(u).to(dtype=torch.int64)
        v_idx = torch.round(v).to(dtype=torch.int64)

        valid = (
            (z > eps)
            & (u_idx >= 0)
            & (u_idx < self._width)
            & (v_idx >= 0)
            & (v_idx < self._height)
        )

        if not valid.any():
            return valid

        depth = self.camera.data.output[self.depth_key][..., 0]
        segmentation = self.camera.data.output[self.segmentation_key][..., 0].to(torch.int64)

        flat_valid = valid.view(-1)
        batch_indices = self._env_id_matrix.view(-1)
        valid_batches = batch_indices[flat_valid]
        valid_u = u_idx.view(-1)[flat_valid]
        valid_v = v_idx.view(-1)[flat_valid]
        sample_depth = z.view(-1)[flat_valid]
        depth_values = depth[valid_batches, valid_v, valid_u]
        seg_values = segmentation[valid_batches, valid_v, valid_u]

        depth_valid = torch.isfinite(depth_values) & (depth_values > 0.0)
        depth_close = torch.abs(depth_values - sample_depth) <= self.depth_tolerance
        object_ids = self._object_instance_ids[valid_batches]
        instance_known = object_ids >= 0
        instance_match = torch.where(
            instance_known, seg_values == object_ids, torch.ones_like(seg_values, dtype=torch.bool)
        )

        visible_flat = torch.zeros_like(flat_valid)
        visible_flat[flat_valid] = depth_valid & depth_close & instance_match
        return visible_flat.view(valid.shape)

    def _select_visible_points(self, visible_mask: torch.Tensor) -> torch.Tensor:
        selected = torch.zeros_like(self._visible_points_w)
        for env_id in range(visible_mask.shape[0]):
            candidate_indices = torch.nonzero(visible_mask[env_id], as_tuple=False).flatten()
            candidate_list = candidate_indices.cpu().tolist()
            visible_set = set(candidate_list)
            chosen: list[int] = []
            if visible_set:
                # keep previously selected indices that remain visible to avoid flicker
                prev = self._selected_indices[env_id].cpu().tolist()
                for idx in prev:
                    if idx in visible_set:
                        chosen.append(idx)
                # append additional visible indices deterministically (ascending order)
                for idx in candidate_list:
                    if len(chosen) >= self.num_points:
                        break
                    if idx not in chosen:
                        chosen.append(idx)
            # if no visible indices, fall back to deterministic candidate order
            if not chosen:
                fallback = self._fallback_indices.cpu().tolist()
                chosen = fallback[: self.num_points]
            # ensure required length by repeating deterministic order
            if len(chosen) < self.num_points:
                repeats = math.ceil(self.num_points / max(len(chosen), 1))
                chosen = (chosen * repeats)[: self.num_points]
            indices_tensor = torch.tensor(chosen, dtype=torch.int64, device=self._device)
            self._selected_indices[env_id] = indices_tensor
            selected[env_id] = self.points_w[env_id, indices_tensor]
        self._visible_points_w = selected
        return selected

    def _format_output(self, points_w: torch.Tensor, flatten: bool, visualize: bool):
        ref_pos_w = self.ref_asset.data.root_pos_w.unsqueeze(1).repeat(1, self.num_points, 1)
        ref_quat_w = self.ref_asset.data.root_quat_w.unsqueeze(1).repeat(1, self.num_points, 1)
        object_point_cloud_pos_b, _ = subtract_frame_transforms(ref_pos_w, ref_quat_w, points_w, None)
        if visualize and getattr(self, "visualizer", None) is not None:
            self.visualizer.visualize(translations=points_w.view(-1, 3))
        return object_point_cloud_pos_b.view(points_w.shape[0], -1) if flatten else object_point_cloud_pos_b

    def _ensure_instance_ids(self):
        missing = torch.nonzero(self._object_instance_ids < 0, as_tuple=False).flatten()
        if missing.numel() == 0:
            return
        info = self.camera.data.info
        for idx in missing.tolist():
            info_entry = self._extract_camera_info_entry(info, idx)
            if not info_entry:
                continue
            seg_info = info_entry.get(self.segmentation_key)
            if seg_info is None:
                continue
            mapping = seg_info.get("idToLabels", {})
            object_path = self._object_prim_paths[idx]
            matched_id = None
            for key, label in mapping.items():
                label_str = self._resolve_label_string(label)
                if label_str and object_path in label_str:
                    try:
                        matched_id = int(key)
                        break
                    except (TypeError, ValueError):
                        continue
            if matched_id is not None:
                self._object_instance_ids[idx] = matched_id

    def _extract_camera_info_entry(self, info_container, env_idx: int) -> dict:
        """Fetch camera info for environment index, handling shared dict or per-env lists."""
        if isinstance(info_container, (list, tuple)):
            if 0 <= env_idx < len(info_container):
                entry = info_container[env_idx]
                return entry if isinstance(entry, dict) else {}
            return {}
        if isinstance(info_container, dict):
            return info_container
        return {}

    def _resolve_label_string(self, entry) -> str:
        if isinstance(entry, str):
            return entry
        if isinstance(entry, dict):
            for value in entry.values():
                resolved = self._resolve_label_string(value)
                if resolved:
                    return resolved
        if isinstance(entry, (list, tuple)):
            for value in entry:
                resolved = self._resolve_label_string(value)
                if resolved:
                    return resolved
        return ""

def fingers_contact_force_b(
    env: ManagerBasedRLEnv,
    contact_sensor_names: list[str],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """base-frame contact forces from listed sensors, concatenated per env.

    Args:
        env: The environment.
        contact_sensor_names: Names of contact sensors in ``env.scene.sensors`` to read.

    Returns:
        Tensor of shape ``(num_envs, 3 * num_sensors)`` with forces stacked horizontally as
        ``[fx, fy, fz]`` per sensor.
    """
    force_w = [env.scene.sensors[name].data.force_matrix_w.view(env.num_envs, 3) for name in contact_sensor_names]
    force_w = torch.stack(force_w, dim=1)
    robot: Articulation = env.scene[asset_cfg.name]
    forces_b = quat_apply_inverse(robot.data.root_link_quat_w.unsqueeze(1).repeat(1, force_w.shape[1], 1), force_w)
    return forces_b
