# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

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
    """Object surface point cloud sampled from segmentation mask and tracked across frames.
    
    This class samples visible object points directly from the camera's segmentation mask
    and depth image, then tracks these keypoints across frames. When a tracked keypoint
    becomes invisible, it is replaced with a new visible point.
    
    The workflow:
    1. Get segmentation mask to find pixels belonging to the object
    2. Back-project visible pixels to 3D world coordinates using depth
    3. Select/track keypoints: keep visible ones, replace invisible with new samples
    4. Transform points to reference frame (e.g., robot base)
    
    Args (from ``cfg.params``):
        object_cfg: Scene entity for the object. Defaults to ``SceneEntityCfg("object")``.
        ref_asset_cfg: Scene entity providing the reference frame. Defaults to ``SceneEntityCfg("robot")``.
        camera_cfg: Scene entity for the camera. Defaults to ``SceneEntityCfg("rgbd_camera")``.
        num_points: Number of keypoints to track. Defaults to ``16``.
        depth_key: Key for depth data in camera output. Defaults to ``"depth"``.
        segmentation_key: Key for segmentation data. Defaults to ``"instance_id_segmentation_fast"``.
        depth_tolerance: Tolerance for depth comparison when checking visibility. Defaults to ``0.01``.
        visualize: Whether to visualize points. Defaults to ``True``.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self._device = env.device
        self._num_envs = env.num_envs
        self.object_cfg: SceneEntityCfg = cfg.params.get("object_cfg", SceneEntityCfg("object"))
        self.ref_asset_cfg: SceneEntityCfg = cfg.params.get("ref_asset_cfg", SceneEntityCfg("robot"))
        self.camera_cfg: SceneEntityCfg = cfg.params.get("camera_cfg", SceneEntityCfg("rgbd_camera"))
        self.num_points: int = cfg.params.get("num_points", 16)
        self.depth_key: str = cfg.params.get("depth_key", "depth")
        self.segmentation_key: str = cfg.params.get("segmentation_key", "instance_id_segmentation_fast")
        self.depth_tolerance: float = cfg.params.get("depth_tolerance", 0.01)
        
        self.object: RigidObject = env.scene[self.object_cfg.name]
        self.ref_asset: Articulation = env.scene[self.ref_asset_cfg.name]
        self.camera = env.scene.sensors[self.camera_cfg.name]
        
        # Visualization
        if cfg.params.get("visualize", True):
            from isaaclab.markers import VisualizationMarkers
            from isaaclab.markers.config import RAY_CASTER_MARKER_CFG

            ray_cfg = RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/ObservationVisiblePointCloud")
            ray_cfg.markers["hit"].radius = 0.0025
            self.visualizer = VisualizationMarkers(ray_cfg)
        else:
            self.visualizer = None

        # Camera dimensions
        self._height = getattr(self.camera.cfg, "height", 1)
        self._width = getattr(self.camera.cfg, "width", 1)
        
        # Tracked keypoints in world frame (num_envs, num_points, 3)
        self._tracked_points_w = torch.zeros((self._num_envs, self.num_points, 3), device=self._device)
        # Validity mask for tracked points (num_envs, num_points) - True if point is valid/initialized
        self._tracked_valid = torch.zeros((self._num_envs, self.num_points), dtype=torch.bool, device=self._device)
        
        # Object instance IDs for segmentation matching
        self._object_instance_ids = torch.full((self._num_envs,), -1, dtype=torch.int64, device=self._device)
        self._object_prim_paths = tuple(self.object.root_physx_view.prim_paths)
        
        # Pre-compute pixel coordinate grids for back-projection
        v_coords, u_coords = torch.meshgrid(
            torch.arange(self._height, device=self._device),
            torch.arange(self._width, device=self._device),
            indexing='ij'
        )
        self._u_grid = u_coords.flatten().float()  # (H*W,)
        self._v_grid = v_coords.flatten().float()  # (H*W,)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        ref_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        camera_cfg: SceneEntityCfg = SceneEntityCfg("rgbd_camera"),
        num_points: int | None = None,
        depth_key: str | None = None,
        segmentation_key: str | None = None,
        depth_tolerance: float | None = None,
        flatten: bool = False,
        visualize: bool = True,
    ):
        """Compute visible object keypoints tracked across frames.
        
        Args:
            env: The environment.
            ref_asset_cfg: Reference frame provider. Defaults to ``SceneEntityCfg("robot")``.
            object_cfg: Object entity. Defaults to ``SceneEntityCfg("object")``.
            camera_cfg: Camera entity. Defaults to ``SceneEntityCfg("rgbd_camera")``.
            num_points: Unused at runtime (set at init).
            depth_key: Unused at runtime (set at init).
            segmentation_key: Unused at runtime (set at init).
            depth_tolerance: Unused at runtime (set at init).
            flatten: If True, return flattened tensor ``(num_envs, 3 * num_points)``.
            visualize: If True, draw markers.
            
        Returns:
            Tensor of shape ``(num_envs, num_points, 3)`` or flattened.
        """
        _ = (ref_asset_cfg, object_cfg, camera_cfg, num_points, depth_key, segmentation_key, depth_tolerance)
        
        if not self._camera_data_ready():
            # Camera not ready - return zeros in reference frame
            return self._format_output(self._tracked_points_w, flatten, visualize)
        
        self._ensure_instance_ids()
        
        # Step 1: Sample visible object points from segmentation mask + depth
        sampled_points_w, sample_valid, _ = self._sample_visible_points_from_mask()
        
        # Step 2: Check visibility of currently tracked points
        tracked_visibility = self._check_tracked_visibility()
        
        # Step 3: Update tracked points - keep visible ones, replace invisible with new samples
        self._update_tracked_points(sampled_points_w, sample_valid, tracked_visibility)
        
        return self._format_output(self._tracked_points_w, flatten, visualize)

    def _camera_data_ready(self) -> bool:
        """Check if camera data is available."""
        outputs = self.camera.data.output
        return self.depth_key in outputs and self.segmentation_key in outputs

    def _sample_visible_points_from_mask(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample 3D points from pixels where object is visible in segmentation mask.
        
        Vectorized implementation that samples a fixed number of points per environment.
        
        Returns:
            sampled_points_w: (num_envs, num_points, 3) - sampled visible points in world frame
            sample_valid: (num_envs, num_points) - True if sample is valid (has visible pixels)
            visible_mask: (num_envs, H*W) - True where object is visible
        """
        depth = self.camera.data.output[self.depth_key][..., 0]  # (num_envs, H, W)
        segmentation = self.camera.data.output[self.segmentation_key][..., 0].to(torch.int64)  # (num_envs, H, W)
        
        # Flatten spatial dimensions
        depth_flat = depth.view(self._num_envs, -1)  # (num_envs, H*W)
        seg_flat = segmentation.view(self._num_envs, -1)  # (num_envs, H*W)
        num_pixels = depth_flat.shape[1]
        
        # Create visibility mask: object instance matches AND valid depth
        object_ids = self._object_instance_ids.unsqueeze(1)  # (num_envs, 1)
        instance_match = (seg_flat == object_ids) & (object_ids >= 0)
        depth_valid = torch.isfinite(depth_flat) & (depth_flat > 0.0)
        visible_mask = instance_match & depth_valid  # (num_envs, H*W)
        
        # Count visible pixels per environment
        visible_counts = visible_mask.sum(dim=1)  # (num_envs,)
        has_visible = visible_counts > 0  # (num_envs,)
        
        # Use multinomial sampling - it's memory efficient and designed for this
        # For envs with no visible pixels, we need to handle specially
        # Create weights: visible pixels get weight 1, others get 0
        # For envs with all zeros, add small uniform weight to avoid multinomial error
        weights = visible_mask.float()  # (num_envs, H*W)

        # For envs with no visible pixels, set uniform weights (will be marked invalid anyway)
        no_visible_mask = ~has_visible  # (num_envs,)
        if no_visible_mask.any():
            weights[no_visible_mask] = 1.0 / num_pixels
        
        # Sample pixel indices using multinomial (with replacement)
        sampled_pixel_idx = torch.multinomial(weights, self.num_points, replacement=True)  # (num_envs, num_points)
        
        # Gather depth values at sampled pixels
        sampled_depth = torch.gather(depth_flat, 1, sampled_pixel_idx)  # (num_envs, num_points)
        
        # Get u, v coordinates for sampled pixels
        sampled_u = self._u_grid[sampled_pixel_idx]  # (num_envs, num_points)
        sampled_v = self._v_grid[sampled_pixel_idx]  # (num_envs, num_points)
        
        # Get camera intrinsics
        fx = self.camera.data.intrinsic_matrices[:, 0, 0].unsqueeze(-1)  # (num_envs, 1)
        fy = self.camera.data.intrinsic_matrices[:, 1, 1].unsqueeze(-1)
        cx = self.camera.data.intrinsic_matrices[:, 0, 2].unsqueeze(-1)
        cy = self.camera.data.intrinsic_matrices[:, 1, 2].unsqueeze(-1)
        
        # Back-project to camera frame
        z_cam = sampled_depth
        x_cam = (sampled_u - cx) * z_cam / fx
        y_cam = (sampled_v - cy) * z_cam / fy
        points_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)  # (num_envs, num_points, 3)
        
        # Transform to world frame
        cam_pos_w = self.camera.data.pos_w.unsqueeze(1)  # (num_envs, 1, 3)
        cam_quat_w = self.camera.data.quat_w_ros.unsqueeze(1).expand(-1, self.num_points, -1)  # (num_envs, num_points, 4)
        
        sampled_points_w = quat_apply(cam_quat_w, points_cam) + cam_pos_w  # (num_envs, num_points, 3)
        
        # Mark validity - samples are valid only if environment has visible pixels
        sample_valid = has_visible.unsqueeze(1).expand(-1, self.num_points)  # (num_envs, num_points)
        
        return sampled_points_w, sample_valid, visible_mask

    def _check_tracked_visibility(self) -> torch.Tensor:
        """Check if currently tracked points are still visible.
        
        Projects tracked points to camera and checks if they match segmentation + depth.
        Fully vectorized implementation for performance.
        
        Returns:
            visibility: (num_envs, num_points) - True if tracked point is still visible
        """
        if not self._tracked_valid.any():
            return torch.zeros_like(self._tracked_valid)
        
        # Transform tracked points to camera frame
        cam_pos_w = self.camera.data.pos_w.unsqueeze(1)  # (num_envs, 1, 3)
        cam_quat_w = self.camera.data.quat_w_ros.unsqueeze(1).expand(-1, self.num_points, -1)
        
        points_cam = quat_apply_inverse(cam_quat_w, self._tracked_points_w - cam_pos_w)  # (num_envs, num_points, 3)
        
        z = points_cam[..., 2]
        x = points_cam[..., 0]
        y = points_cam[..., 1]
        
        # Project to pixel coordinates
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
        
        # Check bounds
        in_bounds = (
            (z > eps)
            & (u_idx >= 0) & (u_idx < self._width)
            & (v_idx >= 0) & (v_idx < self._height)
        )
        
        # Clamp indices to valid range for gathering (will be masked out anyway)
        u_idx_safe = torch.clamp(u_idx, 0, self._width - 1)
        v_idx_safe = torch.clamp(v_idx, 0, self._height - 1)
        
        # Get depth and segmentation
        depth = self.camera.data.output[self.depth_key][..., 0]  # (num_envs, H, W)
        segmentation = self.camera.data.output[self.segmentation_key][..., 0].to(torch.int64)
        
        # Vectorized gather: compute linear indices for (env, v, u) access
        # depth[env_id, v_idx, u_idx] -> use advanced indexing
        env_indices = torch.arange(self._num_envs, device=self._device).unsqueeze(1).expand(-1, self.num_points)
        
        # Gather depth and segmentation values at projected pixel locations
        depth_at_proj = depth[env_indices, v_idx_safe, u_idx_safe]  # (num_envs, num_points)
        seg_at_proj = segmentation[env_indices, v_idx_safe, u_idx_safe]  # (num_envs, num_points)
        
        # Check depth match
        depth_ok = torch.isfinite(depth_at_proj) & (torch.abs(depth_at_proj - z) < self.depth_tolerance)
        
        # Check segmentation match (or object_id < 0 means accept any)
        object_ids = self._object_instance_ids.unsqueeze(1)  # (num_envs, 1)
        seg_ok = (object_ids < 0) | (seg_at_proj == object_ids)
        
        # Combine all conditions
        visibility = in_bounds & self._tracked_valid & depth_ok & seg_ok
        
        return visibility

    def _update_tracked_points(
        self, 
        sampled_points_w: torch.Tensor, 
        sample_valid: torch.Tensor,
        tracked_visibility: torch.Tensor
    ):
        """Update tracked keypoints: keep visible ones, replace invisible with new samples.
        
        Fully vectorized implementation.
        
        Args:
            sampled_points_w: (num_envs, num_points, 3) - newly sampled visible points
            sample_valid: (num_envs, num_points) - True if sample is valid (env has visible pixels)
            tracked_visibility: (num_envs, num_points) - which tracked points are still visible
        """
        # Determine which tracked points to keep (still visible AND were valid)
        keep_mask = tracked_visibility & self._tracked_valid  # (num_envs, num_points)
        
        # Determine which slots need replacement (not keeping)
        needs_replacement = ~keep_mask  # (num_envs, num_points)
        
        # For environments with valid samples, replace non-kept slots with new samples
        # For environments without valid samples, set to ref position and mark invalid
        
        # Get reference positions for invalid envs
        ref_pos_w = self.ref_asset.data.root_pos_w.unsqueeze(1).expand(-1, self.num_points, -1)  # (num_envs, num_points, 3)
        
        # Check which environments have any valid samples
        env_has_valid_samples = sample_valid.any(dim=1)  # (num_envs,)
        
        # Update points: use sampled points where needs_replacement AND sample_valid, else keep tracked
        # For envs without valid samples, use ref_pos
        
        # Create the replacement values
        # Where sample is valid and slot needs replacement -> use sampled_points_w
        # Where sample is valid and slot doesn't need replacement -> keep tracked
        # Where sample is not valid (env has no visible pixels) -> use ref_pos
        
        # Expand env_has_valid_samples for broadcasting
        env_valid_expanded = env_has_valid_samples.unsqueeze(1).unsqueeze(2)  # (num_envs, 1, 1)
        needs_replacement_expanded = needs_replacement.unsqueeze(2)  # (num_envs, num_points, 1)
        
        # Compute new tracked points
        # If env has valid samples:
        #   - If slot needs replacement: use sampled_points_w
        #   - Else: keep self._tracked_points_w
        # Else (no valid samples):
        #   - Use ref_pos_w
        
        new_points = torch.where(
            env_valid_expanded,
            torch.where(needs_replacement_expanded, sampled_points_w, self._tracked_points_w),
            ref_pos_w
        )
        
        # Update validity mask
        # If env has valid samples: all slots become valid (either kept or replaced)
        # Else: all slots become invalid
        new_valid = env_has_valid_samples.unsqueeze(1).expand(-1, self.num_points)
        
        # Write back
        self._tracked_points_w = new_points
        self._tracked_valid = new_valid
        
    def _format_output(self, points_w: torch.Tensor, flatten: bool, visualize: bool) -> torch.Tensor:
        """Transform points to reference frame and format output."""
        ref_pos_w = self.ref_asset.data.root_pos_w.unsqueeze(1).expand(-1, self.num_points, -1)
        ref_quat_w = self.ref_asset.data.root_quat_w.unsqueeze(1).expand(-1, self.num_points, -1)
        
        points_b, _ = subtract_frame_transforms(ref_pos_w, ref_quat_w, points_w, None)
        
        if visualize and self.visualizer is not None:
            self.visualizer.visualize(translations=points_w.view(-1, 3))
        
        return points_b.view(self._num_envs, -1) if flatten else points_b

    def _ensure_instance_ids(self):
        """Populate object instance IDs from camera segmentation info."""
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
        """Extract camera info entry for a specific environment."""
        if isinstance(info_container, (list, tuple)):
            if 0 <= env_idx < len(info_container):
                entry = info_container[env_idx]
                return entry if isinstance(entry, dict) else {}
            return {}
        if isinstance(info_container, dict):
            return info_container
        return {}

    def _resolve_label_string(self, entry) -> str:
        """Recursively resolve label string from segmentation info."""
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
