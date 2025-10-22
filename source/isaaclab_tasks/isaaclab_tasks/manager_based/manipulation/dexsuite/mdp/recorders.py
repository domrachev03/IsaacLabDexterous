# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import Sequence

import isaacsim.core.utils.prims as prim_utils

from isaaclab.managers.recorder_manager import (
    DatasetExportMode,
    RecorderManagerBaseCfg,
    RecorderTerm,
    RecorderTermCfg,
)
from isaaclab.utils import configclass
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error


def _sanitize_label(label: str) -> str:
    """Convert an asset label into a metric-friendly name."""
    return label.replace(" ", "_").replace("/", "_")


class DexsuiteSuccessRecorder(RecorderTerm):
    """Recorder term that tracks per-episode success statistics for Dexsuite tasks."""

    def __init__(self, cfg: RecorderTermCfg, env):
        super().__init__(cfg, env)

        self._device = env.device
        self._command_name = getattr(cfg, "command_name", "object_pose")
        self._position_threshold = getattr(cfg, "position_threshold", 0.05)
        self._orientation_threshold = getattr(cfg, "orientation_threshold", 0.5)

        self._robot = env.scene["robot"]
        self._object = env.scene["object"]

        # Resolve per-environment asset labels and indices.
        prim_paths: Sequence[str] = self._object.root_physx_view.prim_paths[: self._object.num_instances]
        env_labels: list[str] = []
        label_to_index: dict[str, int] = {}
        for path in prim_paths:
            prim = prim_utils.get_prim_at_path(path)
            if prim is None:
                label = "Unknown"
            else:
                label_attr = prim.GetAttribute("isaaclab:spawn:asset_label")
                label = label_attr.Get() if label_attr.IsValid() else "Unknown"
            if label not in label_to_index:
                label_to_index[label] = len(label_to_index)
            env_labels.append(label)

        if not label_to_index:
            # Fallback in case metadata is missing.
            label_to_index["Unknown"] = 0
            env_labels = ["Unknown"] * env.num_envs

        self._label_to_index = label_to_index
        self._labels = [label for label, _ in sorted(label_to_index.items(), key=lambda x: x[1])]
        self._metric_labels = [_sanitize_label(label) for label in self._labels]
        self._env_label_indices = torch.tensor(
            [label_to_index[label] for label in env_labels], dtype=torch.long, device=self._device
        )

        num_labels = len(self._labels)
        self._episode_success = torch.zeros(env.num_envs, dtype=torch.bool, device=self._device)
        self._cumulative_success = torch.zeros(num_labels, dtype=torch.long)
        self._cumulative_counts = torch.zeros(num_labels, dtype=torch.long)
        self._total_success = 0
        self._total_attempts = 0
        self._pending_episode_metrics: dict[str, float] | None = None

        # Pre-compute metric keys so we can always populate every entry.
        self._metric_keys = self._build_metric_keys()
        self._zero_metrics_template = {key: 0.0 for key in self._metric_keys}

    def record_pre_step(self):
        # Nothing to record before the physics step.
        return None, None

    def record_post_step(self):
        """Update success flags using the latest simulation state."""
        command_term = self._env.command_manager.get_term(self._command_name)
        command = command_term.command

        # Transform command from robot base to world frame.
        des_pos_w, des_quat_w = combine_frame_transforms(
            self._robot.data.root_pos_w,
            self._robot.data.root_quat_w,
            command[:, :3],
            command[:, 3:7],
        )
        pos_err, rot_err = compute_pose_error(
            des_pos_w,
            des_quat_w,
            self._object.data.root_pos_w,
            self._object.data.root_quat_w,
        )
        pos_dist = torch.norm(pos_err, dim=-1)
        success_now = pos_dist < self._position_threshold

        # Orientation success check (skip when position-only task or threshold disabled).
        position_only = getattr(command_term.cfg, "position_only", False)
        if not position_only and self._orientation_threshold is not None:
            rot_dist = torch.norm(rot_err, dim=-1)
            success_now &= rot_dist < self._orientation_threshold

        self._episode_success |= success_now
        return None, None

    def record_pre_reset(self, env_ids: Sequence[int] | torch.Tensor | None):
        """Log success metrics for environments that are about to be reset."""

        if env_ids is None:
            env_ids = torch.arange(self._env.num_envs, device=self._device, dtype=torch.long)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self._device, dtype=torch.long)
        if env_ids.numel() == 0:
            return None, None

        success_flags = self._episode_success[env_ids]
        label_indices = self._env_label_indices[env_ids]

        success_count = int(success_flags.sum().item())
        attempt_count = int(success_flags.numel())

        # Prepare summary with default zero values so keys are always present.
        summary = self._zero_metrics_template.copy()
        summary["Success_Rate/Episode/success_rate"] = success_count / attempt_count if attempt_count > 0 else 0.0

        # Per-object episode statistics and cumulative updates.
        unique_indices = label_indices.unique(sorted=False)
        for idx in unique_indices.tolist():
            mask = label_indices == idx
            if not torch.any(mask):
                continue
            label_attempts = int(mask.sum().item())
            label_success_count = int(success_flags[mask].sum().item())
            metric_suffix = self._metric_labels[idx]
            summary[f"Success_Rate/Episode/{metric_suffix}/rate"] = (
                label_success_count / label_attempts if label_attempts > 0 else 0.0
            )

            self._cumulative_success[idx] += label_success_count
            self._cumulative_counts[idx] += label_attempts

        self._total_success += success_count
        self._total_attempts += attempt_count

        summary["Success_Rate/Cumulative/success_rate"] = (
            self._total_success / self._total_attempts if self._total_attempts > 0 else 0.0
        )

        for idx, label in enumerate(self._metric_labels):
            total_attempts = int(self._cumulative_counts[idx].item())
            total_success = float(self._cumulative_success[idx].item())
            summary[f"Success_Rate/Cumulative/{label}/rate"] = (
                total_success / total_attempts if total_attempts > 0 else 0.0
            )

        # Store metrics to be published after resets are processed.
        self._pending_episode_metrics = summary

        # Reset the episode success flags for the selected environments.
        self._episode_success[env_ids] = False

        return None, None

    def record_post_reset(self, env_ids: Sequence[int] | torch.Tensor | None):
        summary = self._pending_episode_metrics or self._zero_metrics_template.copy()
        extras = self._env.extras.setdefault("log", {})
        extras.update(summary)
        self._pending_episode_metrics = None
        return None, None

    def _build_metric_keys(self) -> list[str]:
        keys = [
            "Success_Rate/Episode/success_rate",
            "Success_Rate/Cumulative/success_rate",
        ]
        for label in self._metric_labels:
            keys.extend(
                [
                    f"Success_Rate/Episode/{label}/rate",
                    f"Success_Rate/Cumulative/{label}/rate",
                ]
            )
        return keys


@configclass
class DexsuiteSuccessRecorderCfg(RecorderTermCfg):
    """Configuration for the Dexsuite success recorder."""

    class_type: type[RecorderTerm] = DexsuiteSuccessRecorder
    command_name: str = "object_pose"
    position_threshold: float = 0.05
    orientation_threshold: float | None = 0.5


@configclass
class DexsuiteRecorderManagerCfg(RecorderManagerBaseCfg):
    """Recorder manager configuration for Dexsuite environments."""

    dataset_export_mode: DatasetExportMode = DatasetExportMode.EXPORT_NONE
    export_in_record_pre_reset: bool = False
    success_metrics: DexsuiteSuccessRecorderCfg = DexsuiteSuccessRecorderCfg()
