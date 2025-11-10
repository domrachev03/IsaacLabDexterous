# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from ...dexsuite_env_cfg import VisibleCurriculumCfg, VisibleObservationsCfg
from .dexsuite_ur10_tessolo_env_cfg import (
    DexsuiteUR10TessoloLiftEnvCfg,
    DexsuiteUR10TessoloLiftEnvCfg_PLAY,
    DexsuiteUR10TessoloReorientEnvCfg,
    DexsuiteUR10TessoloReorientEnvCfg_PLAY,
)


@configclass
class DexsuiteUR10TessoloReorientVisibleEnvCfg(DexsuiteUR10TessoloReorientEnvCfg):
    """Camera-visible observation variant of the UR10 Tessolo reorientation task."""

    observations: VisibleObservationsCfg = VisibleObservationsCfg()
    curriculum: VisibleCurriculumCfg | None = VisibleCurriculumCfg()


@configclass
class DexsuiteUR10TessoloReorientVisibleEnvCfg_PLAY(DexsuiteUR10TessoloReorientEnvCfg_PLAY):
    """Evaluation config for the camera-visible UR10 Tessolo reorientation task."""

    observations: VisibleObservationsCfg = VisibleObservationsCfg()
    curriculum: VisibleCurriculumCfg | None = VisibleCurriculumCfg()


@configclass
class DexsuiteUR10TessoloVisibleLiftEnvCfg(DexsuiteUR10TessoloLiftEnvCfg):
    """Camera-visible observation variant of the UR10 Tessolo lift task."""

    observations: VisibleObservationsCfg = VisibleObservationsCfg()
    curriculum: VisibleCurriculumCfg | None = VisibleCurriculumCfg()


@configclass
class DexsuiteUR10TessoloVisibleLiftEnvCfg_PLAY(DexsuiteUR10TessoloLiftEnvCfg_PLAY):
    """Evaluation config for the camera-visible UR10 Tessolo lift task."""

    observations: VisibleObservationsCfg = VisibleObservationsCfg()
    curriculum: VisibleCurriculumCfg | None = VisibleCurriculumCfg()
