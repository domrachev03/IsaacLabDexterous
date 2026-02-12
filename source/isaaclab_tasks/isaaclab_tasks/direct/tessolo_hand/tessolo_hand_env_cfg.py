# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab_assets.robots.tessolo_hand import TESSOLO_HAND_CFG

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg


@configclass
class EventCfg:
    """Configuration for randomization."""

    # -- robot
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        min_step_count_between_reset=720,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (1.0, 1.0),
            "num_buckets": 250,
        },
    )
    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.75, 1.5),
            "damping_distribution_params": (0.3, 3.0),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )
    robot_joint_pos_limits = EventTerm(
        func=mdp.randomize_joint_parameters,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "lower_limit_distribution_params": (0.00, 0.01),
            "upper_limit_distribution_params": (0.00, 0.01),
            "operation": "add",
            "distribution": "gaussian",
        },
    )
    # NOTE: No tendon randomization — Tessolo has no tendons (purely articulated).

    # -- object
    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (1.0, 1.0),
            "num_buckets": 250,
        },
    )
    object_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (0.5, 1.5),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # -- scene
    reset_gravity = EventTerm(
        func=mdp.randomize_physics_scene_gravity,
        mode="interval",
        is_global_time=True,
        interval_range_s=(36.0, 36.0),  # time_s = num_steps * (decimation * dt)
        params={
            "gravity_distribution_params": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.4]),
            "operation": "add",
            "distribution": "gaussian",
        },
    )


@configclass
class TessoloHandEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 10.0
    action_space = 20
    # Full obs: 20 joint_pos + 20 joint_vel + 3 obj_pos + 4 obj_rot + 3 obj_linvel
    #   + 3 obj_angvel + 3 in_hand_pos + 4 goal_rot + 4 rel_goal_rot
    #   + 15 tip_pos + 20 tip_rot + 30 tip_vel + 20 actions = 149
    observation_space = 149
    state_space = 0
    asymmetric_obs = False
    obs_type = "full"

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
        ),
    )
    # robot
    robot_cfg: ArticulationCfg = TESSOLO_HAND_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    actuated_joint_names = [
        "rj_dg_1_1",
        "rj_dg_1_2",
        "rj_dg_1_3",
        "rj_dg_1_4",
        "rj_dg_2_1",
        "rj_dg_2_2",
        "rj_dg_2_3",
        "rj_dg_2_4",
        "rj_dg_3_1",
        "rj_dg_3_2",
        "rj_dg_3_3",
        "rj_dg_3_4",
        "rj_dg_4_1",
        "rj_dg_4_2",
        "rj_dg_4_3",
        "rj_dg_4_4",
        "rj_dg_5_1",
        "rj_dg_5_2",
        "rj_dg_5_3",
        "rj_dg_5_4",
    ]
    fingertip_body_names = [
        "rl_dg_1_tip",
        "rl_dg_2_tip",
        "rl_dg_3_tip",
        "rl_dg_4_tip",
        "rl_dg_5_tip",
    ]

    # in-hand object
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=567.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.132, 0.0, 0.6), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    # goal object
    goal_object_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "goal": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(1.0, 1.0, 1.0),
            )
        },
    )
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=8192, env_spacing=0.75, replicate_physics=True, clone_in_fabric=True
    )

    # reset
    reset_position_noise = 0.01  # range of position at reset
    reset_dof_pos_noise = 0.2  # range of dof pos at reset
    reset_dof_vel_noise = 0.0  # range of dof vel at reset
    # reward scales
    dist_reward_scale = -10.0
    rot_reward_scale = 1.0
    rot_eps = 0.1
    action_penalty_scale = -0.0002
    reach_goal_bonus = 250
    fall_penalty = 0
    fall_dist = 0.24
    vel_obs_scale = 0.2
    success_tolerance = 0.1
    max_consecutive_success = 0
    av_factor = 0.1
    act_moving_average = 1.0
    force_torque_obs_scale = 10.0


@configclass
class TessoloHandOpenAIEnvCfg(TessoloHandEnvCfg):
    # env
    decimation = 3
    episode_length_s = 8.0
    action_space = 20
    # OpenAI obs: 15 tip_pos + 3 obj_pos + 4 rel_goal_rot + 20 actions = 42
    observation_space = 42
    # State: full obs (149) + 30 fingertip_force_torque = 179
    state_space = 179
    asymmetric_obs = True
    obs_type = "openai"
    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
        ),
    )
    # reset
    reset_position_noise = 0.01
    reset_dof_pos_noise = 0.2
    reset_dof_vel_noise = 0.0
    # reward scales
    dist_reward_scale = -10.0
    rot_reward_scale = 1.0
    rot_eps = 0.1
    action_penalty_scale = -0.0002
    reach_goal_bonus = 250
    fall_penalty = -50
    fall_dist = 0.24
    vel_obs_scale = 0.2
    success_tolerance = 0.4
    max_consecutive_success = 50
    av_factor = 0.1
    act_moving_average = 0.3
    force_torque_obs_scale = 10.0
    # domain randomization config (no tendon randomization for Tessolo)
    events: EventCfg = EventCfg()
    # at every time-step add gaussian noise + bias. The bias is a gaussian sampled at reset
    action_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.05, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.015, operation="abs"),
    )
    # at every time-step add gaussian noise + bias. The bias is a gaussian sampled at reset
    observation_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.002, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.0001, operation="abs"),
    )
