import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
# from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg

from isaaclab.sensors import ContactSensorCfg
from isaaclab.sensors import CameraCfg

# from isaaclab.utils.math import quat_to_rot_mats

##
# Configuration
##

from .ur_delto_cfg import DELTO_CFG


# =====================================================================================================
# =====================================================================================================
# =====================================================================================================

@configclass
class DeltoEnvCfg(DirectRLEnvCfg):

    # ======================================================================= env params
    decimation = 2
    episode_length_s = 5.0
    action_space = 20
    observation_space = 63
    state_space = 0
    action_scale = 1
    asymmetric_obs = False
    obs_type = "full"
    num_env = 32

    # ======================================================================= simulation

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

    # ======================================================================= robot

    robot_cfg: ArticulationCfg = DELTO_CFG.replace(prim_path="/World/envs/env_.*/Robot").replace()

    arm_joint_names = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint"
    ]

    hand_joint_names = [
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
        "rj_dg_5_4"
    ]

    hand_position = [30, 0, 0, 0, 0, -50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    start_position = [-90.0, -75.0, 120.0, -50.0, 90.0, -90.0]
    lift_delta_deg = [-0.0, -10.0, -10.0, 20.0, 0.0, 0.0]

    # ======================================================================= sensors

    ft_names = [
        "rl_dg_1_4",
        "rl_dg_2_4",
        "rl_dg_3_4",
        "rl_dg_4_4",
        "rl_dg_5_4",
    ]

    contact_sensors = {}
    for name in ft_names:
        contact_sensors[name] = ContactSensorCfg(
            prim_path=f"/World/envs/env_.*/Robot/dg5f_my/{name}",
            update_period=0.0,
            history_length=1,
            debug_vis=False,
            # filter_prim_paths_expr=["/World/envs/env_.*/Cube"],
            track_air_time=False,
        )

    # ======================================================================= objects

    # in-manipulator object
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",  # один куб на среду
        spawn=sim_utils.CylinderCfg(  # используем Box вместо Cone
            radius=0.03,  # кубик 10 см
            height=0.15,
            axis="Z",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=2.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.3),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0),  # красный
                metallic=0.1,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.23, -0.83, 0.1 + 0.15/2),
            # pos=(0.22, -1.0, 0.1),
        ),
    )
    
    table_cfg: RigidObjectCfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Table",  # один куб на среду
            spawn=sim_utils.CuboidCfg(  # используем Box вместо Cone
                size=(0.5, 0.5, 0.1),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True,
                    kinematic_enabled=True
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=15.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.1, 0.1, 0.1),
                    metallic=0.1,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.30, -0.85, 0.05),
            ),
        )
    
    wall_cfg: RigidObjectCfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Wall",  # один куб на среду
            spawn=sim_utils.CuboidCfg(  # используем Box вместо Cone
                size=(0.05, 2.5, 1.5),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True,
                    kinematic_enabled=False
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.01, 0.01, 0.01),
                    metallic=0.1,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(-0.7, -0.85, 1.5/2),
            ),
        )
    
    # cam_cfg = CameraCfg(
    #     prim_path="/World/envs/env_.*/Robot/ee_link/Camera",
    #     update_period=0.1,
    #     height=480,
    #     width=640,
    #     data_types=["rgb", "distance_to_image_plane"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         # focal_length=3.55, focus_distance=270.0, horizontal_aperture=5.76, vertical_aperture=3.24, clipping_range=(0.01, 20.0)
    #         focal_length=1.51, focus_distance=39.0, horizontal_aperture=5.76, vertical_aperture=4.608, clipping_range=(0.01, 20.0)

    #     ),
    #     offset=CameraCfg.OffsetCfg(pos=(0.05, -0.08, 0.0), rot=(0.70290398, 0.70290398, 0.07698051, 0.07698051), convention="world"),
    # )

    # cam_cfg = CameraCfg(
    #     prim_path="/World/envs/env_.*/Table/Camera",
    #     update_period=0.01,
    #     height=84,
    #     width=84,
    #     data_types=["rgb"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=3.55, focus_distance=270.0, horizontal_aperture=5.76, vertical_aperture=3.24, clipping_range=(0.01, 5.0)
    #         # focal_length=1.51, focus_distance=39.0, horizontal_aperture=5.76, vertical_aperture=4.608, clipping_range=(0.01, 20.0)

    #     ),
    #     offset=CameraCfg.OffsetCfg(pos=(0.3, 0.0, 0.2), rot=(0, 0, 0, 1), convention="world"),
    # )

    # ======================================================================= scene

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=32, 
        env_spacing= 3.0, 
        replicate_physics=True
        )
    
    # === двухфазная логика
    grasp_contact_min = 3            # минимум активных пальцев
    grasp_fc_angle_deg = 110.0       # «распора» (force closure) — угол
    grasp_hold_steps = 20            # сколько тактов подряд держать хват
    grasp_max_dist = 0.07            # м: COM объекта близко к центру хвата

    lift_steps = 120                 # длительность подъёма (шагов симуляции)
    success_height = 0.12            # м: на столько поднять выше поверхности стола
    slip_grace = 10                  # допускаем кратковременную потерю контакта, шагов

    tilt_fail_deg = 55.0 
    
    #  ======================================================================= rewards