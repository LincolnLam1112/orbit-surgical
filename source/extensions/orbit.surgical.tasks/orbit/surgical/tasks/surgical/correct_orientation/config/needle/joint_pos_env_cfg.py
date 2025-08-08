from orbit.surgical.assets import ORBITSURGICAL_ASSETS_DATA_DIR

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import RigidObjectCfg, AssetBaseCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from orbit.surgical.tasks.surgical.correct_orientation import mdp
from orbit.surgical.tasks.surgical.correct_orientation.correct_orientation_env_cfg import CorrOrientationEnvCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from orbit.surgical.assets.psm import PSM_CFG  # isort: skip


@configclass
class NeedleOrientationEnvCfg(CorrOrientationEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Set PSM as robot
        self.scene.robot_1 = PSM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_1")
        self.scene.robot_1.init_state.pos = (-0.075, 0.15, 0.22)
        self.scene.robot_1.init_state.rot = (0.7071, 0.0, 0.0, -0.7071)
        self.scene.robot_2 = PSM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_2")
        self.scene.robot_2.init_state.pos = (-0.2, 0.2, 0.15)
        self.scene.robot_2.init_state.rot = (0.7071, -0.7071, 0.0, 0.0)

        self.scene.robot_1.init_state.joint_pos = {
            "psm_main_insertion_joint": 0.06,  # Set just above minimum limit
            "psm_tool_gripper1_joint": -0.5,  # right gripper
            "psm_tool_gripper2_joint": 0.5,  # left gripper
            "psm_tool_yaw_joint": 0.0,
            "psm_tool_pitch_joint": 0.0,
            "psm_tool_roll_joint": 0.0,
        }

        self.scene.robot_2.init_state.joint_pos = {
            "psm_main_insertion_joint": 0.06,  # Set just above minimum limit
            "psm_tool_gripper1_joint": -0.16,  # right gripper
            "psm_tool_gripper2_joint": 0.165  # left gripper
        }

        # Set actions for the specific robot type (PSM)
        self.actions.body_1_joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot_1",
            joint_names=[
                "psm_yaw_joint",
                "psm_pitch_end_joint",
                "psm_main_insertion_joint",
                "psm_tool_roll_joint",
                "psm_tool_pitch_joint",
                "psm_tool_yaw_joint",
            ],
            # Originally set to 0.7
            scale=0.35,
            use_default_offset=True,
        )

        self.actions.finger_1_joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot_1",
            joint_names=[
                "psm_tool_gripper1_joint",
                "psm_tool_gripper2_joint"
            ],
            # Originally set to 0.5
            scale=0.5,
            use_default_offset=True
        )

        self.actions.finger_2_joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot_2",
            joint_names=[
                "psm_tool_gripper1_joint",
                "psm_tool_gripper2_joint"
            ],
            scale=0.0,  # Zero scale to prevent movement
        )

        # Set the body name for the end effector
        self.commands.ee_1_pose.body_name = "psm_tool_tip_link"

        # Create parent prim: /World/envs/env_X/NeedlePivot
        self.scene.needle_pivot_xform = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/NeedlePivot",
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(-0.2, 0.1435, 0.1505),  # <-- set pivot world position here
                rot=(1.0, 0.0, 0.0, 0.0)
            ),
            spawn=UsdFileCfg(
                usd_path=f"{ORBITSURGICAL_ASSETS_DATA_DIR}/Props/Pivot/pivot.usda",  # must define `def Xform "NeedlePivot" {}`
                scale=(0.001, 0.001, 0.001)
            )
        )

        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/NeedlePivot/Object",  # Can still keep child relationship
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.005, 0.0, -0.01),  # Offset so pivot becomes center of rotation
                rot=(0.66446, 0.66446, -0.24184, 0.24184)
            ),
            spawn=UsdFileCfg(
                # Try changing to the usda file later (for friction)
                usd_path=f"{ORBITSURGICAL_ASSETS_DATA_DIR}/Props/Surgical_needle/needle_sdf.usda",
                scale=(500.0, 500.0, 500.0),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=8,
                    max_angular_velocity=50,
                    max_linear_velocity=50,
                    max_depenetration_velocity=0.2,
                    linear_damping=30.0,
                    angular_damping=30.0,
                    disable_gravity=False,
                ),
            ),
            debug_vis=False
        )

        # Debug frame for needle (this is the rigid body)
        needle_marker_cfg = FRAME_MARKER_CFG.copy()
        needle_marker_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
        needle_marker_cfg.prim_path = "/Visuals/NeedleFrameTransformer"

        self.scene.needle_debug = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/NeedlePivot/Object",
            debug_vis=True,
            visualizer_cfg=needle_marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/NeedlePivot/Object",
                    name="needle_debug",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.0),
                        rot=(1.0, 0.0, 0.0, 0.0)
                    )
                ),
            ],
        )

        # Visual-only pivot marker (NOT a rigid object — just to show pivot center)
        self.scene.pivot_marker = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/NeedlePivot/Marker",
            spawn=UsdFileCfg(
                usd_path=f"{ORBITSURGICAL_ASSETS_DATA_DIR}/Props/Pivot/pivot.usda",  # visual only
                scale=(0.001, 0.001, 0.001),
                rigid_props=None
            ),
            debug_vis=True
        )

        # Visual-only frame transformer on pivot (non-physics)
        pivot_marker_cfg = FRAME_MARKER_CFG.copy()
        pivot_marker_cfg.prim_path = "/Visuals/PivotFrame"
        pivot_marker_cfg.markers["frame"].scale = (0.02, 0.02, 0.02)

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_1_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot_1/psm_base_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot_1/psm_tool_tip_link",
                    name="end_effector",
                ),
            ],
        )

        self.scene.ee_1_gripper_left = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot_1/psm_tool_gripper1_link",  # Adjust to actual gripper link name
            debug_vis=False,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot_1/psm_tool_gripper1_link",
                    name="gripper_left",
                )
            ]
        )

        self.scene.ee_1_gripper_right = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot_1/psm_tool_gripper2_link",  # Adjust to actual gripper link name
            debug_vis=False,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot_1/psm_tool_gripper2_link",
                    name="gripper_right",
                )
            ]
        )

        self.scene.ee_2_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot_2/psm_base_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot_2/psm_tool_tip_link",
                    name="end_effector",
                ),
            ],
        )


@configclass
class NeedleOrientationEnvCfg_PLAY(CorrOrientationEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
