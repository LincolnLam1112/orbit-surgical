from orbit.surgical.assets import ORBITSURGICAL_ASSETS_DATA_DIR

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import RigidObjectCfg, AssetBaseCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils

from orbit.surgical.tasks.surgical.correct_orientation import mdp
from orbit.surgical.tasks.surgical.correct_orientation.correct_orientation_env_cfg import CorrOrientationEnvCfg

##
# Pre‑defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from orbit.surgical.assets.psm import PSM_CFG  # isort: skip


@configclass
class NeedleOrientationEnvCfg(CorrOrientationEnvCfg):
    """
    Environment configuration for the needle orientation task.  This
    implementation derives from the generic correct orientation environment and
    specializes the scene by spawning a PSM robot and a needle mounted on a
    kinematic pivot.  Compared to the upstream version, the damping on the
    needle has been greatly reduced to allow free rotation around the hinge.
    In addition the pivot joint configuration includes the anchor offset used
    when creating the joint.
    """

    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # Set PSM as robot
        self.scene.robot_1 = PSM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_1")
        self.scene.robot_1.init_state.pos = (0.12, 0.0, 0.2)
        self.scene.robot_1.init_state.rot = (0.7071, 0.0, 0.0, -0.7071)

        # Set the initial joint positions for the robot
        self.scene.robot_1.init_state.joint_pos = {
            "psm_main_insertion_joint": 0.06,  # above minimum limit
            "psm_tool_gripper1_joint": -0.5,
            "psm_tool_gripper2_joint": 0.5,
            "psm_tool_yaw_joint": 0.0,
            "psm_tool_pitch_joint": 0.0,
            "psm_tool_roll_joint": 0.0,
        }

        # Define actions for this robot
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
            scale=0.35,
            # scale=0.0,
            use_default_offset=True,
        )
        self.actions.finger_1_joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot_1",
            joint_names=["psm_tool_gripper1_joint", "psm_tool_gripper2_joint"],
            scale=0.5,
            use_default_offset=True,
        )

        # End effector body used by the high‑level command
        self.commands.ee_1_pose.body_name = "psm_tool_tip_link"

        # Kinematic pivot: a tiny rigid object used only for hierarchy
        self.scene.needle_pivot_xform = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/NeedlePivot",
            init_state=RigidObjectCfg.InitialStateCfg(
                # pos=(-0.200, 0.1435, 0.1505),
                # pos=(-0.2, 0.1435, 0.1),
                # pos=(0.0, 0.0, 0.1),
                pos=(0.0, 0.0, 0.1),
                rot=(0.7071, 0.7071, 0.0, 0.0),
            ),
            spawn=UsdFileCfg(
                usd_path=f"{ORBITSURGICAL_ASSETS_DATA_DIR}/Props/Pivot/pivot.usda",
                scale=(0.001, 0.001, 0.001),
                rigid_props=RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                    disable_gravity=True,
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=8,
                    max_linear_velocity=200,
                    max_angular_velocity=200.0,
                ),
            ),
        )

        # NEEDLE: move out from under the pivot
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Needle",   # ← was {ENV_REGEX_NS}/NeedlePivot/Object
            init_state=RigidObjectCfg.InitialStateCfg(
                # Initial pose can be anything; your reset will place it relative to the pivot.
                pos=(0.0, 0.0, 0.0),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
            spawn=UsdFileCfg(
                usd_path=f"{ORBITSURGICAL_ASSETS_DATA_DIR}/Props/Surgical_needle/needle_sdf.usda",
                scale=(0.5, 0.5, 0.5),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=8,
                    max_angular_velocity=50.0,
                    max_linear_velocity=50.0,
                    max_depenetration_velocity=0.2,
                    linear_damping=0.5,
                    angular_damping=150.0,
                    kinematic_enabled=False,
                    disable_gravity=True,
                ),
            ),
            debug_vis=False,
        )

        # DEBUG FRAME: update to new needle path
        needle_marker_cfg = FRAME_MARKER_CFG.copy()
        needle_marker_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
        needle_marker_cfg.prim_path = "/Visuals/NeedleFrameTransformer"
        self.scene.needle_debug = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Needle",     # ← was .../NeedlePivot/Object
            debug_vis=True,
            visualizer_cfg=needle_marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Needle",  # ← was .../NeedlePivot/Object
                    name="needle_debug",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
                )
            ],
        )

        # # PIVOT MARKER stays under pivot but must remain NON-rigid (you already set rigid_props=None)
        # self.scene.pivot_marker = AssetBaseCfg(
        #     prim_path="{ENV_REGEX_NS}/NeedlePivot/Marker",
        #     spawn=UsdFileCfg(
        #         usd_path=f"{ORBITSURGICAL_ASSETS_DATA_DIR}/Props/Pivot/pivot.usda",
        #         scale=(0.001, 0.001, 0.001),
        #         rigid_props=None,  # keep visual-only
        #     ),
        #     debug_vis=True,
        # )

        # Visual frame on the pivot for debugging
        pivot_marker_cfg = FRAME_MARKER_CFG.copy()
        pivot_marker_cfg.prim_path = "/Visuals/PivotFrame"
        pivot_marker_cfg.markers["frame"].scale = (0.02, 0.02, 0.02)

        # End effector frame transformer
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
                )
            ],
        )

        self.scene.ee_1_gripper_left = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot_1/psm_tool_gripper1_link",
            debug_vis=False,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot_1/psm_tool_gripper1_link",
                    name="gripper_left",
                )
            ],
        )

        self.scene.ee_1_gripper_right = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot_1/psm_tool_gripper2_link",
            debug_vis=False,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot_1/psm_tool_gripper2_link",
                    name="gripper_right",
                )
            ],
        )

        # Pivot joint configuration: use a hinge with the anchor offset matching
        # the needle spawn.  The hinge axis defaults to the needle's local x‑axis.
        self.pivot_joint.mode = "HINGE"
        self.pivot_joint.hinge_axis = "needle_x"
        self.pivot_joint.anchor_offset_local = (-0.01, 0.047, 0.0)  # 1.5cm along needle's +X
        self.pivot_joint.anchor_frame = "needle"  # NEW field


@configclass
class NeedleOrientationEnvCfg_PLAY(CorrOrientationEnvCfg):
    """Reduced‑size configuration for interactive play or debugging."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # Disable observation corruption for clarity when playing
        self.observations.policy.enable_corruption = False