from dataclasses import MISSING

from orbit.surgical.assets import ORBITSURGICAL_ASSETS_DATA_DIR

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass

from . import mdp
from .mdp.phased_orientation_reward_wrapper import PhasedOrientationCMORewardWrapper


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the handover scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """
    # robots: will be populated by agent env cfg
    robot_1: ArticulationCfg = MISSING
    robot_2: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_1_frame: FrameTransformerCfg = MISSING
    ee_2_frame: FrameTransformerCfg = MISSING
    ee_1_gripper_left: FrameTransformerCfg = MISSING
    ee_1_gripper_right: FrameTransformerCfg = MISSING
    # target object: will be populated by agent env cfg
    needle_pivot_xform: AssetBaseCfg = MISSING
    object: RigidObjectCfg = MISSING

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.457)),
        spawn=UsdFileCfg(usd_path=f"{ORBITSURGICAL_ASSETS_DATA_DIR}/Props/Table/table.usd"),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, -0.95)),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    ee_1_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot_1",
        body_name=MISSING,  # will be set by agent env cfg
        resampling_time_range=(5.0, 5.0),
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(-0.05, 0.05),
            pos_y=(-0.05, 0.05),
            pos_z=(-0.12, -0.08),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )
    # set the scale of the visualization markers to (0.01, 0.01, 0.01)
    ee_1_pose.goal_pose_visualizer_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
    ee_1_pose.current_pose_visualizer_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)

    initial_needle_pose = mdp.UniformPoseCommandCfg(
        class_type=mdp.UniformPoseCommand,
        resampling_time_range=(0.0, 0.0),
        debug_vis=False,
        asset_name="object",
        body_name="Object",  # <-- This avoids the call to .find_bodies()
        make_quat_unique=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(-0.1976, -0.1976),  # Pivot position
            pos_y=(0.1448, 0.1448),
            pos_z=(0.1397, 0.1397),
            roll=(1.49, 1.49),
            pitch=(-0.667, -0.667),
            yaw=(0.126, 0.126),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    body_1_joint_pos: mdp.JointPositionActionCfg = MISSING
    finger_1_joint_pos: mdp.JointPositionActionCfg = MISSING
    
    # Add fixed configuration for robot_2's gripper
    finger_2_joint_pos: mdp.JointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # Gives live_align (blue, yellow, and center tips)
        grip_contact_distances = ObsTerm(
            func=mdp.gripper_tip_contact_distances
        )

        # How center ee point aligns with path until current index == 8 (second to last)
        center_path = ObsTerm(
            func=mdp.center_to_path
        )

        # How blue ee point aligns with path until current index == 8 (second to last)
        blue_path = ObsTerm(
            func=mdp.blue_to_path
        )

        # How yellow ee point aligns with path until current index == 8 (second to last)
        yellow_path = ObsTerm(
            func=mdp.yellow_to_path
        )

        path2 = ObsTerm(func=mdp.tip_to_path_obs)

        phase = ObsTerm(
            func=mdp.phase_flags_observation
        )

        # Other observations
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot_1")}
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot_1")}
        )
        actions = ObsTerm(func=mdp.last_action)

        # goal = ObsTerm(func=mdp.goal_observation)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""
    reset_all = EventTerm(func=mdp.reset_only_robot1, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_needle_about_pivot,
        mode="reset",
    )

    reset_mode_flags = EventTerm(func=mdp.reset_mode_flags, mode="reset")


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    path_follower = RewTerm(
        func=PhasedOrientationCMORewardWrapper(
            ee_frame_cfg=SceneEntityCfg("ee_1_frame"),
            object_cfg=SceneEntityCfg("object"),
            robot_cfg=SceneEntityCfg("robot_1"),
        ),
        weight=1.0,  # Weight can remain, CMO-PPO handles trade-off
    )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-3)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")}
    )

    needle_fell = DoneTerm(
        func=mdp.needle_below_table,
        time_out=False
    )

    # phase1 = DoneTerm(
    #     func=mdp.reach_phase_success,
    # )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000}
    )


@configclass
class CorrOrientationEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the handover environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 3.0
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.viewer.eye = (0.0, 0.5, 0.2)
        self.viewer.lookat = (0.0, 0.0, 0.05)