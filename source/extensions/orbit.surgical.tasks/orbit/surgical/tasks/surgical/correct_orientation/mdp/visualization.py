import torch
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
import isaaclab.sim as sim_utils
from typing import Union
from isaacsim.core.utils.prims import delete_prim


def quat_to_rot_matrix(quat):
    # quat: (N, 4) in (w, x, y, z)
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    N = quat.shape[0]

    rot = torch.zeros((N, 3, 3), device=quat.device)

    rot[:, 0, 0] = 1 - 2 * (y**2 + z**2)
    rot[:, 0, 1] = 2 * (x * y - z * w)
    rot[:, 0, 2] = 2 * (x * z + y * w)

    rot[:, 1, 0] = 2 * (x * y + z * w)
    rot[:, 1, 1] = 1 - 2 * (x**2 + z**2)
    rot[:, 1, 2] = 2 * (y * z - x * w)

    rot[:, 2, 0] = 2 * (x * z - y * w)
    rot[:, 2, 1] = 2 * (y * z + x * w)
    rot[:, 2, 2] = 1 - 2 * (x**2 + y**2)

    return rot


# Calculates the needle's contact point in world coordinates based on the object's position and orientation.
def _contact_point_world(obj_pos_w: torch.Tensor,
                         obj_quat_w: torch.Tensor,
                         device: torch.device,
                         offset: Union[torch.Tensor, None] = None) -> torch.Tensor:
    """
    Compute a contact point in the world frame, given object position and quaternion.
    Offset is specified in the object's local frame.
    """
    N = obj_pos_w.shape[0]

    # default offset
    if offset is None:
        offset = torch.tensor([0.005, -0.015, 0.0], device=device).unsqueeze(0).repeat(N, 1)
    elif offset.ndim == 1:
        offset = offset.unsqueeze(0).repeat(N, 1)

    # Normalize quaternion to avoid NaNs
    quat_norms = torch.norm(obj_quat_w, dim=1, keepdim=True)
    obj_quat_w = obj_quat_w / (quat_norms + 1e-6)

    # Convert local offset to world frame
    rot_matrix = quat_to_rot_matrix(obj_quat_w)  # (N, 3, 3)
    offset_world = torch.bmm(rot_matrix, offset.unsqueeze(-1)).squeeze(-1)  # (N, 3)

    return obj_pos_w + offset_world


# Calculates the end-effector contact point in world coordinates based on the end-effector frame.
def ee_contact_point_world(ee_frame, device):
    """Computes the actual contact point between grippers (EE + offset)."""
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]       # (N, 3)
    ee_quat = ee_frame.data.target_quat_w[..., 0, :]     # (N, 4)

    # Convert quat to rotation matrix
    ee_rot = quat_to_rot_matrix(ee_quat)  # (N, 3, 3)

    # Offset from EE center to the tip between grippers
    offset_local = torch.tensor([0.0, 0.0, 0.0025], device=device).expand(ee_pos.shape[0], 3)
    offset_world = torch.bmm(ee_rot, offset_local.unsqueeze(-1)).squeeze(-1)  # (N, 3)

    return ee_pos + offset_world


# Visualization marker for contact point on the needle (center of the needle).
contact_marker_cfg = VisualizationMarkersCfg(
    prim_path="/Visuals/ContactPoints",
    markers={
        "sphere": sim_utils.SphereCfg(
            radius=0.0005,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0))
        )
    }
)

contact_marker = VisualizationMarkers(contact_marker_cfg)


def visualize_contact_point(env, env_ids=None):
    needle = env.scene["object"]
    device = env.device
    ids = torch.arange(env.num_envs, device=device) if env_ids is None else env_ids
    N = ids.numel()

    # Get needle pose
    root = needle.data.root_state_w[ids]
    pos = root[:, :3]          # (N, 3)
    quat = root[:, 3:7]        # (N, 4)

    # Convert quat to rot matrix
    rot = quat_to_rot_matrix(quat)  # (N, 3, 3)

    # Define offset in local needle frame: x = lengthwise, y = lateral, z = vertical
    # You can tune these based on physical contact location
    offset_local = torch.tensor([0.005, -0.015, 0.0], device=device).expand(N, 3)  # (N, 3)

    # Rotate offset into world frame
    offset_world = torch.bmm(rot, offset_local.unsqueeze(-1)).squeeze(-1)  # (N, 3)

    # Final contact point
    contact_pts = pos + offset_world

    # Draw markers
    contact_marker.visualize(
        marker_indices=torch.zeros(N, dtype=torch.long),
        translations=contact_pts.cpu().numpy()
    )


# Visualizes the end-effector contact point between the grippers of robot_1.
def visualize_ee_contact_point(env, env_ids=None):
    """Visualizes a point between the grippers of robot_1's end effector."""
    device = env.device
    ids = torch.arange(env.num_envs, device=device) if env_ids is None else env_ids
    N = ids.numel()
    ee_frame = env.scene["ee_1_frame"]

    # Get the base EE contact point
    base_pos = ee_contact_point_world(ee_frame, device)[ids]  # (N, 3)
    ee_quat = ee_frame.data.target_quat_w[ids, 0, :]           # (N, 4)
    rot_matrix = quat_to_rot_matrix(ee_quat)                   # (N, 3, 3)

    # Use the same tip offsets as in the reward
    offset_center = torch.tensor([0.0, 0.0, 0.0], device=device).expand(N, 3)
    contact_pts = base_pos + torch.bmm(rot_matrix, offset_center.unsqueeze(-1)).squeeze(-1)

    if not hasattr(env, "_ee_contact_marker"):
        cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/EEContactPoints",
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=0.0005,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.2, 0.9))
                )
            }
        )
        env._ee_contact_marker = VisualizationMarkers(cfg)

    env._ee_contact_marker.visualize(
        marker_indices=torch.zeros(N, dtype=torch.long),
        translations=contact_pts.cpu().numpy()
    )


# Visualization markers for the gripper tips of robot_1.
gripper_marker_cfg_blue = VisualizationMarkersCfg(
    prim_path="/Visuals/GripperLeft",
    markers={
        "sphere": sim_utils.SphereCfg(
            radius=0.0005,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0))  # Blue
        )
    }
)


gripper_marker_cfg_yellow = VisualizationMarkersCfg(
    prim_path="/Visuals/GripperRight",
    markers={
        "sphere": sim_utils.SphereCfg(
            radius=0.0005,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0))  # Yellow
        )
    }
)


gripper_marker_blue = VisualizationMarkers(gripper_marker_cfg_blue)
gripper_marker_yellow = VisualizationMarkers(gripper_marker_cfg_yellow)


def visualize_gripper_tips(env, env_ids=None):
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    N = ids.numel()

    ee_frame = env.scene["ee_1_frame"]
    device = env.device

    # Get the base EE contact point
    base_pos = ee_contact_point_world(ee_frame, device)[ids]  # (N, 3)
    ee_quat = ee_frame.data.target_quat_w[ids, 0, :]           # (N, 4)
    rot_matrix = quat_to_rot_matrix(ee_quat)                   # (N, 3, 3)

    # Use the same tip offsets as in the reward
    offset_blue = torch.tensor([0.0045, 0.0, 0.0], device=device).expand(N, 3)
    offset_yellow = torch.tensor([-0.0045, 0.0, 0.0], device=device).expand(N, 3)

    tip_blue = base_pos + torch.bmm(rot_matrix, offset_blue.unsqueeze(-1)).squeeze(-1)
    tip_yellow = base_pos + torch.bmm(rot_matrix, offset_yellow.unsqueeze(-1)).squeeze(-1)

    gripper_marker_blue.visualize(
        marker_indices=torch.zeros(N, dtype=torch.long),
        translations=tip_blue.cpu().numpy()
    )
    gripper_marker_yellow.visualize(
        marker_indices=torch.zeros(N, dtype=torch.long),
        translations=tip_yellow.cpu().numpy()
    )


needle_marker_cfg_blue = VisualizationMarkersCfg(
    prim_path="/Visuals/NeedleLeft",
    markers={
        "sphere": sim_utils.SphereCfg(
            radius=0.0005,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0))
        )
    }
)

needle_marker_cfg_yellow = VisualizationMarkersCfg(
    prim_path="/Visuals/NeedleRight",
    markers={
        "sphere": sim_utils.SphereCfg(
            radius=0.0005,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0))
        )
    }
)

needle_marker_blue = VisualizationMarkers(needle_marker_cfg_blue)
needle_marker_yellow = VisualizationMarkers(needle_marker_cfg_yellow)


def visualize_needle_contact_left_right(env, env_ids=None):
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    N = ids.numel()

    needle = env.scene["object"]
    needle_pos = needle.data.root_pos_w[ids]
    needle_quat = needle.data.root_quat_w[ids]

    contact_frame = _contact_point_world(needle_pos, needle_quat, env.device)
    rot_matrix = quat_to_rot_matrix(needle_quat)

    contact_blue_offset = torch.tensor([-0.025, 0.0, 0.0045], device=env.device).expand(N, 3)
    contact_yellow_offset = torch.tensor([-0.025, 0.0, -0.0045], device=env.device).expand(N, 3)

    contact_blue_world = contact_frame + torch.bmm(rot_matrix, contact_blue_offset.unsqueeze(-1)).squeeze(-1)
    contact_yellow_world = contact_frame + torch.bmm(rot_matrix, contact_yellow_offset.unsqueeze(-1)).squeeze(-1)

    needle_marker_blue.visualize(
        marker_indices=torch.zeros(N, dtype=torch.long),
        translations=contact_blue_world.cpu().numpy()
    )
    needle_marker_yellow.visualize(
        marker_indices=torch.zeros(N, dtype=torch.long),
        translations=contact_yellow_world.cpu().numpy()
    )


# Visualization marker for EE-middle contact point on needle
needle_center_marker_cfg = VisualizationMarkersCfg(
    prim_path="/Visuals/NeedleCenterContact",
    markers={
        "sphere": sim_utils.SphereCfg(
            radius=0.0005,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.2, 0.9))  # Purple
        )
    }
)
needle_center_marker = VisualizationMarkers(needle_center_marker_cfg)


def visualize_needle_center_contact(env, env_ids=None):
    """Visualizes a contact point slightly above the needle's main contact center (aligned with EE center)."""
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    N = ids.numel()

    needle = env.scene["object"]
    device = env.device
    needle_pos = needle.data.root_pos_w[ids]
    needle_quat = needle.data.root_quat_w[ids]

    # Needle contact frame
    contact_center = _contact_point_world(needle_pos, needle_quat, device)  # (N, 3)
    needle_rot = quat_to_rot_matrix(needle_quat)  # (N, 3, 3)

    # Offset slightly upward in needle's local Z direction
    offset_local = torch.tensor([-0.025, 0.001, 0.0], device=device).expand(N, 3)
    offset_world = torch.bmm(needle_rot, offset_local.unsqueeze(-1)).squeeze(-1)

    contact_center_above = contact_center + offset_world

    # Visualize
    needle_center_marker.visualize(
        marker_indices=torch.zeros(N, dtype=torch.long),
        translations=contact_center_above.cpu().numpy()
    )


goal_marker_cfg = VisualizationMarkersCfg(
    prim_path="/Visuals/GoalPoint",
    markers={
        "sphere": sim_utils.SphereCfg(
            radius=0.001,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.5, 1.0))  # Purple
        )
    }
)
goal_marker = VisualizationMarkers(goal_marker_cfg)


# def visualize_goal_point(env, env_ids=None):
#     """Visualizes the fixed goal position in world space."""
#     ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
#     N = ids.numel()

#     # Fixed world position (goal point)
#     goal_world = torch.tensor([-0.1863, 0.1419, 0.1296], device=env.device).expand(N, 3)

#     goal_marker.visualize(
#         marker_indices=torch.zeros(N, dtype=torch.long),
#         translations=goal_world.cpu().numpy()
#     )


needle_contact_marker_cfg = VisualizationMarkersCfg(
    prim_path="/Visuals/NeedleContactRel",
    markers={
        "sphere": sim_utils.SphereCfg(
            radius=0.001,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.6, 1.0))  # Light blue
        )
    }
)
needle_contact_marker = VisualizationMarkers(needle_contact_marker_cfg)


def visualize_needle_fixed_contact_relative(env, env_ids=None):
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    device = env.device
    N = ids.numel()

    needle = env.scene["object"]
    needle_pos = needle.data.root_pos_w[ids]         # (N, 3)
    needle_quat = needle.data.root_quat_w[ids]       # (N, 4)
    needle_rot = quat_to_rot_matrix(needle_quat)     # (N, 3, 3)

    # Offset relative to the needle (local frame)
    local_offset = torch.tensor([0.003, 0.0102, 0.0007], device=device).expand(N, 3)

    # Transform to world space
    contact_world = needle_pos + torch.bmm(needle_rot, local_offset.unsqueeze(-1)).squeeze(-1)

    needle_contact_marker.visualize(
        marker_indices=torch.zeros(N, dtype=torch.long),
        translations=contact_world.cpu().numpy()
    )


def visualize_reference_path(env, env_ids=None):
    if not hasattr(env, "reference_path"):
        return
    M = env.num_envs
    device = env.device
    base_ns = getattr(env.scene, "env_ns", "/World/envs/env")

    if env_ids is None:
        env_ids = range(M)

    for eid in env_ids:
        path        = env.reference_path[eid]        # (10,3)
        path_blue   = env.reference_path_blue[eid]
        path_yellow = env.reference_path_yellow[eid]

        for i in range(10):
            # Center (greenish)
            cfg_center = VisualizationMarkersCfg(
                prim_path=f"{base_ns}_{eid}/Visuals/PathDotStep_center_{i}",
                markers={
                    "sphere": sim_utils.SphereCfg(
                        radius=0.0005,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.2, 0.8 - i * 0.05, 0.2 + i * 0.05)
                        )
                    )
                }
            )
            VisualizationMarkers(cfg_center).visualize(
                marker_indices=torch.zeros(1, dtype=torch.long),
                translations=path[i].detach().cpu().unsqueeze(0).numpy(),
            )

            # Blue
            cfg_blue = VisualizationMarkersCfg(
                prim_path=f"{base_ns}_{eid}/Visuals/PathDotStep_blue_{i}",
                markers={
                    "sphere": sim_utils.SphereCfg(
                        radius=0.0005,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.2, 0.3 + i * 0.05, 0.9)
                        )
                    )
                }
            )
            VisualizationMarkers(cfg_blue).visualize(
                marker_indices=torch.zeros(1, dtype=torch.long),
                translations=path_blue[i].detach().cpu().unsqueeze(0).numpy(),
            )

            # Yellow
            cfg_yellow = VisualizationMarkersCfg(
                prim_path=f"{base_ns}_{eid}/Visuals/PathDotStep_yellow_{i}",
                markers={
                    "sphere": sim_utils.SphereCfg(
                        radius=0.0005,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(1.0, 0.8 - i * 0.05, 0.1 + i * 0.05)
                        )
                    )
                }
            )
            VisualizationMarkers(cfg_yellow).visualize(
                marker_indices=torch.zeros(1, dtype=torch.long),
                translations=path_yellow[i].detach().cpu().unsqueeze(0).numpy(),
            )


# Define separate marker configs
grip1_cfg = VisualizationMarkersCfg(
    prim_path="/Visuals/Grip1",
    markers={"sphere": sim_utils.SphereCfg(radius=0.0005, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)))}
)
grip2_cfg = VisualizationMarkersCfg(
    prim_path="/Visuals/Grip2",
    markers={"sphere": sim_utils.SphereCfg(radius=0.001, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)))}
)
tip_cfg = VisualizationMarkersCfg(
    prim_path="/Visuals/Tip",
    markers={"sphere": sim_utils.SphereCfg(radius=0.001, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)))}
)

# Initialize once globally
grip1_marker = VisualizationMarkers(grip1_cfg)
grip2_marker = VisualizationMarkers(grip2_cfg)
tip_marker = VisualizationMarkers(tip_cfg)


def visualize_gripper_links(env, env_ids=None):
    device = env.device
    robot = env.scene["robot_1"]
    ids = torch.arange(env.num_envs, device=device) if env_ids is None else env_ids

    idx_grip1 = robot.body_names.index("psm_tool_gripper1_link")
    idx_grip2 = robot.body_names.index("psm_tool_gripper2_link")
    idx_tip   = robot.body_names.index("psm_tool_tip_link")

    grip1_pos = robot.data.body_pos_w[ids, idx_grip1]
    grip2_pos = robot.data.body_pos_w[ids, idx_grip2]
    tip_pos   = robot.data.body_pos_w[ids, idx_tip]

    marker_ids = torch.zeros(len(ids), dtype=torch.long)

    grip1_marker.visualize(marker_indices=marker_ids, translations=grip1_pos.cpu().numpy())
    grip2_marker.visualize(marker_indices=marker_ids, translations=grip2_pos.cpu().numpy())
    tip_marker.visualize(marker_indices=marker_ids, translations=tip_pos.cpu().numpy())


def visualize_reference_path_2(env, env_id: int = 0, which=("arc_center", "arc_blue", "arc_yellow")):
    """
    Visualize Phase-1 arc paths for a single env.
    Pass `which` as:
      - "arc_center" or ("arc_center",)
      - "arc_blue"   or ("arc_blue",)
      - "arc_yellow" or ("arc_yellow",)
      - any combo, e.g. ("arc_center","arc_blue")
    """
    # normalize input
    if isinstance(which, str):
        which = (which,)

    # gather paths
    paths = {}
    steps = None
    if "arc_center" in which and hasattr(env, "reference_path_2_center") and env.reference_path_2_center.numel() > 0:
        paths["arc_center"] = env.reference_path_2_center[env_id].detach().cpu()
        steps = env.reference_path_2_center.shape[1]
    if "arc_blue" in which and hasattr(env, "reference_path_2_blue") and env.reference_path_2_blue.numel() > 0:
        paths["arc_blue"] = env.reference_path_2_blue[env_id].detach().cpu()
        steps = steps or env.reference_path_2_blue.shape[1]
    if "arc_yellow" in which and hasattr(env, "reference_path_2_yellow") and env.reference_path_2_yellow.numel() > 0:
        paths["arc_yellow"] = env.reference_path_2_yellow[env_id].detach().cpu()
        steps = steps or env.reference_path_2_yellow.shape[1]

    if not paths or steps is None:
        print("[viz] No Phase-1 reference paths to visualize.")
        return

    # clean only selected prefixes
    for prefix in which:
        for i in range(steps):
            delete_prim(f"/Visuals/PathDotStep_{prefix}_{i}")

    palette = {
        "arc_center": (0.2, 0.8, 0.3),
        "arc_blue":   (0.2, 0.4, 0.9),
        "arc_yellow": (0.9, 0.8, 0.2),
    }

    for prefix, pts in paths.items():
        base_color = palette.get(prefix, (0.8, 0.2, 0.5))
        for i in range(steps):
            cfg = VisualizationMarkersCfg(
                prim_path=f"/Visuals/PathDotStep_{prefix}_{i}",
                markers={
                    "sphere": sim_utils.SphereCfg(
                        radius=0.0005,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(
                                max(0.0, min(1.0, base_color[0] - 0.15 * i)),
                                max(0.0, min(1.0, base_color[1] + 0.05 * i)),
                                max(0.0, min(1.0, base_color[2]))
                            )
                        ),
                    )
                },
            )
            VisualizationMarkers(cfg).visualize(
                marker_indices=torch.zeros(1, dtype=torch.long),
                translations=pts[i].unsqueeze(0).numpy(),
            )

def visualize_fixed_point(env):
    point = torch.tensor([-0.21, 0.1419, 0.065])  # hard-coded world coordinate

    cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/FixedPoint",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=0.001,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.1, 0.3))
            )
        }
    )
    marker = VisualizationMarkers(cfg)
    marker.visualize(
        marker_indices=torch.zeros(1, dtype=torch.long),
        translations=point.cpu().unsqueeze(0).numpy()
    )

def visualize_goal_point(env):
    if not hasattr(env, "goal_point"):
        print("No goal_point found in env")
        return

    # visualize the first env’s goal (or loop if you want all)
    point = env.goal_point[0].detach().cpu()

    cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/GoalPoint",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=0.001,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.9, 0.3))
            )
        }
    )
    marker = VisualizationMarkers(cfg)
    marker.visualize(
        marker_indices=torch.zeros(1, dtype=torch.long),
        translations=point.unsqueeze(0).numpy()
    )