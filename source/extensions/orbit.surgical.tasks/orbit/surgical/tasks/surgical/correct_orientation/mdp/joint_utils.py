# joint_utils.py
# Utilities to create/destroy a temporary D6 “clamp” joint between the active gripper tip and the needle.

from __future__ import annotations
from typing import Optional, Sequence
import torch

# If your task already exposes this:
try:
    from .visualization import quat_to_rot_matrix  # type: ignore
except Exception:
    quat_to_rot_matrix = None  # type: ignore


# ----------------------------
# Small quaternion utilities
# ----------------------------
def _quat_normalize(q: torch.Tensor) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)

def _quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.stack([w, -x, -y, -z], dim=-1)

def _quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return torch.stack([w, x, y, z], dim=-1)

def _quat_between_vectors(v_from: torch.Tensor, v_to: torch.Tensor) -> torch.Tensor:
    # returns q that rotates v_from -> v_to
    v_f = v_from / v_from.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    v_t = v_to   / v_to.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    dot = (v_f * v_t).sum(dim=-1, keepdim=True)
    axis = torch.cross(v_f, v_t, dim=-1)
    # handle opposite vectors
    opp = (dot.squeeze(-1) < -0.999999)
    if torch.any(opp):
        ortho = torch.tensor([1.0, 0.0, 0.0], device=v_f.device).expand_as(v_f)
        use_y = (v_f.abs() > 0.9).any(dim=-1, keepdim=True)
        ortho = torch.where(use_y, torch.tensor([0.0, 1.0, 0.0], device=v_f.device).expand_as(v_f), ortho)
        axis = torch.cross(v_f, ortho, dim=-1)
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    s = torch.sqrt((1.0 + dot).clamp(min=0.0)) * 0.5
    xyz = axis * (1.0 / (2.0 * s))
    q = torch.cat([s, xyz], dim=-1)
    return _quat_normalize(q)


# ------------------------------------------
# Frame conversion (world -> local joint)
# ------------------------------------------
def make_local_frame(
    body_pos_w: torch.Tensor,        # (...,3)
    body_quat_w: torch.Tensor,       # (...,4) w,x,y,z
    world_anchor: torch.Tensor,      # (...,3)
    world_axis: Optional[torch.Tensor] = None,  # (...,3) or None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (pos_local, quat_local) for a joint frame on the given body."""
    if quat_to_rot_matrix is not None:
        R = quat_to_rot_matrix(body_quat_w)          # (...,3,3)
    else:
        # Fallback: build rotation matrix from quaternion
        w, x, y, z = body_quat_w.unbind(-1)
        xx, yy, zz = x*x, y*y, z*z
        xy, xz, yz = x*y, x*z, y*z
        wx, wy, wz = w*x, w*y, w*z
        R = torch.stack(
            [
                torch.stack([1 - 2*(yy+zz), 2*(xy-wz),     2*(xz+wy)], dim=-1),
                torch.stack([2*(xy+wz),     1 - 2*(xx+zz), 2*(yz-wx)], dim=-1),
                torch.stack([2*(xz-wy),     2*(yz+wx),     1 - 2*(xx+yy)], dim=-1),
            ],
            dim=-2,
        )
    R_inv = R.transpose(-1, -2)
    rel = world_anchor - body_pos_w
    pos_local = torch.matmul(R_inv, rel.unsqueeze(-1)).squeeze(-1)

    if world_axis is None:
        quat_local = torch.tensor([1.0, 0.0, 0.0, 0.0], device=body_pos_w.device).expand(pos_local.shape[:-1] + (4,))
    else:
        x_axis = torch.tensor([1.0, 0.0, 0.0], device=body_pos_w.device).expand_as(world_axis)
        q_world = _quat_between_vectors(x_axis, world_axis)
        quat_local = _quat_multiply(_quat_conjugate(body_quat_w), q_world)  # to body local
    return pos_local, _quat_normalize(quat_local)


# -------------------------------------------------------
# Create / destroy a temporary clamp (per-env handles)
# -------------------------------------------------------
def create_clamp_joint(
    env,
    env_ids: Sequence[int],
    world_anchor: torch.Tensor,       # (K,3) world coords per env id
    world_axis: Optional[torch.Tensor],  # (K,3) or None
    cfg,                              # env.cfg.no_slip
) -> None:
    """Create a D6 joint between active gripper tip body and needle.
    Linear X/Y/Z locked. Angular depends on cfg.clamp_mode.
    """
    N = env.num_envs
    device = world_anchor.device

    if not hasattr(env, "_clamp_joint_handles"):
        env._clamp_joint_handles = [None] * N
    if not hasattr(env, "_clamp_joint_active"):
        env._clamp_joint_active = torch.zeros(N, dtype=torch.bool, device=device)

    # You already track the active pushing side; for simplicity we attach the clamp to robot_1 root
    # and the needle root. If you have specific tip rigid bodies available, substitute them here.
    try:
        robot = env.scene["robot_1"]
        needle = env.scene["object"]
    except Exception:
        print("[joint_utils] Warning: robot_1/object not found in scene; clamp disabled.")
        return

    for i, eid in enumerate(env_ids):
        # Ensure previous clamp is gone
        if env._clamp_joint_active[eid]:
            destroy_clamp_joint(env, [eid])

        anchor_w = world_anchor[i]
        axis_w = None if world_axis is None else world_axis[i]

        tip_pos = robot.data.root_pos_w[eid]
        tip_quat = robot.data.root_quat_w[eid]
        ned_pos = needle.data.root_pos_w[eid]
        ned_quat = needle.data.root_quat_w[eid]

        tip_p_l, tip_q_l = make_local_frame(tip_pos.unsqueeze(0), tip_quat.unsqueeze(0),
                                            anchor_w.unsqueeze(0),
                                            axis_w.unsqueeze(0) if axis_w is not None else None)
        ned_p_l, ned_q_l = make_local_frame(ned_pos.unsqueeze(0), ned_quat.unsqueeze(0),
                                            anchor_w.unsqueeze(0),
                                            axis_w.unsqueeze(0) if axis_w is not None else None)
        tip_p_l, tip_q_l = tip_p_l[0], tip_q_l[0]
        ned_p_l, ned_q_l = ned_p_l[0], ned_q_l[0]

        handle = None
        try:
            # Prefer a scene-level helper if available
            if hasattr(env.scene, "add_d6_joint"):
                handle = env.scene.add_d6_joint(
                    body0=robot,
                    body1=needle,
                    local_pose0=(tip_p_l.tolist(), tip_q_l.tolist()),
                    local_pose1=(ned_p_l.tolist(), ned_q_l.tolist()),
                    # lock all linear axes
                    linear_locks=(True, True, True),
                    # angular: hinge or free with damping
                    angular_locks=(
                        cfg.clamp_mode != "FREE_ANG_DAMP",   # twist free only if FREE_ANG_DAMP
                        cfg.clamp_mode == "HINGE",
                        cfg.clamp_mode == "HINGE",
                    ),
                    angular_damping=cfg.joint_angular_damping if cfg.clamp_mode == "FREE_ANG_DAMP" else 0.0,
                )
            elif hasattr(env, "create_d6_joint"):
                handle = env.create_d6_joint(
                    robot, needle,
                    tip_p_l, tip_q_l,
                    ned_p_l, ned_q_l,
                    linear_locks=(True, True, True),
                    angular_locks=(
                        cfg.clamp_mode != "FREE_ANG_DAMP",
                        cfg.clamp_mode == "HINGE",
                        cfg.clamp_mode == "HINGE",
                    ),
                    angular_damping=cfg.joint_angular_damping if cfg.clamp_mode == "FREE_ANG_DAMP" else 0.0,
                )
            else:
                print(f"[joint_utils] Warning: No API to create D6 joint for env {eid}; clamp skipped.")
        except Exception as exc:
            print(f"[joint_utils] Warning: failed to create clamp for env {eid}: {exc}")

        env._clamp_joint_handles[eid] = handle
        env._clamp_joint_active[eid] = bool(handle)


def destroy_clamp_joint(env, env_ids: Sequence[int]) -> None:
    """Destroy clamp joints for the given envs (if present)."""
    if not hasattr(env, "_clamp_joint_handles"):
        return
    for eid in env_ids:
        h = env._clamp_joint_handles[eid]
        if h is None:
            env._clamp_joint_active[eid] = False
            continue
        try:
            if hasattr(env.scene, "remove_joint"):
                env.scene.remove_joint(h)
            elif hasattr(env, "destroy_joint"):
                env.destroy_joint(h)
            else:
                print(f"[joint_utils] Warning: No API to destroy D6 joint for env {eid}.")
        except Exception as exc:
            print(f"[joint_utils] Warning: failed to destroy clamp for env {eid}: {exc}")
        env._clamp_joint_handles[eid] = None
        env._clamp_joint_active[eid] = False
