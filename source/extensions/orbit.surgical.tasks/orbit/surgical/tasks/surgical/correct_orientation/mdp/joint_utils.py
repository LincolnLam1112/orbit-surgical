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

# add near top
from pxr import Usd, UsdPhysics, Gf, Sdf


def _quat_conj(q):  # q: (4,)
    return torch.tensor([ q[0], -q[1], -q[2], -q[3]], device=q.device)

def _quat_mul(a, b):  # both (4,)
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return torch.stack([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw
    ])

def _twist_angle_x(q_rel):  # radians; extract twist about X from relative quat
    w, x, y, z = q_rel
    denom = torch.sqrt(w*w + x*x).clamp(min=1e-9)
    w_t =  w / denom
    x_t =  x / denom
    # sign preserves rotation direction around +X
    ang = 2.0 * torch.atan2(x_t, w_t)
    return ang.item()

def set_hinge_target_to_current(env, eid: int, axis="X"):
    # resolve joint prim path (we stored it when creating the hinge)
    joint_handle = getattr(env, "_pivot_joint_handles", [None]*env.num_envs)[eid]
    if not joint_handle:
        # fallback to constructed path
        pivot = env.scene["needle_pivot_xform"]
        pivot_path = pivot.cfg.prim_path.replace("{ENV_REGEX_NS}", f"{getattr(env.scene,'env_ns','/World/envs/env')}_{eid}")
        joint_handle = f"{pivot_path}/PivotHingeJoint_{eid}"

    jprim = env.scene.stage.GetPrimAtPath(joint_handle)
    if not jprim or not jprim.IsValid():
        print(f"[hinge] joint prim not found for env {eid}: {joint_handle}")
        return

    # world quats (w,x,y,z)
    pivot_q  = env.scene["needle_pivot_xform"].data.root_quat_w[eid]
    needle_q = env.scene["object"].data.root_quat_w[eid]
    q_rel = _quat_mul(_quat_conj(pivot_q), needle_q)  # pivot->needle

    if axis.upper() != "X":
        print("[hinge] only X axis supported in this helper")
    target = _twist_angle_x(q_rel)

    drv = UsdPhysics.DriveAPI(jprim, UsdPhysics.Tokens.angular)
    if not drv:
        drv = UsdPhysics.DriveAPI.Apply(jprim, UsdPhysics.Tokens.angular)
    drv.GetTargetPositionAttr().Set(float(target))
    # keep your small stiffness/damping/maxForce so gripper can overcome
    print(f"[hinge] env {eid} targetPosition set to {target:.3f} rad")

def _quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    vq = torch.cat([torch.zeros_like(v[..., :1]), v], dim=-1)
    return _quat_multiply(_quat_multiply(q, vq), _quat_conjugate(q))[..., 1:4]


def _usd_define_joint_prim(stage, joint_path: str, joint_type: str):
    if joint_type == "fixed":
        return UsdPhysics.FixedJoint.Define(stage, Sdf.Path(joint_path))
    elif joint_type == "revolute":
        return UsdPhysics.RevoluteJoint.Define(stage, Sdf.Path(joint_path))
    else:
        raise ValueError(f"Unsupported joint_type: {joint_type}")

def _usd_set_local_pose(joint, body0_path: str, body1_path: str,
                        p0: torch.Tensor, q0: torch.Tensor,
                        p1: torch.Tensor, q1: torch.Tensor):
    # Bind bodies
    joint.CreateBody0Rel().SetTargets([Sdf.Path(body0_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body1_path)])
    # Local poses (author as Gf)
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(float(p0[0]), float(p0[1]), float(p0[2])))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(float(p1[0]), float(p1[1]), float(p1[2])))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(float(q0[0]), Gf.Vec3f(float(q0[1]), float(q0[2]), float(q0[3]))))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(float(q1[0]), Gf.Vec3f(float(q1[1]), float(q1[2]), float(q1[3]))))

def _create_usd_weld_joint(env, eid: int,
                           body0_prim_path: str, body1_prim_path: str,
                           p0_l: torch.Tensor, q0_l: torch.Tensor,
                           p1_l: torch.Tensor, q1_l: torch.Tensor):
    stage = env.scene.stage  # most IsaacLab envs expose the USD stage here
    joint_path = f"{body0_prim_path}/PivotWeldJoint_{eid}"
    j = _usd_define_joint_prim(stage, joint_path, "fixed")
    _usd_set_local_pose(j, body0_prim_path, body1_prim_path, p0_l, q0_l, p1_l, q1_l)
    return joint_path

from pxr import PhysxSchema

def _create_usd_hinge_joint(env, eid: int,
                            body0_prim_path: str, body1_prim_path: str,
                            p0_l: torch.Tensor, q0_l: torch.Tensor,
                            p1_l: torch.Tensor, q1_l: torch.Tensor,
                            axis: str = "X",
                            hold_stiffness: float = 0.0,
                            hold_damping: float = 2.0,
                            hold_max_torque: float = 0.0000001):
    stage = env.scene.stage
    joint_path = f"{body0_prim_path}/PivotHingeJoint_{eid}"
    j = _usd_define_joint_prim(stage, joint_path, "revolute")
    _usd_set_local_pose(j, body0_prim_path, body1_prim_path, p0_l, q0_l, p1_l, q1_l)
    axis_token = getattr(UsdPhysics.Tokens, axis.lower())
    j.CreateAxisAttr().Set(axis_token)

    # Low-torque “holding brake”: keeps the randomized angle under gravity,
    # but yields when the gripper applies enough torque.
    drive = UsdPhysics.DriveAPI.Apply(j.GetPrim(), UsdPhysics.Tokens.angular)
    drive.CreateTargetPositionAttr().Set(0.0)                  # current pose == 0
    drive.CreateStiffnessAttr().Set(float(hold_stiffness))     # Kp
    drive.CreateDampingAttr().Set(float(hold_damping))         # Kd
    drive.CreateMaxForceAttr().Set(float(hold_max_torque))     # Nm cap (let gripper win)
    drive.CreateTargetVelocityAttr().Set(0.0) 

    # Debug print so you know it actually authored:
    # try:
    #     print("[DEBUG] Hinge created (no PhysxSchema):",
    #           "stiffness=", drive.GetStiffnessAttr().Get(),
    #           "damping=", drive.GetDampingAttr().Get(),
    #           "max_force=", drive.GetMaxForceAttr().Get())
    # except Exception:
    #     pass
    return joint_path

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
def _axis_world_from_choice(needle_quat_w: torch.Tensor, hinge_axis: str, device) -> torch.Tensor:
    """Return a world-space axis (3,) given 'world_x|y|z' or 'needle_x|y|z'."""
    hinge_axis = (hinge_axis or "needle_x").lower()
    base = {
        "x": torch.tensor([1.0, 0.0, 0.0], device=device),
        "y": torch.tensor([0.0, 1.0, 0.0], device=device),
        "z": torch.tensor([0.0, 0.0, 1.0], device=device),
    }
    if hinge_axis.startswith("world_"):
        return base[hinge_axis[-1]]
    # rotate local needle axis into world
    local = base[hinge_axis[-1]]
    # q * v * q_conj
    q = needle_quat_w
    v = torch.tensor([0.0, *local.tolist()], device=device)  # as quaternion (0, x, y, z)
    qv = _quat_multiply(q, v)
    qvq = _quat_multiply(qv, _quat_conjugate(q))
    return qvq[..., 1:4]

def create_pivot_joint(
    env,
    env_ids: Sequence[int],
    mode: str = "WELD",                  # "WELD" or "HINGE"
    hinge_axis: Optional[str] = "needle_x",
    pivot_key: str = "needle_pivot_xform",
    needle_key: str = "object",
    anchor_offset_local=(0.03, 0.042, 0.0),
) -> None:
    
    """
    Create a PhysX D6 joint per env that pins the needle to the pivot at the pivot's world pose.
    - Locks all linear axes.
    - WELD: lock all angular axes.
    - HINGE: free twist (x) in the joint frame; lock (y,z).
    """
    if isinstance(env_ids, int):
        env_ids = [env_ids]
    device = env.device

    # Allocate storage once
    if not hasattr(env, "_pivot_joint_handles"):
        env._pivot_joint_handles = [None] * env.num_envs
    if not hasattr(env, "_pivot_joint_active"):
        env._pivot_joint_active = [False] * env.num_envs

    try:
        pivot = env.scene[pivot_key]
        needle = env.scene[needle_key]
    except Exception as exc:
        print(f"[joint_utils] Pivot/Needle not found ({pivot_key}/{needle_key}): {exc}")
        return

    for i, eid in enumerate(env_ids):
        # Skip if already active
        if env._pivot_joint_active[eid]:
            continue

        # Anchor at the pivot’s world pose (position; axis defines joint frame x-axis)
        pivot_pos_w  = pivot.data.root_pos_w[eid]
        pivot_quat_w = pivot.data.root_quat_w[eid]
        needle = env.scene["object"]
        needle_pos_w  = needle.data.root_pos_w[eid]   # world position (3,)
        needle_quat_w = needle.data.root_quat_w[eid]  # world orientation (w,x,y,z)

        off_l = torch.tensor(anchor_offset_local, device=env.device)

        # world anchor = pivot_pos + R(pivot_quat) * offset_local
        anchor_w = pivot_pos_w + _quat_apply(pivot_quat_w, off_l)
        axis_w = None
        if mode.upper() == "HINGE":
            axis_w = _axis_world_from_choice(needle.data.root_quat_w[eid], hinge_axis, device)

        # Build local joint frames on pivot (parent) and needle (child)
        p_pos_l, p_quat_l = make_local_frame(pivot_pos_w.unsqueeze(0), pivot_quat_w.unsqueeze(0),
                                            anchor_w.unsqueeze(0),
                                            None if axis_w is None else axis_w.unsqueeze(0))
        n_pos_l, n_quat_l = make_local_frame(needle_pos_w.unsqueeze(0), needle_quat_w.unsqueeze(0),
                                            anchor_w.unsqueeze(0),
                                            None if axis_w is None else axis_w.unsqueeze(0))

        p_pos_l, p_quat_l = p_pos_l[0], p_quat_l[0]
        n_pos_l, n_quat_l = n_pos_l[0], n_quat_l[0]

        # Locks
        linear_locks = (True, True, True)
        if mode.upper() == "WELD":
            angular_locks = (True, True, True)
        else:  # HINGE — free twist around joint-frame x
            angular_locks = (False, True, True)

        handle = None
        try:
            if hasattr(env.scene, "add_d6_joint"):
                handle = env.scene.add_d6_joint(
                    body0=pivot,
                    body1=needle,
                    local_pose0=(p_pos_l.tolist(), p_quat_l.tolist()),
                    local_pose1=(n_pos_l.tolist(), n_quat_l.tolist()),
                    linear_locks=linear_locks,
                    angular_locks=angular_locks,
                )
            elif hasattr(env, "create_d6_joint"):
                handle = env.create_d6_joint(
                    pivot, needle,
                    p_pos_l, p_quat_l,
                    n_pos_l, n_quat_l,
                    linear_locks=linear_locks,
                    angular_locks=angular_locks,
                )
            else:
                pivot = env.scene["needle_pivot_xform"]
                needle = env.scene["object"]


                # Prefer cfg.prim_path; if you already got a string from elsewhere, pass it in here.
                pivot_path_tmpl  = getattr(pivot.cfg,  "prim_path",  "")
                needle_path_tmpl = getattr(needle.cfg, "prim_path",  "")


                pivot_path  = _expand_env_path(pivot_path_tmpl,  env, eid)
                needle_path = _expand_env_path(needle_path_tmpl, env, eid)


                # Debug (first env)
                if eid == 0:
                    if hasattr(env, "logger"):
                        env.logger.info(f"[joint_utils] env {eid} pivot_path={pivot_path}")
                        env.logger.info(f"[joint_utils] env {eid} needle_path={needle_path}")
                    # else:
                        # print("[joint_utils] pivot_path:", pivot_path)
                        # print("[joint_utils] needle_path:", needle_path)


                if mode.upper() == "WELD":
                    joint_prim_path = _create_usd_weld_joint(env, eid, pivot_path, needle_path,
                                                            p_pos_l, p_quat_l, n_pos_l, n_quat_l)
                else:
                    # Choose which axis is the hinge in the joint frame; our make_local_frame aligns x to hinge
                    joint_prim_path = _create_usd_hinge_joint(env, eid, pivot_path, needle_path,
                                                            p_pos_l, p_quat_l, n_pos_l, n_quat_l,
                                                            axis="X")


                env._pivot_joint_handles[eid] = joint_prim_path  # store the path as handle
                env._pivot_joint_active[eid] = True
        except Exception as exc:
            print(f"[joint_utils] Failed to create pivot joint for env {eid}: {exc}")

        env._pivot_joint_handles[eid] = handle
        env._pivot_joint_active[eid] = bool(handle)
        if hasattr(env, "logger") and handle:
            env.logger.info(
                f"[PivotJoint] Env {eid}: Created at {anchor_w.cpu().numpy()}, "
                f"mode={mode}, hinge_axis={hinge_axis}, locks=(lin={linear_locks}, ang={angular_locks})"
            )

def destroy_pivot_joint(env, env_ids):
    if isinstance(env_ids, int):
        env_ids = [env_ids]
    stage = env.scene.stage
    for eid in env_ids:
        handle = getattr(env, "_pivot_joint_handles", [None]*env.num_envs)[eid]
        if handle:
            prim = stage.GetPrimAtPath(handle)
            if prim.IsValid():
                stage.RemovePrim(prim.GetPath())
        if hasattr(env, "_pivot_joint_handles"):
            env._pivot_joint_handles[eid] = None
        if hasattr(env, "_pivot_joint_active"):
            env._pivot_joint_active[eid] = False


def setup_needle_pivot_joint(env, env_ids, mode: str | None = None, hinge_axis: str | None = None) -> None:
    # pull defaults from cfg if not provided
    if mode is None and hasattr(env, "cfg") and hasattr(env.cfg, "pivot_joint"):
        mode = getattr(env.cfg.pivot_joint, "mode", "WELD")
    if hinge_axis is None and hasattr(env, "cfg") and hasattr(env.cfg, "pivot_joint"):
        hinge_axis = getattr(env.cfg.pivot_joint, "hinge_axis", "needle_x")
    # env_ids is provided by EventManager; create joints for exactly these envs
    create_pivot_joint(env, env_ids, mode=mode or "WELD", hinge_axis=hinge_axis or "needle_x")


def teardown_needle_pivot_joint(env, env_ids) -> None:
    destroy_pivot_joint(env, env_ids)


import re

def _env_ns(env) -> str:
    # Use scene's namespace if available, else fall back to the common default.
    return getattr(env.scene, "env_ns", "/World/envs/env")

def _expand_env_path(tmpl: str, env, eid: int) -> str:
    """
    Turn a template/regex prim path into a concrete per-env path.
    Handles both {ENV_REGEX_NS} and the baked 'env_.*' regex variant.
    """
    per_env_ns = f"{_env_ns(env)}_{eid}"            # e.g., "/World/envs/env_0"
    s = tmpl
    # 1) Replace template token, if present
    s = s.replace("{ENV_REGEX_NS}", per_env_ns)
    # 2) Replace regex form "env_.*" with "env_{eid}"
    s = re.sub(r"env_\.\*", f"env_{eid}", s)
    return s

def _get_env_prim_path(asset, eid: int) -> str:
    """
    Resolve per-env USD prim path for an IsaacLab asset.
    1) Prefer cfg.prim_path with {ENV_REGEX_NS}.
    2) Fallback to PhysX view internals if available.
    """
    # 1) From cfg template
    if hasattr(asset, "cfg") and hasattr(asset.cfg, "prim_path"):
        tmpl = asset.cfg.prim_path
        if isinstance(tmpl, str) and tmpl:
            if "{ENV_REGEX_NS}" in tmpl:
                return tmpl.replace("{ENV_REGEX_NS}", _env_ns(eid))
            # If no placeholder, assume it's already a full path (single env)
            return tmpl

    # 2) From PhysX view (implementation-dependent)
    # Try root_physx_view or _root_physx_view
    view = getattr(asset, "root_physx_view", None) or getattr(asset, "_root_physx_view", None)
    if view is not None:
        # Common private attr in many builds
        if hasattr(view, "_body_paths"):
            paths = getattr(view, "_body_paths")
            return str(paths[eid])
        # Some builds expose a getter
        if hasattr(view, "get_prim_paths"):
            paths = view.get_prim_paths()
            return str(paths[eid])

    # 3) Give a helpful error
    raise AttributeError(
        f"Cannot determine prim path for asset {type(asset).__name__}. "
        f"Expected asset.cfg.prim_path to be set (with '{{ENV_REGEX_NS}}')."
    )