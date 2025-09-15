# joint_utils.py (modified)
# Utilities to create/destroy a temporary joint between the pivot and needle.

"""
This module provides helpers for constructing a temporary joint between the
kinematic pivot and the dynamic surgical needle.  The original implementation
only supported a welded joint and exposed a hinge via a D6 (six‑degree of
freedom) constraint.  In practice the D6 joint proved numerically stiff
because it locks two of the rotational axes with extremely high damping.  As
a result the needle could not rotate freely about its long axis and the RL
policy stopped exploring.  This rewritten version adds the following
improvements:

* The anchor offset can be passed in from the environment configuration.  If
  omitted, the offset will be inferred from the needle’s initial pose.  Using
  the correct offset is important when the needle is spawned with an offset
  relative to the pivot (for example `(0.007, 0.0, -0.01)` in the provided
  config).
* For hinge mode we always create a native USD `RevoluteJoint` instead of
  using a D6 joint.  Revolute joints are considerably more stable when only
  a single rotational degree of freedom is required.
* Joint creation no longer overwrites the internal handle on failure and the
  fallback branch is executed deterministically.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch

# If your task already exposes this:
try:
    from .visualization import quat_to_rot_matrix  # type: ignore
except Exception:
    quat_to_rot_matrix = None  # type: ignore

from pxr import Usd, UsdPhysics, Gf, Sdf


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
                        p1: torch.Tensor, q1: torch.Tensor) -> None:
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
                           p1_l: torch.Tensor, q1_l: torch.Tensor) -> str:
    stage = env.scene.stage
    joint_path = f"{body0_prim_path}/PivotWeldJoint_{eid}"
    j = _usd_define_joint_prim(stage, joint_path, "fixed")
    _usd_set_local_pose(j, body0_prim_path, body1_prim_path, p0_l, q0_l, p1_l, q1_l)
    return joint_path


def _create_usd_hinge_joint(env, eid: int,
                            body0_prim_path: str, body1_prim_path: str,
                            p0_l: torch.Tensor, q0_l: torch.Tensor,
                            p1_l: torch.Tensor, q1_l: torch.Tensor,
                            axis: str = "X") -> str:
    stage = env.scene.stage
    joint_path = f"{body0_prim_path}/PivotHingeJoint_{eid}"
    j = _usd_define_joint_prim(stage, joint_path, "revolute")
    _usd_set_local_pose(j, body0_prim_path, body1_prim_path, p0_l, q0_l, p1_l, q1_l)
    # Axis can be "X", "Y", or "Z"
    axis_token = getattr(UsdPhysics.Tokens, axis.lower())
    j.CreateAxisAttr().Set(axis_token)
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
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)


def _quat_between_vectors(v_from: torch.Tensor, v_to: torch.Tensor) -> torch.Tensor:
    # returns q that rotates v_from -> v_to
    v_f = v_from / v_from.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    v_t = v_to / v_to.norm(dim=-1, keepdim=True).clamp(min=1e-8)
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
    body_pos_w: torch.Tensor,  # (...,3)
    body_quat_w: torch.Tensor,  # (...,4) w,x,y,z
    world_anchor: torch.Tensor,  # (...,3)
    world_axis: Optional[torch.Tensor] = None,  # (...,3) or None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (pos_local, quat_local) for a joint frame on the given body."""
    # Convert rotation to matrix
    if quat_to_rot_matrix is not None:
        R = quat_to_rot_matrix(body_quat_w)
    else:
        w, x, y, z = body_quat_w.unbind(-1)
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z
        R = torch.stack(
            [
                torch.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)], dim=-1),
                torch.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)], dim=-1),
                torch.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], dim=-1),
            ],
            dim=-2,
        )
    R_inv = R.transpose(-1, -2)
    rel = world_anchor - body_pos_w
    pos_local = torch.matmul(R_inv, rel.unsqueeze(-1)).squeeze(-1)

    if world_axis is None:
        quat_local = torch.tensor([1.0, 0.0, 0.0, 0.0], device=body_pos_w.device).expand(
            pos_local.shape[:-1] + (4,)
        )
    else:
        x_axis = torch.tensor([1.0, 0.0, 0.0], device=body_pos_w.device).expand_as(world_axis)
        q_world = _quat_between_vectors(x_axis, world_axis)
        quat_local = _quat_multiply(_quat_conjugate(body_quat_w), q_world)
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
    q = needle_quat_w
    v = torch.tensor([0.0, *local.tolist()], device=device)
    qv = _quat_multiply(q, v)
    qvq = _quat_multiply(qv, _quat_conjugate(q))
    return qvq[..., 1:4]


def create_pivot_joint(
    env,
    env_ids: Sequence[int],
    mode: str = "WELD",  # "WELD" or "HINGE"
    hinge_axis: Optional[str] = "needle_x",
    pivot_key: str = "needle_pivot_xform",
    needle_key: str = "object",
    anchor_offset_local: Optional[Sequence[float]] = None,
) -> None:
    """
    Create a PhysX joint between the pivot and needle in each environment.  The
    pivot joint locks all translations.  For weld mode all angular axes are
    also locked; for hinge mode the joint is free to rotate about its x‑axis.

    The anchor offset may be overridden on a per‑task basis.  If not provided
    the function will attempt to infer it from the needle’s spawn position
    relative to the pivot.
    """
    if isinstance(env_ids, int):
        env_ids = [env_ids]
    device = env.device

    # Allocate storage once on the environment
    if not hasattr(env, "_pivot_joint_handles"):
        env._pivot_joint_handles = [None] * env.num_envs
    if not hasattr(env, "_pivot_joint_active"):
        env._pivot_joint_active = [False] * env.num_envs

    # Resolve scene objects
    try:
        pivot_asset = env.scene[pivot_key]
        needle_asset = env.scene[needle_key]
    except Exception as exc:
        print(f"[joint_utils] Pivot/Needle not found ({pivot_key}/{needle_key}): {exc}")
        return

    # Determine default anchor offset from the needle’s spawn definition if none provided
    if anchor_offset_local is None:
        # Use the rigid object initial spawn offset if available
        default_offset = (0.0, 0.0, 0.0)
        try:
            init_state = getattr(needle_asset.cfg, "init_state", None)
            if init_state is not None and hasattr(init_state, "pos"):
                default_offset = tuple(init_state.pos)
        except Exception:
            pass
        anchor_offset_local = default_offset

    off_l_tensor = torch.tensor(anchor_offset_local, device=device)

    for eid in env_ids:
        # Skip if already active
        if env._pivot_joint_active[eid]:
            continue

        # Compute anchor in world space from pivot pose and local offset
        pivot_pos_w = pivot_asset.data.root_pos_w[eid]
        pivot_quat_w = pivot_asset.data.root_quat_w[eid]
        needle_pos_w = needle_asset.data.root_pos_w[eid]
        needle_quat_w = needle_asset.data.root_quat_w[eid]

        anchor_w = pivot_pos_w + _quat_apply(pivot_quat_w, off_l_tensor)
        axis_w = None
        if mode.upper() == "HINGE":
            axis_w = _axis_world_from_choice(needle_asset.data.root_quat_w[eid], hinge_axis, device)

        # Build local joint frames on pivot (parent) and needle (child)
        p_pos_l, p_quat_l = make_local_frame(
            pivot_pos_w.unsqueeze(0), pivot_quat_w.unsqueeze(0), anchor_w.unsqueeze(0),
            None if axis_w is None else axis_w.unsqueeze(0)
        )
        n_pos_l, n_quat_l = make_local_frame(
            needle_pos_w.unsqueeze(0), needle_quat_w.unsqueeze(0), anchor_w.unsqueeze(0),
            None if axis_w is None else axis_w.unsqueeze(0)
        )
        p_pos_l, p_quat_l = p_pos_l[0], p_quat_l[0]
        n_pos_l, n_quat_l = n_pos_l[0], n_quat_l[0]

        # Locks: translations locked, angular depends on mode
        linear_locks = (True, True, True)
        if mode.upper() == "WELD":
            angular_locks = (True, True, True)
        else:
            # free rotation around joint frame x
            angular_locks = (False, True, True)

        # Attempt to create a D6 joint only for weld joints.  Revolute joints
        # provide much better numerical stability for hinge joints and are
        # handled in the fallback path below.
        handle = None
        try:
            if mode.upper() == "WELD":
                if hasattr(env.scene, "add_d6_joint"):
                    handle = env.scene.add_d6_joint(
                        body0=pivot_asset,
                        body1=needle_asset,
                        local_pose0=(p_pos_l.tolist(), p_quat_l.tolist()),
                        local_pose1=(n_pos_l.tolist(), n_quat_l.tolist()),
                        linear_locks=linear_locks,
                        angular_locks=angular_locks,
                    )
                elif hasattr(env, "create_d6_joint"):
                    handle = env.create_d6_joint(
                        pivot_asset, needle_asset,
                        p_pos_l, p_quat_l,
                        n_pos_l, n_quat_l,
                        linear_locks=linear_locks,
                        angular_locks=angular_locks,
                    )
        except Exception as exc:
            print(f"[joint_utils] Failed to create D6 joint for env {eid}: {exc}")
            handle = None

        if handle is not None:
            env._pivot_joint_handles[eid] = handle
            env._pivot_joint_active[eid] = True
            if hasattr(env, "logger"):
                env.logger.info(
                    f"[PivotJoint] Env {eid}: Created D6 joint at {anchor_w.cpu().numpy()}, "
                    f"mode={mode}, hinge_axis={hinge_axis}, locks=(lin={linear_locks}, ang={angular_locks})"
                )
            continue

        # Fallback: author a USD joint (Fixed or Revolute) directly.  This
        # branch will also be executed for hinge joints even if the scene
        # exposes an `add_d6_joint` API.
        pivot_path_tmpl = getattr(pivot_asset.cfg, "prim_path", "")
        needle_path_tmpl = getattr(needle_asset.cfg, "prim_path", "")

        pivot_path = _expand_env_path(pivot_path_tmpl, env, eid)
        needle_path = _expand_env_path(needle_path_tmpl, env, eid)

        if mode.upper() == "WELD":
            joint_path = _create_usd_weld_joint(env, eid, pivot_path, needle_path,
                                                p_pos_l, p_quat_l, n_pos_l, n_quat_l)
        else:
            joint_path = _create_usd_hinge_joint(env, eid, pivot_path, needle_path,
                                                 p_pos_l, p_quat_l, n_pos_l, n_quat_l,
                                                 axis="X")
        env._pivot_joint_handles[eid] = joint_path
        env._pivot_joint_active[eid] = True
        if hasattr(env, "logger"):
            env.logger.info(
                f"[PivotJoint] Env {eid}: Authored USD joint at {anchor_w.cpu().numpy()}, "
                f"mode={mode}, hinge_axis={hinge_axis}, locks=(lin={linear_locks}, ang={angular_locks})"
            )


def destroy_pivot_joint(env, env_ids):
    if isinstance(env_ids, int):
        env_ids = [env_ids]
    stage = env.scene.stage
    for eid in env_ids:
        handle = getattr(env, "_pivot_joint_handles", [None] * env.num_envs)[eid]
        if handle:
            if isinstance(handle, str):
                prim = stage.GetPrimAtPath(handle)
                if prim.IsValid():
                    stage.RemovePrim(prim.GetPath())
            else:
                # handle is a PhysX joint; destroy via scene API if available
                try:
                    if hasattr(env.scene, "remove_joint"):
                        env.scene.remove_joint(handle)
                except Exception:
                    pass
        if hasattr(env, "_pivot_joint_handles"):
            env._pivot_joint_handles[eid] = None
        if hasattr(env, "_pivot_joint_active"):
            env._pivot_joint_active[eid] = False


def setup_needle_pivot_joint(env, env_ids, mode: Optional[str] = None, hinge_axis: Optional[str] = None) -> None:
    """Wrapper called from the task's event system to create the pivot joint."""
    # Pull defaults from the environment configuration if not provided
    if mode is None and hasattr(env, "cfg") and hasattr(env.cfg, "pivot_joint"):
        mode = getattr(env.cfg.pivot_joint, "mode", "WELD")
    if hinge_axis is None and hasattr(env, "cfg") and hasattr(env.cfg, "pivot_joint"):
        hinge_axis = getattr(env.cfg.pivot_joint, "hinge_axis", "needle_x")
    # Get optional anchor offset from config
    anchor_offset_local = None
    if hasattr(env, "cfg") and hasattr(env.cfg, "pivot_joint"):
        anchor_offset_local = getattr(env.cfg.pivot_joint, "anchor_offset_local", None)
    # Create joint for the specified envs
    create_pivot_joint(
        env, env_ids, mode=mode or "WELD", hinge_axis=hinge_axis or "needle_x",
        anchor_offset_local=anchor_offset_local
    )


def teardown_needle_pivot_joint(env, env_ids) -> None:
    """Wrapper called from the task's event system to destroy the pivot joint."""
    destroy_pivot_joint(env, env_ids)


import re


def _env_ns(env) -> str:
    # Use scene's namespace if available, else fall back to the common default.
    return getattr(env.scene, "env_ns", "/World/envs/env")


def _expand_env_path(tmpl: str, env, eid: int) -> str:
    """
    Turn a template/regex prim path into a concrete per‑env path.  Handles both
    `{ENV_REGEX_NS}` and the baked `env_.*` regex variant.
    """
    per_env_ns = f"{_env_ns(env)}_{eid}"
    s = tmpl
    s = s.replace("{ENV_REGEX_NS}", per_env_ns)
    s = re.sub(r"env_\.\*", f"env_{eid}", s)
    return s


def _get_env_prim_path(asset, eid: int) -> str:
    """
    Resolve per‑env USD prim path for an IsaacLab asset.
    1) Prefer cfg.prim_path with {ENV_REGEX_NS}.
    2) Fallback to PhysX view internals if available.
    """
    if hasattr(asset, "cfg") and hasattr(asset.cfg, "prim_path"):
        tmpl = asset.cfg.prim_path
        if isinstance(tmpl, str) and tmpl:
            if "{ENV_REGEX_NS}" in tmpl:
                return tmpl.replace("{ENV_REGEX_NS}", _env_ns(eid))
            return tmpl
    # 2) From PhysX view (implementation‑dependent)
    view = getattr(asset, "root_physx_view", None) or getattr(asset, "_root_physx_view", None)
    if view is not None:
        try:
            return view.get_prim_path(eid)
        except Exception:
            pass
    return ""