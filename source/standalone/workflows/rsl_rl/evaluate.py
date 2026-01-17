# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint of a CMO-PPO agent from RSL-RL."""

import argparse
import random
import numpy as np
import torch
from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a CMO-PPO agent with RSL-RL.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# -----------------------------------------------------------------------------
# Deterministic evaluation utilities
#
# We set global seeds before constructing the environment to ensure that the
# simulator, Python, NumPy and PyTorch all use the same random seed. This is
# critical for reproducible evaluation results and reduces run‑to‑run jitter.
#
def _set_global_seeds(seed: int | None) -> None:
    """Seed Python, NumPy and PyTorch for deterministic evaluation."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # When available, also seed CUDA; silently ignore on CPU‑only machines
    try:
        torch.cuda.manual_seed_all(seed)  # type: ignore[attr-defined]
    except Exception:
        pass

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import os
import torch

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import (
   RslRlOnPolicyRunnerCfg,
   RslRlVecEnvWrapper,
   export_policy_as_jit,
   export_policy_as_onnx,
)
import orbit.surgical.tasks  # noqa: F401
from isaaclab.managers import SceneEntityCfg


def main():
    """Play with a CMO-PPO agent."""
    # set seeds for deterministic behaviour
    _set_global_seeds(args_cli.seed)
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # turn off exploration during evaluation by zeroing noise and entropy
    try:
        agent_cfg.policy.init_noise_std = 0.0
    except Exception:
        pass
    try:
        agent_cfg.algorithm.entropy_coef = 0.0  # type: ignore[assignment]
    except Exception:
        pass

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)

    policy = runner.get_inference_policy(device=env.unwrapped.device)

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    # export_policy_as_jit(runner.alg.actor_critic, runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
    # export_policy_as_onnx(runner.alg.actor_critic, path=export_model_dir, filename="policy.onnx")

    obs, _ = env.get_observations()

    num_envs = env.unwrapped.num_envs
    device = env.unwrapped.device

    # metrics containers
    warmup_episodes = 3  # skip first 3 episodes per env before evaluation
    reach_success  = torch.zeros(num_envs, dtype=torch.bool, device=device)
    orient_success = torch.zeros(num_envs, dtype=torch.bool, device=device)
    running_mask   = torch.ones(num_envs, dtype=torch.bool, device=device)
    episode_counter_per_env = torch.zeros(num_envs, dtype=torch.long, device=device)

    # variables for measuring action jitter during eval
    prev_actions: torch.Tensor | None = None
    action_delta_sum = 0.0
    action_count = 0

    while simulation_app.is_running():
        with torch.inference_mode():
            # --- save "pre-step" metrics so you can evaluate terminal state even if wrapper auto-resets ---
            raw_env = env.unwrapped
            prev_phase = raw_env.mode_flags.clone()
            prev_reached = raw_env.check_reached.clone()
            prev_orient = raw_env.check_orient.clone()

            # recompute needle and goal positions before stepping
            needle = raw_env.scene["object"]
            needle_center_prev = needle.data.root_pos_w
            needle_quat_prev = needle.data.root_quat_w
            ee_frame = raw_env.scene["ee_1_frame"]
            N = raw_env.num_envs

            from orbit.surgical.tasks.surgical.correct_orientation.mdp.visualization import quat_to_rot_matrix, _contact_point_world

            contact_center_prev = _contact_point_world(needle_center_prev, needle_quat_prev, needle_center_prev.device)
            needle_rot_prev = quat_to_rot_matrix(needle_quat_prev)
            offset_local = torch.tensor([-0.025, 0.0, 0.0], device=needle_center_prev.device).expand(N, 3)
            offset_world = torch.bmm(needle_rot_prev, offset_local.unsqueeze(-1)).squeeze(-1)
            needle_center_prev = contact_center_prev + offset_world
            
            # per-env goal: compute goal location using environment origins for consistent spacing
            if hasattr(raw_env, "goal_point"):
                goal_point = raw_env.goal_point
            elif hasattr(raw_env.scene, "env_origins"):
                goal_point = raw_env.scene.env_origins + torch.tensor([-0.1863, 0.1419, 0.1296], device=device)
            else:
                # fallback to original spacing-based heuristic
                spacing = getattr(raw_env.scene, "env_spacing", 1.0)
                env_indices = torch.arange(N, device=device).unsqueeze(1)
                env_bases = torch.zeros((N, 3), device=device)
                env_bases[:, 0] = env_indices.squeeze(1) * spacing
                relative_offset = torch.tensor([-0.1863, 0.1419, 0.1296], device=device)
                goal_point = env_bases + relative_offset.unsqueeze(0)

            dist_to_goal_prev = torch.norm(needle_center_prev - goal_point, dim=1)

            # --- step the environment ---
            actions = policy(obs)

            # compute action jitter: accumulate absolute delta against previous actions
            if prev_actions is not None:
                delta = torch.abs(actions - prev_actions)
                action_delta_sum += float(delta.sum().item())
                action_count += delta.numel()
            prev_actions = actions.clone().detach()

            obs, _, dones, _ = env.step(actions)

            # Use the saved pre-step values for evaluation of those that just finished
            done_idx = torch.nonzero(dones, as_tuple=False).squeeze(-1)
            if done_idx.numel() > 0:
                # increment episode counters
                episode_counter_per_env[done_idx] += 1
                for env_id in done_idx:
                    if episode_counter_per_env[env_id] <= warmup_episodes:
                        continue

                    # --- SUCCESS DECISION (use pre-step flags, plus geometric fallback for reach) ---
                    reach  = (prev_reached[env_id] == 1) 
                    # or (dist_to_goal_prev[env_id] < 0.020)  # 2 cm reach fallback
                    orient = reach and (prev_orient[env_id] == 1) 
                                        # or (dist_to_goal_prev[env_id] < 0.004))  # 4 mm orient

                    reach_success[env_id]  = reach
                    orient_success[env_id] = orient
                    running_mask[env_id]   = False
                # stop when all envs have been evaluated once beyond warmup
                if not running_mask.any():
                    break

    reach_count = reach_success.sum().item()
    orient_count = orient_success.sum().item()

    print("\n=== EVALUATION SUMMARY ===")
    print(f"Reach Phase Success: {reach_count}/{num_envs} ({100*reach_count/num_envs:.1f}%)")
    print(f"Orient Success (<2mm): {orient_count}/{num_envs} ({100*orient_count/num_envs:.1f}%)")
    # compute a proxy for action jitter; avoid division by zero
    mean_action_delta = 0.0 if action_count == 0 else (action_delta_sum / action_count)
    print(f"Mean absolute action delta per step: {mean_action_delta:.4f}")
    print("==========================\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()