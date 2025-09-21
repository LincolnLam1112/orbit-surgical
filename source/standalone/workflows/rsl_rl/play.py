# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

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

def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    #export_policy_as_jit(
        #ppo_runner.alg.actor_critic, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
    #)
    #export_policy_as_onnx(ppo_runner.alg.actor_critic, path=export_model_dir, filename="policy.onnx")

    # # reset environment
    # obs, _ = env.get_observations()

    # # --- smoothing state (persistent across steps) ---
    # _sm_prev = None        # low-pass internal state
    # _sm_last = None        # last output after slew-limit
    # _sm_repeat = None      # per-env repeat counter

    # # knobs (tweak these to feel the effect)
    # ALPHA = 1.0           # closer to 0 = heavier smoothing
    # REPEAT = 1             # 1 = no repeat; 2+ = hold output for k-1 steps
    # SLEW  = 0.15           # per-dim max delta per step; 0 disables

    # step_i = 0

    # # simulate environment
    # while simulation_app.is_running():
    #     with torch.inference_mode():
    #         # agent stepping (policy is already deterministic in get_inference_policy)
    #         actions = policy(obs)  # shape: [num_envs, act_dim], on the right device

    #         # --- PRE-STEP SMOOTHING (runs every control step) ---
    #         if _sm_prev is None:
    #             # first step init
    #             _sm_prev   = actions.clone()
    #             _sm_last   = actions.clone()
    #             _sm_repeat = torch.zeros(actions.shape[0], dtype=torch.long, device=actions.device)

    #         # choose which envs recompute a new command this step
    #         recompute = (_sm_repeat == 0)
    #         if recompute.any():
    #             idx = torch.nonzero(recompute, as_tuple=False).squeeze(-1)
    #             prev = _sm_prev[idx]

    #             # low-pass: ā_t = α a_t + (1-α) ā_{t-1}
    #             new_smooth = ALPHA * actions[idx] + (1.0 - ALPHA) * prev

    #             # slew-rate limit (per-dim clamp on change from last output)
    #             if SLEW > 0.0:
    #                 delta = (new_smooth - _sm_last[idx]).clamp(-SLEW, SLEW)
    #                 out = _sm_last[idx] + delta
    #             else:
    #                 out = new_smooth

    #             # update internal states
    #             _sm_prev[idx] = new_smooth
    #             _sm_last[idx] = out

    #         # action repeat bookkeeping
    #         hold = ~recompute
    #         _sm_repeat[recompute] = REPEAT - 1
    #         _sm_repeat[hold] = torch.clamp(_sm_repeat[hold] - 1, min=0)

    #         smoothed_actions = _sm_last

    #         # (optional) light debug to confirm it’s running every step
    #         if step_i % 200 == 0:
    #             i = 0  # first env
    #             # print("[smooth] raw[0]:", actions[i].detach().cpu().numpy(), " -> smoothed[0]:", smoothed_actions[i].detach().cpu().numpy())

    #         # env stepping WITH SMOOTHED ACTIONS
    #         obs, _, _, _ = env.step(smoothed_actions)

    #         step_i += 1

    # reset environment
    obs, _ = env.get_observations()
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
            # print(actions)
            # print(obs)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()