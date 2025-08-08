# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to collect demonstrations with Isaac Lab environments."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Collect demonstrations for Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--teleop_device", type=str, default="keyboard", help="Device for interacting with environment")
parser.add_argument("--num_demos", type=int, default=3, help="Number of episodes to store in the dataset.")
parser.add_argument("--filename", type=str, default="hdf_dataset", help="Basename of output file.")
parser.add_argument("--num_success_steps", type=int, default = 3, help = "Number of continuous steps with task success for concluding a demo as successful. Default is 10.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import time

import contextlib
import gymnasium as gym
import os
import torch

from isaaclab.devices import Se3Keyboard, Se3SpaceMouse

import isaaclab_mimic.envs  # noqa: F401
from isaaclab_mimic.ui.instruction_display import InstructionDisplay, show_subtask_instructions

from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.io import dump_pickle, dump_yaml

import isaaclab_tasks  # noqa: F401
#from isaaclab_tasks.utils.data_collector import RobomimicDataCollector
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from isaaclab.devices.openxr.retargeters.manipulator import GripperRetargeter, Se3AbsRetargeter, Se3RelRetargeter
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.envs.ui import EmptyWindow
from isaaclab.managers import DatasetExportMode, ObservationTermCfg, ActionTermCfg

import orbit.surgical.tasks  # noqa: F401
from orbit.surgical.tasks.surgical.lift import mdp
from orbit.surgical.tasks.surgical.lift.lift_env_cfg import LiftEnvCfg


def pre_process_actions(delta_pose: torch.Tensor, gripper_command: bool) -> torch.Tensor:
    """Pre-process actions for the environment."""
    # compute actions based on environment
    if "Reach" in args_cli.task:
        # note: reach is the only one that uses a different action space
        # compute actions
        return delta_pose
    else:
        # resolve gripper command
        gripper_vel = torch.zeros((delta_pose.shape[0], 1), dtype=torch.float, device=delta_pose.device)
        gripper_vel[:] = -1 if gripper_command else 1
        # compute actions
        return torch.concat([delta_pose, gripper_vel], dim=1)


def main():
    """Collect demonstrations from the environment using teleop interfaces."""
    assert (
        args_cli.task == "Isaac-Lift-Needle-PSM-IK-Rel-v0"
    ), "Only 'Isaac-Lift-Needle-PSM-IK-Rel-v0' is supported currently."
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)

    # modify configuration such that the environment runs indefinitely
    # until goal is reached
    env_cfg.terminations.time_out = None
    # set the resampling time range to large number to avoid resampling
    env_cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)
    # we want to have the terms in the observations returned as a dictionary
    # rather than a concatenated tensor
    env_cfg.observations.policy.concatenate_terms = False

    # add termination condition for reaching the goal otherwise the environment won't reset
    env_cfg.terminations.object_reached_goal = None

    # specify directory for logging experiments
    log_dir = os.path.join("./datasets", args_cli.task)
    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)

    # create data-collector
    # ─── 1. Before creating `env`, configure the recorder ──────────────────────

    # What we want to record
    env_cfg = LiftEnvCfg()

    filename = args_cli.filename
    for cfg in env_cfg.recorders.values():
        cfg.dataset_export_dir_path = log_dir
        cfg.dataset_filename = filename


    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    print("Recorder terms:")
    env.recorder_manager.print_active_terms()

    # filename = args_cli.filename

    # env_cfg.recorders = ActionStateRecorderManagerCfg()
    # env_cfg.recorders.dataset_export_dir_path = log_dir
    # env_cfg.recorders.dataset_filename = filename
    # # Only export episodes marked as successful
    # env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL

        # create controller
    if args_cli.teleop_device.lower() == "keyboard":
        teleop_interface = Se3Keyboard(pos_sensitivity=0.005, rot_sensitivity=0.05)
    elif args_cli.teleop_device.lower() == "spacemouse":
        teleop_interface = Se3SpaceMouse(pos_sensitivity=0.05, rot_sensitivity=0.005)
    else:
        raise ValueError(f"Invalid device interface '{args_cli.teleop_device}'. Supported: 'keyboard', 'spacemouse'.")
    # add teleoperation key for env reset
    teleop_interface.add_callback("L", env.reset)
    # print helper
    print(teleop_interface)

    # ─── 2. Initialize/reset before entering the loop ───────────────────────────
    obs_dict, _ = env.reset()
    teleop_interface.reset()
    
    num_demos_exported = 0
    success_step_count = 0
    # How many consecutive “success” steps are required to declare an episode successful
    SUCCESS_STEPS_REQUIRED = 3 if args_cli.num_success_steps is None else args_cli.num_success_steps


    # ─── 3. Main teleop & record loop ─────────────────────────────────────────
    with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
        while True:
            # 3a. Get teleop data → pre‐process into a PyTorch action tensor
            delta_pose, gripper_command = teleop_interface.advance()
            delta_pose = torch.tensor(delta_pose, dtype=torch.float32, device=env.device).repeat(env.num_envs, 1)
            actions = pre_process_actions(delta_pose, gripper_command)

            # 3b. Step the environment
            #print(env.scene["object"].data.root_pos_w[:,2])
            obs_dict, rewards, terminated, truncated, info = env.step(actions)
            dones = terminated | truncated

            # 3c. Check success‐term (if defined) to accumulate “success steps”
            success_term = DoneTerm(func=mdp.object_reached_goal)
            
            if success_term is not None:
                # 'success_term.func(env, **success_term.params)` returns a tensor of shape [B] (B = num_envs)
                success_flags = success_term.func(env)  # bool tensor

                if bool(success_flags):
                    success_step_count += 1
                    #print(success_step_count)

                # 3d. If we have enough consecutive success steps, export the episode
                #if success_step_count >= SUCCESS_STEPS_REQUIRED:
                    # 1) Mark it for recorder to keep
                    #env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
                    # 2) Flag this episode as “successful”
                    #env.recorder_manager.set_success_to_episodes(
                        #[0], torch.tensor([[True]], dtype=torch.bool, device=env.device)
                    #)
                    # 3) Actually export the HDF5 data for that episode
                success_result = success_term.func(env, **success_term.params)
                if success_result.any():
                    print("Recorder manager:", env.recorder_manager)
                    env.recorder_manager.record_pre_reset([0], force_export_or_skip=True)
                    env.recorder_manager.set_success_to_episodes([0], torch.tensor([[True]], dtype=torch.bool, device=env.device))
                    success = env.recorder_manager.export_episodes([0])
                    print("Export result:", success)

                    num_demos_exported += 1
                    print(f"[Demo] Exported episode #{num_demos_exported} as successful.")

                    # 3e. If we’ve reached the desired number of demos, break out
                    if args_cli.num_demos > 0 and num_demos_exported >= args_cli.num_demos:
                        print(f"[Demo] Recorded all {num_demos_exported} demos. Exiting.")
                        break

                    # 3f. Otherwise, force‐reset for the next demo
                    env.sim.reset()
                    env.recorder_manager.reset()
                    obs_dict, _ = env.reset()
                    teleop_interface.reset()
                    success_step_count = 0
                    continue


            # 3g. If the env ended for any other reason (timeout, truncated), manually reset:
            if dones.any():
                env.sim.reset()
                env.recorder_manager.reset()
                obs_dict, _ = env.reset()
                teleop_interface.reset()
                success_step_count = 0
                continue

            # 3h. Otherwise, keep looping – the recorder is automatically collecting obs/actions

            # 3i. Optionally, break if the simulator is stopped
            if env.sim.is_stopped():
                print("[Demo] Simulator stopped. Exiting.")
                break

    # ─── 4. Clean up ─────────────────────────────────────────────────────────────
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
