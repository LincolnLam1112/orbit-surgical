import torch


def get_mode_flags(env):
    """
    Initialize or return existing mode flags.
    Now uses numerical stage flags instead of named strings.
    """
    if not hasattr(env, "mode_flags"):
        N = env.num_envs
        device = env.device
        env.mode_flags = torch.zeros(N, dtype=torch.long, device=device)  # stage 0 by default
    return env.mode_flags
