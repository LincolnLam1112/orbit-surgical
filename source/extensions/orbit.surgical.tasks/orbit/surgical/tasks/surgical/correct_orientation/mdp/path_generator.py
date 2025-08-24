import torch


class LinearPathGenerator:
    def __init__(self, num_steps: int = 10):
        self.num_steps = num_steps

    def generate(self, start_pos: torch.Tensor, end_pos: torch.Tensor) -> torch.Tensor:
        """
        Linearly interpolate a path of positions.
        start_pos: (N, 3)
        end_pos: (N, 3)
        Returns: (N, num_steps, 3)
        """
        N = start_pos.shape[0]
        alpha = torch.linspace(0, 1, self.num_steps, device=start_pos.device).view(1, self.num_steps, 1)
        start = start_pos.view(N, 1, 3)
        end = end_pos.view(N, 1, 3)
        return start + alpha * (end - start)
