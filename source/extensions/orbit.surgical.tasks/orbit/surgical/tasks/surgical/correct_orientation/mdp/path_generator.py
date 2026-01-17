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


class ArcPathGenerator:
    """
    Generate points on a circular arc from start_pos to end_pos around a given center.
    All tensors are (N, 3). Returns (N, num_steps, 3).
    """
    def __init__(self, num_steps: int = 5, shortest: bool = True):
        self.num_steps = num_steps
        self.shortest = shortest

    def generate(self, start_pos: torch.Tensor, end_pos: torch.Tensor, center_pos: torch.Tensor) -> torch.Tensor:
        N = start_pos.shape[0]
        device = start_pos.device

        # vectors from center
        a = start_pos - center_pos     # (N,3)
        b = end_pos   - center_pos     # (N,3)
        ra = torch.norm(a, dim=1, keepdim=True).clamp_min(1e-9)
        rb = torch.norm(b, dim=1, keepdim=True).clamp_min(1e-9)

        # if radii mismatch (numerics), normalize to average radius
        r = 0.5 * (ra + rb)
        a_hat = a / ra
        b_hat = b / rb

        # plane basis
        n = torch.cross(a_hat, b_hat, dim=1)
        n_norm = torch.norm(n, dim=1, keepdim=True).clamp_min(1e-9)
        n_hat = n / n_norm

        # build orthonormal basis (u = a_hat, v = n̂ × u)
        u = a_hat
        v = torch.cross(n_hat, u, dim=1)

        # angles
        cos_t = (a_hat * b_hat).sum(1).clamp(-1.0, 1.0)
        sin_t = (b_hat * v).sum(1)     # signed, since v ⟂ u in plane
        theta = torch.atan2(sin_t, cos_t)  # (N,)

        if self.shortest:
            # wrap to (-pi, pi]
            theta = (theta + torch.pi) % (2 * torch.pi) - torch.pi

        # interpolation
        t = torch.linspace(0, 1, self.num_steps, device=device).view(1, self.num_steps, 1)  # (1,S,1)
        theta_t = theta.view(N, 1, 1) * t                                                   # (N,S,1)
        cos_tt = torch.cos(theta_t)
        sin_tt = torch.sin(theta_t)                            # (N,S,1)

        arc_pts = center_pos.view(N, 1, 3) + r.view(N, 1, 1) * (u.view(N,1,3)*cos_tt + v.view(N,1,3)*sin_tt)
        return arc_pts