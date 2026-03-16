import torch

def sample_uniform(
    low: float | torch.Tensor,
    high: float | torch.Tensor,
    shape: tuple[int, ...],
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Uniform sampler mirroring IsaacLab's ``sample_uniform`` helper."""
    low_t = torch.as_tensor(low, dtype=torch.float32, device=device)
    high_t = torch.as_tensor(high, dtype=torch.float32, device=device)
    return low_t + (high_t - low_t) * torch.rand(shape, device=device)