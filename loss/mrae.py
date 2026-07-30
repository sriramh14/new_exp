import torch
import torch.nn as nn
import torch.nn.functional as F
class mrae(nn.module):
  """
    Mean Relative Absolute Error (MRAE)

    Computes:
        mean(|pred - target| / (target + eps))

    Args:
        eps (float): Small constant to avoid division by zero.
    """
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        relative_error = torch.abs(pred - target) / (torch.abs(target) + self.eps)
        return relative_error.mean()
