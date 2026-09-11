import torch
import torch.nn.functional as F


def binary_segmentation_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    loss_type: str = "bce",
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
) -> torch.Tensor:
    """Masked BCE or binary focal loss over valid character gaps."""
    per_gap_bce = F.binary_cross_entropy_with_logits(
        logits.float(), labels.float(), reduction="none"
    )
    if loss_type == "bce":
        per_gap_loss = per_gap_bce
    elif loss_type == "focal":
        if not 0.0 <= focal_alpha <= 1.0:
            raise ValueError("focal_alpha must be in [0, 1]")
        if focal_gamma < 0.0:
            raise ValueError("focal_gamma must be non-negative")
        targets = labels.float()
        alpha_t = focal_alpha * targets + (1.0 - focal_alpha) * (1.0 - targets)
        probability_t = torch.exp(-per_gap_bce)
        per_gap_loss = alpha_t * (1.0 - probability_t).pow(focal_gamma) * per_gap_bce
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type!r}; expected 'bce' or 'focal'")

    weights = mask.to(per_gap_loss.dtype)
    return (per_gap_loss * weights).sum() / weights.sum().clamp_min(1)
