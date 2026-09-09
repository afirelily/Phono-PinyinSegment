import pytest
import torch
import torch.nn.functional as F

from loss import binary_segmentation_loss


def test_bce_matches_pytorch_on_valid_gaps():
    logits = torch.tensor([[0.0, 1.0, -1.0]])
    labels = torch.tensor([[0.0, 1.0, 1.0]])
    mask = torch.tensor([[True, True, False]])
    actual = binary_segmentation_loss(logits, labels, mask, loss_type="bce")
    expected = F.binary_cross_entropy_with_logits(logits[:, :2], labels[:, :2])
    assert torch.allclose(actual, expected)


def test_focal_gamma_zero_is_alpha_weighted_bce():
    logits = torch.tensor([[0.0, 0.0]])
    labels = torch.tensor([[1.0, 0.0]])
    mask = torch.ones_like(labels, dtype=torch.bool)
    actual = binary_segmentation_loss(
        logits, labels, mask, loss_type="focal", focal_alpha=0.25, focal_gamma=0.0
    )
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    expected = (bce * torch.tensor([[0.25, 0.75]])).mean()
    assert torch.allclose(actual, expected)


def test_focal_downweights_easy_examples():
    logits = torch.tensor([[8.0, -8.0, 0.0]])
    labels = torch.tensor([[1.0, 0.0, 1.0]])
    mask = torch.ones_like(labels, dtype=torch.bool)
    focal = binary_segmentation_loss(
        logits, labels, mask, loss_type="focal", focal_alpha=0.5, focal_gamma=2.0
    )
    bce = binary_segmentation_loss(logits, labels, mask, loss_type="bce")
    assert 0.0 < focal < bce


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"loss_type": "unknown"}, "Unsupported loss_type"),
        ({"loss_type": "focal", "focal_alpha": 1.1}, "focal_alpha"),
        ({"loss_type": "focal", "focal_gamma": -1.0}, "focal_gamma"),
    ],
)
def test_invalid_loss_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        binary_segmentation_loss(
            torch.zeros((1, 1)), torch.zeros((1, 1)),
            torch.ones((1, 1), dtype=torch.bool), **kwargs,
        )
