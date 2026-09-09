import pytest
import torch
from transformers import get_cosine_with_min_lr_schedule_with_warmup


def test_warmup_decay_uses_full_training_horizon():
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)
    scheduler = get_cosine_with_min_lr_schedule_with_warmup(
        optimizer,
        num_warmup_steps=2,
        num_training_steps=10,
        min_lr=1e-4,
    )

    learning_rates = []
    for _ in range(10):
        optimizer.step()
        scheduler.step()
        learning_rates.append(optimizer.param_groups[0]["lr"])

    assert learning_rates[1] == pytest.approx(1e-3)
    assert learning_rates[-1] == pytest.approx(1e-4)
    assert all(a >= b for a, b in zip(learning_rates[1:], learning_rates[2:]))
