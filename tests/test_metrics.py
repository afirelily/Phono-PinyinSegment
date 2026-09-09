import torch


def test_accuracy_and_sentence_accuracy():
    from metrics import SegmentationMetrics
    metrics = SegmentationMetrics()
    logits = torch.tensor([[10.0, -10.0, 10.0], [-10.0, 10.0, 10.0]])
    labels = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    mask = torch.ones((2, 4), dtype=torch.bool)
    metrics.update(logits, labels, mask)
    result = metrics.compute()
    assert result["acc"] == 5 / 6
    assert result["s_acc"] == 1 / 2


def test_padding_is_excluded():
    from metrics import SegmentationMetrics
    metrics = SegmentationMetrics()
    metrics.update(
        torch.tensor([[10.0, 10.0, 10.0]]),
        torch.tensor([[1.0, 0.0, 0.0]]),
        torch.tensor([[1, 1, 0, 0]], dtype=torch.bool),
    )
    assert metrics.compute() == {"acc": 1.0, "s_acc": 1.0}
