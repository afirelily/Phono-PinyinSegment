import torch


class SegmentationMetrics:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.reset()

    def update(self, logits, labels, attention_mask):
        mask = attention_mask[:, :-1].bool() & attention_mask[:, 1:].bool()
        predictions = torch.sigmoid(logits) >= self.threshold
        correct = predictions.eq(labels.bool()) & mask
        self.correct += int(correct.sum().item())
        self.total += int(mask.sum().item())
        self.sentences += int(mask.shape[0])
        self.correct_sentences += int(((predictions.eq(labels.bool()) | ~mask).all(dim=1)).sum().item())

    def compute(self):
        return {
            "acc": self.correct / self.total if self.total else 0.0,
            "s_acc": self.correct_sentences / self.sentences if self.sentences else 0.0,
        }

    def reset(self):
        self.correct = self.total = self.correct_sentences = self.sentences = 0
