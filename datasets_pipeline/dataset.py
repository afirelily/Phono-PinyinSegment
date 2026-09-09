"""Adapters over PhonoP2C's existing flat, uniquely disambiguated pinyin data."""

import json

import torch
from datasets import load_from_disk
from streaming import StreamingDataset

from datasets_pipeline.pinyin import augment_pinyin_sequence


def transform_sample(pinyin_value, tokenizer, augmentation=None, rng=None):
    syllables = json.loads(pinyin_value) if isinstance(pinyin_value, str) else list(pinyin_value)
    syllables = augment_pinyin_sequence(syllables, augmentation or {}, rng=rng)
    text = "".join(syllables)
    input_ids = tokenizer.encode(text)

    # Label gap i when it follows the final character of a non-final syllable.
    labels = [0.0] * max(0, len(input_ids) - 1)
    offset = 0
    for syllable in syllables[:-1]:
        offset += len(syllable)
        if 0 < offset <= len(labels):
            labels[offset - 1] = 1.0
    return {"input_ids": input_ids, "labels": labels, "length": len(input_ids)}


def transform_batch(batch, tokenizer, augmentation=None):
    samples = [
        transform_sample(value, tokenizer, augmentation)
        for value in batch["pinyin_list"]
    ]
    return {key: [sample[key] for sample in samples] for key in samples[0]}


def make_collate_fn(pad_token_id: int):
    def collate(samples):
        batch_size = len(samples)
        max_length = max(2, max(sample["length"] for sample in samples))
        input_ids = torch.full((batch_size, max_length), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
        labels = torch.zeros((batch_size, max_length - 1), dtype=torch.float32)
        for row, sample in enumerate(samples):
            length = sample["length"]
            if length:
                input_ids[row, :length] = torch.tensor(sample["input_ids"])
                attention_mask[row, :length] = True
            if length > 1:
                labels[row, :length - 1] = torch.tensor(sample["labels"])
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    return collate


def create_dataset(path: str, keep_in_memory=False):
    return load_from_disk(path, keep_in_memory=keep_in_memory)


class PinyinSegmentStreamingDataset(StreamingDataset):
    def __init__(self, local, tokenizer, augmentation, **kwargs):
        super().__init__(local=local, **kwargs)
        self.tokenizer = tokenizer
        self.augmentation = augmentation

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        return transform_sample(sample["pinyin_list"], self.tokenizer, self.augmentation)
