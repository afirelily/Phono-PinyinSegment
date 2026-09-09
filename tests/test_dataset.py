import json

import torch


def test_boundary_labels(tokenizer):
    from datasets_pipeline.dataset import transform_sample
    sample = transform_sample(json.dumps(["ni", "hao", "ma"]), tokenizer, {})
    assert tokenizer.decode(sample["input_ids"]) == "nihaoma"
    assert sample["labels"] == [0, 1, 0, 0, 1, 0]
    assert len(sample["labels"]) == sample["length"] - 1


def test_single_syllable_has_no_positive_boundary(tokenizer):
    from datasets_pipeline.dataset import transform_sample
    sample = transform_sample(["zhong"], tokenizer, {})
    assert sample["labels"] == [0, 0, 0, 0]


def test_collate_padding_and_gap_mask(tokenizer):
    from datasets_pipeline.dataset import make_collate_fn, transform_sample
    samples = [transform_sample(["ni", "hao"], tokenizer, {}), transform_sample(["a"], tokenizer, {})]
    batch = make_collate_fn(tokenizer.pad_token_id)(samples)
    assert batch["input_ids"].shape == (2, 5)
    assert batch["labels"].shape == (2, 4)
    assert batch["attention_mask"].tolist() == [
        [True, True, True, True, True], [True, False, False, False, False]
    ]
    assert torch.equal(batch["labels"][0], torch.tensor([0.0, 1.0, 0.0, 0.0]))


def test_only_supported_augmentation_keys_matter(tokenizer):
    from datasets_pipeline.dataset import transform_sample
    plain = transform_sample(["ni", "hao"], tokenizer, {"heteronym_confusion": 1.0, "no_context_prob": 1.0})
    assert tokenizer.decode(plain["input_ids"]) == "nihao"
