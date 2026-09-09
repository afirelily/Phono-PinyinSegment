import pathlib
import sys

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture
def tokenizer():
    from tokenizer import PinyinCharTokenizer
    return PinyinCharTokenizer.from_config(str(ROOT / "vocabs" / "config.yaml"))


@pytest.fixture
def tiny_model(tokenizer):
    from model.config import PinyinSegmentConfig
    from model.model import PinyinSegmentModel
    config = PinyinSegmentConfig(
        vocab_size=tokenizer.vocab_size, pad_token_id=tokenizer.pad_token_id,
        model_dim=16, num_layers=2, kernel_size=3, ffn_dim=32,
    )
    return PinyinSegmentModel(config)
