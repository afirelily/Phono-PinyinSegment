from transformers import PretrainedConfig


class PinyinSegmentConfig(PretrainedConfig):
    model_type = "phono_pinyin_segment"

    def __init__(
        self,
        vocab_size: int = 28,
        model_dim: int = 128,
        num_layers: int = 8,
        kernel_size: int = 3,
        ffn_dim: int = 256,
        pad_token_id: int = 0,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.ffn_dim = ffn_dim


def build_config_from_dict(d: dict, vocab_size: int, pad_token_id: int):
    return PinyinSegmentConfig(
        vocab_size=vocab_size,
        pad_token_id=pad_token_id,
        model_dim=d.get("model_dim", 128),
        num_layers=d.get("num_layers", 8),
        kernel_size=d.get("kernel_size", 3),
        ffn_dim=d.get("ffn_dim", 256),
    )
