"""Lightweight pinyin boundary detector using BHWC depthwise convolutions."""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from model.config import PinyinSegmentConfig
from loss import binary_segmentation_loss


@dataclass
class PinyinSegmentOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None


class SwiGLU(nn.Module):
    """Pointwise Up-Act-Gate-Down projection, applied on BHWC tensors."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.up_proj(x)) * self.gate_proj(x))


class DSCResidualBlock(nn.Module):
    """Pre-norm depthwise convolution followed by a pointwise SwiGLU."""

    def __init__(self, dim: int, kernel_size: int, ffn_dim: int):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("Residual DSC kernel_size must be odd")
        self.norm1 = nn.LayerNorm(dim)
        self.depthwise = nn.Conv2d(
            dim, dim, kernel_size=(1, kernel_size),
            padding=(0, kernel_size // 2), groups=dim, bias=False,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.pointwise = SwiGLU(dim, ffn_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is logically BHWC. Conv2d consumes BCHW but is explicitly run in
        # channels-last memory format so training and export use NHWC kernels.
        residual = x
        hidden = self.norm1(x).permute(0, 3, 1, 2).contiguous(
            memory_format=torch.channels_last
        )
        hidden = self.depthwise(hidden)
        hidden = hidden.permute(0, 2, 3, 1)
        x = residual + hidden
        return x + self.pointwise(self.norm2(x))


class PinyinSegmentModel(PreTrainedModel):
    config_class = PinyinSegmentConfig
    base_model_prefix = "phono_pinyin_segment"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DSCResidualBlock"]

    def __init__(self, config: PinyinSegmentConfig):
        super().__init__(config)
        self.embed = nn.Embedding(
            config.vocab_size, config.model_dim, padding_idx=config.pad_token_id
        )
        self.layers = nn.ModuleList([
            DSCResidualBlock(config.model_dim, config.kernel_size, config.ffn_dim)
            for _ in range(config.num_layers)
        ])
        self.final_norm = nn.LayerNorm(config.model_dim)
        # A valid width-2 contraction creates exactly one feature per gap.
        self.contraction = nn.Conv2d(
            config.model_dim, config.model_dim, kernel_size=(1, 2), padding=0,
            groups=config.model_dim, bias=False,
        )
        # Bias-free keeps the binary head friendly to XNNPACK dynamic
        # quantization, matching the other pointwise projections.
        self.classifier = nn.Linear(config.model_dim, 1, bias=False)
        self.gradient_checkpointing = False
        self.post_init()

    def _run_layer(self, layer, hidden):
        if self.gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                layer, hidden, use_reentrant=False
            )
        return layer(hidden)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        loss_type: str = "bce",
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        return_dict: Optional[bool] = None,
    ):
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.shape[1] < 2:
            raise ValueError("input sequences must be padded to at least length 2")

        use_return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        hidden = self.embed(input_ids).unsqueeze(1)  # BHWC: [B, 1, L, C]
        hidden_mask = None
        if attention_mask is not None:
            hidden_mask = attention_mask[:, None, :, None].to(hidden.dtype)
            hidden = hidden * hidden_mask
        for layer in self.layers:
            hidden = self._run_layer(layer, hidden)
            # Prevent padded positions from becoming a recurrent halo that
            # feeds back into real tokens in later convolutional layers.
            if hidden_mask is not None:
                hidden = hidden * hidden_mask
        hidden = self.final_norm(hidden)
        hidden = hidden.permute(0, 3, 1, 2).contiguous(
            memory_format=torch.channels_last
        )
        hidden = self.contraction(hidden).permute(0, 2, 3, 1)
        logits = self.classifier(hidden).squeeze(1).squeeze(-1)  # [B, L-1]

        loss = None
        if labels is not None:
            if attention_mask is None:
                gap_mask = torch.ones_like(logits, dtype=torch.bool)
            else:
                gap_mask = attention_mask[:, :-1].bool() & attention_mask[:, 1:].bool()
            loss = binary_segmentation_loss(
                logits, labels, gap_mask, loss_type, focal_alpha, focal_gamma
            )

        if not use_return_dict:
            return ((loss, logits) if loss is not None else (logits,))
        return PinyinSegmentOutput(loss=loss, logits=logits)
