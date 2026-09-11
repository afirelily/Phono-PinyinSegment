"""Command-line demo for pinyin segmentation."""

import argparse
from pathlib import Path

import torch

from algo import decode_legal_path, load_pinyin_vocabulary
from model.model import PinyinSegmentModel
from tokenizer import PinyinCharTokenizer


PROJECT_DIR = Path(__file__).resolve().parent
DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


@torch.no_grad()
def segment_pinyin(text, model, tokenizer, device, vocabulary):
    compact = "".join(text.lower().split())
    if not compact:
        return None, [], []
    if len(compact) < 3:
        logits = [0.0] * (len(compact) - 1)
    else:
        ids = torch.tensor([tokenizer.encode(compact)], dtype=torch.long, device=device)
        logits = model(input_ids=ids).logits[0].cpu().tolist()
    pieces = decode_legal_path(compact, logits, vocabulary)
    probabilities = torch.sigmoid(torch.tensor(logits)).tolist()
    return pieces, logits, probabilities


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pinyin", help="concatenated, tone-free pinyin to segment")
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="model checkpoint directory"
    )
    parser.add_argument(
        "--vocab-config", type=Path, default=PROJECT_DIR / "vocabs" / "config.yaml",
        help="character tokenizer configuration",
    )
    parser.add_argument(
        "--pinyin-vocab", type=Path, required=True,
        help="newline-delimited legal pinyin syllables",
    )
    parser.add_argument("--device", default="auto", help="cpu, cuda, or cuda:N")
    parser.add_argument("--dtype", choices=("auto", *DTYPES), default="auto")
    return parser


def resolve_runtime(device_name, dtype_name):
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available()
        else "cpu" if device_name == "auto" else device_name
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    dtype = (
        torch.bfloat16 if dtype_name == "auto" and device.type == "cuda"
        else torch.float32 if dtype_name == "auto" else DTYPES[dtype_name]
    )
    return device, dtype


def main(argv=None):
    args = build_parser().parse_args(argv)

    device, dtype = resolve_runtime(args.device, args.dtype)
    tokenizer = PinyinCharTokenizer.from_config(str(args.vocab_config.expanduser()))
    vocabulary = load_pinyin_vocabulary(args.pinyin_vocab)
    model = PinyinSegmentModel.from_pretrained(args.checkpoint.expanduser()).to(
        device=device, dtype=dtype
    ).eval()
    model.to(memory_format=torch.channels_last)
    pieces, logits, probabilities = segment_pinyin(
        args.pinyin, model, tokenizer, device, vocabulary
    )
    print("Input:       ", "".join(args.pinyin.split()))
    print("Segmentation:", " ".join(pieces) if pieces else "<invalid pinyin>")
    print("Gap logits:  ", " ".join(f"{value:.8f}" for value in logits))
    print("Gap probs:   ", " ".join(f"{p:.8f}" for p in probabilities))


if __name__ == "__main__":
    main()
