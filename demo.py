import argparse

import torch

from model.model import PinyinSegmentModel
from tokenizer import PinyinCharTokenizer


@torch.no_grad()
def segment_pinyin(text, model, tokenizer, device, threshold=0.5):
    compact = "".join(text.lower().split())
    if len(compact) < 2:
        return [compact], []
    ids = torch.tensor([tokenizer.encode(compact)], dtype=torch.long, device=device)
    logits = model(input_ids=ids).logits[0]
    probabilities = torch.sigmoid(logits).cpu().tolist()
    pieces, start = [], 0
    for index, probability in enumerate(probabilities):
        if probability >= threshold:
            pieces.append(compact[start:index + 1])
            start = index + 1
    pieces.append(compact[start:])
    return pieces, probabilities


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pinyin", nargs="?", default="nihaoma")
    parser.add_argument("--checkpoint", default="./checkpoints/v1_0-small/final_model")
    parser.add_argument("--vocab-config", default="./vocabs/config.yaml")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    tokenizer = PinyinCharTokenizer.from_config(args.vocab_config)
    model = PinyinSegmentModel.from_pretrained(args.checkpoint).to(device).eval()
    model.to(memory_format=torch.channels_last)
    pieces, probabilities = segment_pinyin(
        args.pinyin, model, tokenizer, device, args.threshold
    )
    print("Input:       ", "".join(args.pinyin.split()))
    print("Segmentation:", " ".join(pieces))
    print("Gap probs:   ", " ".join(f"{p:.8f}" for p in probabilities))


if __name__ == "__main__":
    main()
