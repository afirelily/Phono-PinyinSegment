import argparse

import torch

from decoder import decode_legal_path, load_pinyin_vocabulary
from model.model import PinyinSegmentModel
from tokenizer import PinyinCharTokenizer


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pinyin", nargs="?", default="nihaoma")
    parser.add_argument(
        "--checkpoint", default="./checkpoints/v1_0-small-alpha02/final_model"
    )
    parser.add_argument("--vocab-config", default="./vocabs/config.yaml")
    parser.add_argument("--pinyin-vocab", default="../PhonoP2C/vocabs/pinyin_vocab.txt")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    tokenizer = PinyinCharTokenizer.from_config(args.vocab_config)
    vocabulary = load_pinyin_vocabulary(args.pinyin_vocab)
    model = PinyinSegmentModel.from_pretrained(args.checkpoint).to(device).eval()
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
