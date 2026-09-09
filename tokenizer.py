"""Character tokenizer for concatenated, tone-free pinyin."""

import json
import os


class PinyinCharTokenizer:
    def __init__(self, vocab_path: str):
        with open(vocab_path, "r", encoding="utf-8") as f:
            tokens = [line.strip() for line in f if line.strip()]
        if len(tokens) != len(set(tokens)):
            raise ValueError("vocab contains duplicate tokens")
        self.token_to_id = {token: idx for idx, token in enumerate(tokens)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        if "<pad>" not in self.token_to_id or "<unk>" not in self.token_to_id:
            raise ValueError("vocab must contain <pad> and <unk>")
        self.pad_token_id = self.token_to_id["<pad>"]
        self.unk_token_id = self.token_to_id["<unk>"]

    @classmethod
    def from_config(cls, config_path: str):
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        path = os.path.join(os.path.dirname(os.path.abspath(config_path)), cfg["vocab"])
        return cls(path)

    @property
    def vocab_size(self):
        return len(self.token_to_id)

    def encode(self, text: str):
        return [self.token_to_id.get(char, self.unk_token_id) for char in text.lower()]

    def decode(self, ids):
        return "".join(self.id_to_token[int(i)] for i in ids if int(i) != self.pad_token_id)

    def save_pretrained(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "vocab.json"), "w", encoding="utf-8") as f:
            json.dump(self.token_to_id, f, ensure_ascii=False, indent=2)
