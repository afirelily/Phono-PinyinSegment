"""Reference legal-path decoder for the gap scorer demo.

The production implementation lives in phono-core. This small Python version
keeps the training-side demo honest: model logits score only Trie-legal paths
instead of being thresholded independently.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class _Path:
    score: float
    pieces: tuple[str, ...]


def _prefer(candidate: _Path, current: _Path | None) -> bool:
    if current is None:
        return True
    if candidate.score != current.score:
        return candidate.score > current.score
    if len(candidate.pieces) != len(current.pieces):
        return len(candidate.pieces) < len(current.pieces)
    return tuple(map(len, candidate.pieces)) > tuple(map(len, current.pieces))


def decode_legal_path(
    text: str, gap_logits: list[float], vocabulary: set[str]
) -> list[str] | None:
    """Return the maximum-score legal segmentation, or ``None`` if unreachable."""

    if not text:
        return None
    if len(gap_logits) != len(text) - 1:
        raise ValueError("gap_logits must contain len(text) - 1 values")

    best: list[_Path | None] = [None] * (len(text) + 1)
    best[0] = _Path(0.0, ())
    max_token_chars = max(map(len, vocabulary), default=0)
    for begin in range(len(text)):
        prefix = best[begin]
        if prefix is None:
            continue
        for end in range(begin + 1, min(len(text), begin + max_token_chars) + 1):
            token = text[begin:end]
            if token not in vocabulary:
                continue
            score = prefix.score + (gap_logits[end - 1] if end < len(text) else 0.0)
            candidate = _Path(score, prefix.pieces + (token,))
            if _prefer(candidate, best[end]):
                best[end] = candidate
    return list(best[-1].pieces) if best[-1] is not None else None


def load_pinyin_vocabulary(path: str) -> set[str]:
    with open(path, encoding="utf-8") as file:
        return {line.strip() for line in file if line.strip()}
