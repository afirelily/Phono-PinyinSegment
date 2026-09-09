# Gap scorer and legal-path MAP decoding

Phono-PinyinSegment is a scorer, not a standalone greedy segmenter. For an
input of `n` normalized letters, it emits one logit `z_k` for each of the
`n - 1` character gaps. A legal segmentation path `Y` assigns every gap a
binary state `y_k`: one means a syllable boundary and zero means continuation.

With `p_k = sigmoid(z_k)`, the Bernoulli log probability of a path is

```text
log P(Y | X) = sum_k [y_k log p_k + (1-y_k) log(1-p_k)].
```

Using `log p_k - log(1-p_k) = z_k`, this becomes

```text
log P(Y | X) = sum_k log(1-p_k) + sum_k y_k z_k.
```

The first sum depends on the input and model output but not on the candidate
path. Therefore, for MAP decoding only,

```text
argmax_Y log P(Y | X) = argmax_Y sum_{k: y_k=1} z_k.
```

This is why the runtime may add raw logits only at selected boundaries. It is
an exact argmax simplification, not a calibrated probability for the whole
path. Every path assigns the same `n - 1` aligned gaps, so full-pinyin and
jianpin paths need neither geometric-mean normalization nor a syllable-count
penalty.

The pinyin vocabulary Trie enumerates all legal edges `(i, j)` in a character
DAG. An edge contributes `z_(j-1)` when `j < n`, and zero when it reaches the
end. Dynamic programming stores the best prefix path at each character offset;
the complexity is `O(E)`, where `E` is the number of Trie matches. Exact-score
ties prefer fewer syllables and then longer earlier syllables, making results
deterministic without changing the MAP result in the ordinary case.

The production decoder, normalization, strict/safe invalid-input handling and
per-character repair live in phono-core. This repository's `demo.py` contains
a compact reference legal-path decoder for validating scorer checkpoints.
