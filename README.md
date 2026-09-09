# Phono-PinyinSegment

Phono-PinyinSegment is the lightweight pinyin gap scorer paired with PhonoP2C.
It consumes concatenated, tone-free pinyin letters and emits one boundary
logit for every adjacent character pair. Production decoding combines these
scores with a legal pinyin Trie DAG instead of thresholding gaps greedily.

For example, `nihaoma` has seven characters and six output positions. The
target boundaries recover `ni hao ma`.

## Architecture

The Hugging Face `PinyinSegmentModel` uses an embedding followed by pre-norm
residual depthwise-separable convolution blocks. Depthwise convolutions run in
channels-last memory format; the pointwise path is a Linear
Up-SiLU-Gate-Down SwiGLU. A final valid width-2 depthwise contraction maps a
length-L input to L-1 gap features before the binary classifier.

## Data

No preprocessing code is included. Training reads the existing PhonoP2C MDS
dataset and validation reads `val_original`. Their flat `pinyin_list` field is
already the uniquely disambiguated pronunciation sequence. Online augmentation
supports only `drop_vowels`, `drop_last_vowel`, and `vowels_droprate`.

## Commands

```bash
pixi run test
pixi run train
pixi run demo -- nihaoma --checkpoint ./checkpoints/v1_0-small-alpha02/final_model
pixi run python export.py --checkpoint ./checkpoints/v1_0-small/final_model --quantization w8a8
```

Hydra overrides work as in PhonoP2C, for example
`pixi run train system.optim_8bit=false model=tiny`.
Use `task.loss_type=focal` to compare focal loss against the default BCE;
`task.focal_loss_alpha` and `task.focal_loss_gamma` control its class weighting
and focusing strength.

Checkpoints contain only Hugging Face model configuration/weights plus the
resolved training configuration. Optimizer and scheduler state are deliberately
not stored. Validation reports BCE loss, per-gap accuracy (`ACC`), and exact
whole-sequence accuracy (`S-ACC`).

The exported mobile graph has a static batch size of one and a dynamic input
length from 3 through 512. phono-core routes shorter input directly to its
checked FMM path and does not invoke or pad the scorer.

See [Gap scorer and legal-path MAP decoding](docs/scorer-decoding.md) for the
exact logit-sum derivation and Trie-DAG dynamic program.
