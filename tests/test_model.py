import torch


def test_output_contract_and_backward(tiny_model):
    input_ids = torch.randint(2, tiny_model.config.vocab_size, (3, 9))
    labels = torch.randint(0, 2, (3, 8)).float()
    mask = torch.ones_like(input_ids, dtype=torch.bool)
    output = tiny_model(input_ids=input_ids, attention_mask=mask, labels=labels)
    assert output.logits.shape == (3, 8)
    assert output.loss.ndim == 0 and torch.isfinite(output.loss)
    output.loss.backward()
    assert tiny_model.contraction.weight.grad is not None
    assert tiny_model.classifier.weight.grad is not None


def test_padded_gaps_do_not_affect_loss(tiny_model):
    ids = torch.randint(2, tiny_model.config.vocab_size, (1, 6))
    mask = torch.tensor([[1, 1, 1, 0, 0, 0]], dtype=torch.bool)
    labels1 = torch.tensor([[0, 1, 0, 0, 0]], dtype=torch.float32)
    labels2 = torch.tensor([[0, 1, 1, 1, 1]], dtype=torch.float32)
    loss1 = tiny_model(input_ids=ids, attention_mask=mask, labels=labels1).loss
    loss2 = tiny_model(input_ids=ids, attention_mask=mask, labels=labels2).loss
    assert torch.equal(loss1, loss2)


def test_extra_batch_padding_does_not_change_valid_logits(tiny_model):
    tiny_model.eval()
    short = torch.tensor([[2, 3, 4, 5]])
    padded = torch.tensor([[2, 3, 4, 5, 0, 0, 0]])
    short_mask = torch.ones_like(short, dtype=torch.bool)
    padded_mask = padded.ne(0)
    short_logits = tiny_model(input_ids=short, attention_mask=short_mask).logits
    padded_logits = tiny_model(input_ids=padded, attention_mask=padded_mask).logits[:, :3]
    assert torch.allclose(short_logits, padded_logits, atol=1e-6)


def test_depthwise_convolutions_use_channels_last(tiny_model):
    observed = []
    hooks = []
    for module in [layer.depthwise for layer in tiny_model.layers] + [tiny_model.contraction]:
        hooks.append(module.register_forward_pre_hook(
            lambda _module, args: observed.append(args[0].is_contiguous(memory_format=torch.channels_last))
        ))
    tiny_model(input_ids=torch.ones((2, 8), dtype=torch.long))
    for hook in hooks:
        hook.remove()
    assert all(observed)


def test_huggingface_roundtrip(tiny_model, tmp_path):
    tiny_model.save_pretrained(tmp_path)
    from model.model import PinyinSegmentModel
    loaded = PinyinSegmentModel.from_pretrained(tmp_path)
    ids = torch.randint(2, tiny_model.config.vocab_size, (1, 7))
    tiny_model.eval()
    loaded.eval()
    assert torch.allclose(tiny_model(input_ids=ids).logits, loaded(input_ids=ids).logits)


def test_gradient_checkpointing(tiny_model):
    tiny_model.gradient_checkpointing_enable()
    tiny_model.train()
    output = tiny_model(
        input_ids=torch.ones((2, 6), dtype=torch.long),
        attention_mask=torch.ones((2, 6), dtype=torch.bool),
        labels=torch.zeros((2, 5)),
    )
    output.loss.backward()
    assert tiny_model.gradient_checkpointing
