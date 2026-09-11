from algo import decode_legal_path


def test_decoder_uses_logits_only_at_selected_boundaries():
    vocabulary = {"xi", "an", "xian"}
    assert decode_legal_path("xian", [0.0, -2.0, 0.0], vocabulary) == ["xian"]
    assert decode_legal_path("xian", [0.0, 2.0, 0.0], vocabulary) == ["xi", "an"]


def test_decoder_rejects_illegal_input_and_checks_shape():
    assert decode_legal_path("pv", [0.0], {"p"}) is None
    try:
        decode_legal_path("abc", [0.0], {"a", "bc"})
    except ValueError:
        pass
    else:
        raise AssertionError("misaligned logits must fail")


def test_decoder_tie_break_is_deterministic():
    vocabulary = {"a", "ab", "bc", "c"}
    assert decode_legal_path("abc", [0.0, 0.0], vocabulary) == ["ab", "c"]
