def test_character_roundtrip(tokenizer):
    ids = tokenizer.encode("NiHaoMa")
    assert len(ids) == 7
    assert tokenizer.decode(ids) == "nihaoma"


def test_unknown_character(tokenizer):
    assert tokenizer.encode("ü") == [tokenizer.unk_token_id]
