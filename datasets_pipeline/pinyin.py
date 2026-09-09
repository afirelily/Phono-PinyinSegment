import random


_COMPOUND_INITIALS = {"zh", "ch", "sh"}
_SINGLE_INITIALS = set("bpmfdtnlgkhjqxrywzcs")


def split_initial_final(syllable: str):
    if syllable[:2] in _COMPOUND_INITIALS:
        return syllable[:2], syllable[2:]
    if syllable[:1] in _SINGLE_INITIALS:
        return syllable[:1], syllable[1:]
    return "", syllable


def augment_pinyin_sequence(pinyin_list, cfg, rng=None):
    """Apply only the three segmentation-relevant abbreviation controls."""
    rng = rng or random
    result = list(pinyin_list)
    if rng.random() < cfg.get("drop_vowels", 0.0):
        low, high = cfg.get("vowels_droprate", [0.0, 1.0])
        for index, syllable in enumerate(result):
            initial, final = split_initial_final(syllable)
            if initial and final and rng.random() < rng.uniform(low, high):
                result[index] = initial[0]
    if result and rng.random() < cfg.get("drop_last_vowel", 0.0):
        initial, final = split_initial_final(result[-1])
        if initial and final:
            result[-1] = initial[0]
    return result
