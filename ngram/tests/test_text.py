from ngramlm.text import clean, normalize, tokenize


def toks(raw: str):
    return tokenize(normalize(raw))


def test_elision_is_split_after_apostrophe():
    assert toks("Je t'en laisse la décision.") == ["je", "t'", "en", "laisse", "la", "décision"]
    assert toks("L'enfant bégaye.") == ["l'", "enfant", "bégaye"]


def test_hyphenated_words_stay_whole():
    assert toks("Pourquoi m'avez-vous tiré dessus ?") == ["pourquoi", "m'", "avez-vous", "tiré", "dessus"]
    assert toks("Qu'est-ce que c'est ?") == ["qu'", "est-ce", "que", "c'", "est"]
    assert toks("A-t-il peut-être raison ?") == ["a-t-il", "peut-être", "raison"]


def test_curly_and_straight_apostrophes_are_equivalent():
    assert toks("J’en suis sûre maintenant.") == toks("J'en suis sûre maintenant.")
    assert toks("C’est là votre point fort.") == ["c'", "est", "là", "votre", "point", "fort"]


def test_aujourdhui_is_one_token():
    assert toks("Aujourd'hui, il pleut.") == ["aujourd'hui", "il", "pleut"]


def test_numbers_are_mapped_to_num():
    assert toks("Molière est né en 1622.") == ["molière", "est", "né", "en", "<num>"]
    assert toks("Ça coûte 3,50 euros.") == ["ça", "coûte", "<num>", "euros"]


def test_ligatures_and_accents_are_kept():
    assert toks("J'aime bien les œufs.") == ["j'", "aime", "bien", "les", "œufs"]


def test_invisible_and_special_spaces_do_not_change_tokens():
    plain = toks("Es-tu chez toi ?")
    assert toks("Es-tu chez toi ?") == plain
    assert toks("Es-tu​ chez toi ?") == plain


def test_punctuation_is_dropped():
    assert toks("« Bonjour », dit-il… « Ça va ? »") == ["bonjour", "dit-il", "ça", "va"]


def test_clean_rejects_bad_sentences():
    assert clean("Prout !") is None  # one word
    assert clean("Привет, мир") is None  # non-Latin script
    assert clean("mot " * 41) is None  # too long
    assert clean("!!! ???") is None  # no tokens


def test_clean_accepts_normal_sentence():
    assert clean("Je ne supporte pas ce type.") == ["je", "ne", "supporte", "pas", "ce", "type"]
