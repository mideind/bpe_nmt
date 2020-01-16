import itertools

from bpe_nmt.bpe import BPEEncoder, EOW, RESERVED_TOKENS, line_gen, roundrobin
from tensor2tensor.data_generators import tokenizer as t2t_tokenizer


def test_build_from_frequencies():
    """
    {a,b,c,d}

    2*'a b'  'a b c'  'b c d'  'c d'
    3*(a,b) 2*(b,c) (c,d) -> ab
    [a,b,c,d,ab]

    2*'ab'  'ab c'  'b c d'  'c d'
    (ab,c) (b,c) 2*(c,d) -> cd
    [a,b,c,d,ab,cd]
    """

    token_counts = dict(ab=2, abc=1, bcd=1, cd=1)
    alphabet = ["a", "b", "c", "d"]
    merges = [("a", "b"), ("c", "d")]

    enc = BPEEncoder.build_from_token_counts(token_counts, 6, use_eow=False)
    assert enc.alphabet == alphabet
    assert enc.merge_table == merges


def test_bpe_encoder_serialization():
    token_counts = dict(ab=2, abc=1, bcd=1, cd=1)
    max_vocab_size = 15
    separate_case = False
    enc1 = BPEEncoder.build_from_token_counts(
        token_counts, max_vocab_size, separate_case
    )
    enc1.store_to_file("vocab_test.json")
    enc2 = BPEEncoder.load_from_file("vocab_test.json")
    assert enc1.alphabet == enc2.alphabet
    assert enc1.merge_table == enc2.merge_table
    assert enc1.symbols == enc2.symbols
    assert enc1.separate_case == enc2.separate_case


def test_bpe_encoder_build():
    """
    ab_ ab_ abc_ bcd_ cd_
    {a,b,c,d}

    2*'a b _'  'a b c _'  'b c d _'  'c d _'
    """
    token_counts = dict(ab=2, abc=1, bcd=1, cd=1)
    enc = BPEEncoder.build_from_token_counts(token_counts, 5, separate_case=False)
    assert enc.symbols == ["a", "b", "c", "d", EOW]
    enc = BPEEncoder.build_from_token_counts(token_counts, 7, separate_case=False)
    assert enc.symbols == ["a", "b", "c", "d", EOW, "ab", "ab" + EOW]
    assert enc.all_symbols == RESERVED_TOKENS + [
        "a",
        "b",
        "c",
        "d",
        EOW,
        "ab",
        "ab" + EOW,
    ]


def test_bpe_encoder_encode_greedy():
    token_counts = dict(ab=2, abc=1, bcd=1, cd=1)
    enc = BPEEncoder.build_from_token_counts(token_counts, 15, False)
    text = "ab abc bcd cd"
    assert enc.decode(enc.encode(text)) == text


def test_bpe_encoder_eow_token():
    token_counts = dict(ab=2, abc=1, bcd=1, cd=1)
    enc = BPEEncoder.build_from_token_counts(token_counts, 15, False)
    text = "b"
    assert enc.decode(enc.encode(text)) == text


def test_bpe_encoder_optimal():
    """
    vocabulary:
      a b c d e ab bc de bcde
    tokenized text:
      abcde
    greedy encoding:
      ab c de
    optimal encoding:
      a bcde
    """
    corpus = "ab ab ab ab ab ab bc de bcde bcde"
    text = "abcde"
    enc = BPEEncoder.build_from_generator([corpus], 20, False, use_eow=False)

    greedy_ids = enc.encode(text)
    greedy_symbols = [enc.decode([token_id]) for token_id in greedy_ids]
    assert greedy_symbols == ["ab", "c", "de"]

    optimal_ids = enc.encode(text, greedy=False)
    optimal_symbols = [enc.decode([token_id]) for token_id in optimal_ids]
    assert optimal_symbols == ["a", "bcde"]


def test_dropout():
    corpus = "abcdefgh abcdefgh abcdefgh abcdefgh abcdefgh"
    text = "abcdefgh"
    enc = BPEEncoder.build_from_generator([corpus], 30, False, use_eow=False)
    encodings = set()
    for _ in range(50):
        dropout_ids = enc.encode_with_dropout(text, dropout=0.2)
        encodings.add(tuple(dropout_ids))
    assert len(encodings) > 1


def debug_bpe_encoder_build_large():
    max_line_count = 10_000
    max_vocab_size = 1_000
    separate_case = True

    corpus = line_gen("/home/haukur/Projects/bpe_nmt/eng-isl.tsv")
    corpus = itertools.islice(corpus, max_line_count)

    enc = BPEEncoder.build_from_generator(
        corpus, max_vocab_size, separate_case, verbose=True
    )
    enc.store_to_file("tests/vocab_large.json")


def debug_large_01():
    enc = BPEEncoder.load_from_file("tests/vocab_large.json")
    text = "Upplýsingar sem eiga að koma fram í áliti Matvæla öryggisstofnunarinnar"
    greedy_ids = enc.encode(text)
    print([enc.all_symbols[token_id] for token_id in greedy_ids])
    """Greedy encoding:
      ['↑', 'upplýsingar␃', 'sem␃', 'eig', 'a␃', 'að␃', 'koma␃', 'fram␃', 'í␃', 'ál',
       'it', 'i␃', '↑ma', 't', 'v', 'æ', 'la␃', 'ör', 'y', 'gg', 'i', 'ss', 'to', 'f',
       'n', 'un', 'ar', 'inn', 'ar␃']
    """
    optimal_ids = enc.encode(text, greedy=False)
    print([enc.all_symbols[token_id] for token_id in optimal_ids])


def debug_bpe_large_02():
    enc = BPEEncoder.load_from_file("tests/vocab_large.json")
    max_line_count = 100_000

    corpus = line_gen("/home/haukur/Projects/bpe_nmt/eng-isl.tsv")
    corpus = itertools.islice(corpus, max_line_count)

    total_matches = 0
    for line in corpus:
        greedy_ids = enc.encode(line)
        optimal_ids = enc.encode(line, greedy=False)
        total_matches += int(greedy_ids == optimal_ids)
        if greedy_ids != optimal_ids:
            print(line)
            toks = t2t_tokenizer.encode(line)
            toks = [
                tok for tok in toks if enc.encode(tok) != enc.encode(tok, greedy=False)
            ]
            toks = " ".join(toks)
            print([enc.all_symbols[token_id] for token_id in enc.encode(toks)])
            print(
                [
                    enc.all_symbols[token_id]
                    for token_id in enc.encode(toks, greedy=False)
                ]
            )
            print()

    print(total_matches / max_line_count)
