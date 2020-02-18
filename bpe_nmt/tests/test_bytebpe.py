"""
    BPE-NMT Encoder

    Copyright (C) 2020 Miðeind ehf.

       This program is free software: you can redistribute it and/or modify
       it under the terms of the GNU General Public License as published by
       the Free Software Foundation, either version 3 of the License, or
       (at your option) any later version.
       This program is distributed in the hope that it will be useful,
       but WITHOUT ANY WARRANTY; without even the implied warranty of
       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
       GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see http://www.gnu.org/licenses/.
"""

import itertools

from bpe_nmt.bpe import ByteBPEEncoder, EOW, RESERVED_TOKENS, line_gen
from tensor2tensor.data_generators import tokenizer as t2t_tokenizer

try:
    from icecream import ic
    ic.configureOutput(includeContext=True)
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa


def test_bpe_encoder_build():
    corpus = (
        "þrá þráum þráð"
        " þrái þráð þrái þræði"
        " þráir þráðir þráir þræðir"
        " þráir þráði þrái þræði"
        " þráum þráðum þráum þræðum"
        " þráið þráðuð þráið þræðuð"
        " þrá þráðu þrái þræðu"
    )
    enc = ByteBPEEncoder.build_from_generator([corpus], 255 + 20, False, use_eow=True)
    ids = enc.encode(corpus)
    decoded = enc.decode(ids)
    assert corpus == decoded


def construct_bpe_encoder_build_large():
    max_line_count = 10_000
    max_vocab_size = 1_000
    separate_case = True

    corpus = line_gen("/home/haukur/Projects/bpe_nmt/eng-isl.tsv")
    corpus = itertools.islice(corpus, max_line_count)

    enc = ByteBPEEncoder.build_from_generator(
        corpus, max_vocab_size, separate_case, verbose=True
    )
    enc.store_to_file("tests/vocab_bbpe_large.json")


def test_large_greedy():
    # construct_bpe_encoder_build_large()
    enc = ByteBPEEncoder.load_from_file("tests/vocab_bbpe_large.json")
    text = "Upplýsingar sem eiga að koma fram í áliti Matvæla öryggisstofnunarinnar"
    greedy_ids = enc.encode(text)
    decoded = enc.decode(greedy_ids)
    print(text)
    print(decoded)
    assert text == decoded


def test_large_dropout():
    enc = ByteBPEEncoder.load_from_file("tests/vocab_bbpe_large.json")
    text = "Upplýsingar sem eiga að koma fram í áliti Matvæla öryggisstofnunarinnar"
    ids = enc.encode_with_dropout(text, 0.1)
    print(ids)
    print(enc.decode_list(ids))
    print(":".join(enc.decode_list(ids)))
    ic(enc.decode_list_human(ids))
    decoded = enc.decode(ids)
    print(decoded)
    assert text == decoded


def test_large_optimal():
    enc = ByteBPEEncoder.load_from_file("tests/vocab_bbpe_large.json")
    text = "Upplýsingar sem eiga að koma fram í áliti Matvæla öryggisstofnunarinnar"
    greedy_ids = enc.encode(text, greedy=True)
    opt_ids = enc.encode(text, greedy=False)
    decoded = enc.decode(opt_ids)
    print(opt_ids)
    print(enc.decode_list(opt_ids))
    print(enc.decode_list_human(opt_ids))
    print(decoded)
    assert len(opt_ids) <= len(greedy_ids)
    assert text == decoded


def test_large_optimal():
    enc = ByteBPEEncoder.load_from_file("tests/vocab_bbpe_large.json")
    text = """Upplýsingar sem eiga að koma fram í áliti Matvæla öryggisstofnunarinnar
Ef leiðbeiningunum fyrir Apidra er ekki fylgt getur það valdið alvarlegum aukaverkunum.
Ekki má nota nálar úr 1-mánaða paliperidon palmitat stungulyfi eða öðrum nálum sem fáanlegar eru á markaði við gjöf TREVICTA (sjá Upplýsingar ætlaðar heilbrigðisstarfsfólki).
Red frændi vildi ekki leyfa mér að koma með til Nashville...
Þetta ætti að stuðla að því að komist verði hjá ónauðsynlegri orkunotkun og tryggja þægileg loftskilyrði innandyra (hitaþægindi) í hlutfalli við hitastig utandyra."""
    import numpy as np
    for line in text.split("\n"):
        samples = []
        for i in range(10):
            samples.append(len(enc.encode_with_dropout(line, 0.15)))
        print(len(enc.encode(line, greedy=False)), len(enc.encode(line, greedy=True)), min(samples), max(samples))
        print(np.mean(samples), np.std(samples))
        print()
