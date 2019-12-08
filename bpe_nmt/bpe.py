import itertools
import re, collections, string
import os
import json
import random
from pprint import pprint

from tensor2tensor.data_generators import tokenizer as t2t_tokenizer
from tensor2tensor.data_generators.text_encoder import (
    native_to_unicode,
    _escape_token,
    _unescape_token,
    unicode_to_native,
    strip_ids,
)


PAD = "<pad>"
EOS = "<EOS>"
RESERVED_TOKENS = [PAD, EOS]
NUM_RESERVED_TOKENS = len(RESERVED_TOKENS)
PAD_ID = RESERVED_TOKENS.index(PAD)  # Normally 0
EOS_ID = RESERVED_TOKENS.index(EOS)  # Normally 1


PUA_OFFSET = 0x100000  # Unicode Private Usage Area
SYMBOL_DELIMITER = chr(PUA_OFFSET + 1)
SYMBOL_DELIMITER = "_"
UP_ARROW = chr(0x2191)  # ↑
END_OF_TEXT = chr(0x2403)
END_OF_TRANSMISSION = chr(0x2404)
END_OF_TRANSMISSION_BLOCK = chr(0x2417)
EOW = END_OF_TEXT

PUNCT = re.compile("[" + re.escape(string.punctuation) + "]")
WHITESPACE = re.compile(r"\s+")
MIN_COUNT = 5


def encode_capitalization(token):
    uppers = set(c for c in token if c.isupper())
    for c in uppers:
        token = token.replace(c, UP_ARROW + c.lower())
    return token


def decode_capitalization(token):
    if UP_ARROW in token:
        idxs = [idx for (idx, char) in enumerate(token) if char == UP_ARROW]
        chars = list(token)
        for idx in reversed(idxs):
            if idx + 1 < len(token):
                chars[idx : idx + 2] = chars[idx + 1].upper()
        token = "".join(chars)
    return token


def add_meta_symbols(token):
    token = encode_capitalization(token)
    token = token + EOW
    return token


def remove_meta_symbols(token):
    token = decode_capitalization(token)
    token = token.replace(EOW, "")
    return token


class BPEEncoder:
    def __init__(
        self,
        alphabet,
        merge_table,
        symbols,
        separate_case=True,
        num_reserved_ids=NUM_RESERVED_TOKENS,
        omit_eow=False,
    ):
        self._separate_case = separate_case
        self.omit_eow = omit_eow
        self._num_reserved_ids = num_reserved_ids
        self.alphabet = alphabet
        self.merge_table = merge_table
        self.symbols = symbols
        self.all_symbols = RESERVED_TOKENS + symbols
        self._subtoken_string_to_id = {
            sym: idx for (idx, sym) in enumerate(self.all_symbols)
        }
        self._max_subtoken_len = max(len(sym) for sym in self.symbols)

    def maybe_add_meta_symbols(self, token, eow=EOW):
        if self.separate_case:
            token = encode_capitalization(token)
        if not self.omit_eow:
            token = token + eow
        return token

    def encode(self, text, greedy=True):
        # TODO: implement dropout
        tokens = [
            self.maybe_add_meta_symbols(tok) for tok in t2t_tokenizer.encode(text)
        ]
        return self._encode_tokens(tokens, greedy=greedy)

    def _encode_tokens(self, tokens, greedy=True):
        encode_token = lambda x: self._encode_token_greedy(x)
        if not greedy:
            encode_token = lambda x: self._encode_token_optimal(x)
        ret = []
        for token in tokens:
            ret.extend(encode_token(token))
        return ret

    def _encode_token_greedy(self, escaped_token):
        ret = []
        start = 0
        token_len = len(escaped_token)
        while start < token_len:
            for end in range(min(token_len, start + self._max_subtoken_len), start, -1):
                subtoken = escaped_token[start:end]
                if subtoken in self._subtoken_string_to_id:
                    ret.append(subtoken)
                    start = end
                    break

            else:  # Did not break
                # If there is no possible encoding of the escaped token then one of the
                # characters in the token is not in the alphabet. This should be
                # impossible and would be indicative of a bug.
                assert False, "Token substring not found in subtoken vocabulary."
        return [self._subtoken_string_to_id[substr] for substr in ret]

    def _encode_token_optimal(self, escaped_token, verbose=False):
        """
        nicest implementation uses a trie (prefix tree)

        graph[i] points to the nearest node which is on the shortest path from graph[i] to graph[0]
        via the BPE vocabulary

        graph[i] contains the longest substring in escaped_token starting at i that is in the BPE
        vocabulary.

        vocabulary:
          a b c d e ab bc de fg bcde
        tokenized text:
          abcde
        greedy:
          ab c d e
        optimal:
          a bcde

        graph:
          default
            [a, b, c, d, e]
          longest
            [a, bcdefg, c, d, e]

        shortest path:
          default  (path_length, token_pos, symbol)
            [(0; 0 ɛ),
             (1; 0 a),
             (2; 1 b),
             (3; 2 c),
             (4; 3 d),
             (5; 4 e)]
          optimal
            [(0; 0 ɛ),
             (1; 0 a),
             (2; 1 b),
             (2; 1 bc),
             (4; 3 d),
             (2; 1 bcde)]
        """
        _print = (
            (lambda *ar, **kw: print(*ar, **kw))
            if verbose
            else (lambda *ar, **kw: None)
        )
        _pprint = (
            (lambda *ar, **kw: print(*ar, **kw))
            if verbose
            else (lambda *ar, **kw: None)
        )
        Path = collections.namedtuple(
            "Path", ["length", "start", "symbol"]  # start in path, not token
        )
        if not escaped_token:
            return []
        start = 0
        end = 1
        path = [Path(0, 0, "")] + [
            Path(pos + 1, pos, escaped_token[pos]) for pos in range(len(escaped_token))
        ]

        _pprint(path)
        max_len = len(escaped_token)
        _print("Entering path loop")
        while start < max_len:
            substring = escaped_token[start:end]
            if substring in self._subtoken_string_to_id:
                end_pos = start + len(substring)
                if path[start].length < path[end_pos].length:
                    prev = path[end_pos]
                    path[end_pos] = Path(path[start].length + 1, start, substring)
                    _print(prev, " ", path[end_pos])
            end += 1
            if (end - start) > self._max_subtoken_len:
                start += 1
                end = start + 1
        _print("Exited path loop")
        _pprint(path)
        subtokens = []
        pointer = path[-1]
        _print("Constructing shortest path")
        last = pointer
        while pointer.symbol:
            last = pointer.start
            _print(pointer.start)
            subtokens.append(pointer.symbol)
            pointer = path[pointer.start]
            if last == pointer.start:
                _print("exiting")
                break
        _print("constructed shortest path")
        subtokens = list(reversed(subtokens))
        ids = [self._subtoken_string_to_id[idx] for idx in subtokens]
        _print(subtokens)
        _print(ids)
        return ids

    def encode_with_dropout(self, text, dropout):
        tokens = [
            self.maybe_add_meta_symbols(tok) for tok in t2t_tokenizer.encode(text)
        ]
        ret = []
        for token in tokens:
            ret.extend(self._encode_token_with_dropout(token, dropout))
        return ret

    def _encode_token_with_dropout(self, token, dropout):
        atoms = list(token)
        table = {pair: idx for (idx, pair) in enumerate(self.merge_table)}
        while True:
            best_idx = len(table)
            for i in range(len(atoms) - 1):
                pair = (atoms[i], atoms[i + 1])
                if pair in table:
                    if random.random() < dropout:
                        continue
                    best_idx = min(table[pair], best_idx)
            if best_idx >= len(table):
                break

            pair = self.merge_table[best_idx]
            merged_pair = "".join(pair)
            idxs = [
                i for i in range(len(atoms) - 1) if (atoms[i], atoms[i + 1]) == pair
            ]
            for idx in reversed(idxs):
                atoms[idx : idx + 2] = [merged_pair]
        ret = [self._subtoken_string_to_id[idx] for idx in atoms]
        return ret

    def decode(self, ids, strip_extraneous=False):
        if strip_extraneous:
            ids = strip_ids(ids, list(range(self._num_reserved_ids or 0)))
        substrings = [remove_meta_symbols(self.all_symbols[idx]) for idx in ids]
        return t2t_tokenizer.decode([token for token in substrings if token])

    def decode_list(self, ids):
        pass

    @property
    def vocab_size(self):
        return len(self.all_symbols)

    @property
    def separate_case(self):
        return self._separate_case

    @classmethod
    def build_from_generator(
        cls, generator, max_size, separate_case=True, verbose=False, omit_eow=False
    ):
        token_counts = collections.defaultdict(int)
        for line in generator:
            for token in t2t_tokenizer.encode(line):
                token_counts[token] += 1
        return cls.build_from_token_counts(
            token_counts,
            max_size,
            separate_case=separate_case,
            verbose=verbose,
            omit_eow=omit_eow,
        )

    @classmethod
    def build_from_token_counts(
        cls, token_counts, max_size, separate_case=True, verbose=False, omit_eow=False
    ):
        token_counts_with_eow = collections.defaultdict(int)
        maybe_append_eow = (lambda s: s) if omit_eow else (lambda s: s + EOW)
        maybe_encode_capitalization = (
            (lambda s: s) if not separate_case else encode_capitalization
        )
        for (token, count) in token_counts.items():
            token = maybe_encode_capitalization(token)
            token = maybe_append_eow(token)
            token_counts_with_eow[token] = count
        alphabet, merge_table, symbols = build_from_token_counts(
            token_counts_with_eow, max_size, verbose=verbose
        )
        return cls(
            alphabet, merge_table, symbols, separate_case=True, omit_eow=omit_eow
        )

    def store_to_file(self, path):
        with open(path, "w", encoding="utf-8") as file_handle:
            obj = dict(
                alphabet=self.alphabet,
                merge_table=self.merge_table,
                symbols=self.symbols,
                separate_case=self.separate_case,
            )
            json.dump(obj, file_handle, ensure_ascii=False, indent=2)

    @classmethod
    def load_from_file(cls, path):
        with open(path, "r", encoding="utf-8") as file_handle:
            obj = json.load(file_handle)
            merge_table = [tuple(item) for item in obj["merge_table"]]
            enc = cls(
                obj["alphabet"], merge_table, obj["symbols"], obj["separate_case"]
            )
            return enc


def merge_pair_in_vocab(vocab_old, pair, symbol_delimiter=" "):
    vocab_new = {}
    bigram = re.escape(symbol_delimiter.join(pair))
    pattern = re.compile(
        r"(?<![^"
        + re.escape(symbol_delimiter)
        + r"])"
        + bigram
        + r"(?![^"
        + re.escape(symbol_delimiter)
        + r"])"
    )
    new_symbol = "".join(pair)
    for token in vocab_old:
        new_encoding = pattern.sub(new_symbol, token)
        vocab_new[new_encoding] = vocab_old[token]
    return vocab_new


def compute_pair_counts(vocab, symbol_delimiter=" "):
    pair_counts = collections.defaultdict(int)
    for token, count in vocab.items():
        symbols = token.split(symbol_delimiter)
        for i in range(len(symbols) - 1):
            symbol_pair = (symbols[i], symbols[i + 1])
            pair_counts[symbol_pair] += count
    return pair_counts


def build_from_token_counts(
    token_counts, max_size, verbose=False, symbol_delimiter=SYMBOL_DELIMITER
):
    """Description
    Args:
      token_counts: dict of word to count
    Returns:
      ?
    """
    _print = (
        (lambda *ar, **kw: print(*ar, **kw)) if verbose else (lambda *ar, **kw: None)
    )
    alphabet = set()
    atomized_vocab = dict()
    for (token, count) in token_counts.items():
        alphabet.update(token)
        atomized_token = symbol_delimiter.join(list(token))
        atomized_vocab[atomized_token] = count
    del token_counts

    alphabet = sorted(list(alphabet))
    symbols = list(alphabet)
    num_merges = int(max_size) - len(symbols)
    if num_merges < 0:
        raise ValueError("Invalid vocab size, must be at least enough for alphabet")

    merge_table = []
    for iter_idx in range(num_merges):
        _print("Iteration {:>5d}".format(iter_idx))
        pair_counts = compute_pair_counts(
            atomized_vocab, symbol_delimiter=SYMBOL_DELIMITER
        )
        if not pair_counts:
            _print("symbol pairs exhausted")
            _print(atomized_vocab)
            _print(symbols)
            break
        best_pair = max(pair_counts, key=pair_counts.get)
        _print(
            " " * 4,
            "new_symbol ",
            "'{}'".format("".join(best_pair)),
            " from ",
            best_pair,
            sep="",
        )
        atomized_vocab = merge_pair_in_vocab(
            atomized_vocab, best_pair, symbol_delimiter=SYMBOL_DELIMITER
        )
        symbols.append("".join(best_pair))
        merge_table.append(best_pair)
        if len(symbols) >= max_size:
            _print("reached max size:", max_size)
            _print(symbols)
            break
    return (alphabet, merge_table, symbols)


def line_gen(path):
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if "\t" in line:
                yield from line.split("\t")
                continue
            yield line


def roundrobin(*iterables):
    "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
    # Recipe credited to George Sakkis
    pending = len(iterables)
    nexts = itertools.cycle(iter(it).next for it in iterables)
    while pending:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            pending -= 1
            nexts = itertools.cycle(itertools.islice(nexts, pending))
