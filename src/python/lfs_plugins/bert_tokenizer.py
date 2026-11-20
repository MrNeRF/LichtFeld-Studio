# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""BERT uncased WordPiece for LLMDet; no inference framework or network access.

Follows Hugging Face's Apache-2.0 BasicTokenizer/WordpieceTokenizer rules.
"""
import gzip
import hashlib
import unicodedata

from _lfs_path_io import Path


def _chinese(code):
    return any(lo <= code <= hi for lo, hi in (
        (0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x20000, 0x2A6DF),
        (0x2A700, 0x2B73F), (0x2B740, 0x2B81F), (0x2B820, 0x2CEAF),
        (0xF900, 0xFAFF), (0x2F800, 0x2FA1F)))


def _basic_tokens(text):
    cleaned = []
    for char in text:
        category = unicodedata.category(char)
        if char in " \t\r\n" or category == "Zs":
            cleaned.append(" ")
        elif ord(char) not in (0, 0xFFFD) and category[0] != "C":
            cleaned.append(" " + char + " " if _chinese(ord(char)) else char)
    for word in unicodedata.normalize("NFC", "".join(cleaned)).split():
        word = "".join(char for char in unicodedata.normalize("NFD", word.lower())
                       if unicodedata.category(char) != "Mn")
        current = []
        for char in word:
            code = ord(char)
            punctuation = (33 <= code <= 47 or 58 <= code <= 64 or
                           91 <= code <= 96 or 123 <= code <= 126 or
                           unicodedata.category(char)[0] == "P")
            if punctuation:
                if current:
                    yield "".join(current)
                    current.clear()
                yield char
            else:
                current.append(char)
        if current:
            yield "".join(current)


class BertTokenizer:
    def __init__(self, vocabulary):
        self._vocab = {word: index for index, word in enumerate(vocabulary)}
        if (len(self._vocab) != 30522 or
                any(self._vocab.get(word) != index for word, index in
                    (("[PAD]", 0), ("[UNK]", 100), ("[CLS]", 101), ("[SEP]", 102), ("[MASK]", 103)))):
            raise ValueError("Invalid BERT uncased vocabulary")

    @classmethod
    def load(cls):
        path = Path(__file__).parent / "assets" / "bert" / "vocab.txt.gz"
        with path.open("rb") as file, gzip.GzipFile(fileobj=file) as stream:
            raw = stream.read(231509)
        if len(raw) != 231508 or hashlib.sha256(raw).hexdigest() != VOCAB_SHA256:
            raise ValueError("BERT vocabulary resource is corrupt")
        return cls(raw.decode("utf-8").splitlines())

    def _wordpiece(self, word):
        if len(word) > 100:
            return [100]
        ids, begin = [], 0
        while begin < len(word):
            end = len(word)
            while end > begin:
                piece = ("##" if begin else "") + word[begin:end]
                if piece in self._vocab:
                    break
                end -= 1
            if begin == end:
                return [100]
            ids.append(self._vocab[piece])
            begin = end
        return ids

    def encode(self, text):
        if not isinstance(text, str):
            raise TypeError("BERT text must be a string")
        if len(text) > 1024 or len(text.encode("utf-8")) > 4096:
            raise ValueError("BERT text exceeds the input limit")
        if any(token in text for token in ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]")):
            raise ValueError("BERT special tokens are not allowed in descriptions")
        ids = [101]
        for word in _basic_tokens(text):
            ids.extend(self._wordpiece(word))
            if len(ids) > 254:
                raise ValueError("LLMDet text exceeds the 256-token context")
        if len(ids) == 1:
            raise ValueError("LLMDet description is empty")
        # LLMDet separates category spans with full stops. The service joins
        # these complete, untruncated spans into one bounded BERT context.
        if ids[-1] != 1012:
            ids.append(1012)
        return tuple(ids + [102])


VOCAB_SHA256 = "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3"
