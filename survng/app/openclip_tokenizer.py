"""Small NumPy-only OpenCLIP BPE tokenizer.

Derived from OpenAI CLIP and OpenCLIP's MIT-licensed SimpleTokenizer. Keeping
this implementation local avoids importing PyTorch in the SurvNG server.
"""

from __future__ import annotations

import gzip
import html
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import ftfy
import numpy as np
import regex


@lru_cache(maxsize=1)
def bytes_to_unicode() -> dict[int, str]:
    values = list(range(ord("!"), ord("~") + 1))
    values += list(range(ord("¡"), ord("¬") + 1))
    values += list(range(ord("®"), ord("ÿ") + 1))
    characters = values[:]
    extra = 0
    for value in range(256):
        if value not in values:
            values.append(value)
            characters.append(256 + extra)
            extra += 1
    return dict(zip(values, (chr(value) for value in characters), strict=True))


def _pairs(word: tuple[str, ...]) -> set[tuple[str, str]]:
    return set(zip(word, word[1:]))


class OpenClipBpeTokenizer:
    CONTEXT_LENGTH = 77

    def __init__(self, vocab_path: Path, context_length: int = CONTEXT_LENGTH) -> None:
        self.context_length = int(context_length)
        if not 2 <= self.context_length <= 4096:
            raise ValueError("OpenCLIP tokenizer context length is invalid")
        try:
            merge_lines = gzip.open(vocab_path, "rt", encoding="utf-8").read().splitlines()
        except (OSError, UnicodeError) as exc:
            raise RuntimeError(f"OpenCLIP BPE vocabulary is unreadable: {vocab_path}") from exc
        merges = [tuple(line.split()) for line in merge_lines[1:49152 - 256 - 2 + 1]]
        if not merges or any(len(merge) != 2 for merge in merges):
            raise RuntimeError("OpenCLIP BPE vocabulary is invalid")
        byte_values = list(bytes_to_unicode().values())
        vocabulary = byte_values + [f"{value}</w>" for value in byte_values]
        vocabulary.extend("".join(merge) for merge in merges)
        vocabulary.extend(["<start_of_text>", "<end_of_text>"])
        self.encoder = dict(zip(vocabulary, range(len(vocabulary)), strict=True))
        if len(self.encoder) != 49408:
            raise RuntimeError("OpenCLIP BPE vocabulary has an unexpected size")
        self.byte_encoder = bytes_to_unicode()
        self.ranks = dict(zip(merges, range(len(merges)), strict=True))
        self.cache = {
            "<start_of_text>": "<start_of_text>",
            "<end_of_text>": "<end_of_text>",
        }
        self.start_token = self.encoder["<start_of_text>"]
        self.end_token = self.encoder["<end_of_text>"]
        self.pattern = regex.compile(
            r"<start_of_text>|<end_of_text>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+",
            regex.IGNORECASE,
        )

    def _bpe(self, token: str) -> str:
        cached = self.cache.get(token)
        if cached is not None:
            return cached
        word = tuple(token[:-1]) + (f"{token[-1]}</w>",)
        pairs = _pairs(word)
        if not pairs:
            return f"{token}</w>"
        while pairs:
            first, second = min(pairs, key=lambda pair: self.ranks.get(pair, float("inf")))
            if (first, second) not in self.ranks:
                break
            merged: list[str] = []
            index = 0
            while index < len(word):
                try:
                    next_index = word.index(first, index)
                except ValueError:
                    merged.extend(word[index:])
                    break
                merged.extend(word[index:next_index])
                index = next_index
                if index < len(word) - 1 and word[index + 1] == second:
                    merged.append(first + second)
                    index += 2
                else:
                    merged.append(word[index])
                    index += 1
            word = tuple(merged)
            if len(word) == 1:
                break
            pairs = _pairs(word)
        result = " ".join(word)
        self.cache[token] = result
        return result

    def encode(self, text: str) -> list[int]:
        cleaned = " ".join(html.unescape(html.unescape(ftfy.fix_text(text))).strip().split()).lower()
        encoded: list[int] = []
        for token in regex.findall(self.pattern, cleaned):
            byte_token = "".join(self.byte_encoder[value] for value in token.encode("utf-8"))
            encoded.extend(self.encoder[piece] for piece in self._bpe(byte_token).split(" "))
        return encoded

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        result = np.zeros((len(texts), self.context_length), dtype=np.int64)
        for row, text in enumerate(texts):
            tokens = [self.start_token, *self.encode(str(text)), self.end_token]
            if len(tokens) > self.context_length:
                tokens = tokens[: self.context_length]
                tokens[-1] = self.end_token
            result[row, : len(tokens)] = tokens
        return result
