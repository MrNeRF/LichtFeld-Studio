#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Small numerical BERT oracle from the official LLMDet checkpoint, without Torch.

Uses independent NumPy float64 equations for the inference-only BERT-base tower.
The model architecture follows Transformers v4.55.0 modeling_bert.py (Apache-2.0).
This is a development reference, not a second application inference backend.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from nn_export.lfw import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--vocab", type=Path, required=True)
    args = parser.parse_args()
    with args.checkpoint.open("rb") as stream:
        digest = sha256_file(stream)
        if digest != "f0e476be3d65c7c50e13029b0adc4bb5d4e05f3eb6d7bb04856578095e6ad92f":
            raise ValueError("Expected the pinned official LLMDet Tiny checkpoint")
        stream.seek(0)
        length = int.from_bytes(stream.read(8), "little")
        header = json.loads(stream.read(length))
    mapped = np.memmap(args.checkpoint, mode="r", dtype="u1")

    def weight(name):
        record = header["model.text_backbone." + name]
        return np.ndarray(record["shape"], dtype="<f4", buffer=mapped,
                          offset=8 + length + record["data_offsets"][0]).astype(np.float64)

    def norm(x, prefix):
        return (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-12) * weight(prefix + ".weight") + weight(prefix + ".bias")

    def linear(x, prefix):
        return x @ weight(prefix + ".weight").T + weight(prefix + ".bias")

    from tokenizers import BertWordPieceTokenizer
    vocabulary = args.vocab.read_bytes()
    if hashlib.sha256(vocabulary).hexdigest() != "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3":
        raise ValueError("Expected the official BERT uncased vocabulary")
    tokenizer = BertWordPieceTokenizer(str(args.vocab), lowercase=True)
    descriptions = ["person", "people", "PERSON", "a red chair", "città", "citta\u0300",
                    "chairs, tables & lamps", "yellow school bus.", "person\t\n standing", "東京",
                    "a\u0000cat", "a\u200bcat", "unknown🦄animal", "someone’s left hand", "x" * 101]
    lexical = {text: tokenizer.encode(text if text.rstrip().endswith(".") else text + ".").ids
               for text in descriptions}
    cases = []
    # The second case isolates two category spans. Cross-span attention is
    # forbidden, and positions reset, exactly as MM Grounding DINO requires.
    for tokens, positions, blocks in [([101, 2711, 1012, 102], [0, 0, 1, 0], [(0, 1), (1, 3), (3, 4)]),
                                      ([101, 2711, 1012, 2482, 1012, 102], [0, 0, 1, 0, 1, 0], [(0, 1), (1, 3), (3, 5), (5, 6)])]:
        n = len(tokens)
        mask = np.full((n, n), -np.inf)
        for begin, end in blocks:
            mask[begin:end, begin:end] = 0
        x = (weight("embeddings.word_embeddings.weight")[tokens] +
             weight("embeddings.position_embeddings.weight")[positions] +
             weight("embeddings.token_type_embeddings.weight")[0])
        x = norm(x, "embeddings.LayerNorm")
        for layer in range(12):
            p = f"encoder.layer.{layer}"
            q, k, v = (linear(x, p + ".attention.self." + name).reshape(n, 12, 64).transpose(1, 0, 2)
                       for name in ("query", "key", "value"))
            scores = (q @ k.transpose(0, 2, 1)) / 8 + mask
            probabilities = np.exp(scores - scores.max(-1, keepdims=True))
            probabilities /= probabilities.sum(-1, keepdims=True)
            context = (probabilities @ v).transpose(1, 0, 2).reshape(n, 768)
            x = norm(x + linear(context, p + ".attention.output.dense"), p + ".attention.output.LayerNorm")
            hidden = linear(x, p + ".intermediate.dense")
            hidden = .5 * hidden * (1 + np.vectorize(math.erf)(hidden / math.sqrt(2)))
            x = norm(x + linear(hidden, p + ".output.dense"), p + ".output.LayerNorm")
        flat = x.ravel()
        indices = np.linspace(0, flat.size - 1, 48, dtype=np.int64)
        cases.append({"tokens": tokens, "positions": positions,
                      "blocks": blocks, "indices": indices.tolist(),
                      "values": flat[indices].tolist(), "mean": float(flat.mean()),
                      "squared_norm": float(flat @ flat)})
    fixture = Path(__file__).resolve().parents[1] / "tests/data/nn/bert_text_ref_fixture.json"
    fixture.write_text("{\n" +
                       '  "checkpoint_sha256": ' + json.dumps(digest) + ",\n" +
                       '  "reference": "NumPy float64 BERT-base equations; Transformers v4.55.0",\n' +
                       '  "tokenizer_cases": ' + json.dumps(lexical, ensure_ascii=False) + ",\n" +
                       '  "cases": [\n' + ",\n".join("    " + json.dumps(case) for case in cases) +
                       "\n  ]\n}\n", encoding="utf-8")



if __name__ == "__main__":
    main()
