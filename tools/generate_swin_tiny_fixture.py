#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate a small Swin-T numerical oracle with NumPy float64, without Torch.

Inference equations follow Transformers v4.55.0 modeling_swin.py (Apache-2.0).
This independent development oracle uses full window masks rather than the
native backbone's interior/boundary batching optimization.
"""
import argparse
import functools
import json
import math
from pathlib import Path

import numpy as np

from nn_export.lfw import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    with args.checkpoint.open("rb") as stream:
        digest = sha256_file(stream)
        if digest != "f0e476be3d65c7c50e13029b0adc4bb5d4e05f3eb6d7bb04856578095e6ad92f":
            raise ValueError("Expected the pinned official LLMDet Tiny checkpoint")
        stream.seek(0)
        length = int.from_bytes(stream.read(8), "little")
        header = json.loads(stream.read(length))
    mapped = np.memmap(args.checkpoint, mode="r", dtype="u1")

    @functools.lru_cache(None)
    def weight(name):
        record = header["model.backbone.conv_encoder.model." + name]
        return np.ndarray(record["shape"], dtype="<f4", buffer=mapped,
                          offset=8 + length + record["data_offsets"][0]).astype(np.float64)

    def norm(x, p):
        return ((x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-5)
                * weight(p + ".weight") + weight(p + ".bias"))

    def dense(x, p):
        return x @ weight(p + ".weight").T + weight(p + ".bias")

    def pad(x, multiple):
        h, w = x.shape[1:3]
        return np.pad(x, ((0, 0), (0, -h % multiple), (0, -w % multiple), (0, 0)))

    def partition(x):
        _, h, w, c = x.shape
        return x.reshape(1, h // 7, 7, w // 7, 7, c).transpose(0, 1, 3, 2, 4, 5).reshape(-1, 49, c)

    def reverse(x, h, w):
        c = x.shape[-1]
        return x.reshape(1, h // 7, w // 7, 7, 7, c).transpose(0, 1, 3, 2, 4, 5).reshape(1, h, w, c)

    grid = np.stack(np.meshgrid(np.arange(7), np.arange(7), indexing="ij")).reshape(2, -1)
    delta = grid[:, :, None] - grid[:, None, :]
    relative = (delta[0] + 6) * 13 + delta[1] + 6
    cases = []
    for height, width in [(31, 35), (17, 19)]:
        pixels = ((np.arange(3 * height * width) * 37 % 251 - 125) / 127).astype(np.float32)
        x = pad(pixels.astype(np.float64).reshape(1, 3, height, width).transpose(0, 2, 3, 1), 4)
        h, w = x.shape[1:3]
        patches = x.reshape(1, h // 4, 4, w // 4, 4, 3).transpose(0, 1, 3, 5, 2, 4).reshape(1, h // 4, w // 4, 48)
        x = (patches @ weight("embeddings.patch_embeddings.projection.weight").reshape(96, 48).T
             + weight("embeddings.patch_embeddings.projection.bias"))
        x = norm(x, "embeddings.norm")
        outputs = []
        for stage, depth in enumerate([2, 2, 6, 2]):
            h, w, c = x.shape[1:]
            heads = c // 32
            pstage = f"encoder.layers.{stage}"
            for block in range(depth):
                p = f"{pstage}.blocks.{block}"
                shortcut = x
                z = pad(norm(x, p + ".layernorm_before"), 7)
                hp, wp = z.shape[1:3]
                mask = None
                if block % 2:
                    z = np.roll(z, (-3, -3), axis=(1, 2))
                    labels = np.zeros((1, hp, wp, 1))
                    for iy, ys in enumerate([slice(0, -7), slice(-7, -3), slice(-3, None)]):
                        for ix, xs in enumerate([slice(0, -7), slice(-7, -3), slice(-3, None)]):
                            labels[:, ys, xs, :] = iy * 3 + ix
                    labels = partition(labels).squeeze(-1)
                    mask = np.where(labels[:, :, None] == labels[:, None, :], 0., -100.)[:, None]
                z = partition(z)
                q, k, v = [dense(z, p + ".attention.self." + name).reshape(-1, 49, heads, 32).transpose(0, 2, 1, 3)
                           for name in ["query", "key", "value"]]
                scores = q @ k.swapaxes(-1, -2) / math.sqrt(32)
                scores += weight(p + ".attention.self.relative_position_bias_table")[relative].transpose(2, 0, 1)[None]
                if mask is not None:
                    scores += mask
                scores = np.exp(scores - scores.max(-1, keepdims=True))
                scores /= scores.sum(-1, keepdims=True)
                z = (scores @ v).transpose(0, 2, 1, 3).reshape(-1, 49, c)
                z = reverse(dense(z, p + ".attention.output.dense"), hp, wp)
                if block % 2:
                    z = np.roll(z, (3, 3), axis=(1, 2))
                x = shortcut + z[:, :h, :w]
                z = dense(norm(x, p + ".layernorm_after"), p + ".intermediate.dense")
                z = .5 * z * (1 + np.vectorize(math.erf)(z / math.sqrt(2)))
                x = x + dense(z, p + ".output.dense")
            if stage:
                feature = norm(x, f"hidden_states_norms.stage{stage + 1}").transpose(0, 3, 1, 2)
                flat = feature.ravel()
                indices = np.linspace(0, flat.size - 1, 48, dtype=np.int64)
                outputs.append({"shape": list(feature.shape), "indices": indices.tolist(),
                                "values": flat[indices].tolist(), "mean": float(flat.mean()),
                                "squared_norm": float(flat @ flat)})
            if stage < 3:
                x = pad(x, 2)
                x = np.concatenate([x[:, 0::2, 0::2], x[:, 1::2, 0::2], x[:, 0::2, 1::2], x[:, 1::2, 1::2]], -1)
                x = norm(x, pstage + ".downsample.norm") @ weight(pstage + ".downsample.reduction.weight").T
        cases.append({"height": height, "width": width, "outputs": outputs})
    output = Path(__file__).resolve().parents[1] / "tests/data/nn/swin_tiny_ref_fixture.json"
    output.write_text('{\n  "checkpoint_sha256": ' + json.dumps(digest) + ',\n'
                      '  "reference": "NumPy float64 Swin-T equations; Transformers v4.55.0",\n'
                      '  "cases": [\n' + ',\n'.join('    ' + json.dumps(case) for case in cases) + '\n  ]\n}\n', encoding="utf-8")


if __name__ == "__main__":
    main()
