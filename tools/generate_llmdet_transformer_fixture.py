#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Small independent NumPy float64 oracles for LLMDet's encoder and decoder.

Equations follow Transformers v4.55.0 modeling_mm_grounding_dino.py
(Apache-2.0). Uses synthetic backbone features, not native intermediate output.
No Torch or application inference dependency is needed.
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
        record = header[name if name.startswith("bbox_embed.") else "model." + name]
        return np.ndarray(record["shape"], dtype="<f4", buffer=mapped,
                          offset=8 + length + record["data_offsets"][0]).astype(np.float64)

    def norm(x, p):
        return ((x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-5)
                * weight(p + ".weight") + weight(p + ".bias"))

    def dense(x, p):
        return x @ weight(p + ".weight").T + weight(p + ".bias")

    def softmax(x):
        x = np.exp(x - x.max(-1, keepdims=True))
        return x / x.sum(-1, keepdims=True)

    def heads(x, count):
        return x.reshape(x.shape[0], count, -1).transpose(1, 0, 2)

    def merge(x):
        return x.transpose(1, 0, 2).reshape(x.shape[1], -1)

    def sine(x, features, temperature):
        angle = np.asarray(x)[..., None] * (2 * np.pi) / temperature ** (np.arange(features // 2) * 2 / features)
        return np.stack([np.sin(angle), np.cos(angle)], -1).reshape(*angle.shape[:-1], features)

    def synthetic(shape):
        # Binary-representable values; identical inputs in C++ and this oracle.
        return ((np.arange(math.prod(shape)) * 37 % 251 - 125) / 128).reshape(shape).astype(np.float64)

    def deformable(values, locations, a, shapes):
        attended = np.zeros((locations.shape[0], 8, 32))
        first = 0
        for level, (h, w) in enumerate(shapes):
            for query in range(len(attended)):
                for head in range(8):
                    for point in range(4):
                        x, y = locations[query, head, level, point] * [w, h] - .5
                        x0, y0 = math.floor(x), math.floor(y)
                        for sy in [y0, y0 + 1]:
                            for sx in [x0, x0 + 1]:
                                if 0 <= sx < w and 0 <= sy < h:
                                    attended[query, head] += (values[first + sy * w + sx, head] * (1 - abs(x - sx)) * (1 - abs(y - sy))
                                                             * a[query, head, level, point])
            first += h * w
        return attended.reshape(-1, 256)

    inputs = [synthetic((192, 3, 5)), synthetic((384, 2, 3)), synthetic((768, 1, 2))]
    maps, positions, references, shapes = [], [], [], []
    for level in range(4):
        p = f"input_proj_vision.{level}"
        source = inputs[min(level, 2)]
        kernel = weight(p + ".0.weight")
        if level < 3:
            x = np.einsum("oihw,iyx->oyx", kernel, source) + weight(p + ".0.bias")[:, None, None]
        else:
            source = np.pad(source, ((0, 0), (1, 1), (1, 1)))
            x = np.empty((256, 1, 1))
            x[:, 0, 0] = np.einsum("oihw,ihw->o", kernel, source[:, :3, :3]) + weight(p + ".0.bias")
        grouped = x.reshape(32, -1)
        x = ((grouped - grouped.mean(-1, keepdims=True)) / np.sqrt(grouped.var(-1, keepdims=True) + 1e-5)).reshape(x.shape)
        x = x * weight(p + ".1.weight")[:, None, None] + weight(p + ".1.bias")[:, None, None]
        h, w = x.shape[1:]
        shapes.append([h, w])
        maps.append(x.reshape(256, -1).T)
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        positions.append(np.concatenate([sine((yy + 1) / (h + 1e-6), 128, 20), sine((xx + 1) / (w + 1e-6), 128, 20)], -1).reshape(-1, 256)
                         + weight("level_embed")[level])
        references.append(np.stack([(xx + .5) / w, (yy + .5) / h], -1).reshape(-1, 2))
    vision = np.concatenate(maps)
    text = dense(synthetic((6, 768)), "text_projection")
    vision_position = np.concatenate(positions)
    reference = np.concatenate(references)
    normalizer = np.array([[w, h] for h, w in shapes])
    position_ids = [0, 0, 1, 0, 1, 0]
    text_position = sine(position_ids, 256, 10000)
    text_mask = np.full((6, 6), -np.inf)
    for first, last in [(0, 1), (1, 3), (3, 5), (5, 6)]:
        text_mask[first:last, first:last] = 0
    for layer in range(6):
        p = f"encoder.layers.{layer}"
        f = p + ".fusion_layer"
        vision = norm(vision, f + ".layer_norm_vision")
        text = norm(text, f + ".layer_norm_text")
        qv = heads(dense(vision, f + ".attn.vision_proj"), 4) / 16
        kt = heads(dense(text, f + ".attn.text_proj"), 4)
        vv = heads(dense(vision, f + ".attn.values_vision_proj"), 4)
        vt = heads(dense(text, f + ".attn.values_text_proj"), 4)
        scores = qv @ kt.swapaxes(-1, -2)
        scores = np.clip(scores - scores.max(), -50000, 50000)
        transposed = scores.swapaxes(-1, -2)
        tscores = np.clip(transposed - transposed.max(-1, keepdims=True), -50000, 50000)
        dv = dense(merge(softmax(scores) @ vt), f + ".attn.out_vision_proj")
        dt = dense(merge(softmax(tscores) @ vv), f + ".attn.out_text_proj")
        vision = vision + dv * weight(f + ".vision_param")
        text = text + dt * weight(f + ".text_param")
        t = p + ".text_enhancer_layer"
        q = heads(dense(text + text_position, t + ".self_attn.query"), 4)
        k = heads(dense(text + text_position, t + ".self_attn.key"), 4)
        v = heads(dense(text, t + ".self_attn.value"), 4)
        attention = merge(softmax(q @ k.swapaxes(-1, -2) / 8 + text_mask) @ v)
        text = norm(text + dense(attention, t + ".self_attn.out_proj"), t + ".layer_norm_before")
        text = norm(text + dense(np.maximum(dense(text, t + ".fc1"), 0), t + ".fc2"), t + ".layer_norm_after")
        d = p + ".deformable_layer"
        queries = vision + vision_position
        offsets = dense(queries, d + ".self_attn.sampling_offsets").reshape(-1, 8, 4, 4, 2)
        locations = reference[:, None, None, None, :] + offsets / normalizer[None, None, :, None, :]
        a = softmax(dense(queries, d + ".self_attn.attention_weights").reshape(-1, 8, 16)).reshape(-1, 8, 4, 4)
        values = dense(vision, d + ".self_attn.value_proj").reshape(-1, 8, 32)
        attended = dense(deformable(values, locations, a, shapes), d + ".self_attn.output_proj")
        vision = norm(vision + attended, d + ".self_attn_layer_norm")
        vision = norm(vision + dense(np.maximum(dense(vision, d + ".fc1"), 0), d + ".fc2"), d + ".final_layer_norm")
    def samples(x):
        flat = x.ravel()
        indices = np.linspace(0, flat.size - 1, 64, dtype=np.int64)
        return {"shape": [1, *x.shape], "indices": indices.tolist(), "values": flat[indices].tolist(),
                "mean": float(flat.mean()), "squared_norm": float(flat @ flat)}
    fixture = {"checkpoint_sha256": digest, "reference": "NumPy float64 MM Grounding DINO encoder; Transformers v4.55.0",
               "spatial_shapes": shapes, "vision": samples(vision), "text": samples(text)}
    output = Path(__file__).resolve().parents[1] / "tests/data/nn/llmdet_encoder_ref_fixture.json"
    def write(path, value):
        path.write_text("{\n" + ",\n".join("  " + json.dumps(key) + ": " + json.dumps(item)
                                        for key, item in value.items()) + "\n}\n", encoding="utf-8")
    write(output, fixture)

    def decoder_input(shape):
        ids = np.arange(math.prod(shape), dtype=np.uint64)
        bits = (ids * 1664525 + 1013904223) & 0xffffffff
        return (((bits >> 8) & 65535).astype(np.int32) - 32768).reshape(shape).astype(np.float64) / 32768

    def bbox(x, p):
        return dense(np.maximum(dense(np.maximum(dense(x, p + ".layers.0"), 0), p + ".layers.1"), 0), p + ".layers.2")

    def bbox_prefix(i):
        return "bbox_embed.0" if i == 0 else f"decoder.bbox_embed.{i}"

    def sigmoid(x):
        return .5 * (np.tanh(x / 2) + 1)

    def logit(x):
        x = np.clip(x, 1e-5, 1 - 1e-5)
        return np.log(x / (1 - x))

    shapes = [[20, 40], [10, 20], [5, 10], [3, 5]]
    count = sum(h * w for h, w in shapes)
    text = decoder_input((6, 256))
    pool = decoder_input((8192, 256))
    pool_query = norm(dense(pool, "enc_output"), "enc_output_norm")
    pool_scores = (pool_query @ text.T / 16 + weight("encoder_output_class_embed.bias")).max(-1)
    gap = float(np.ptp(pool_scores)) / (count * 2)
    ordered_pool = np.argsort(pool_scores, kind="stable")
    for _ in range(16):
        rows = []
        last_score = -np.inf
        for index in ordered_pool:
            if pool_scores[index] - last_score >= gap:
                rows.append(int(index))
                last_score = pool_scores[index]
        if len(rows) >= count:
            break
        gap *= .5
    if len(rows) < count:
        raise ValueError("Cannot construct numerically separated proposal ranking")
    rows = rows[:count]
    vision = pool[rows]
    proposals = []
    for level, (h, w) in enumerate(shapes):
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        proposals.append(np.stack([(xx + .5) / w, (yy + .5) / h, np.full((h, w), .05 * 2 ** level), np.full((h, w), .05 * 2 ** level)], -1).reshape(-1, 4))
    proposals = np.concatenate(proposals)
    valid = ((proposals > .01) & (proposals < .99)).all(-1)
    object_query = norm(dense(vision * valid[:, None], "enc_output"), "enc_output_norm")
    proposal_logits = np.where(valid[:, None], np.log(proposals / (1 - proposals)), np.inf)
    score = (object_query @ text.T / 16 + weight("encoder_output_class_embed.bias")).max(-1)
    order = np.argsort(-score, kind="stable")[:900]
    # No exact ties: ranking comparisons must not depend on an unspecified tie rule.
    if np.min(np.abs(np.diff(np.sort(score)))) == 0:
        raise ValueError("Decoder reference requires distinct proposal scores")
    reference = sigmoid((bbox(object_query, "encoder_output_bbox_embed") + proposal_logits)[order])
    hidden = weight("query_position_embeddings.weight").copy()
    for i in range(6):
        last_reference = reference
        position = np.concatenate([sine(reference[:, axis], 128, 10000) for axis in [1, 0, 2, 3]], -1)
        position = dense(np.maximum(dense(position, "decoder.reference_points_head.layers.0"), 0), "decoder.reference_points_head.layers.1")
        p = f"decoder.layers.{i}"
        def mha(q, k, v, prefix):
            q = heads(dense(q, prefix + ".query"), 8)
            k = heads(dense(k, prefix + ".key"), 8)
            v = heads(dense(v, prefix + ".value"), 8)
            return dense(merge(softmax(q @ k.swapaxes(-1, -2) / math.sqrt(32)) @ v), prefix + ".out_proj")
        hidden = norm(hidden + mha(hidden + position, hidden + position, hidden, p + ".self_attn"), p + ".self_attn_layer_norm")
        hidden = norm(hidden + mha(hidden + position, text, text, p + ".encoder_attn_text"), p + ".encoder_attn_text_layer_norm")
        queries = hidden + position
        offsets = dense(queries, p + ".encoder_attn.sampling_offsets").reshape(900, 8, 4, 4, 2)
        locations = reference[:, None, None, None, :2] + offsets * reference[:, None, None, None, 2:] / 8
        a = softmax(dense(queries, p + ".encoder_attn.attention_weights").reshape(900, 8, 16)).reshape(900, 8, 4, 4)
        values = dense(vision, p + ".encoder_attn.value_proj").reshape(-1, 8, 32)
        attended = dense(deformable(values, locations, a, shapes), p + ".encoder_attn.output_proj")
        hidden = norm(hidden + attended, p + ".encoder_attn_layer_norm")
        hidden = norm(hidden + dense(np.maximum(dense(hidden, p + ".fc1"), 0), p + ".fc2"), p + ".final_layer_norm")
        reference = sigmoid(bbox(hidden, bbox_prefix(i)) + logit(reference))
    final = norm(hidden, "decoder.layer_norm")
    boxes = sigmoid(bbox(final, bbox_prefix(5)) + logit(last_reference))
    logits = final @ text.T / 16 + weight("decoder.class_embed.5.bias")
    write(output.with_name("llmdet_decoder_ref_fixture.json"), {
        "checkpoint_sha256": digest, "reference": "NumPy float64 MM Grounding DINO decoder; Transformers v4.55.0",
        "spatial_shapes": shapes, "source_rows": rows, "minimum_rank_gap": float(np.min(np.abs(np.diff(np.sort(score))))),
        "boxes": samples(boxes), "logits": samples(logits)})


if __name__ == "__main__":
    main()
