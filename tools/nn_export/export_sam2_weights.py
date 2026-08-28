#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Export a SAM 2.1 checkpoint to a .lfw weight file and optional fixtures.

Weight export needs only torch. The optional `--fixture` path loads the official
`sam2` package to dump CPU fp32 reference activations.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from export_onnx_weights import downsample_values, numpy_dtype_name, write_lfw

MODEL_ID = "sam2.1-hiera-base-plus"
SOURCE_NAME = "sam2.1_hiera_base_plus.pt"
CONFIG_FILE = "configs/sam2.1/sam2.1_hiera_b+.yaml"
IMAGE_SIZE = 1024
INPUT_SEED = 0
TAP_LIMIT = 256
LOW_RES_LIMIT = 4096
PROBE_GRID = 8
FIXTURE_BUDGET = 400 * 1024

KEEP_PREFIXES = (
    "image_encoder.",
    "sam_prompt_encoder.",
    "sam_mask_decoder.",
)
KEEP_EXACT = frozenset({"no_mem_embed"})

POINT_COORDS = np.array([[312.0, 505.0], [700.0, 340.0]], dtype=np.float32)
POINT_LABELS = np.array([1, 0], dtype=np.int32)
BOX_XYXY = np.array([200.0, 150.0, 800.0, 850.0], dtype=np.float32)


def keep_weight_name(name: str) -> bool:
    # Image predict (SAM2ImagePredictor.set_image / _predict) never reads the
    # video memory stack: memory_attention, memory_encoder, obj_ptr*, mask_downsample.
    return name in KEEP_EXACT or name.startswith(KEEP_PREFIXES)


def _to_numpy(tensor) -> np.ndarray:
    array = tensor.detach().cpu().numpy()
    return np.ascontiguousarray(array)


def export_checkpoint(checkpoint: Path, out_path: Path, fp16: bool) -> dict[str, np.ndarray]:
    import torch

    ckpt = torch.load(checkpoint, map_location="cpu")
    state = ckpt["model"]
    tensors: dict[str, np.ndarray] = {}
    for name, value in state.items():
        if not keep_weight_name(name):
            continue
        array = _to_numpy(value)
        if fp16 and array.dtype == np.float32:
            array = array.astype(np.float16)
        tensors[name] = np.ascontiguousarray(array)
    write_lfw(
        out_path,
        tensors,
        {
            "model": MODEL_ID,
            "fp16": bool(fp16),
            "license": "Apache-2.0",
            "source": SOURCE_NAME,
        },
    )
    return tensors


def tensor_stats(array: np.ndarray) -> dict[str, Any]:
    flat = np.ascontiguousarray(array).reshape(-1).astype(np.float32)
    return {
        "shape": [int(d) for d in array.shape],
        "max_abs": float(np.max(np.abs(flat))) if flat.size else 0.0,
        "mean": float(np.mean(flat)) if flat.size else 0.0,
        "std": float(np.std(flat)) if flat.size else 0.0,
    }


def full_small(array: np.ndarray) -> dict[str, Any]:
    arr = np.ascontiguousarray(array).astype(np.float32)
    return {
        "shape": [int(d) for d in arr.shape],
        "dtype": numpy_dtype_name(arr.dtype),
        "values": arr.reshape(-1).tolist(),
    }


def make_image(size: int = IMAGE_SIZE) -> np.ndarray:
    rng = np.random.default_rng(INPUT_SEED)
    return rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)


def probe_pixels(height: int, width: int, n: int = PROBE_GRID) -> np.ndarray:
    ys = np.linspace(0, height - 1, n, dtype=np.int64)
    xs = np.linspace(0, width - 1, n, dtype=np.int64)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    return np.stack([yy.reshape(-1), xx.reshape(-1)], axis=1)


def _stage_counts(stage_ends: list[int]) -> list[int]:
    counts = []
    prev = -1
    for end in stage_ends:
        counts.append(int(end - prev))
        prev = end
    return counts


def model_config(model) -> dict[str, Any]:
    trunk = model.image_encoder.trunk
    neck = model.image_encoder.neck
    decoder = model.sam_mask_decoder
    prompt = model.sam_prompt_encoder
    stage_ends = [int(x) for x in trunk.stage_ends]
    feat_sizes = [[256, 256], [128, 128], [64, 64]]
    attn = decoder.transformer.final_attn_token_to_image
    downsample_rate = int(attn.embedding_dim // attn.internal_dim)
    return {
        "image_size": int(model.image_size),
        "backbone_stride": int(model.backbone_stride),
        "hidden_dim": int(model.hidden_dim),
        "prompt_embed_dim": int(model.sam_prompt_embed_dim),
        "image_embedding_size": int(model.sam_image_embedding_size),
        "embed_dim": int(trunk.blocks[0].dim),
        "num_heads": [int(trunk.blocks[end].attn.num_heads) for end in trunk.stage_ends],
        "stage_dims": [int(trunk.blocks[end].dim_out) for end in trunk.stage_ends],
        "stages": _stage_counts(stage_ends),
        "stage_ends": stage_ends,
        "q_pool_blocks": [int(x) for x in trunk.q_pool_blocks],
        "q_stride": [int(x) for x in trunk.q_stride],
        "window_spec": [int(x) for x in trunk.window_spec],
        "global_att_blocks": [int(x) for x in trunk.global_att_blocks],
        "block_window_sizes": [int(b.window_size) for b in trunk.blocks],
        "channel_list": [int(x) for x in trunk.channel_list],
        "window_pos_embed_bkg_spatial_size": [
            int(x) for x in trunk.window_pos_embed_bkg_spatial_size
        ],
        "patch_embed": {"kernel": [7, 7], "stride": [4, 4], "padding": [3, 3]},
        "scalp": int(model.image_encoder.scalp),
        "fpn_top_down_levels": [int(x) for x in neck.fpn_top_down_levels],
        "fpn_interp": str(neck.fpn_interp_model),
        "fpn_fuse": str(neck.fuse_type),
        "d_model": int(neck.d_model),
        "backbone_channel_list": [int(x) for x in neck.backbone_channel_list],
        "num_feature_levels": int(model.num_feature_levels),
        "directly_add_no_mem_embed": bool(model.directly_add_no_mem_embed),
        "use_high_res_features_in_sam": bool(model.use_high_res_features_in_sam),
        "decoder_feat_resolutions": feat_sizes,
        "imagenet_mean": [0.485, 0.456, 0.406],
        "imagenet_std": [0.229, 0.224, 0.225],
        "prompt": {
            "image_embedding_size": [int(x) for x in prompt.image_embedding_size],
            "input_image_size": [int(x) for x in prompt.input_image_size],
            "mask_input_size": [int(x) for x in prompt.mask_input_size],
        },
        "decoder": {
            "transformer_depth": int(decoder.transformer.depth),
            "transformer_heads": int(decoder.transformer.num_heads),
            "transformer_mlp_dim": int(decoder.transformer.mlp_dim),
            "attention_downsample_rate": downsample_rate,
            "num_mask_tokens": int(decoder.num_mask_tokens),
            "num_multimask_outputs": int(decoder.num_multimask_outputs),
            "pred_obj_scores": bool(decoder.pred_obj_scores),
            "iou_sigmoid": bool(decoder.iou_prediction_head.sigmoid_output),
        },
        "layernorm_eps": 1e-6,
        "hiera_norm_eps": float(trunk.blocks[0].norm1.eps),
    }


def _as_numpy(tensor) -> np.ndarray:
    import torch

    if not torch.is_tensor(tensor):
        raise TypeError(f"expected tensor, got {type(tensor)}")
    return np.ascontiguousarray(tensor.detach().float().cpu().numpy())


class _Taps:
    def __init__(self) -> None:
        self.data: dict[str, np.ndarray] = {}
        self._handles: list[Any] = []

    def _add(self, module, fn) -> None:
        self._handles.append(module.register_forward_hook(fn))

    def tensor(self, name: str, module) -> None:
        def hook(_m, _inp, out):
            self.data[name] = _as_numpy(out)

        self._add(module, hook)

    def list_prefix(self, prefix: str, module) -> None:
        def hook(_m, _inp, out):
            for i, value in enumerate(out):
                self.data[f"{prefix}{i}"] = _as_numpy(value)

        self._add(module, hook)

    def keyed_list(self, prefix: str, module, key: str) -> None:
        def hook(_m, _inp, out):
            for i, value in enumerate(out[key]):
                self.data[f"{prefix}{i}"] = _as_numpy(value)

        self._add(module, hook)

    def pair(self, first: str, second: str, module) -> None:
        def hook(_m, _inp, out):
            self.data[first] = _as_numpy(out[0])
            self.data[second] = _as_numpy(out[1])

        self._add(module, hook)

    def pick(self, name: str, module, index: int) -> None:
        def hook(_m, _inp, out):
            self.data[name] = _as_numpy(out[index])

        self._add(module, hook)

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def _pack_group(arrays: dict[str, np.ndarray], limit: int) -> tuple[dict[str, Any], dict[str, Any]]:
    nodes: dict[str, Any] = {}
    stats: dict[str, Any] = {}
    for name, array in arrays.items():
        nodes[name] = downsample_values(array, limit)
        stats[name] = tensor_stats(array)
    return nodes, stats


def _serialize(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")


def dump_fixture(checkpoint: Path, out_json: Path, full_refs: Path) -> None:
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(0)

    model = build_sam2(CONFIG_FILE, ckpt_path=str(checkpoint), device="cpu", mode="eval")
    model.eval()
    predictor = SAM2ImagePredictor(model)
    image = make_image()
    pixels = probe_pixels(IMAGE_SIZE, IMAGE_SIZE)

    taps = _Taps()
    trunk = model.image_encoder.trunk
    taps.tensor("patch_embed", trunk.patch_embed)
    for i, block in enumerate(trunk.blocks):
        taps.tensor(f"block{i}", block)
    taps.list_prefix("trunk_stage", trunk)
    taps.keyed_list("fpn", model.image_encoder, "backbone_fpn")
    taps.pair("sparse_embeddings", "dense_embeddings", model.sam_prompt_encoder)
    taps.pick("transformer_tokens", model.sam_mask_decoder.transformer, 0)
    taps.tensor("upscaled_embedding", model.sam_mask_decoder.output_upscaling[4])
    taps.pick("object_score_logits", model.sam_mask_decoder, 3)

    shared: dict[str, np.ndarray] = {}
    cases_raw: dict[str, dict[str, Any]] = {}
    npz: dict[str, np.ndarray] = {"image": image}

    try:
        with torch.inference_mode():
            predictor.set_image(image)
            feats = predictor._features
            for i, feat in enumerate(feats["high_res_feats"]):
                shared[f"high_res_feats{i}"] = _as_numpy(feat)
            shared["image_embed"] = _as_numpy(feats["image_embed"])
            shared["dense_pe"] = _as_numpy(model.sam_prompt_encoder.get_dense_pe())
            for key in (
                ["patch_embed"]
                + [f"block{i}" for i in range(len(trunk.blocks))]
                + [f"trunk_stage{i}" for i in range(4)]
                + [f"fpn{i}" for i in range(3)]
            ):
                shared[key] = taps.data[key]

            npz["image_embed"] = shared["image_embed"]
            npz["high_res_feats0"] = shared["high_res_feats0"]
            npz["high_res_feats1"] = shared["high_res_feats1"]

            case_specs = (
                (
                    "points",
                    {
                        "point_coords": POINT_COORDS,
                        "point_labels": POINT_LABELS,
                        "box": None,
                    },
                ),
                (
                    "box",
                    {
                        "point_coords": None,
                        "point_labels": None,
                        "box": BOX_XYXY,
                    },
                ),
            )
            for name, prompts in case_specs:
                masks, ious, low_res = predictor.predict(
                    point_coords=prompts["point_coords"],
                    point_labels=prompts["point_labels"],
                    box=prompts["box"],
                    multimask_output=True,
                    return_logits=True,
                )
                case_taps = {
                    "sparse_embeddings": taps.data["sparse_embeddings"],
                    "dense_embeddings": taps.data["dense_embeddings"],
                    "transformer_tokens": taps.data["transformer_tokens"],
                    "upscaled_embedding": taps.data["upscaled_embedding"],
                }
                cases_raw[name] = {
                    "prompts": prompts,
                    "taps": case_taps,
                    "masks": np.ascontiguousarray(masks.astype(np.float32)),
                    "iou": np.ascontiguousarray(np.asarray(ious, dtype=np.float32)),
                    "low_res": np.ascontiguousarray(low_res.astype(np.float32)),
                    "object_score_logits": taps.data.get("object_score_logits"),
                }
                npz[f"{name}_masks"] = cases_raw[name]["masks"]
                npz[f"{name}_iou"] = cases_raw[name]["iou"]
                npz[f"{name}_low_res"] = cases_raw[name]["low_res"]
                npz[f"{name}_sparse"] = case_taps["sparse_embeddings"]
                npz[f"{name}_dense"] = case_taps["dense_embeddings"]
    finally:
        taps.close()

    cfg = model_config(model)
    cfg["mask_probe_pixels"] = pixels.astype(np.int64).tolist()

    shared_order = (
        ["patch_embed"]
        + [f"block{i}" for i in range(24)]
        + [f"trunk_stage{i}" for i in range(4)]
        + [f"fpn{i}" for i in range(3)]
        + ["high_res_feats0", "high_res_feats1", "image_embed", "dense_pe"]
    )
    shared = {k: shared[k] for k in shared_order}

    body = b""
    for tap_limit in (TAP_LIMIT, 128):
        nodes, stats = _pack_group(shared, tap_limit)
        cases: dict[str, Any] = {}
        for name, raw in cases_raw.items():
            prompts = raw["prompts"]
            case_nodes, case_stats = _pack_group(raw["taps"], tap_limit)
            case_nodes["low_res_logits"] = downsample_values(raw["low_res"], LOW_RES_LIMIT)
            case_stats["low_res_logits"] = tensor_stats(raw["low_res"])
            masks = raw["masks"]
            probe = masks[:, pixels[:, 0], pixels[:, 1]]
            entry: dict[str, Any] = {
                "point_coords": None
                if prompts["point_coords"] is None
                else prompts["point_coords"].astype(np.float32).tolist(),
                "point_labels": None
                if prompts["point_labels"] is None
                else prompts["point_labels"].astype(np.int32).tolist(),
                "box": None if prompts["box"] is None else prompts["box"].astype(np.float32).tolist(),
                "multimask_output": True,
                "return_logits": True,
                "nodes": case_nodes,
                "stats": case_stats,
                "full": {
                    "iou_predictions": full_small(raw["iou"]),
                },
                "mask_area": [int(v) for v in (masks > 0).reshape(masks.shape[0], -1).sum(axis=1)],
                "mask_probe_values": probe.astype(np.float32).tolist(),
            }
            if raw["object_score_logits"] is not None:
                entry["full"]["object_score_logits"] = full_small(raw["object_score_logits"])
            cases[name] = entry
        payload = {
            "model": MODEL_ID,
            "source": SOURCE_NAME,
            "input_shape": [IMAGE_SIZE, IMAGE_SIZE, 3],
            "input_layout": "HWC",
            "input_dtype": "uint8",
            "input_seed": INPUT_SEED,
            "config": cfg,
            "nodes": nodes,
            "stats": stats,
            "cases": cases,
        }
        body = _serialize(payload)
        if len(body) <= FIXTURE_BUDGET:
            break
    else:
        raise SystemExit(
            f"serialized SAM2 fixture is {len(body)} bytes, exceeds "
            f"FIXTURE_BUDGET={FIXTURE_BUDGET} even with tap_limit=128"
        )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_bytes(body)
    full_refs.parent.mkdir(parents=True, exist_ok=True)
    np.savez(full_refs, **npz)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="SAM 2.1 .pt checkpoint")
    parser.add_argument("--out", type=Path, required=True, help="Output .lfw path")
    parser.add_argument("--fp16", action="store_true", help="Store float32 tensors as float16")
    parser.add_argument(
        "--fixture",
        type=Path,
        help="Write sampled CPU fp32 reference activations (requires sam2)",
    )
    args = parser.parse_args()

    export_checkpoint(args.checkpoint, args.out, args.fp16)
    if args.fixture is not None:
        dump_fixture(args.checkpoint, args.fixture, args.out.parent / "refs" / "full_refs.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
