#!/usr/bin/env python3
"""Export LPIPS v0.1 VGG16 weights.

Pass --fixture tests/data/nn/lpips_ref_fixture.json to also regenerate
parity fixtures and lossless crops from the local data/bicycle dataset.

The synthetic pair is 1x3x64x96.  For pixel (x,y), image A has channels
((x+2*y)/(95+2*63), (x%17)/16, ((3*x+5*y)%29)/28), while image B has
(1-A.r, ((x+3*y)%19)/18, ((7*x+2*y)%31)/30).  The same formula is in
tests/test_lpips.cpp.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


FIXTURE_BUDGET = 400 * 1024
CROP_SIZE = 256
CROP_BOX = (400, 250, 400 + CROP_SIZE, 250 + CROP_SIZE)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_lfw(path: Path, tensors: dict[str, np.ndarray], meta: dict[str, object]) -> None:
    payload_offset = 0
    entries: dict[str, dict[str, object]] = {}
    blobs: list[tuple[int, bytes]] = []
    for name, value in tensors.items():
        array = np.ascontiguousarray(value, dtype=np.float32)
        blob = array.tobytes(order="C")
        payload_offset = (payload_offset + 63) & ~63
        entries[name] = {
            "dtype": "float32",
            "shape": [int(dimension) for dimension in array.shape],
            "offset": payload_offset,
            "length": len(blob),
        }
        blobs.append((payload_offset, blob))
        payload_offset += len(blob)

    header = json.dumps(
        {"format": "lfw", "version": 1, "meta": meta, "tensors": entries},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    file_payload_offset = (8 + len(header) + 63) & ~63
    output = bytearray(file_payload_offset + payload_offset)
    output[:4] = b"LFW1"
    output[4:8] = len(header).to_bytes(4, "little")
    output[8 : 8 + len(header)] = header
    for offset, blob in blobs:
        start = file_payload_offset + offset
        output[start : start + len(blob)] = blob
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(output)


def synthetic_pair(height: int = 64, width: int = 96) -> tuple[np.ndarray, np.ndarray]:
    a = np.empty((1, 3, height, width), dtype=np.float32)
    b = np.empty_like(a)
    for y in range(height):
        for x in range(width):
            a[0, 0, y, x] = np.float32((x + 2 * y) / (width - 1 + 2 * (height - 1)))
            a[0, 1, y, x] = np.float32((x % 17) / 16.0)
            a[0, 2, y, x] = np.float32(((3 * x + 5 * y) % 29) / 28.0)
            b[0, 0, y, x] = np.float32(1.0) - a[0, 0, y, x]
            b[0, 1, y, x] = np.float32((x + 3 * y) % 19) / np.float32(18.0)
            b[0, 2, y, x] = np.float32((7 * x + 2 * y) % 31) / np.float32(30.0)
    return a, b


def sampled(array, count: int = 8) -> dict[str, object]:
    flat = np.asarray(array, dtype=np.float32).reshape(-1)
    indices = np.linspace(0, flat.size - 1, min(count, flat.size), dtype=np.int64)
    return {
        "shape": [int(dimension) for dimension in array.shape],
        "indices": [int(index) for index in indices],
        "values": [float(flat[index]) for index in indices],
    }


def export(args: argparse.Namespace) -> None:
    import torch
    import torchvision
    import lpips

    pip_model = lpips.LPIPS(net="vgg", version="0.1")
    pip_model.eval()
    pip_linears = list(pip_model.lin) if hasattr(pip_model, "lin") else [getattr(pip_model, f"lin{i}") for i in range(5)]

    def linear_conv(layer):
        return layer[1] if hasattr(layer, "__getitem__") else layer.model[1]

    features = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1).features
    tensors: dict[str, np.ndarray] = {
        "scaling.shift": np.array([[-0.030, -0.088, -0.188]], dtype=np.float32).reshape(1, 3, 1, 1),
        "scaling.scale": np.array([[0.458, 0.448, 0.450]], dtype=np.float32).reshape(1, 3, 1, 1),
    }
    for index, layer in enumerate(features):
        if isinstance(layer, torch.nn.Conv2d):
            tensors[f"vgg.features.{index}.weight"] = layer.weight.detach().cpu().numpy()
            tensors[f"vgg.features.{index}.bias"] = layer.bias.detach().cpu().numpy()
    for index, linear in enumerate(pip_linears):
        tensors[f"lin{index}.weight"] = linear_conv(linear).weight.detach().cpu().numpy()

    source_vgg = Path(torchvision.models.vgg.__file__).read_bytes()
    source_lpips = Path(lpips.__file__).read_bytes()
    meta = {
        "model": "lpips-vgg16-v0.1",
        "net": "vgg",
        "lpips_version": "0.1",
        "torchvision_weight_id": "IMAGENET1K_V1",
        "storage_dtype": "float32",
        "linear_weight_shape": "[1,C,1,1]",
        "input_scaling_default": "Identity",
        "torchvision_vgg_source_sha256": sha256_bytes(source_vgg),
        "lpips_package_source_sha256": sha256_bytes(source_lpips),
    }
    write_lfw(args.out, tensors, meta)
    print(f"wrote {args.out} sha256={sha256_bytes(args.out.read_bytes())}")
    if args.fixture is None:
        return

    pair_a, pair_b = synthetic_pair()
    ta = torch.from_numpy(pair_a)
    tb = torch.from_numpy(pair_b)

    with torch.no_grad():
        identity_value, identity_res = pip_model(ta, tb, retPerLayer=True, normalize=False)
        normalize_value, normalize_res = pip_model(ta, tb, retPerLayer=True, normalize=True)
        identity_features_a = pip_model.net(pip_model.scaling_layer(ta))
        normalize_features_a = pip_model.net(pip_model.scaling_layer(ta * 2 - 1))
        identity_taps = [lpips.normalize_tensor(x).detach().cpu().numpy() for x in identity_features_a]
        normalize_taps = [lpips.normalize_tensor(x).detach().cpu().numpy() for x in normalize_features_a]

    real_files = sorted((args.repo / "data/bicycle/images_4").glob("*.JPG"))
    first = args.repo / "data/bicycle/images_4/_DSC8679.JPG"
    real_next = next(path for path in real_files if path > first)

    def pil_chw(path: Path):
        array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / np.float32(255.0)
        return np.ascontiguousarray(array.transpose(2, 0, 1)[None])

    real_a, real_b = pil_chw(first), pil_chw(real_next)
    with torch.no_grad():
        real_ta = torch.from_numpy(real_a)
        real_tb = torch.from_numpy(real_b)
        real_identity = float(pip_model(real_ta, real_tb, normalize=False).item())
        real_normalize = float(pip_model(real_ta, real_tb, normalize=True).item())

    crop_a_path = args.repo / "tests/data/nn/lpips_crop_a.png"
    crop_b_path = args.repo / "tests/data/nn/lpips_crop_b.png"
    crop_a_path.parent.mkdir(parents=True, exist_ok=True)
    crop_a_image = Image.open(first).convert("RGB").crop(CROP_BOX)
    crop_b_image = Image.open(real_next).convert("RGB").crop(CROP_BOX)
    crop_a_image.save(crop_a_path, format="PNG")
    crop_b_image.save(crop_b_path, format="PNG")
    crop_a = pil_chw(crop_a_path)
    crop_b = pil_chw(crop_b_path)
    with torch.no_grad():
        crop_ta = torch.from_numpy(crop_a)
        crop_tb = torch.from_numpy(crop_b)
        crop_identity = float(pip_model(crop_ta, crop_tb, normalize=False).item())
        crop_normalize = float(pip_model(crop_ta, crop_tb, normalize=True).item())
        crop_features_a = pip_model.net(pip_model.scaling_layer(crop_ta))
        crop_features_b = pip_model.net(pip_model.scaling_layer(crop_tb))
        crop_taps_a = [lpips.normalize_tensor(x) for x in crop_features_a]
        crop_taps_b = [lpips.normalize_tensor(x) for x in crop_features_b]
        crop_per_tap = [
            float(linear((fa - fb).square()).mean().item())
            for fa, fb, linear in zip(crop_taps_a, crop_taps_b, pip_linears)
        ]

    fixture = {
        "model": meta,
        "synthetic": {
            "input_shape": [1, 3, 64, 96],
            "formula": __doc__.split("\n\n", 2)[2].strip(),
            "identity": {
                "normalized_taps": [sampled(tap) for tap in identity_taps],
                "per_tap_scalars": [float(value.item()) for value in identity_res],
                "lpips": float(identity_value.item()),
            },
            "normalize": {
                "normalized_taps": [sampled(tap) for tap in normalize_taps],
                "per_tap_scalars": [float(value.item()) for value in normalize_res],
                "lpips": float(normalize_value.item()),
            },
        },
        "real": {
            "pred": first.name,
            "target": real_next.name,
            "shape": [1, 3, int(real_a.shape[2]), int(real_a.shape[3])],
            "identity": real_identity,
            "normalize": real_normalize,
        },
        "crop_pair": {
            "files": ["lpips_crop_a.png", "lpips_crop_b.png"],
            "crop_box_xyxy": list(CROP_BOX),
            "lpips_identity": crop_identity,
            "lpips_normalize": crop_normalize,
            "per_tap_identity": crop_per_tap,
        },
    }
    args.fixture.parent.mkdir(parents=True, exist_ok=True)
    args.fixture.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")
    if args.fixture.stat().st_size >= FIXTURE_BUDGET:
        raise RuntimeError(f"fixture is {args.fixture.stat().st_size} bytes, over budget")
    print(f"wrote {args.fixture} bytes={args.fixture.stat().st_size}")
    print(f"real pair={first.name} vs {real_next.name}")
    print(f"wrote {crop_a_path} and {crop_b_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path.home() / ".lichtfeld/onnx/lpips-vgg16-v0.1.lfw")
    parser.add_argument("--fixture", type=Path, help="Also regenerate parity fixtures; requires data/bicycle")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    export(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
