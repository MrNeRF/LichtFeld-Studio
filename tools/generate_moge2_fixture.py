#!/usr/bin/env python3
"""Generate the small ONNX-reference fixture used by the native MoGe-2 tests.

The test feeds a 1x3x70x70 RGB image made directly from the same CHW formula
as ``make_test_image`` in ``tests/test_moge2.cpp`` and asks the model for 400
tokens.  This script runs the shipped ONNX graph in ONNX Runtime, promotes
the named intermediate tensors to graph outputs, and writes sampled values for
each tap plus the complete postprocessed ``normal`` output.

Usage (from the repository root)::

    python3 tools/generate_moge2_fixture.py

Use ``--onnx``, ``--height``, ``--width``, ``--num-tokens``, ``--samples``,
and ``--output`` to override the defaults.  The fixture intentionally stores
only sampled values for intermediate/output taps; ``full.normal.values`` is
the sole complete array because the C++ parity test checks its Linf error.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


# Native tap name -> exact ONNX value carrying the same layout and semantic
# stage.  The native implementation uses the post-Reshape/Transpose patch
# tokens, terminal block residual adds, and the final tensors in each head.
TAP_TO_ONNX = {
    "patch_embed": "/encoder/patch_embed/Transpose_output_0",
    **{
        f"block{i}": f"/encoder/blocks.{i}/Add_1_output_0"
        for i in range(12)
    },
    "encoder_feat": "/encoder/ReduceSum_output_0",
    "neck0": "/neck/input_blocks.0/Conv_output_0",
    "neck1": "/neck/res_blocks.1/res_blocks.1.0/Add_output_0",
    "neck2": "/neck/res_blocks.2/res_blocks.2.0/Add_output_0",
    "neck3": "/neck/res_blocks.3/res_blocks.3.0/Add_output_0",
    "neck4": "/neck/Add_3_output_0",
    "points_head": "/points_head/output_blocks.4/Conv_output_0",
    "normal_head": "/normal_head/output_blocks.4/Conv_output_0",
    "mask_head": "/mask_head/output_blocks.4/Conv_output_0",
    "points": "points",
    "normal": "normal",
    "mask": "mask",
    "metric_scale": "metric_scale",
}


def make_test_image(height: int, width: int) -> np.ndarray:
    """Return the exact float32 NCHW image constructed by the C++ test."""

    image = np.empty((1, 3, height, width), dtype=np.float32)
    plane = height * width
    # Keep the scalar casts explicit: C++ casts numerator and denominator to
    # float before division, then stores the result in its float vector.
    for y in range(height):
        for x in range(width):
            i = y * width + x
            image.flat[i] = np.float32(
                np.float32(x) / np.float32(max(width - 1, 1))
            )
            image.flat[plane + i] = np.float32(
                np.float32(y) / np.float32(max(height - 1, 1))
            )
            image.flat[2 * plane + i] = np.float32(
                np.float32(x + y) / np.float32(max(width + height - 2, 1))
            )
    return image


def promote_outputs(model: onnx.ModelProto, names: list[str]) -> None:
    """Add intermediate values to graph outputs without changing the file."""

    produced = {output for node in model.graph.node for output in node.output}
    graph_outputs = {output.name for output in model.graph.output}
    missing = [name for name in names if name not in produced and name not in graph_outputs]
    if missing:
        raise RuntimeError("ONNX values are not produced by the graph: " + ", ".join(missing))
    for name in names:
        if name not in graph_outputs:
            model.graph.output.append(
                onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None)
            )
            graph_outputs.add(name)


def sampled(array: np.ndarray, count: int) -> dict[str, object]:
    """Encode evenly distributed flat samples in the C++ fixture schema."""

    flat = np.asarray(array, dtype=np.float32).reshape(-1)
    if not np.isfinite(flat).all():
        raise RuntimeError("non-finite value in ONNX output with shape " + str(array.shape))
    sample_count = min(count, flat.size)
    indices = np.linspace(0, flat.size - 1, sample_count, dtype=np.int64)
    indices = np.unique(indices)
    return {
        "shape": [int(dimension) for dimension in array.shape],
        "indices": [int(index) for index in indices],
        "values": [float(flat[index]) for index in indices],
    }


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[3]
    lane = Path(__file__).resolve().parent
    home = Path.home()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--onnx",
        type=Path,
        default=home / ".lichtfeld/onnx/moge-2-vitb-normal.onnx",
        help="MoGe-2 ViT-B ONNX reference (default: %(default)s)",
    )
    parser.add_argument("--height", type=int, default=70)
    parser.add_argument("--width", type=int, default=70)
    parser.add_argument("--num-tokens", type=int, default=400)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument(
        "--output",
        type=Path,
        default=lane / "moge2_ref_fixture.json",
        help="fixture JSON output (default: %(default)s)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="ORT intra-op threads (default: %(default)s for reproducibility)",
    )
    parser.set_defaults(repo_root=root)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.height < 1 or args.width < 1 or args.num_tokens < 1 or args.samples < 1:
        raise SystemExit("height, width, num-tokens, and samples must be positive")
    if not args.onnx.is_file():
        raise SystemExit(f"ONNX model not found: {args.onnx}")

    image = make_test_image(args.height, args.width)
    model = onnx.load(str(args.onnx), load_external_data=False)
    promote_outputs(model, list(TAP_TO_ONNX.values()))
    session_options = ort.SessionOptions()
    if args.threads > 0:
        session_options.intra_op_num_threads = args.threads
    session = ort.InferenceSession(
        model.SerializeToString(), session_options, providers=["CPUExecutionProvider"]
    )
    feed = {"image": image, "num_tokens": np.asarray(args.num_tokens, dtype=np.int64)}
    output_names = [output.name for output in session.get_outputs()]
    outputs = dict(zip(output_names, session.run(None, feed)))

    absent = [name for name in TAP_TO_ONNX.values() if name not in outputs]
    if absent:
        raise RuntimeError("promoted ONNX outputs missing from ORT result: " + ", ".join(absent))

    nodes = {}
    for tap, onnx_name in TAP_TO_ONNX.items():
        nodes[tap] = sampled(outputs[onnx_name], args.samples)
    normal = np.asarray(outputs["normal"], dtype=np.float32)
    if not np.isfinite(normal).all():
        raise RuntimeError("non-finite value in complete normal output")

    payload = {
        "input_shape": [1, 3, args.height, args.width],
        "num_tokens": args.num_tokens,
        "nodes": nodes,
        "full": {
            "normal": {
                "shape": [int(dimension) for dimension in normal.shape],
                "values": [float(value) for value in normal.reshape(-1)],
            }
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    size = args.output.stat().st_size
    if size >= 2 * 1024 * 1024:
        raise RuntimeError(f"fixture is too large: {size} bytes")
    print(f"wrote {args.output} ({size} bytes)")
    print(f"input shape={payload['input_shape']} num_tokens={args.num_tokens}")
    for tap, node in nodes.items():
        print(f"{tap}: {node['shape']} <- {TAP_TO_ONNX[tap]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
