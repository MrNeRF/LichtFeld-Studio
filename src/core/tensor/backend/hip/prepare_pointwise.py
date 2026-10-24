# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate HIP dispatch from the shared catalog without editing other backends."""
import argparse
from pathlib import Path
import re


SCALARS = {
    "Add": "add", "Sub": "sub", "Mul": "mul", "Div": "div", "Pow": "pow",
    "Mod": "mod", "Maximum": "maximum", "Minimum": "minimum", "Equal": "equal",
    "NotEqual": "not_equal", "Less": "less", "LessEqual": "less_equal",
    "Greater": "greater", "GreaterEqual": "greater_equal",
}
COMPOSITES = {
    "ExpThenMulScalar": "ops::exp_op{}(x) * scalar",
    "MulScalarThenAbs": "ops::abs_op{}(x * scalar)",
    "MulScalarThenRelu": "ops::relu_op{}(x * scalar)",
}


def generate(source):
    functors = (source / "tensor_functors.hpp").read_text(encoding="utf-8")
    functors = functors.replace("__CUDACC__", "__HIPCC__").replace(
        "__CUDA_ARCH__", "__HIP_DEVICE_COMPILE__")
    # HIP fmaxf selects +0 for (-0,+0); keep the public CPU ReLU's signed zero.
    original = "isnan(static_cast<float>(x)) ? x : fmaxf(x, T(0))"
    if functors.count(original) != 1:
        raise ValueError("Shared ReLU changed: review the HIP signed-zero adaptation")
    functors = functors.replace(original, "isnan(static_cast<float>(x)) ? x : (x < T(0) ? T(0) : x)")
    dispatch = {"f32": [], "f16": []}
    identifiers = set()
    for line in (source / "pointwise_ops.def").read_text(encoding="utf-8").splitlines():
        line = line.split("//", 1)[0].strip()
        if not line:
            continue
        row = re.fullmatch(r'LFS_POINTWISE_OP\((\w+),\s*([\w:]+),\s*"[\w]+"\)', line)
        if not row:
            raise ValueError(f"Unsupported pointwise catalog row: {line}")
        identifier, functor = row.groups()
        if identifier in identifiers:
            raise ValueError(f"Duplicate pointwise operation: {identifier}")
        identifiers.add(identifier)
        mode = 0
        if identifier in COMPOSITES:
            functor = "[=] __device__ (float x) { return " + COMPOSITES[identifier] + "; }"
        elif identifier.endswith("Scalar"):
            base = identifier[:-6]
            if base not in SCALARS:
                raise ValueError(f"Unsupported scalar operation: {identifier}")
            mode, functor = 1, "ops::" + SCALARS[base] + "_op{}"
        else:
            if not functor.startswith("ops::"):
                raise ValueError(f"Unsupported pointwise functor: {functor}")
            mode = 2 if identifier.endswith("Tensor") else 0
            functor += "{}"
        predicate = identifier.startswith(("Equal", "NotEqual", "Less", "Greater", "Logical", "IsNan", "IsInf", "IsFinite"))
        for storage, dtype in (("f32", "float"), ("f16", "__half")):
            output = "uint8_t" if predicate else dtype
            dispatch[storage].append(
                f"case LFS_HIP_OP_{identifier}: launch_pointwise<{mode}, {output}, {dtype}>"
                f"(context, lhs, rhs, output, count, scalar, {functor}); break;")
    if not identifiers:
        raise ValueError("The pointwise operation catalog is empty")
    return {
        "hip_pointwise_functors.hpp": functors,
        **{f"hip_pointwise_{dtype}.inc": "\n".join(rows) + "\n" for dtype, rows in dispatch.items()},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generated = generate(args.source)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, contents in generated.items():
        target = args.output / name
        if not target.exists() or target.read_text(encoding="utf-8") != contents:
            target.write_text(contents, encoding="utf-8")


if __name__ == "__main__":
    main()
