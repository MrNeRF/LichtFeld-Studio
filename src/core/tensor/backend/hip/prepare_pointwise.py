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
    # Retain the shared device ReLU, including CUDA's +0 result for -0.
    if functors.count("isnan(static_cast<float>(x)) ? x : fmaxf(x, T(0))") != 1:
        raise ValueError("Shared ReLU changed: review the HIP device contract")
    # Explicit contextual conversions preserve the shared logical semantics.
    functors = functors.replace("(a && b)", "(bool(a) && bool(b))").replace("(a || b)", "(bool(a) || bool(b))").replace("return !x;", "return !bool(x);")
    dispatch = {"f32": [], "f16": []}
    flags, evaluation, broadcast, half_broadcast, scalar_left = [], [], [], [], []
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
        flags.append(str(mode | (4 if predicate else 0)))
        if not predicate:
            expression = COMPOSITES.get(identifier)
            if expression is None:
                arguments = "x, y" if mode == 2 else "x, scalar" if mode == 1 else "x"
                expression = f"{functor}({arguments})"
            evaluation.append(f"case LFS_HIP_OP_{identifier}: return {expression};")
        if mode == 2:
            output_type = "uint8_t" if predicate else "float"
            broadcast.append(f"case LFS_HIP_OP_{identifier}: launch_broadcast<{output_type}>(context, lhs, rhs, output, a, b, out, {functor}); break;")
            half_type = "uint8_t" if predicate else "__half"
            half_broadcast.append(f"case LFS_HIP_OP_{identifier}: launch_broadcast<{half_type}, __half>(context, lhs, rhs, output, a, b, out, {functor}); break;")
        if mode == 1:
            output_type = "uint8_t" if predicate else "Storage"
            scalar_left.append(f"case LFS_HIP_OP_{identifier}: launch_pointwise<0, {output_type}, Storage>(context, lhs, {{}}, output, count, scalar, [=] __device__ (float x) {{ return {functor}(scalar, x); }}); break;")
        for storage, dtype in (("f32", "float"), ("f16", "__half")):
            output = "uint8_t" if predicate else dtype
            dispatch[storage].append(
                f"case LFS_HIP_OP_{identifier}: launch_pointwise<{mode}, {output}, {dtype}>"
                f"(context, lhs, rhs, output, count, scalar, {functor}); break;")
    if not identifiers:
        raise ValueError("The pointwise operation catalog is empty")
    return {
        "hip_pointwise_functors.hpp": functors,
        "hip_pointwise_flags.inc": ",".join(flags),
        "hip_pointwise_eval.inc": "\n".join(evaluation) + "\n",
        "hip_pointwise_broadcast.inc": "\n".join(broadcast) + "\n",
        "hip_pointwise_half_broadcast.inc": "\n".join(half_broadcast) + "\n",
        "hip_pointwise_scalar_left.inc": "\n".join(scalar_left) + "\n",
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
