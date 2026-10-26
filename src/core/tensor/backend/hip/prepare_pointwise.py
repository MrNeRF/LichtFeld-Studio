# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate HIP dispatch from the shared catalog without editing other backends."""
import argparse
from pathlib import Path
import re


HALF_OPERATIONS = frozenset({
    "AddTensor", "SubTensor", "MulTensor", "DivTensor",
    "EqualTensor", "NotEqualTensor", "LessTensor", "LessEqualTensor",
    "GreaterTensor", "GreaterEqualTensor", "LogicalAndTensor", "LogicalOrTensor",
    "LogicalXorTensor", "IsNan", "IsInf", "IsFinite", "LogicalNot",
})


def generate(source):
    lowering = (source / "pointwise_lowering.hpp").read_text(encoding="utf-8")
    aliases = dict(re.findall(r'using (pointwise_\w+_functor)\s*=\s*([^;]+);', lowering))

    def resolve(name, argument="x"):
        expression = " ".join(aliases.get(name, name).split())
        match = re.fullmatch(r'ops::scalar_right_op<(ops::\w+), \w+>', expression)
        if match:
            return f"{match[1]}{{}}({argument}, scalar)"
        match = re.fullmatch(r'ops::composed_unary_op<([\w:]+), ([\w:]+)>', expression)
        if match:
            return resolve(match[2], resolve(match[1], argument))
        if re.fullmatch(r'ops::\w+', expression):
            return f"{expression}{{}}({argument})"
        raise ValueError(f"Unsupported shared pointwise lowering: {name}")

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
        composite = "ops::composed_unary_op<" in aliases.get(functor, "")
        if composite:
            functor = "[=] __device__ (float x) { return " + resolve(functor) + "; }"
        elif identifier.endswith("Scalar"):
            alias = aliases.get(functor, "")
            match = re.fullmatch(r'ops::scalar_right_op<(ops::\w+), \w+>', " ".join(alias.split()))
            if not match or identifier[:-6].lower() != match[1].removeprefix("ops::").removesuffix("_op").replace("_", ""):
                raise ValueError(f"Unsupported scalar operation: {identifier}")
            mode, functor = 1, match[1] + "{}"
        else:
            if not functor.startswith("ops::"):
                raise ValueError(f"Unsupported pointwise functor: {functor}")
            mode = 2 if identifier.endswith("Tensor") else 0
            functor += "{}"
        predicate = identifier.startswith(("Equal", "NotEqual", "Less", "Greater", "Logical", "IsNan", "IsInf", "IsFinite"))
        flags.append(str(mode | (4 if predicate else 0)))
        if not predicate:
            arguments = "x, y" if mode == 2 else "x, scalar" if mode == 1 else "x"
            expression = f"{functor}({arguments})"
            evaluation.append(f"case LFS_HIP_OP_{identifier}: return {expression};")
        if mode == 2:
            output_type = "uint8_t" if predicate else "float"
            broadcast.append(f"case LFS_HIP_OP_{identifier}: launch_broadcast<{output_type}>(context, lhs, rhs, output, a, b, out, {functor}); break;")
            half_type = "uint8_t" if predicate else "__half"
            half_broadcast.append(f"case LFS_HIP_OP_{identifier}: launch_broadcast<{half_type}, __half>(context, lhs, rhs, output, a, b, out, {functor}); break;")
        if mode == 1 and identifier in {"AddScalar", "SubScalar", "MulScalar", "DivScalar"}:
            output_type = "uint8_t" if predicate else "Storage"
            scalar_left.append(f"case LFS_HIP_OP_{identifier}: launch_pointwise<0, {output_type}, Storage>(context, lhs, {{}}, output, count, scalar, [=] __device__ (float x) {{ return {functor}(scalar, x); }}); break;")
        for storage, dtype in (("f32", "float"), ("f16", "__half")):
            if storage == "f16" and identifier not in HALF_OPERATIONS:
                continue
            output = "uint8_t" if predicate else dtype
            dispatch[storage].append(
                f"case LFS_HIP_OP_{identifier}: launch_pointwise<{mode}, {output}, {dtype}>"
                f"(context, lhs, rhs, output, count, scalar, {functor}); break;")
    if not identifiers:
        raise ValueError("The pointwise operation catalog is empty")
    return {
        "hip_pointwise_functors.hpp": '#pragma once\n#include "tensor_functors.hpp"\n',
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
