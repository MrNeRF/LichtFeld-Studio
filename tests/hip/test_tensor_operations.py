# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Native Tensor operator contracts against independent small CPU references."""
import argparse
from contextlib import contextmanager
import ctypes as c
import math
import os
from pathlib import Path
import struct
import unittest

import test_native_runtime as native

View = native.View
Error = native.Error
FORMATS = ("f", "e", "i", "q", "B", "B", "I")


class Layout(c.Structure):
    _fields_ = [("rank", c.c_uint32), ("dims", c.c_uint64 * 8),
                ("strides", c.c_uint64 * 8), ("elements", c.c_uint64)]


def layout(dims, strides=None):
    if strides is None:
        strides, step = [], 1
        for dim in reversed(dims):
            strides.insert(0, step)
            step *= dim
    return Layout(len(dims), (c.c_uint64 * 8)(*dims),
                  (c.c_uint64 * 8)(*strides), math.prod(dims))


class LayoutCopy(c.Structure):
    _fields_ = [("input", View), ("output", View), ("input_layout", Layout),
                ("output_layout", Layout), ("element_bytes", c.c_uint32)]


class MemoryInfo(c.Structure):
    _fields_ = [(name, c.c_uint64) for name in ("total", "free", "owned", "workspace")]


class ChainOp(c.Structure):
    _fields_ = [("op", c.c_uint32), ("scalar", c.c_float), ("rhs", View)]


class Chain(c.Structure):
    _fields_ = [("count", c.c_uint32), ("ops", ChainOp * 16)]


class Random(c.Structure):
    _fields_ = [("kind", c.c_uint32), ("seed", c.c_uint64), ("count", c.c_size_t),
                ("first", c.c_float), ("second", c.c_float), ("low", c.c_int32), ("high", c.c_int32)]


class Predicate(c.Structure):
    _fields_ = [("struct_bytes", c.c_uint32), ("op", c.c_uint32), ("dtype", c.c_uint32),
                ("scalar_kind", c.c_uint32), ("count", c.c_uint64), ("integer_scalar", c.c_int64),
                ("float_scalar", c.c_float), ("scalar_left", c.c_uint32)]


class Gemm(c.Structure):
    _fields_ = [("struct_bytes", c.c_uint32), ("transpose_b", c.c_uint32), ("relu", c.c_uint32),
                ("batch", c.c_uint64), ("m", c.c_uint64), ("n", c.c_uint64), ("k", c.c_uint64)]


class Linear(c.Structure):
    _fields_ = [(name, c.c_uint64) for name in ("batch", "m", "n", "k")] + [(name, c.c_uint32) for name in ("dtype", "transpose_b", "batched_b", "activation")]


class Where(c.Structure):
    _fields_ = [("struct_bytes", c.c_uint32), ("dtype", c.c_uint32),
                ("condition", Layout), ("x", Layout), ("y", Layout), ("output", Layout)]


class Selection(c.Structure):
    _fields_ = [("struct_bytes", c.c_uint32), ("dtype", c.c_uint32), ("kind", c.c_uint32),
                ("count_known", c.c_uint32), ("count", c.c_uint64),
                ("capacity", c.c_uint64), ("selected_count", c.c_uint64)]


class Gather(c.Structure):
    _fields_ = [(name, c.c_uint32) for name in ("struct_bytes", "dtype", "index_dtype", "dim", "boundary_mode", "index_select")] + [("input_layout", Layout), ("index_layout", Layout)]


class IndexUpdate(c.Structure):
    _fields_ = [(name, c.c_uint32) for name in ("struct_bytes", "dtype", "index_dtype", "dim", "add")] + [("index_count", c.c_uint64), ("output_layout", Layout)]


class IndexFill(c.Structure):
    _fields_ = [(name, c.c_uint32) for name in ("struct_bytes", "dtype", "index_dtype", "dim")] + [("index_count", c.c_uint64), ("value_bits", c.c_uint64), ("output_layout", Layout)]


class TensorOperations(native.RuntimeContract):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        P, U, Z = c.c_void_p, c.c_uint32, c.c_size_t
        E, L = c.POINTER(Error), c.POINTER(Layout)
        signatures = {
            "scalar_left_typed": [P, U, View, View, Z, c.c_float, U, E],
            "broadcast_typed": [P, U, View, View, View, L, L, L, U, U, E],
            "convert": [P, View, U, View, U, Z, E],
            "copy_layouts": [P, c.POINTER(LayoutCopy), Z, E],
            "strided_copy": [P, View, View, L, U, E],
            "strided_scatter": [P, View, View, L, U, U, E],
            "fill_layout_bits": [P, View, L, U, c.c_uint64, E],
            "reduce_f32": [P, View, Z, U, c.POINTER(c.c_float), E],
            "reduce_layout_f32": [P, View, View, L, U, U, E],
            "reduce_typed_layout": [P, View, View, L, U, U, U, U, E],
            "arg_extreme_f32": [P, View, View, View, Z, Z, Z, U, E],
            "chain_f32": [P, View, View, Z, c.POINTER(Chain), E],
            "chain_reduce_f32": [P, View, View, Z, Z, c.POINTER(Chain), U, E],
            "integer_pointwise": [P, View, View, View, c.POINTER(Predicate), E],
            "predicate_typed": [P, View, View, View, c.POINTER(Predicate), E],
            "clamp_i32": [P, View, Z, c.c_int32, c.c_int32, E],
            "cumsum": [P, View, U, L, U, E],
            "sort_pairs": [P, View, View, Z, Z, Z, U, E],
            "select": [P, View, View, View, c.POINTER(Selection), c.POINTER(c.c_uint64), E],
            "masked_scatter": [P, View, View, View, U, Z, Z, E],
            "where_layout": [P, View, View, View, View, c.POINTER(Where), E],
            "masked_fill": [P, View, View, c.c_uint64, U, c.c_float, E],
            "where_scalar": [P, View, View, View, Z, U, c.c_float, E],
            "random": [P, View, c.POINTER(Random), E],
            "count_nonzero": [P, View, U, Z, c.POINTER(c.c_uint64), E],
            "index_cast": [P, View, View, Z, Z, E],
            "gather": [P, View, View, View, c.POINTER(Gather), E],
            "gather_unary_f32": [P, View, View, U, View, Z, Z, U, E],
            "index_put_f32": [P, View, View, U, View, Z, Z, E],
            "index_update": [P, View, View, View, c.POINTER(IndexUpdate), E],
            "index_fill": [P, View, View, c.POINTER(IndexFill), E],
            "gemm_f32": [P, View, View, View, View, c.POINTER(Gemm), E],
            "nn_linear": [P, c.POINTER(View), View, c.POINTER(Linear), E],
            "cdist_f32": [P, View, View, View, Z, Z, Z, c.c_float, E],
            "memory_info": [P, c.POINTER(MemoryInfo), E],
            "trim_workspace": [P, E],
            "multinomial": [P, View, View, Z, Z, c.c_uint64, U, E],
        }
        cls.calls.update({name: native.bind(cls.dll, name, args) for name, args in signatures.items()})
        cls.operations = native.pointwise_operations()

    def test_exported_operator_boundary_without_gpu_queries(self):
        self.assertGreaterEqual(len(self.calls), 60)
        self.assertEqual(self.calls["runtime_abi"](), native.EXPECTED_ABI)

    @contextmanager
    def resources(self):
        context = c.c_void_p()
        self.checked("context_create", native.options.device, c.byref(context))
        try:
            yield context
        finally:
            self.checked("context_destroy", context)

    def tensor(self, context, values, dtype=0, *, extra=0):
        values = list(values)
        data = b"".join(struct.pack("<" + FORMATS[dtype], value) for value in values)
        handle = c.c_void_p()
        self.checked("buffer_allocate", context, max(1, len(data) + extra), c.byref(handle))
        if data:
            self.checked("upload", context, handle, 0, c.create_string_buffer(data), len(data))
        return View(handle, 0)

    def read(self, context, view, count, dtype=0):
        data = (c.c_ubyte * (count * struct.calcsize(FORMATS[dtype])))()
        self.checked("download", context, view.buffer, view.offset, data, len(data))
        return [row[0] for row in struct.iter_unpack("<" + FORMATS[dtype], bytes(data))]

    def reject(self, name, *arguments):
        error = Error()
        self.assertNotEqual(self.calls[name](*arguments, c.byref(error)), 0)
        self.assertTrue(error.message)
        return error

    def gather_checked(self, context, source, dtype, indices, index_dtype, output, source_layout, index_layout, dim, boundary, select):
        p = Gather(c.sizeof(Gather), dtype, index_dtype, dim, boundary, select, c.cast(source_layout, c.POINTER(Layout)).contents, c.cast(index_layout, c.POINTER(Layout)).contents)
        self.checked("gather", context, source, indices, output, c.byref(p))

    def index_update_checked(self, context, output, dtype, indices, index_dtype, source, shape, dim, count, add):
        p = IndexUpdate(c.sizeof(IndexUpdate), dtype, index_dtype, dim, add, count, c.cast(shape, c.POINTER(Layout)).contents)
        self.checked("index_update", context, output, indices, source, c.byref(p))

    def index_fill_checked(self, context, output, dtype, indices, index_dtype, shape, dim, count, bits):
        p = IndexFill(c.sizeof(IndexFill), dtype, index_dtype, dim, count, bits, c.cast(shape, c.POINTER(Layout)).contents)
        self.checked("index_fill", context, output, indices, c.byref(p))

    def select_checked(self, context, source, mask, output, dtype, kind, count, capacity, selected, *, known=None):
        p = Selection(c.sizeof(Selection), dtype, kind, known is not None, count, capacity, known or 0)
        self.checked("select", context, source, mask, output, c.byref(p), selected)

    def select_reject(self, context, source, mask, output, dtype, kind, count, capacity, selected):
        p = Selection(c.sizeof(Selection), dtype, kind, 0, count, capacity, 0)
        return self.reject("select", context, source, mask, output, c.byref(p), selected)

    def fill_float(self, action, context, output, shape, value):
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
        return getattr(self, action)("fill_layout_bits", context, output, shape, 4, bits)

    def test_all_conversion_pairs_and_scalar_left(self):
        with self.resources() as context:
            for input_dtype in range(7):
                values = [0, 1] if input_dtype == 5 else [0, 1, 3, 17]
                source = self.tensor(context, values, input_dtype)
                for output_dtype in range(7):
                    with self.subTest(source=input_dtype, destination=output_dtype):
                        output = self.tensor(context, [99] * len(values), output_dtype)
                        targets = {0: {0, 1, 2, 3, 4, 6}, 1: {0, 1, 2, 3, 4}, 2: {0, 1, 2, 3, 4},
                                   3: {0, 1, 2, 3, 4}, 4: {0, 1, 2, 3, 4, 5}, 5: {0, 1, 2, 3, 4, 5}, 6: {0, 3, 6}}
                        if output_dtype not in targets[input_dtype]:
                            self.reject("convert", context, source, input_dtype, output, output_dtype, len(values))
                            self.assertEqual(self.read(context, output, len(values), output_dtype), [99] * len(values))
                            continue
                        self.checked("convert", context, source, input_dtype, output, output_dtype, len(values))
                        expected = [int(bool(x)) for x in values] if output_dtype == 5 else values
                        self.assertEqual(self.read(context, output, len(values), output_dtype), expected)
            for dtype in (0,):
                source = self.tensor(context, [1, 2, 4], dtype)
                output = self.tensor(context, [0, 0, 0], dtype)
                self.checked("scalar_left_typed", context, self.operations["DivScalar"], source, output, 3, 8, dtype)
                self.assertEqual(self.read(context, output, 3, dtype), [8, 4, 2])
            source = self.tensor(context, [-257.9, -1.9, 256.9, float("nan"), float("inf")])
            output = self.tensor(context, [77] * 5, 4)
            self.checked("convert", context, source, 0, output, 4, 5)
            self.assertEqual(self.read(context, output, 5, 4), [255, 255, 0, 0, 0])

    def test_layout_broadcast_copy_scatter_and_fill(self):
        with self.resources() as context:
            a, b, result = layout([2, 1]), layout([1, 3]), layout([2, 3])
            for dtype in range(7):
                left, right = self.tensor(context, [1, 0], dtype), self.tensor(context, [1, 0, 1], dtype)
                output = self.tensor(context, [0] * 6, dtype)
                operation = "LogicalOrTensor" if dtype == 5 else "AddTensor"
                width = struct.calcsize(FORMATS[dtype])
                self.checked("broadcast_typed", context, self.operations[operation], left, right, output,
                             c.byref(a), c.byref(b), c.byref(result), dtype, width)
                self.assertEqual(self.read(context, output, 6, dtype), [1, 1, 1, 1, 0, 1] if dtype == 5 else [2, 1, 2, 1, 0, 1])
            source = self.tensor(context, range(6))
            output = self.tensor(context, [99] * 6)
            transposed = layout([3, 2], [1, 3])
            self.checked("strided_copy", context, source, output, c.byref(transposed), 4)
            self.assertEqual(self.read(context, output, 6), [0, 3, 1, 4, 2, 5])
            dense = layout([3, 2])
            self.checked("strided_scatter", context, output, source, c.byref(transposed), 4, 0)
            self.assertEqual(self.read(context, source, 6), list(range(6)))
            copies = (LayoutCopy * 1)(LayoutCopy(source, output, transposed, dense, 4))
            self.checked("copy_layouts", context, copies, 1)
            self.assertEqual(self.read(context, output, 6), [0, 3, 1, 4, 2, 5])
            holes = self.tensor(context, [99] * 8)
            sparse = layout([2, 2], [4, 1])
            self.fill_float("checked", context, holes, c.byref(sparse), 7)
            self.assertEqual(self.read(context, holes, 8), [7, 7, 99, 99, 7, 7, 99, 99])
            bits = struct.unpack("<I", struct.pack("<f", 3))[0]
            self.checked("fill_layout_bits", context, holes, c.byref(sparse), 4, bits)
            self.assertEqual(self.read(context, holes, 8), [3, 3, 99, 99, 3, 3, 99, 99])
            self.reject("broadcast_typed", context, self.operations["AddTensor"], source, output, holes,
                        c.byref(layout([2])), c.byref(layout([3])), c.byref(layout([3])), 0, 4)
            self.fill_float("reject", context, holes, c.byref(layout([2, 2], [0, 1])), 8)

    def test_integer_unary_functors_preserve_precision_and_wraparound(self):
        with self.resources() as context:
            for dtype, values in ((2, [-2**31, -1, 0, 2**31-1]), (3, [-2**63, -1, 0, 2**63-1]),
                                  (4, [0, 1, 127, 255]), (6, [0, 1, 2**31, 2**32-1])):
                source = self.tensor(context, values, dtype)
                output = self.tensor(context, [0] * len(values), dtype)
                bits = 8 * struct.calcsize(FORMATS[dtype])
                signed = dtype in (2, 3)
                def wrap(value):
                    value %= 1 << bits
                    return value - (1 << bits) if signed and value >= 1 << (bits - 1) else value
                for operation, reference in (("Abs", abs), ("Neg", lambda x: -x), ("Square", lambda x: x*x),
                                             ("Relu", lambda x: max(x, 0)), ("Sign", lambda x: (x > 0) - (x < 0))):
                    p = Predicate(c.sizeof(Predicate), self.operations[operation], dtype, 0, len(values))
                    self.checked("integer_pointwise", context, source, View(), output, c.byref(p))
                    self.assertEqual(self.read(context, output, len(values), dtype), [wrap(reference(x)) for x in values])

    def test_reduction_dispatch_thresholds(self):
        with self.resources() as context:
            for size in (1, 31, 32, 33, 257, 4097, 10001):
                values = [(i % 5 - 2) * 0.125 for i in range(2 * size)]
                shape = layout([2, size])
                for dtype in (0, 1):
                    source = self.tensor(context, values, dtype)
                    output = self.tensor(context, [0, 0], dtype)
                    for operation, reference in ((0, sum), (1, lambda x: sum(x)/len(x)), (2, min), (3, max)):
                        if dtype:
                            self.checked("reduce_typed_layout", context, source, output, c.byref(shape), 2, operation, 1, 1)
                        else:
                            self.checked("reduce_layout_f32", context, source, output, c.byref(shape), 2, operation)
                        expected = [reference(values[r*size:(r+1)*size]) for r in range(2)]
                        if dtype:
                            expected = [struct.unpack("<e", struct.pack("<e", x))[0] for x in expected]
                        for actual, wanted in zip(self.read(context, output, 2, dtype), expected):
                            self.assertAlmostEqual(actual, wanted, delta=1e-7)
                large = [2**53 + 1 + i % 2 for i in range(2 * size)]
                source = self.tensor(context, large, 3)
                output = self.tensor(context, [0, 0], 3)
                for operation, reference in ((0, sum), (2, min), (3, max)):
                    self.checked("reduce_typed_layout", context, source, output, c.byref(shape), 2, operation, 3, 3)
                    expected = [reference(large[r*size:(r+1)*size]) for r in range(2)]
                    expected = [(x + 2**63) % 2**64 - 2**63 for x in expected]
                    self.assertEqual(self.read(context, output, 2, 3), expected)

    def test_reductions_chains_and_workspace_lifetime(self):
        with self.resources() as context:
            values = [float(i % 11 - 5) for i in range(8201)]
            source = self.tensor(context, values)
            result = c.c_float()
            for operation, expected in enumerate((sum(values), sum(values) / len(values), min(values), max(values), 0, 0)):
                self.checked("reduce_f32", context, source, len(values), operation, c.byref(result))
                self.assertAlmostEqual(result.value, expected, delta=2e-5)
            matrix = self.tensor(context, [1, 2, 3, 4, 5, 6])
            output = self.tensor(context, [0, 0])
            shape = layout([2, 3])
            self.checked("reduce_layout_f32", context, matrix, output, c.byref(shape), 2, 0)
            self.assertEqual(self.read(context, output, 2), [6, 15])
            for dtype in (2, 3, 4, 5, 6):
                integers = self.tensor(context, [1, 0, 1, 1, 1, 1], dtype)
                reduced = self.tensor(context, [0, 0], 3)
                self.checked("reduce_typed_layout", context, integers, reduced, c.byref(shape), 2, 0, dtype, 3)
                self.assertEqual(self.read(context, reduced, 2, 3), [2, 3])
            chain = Chain(2, (ChainOp * 16)(ChainOp(self.operations["MulScalar"], 2, View()),
                                          ChainOp(self.operations["AddScalar"], 1, View())))
            transformed = self.tensor(context, [0] * 6)
            self.checked("chain_f32", context, matrix, transformed, 6, c.byref(chain))
            self.assertEqual(self.read(context, transformed, 6), [3, 5, 7, 9, 11, 13])
            self.checked("chain_reduce_f32", context, matrix, output, 2, 3, c.byref(chain), 0)
            self.assertEqual(self.read(context, output, 2), [15, 33])
            wide = self.tensor(context, [1] * 20000)
            self.checked("chain_reduce_f32", context, wide, output, 2, 10000, c.byref(chain), 0)
            self.assertEqual(self.read(context, output, 2), [30000, 30000])
            self.checked("chain_reduce_f32", context, matrix, output, 2, 0, c.byref(chain), 0)
            self.assertEqual(self.read(context, output, 2), [0, 0])
            self.checked("chain_reduce_f32", context, matrix, output, 2, 0, c.byref(chain), 1)
            self.assertTrue(all(math.isnan(x) for x in self.read(context, output, 2)))

            info = MemoryInfo()
            self.checked("memory_info", context, c.byref(info))
            self.assertGreater(info.workspace, 0)
            self.checked("trim_workspace", context)
            self.checked("memory_info", context, c.byref(info))
            self.assertEqual(info.workspace, 0)

    def byte_predicate(self, action, context, operation, source, output, count, scalar, dtype=4):
        p = Predicate(c.sizeof(Predicate), self.operations[operation], dtype, 0, count, 0, scalar, 0)
        getattr(self, action)("predicate_typed", context, source, View(), output, c.byref(p))

    def test_byte_scalar_cast_matches_typed_comparison(self):
        with self.resources() as context:
            for dtype, values, scalar, expected in (
                (4, [0, 2, 3], 2.5, [1, 0, 0]),
                (4, [0, 1, 2], 0.5, [0, 0, 0]),
                (5, [0, 1, 1], 0.5, [1, 0, 0]),
            ):
                source = self.tensor(context, values, dtype)
                output = self.tensor(context, [99] * 3, 4)
                p = Predicate(c.sizeof(Predicate), self.operations["LessScalar"], dtype, 0, 3, 0, scalar, 0)
                self.checked("predicate_typed", context, source, View(), output, c.byref(p))
                self.assertEqual(self.read(context, output, 3, 4), expected)
            source = self.tensor(context, [0, 1, 2], 4)
            self.byte_predicate("checked", context, "EqualScalar", source, output, 3, 0.5)
            self.assertEqual(self.read(context, output, 3, 4), [1, 0, 0])

    def test_integer_precision_predicates_and_boolean_masks(self):
        with self.resources() as context:
            for dtype, values in ((2, [-3, 0, 7]), (3, [2**53 + 1, 2**53 + 2, -2**53 - 1]), (6, [0, 2**31, 2**32 - 1]), (4, [0, 127, 255])):
                source = self.tensor(context, values, dtype)
                output = self.tensor(context, [0] * 3, dtype)
                p = Predicate(c.sizeof(Predicate), self.operations["AddScalar"], dtype, 1, 3, 1, 0, 0)
                self.checked("integer_pointwise", context, source, View(), output, c.byref(p))
                bits = 8 * struct.calcsize(FORMATS[dtype])
                signed = dtype in (2, 3)
                expected = []
                for value in values:
                    v = (value + 1) % (1 << bits)
                    expected.append(v - (1 << bits) if signed and v >= 1 << (bits - 1) else v)
                self.assertEqual(self.read(context, output, 3, dtype), expected)
            source = self.tensor(context, [2**53, 2**53 + 1, 2**53 + 2], 3)
            mask = self.tensor(context, [0] * 3, 4)
            p = Predicate(c.sizeof(Predicate), self.operations["EqualScalar"], 3, 1, 3, 2**53 + 1, 0, 0)
            self.checked("predicate_typed", context, source, View(), mask, c.byref(p))
            self.assertEqual(self.read(context, mask, 3, 4), [0, 1, 0])
            boolean = self.tensor(context, [0, 1, 1], 5)
            self.byte_predicate("checked", context, "LogicalNot", boolean, mask, 3, 0, dtype=5)
            self.assertEqual(self.read(context, mask, 3, 4), [1, 0, 0])
            integers = self.tensor(context, [-9, 0, 9], 2)
            self.checked("clamp_i32", context, integers, 3, -2, 4)
            self.assertEqual(self.read(context, integers, 3, 2), [-2, 0, 4])

    def test_scalar_left_integer_and_half_predicate_edges(self):
        with self.resources() as context:
            for dtype in (2, 3, 6, 4):
                source = self.tensor(context, [1, 2, 4], dtype)
                output = self.tensor(context, [0] * 3, dtype)
                p = Predicate(c.sizeof(Predicate), self.operations["SubScalar"], dtype, 1, 3, 8, 0, 1)
                self.checked("integer_pointwise", context, source, View(), output, c.byref(p))
                self.assertEqual(self.read(context, output, 3, dtype), [7, 6, 4])
                if dtype in (2, 3, 6):
                    mask = self.tensor(context, [99] * 3, 4)
                    p.op, p.integer_scalar = self.operations["LessScalar"], 2
                    self.checked("predicate_typed", context, source, View(), mask, c.byref(p))
                    self.assertEqual(self.read(context, mask, 3, 4), [0, 0, 1])
            half = self.tensor(context, [-0.0, -2, float("nan"), float("inf"), -float("inf")], 1)
            mask = self.tensor(context, [99] * 5, 4)
            for name, expected in (("IsNan", [0, 0, 1, 0, 0]), ("IsInf", [0, 0, 0, 1, 1]), ("IsFinite", [1, 1, 0, 0, 0])):
                p = Predicate(c.sizeof(Predicate), self.operations[name], 1, 0, 5)
                self.checked("predicate_typed", context, half, View(), mask, c.byref(p))
                self.assertEqual(self.read(context, mask, 5, 4), expected)

    def test_noncontiguous_multi_axis_reduction_and_long_strided_scans(self):
        with self.resources() as context:
            source = self.tensor(context, range(24))
            output = self.tensor(context, [99, 99])
            shape = layout([3, 2, 4], [4, 12, 1])
            self.checked("reduce_layout_f32", context, source, output, c.byref(shape), 5, 0)
            self.assertEqual(self.read(context, output, 2), [66, 210])
            wide = self.tensor(context, [1, 2] * 10001)
            shape = layout([10001, 2])
            self.checked("reduce_layout_f32", context, wide, output, c.byref(shape), 1, 0)
            self.assertEqual(self.read(context, output, 2), [10001, 20002])
            for dtype in (0, 2, 3, 6):
                source = self.tensor(context, [1, 2] * 513, dtype)
                shape = layout([2, 513], [1, 2])
                self.checked("cumsum", context, source, dtype, c.byref(shape), 1)
                expected = [value for i in range(1, 514) for value in (i, 2 * i)]
                self.assertEqual(self.read(context, source, 1026, dtype), expected)

    def test_scan_sort_and_stable_selection(self):
        with self.resources() as context:
            for dtype in (0, 2, 3, 6):
                source = self.tensor(context, [1, 2, 3, 4, 5, 6], dtype)
                shape = layout([2, 3])
                self.checked("cumsum", context, source, dtype, c.byref(shape), 1)
                self.assertEqual(self.read(context, source, 6, dtype), [1, 3, 6, 4, 9, 15])
            for outer, dimension, inner in ((1, 6, 1), (2, 3, 1), (1, 3, 2)):
                original = [3, 1, 1, 8, -2, 4]
                source = self.tensor(context, original)
                indices = self.tensor(context, [99] * 6, 3)
                self.checked("sort_pairs", context, source, indices, outer, dimension, inner, 0)
                actual, order = self.read(context, source, 6), self.read(context, indices, 6, 3)
                for batch in range(outer):
                    for lane in range(inner):
                        positions = [(batch * dimension + j) * inner + lane for j in range(dimension)]
                        expected = sorted(range(dimension), key=lambda j: original[positions[j]])
                        self.assertEqual([order[x] for x in positions], expected)
                        self.assertEqual([actual[x] for x in positions], [original[positions[j]] for j in expected])
            source = self.tensor(context, [7, 9, 4, 8, 1])
            mask = self.tensor(context, [1, 0, 1, 0, 1], 4)
            output = self.tensor(context, [99] * 5)
            selected = c.c_uint64()
            self.select_checked(context, source, mask, output, 0, 0, 5, 5, c.byref(selected))
            self.assertEqual(selected.value, 3)
            self.assertEqual(self.read(context, output, 3), [7, 4, 1])
            positions = self.tensor(context, [99] * 5, 3)
            self.select_checked(context, mask, View(), positions, 4, 1, 5, 5, c.byref(selected))
            self.assertEqual(self.read(context, positions, 3, 3), [0, 2, 4])
            replacement = self.tensor(context, [10, 20, 30])
            self.checked("masked_scatter", context, source, mask, replacement, 0, 5, 3)
            self.assertEqual(self.read(context, source, 5), [10, 9, 20, 8, 30])
            self.checked("masked_fill", context, mask, source, 5, 0, -2)
            self.assertEqual(self.read(context, source, 5), [-2, 9, -2, 8, -2])

    def test_precounted_selection_and_scatter_count_errors(self):
        with self.resources() as context:
            mask = self.tensor(context, [1, 0, 1, 0, 1], 4)
            source = self.tensor(context, [3, 8, 5, 9, 7])
            result = self.tensor(context, [99] * 4)
            selected = c.c_uint64(99)
            self.select_checked(context, source, mask, result, 0, 0, 5, 3, c.byref(selected), known=3)
            self.assertEqual(selected.value, 3)
            self.assertEqual(self.read(context, result, 4), [3, 5, 7, 99])
            positions = self.tensor(context, [99] * 4, 3)
            self.select_checked(context, mask, View(), positions, 4, 1, 5, 3, c.byref(selected), known=3)
            self.assertEqual(self.read(context, positions, 4, 3), [0, 2, 4, 99])

    def test_invalid_scatter_count_is_asynchronous_and_preserves_output(self):
        with self.resources() as context:
            output = self.tensor(context, [99] * 5)
            mask = self.tensor(context, [1, 0, 1, 0, 1], 4)
            source = self.tensor(context, [3, 5])
            self.checked("masked_scatter", context, output, mask, source, 0, 5, 2)
            self.reject("context_synchronize", context)
            self.assertEqual(self.read(context, output, 5), [99] * 5)
            source = self.tensor(context, [3, 5, 7])
            self.checked("masked_scatter", context, output, mask, source, 0, 5, 3)
            self.assertEqual(self.read(context, output, 5), [3, 99, 5, 99, 7])

    def test_segmented_sort_preserves_payloads_and_stable_special_value_order(self):
        payloads = [0, 0x80000000, 0x7FC00001, 0xFFC01234, 0x3F800000, 0xBF800000, 0x80000000, 0]
        long_values = [float("nan"), -float("inf"), -0.0, 0.0, float("inf"), 3.0, -2.0] * 151
        long_bits = [struct.unpack("<I", struct.pack("<f", x))[0] for x in long_values]
        with self.resources() as context:
            for bits, outer in ((payloads, 1), (long_bits, 2)):
                values = [struct.unpack("<f", struct.pack("<I", x))[0] for x in bits]
                for descending in (0, 1):
                    with self.subTest(segments=outer, descending=descending):
                        source = self.tensor(context, bits * outer, 6)
                        indices = self.tensor(context, [99] * (len(bits) * outer), 3)
                        self.checked("sort_pairs", context, source, indices, outer, len(bits), 1, descending)
                        key = lambda i: (math.isnan(values[i]), 0 if math.isnan(values[i]) else values[i])
                        expected = sorted(range(len(bits)), key=key, reverse=bool(descending))
                        self.assertEqual(self.read(context, indices, len(bits) * outer, 3), expected * outer)
                        self.assertEqual(self.read(context, source, len(bits) * outer, 6), [bits[i] for i in expected] * outer)

    def test_selection_preserves_all_storage_bits(self):
        with self.resources() as context:
            for dtype in range(7):
                values = [0, 1, 0, 1] if dtype == 5 else [0, 3, 0, 7]
                source = self.tensor(context, values, dtype)
                mask = self.tensor(context, [0, 1, 0, 1], 4)
                output = self.tensor(context, [0, 0], dtype)
                selected = c.c_uint64()
                self.select_checked(context, source, mask, output, dtype, 0, 4, 2, c.byref(selected))
                self.assertEqual(self.read(context, output, 2, dtype), [values[1], values[3]])
            bits = [0xffc12345, 0x80000000, 0x7fc54321, 0]
            source = self.tensor(context, bits, 6)
            output = self.tensor(context, [0, 0], 6)
            mask = self.tensor(context, [1, 1, 0, 0], 4)
            self.select_checked(context, source, mask, output, 0, 0, 4, 2, c.byref(selected))
            self.assertEqual(self.read(context, output, 2, 6), bits[:2])

    def test_sparse_selection_retains_no_full_result_scratch(self):
        with self.resources() as context:
            size = 100003
            source = self.tensor(context, [float(i) for i in range(size)])
            mask = self.tensor(context, [1] + [0] * (size - 2) + [1], 4)
            output = self.tensor(context, [99, 99])
            selected = c.c_uint64()
            self.select_checked(context, source, mask, output, 0, 0, size, 2, c.byref(selected))
            self.assertEqual(selected.value, 2)
            self.assertEqual(self.read(context, output, 2), [0, size - 1])
            info = MemoryInfo()
            self.checked("memory_info", context, c.byref(info))
            self.assertLess(info.workspace, size * 4)

    def test_reduction_opcodes_are_shared_across_dtypes(self):
        with self.resources() as context:
            for dtype in (1, 2, 3, 4, 6):
                source = self.tensor(context, [3, 0, 7, 2], dtype)
                for operation, expected in ((2, 0), (3, 7)):
                    output = self.tensor(context, [99], dtype)
                    self.checked("reduce_typed_layout", context, source, output, c.byref(layout([4])), 1, operation, dtype, dtype)
                    self.assertEqual(self.read(context, output, 1, dtype), [expected])
                for operation, expected in ((6, 1), (7, 0)):
                    output = self.tensor(context, [99], 5)
                    self.checked("reduce_typed_layout", context, source, output, c.byref(layout([4])), 1, operation, dtype, 5)
                    self.assertEqual(self.read(context, output, 1, 5), [expected])

    def test_selection_capacity_and_all_dtype_nonzero(self):
        with self.resources() as context:
            count = c.c_uint64(123)
            for dtype in range(7):
                values = [0, 1, 0, 1] if dtype == 5 else [0, 3, 0, 7]
                source = self.tensor(context, values, dtype)
                output = self.tensor(context, [99] * 4, 3)
                self.checked("count_nonzero", context, source, dtype, 4, c.byref(count))
                self.assertEqual(count.value, 2)
                self.select_checked(context, source, View(), output, dtype, 1, 4, 4, c.byref(count))
                self.assertEqual(self.read(context, output, 2, 3), [1, 3])
                output = self.tensor(context, [99], 3)
                self.select_reject(context, source, View(), output, dtype, 1, 4, 1, c.byref(count))
                self.assertEqual(count.value, 0)
                self.assertEqual(self.read(context, output, 1, 3), [99])

    def test_where_broadcast_preserves_integer_precision_and_inplace_values(self):
        with self.resources() as context:
            for dtype in range(7):
                high = 2**53 + 1 if dtype == 3 else 1
                condition = self.tensor(context, [1, 0, 1], 5)
                x = self.tensor(context, [high, 0], dtype)
                y = self.tensor(context, [0, 1, 0], dtype)
                output = self.tensor(context, [0] * 6, dtype)
                p = Where(c.sizeof(Where), dtype, layout([1, 3]), layout([2, 1]), layout([1, 3]), layout([2, 3]))
                self.checked("where_layout", context, condition, x, y, output, c.byref(p))
                self.assertEqual(self.read(context, output, 6, dtype), [high, 1, high, 0, 1, 0])
            mask = self.tensor(context, [1, 0, 1], 5)
            for dtype in (0, 1):
                source = self.tensor(context, [2, 4, 6], dtype)
                self.checked("where_scalar", context, mask, source, source, 3, dtype, -5)
                self.assertEqual(self.read(context, source, 3, dtype), [-5, 4, -5])

    def test_arg_extreme_nan_ties_and_partitioned_axes(self):
        with self.resources() as context:
            for reduced in (3, 10001):
                for maximum in (0, 1):
                    original = [1] * (2 * reduced)
                    original[1] = original[2] = 9 if maximum else -9
                    original[reduced + 1] = float("nan")
                    source = self.tensor(context, original)
                    values, indices = self.tensor(context, [99, 99]), self.tensor(context, [99, 99], 3)
                    self.checked("arg_extreme_f32", context, source, values, indices, 2, reduced, 1, maximum)
                    self.assertEqual(self.read(context, indices, 2, 3), [1, 1])
                    actual = self.read(context, values, 2)
                    self.assertEqual(actual[0], 9 if maximum else -9)
                    self.assertTrue(math.isnan(actual[1]))

    def test_reduction_nan_empty_axes_and_boolean_identities(self):
        with self.resources() as context:
            source = self.tensor(context, [1, float("nan"), float("inf")])
            result = c.c_float()
            for operation in range(6):
                self.checked("reduce_f32", context, source, 3, operation, c.byref(result))
                self.assertTrue(math.isnan(result.value) if operation < 4 else result.value == 1)
            empty = self.tensor(context, [])
            for operation, expected in ((0, 0), (4, 0), (5, 0)):
                self.checked("reduce_f32", context, empty, 0, operation, c.byref(result))
                self.assertEqual(result.value, expected)
            self.checked("reduce_f32", context, empty, 0, 1, c.byref(result))
            self.assertTrue(math.isnan(result.value))
            self.reject("reduce_f32", context, empty, 0, 2, c.byref(result))
            shape = layout([2, 0])
            for operation, expected in ((6, [0, 0]), (7, [1, 1])):
                output = self.tensor(context, [99, 99], 5)
                self.checked("reduce_typed_layout", context, empty, output, c.byref(shape), 2, operation, 5, 5)
                self.assertEqual(self.read(context, output, 2, 5), expected)
            half = self.tensor(context, [1, 2, 3, 4, 5, 6], 1)
            output = self.tensor(context, [0, 0], 1)
            shape = layout([2, 3])
            self.checked("reduce_typed_layout", context, half, output, c.byref(shape), 2, 1, 1, 1)
            self.assertEqual(self.read(context, output, 2, 1), [2, 5])

    def test_invalid_descriptor_batch_preserves_all_outputs(self):
        with self.resources() as context:
            source = self.tensor(context, [1, 2, 3, 4])
            output = self.tensor(context, [99] * 4)
            good = LayoutCopy(source, output, layout([4]), layout([4]), 4)
            bad = LayoutCopy(source, output, layout([4]), layout([4], [0]), 4)
            self.reject("copy_layouts", context, (LayoutCopy * 2)(good, bad), 2)
            self.assertEqual(self.read(context, output, 4), [99] * 4)
            huge = layout([2, 2], [c.c_uint64(-1).value, 1])
            self.reject("strided_copy", context, source, output, c.byref(huge), 4)
            self.reject("chain_f32", context, source, output, 4, c.byref(Chain(17)))
            self.reject("random", context, output, c.byref(Random(1, 0, 4, 1.1, 1, 0, 0)))
            self.reject("gemm_f32", context, source, source, View(), output, c.byref(Gemm(0, 0, 0, 1, 2, 2, 2)))
            self.assertEqual(self.read(context, output, 4), [99] * 4)

    def test_indexing_and_asynchronous_invalid_indices(self):
        with self.resources() as context:
            source = self.tensor(context, [10, 20, 30, 40, 50, 60])
            indices = self.tensor(context, [2, 0], 3)
            output = self.tensor(context, [99] * 4)
            matrix, index_layout = layout([2, 3]), layout([2])
            self.gather_checked(context, source, 0, indices, 3, output,
                         c.byref(matrix), c.byref(index_layout), 1, 0, 1)
            self.assertEqual(self.read(context, output, 4), [30, 10, 60, 40])
            invalid = self.tensor(context, [0, 5], 3)
            self.gather_checked(context, source, 0, invalid, 3, output,
                         c.byref(matrix), c.byref(index_layout), 1, 0, 1)
            error = self.reject("context_synchronize", context)
            self.assertEqual(error.native_code, -2)
            # Valid elements may complete; invalid elements never access source/output.
            self.assertEqual(self.read(context, output, 4), [10, 10, 40, 40])
            self.checked("context_synchronize", context)
            values = self.tensor(context, [3, 4, 5, 6])
            self.index_update_checked(context, source, 0, indices, 3, values, c.byref(matrix), 1, 2, 0)
            self.assertEqual(self.read(context, source, 6), [4, 20, 3, 6, 50, 5])
            bits = struct.unpack("<I", struct.pack("<f", 7))[0]
            self.index_fill_checked(context, source, 0, indices, 3, c.byref(matrix), 1, 2, bits)
            self.assertEqual(self.read(context, source, 6), [7, 20, 7, 7, 50, 7])
            flat_indices = self.tensor(context, [-1, 0, 100], 3)
            flat_output = self.tensor(context, [0] * 3)
            self.checked("gather_unary_f32", context, source, flat_indices, 3, flat_output, 6, 3, self.operations["Neg"])
            self.assertEqual(self.read(context, flat_output, 3), [-7, -7, -7])
            self.checked("index_put_f32", context, source, indices, 3, self.tensor(context, [2, 8]), 6, 2)
            self.assertEqual(self.read(context, source, 3), [8, 20, 2])
            cast = self.tensor(context, [0, 0], 2)
            self.checked("index_cast", context, indices, cast, 2, 6)
            self.assertEqual(self.read(context, cast, 2, 2), [2, 0])

    @contextmanager
    def readback_slot(self, context):
        slot = c.c_void_p()
        self.checked("readback_create", context, c.byref(slot))
        try:
            yield slot
        finally:
            # Fault tests deliberately leave an execution error in the context.
            # Draining it is independent of the snapshot's own error lifetime.
            error = Error()
            self.calls["context_synchronize"](context, c.byref(error))
            self.checked("readback_destroy", context, slot)

    def test_invalid_index_readback_never_exposes_payload_and_can_be_discarded(self):
        with self.resources() as context, self.readback_slot(context) as slot:
            source = self.tensor(context, [1, 2, 3])
            invalid = self.tensor(context, [-1, 3], 3)
            valid = self.tensor(context, [2, 0], 3)
            output = self.tensor(context, [99, 99])
            p = Gather(c.sizeof(Gather), 0, 3, 0, 0, 1, layout([3]), layout([2]))
            self.checked("gather", context, source, invalid, output, c.byref(p))
            self.checked("readback_enqueue", context, slot, output.buffer, output.offset, 8)
            self.assertEqual(self.reject("readback_wait", context, slot).native_code, -2)
            destination, ready = (c.c_float * 2)(-7, -8), c.c_int32(9)
            self.assertEqual(self.reject("readback_poll", context, slot, destination, 8, c.byref(ready)).native_code, -2)
            self.assertEqual(ready.value, 0)
            self.assertEqual(list(destination), [-7, -8])
            self.assertEqual(self.reject("context_synchronize", context).native_code, -2)
            # Consuming the context error must not validate an already bad snapshot.
            self.assertEqual(self.reject("readback_wait", context, slot).native_code, -2)
            self.assertEqual(self.reject("readback_poll", context, slot, None, 0, c.byref(ready)).native_code, -2)
            self.assertEqual(ready.value, 0)
            self.checked("gather", context, source, valid, output, c.byref(p))
            self.checked("readback_enqueue", context, slot, output.buffer, output.offset, 8)
            self.checked("readback_wait", context, slot)
            self.checked("readback_poll", context, slot, destination, 8, c.byref(ready))
            self.assertEqual(ready.value, 1)
            self.assertEqual(list(destination), [3, 1])

    def test_later_index_error_does_not_poison_earlier_snapshot(self):
        with self.resources() as context, self.readback_slot(context) as earlier, self.readback_slot(context) as later:
            source = self.tensor(context, [1, 2, 3])
            valid = self.tensor(context, [2, 0], 3)
            invalid = self.tensor(context, [-1, 3], 3)
            output = self.tensor(context, [99, 99])
            p = Gather(c.sizeof(Gather), 0, 3, 0, 0, 1, layout([3]), layout([2]))
            self.checked("gather", context, source, valid, output, c.byref(p))
            self.checked("readback_enqueue", context, earlier, output.buffer, 0, 8)
            self.checked("gather", context, source, invalid, output, c.byref(p))
            self.checked("readback_enqueue", context, later, output.buffer, 0, 8)
            destination, ready = (c.c_float * 2)(-7, -8), c.c_int32()
            self.checked("readback_wait", context, earlier)
            self.checked("readback_poll", context, earlier, destination, 8, c.byref(ready))
            self.assertEqual(ready.value, 1)
            self.assertEqual(list(destination), [3, 1])
            self.assertEqual(self.reject("context_synchronize", context).native_code, -2)
            destination[:] = [-7, -8]
            self.assertEqual(self.reject("readback_poll", context, later, destination, 8, c.byref(ready)).native_code, -2)
            self.assertEqual(ready.value, 0)
            self.assertEqual(list(destination), [-7, -8])
            self.assertEqual(self.reject("readback_wait", context, later).native_code, -2)
            self.assertEqual(self.reject("readback_poll", context, later, None, 0, c.byref(ready)).native_code, -2)
            self.assertEqual(ready.value, 0)

    def test_packed_readback_preserves_faults_across_appends_and_context_reset(self):
        with self.resources() as context, self.readback_slot(context) as slot:
            source = self.tensor(context, [1, 2, 3])
            invalid = self.tensor(context, [-1, 3], 3)
            valid = self.tensor(context, [2, 0], 3)
            output = self.tensor(context, [99, 99])
            host = c.c_uint64()
            self.checked("readback_reserve", context, slot, 8, c.byref(host))
            p = Gather(c.sizeof(Gather), 0, 3, 0, 0, 1, layout([3]), layout([2]))
            self.checked("gather", context, source, invalid, output, c.byref(p))
            self.checked("readback_append", context, slot, output.buffer, 0, 4, 0)
            self.assertEqual(self.reject("context_synchronize", context).native_code, -2)
            self.checked("gather", context, source, valid, output, c.byref(p))
            self.checked("readback_append", context, slot, output.buffer, 4, 4, 4)
            self.checked("context_synchronize", context)
            self.assertEqual(self.reject("readback_wait", context, slot).native_code, -2)
            destination, ready = (c.c_float * 2)(-7, -8), c.c_int32(9)
            self.assertEqual(self.reject("readback_poll", context, slot, destination, 8, c.byref(ready)).native_code, -2)
            self.assertEqual(list(destination), [-7, -8])
            self.assertEqual(ready.value, 0)
            self.assertEqual(self.reject("readback_poll", context, slot, None, 0, c.byref(ready)).native_code, -2)
            self.checked("readback_append", context, slot, output.buffer, 0, 8, 0)
            self.checked("readback_wait", context, slot)
            self.checked("readback_poll", context, slot, destination, 8, c.byref(ready))
            self.assertEqual(ready.value, 1)
            self.assertEqual(list(destination), [3, 1])

    def test_readback_reports_masked_scatter_cardinality_error(self):
        with self.resources() as context, self.readback_slot(context) as slot:
            output = self.tensor(context, [99, 99, 99])
            mask = self.tensor(context, [1, 0, 1], 4)
            source = self.tensor(context, [7])
            self.checked("masked_scatter", context, output, mask, source, 0, 3, 1)
            self.checked("readback_enqueue", context, slot, output.buffer, 0, 12)
            self.assertEqual(self.reject("readback_wait", context, slot).native_code, -3)
            destination, ready = (c.c_float * 3)(-7, -8, -9), c.c_int32(9)
            self.assertEqual(self.reject("readback_poll", context, slot, destination, 12, c.byref(ready)).native_code, -3)
            self.assertEqual(ready.value, 0)
            self.assertEqual(list(destination), [-7, -8, -9])
            self.assertEqual(self.reject("context_synchronize", context).native_code, -3)
            self.assertEqual(self.reject("readback_poll", context, slot, None, 0, c.byref(ready)).native_code, -3)

    def test_empty_readback_and_ordering_marker_capture_index_errors(self):
        for marker in (False, True):
            with self.subTest(marker=marker), self.resources() as context, self.readback_slot(context) as slot:
                source = self.tensor(context, [1])
                indices = self.tensor(context, [1], 3)
                output = self.tensor(context, [99])
                p = Gather(c.sizeof(Gather), 0, 3, 0, 0, 1, layout([1]), layout([1]))
                self.checked("gather", context, source, indices, output, c.byref(p))
                if marker:
                    self.checked("marker_record", context, slot)
                else:
                    self.checked("readback_enqueue", context, slot, None, 0, 0)
                self.assertEqual(self.reject("readback_wait", context, slot).native_code, -2)
                ready = c.c_int32(9)
                self.assertEqual(self.reject("readback_poll", context, slot, None, 0, c.byref(ready)).native_code, -2)
                self.assertEqual(ready.value, 0)

    def test_invalid_update_fill_and_cast_skip_out_of_range_lanes(self):
        with self.resources() as context:
            output = self.tensor(context, [99] * 6)
            indices = self.tensor(context, [-1, 3], 3)
            values = self.tensor(context, [1, 2, 3, 4])
            for add in (0, 1):
                self.index_update_checked(context, output, 0, indices, 3, values, c.byref(layout([2, 3])), 1, 2, add)
                self.assertEqual(self.reject("context_synchronize", context).native_code, -2)
                self.assertEqual(self.read(context, output, 6), [99] * 6)
            self.index_fill_checked(context, output, 0, indices, 3, c.byref(layout([2, 3])), 1, 2, 0)
            self.assertEqual(self.reject("context_synchronize", context).native_code, -2)
            self.assertEqual(self.read(context, output, 6), [99] * 6)
            cast = self.tensor(context, [77, 77], 2)
            self.checked("index_cast", context, indices, cast, 2, 3)
            self.assertEqual(self.reject("context_synchronize", context).native_code, -2)
            self.assertEqual(self.read(context, cast, 2, 2), [77, 77])

    def test_byte_predicates_and_integer_clamp_cover_capped_grid_tail(self):
        count = 4096 * 256 + 17
        with self.resources() as context:
            source = self.tensor(context, [0] * count, 4)
            output = self.tensor(context, [77] * count, 4)
            self.byte_predicate("checked", context, "LogicalNot", source, output, count, 0)
            self.assertEqual(self.read(context, View(output.buffer, count - 17), 17, 4), [1] * 17)
            integers = self.tensor(context, [-7] * count, 2)
            self.checked("clamp_i32", context, integers, count, -2, 3)
            self.assertEqual(self.read(context, View(integers.buffer, (count - 17) * 4), 17, 2), [-2] * 17)

    def test_byte_predicates_reject_shifted_aliases_but_allow_exact_inplace(self):
        with self.resources() as context:
            values = self.tensor(context, [0, 1, 0, 1, 0], 4)
            shifted = View(values.buffer, 1)
            self.byte_predicate("reject", context, "LogicalNot", values, shifted, 4, 0)
            self.assertEqual(self.read(context, values, 5, 4), [0, 1, 0, 1, 0])
            self.byte_predicate("checked", context, "LogicalNot", values, values, 5, 0)
            self.assertEqual(self.read(context, values, 5, 4), [1, 0, 1, 0, 1])

    def test_indexed_add_accumulates_duplicate_indices(self):
        with self.resources() as context:
            for dtype in (0, 2):
                with self.subTest(dtype=dtype):
                    destination = self.tensor(context, [0] * 6, dtype)
                    indices = self.tensor(context, [2, 2, 0], 2)
                    values = self.tensor(context, [1, 2, 3, 4, 5, 6], dtype)
                    shape = layout([2, 3])
                    self.index_update_checked(context, destination, dtype, indices, 2,
                                 values, c.byref(shape), 1, 3, 1)
                    self.assertEqual(self.read(context, destination, 6, dtype), [3, 0, 3, 6, 0, 9])

    def test_multinomial_rejects_invalid_weights_without_writing_output(self):
        with self.resources() as context:
            output = self.tensor(context, [99], 3)
            for weights in ([0, 0, 0], [0, -1, 0], [0, float("nan"), 0], [0, float("inf"), 0]):
                with self.subTest(weights=weights):
                    source = self.tensor(context, weights)
                    self.reject("multinomial", context, source, output, 3, 1, 42, 1)
                    self.assertEqual(self.read(context, output, 1, 3), [99])
            source = self.tensor(context, [0, 1, 0])
            self.checked("multinomial", context, source, output, 3, 1, 42, 1)
            self.assertEqual(self.read(context, output, 1, 3), [1])

    def test_random_reproducibility_bounds_and_distribution(self):
        with self.resources() as context:
            count = 4097
            for kind in range(4):
                dtype = 2 if kind == 2 else 0
                first, second = (0.25, 1) if kind == 1 else (-2, 3) if kind == 0 else (0, 1)
                program = Random(kind, 1234567, count, first, second, -7, 11)
                output = self.tensor(context, [0] * count, dtype)
                other = self.tensor(context, [0] * count, dtype)
                self.checked("random", context, output, c.byref(program))
                self.checked("random", context, other, c.byref(program))
                values = self.read(context, output, count, dtype)
                self.assertEqual(values, self.read(context, other, count, dtype))
                if kind == 0:
                    self.assertTrue(all(-2 <= x < 3 for x in values))
                elif kind == 1:
                    self.assertTrue(set(values) <= {0, 1})
                    self.assertAlmostEqual(sum(values) / count, 0.25, delta=0.04)
                elif kind == 2:
                    self.assertTrue(all(-7 <= x < 11 for x in values))
                else:
                    self.assertTrue(all(math.isfinite(x) for x in values))
                    self.assertAlmostEqual(sum(values) / count, 0, delta=0.08)
                    self.assertAlmostEqual(sum(x * x for x in values) / count, 1, delta=0.12)
            nonzero = self.tensor(context, [0, -1, 2, 0, float("nan")])
            total = c.c_uint64()
            self.checked("count_nonzero", context, nonzero, 0, 5, c.byref(total))
            self.assertEqual(total.value, 3)

    def test_cdist_short_and_tiled_norms_match_independent_reference(self):
        with self.resources() as context:
            for columns in (0, 3, 21):
                a = [float(i % 9 - 4) / 4 for i in range(17 * columns)]
                b = [float(i % 7 - 3) / 4 for i in range(19 * columns)]
                lhs, rhs = self.tensor(context, a), self.tensor(context, b)
                output = self.tensor(context, [99] * (17 * 19))
                for norm in (0, 1, 2, float("inf"), 1.5):
                    with self.subTest(columns=columns, norm=norm):
                        self.checked("cdist_f32", context, lhs, rhs, output, 17, 19, columns, norm)
                        expected = []
                        for row in range(17):
                            for column in range(19):
                                differences = [abs(a[row * columns + k] - b[column * columns + k]) for k in range(columns)]
                                if norm == 0:
                                    value = sum(x != 0 for x in differences)
                                elif math.isinf(norm):
                                    value = max(differences, default=0)
                                else:
                                    value = sum(x ** norm for x in differences) ** (1 / norm)
                                expected.append(value)
                        for actual, reference in zip(self.read(context, output, 17 * 19), expected):
                            self.assertAlmostEqual(actual, reference, delta=2e-5 * max(1, reference))

    def test_gemm_batches_transpose_skinny_and_tiled(self):
        with self.resources() as context:
            for m, n, k, batches, transpose in ((2, 3, 4, 2, False), (17, 19, 21, 1, True), (5, 7, 0, 1, False)):
                with self.subTest(m=m, n=n, k=k, batch=batches):
                    a = [float(i % 7 - 3) / 4 for i in range(batches * m * k)]
                    b = [float(i % 5 - 2) / 4 for i in range(batches * k * n)]
                    bias = [float(i - 2) for i in range(n)]
                    aa, bb, bias_view = self.tensor(context, a), self.tensor(context, b), self.tensor(context, bias)
                    output = self.tensor(context, [99] * (batches * m * n))
                    p = Gemm(c.sizeof(Gemm), transpose, 1, batches, m, n, k)
                    self.checked("gemm_f32", context, aa, bb, bias_view, output, c.byref(p))
                    expected = []
                    for batch in range(batches):
                        for row in range(m):
                            for column in range(n):
                                value = sum(a[(batch * m + row) * k + t] * b[batch * k * n + (column * k + t if transpose else t * n + column)] for t in range(k))
                                expected.append(max(0, value + bias[column]))
                    self.assertEqual(self.read(context, output, len(expected)), expected)
            a, b, output = self.tensor(context, [0, 0, 3, 4]), self.tensor(context, [0, 4]), self.tensor(context, [99, 99])
            self.checked("cdist_f32", context, a, b, output, 2, 1, 2, 2)
            self.assertEqual(self.read(context, output, 2), [4, 3])

    def test_nn_linear_shared_product_preserves_epilogues(self):
        with self.resources() as context:
            for dtype in (0, 1):
                for m, n, k, batches in ((2, 3, 4, 2), (17, 19, 21, 1), (5, 7, 0, 1), (65, 67, 17, 1)):
                    for transpose in (False, True):
                        for activation in range(5):
                            with self.subTest(dtype=dtype, m=m, transpose=transpose, activation=activation):
                                batched = activation % 2
                                a = [(i % 7 - 3) / 8 for i in range(batches*m*k)]
                                b = [(i % 5 - 2) / 8 for i in range((batches if batched else 1)*k*n)]
                                bias = [(i % 3 - 1) / 8 for i in range(n)]
                                scale = [(i % 3 + 1) / 4 for i in range(n)]
                                residual = [(i % 5 - 2) / 8 for i in range(batches*m*n)]
                                inputs = (View * 5)(self.tensor(context, a, dtype), self.tensor(context, b, dtype),
                                    self.tensor(context, bias, dtype), self.tensor(context, scale, dtype), self.tensor(context, residual, dtype))
                                output = self.tensor(context, [99]*(batches*m*n), dtype)
                                descriptor = Linear(batches,m,n,k,dtype,transpose,batched,activation)
                                self.checked("nn_linear",context,inputs,output,c.byref(descriptor))
                                expected=[]
                                for batch in range(batches):
                                    for row in range(m):
                                        for col in range(n):
                                            value=sum(a[(batch*m+row)*k+i]*b[(batch if batched else 0)*k*n+(col*k+i if transpose else i*n+col)] for i in range(k))+bias[col]
                                            if activation==1: value=max(0,value)
                                            elif activation==2: value=.5*value*(1+math.tanh(.7978845608028654*(value+.044715*value**3)))
                                            elif activation==3: value=.5*value*(1+math.erf(value/math.sqrt(2)))
                                            elif activation==4: value=value/(1+math.exp(-value))
                                            expected.append(value*scale[col]+residual[(batch*m+row)*n+col])
                                result=self.read(context,output,len(expected),dtype)
                                tolerance=0.002 if dtype else 0.00001
                                for actual,reference in zip(result,expected):
                                    self.assertAlmostEqual(actual,reference,delta=tolerance*max(1,abs(reference)))


class TensorFaultRecovery(TensorOperations):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.fail_once = native.bind(cls.dll, "test_fail_once", [c.c_int], None)
        cls.live_resources = native.bind(cls.dll, "test_live_resources", [], c.c_int)

    def tearDown(self):
        self.fail_once(0)
        self.assertEqual(self.live_resources(), 0)

    def test_extended_workspace_growth_failure_is_retryable(self):
        for fault in (7, 9, 8):
            with self.subTest(fault=fault), self.resources() as context:
                small = self.tensor(context, [0, 1, 0])
                large = self.tensor(context, [0, 1] + [0] * 4095)
                output = self.tensor(context, [99], 3)
                self.checked("multinomial", context, small, output, 3, 1, 42, 1)
                self.checked("context_synchronize", context)
                self.fail_once(fault)
                self.reject("multinomial", context, large, output, 4097, 1, 42, 1)
                self.assertEqual(self.read(context, output, 1, 3), [1])
                self.checked("multinomial", context, large, output, 4097, 1, 42, 1)
                self.assertEqual(self.read(context, output, 1, 3), [1])

    def test_scalar_staging_survives_copy_sync_and_release_failures(self):
        for fault in (1, 6, 7):
            with self.subTest(fault=fault), self.resources() as context:
                source = self.tensor(context, [0, 1, 0])
                output = self.tensor(context, [99], 3)
                if fault != 1:
                    self.checked("multinomial", context, source, output, 3, 1, 42, 1)
                    self.checked("context_synchronize", context)
                    self.checked("memset", context, output, 8, 0x63)
                self.fail_once(fault)
                self.reject("multinomial", context, source, output, 3, 1, 42, 1)
                self.checked("context_synchronize", context)
                expected = 99 if fault == 1 else 0x6363636363636363
                self.assertEqual(self.read(context, output, 1, 3), [expected])
                self.checked("multinomial", context, source, output, 3, 1, 42, 1)
                self.assertEqual(self.read(context, output, 1, 3), [1])
                self.fail_once(2)
                self.reject("context_destroy", context)
                # Destruction retains the pinned staging after failed release.

    def test_readback_error_capture_failures_retain_all_owners(self):
        for fault in (1, 8, 10, 6, 3):
            with self.subTest(fault=fault), self.resources() as context:
                source = self.tensor(context, [1, 2, 3])
                indices = self.tensor(context, [2, 0], 3)
                output = self.tensor(context, [99, 99])
                p = Gather(c.sizeof(Gather), 0, 3, 0, 0, 1, layout([3]), layout([2]))
                self.checked("gather", context, source, indices, output, c.byref(p))
                with self.readback_slot(context) as slot:
                    host = c.c_uint64()
                    self.checked("readback_reserve", context, slot, 4, c.byref(host))
                    self.fail_once(fault)
                    self.reject("readback_append", context, slot, output.buffer, 0, 0, 0)
                    ready = c.c_int32(9)
                    if fault == 3:
                        # The payload/error capture succeeded; a failed event
                        # record falls back to a stream completion query.
                        self.checked("readback_wait", context, slot)
                        self.checked("readback_poll", context, slot, None, 0, c.byref(ready))
                        self.assertEqual(ready.value, 1)
                    else:
                        self.reject("readback_wait", context, slot)
                        self.reject("readback_poll", context, slot, None, 0, c.byref(ready))
                        self.assertEqual(ready.value, 0)
                    self.checked("readback_append", context, slot, output.buffer, 0, 0, 0)
                    self.checked("readback_wait", context, slot)
                    self.checked("readback_poll", context, slot, None, 0, c.byref(ready))
                    self.assertEqual(ready.value, 1)
                    self.fail_once(2)
                    self.reject("readback_destroy", context, slot)
                    self.fail_once(9)
                    self.reject("readback_destroy", context, slot)
                    # The fixture retries destruction; pinned/device snapshot
                    # storage remains owned through both release failures.

    def test_index_error_storage_allocation_and_cleanup_can_retry(self):
        for fault in (1, 8):
            with self.subTest(fault=fault), self.resources() as context:
                source = self.tensor(context, [1, 2, 3])
                indices = self.tensor(context, [2, 0], 3)
                output = self.tensor(context, [99, 99])
                p = Gather(c.sizeof(Gather), 0, 3, 0, 0, 1, layout([3]), layout([2]))
                self.fail_once(fault)
                self.reject("gather", context, source, indices, output, c.byref(p))
                self.assertEqual(self.read(context, output, 2), [99, 99])
                self.checked("gather", context, source, indices, output, c.byref(p))
                self.assertEqual(self.read(context, output, 2), [3, 1])
                self.fail_once(2)
                self.reject("context_destroy", context)
                # The context retains only resources whose release failed;
                # the resources() finalizer retries destruction.

    def test_index_error_survives_failed_sync_and_is_consumed_once(self):
        with self.resources() as context:
            source = self.tensor(context, [1, 2, 3])
            indices = self.tensor(context, [-1, 3], 3)
            output = self.tensor(context, [99, 99])
            self.gather_checked(context, source, 0, indices, 3, output, c.byref(layout([3])), c.byref(layout([2])), 0, 0, 1)
            self.fail_once(7)
            self.reject("context_synchronize", context)
            self.assertEqual(self.reject("context_synchronize", context).native_code, -2)
            self.checked("context_synchronize", context)
            self.assertEqual(self.read(context, output, 2), [99, 99])

    def test_precounted_selection_and_scatter_do_not_drain_the_stream(self):
        with self.resources() as context:
            source = self.tensor(context, [1, 2, 3])
            mask = self.tensor(context, [1, 0, 1], 4)
            output = self.tensor(context, [99, 99])
            selected = c.c_uint64()
            self.select_checked(context, source, mask, output, 0, 0, 3, 2, c.byref(selected), known=2)
            self.checked("context_synchronize", context)
            self.fail_once(7)
            self.select_checked(context, source, mask, output, 0, 0, 3, 2, c.byref(selected), known=2)
            self.reject("context_synchronize", context)
            self.checked("context_synchronize", context)
            self.assertEqual(self.read(context, output, 2), [1, 3])
            destination = self.tensor(context, [99, 99, 99])
            self.checked("masked_scatter", context, destination, mask, output, 0, 3, 2)
            self.checked("context_synchronize", context)
            self.fail_once(7)
            self.checked("masked_scatter", context, destination, mask, output, 0, 3, 2)
            self.reject("context_synchronize", context)
            self.checked("context_synchronize", context)
            self.assertEqual(self.read(context, destination, 3), [1, 99, 3])

    def test_completed_fence_does_not_enqueue_an_error_flag_reset(self):
        with self.resources() as context:
            source = self.tensor(context, [1, 2, 3])
            indices = self.tensor(context, [2, 0], 3)
            output = self.tensor(context, [99, 99])
            p = Gather(c.sizeof(Gather), 0, 3, 0, 0, 1, layout([3]), layout([2]))
            self.checked("gather", context, source, indices, output, c.byref(p))
            self.fail_once(10)
            self.checked("context_synchronize", context)
            # The reset belongs to the next submission, not the completed fence.
            self.reject("gather", context, source, indices, output, c.byref(p))
            self.assertEqual(self.read(context, output, 2), [3, 1])
            self.checked("gather", context, source, indices, output, c.byref(p))
            self.assertEqual(self.read(context, output, 2), [3, 1])

    def test_workspace_release_and_allocation_failure_are_retryable(self):
        with self.resources() as context:
            source = self.tensor(context, [3, 1, 2, 0])
            indices = self.tensor(context, [0] * 4, 3)
            self.checked("sort_pairs", context, source, indices, 1, 4, 1, 0)
            for fault in (7, 9):
                self.fail_once(fault)
                error = Error()
                self.assertNotEqual(self.calls["trim_workspace"](context, c.byref(error)), 0)
                self.checked("trim_workspace", context)
                self.checked("sort_pairs", context, source, indices, 1, 4, 1, 0)
            self.checked("trim_workspace", context)
            self.fail_once(8)
            error = Error()
            self.assertNotEqual(self.calls["sort_pairs"](context, source, indices, 1, 4, 1, 0, c.byref(error)), 0)
            self.assertEqual(self.read(context, source, 4), [0, 1, 2, 3])
            self.checked("sort_pairs", context, source, indices, 1, 4, 1, 0)

    def test_workspace_growth_failure_preserves_inputs_and_can_retry(self):
        for fault in (7, 9, 8):
            with self.subTest(fault=fault), self.resources() as context:
                small = self.tensor(context, [3, 1, 2, 0])
                small_indices = self.tensor(context, [0] * 4, 3)
                self.checked("sort_pairs", context, small, small_indices, 1, 4, 1, 0)
                count = 65537
                original = list(range(count, 0, -1))
                large = self.tensor(context, original)
                indices = self.tensor(context, [0] * count, 3)
                self.fail_once(fault)
                error = Error()
                self.assertNotEqual(self.calls["sort_pairs"](context, large, indices, 1, count, 1, 0, c.byref(error)), 0)
                self.assertEqual(self.read(context, large, count), original)
                self.checked("sort_pairs", context, large, indices, 1, count, 1, 0)
                self.assertEqual(self.read(context, large, count), list(range(1, count + 1)))

    def test_context_destroy_free_failure_retains_ownership_for_retry(self):
        context = c.c_void_p()
        self.checked("context_create", native.options.device, c.byref(context))
        try:
            self.tensor(context, [1, 2, 3])
            self.fail_once(9)
            error = Error()
            self.assertNotEqual(self.calls["context_destroy"](context, c.byref(error)), 0)
        finally:
            self.fail_once(0)
            self.checked("context_destroy", context)

    def test_scalar_read_drain_failure_preserves_host_output(self):
        with self.resources() as context:
            source = self.tensor(context, [1, 2, 3])
            result = c.c_float(91)
            self.checked("reduce_f32", context, source, 3, 0, c.byref(result))
            result.value = 91
            self.fail_once(7)
            error = Error()
            self.assertNotEqual(self.calls["reduce_f32"](context, source, 3, 0, c.byref(result), c.byref(error)), 0)
            self.assertEqual(result.value, 91)
            self.checked("reduce_f32", context, source, 3, 0, c.byref(result))
            self.assertEqual(result.value, 6)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("module", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--runtime-dir", type=Path)
    parser.add_argument("--abi-only", action="store_true")
    parser.add_argument("--faults", action="store_true")
    args = parser.parse_args()
    if args.abi_only and args.faults:
        parser.error("--abi-only and --faults are mutually exclusive")
    if not 0 <= args.device <= 2147483647:
        parser.error("--device must be an ordinal in 0..2147483647")
    native.options = argparse.Namespace(module=args.module, device=args.device, faults=args.faults)
    directory = os.add_dll_directory(str(args.runtime_dir.resolve())) if os.name == "nt" and args.runtime_dir else None
    try:
        names = ["TensorOperations.test_exported_operator_boundary_without_gpu_queries"] if args.abi_only else ["TensorOperations." + name for name in TensorOperations.__dict__ if name.startswith("test_")]
        if args.faults:
            names = ["TensorFaultRecovery." + name for name in TensorFaultRecovery.__dict__ if name.startswith("test_")]
        unittest.main(argv=[__file__], defaultTest=names)
    finally:
        if directory:
            directory.close()
