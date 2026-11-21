# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

import pytest


@pytest.mark.parametrize("dtype", ["int32", "long", "byte"])
def test_integer_arange_preserves_rows_above_float_precision(lf, dtype):
    start = 2**24 + 1
    result = lf.Tensor.arange(start, start + 3, device="cpu", dtype=dtype)
    expected = [1, 2, 3] if dtype == "byte" else list(range(start, start + 3))
    assert result.tolist() == expected


def test_integer_arange_accepts_numpy_integers_and_descending_steps(lf, numpy):
    end = 2**53
    result = lf.Tensor.arange(numpy.int64(end + 5), end, -2, "cpu", "int64")
    assert result.tolist() == list(range(end + 5, end, -2))
    assert lf.Tensor.arange(end=3, device="cpu", dtype="int").tolist() == [0, 1, 2]
    assert lf.Tensor.arange(5, 3, device="cpu", dtype="long").shape == (0,)


@pytest.mark.parametrize("kind", ["builtin", "uint64"])
def test_oversized_integer_stride_keeps_float_result(lf, numpy, kind):
    step = 2**63 + 1
    if kind == "uint64":
        step = numpy.uint64(step)
    assert lf.Tensor.arange(0, 3, step, "cpu", "byte").tolist() == [0]


def test_oversized_bounds_and_descending_stride_keep_float_result(lf):
    assert lf.Tensor.arange(2**63, 2**63 + 1, device="cpu", dtype="int64").shape == (0,)
    assert lf.Tensor.arange(2, 0, -(2**63) - 1, "cpu", "long").tolist() == [2]


@pytest.mark.parametrize("name", ["intc", "uintc", "long", "ulong", "int64", "uint64"])
def test_numpy_integer_scalar_types_are_exact(lf, numpy, name):
    scalar = vars(numpy).get(name)
    if not isinstance(scalar, type) or not issubclass(scalar, numpy.integer):
        pytest.skip(f"{name} is unavailable as a NumPy integer scalar type")
    start = 2**24 + 1
    assert lf.Tensor.arange(scalar(start), start + 3, device="cpu", dtype="long").tolist() == list(range(start, start + 3))


def test_arange_ignores_proxy_class_when_selecting_integer_route(lf, numpy):
    calls = []

    class Proxy:
        @property
        def __class__(self):
            calls.append("class")
            return numpy.int64

        def __float__(self):
            calls.append("float")
            return 3.0

        def __index__(self):
            calls.append("index")
            return 2

    assert lf.Tensor.arange(Proxy(), 5, device="cpu", dtype="int32").tolist() == [3, 4]
    assert calls == ["float"]


def test_numpy_integer_subclass_is_exact(lf, numpy):
    class Row(numpy.int64):
        pass

    start = 2**53 + 1
    assert lf.Tensor.arange(Row(start), start + 2, device="cpu", dtype="long").tolist() == [start, start + 1]


def test_integer_arange_checks_int32_results(lf):
    with pytest.raises(RuntimeError, match="outside the representable range"):
        lf.Tensor.arange(2**31, 2**31 + 1, device="cpu", dtype="int32")
    assert lf.Tensor.arange(2**31 - 1, 2**31, device="cpu", dtype="int32").tolist() == [2**31 - 1]
    floats = lf.Tensor.arange(2**31, 2**31 + 256, 128.0, "cpu")
    assert lf.Tensor.arange(2**31, 2**31 + 256, 128.0, "cpu", "int32").tolist() == floats.to("int32").tolist()


def test_arange_keeps_float_values_for_mixed_bounds_or_default_dtype(lf, numpy):
    start = 2**24
    rounded = [start, start, start + 2, start + 4]
    assert lf.Tensor.arange(start, start + 4, 1.0, "cpu", "int32").tolist() == rounded
    assert lf.Tensor.arange(start, start + 4, device="cpu").tolist() == rounded
    assert lf.Tensor.arange(-1, 1.0, 0.5, "cpu", "int64").tolist() == [-1, 0, 0, 0]
    assert lf.Tensor.arange(-1, 1, numpy.array(0.5), "cpu", "int64").tolist() == [-1, 0, 0, 0]


@pytest.mark.parametrize("index_result", [2, TypeError("no index"), ValueError("no index")])
def test_arange_uses_float_protocol_for_custom_numbers(lf, index_result):
    calls = []

    class Number:
        def __float__(self):
            calls.append("float")
            return 3.0

        def __index__(self):
            calls.append("index")
            if isinstance(index_result, Exception):
                raise index_result
            return index_result

    assert lf.Tensor.arange(Number(), 5, device="cpu", dtype="int32").tolist() == [3, 4]
    assert calls == ["float"]


def test_arange_converts_other_numbers_in_argument_order(lf):
    calls = []

    class Number:
        def __init__(self, position):
            self.position = position

        def __index__(self):
            raise TypeError("no index")

        def __float__(self):
            calls.append(self.position)
            if self.position == "end":
                raise TypeError("bad end")
            return 1.0

    with pytest.raises(TypeError, match="bad end"):
        lf.Tensor.arange(Number("start"), Number("end"), Number("step"), "cpu", "byte")
    assert calls == ["start", "end"]


def test_arange_bool_and_int_subclasses_keep_float_conversion(lf):
    class Bound(int):
        def __float__(self):
            return 2.0

    assert lf.Tensor.arange(start=Bound(4), end=5, step=True, device="cpu", dtype="int64").tolist() == [2, 3, 4]
    start = 2**24
    assert lf.Tensor.arange(start, start + 4, True, "cpu", "int64").tolist() == [start, start, start + 2, start + 4]
    with pytest.raises(RuntimeError, match="step cannot be zero"):
        lf.Tensor.arange(0, 1, False, "cpu", "int32")
