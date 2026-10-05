# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Restored tensor bindings, exercised on CPU and the selected GPU backend."""

import pytest


@pytest.fixture(params=["cpu", "gpu"])
def device(request, gpu_available):
    if request.param == "gpu" and not gpu_available:
        pytest.skip("GPU backend unavailable")
    return request.param


@pytest.fixture
def tensor(lf, numpy, device):
    def make(values, dtype="float32"):
        result = lf.Tensor.from_numpy(numpy.asarray(values, dtype="float32" if dtype == "float16" else dtype))
        result = result if device == "cpu" else result.gpu()
        return result.to(dtype) if dtype == "float16" else result
    return make


def assert_values(numpy, actual, expected, **kwargs):
    if actual.dtype == "float16":
        actual = actual.to("float32")
    numpy.testing.assert_allclose(actual.cpu().numpy(), expected, **kwargs)


@pytest.mark.parametrize("dtype", ["float32", "float"])
def test_normal(lf, numpy, device, dtype):
    result = lf.Tensor.normal([2, 3], mean=2.5, std=0, device=device, dtype=dtype)
    assert result.shape == (2, 3)
    assert result.dtype == "float32"
    assert result.backend == lf.Tensor.zeros([1], device=device).backend
    assert_values(numpy, result, numpy.full((2, 3), 2.5))
    samples = lf.Tensor.normal([20000], mean=3, std=2, device=device)
    assert samples.mean().item() == pytest.approx(3, abs=0.1)
    assert samples.std().item() == pytest.approx(2, abs=0.1)


@pytest.mark.parametrize("p", [0, 0.25, 1])
def test_bernoulli(lf, numpy, device, p):
    result = lf.Tensor.bernoulli([20000], p=p, device=device, dtype="float32")
    values = result.cpu().numpy()
    assert set(numpy.unique(values)) <= {0, 1}
    assert values.mean() == pytest.approx(p, abs=0.02)
    assert result.backend == lf.Tensor.zeros([1], device=device).backend


def test_multinomial(lf, numpy, tensor):
    weights = tensor([[0, 9], [1, 9], [0, 9]])[:, 0]
    draws = lf.Tensor.multinomial(weights, 32, replacement=True, seed=42)
    assert draws.dtype == "int64"
    assert draws.backend == weights.backend
    assert_values(numpy, draws, numpy.ones(32))
    weights = tensor([1, 2, 3, 4])
    assert sorted(lf.Tensor.multinomial(weights, 4).tolist()) == [0, 1, 2, 3]
    a = lf.Tensor.multinomial(weights, 32, True, seed=7)
    b = lf.Tensor.multinomial(weights, 32, True, seed=7)
    assert a.tolist() == b.tolist()


def test_diag(lf, numpy, tensor):
    diagonal = tensor([[2, 99], [3, 99]])[:, 0]
    result = lf.Tensor.diag(diagonal)
    assert result.backend == diagonal.backend
    assert_values(numpy, result, [[2, 0], [0, 3]])


@pytest.mark.parametrize("p,expected", [(2, 5), (1, 7), (0, 2), (float("inf"), 4)])
def test_cdist(numpy, tensor, p, expected):
    result = tensor([[0, 0], [3, 4]]).cdist(tensor([[0, 0]]), p=p)
    assert_values(numpy, result, [[0], [expected]])


def test_normalize(numpy, tensor):
    values = numpy.array([[1, 3], [2, 8]], dtype=numpy.float32)
    x = tensor(values)
    assert_values(numpy, x.normalize(), (values - values.mean()) / values.std(), atol=1e-6)
    assert_values(numpy, x.normalize(dim=1, eps=0.5),
                  (values - values.mean(axis=1, keepdims=True)) /
                  (values.std(axis=1, keepdims=True) + 0.5), atol=1e-6)


def test_mod(numpy, tensor):
    assert_values(numpy, tensor([-5, 5]).mod(tensor([3, -3])), [-2, 2])


@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_clamp_and_strided_inplace(numpy, tensor, dtype):
    x = tensor([[-4, 99], [0.5, 99], [4, 99], [float("nan"), 99]], dtype)
    view = x[:, :1]
    assert_values(numpy, view.clamp(min=-1, max=1), [[-1], [0.5], [1], [numpy.nan]])
    assert view.clamp_min_(min=-2) is view
    assert view.clamp_(min=-2, max=2) is view
    assert_values(numpy, x, [[-2, 99], [0.5, 99], [2, 99], [numpy.nan, 99]])


@pytest.mark.parametrize("name,expected", [
    ("SUM", 10), ("MEAN", 2.5), ("MAX", 4), ("MIN", 1), ("PROD", 24),
    ("ANY", 1), ("ALL", 1), ("STD", (5 / 3) ** 0.5), ("VAR", 5 / 3),
    ("ARGMAX", 3), ("ARGMIN", 0),
])
def test_reduce_enum(lf, tensor, name, expected):
    dtype = "bool" if name in ("ANY", "ALL") else "float32"
    result = tensor([1, 2, 3, 4], dtype).reduce(getattr(lf.ReduceOp, name))
    assert result.item() == pytest.approx(expected, rel=1e-5)


def test_reduce_dimension(lf, numpy, tensor):
    result = tensor([[1, 2], [3, 4]]).reduce(lf.ReduceOp.SUM, dim=-1, keepdim=True)
    assert result.shape == (2, 1)
    assert_values(numpy, result, [[3], [7]])


@pytest.mark.parametrize("name", ["all_close", "allclose"])
def test_all_close(tensor, name):
    x = tensor([1, 2])
    close = getattr(x, name)
    assert close(x.clone())
    assert not close(x + 1)
    assert close(x + 0.01, rtol=0, atol=0.02)
    assert close(x + 0.01, rtol=0.02, atol=0)
    assert not getattr(tensor([float("nan")]), name)(tensor([float("nan")]))


def test_nonzero_split(numpy, tensor):
    x = tensor([[0, 2], [3, 0]])
    indices = x.nonzero_split()
    assert len(indices) == 2
    for actual, expected in zip(indices, ([0, 1], [1, 0])):
        assert actual.dtype == "int64"
        assert actual.backend == x.backend
        assert_values(numpy, actual, expected)
    assert all(t.numel == 0 for t in tensor([[0, 0]]).nonzero_split())


@pytest.mark.parametrize("bias", [False, True])
def test_linear_and_conv1x1(numpy, tensor, bias):
    x = tensor([[1, 2], [3, 4]])
    weight = tensor([[2, -1], [1, 3]])
    expected = numpy.array([[0, 7], [2, 15]], dtype=numpy.float32)
    b = tensor([5, -2]) if bias else None
    if bias:
        expected += [5, -2]
    assert_values(numpy, x.linear(weight, bias=b), expected)
    if not bias:
        assert_values(numpy, x.linear(weight), expected)
    image = x.t().contiguous().reshape([1, 2, 1, 2])
    assert_values(numpy, image.conv1x1(weight, bias=b), expected.T.reshape(1, 2, 1, 2))
    if not bias:
        assert_values(numpy, image.conv1x1(weight), expected.T.reshape(1, 2, 1, 2))


def test_element_assignment(numpy, tensor):
    x = tensor([[1, 2], [3, 4]])
    x[0, 1] = 7
    x[-1, -1] = -2
    x.t()[1, 0] = 9
    assert_values(numpy, x, [[1, 9], [3, -2]])


def test_where_into_overlap(numpy, tensor):
    x = tensor([1, 2, 3, 4])
    output = x[1:]
    condition = tensor([0, 1, 0], "bool")
    assert output.where_into_(condition, value=8, source=x[:3]) is output
    assert_values(numpy, x, [1, 1, 8, 3])


@pytest.mark.parametrize("dtype", ["float32", "float16", "int32"])
def test_gather_evaluated(numpy, tensor, dtype):
    x = tensor([4, -1, 9, -3], dtype)
    indices = tensor([[3, 9], [0, 9], [1, 9]], "int32")[:, 0]
    result = x.gather_lazy(indices)
    assert result.backend == x.backend
    assert result.dtype == dtype
    x[3] = tensor([42], dtype)
    assert_values(numpy, result, [-3, 4, -1])


def test_inspection(numpy, tensor):
    x = tensor([[1, 99], [3, 99], [5, 99]])[:, 0]
    validation = x.validate()
    assert validation == dict(is_valid=True, has_nan=False, has_inf=False,
                              nan_count=0, inf_count=0, min_val=1, max_val=5, mean_val=3)
    invalid = tensor([1, float("nan"), float("inf"), -float("inf")]).validate()
    assert not invalid["is_valid"]
    assert invalid["has_nan"] and invalid["has_inf"]
    assert invalid["nan_count"] == 1 and invalid["inf_count"] == 2
    stats = x.stats()
    assert stats["mean"] == 3 and stats["min"] == 1 and stats["max"] == 5
    assert stats["std"] == pytest.approx(numpy.std([1, 3, 5]))
    assert stats["shape"] == (3,) and stats["numel"] == 3
    assert stats["dtype"] == "float32" and stats["backend"] == x.backend
    assert stats["is_cuda"] == x.is_cuda
    diff = x.diff(tensor([1, 3, 7]), tolerance=0.5)
    assert diff["shapes_match"] and diff["dtypes_match"]
    assert diff["num_different"] == 1 and diff["total_elements"] == 3
    assert diff["max_abs_diff"] == 2
    assert diff["mean_abs_diff"] == pytest.approx(2 / 3)
    assert diff["max_rel_diff"] == pytest.approx(0.4)
    assert x.diff(x.clone())["num_different"] == 0
    assert not x.diff(tensor([1]))["shapes_match"]
    assert not x.diff(x.to("float16"))["dtypes_match"]


def test_reserved_bytes(lf, device):
    x = lf.Tensor.zeros([4, 3], device=device)
    assert x.reserved_allocation_bytes >= 4 * 3 * 4
    view_bytes = x[:2].reserved_allocation_bytes
    assert view_bytes is None or view_bytes >= 2 * 3 * 4
    assert lf.Tensor().reserved_allocation_bytes is None


@pytest.mark.parametrize("operation", [
    lambda lf, t: lf.Tensor.normal([2], device="invalid"),
    lambda lf, t: lf.Tensor.normal([2], device="cpu", dtype="float16"),
    lambda lf, t: t([1, 2]).reduce(lf.ReduceOp.COUNT_NONZERO),
    lambda lf, t: t([1, 2]).reduce(lf.ReduceOp.NORM),
    lambda lf, t: t([1, 2]).gather_lazy(t([0], "int64")),
    lambda lf, t: lf.Tensor.bernoulli([2], dtype="invalid"),
    lambda lf, t: lf.Tensor.bernoulli([2], p=2, device="cpu"),
    lambda lf, t: lf.Tensor.multinomial(t([-1, 2]), 1),
    lambda lf, t: lf.Tensor.diag(t([[1, 2]])),
    lambda lf, t: t([[1, 2]]).cdist(t([[1]])),
    lambda lf, t: t([1, 2]).normalize(eps=0),
    lambda lf, t: t([1, 2]).reduce("SUM"),
    lambda lf, t: t([1, 2]).linear(t([[1]])),
    lambda lf, t: t([1, 2]).conv1x1(t([[1, 2]])),
    lambda lf, t: t([1, 2]).diff(t([1, 2]), tolerance=-1),
])
def test_errors_reach_python(lf, tensor, operation):
    with pytest.raises((RuntimeError, ValueError, TypeError, lf.Error)):
        operation(lf, tensor)


def test_inplace_methods_in_generated_stubs():
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    stub = ast.parse((root / "src/python/stubs/lichtfeld/__init__.pyi").read_text())
    tensor_class = next(node for node in stub.body if isinstance(node, ast.ClassDef) and node.name == "Tensor")
    methods = {node.name for node in tensor_class.body if isinstance(node, ast.FunctionDef)}
    assert {"clamp_", "clamp_min_", "where_into_", "fill_", "zero_", "masked_fill_"} <= methods
    assert not any(node.name.startswith("_testing") for node in stub.body if isinstance(node, ast.ClassDef))
