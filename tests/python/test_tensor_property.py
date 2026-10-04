# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for TensorProperty - GPU-native tensor storage in property groups.

These tests verify that TensorProperty correctly validates shape, dtype, and
device constraints, and integrates properly with the property system.
"""

import sys
from pathlib import Path

import pytest


@pytest.fixture
def lfs_props():
    """Import lfs_plugins.props module."""
    project_root = Path(__file__).parent.parent.parent
    build_python = project_root / "build" / "src" / "python"
    if str(build_python) not in sys.path:
        sys.path.insert(0, str(build_python))

    try:
        from lfs_plugins import props

        return props
    except ImportError as e:
        pytest.skip(f"lfs_plugins.props module not available: {e}")


@pytest.fixture
def lfs_types():
    """Import lfs_plugins.types module."""
    project_root = Path(__file__).parent.parent.parent
    build_python = project_root / "build" / "src" / "python"
    if str(build_python) not in sys.path:
        sys.path.insert(0, str(build_python))

    try:
        from lfs_plugins import types

        return types
    except ImportError as e:
        pytest.skip(f"lfs_plugins.types module not available: {e}")


class TestTensorPropertyBasic:
    """Basic tests for TensorProperty instantiation and attributes."""

    def test_tensor_property_exists(self, lfs_props):
        """TensorProperty class should exist in props module."""
        assert hasattr(lfs_props, "TensorProperty")

    def test_tensor_property_default_values(self, lfs_props):
        """TensorProperty should have sensible defaults."""
        prop = lfs_props.TensorProperty()
        assert prop.shape == ()
        assert prop.dtype == "float32"
        assert prop.device == "cuda"
        assert prop.default is None

    def test_tensor_property_custom_shape(self, lfs_props):
        """TensorProperty should accept custom shape."""
        prop = lfs_props.TensorProperty(shape=(-1, 3))
        assert prop.shape == (-1, 3)

    def test_tensor_property_custom_dtype(self, lfs_props):
        """TensorProperty should accept custom dtype."""
        prop = lfs_props.TensorProperty(dtype="float16")
        assert prop.dtype == "float16"

    def test_tensor_property_custom_device(self, lfs_props):
        """TensorProperty should accept custom device."""
        prop = lfs_props.TensorProperty(device="cpu")
        assert prop.device == "cpu"

    def test_tensor_property_with_name(self, lfs_props):
        """TensorProperty should accept name parameter."""
        prop = lfs_props.TensorProperty(name="My Tensor", shape=(-1, 3))
        assert prop.name == "My Tensor"


class TestTensorPropertyValidation:
    """Tests for TensorProperty validation logic."""

    def test_validate_none_allowed(self, lfs_props):
        """None should be a valid value (empty property)."""
        prop = lfs_props.TensorProperty(shape=(-1, 3), dtype="float32")
        result = prop.validate(None)
        assert result is None

    def test_validate_correct_tensor(self, lf, lfs_props, numpy):
        """Correctly shaped tensor should pass validation."""
        prop = lfs_props.TensorProperty(shape=(-1, 3), dtype="float32", device="cpu")

        t = lf.Tensor.zeros([100, 3], dtype="float32", device="cpu")
        result = prop.validate(t)
        assert result is t

    def test_validate_wrong_dtype_raises(self, lf, lfs_props, numpy):
        """Wrong dtype should raise ValueError."""
        prop = lfs_props.TensorProperty(shape=(-1, 3), dtype="float32", device="cpu")

        t = lf.Tensor.zeros([100, 3], dtype="float16", device="cpu")
        with pytest.raises(ValueError, match="dtype"):
            prop.validate(t)

    def test_validate_wrong_device_raises(self, lf, lfs_props, numpy):
        """Wrong device should raise ValueError."""
        prop = lfs_props.TensorProperty(shape=(-1, 3), dtype="float32", device="cuda")

        t = lf.Tensor.zeros([100, 3], dtype="float32", device="cpu")
        with pytest.raises(ValueError, match="device"):
            prop.validate(t)

    def test_validate_wrong_ndim_raises(self, lf, lfs_props, numpy):
        """Wrong number of dimensions should raise ValueError."""
        prop = lfs_props.TensorProperty(shape=(-1, 3), dtype="float32", device="cpu")

        # 1D tensor when expecting 2D
        t = lf.Tensor.zeros([100], dtype="float32", device="cpu")
        with pytest.raises(ValueError, match="tensor"):
            prop.validate(t)

    def test_validate_wrong_fixed_dim_raises(self, lf, lfs_props, numpy):
        """Wrong fixed dimension size should raise ValueError."""
        prop = lfs_props.TensorProperty(shape=(-1, 3), dtype="float32", device="cpu")

        # Second dimension is 4, not 3
        t = lf.Tensor.zeros([100, 4], dtype="float32", device="cpu")
        with pytest.raises(ValueError, match="Shape mismatch"):
            prop.validate(t)

    def test_validate_variable_dim_any_size(self, lf, lfs_props, numpy):
        """Variable dimension (-1) should accept any size."""
        prop = lfs_props.TensorProperty(shape=(-1, 3), dtype="float32", device="cpu")

        # All these should be valid
        for n in [1, 10, 100, 1000]:
            t = lf.Tensor.zeros([n, 3], dtype="float32", device="cpu")
            result = prop.validate(t)
            assert result is t

    def test_validate_non_tensor_raises(self, lfs_props, numpy):
        """Non-tensor value should raise TypeError."""
        prop = lfs_props.TensorProperty(shape=(-1, 3), dtype="float32", device="cpu")

        with pytest.raises(TypeError, match="Tensor"):
            prop.validate([1, 2, 3])

        with pytest.raises(TypeError, match="Tensor"):
            prop.validate(numpy.zeros([100, 3]))

    def test_validate_exact_shape(self, lf, lfs_props, numpy):
        """Exact shape (no -1) should only accept matching tensor."""
        prop = lfs_props.TensorProperty(shape=(100, 3), dtype="float32", device="cpu")

        # Exact match
        t1 = lf.Tensor.zeros([100, 3], dtype="float32", device="cpu")
        assert prop.validate(t1) is t1

        # Wrong first dimension
        t2 = lf.Tensor.zeros([50, 3], dtype="float32", device="cpu")
        with pytest.raises(ValueError, match="Shape mismatch"):
            prop.validate(t2)


class TestTensorPropertyDtypes:
    """Tests for various dtype support."""

    @pytest.mark.parametrize(
        "dtype",
        ["float32", "float16", "int32", "int64", "bool"],
    )
    def test_dtype_validation(self, lf, lfs_props, numpy, dtype):
        """TensorProperty should validate various dtypes correctly."""
        prop = lfs_props.TensorProperty(shape=(-1,), dtype=dtype, device="cpu")

        # Correct dtype passes
        t_correct = lf.Tensor.zeros([10], dtype=dtype, device="cpu")
        assert prop.validate(t_correct) is t_correct

        # Wrong dtype fails
        wrong_dtype = "int32" if dtype != "int32" else "float32"
        t_wrong = lf.Tensor.zeros([10], dtype=wrong_dtype, device="cpu")
        with pytest.raises(ValueError, match="dtype"):
            prop.validate(t_wrong)


class TestTensorPropertyInPropertyGroup:
    """Tests for TensorProperty used within PropertyGroup."""

    def test_property_group_with_tensor(self, lf, lfs_props, lfs_types, numpy):
        """PropertyGroup should support TensorProperty."""

        class Settings(lfs_types.PropertyGroup):
            positions = lfs_props.TensorProperty(
                shape=(-1, 3), dtype="float32", device="cpu"
            )

        # Create instance
        settings = Settings()

        # Initially None
        assert settings.positions is None

        # Set valid tensor
        t = lf.Tensor.zeros([100, 3], dtype="float32", device="cpu")
        settings.positions = t
        assert settings.positions is not None
        assert tuple(settings.positions.shape) == (100, 3)

    def test_property_group_multiple_tensors(self, lf, lfs_props, lfs_types, numpy):
        """PropertyGroup should support multiple TensorProperties."""

        class MultiTensorSettings(lfs_types.PropertyGroup):
            positions = lfs_props.TensorProperty(
                shape=(-1, 3), dtype="float32", device="cpu"
            )
            colors = lfs_props.TensorProperty(
                shape=(-1, 3), dtype="float32", device="cpu"
            )
            mask = lfs_props.TensorProperty(shape=(-1,), dtype="bool", device="cpu")

        settings = MultiTensorSettings()

        # Set all properties
        settings.positions = lf.Tensor.zeros([100, 3], dtype="float32", device="cpu")
        settings.colors = lf.Tensor.ones([100, 3], dtype="float32", device="cpu")
        settings.mask = lf.Tensor.zeros([100], dtype="bool", device="cpu")

        assert tuple(settings.positions.shape) == (100, 3)
        assert tuple(settings.colors.shape) == (100, 3)
        assert tuple(settings.mask.shape) == (100,)

    def test_property_group_tensor_validation_on_set(self, lf, lfs_props, lfs_types, numpy):
        """Setting invalid tensor on PropertyGroup should raise."""

        class ValidatedSettings(lfs_types.PropertyGroup):
            data = lfs_props.TensorProperty(
                shape=(-1, 3), dtype="float32", device="cpu"
            )

        settings = ValidatedSettings()

        # Valid tensor works
        t_valid = lf.Tensor.zeros([10, 3], dtype="float32", device="cpu")
        settings.data = t_valid

        # Invalid shape fails
        t_invalid = lf.Tensor.zeros([10, 4], dtype="float32", device="cpu")
        with pytest.raises(ValueError):
            settings.data = t_invalid

    def test_property_group_clear_tensor(self, lf, lfs_props, lfs_types, numpy):
        """Should be able to clear tensor by setting to None."""

        class ClearableSettings(lfs_types.PropertyGroup):
            tensor = lfs_props.TensorProperty(
                shape=(-1, 3), dtype="float32", device="cpu"
            )

        settings = ClearableSettings()

        # Set tensor
        t = lf.Tensor.zeros([50, 3], dtype="float32", device="cpu")
        settings.tensor = t
        assert settings.tensor is not None

        # Clear tensor
        settings.tensor = None
        assert settings.tensor is None


class TestTensorPropertyShapePatterns:
    """Tests for various shape pattern specifications."""

    def test_scalar_shape(self, lf, lfs_props, numpy):
        """Empty shape tuple means any shape (like a wildcard)."""
        prop = lfs_props.TensorProperty(shape=(), dtype="float32", device="cpu")

        # Any shape should be valid with empty shape tuple
        t1 = lf.Tensor.zeros([10], dtype="float32", device="cpu")
        t2 = lf.Tensor.zeros([10, 20], dtype="float32", device="cpu")
        t3 = lf.Tensor.zeros([10, 20, 30], dtype="float32", device="cpu")

        # All should pass (empty shape means no shape constraint)
        assert prop.validate(t1) is t1
        assert prop.validate(t2) is t2
        assert prop.validate(t3) is t3

    def test_1d_variable(self, lf, lfs_props, numpy):
        """1D variable shape (-1,)."""
        prop = lfs_props.TensorProperty(shape=(-1,), dtype="float32", device="cpu")

        t1 = lf.Tensor.zeros([10], dtype="float32", device="cpu")
        t2 = lf.Tensor.zeros([1000], dtype="float32", device="cpu")

        assert prop.validate(t1) is t1
        assert prop.validate(t2) is t2

        # 2D should fail
        t_2d = lf.Tensor.zeros([10, 3], dtype="float32", device="cpu")
        with pytest.raises(ValueError):
            prop.validate(t_2d)

    def test_2d_n_by_3(self, lf, lfs_props, numpy):
        """Common pattern: Nx3 for positions/colors."""
        prop = lfs_props.TensorProperty(shape=(-1, 3), dtype="float32", device="cpu")

        t = lf.Tensor.zeros([1000, 3], dtype="float32", device="cpu")
        assert prop.validate(t) is t

    def test_2d_n_by_4(self, lf, lfs_props, numpy):
        """Common pattern: Nx4 for quaternions/RGBA."""
        prop = lfs_props.TensorProperty(shape=(-1, 4), dtype="float32", device="cpu")

        t = lf.Tensor.zeros([500, 4], dtype="float32", device="cpu")
        assert prop.validate(t) is t

    def test_3d_shape(self, lf, lfs_props, numpy):
        """3D tensor shape."""
        prop = lfs_props.TensorProperty(
            shape=(-1, -1, 3), dtype="float32", device="cpu"
        )

        t = lf.Tensor.zeros([10, 20, 3], dtype="float32", device="cpu")
        assert prop.validate(t) is t

        # Wrong last dimension
        t_wrong = lf.Tensor.zeros([10, 20, 4], dtype="float32", device="cpu")
        with pytest.raises(ValueError):
            prop.validate(t_wrong)

    def test_all_variable_dims(self, lf, lfs_props, numpy):
        """All variable dimensions."""
        prop = lfs_props.TensorProperty(
            shape=(-1, -1, -1), dtype="float32", device="cpu"
        )

        t1 = lf.Tensor.zeros([10, 20, 30], dtype="float32", device="cpu")
        t2 = lf.Tensor.zeros([1, 1, 1], dtype="float32", device="cpu")
        t3 = lf.Tensor.zeros([100, 100, 100], dtype="float32", device="cpu")

        assert prop.validate(t1) is t1
        assert prop.validate(t2) is t2
        assert prop.validate(t3) is t3

        # 2D should fail
        t_2d = lf.Tensor.zeros([10, 20], dtype="float32", device="cpu")
        with pytest.raises(ValueError):
            prop.validate(t_2d)


class TestTensorPropertyGPU:
    """Tests for GPU tensor support."""

    @pytest.mark.gpu
    def test_cuda_tensor(self, lf, lfs_props, numpy, gpu_available):
        """TensorProperty should work with CUDA tensors."""
        if not gpu_available:
            pytest.skip("GPU not available")

        prop = lfs_props.TensorProperty(shape=(-1, 3), dtype="float32", device="cuda")

        t = lf.Tensor.zeros([100, 3], dtype="float32", device="cuda")
        assert prop.validate(t) is t

    @pytest.mark.gpu
    def test_cuda_device_validation(self, lf, lfs_props, numpy, gpu_available):
        """CPU tensor should fail validation for cuda property."""
        if not gpu_available:
            pytest.skip("GPU not available")

        prop = lfs_props.TensorProperty(shape=(-1, 3), dtype="float32", device="cuda")

        t_cpu = lf.Tensor.zeros([100, 3], dtype="float32", device="cpu")
        with pytest.raises(ValueError, match="device"):
            prop.validate(t_cpu)

    @pytest.mark.gpu
    def test_property_group_with_cuda_tensor(self, lf, lfs_props, lfs_types, numpy, gpu_available):
        """PropertyGroup should work with CUDA TensorProperty."""
        if not gpu_available:
            pytest.skip("GPU not available")

        class GPUSettings(lfs_types.PropertyGroup):
            gpu_data = lfs_props.TensorProperty(
                shape=(-1, 3), dtype="float32", device="cuda"
            )

        settings = GPUSettings()
        t = lf.Tensor.zeros([50, 3], dtype="float32", device="cuda")
        settings.gpu_data = t

        assert settings.gpu_data is not None
        assert str(settings.gpu_data.device) == "cuda"


class TestTensorPropertyGetAllProperties:
    """Tests for TensorProperty with get_all_properties introspection."""

    def test_tensor_property_in_all_properties(self, lf, lfs_props, lfs_types, numpy):
        """TensorProperty should appear in get_all_properties."""

        class IntrospectionSettings(lfs_types.PropertyGroup):
            tensor_data = lfs_props.TensorProperty(
                shape=(-1, 3),
                dtype="float32",
                device="cpu",
                name="Tensor Data",
                description="Test tensor property",
            )

        settings = IntrospectionSettings()
        all_props = settings.get_all_properties()

        assert "tensor_data" in all_props
        prop_desc = all_props["tensor_data"]
        assert prop_desc.__class__.__name__ == "TensorProperty"

    def test_tensor_property_attributes_preserved(self, lf, lfs_props, lfs_types, numpy):
        """TensorProperty metadata should be preserved."""

        class MetadataSettings(lfs_types.PropertyGroup):
            data = lfs_props.TensorProperty(
                shape=(-1, 4),
                dtype="float16",
                device="cpu",
                name="Quaternions",
                description="Rotation quaternions",
            )

        settings = MetadataSettings()
        all_props = settings.get_all_properties()
        prop_desc = all_props["data"]

        assert prop_desc.shape == (-1, 4)
        assert prop_desc.dtype == "float16"
        assert prop_desc.device == "cpu"
        assert prop_desc.name == "Quaternions"
        assert prop_desc.description == "Rotation quaternions"
