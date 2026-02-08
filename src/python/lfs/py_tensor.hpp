/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"
#include <dlpack/dlpack.h>
#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

namespace lfs::python {

    class PyTensor {
    public:
        PyTensor() = default;
        explicit PyTensor(core::Tensor tensor, bool owns_data = true);
        ~PyTensor();

        // Copy and move operations
        PyTensor(const PyTensor& other);
        PyTensor& operator=(const PyTensor& other);
        PyTensor(PyTensor&& other) noexcept;
        PyTensor& operator=(PyTensor&& other) noexcept;

        // Properties
        nb::tuple shape() const;
        size_t ndim() const;
        size_t numel() const;
        std::string device() const;
        std::string dtype() const;
        bool is_contiguous() const;
        bool is_cuda() const;
        size_t size(int dim) const;

        // Memory operations
        PyTensor clone() const;
        PyTensor cpu() const;
        PyTensor cuda() const;
        PyTensor contiguous() const;
        void sync() const;

        // Scalar extraction
        float item() const;
        float item_float() const { return item(); }
        int64_t item_int() const;
        bool item_bool() const;

        // NumPy conversion
        nb::object numpy(bool copy = true) const;

        // Static factory: create from NumPy
        static PyTensor from_numpy(nb::ndarray<> arr, bool copy = true);

        // Slicing (Phase 3)
        PyTensor getitem(const nb::object& key) const;
        void setitem(const nb::object& key, const nb::object& value);

        // Arithmetic operators (Phase 4)
        PyTensor add(const PyTensor& other) const;
        PyTensor add_scalar(float scalar) const;
        PyTensor sub(const PyTensor& other) const;
        PyTensor sub_scalar(float scalar) const;
        PyTensor rsub_scalar(float scalar) const;
        PyTensor mul(const PyTensor& other) const;
        PyTensor mul_scalar(float scalar) const;
        PyTensor div(const PyTensor& other) const;
        PyTensor div_scalar(float scalar) const;
        PyTensor rdiv_scalar(float scalar) const;
        PyTensor neg() const;
        PyTensor abs() const;
        PyTensor sigmoid() const;
        PyTensor exp() const;
        PyTensor log() const;
        PyTensor sqrt() const;
        PyTensor relu() const;

        // Additional unary math operations
        PyTensor sin() const;
        PyTensor cos() const;
        PyTensor tan() const;
        PyTensor tanh() const;
        PyTensor floor() const;
        PyTensor ceil() const;
        PyTensor round() const;

        // Extended unary operations
        PyTensor log2() const;
        PyTensor log10() const;
        PyTensor log1p() const;
        PyTensor exp2() const;
        PyTensor rsqrt() const;
        PyTensor square() const;
        PyTensor asin() const;
        PyTensor acos() const;
        PyTensor atan() const;
        PyTensor sinh() const;
        PyTensor cosh() const;
        PyTensor trunc() const;
        PyTensor sign() const;
        PyTensor reciprocal() const;
        PyTensor gelu() const;
        PyTensor swish() const;
        PyTensor isnan() const;
        PyTensor isinf() const;
        PyTensor isfinite() const;

        // Power operations
        PyTensor pow(float exponent) const;
        PyTensor pow(const PyTensor& exponent) const;

        // In-place arithmetic
        PyTensor& iadd(const PyTensor& other);
        PyTensor& iadd_scalar(float scalar);
        PyTensor& isub(const PyTensor& other);
        PyTensor& isub_scalar(float scalar);
        PyTensor& imul(const PyTensor& other);
        PyTensor& imul_scalar(float scalar);
        PyTensor& idiv(const PyTensor& other);
        PyTensor& idiv_scalar(float scalar);
        PyTensor& fill_(float value);
        PyTensor& zero_();

        // Comparison operators (return Bool tensor)
        PyTensor eq(const PyTensor& other) const;
        PyTensor eq_scalar(float scalar) const;
        PyTensor ne(const PyTensor& other) const;
        PyTensor ne_scalar(float scalar) const;
        PyTensor lt(const PyTensor& other) const;
        PyTensor lt_scalar(float scalar) const;
        PyTensor le(const PyTensor& other) const;
        PyTensor le_scalar(float scalar) const;
        PyTensor gt(const PyTensor& other) const;
        PyTensor gt_scalar(float scalar) const;
        PyTensor ge(const PyTensor& other) const;
        PyTensor ge_scalar(float scalar) const;

        // Logical operators (Bool tensors)
        PyTensor logical_and(const PyTensor& other) const;
        PyTensor logical_or(const PyTensor& other) const;
        PyTensor logical_not() const;

        // Reduction operations
        PyTensor sum(std::optional<int> dim = std::nullopt, bool keepdim = false) const;
        PyTensor mean(std::optional<int> dim = std::nullopt, bool keepdim = false) const;
        PyTensor max(std::optional<int> dim = std::nullopt, bool keepdim = false) const;
        PyTensor min(std::optional<int> dim = std::nullopt, bool keepdim = false) const;
        float sum_scalar() const;
        float mean_scalar() const;
        float max_scalar() const;
        float min_scalar() const;

        // Extended reductions
        PyTensor prod(std::optional<int> dim = std::nullopt, bool keepdim = false) const;
        PyTensor std(std::optional<int> dim = std::nullopt, bool keepdim = false) const;
        PyTensor var(std::optional<int> dim = std::nullopt, bool keepdim = false) const;
        PyTensor argmax(std::optional<int> dim = std::nullopt, bool keepdim = false) const;
        PyTensor argmin(std::optional<int> dim = std::nullopt, bool keepdim = false) const;
        PyTensor all(std::optional<int> dim = std::nullopt, bool keepdim = false) const;
        PyTensor any(std::optional<int> dim = std::nullopt, bool keepdim = false) const;
        PyTensor norm(float p = 2.0f) const;
        float norm_scalar(float p = 2.0f) const;

        // Shape operations
        PyTensor reshape(const std::vector<int64_t>& new_shape) const;
        PyTensor view(const std::vector<int64_t>& new_shape) const;
        PyTensor squeeze(std::optional<int> dim = std::nullopt) const;
        PyTensor unsqueeze(int dim) const;
        PyTensor transpose(int dim0, int dim1) const;
        PyTensor permute(const std::vector<int>& dims) const;
        PyTensor flatten(int start_dim = 0, int end_dim = -1) const;
        PyTensor expand(const std::vector<int64_t>& sizes) const;
        PyTensor repeat(const std::vector<int64_t>& repeats) const;
        PyTensor t() const;

        // Advanced indexing
        PyTensor index_select(int dim, const PyTensor& indices) const;
        PyTensor gather(int dim, const PyTensor& indices) const;
        PyTensor masked_select(const PyTensor& mask) const;
        PyTensor masked_fill(const PyTensor& mask, float value) const;
        PyTensor& masked_fill_(const PyTensor& mask, float value);
        PyTensor nonzero() const;

        // Linear algebra
        PyTensor matmul(const PyTensor& other) const;
        PyTensor mm(const PyTensor& other) const;
        PyTensor bmm(const PyTensor& other) const;
        PyTensor dot(const PyTensor& other) const;

        // Element-wise operations
        PyTensor clamp(float min_val, float max_val) const;
        PyTensor maximum(const PyTensor& other) const;
        PyTensor minimum(const PyTensor& other) const;

        // Type conversion
        PyTensor to_dtype(const std::string& dtype) const;

        // String representation
        std::string repr() const;

        // DLPack protocol for zero-copy tensor exchange
        nb::tuple dlpack_device() const;
        nb::capsule dlpack(nb::object stream = nb::none()) const;
        static PyTensor from_dlpack(nb::object obj);

        // Static creation functions
        static PyTensor zeros(const std::vector<int64_t>& shape,
                              const std::string& device = "cuda",
                              const std::string& dtype = "float32");
        static PyTensor ones(const std::vector<int64_t>& shape,
                             const std::string& device = "cuda",
                             const std::string& dtype = "float32");
        static PyTensor full(const std::vector<int64_t>& shape, float value,
                             const std::string& device = "cuda",
                             const std::string& dtype = "float32");
        static PyTensor arange(float end);
        static PyTensor arange(float start, float end, float step = 1.0f,
                               const std::string& device = "cuda",
                               const std::string& dtype = "float32");
        static PyTensor linspace(float start, float end, int64_t steps,
                                 const std::string& device = "cuda",
                                 const std::string& dtype = "float32");
        static PyTensor eye(int64_t n, const std::string& device = "cuda",
                            const std::string& dtype = "float32");
        static PyTensor eye(int64_t m, int64_t n, const std::string& device = "cuda",
                            const std::string& dtype = "float32");

        // Random tensor creation
        static PyTensor rand(const std::vector<int64_t>& shape,
                             const std::string& device = "cuda",
                             const std::string& dtype = "float32");
        static PyTensor randn(const std::vector<int64_t>& shape,
                              const std::string& device = "cuda",
                              const std::string& dtype = "float32");
        static PyTensor empty(const std::vector<int64_t>& shape,
                              const std::string& device = "cuda",
                              const std::string& dtype = "float32");
        static PyTensor randint(int64_t low, int64_t high,
                                const std::vector<int64_t>& shape,
                                const std::string& device = "cuda");

        // *_like variants
        static PyTensor zeros_like(const PyTensor& other);
        static PyTensor ones_like(const PyTensor& other);
        static PyTensor rand_like(const PyTensor& other);
        static PyTensor randn_like(const PyTensor& other);
        static PyTensor empty_like(const PyTensor& other);
        static PyTensor full_like(const PyTensor& other, float value);

        // Tensor combination
        static PyTensor cat(const std::vector<PyTensor>& tensors, int dim = 0);
        static PyTensor stack(const std::vector<PyTensor>& tensors, int dim = 0);
        static PyTensor where(const PyTensor& condition, const PyTensor& x, const PyTensor& y);

        // Access underlying tensor (for internal use)
        const core::Tensor& tensor() const { return tensor_; }
        core::Tensor& tensor() { return tensor_; }

        // Factory for non-owning view with generation tracking
        static PyTensor view_of(core::Tensor& t, uint64_t generation);

        // Validate tensor is still valid (for non-owning views)
        void validate() const;

    private:
        core::Tensor tensor_;
        bool owns_data_ = true;
        uint64_t source_gen_ = 0; // 0 = owning, no check needed

        // DLPack: shared ownership of managed tensor - deleter called when last copy destroyed
        std::shared_ptr<DLManagedTensor> dlpack_managed_;

        // Helper to parse Python slice
        struct SliceInfo {
            int64_t start;
            int64_t stop;
            int64_t step;
        };
        SliceInfo parse_slice(const nb::slice& sl, size_t dim_size) const;
    };

    // Register PyTensor with nanobind
    void register_tensor(nb::module_& m);

} // namespace lfs::python
