/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "core/mesh_data.hpp"
#include "core/point_cloud.hpp"
#include "core/tensor.hpp"
#include "licht_test_support.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <format>
#include <memory>
#include <new>
#include <span>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace lfs::core {

    std::atomic<std::size_t> Tensor::next_id_{1};

    uint64_t MeshData::next_id() {
        static std::atomic<uint64_t> counter{0};
        return counter.fetch_add(1, std::memory_order_relaxed) + 1;
    }

    namespace {

        [[noreturn]] void unsupported_cuda() {
            throw std::runtime_error(
                "the CPU-only format test runtime cannot create CUDA tensors");
        }

        double tensor_value(const Tensor& tensor, const std::size_t index) {
            switch (tensor.dtype()) {
            case DataType::Float32:
                return tensor.ptr<float>()[index];
            case DataType::Float16:
                return __half2float(tensor.ptr<__half>()[index]);
            case DataType::Int32:
                return tensor.ptr<std::int32_t>()[index];
            case DataType::Int64:
                return static_cast<double>(tensor.ptr<std::int64_t>()[index]);
            case DataType::UInt8:
            case DataType::Bool:
                return tensor.ptr<std::uint8_t>()[index];
            }
            throw std::runtime_error("unsupported tensor dtype");
        }

        void set_tensor_value(
            Tensor& tensor, const std::size_t index, const double value) {
            switch (tensor.dtype()) {
            case DataType::Float32:
                tensor.ptr<float>()[index] = static_cast<float>(value);
                return;
            case DataType::Float16:
                tensor.ptr<__half>()[index] =
                    __float2half(static_cast<float>(value));
                return;
            case DataType::Int32:
                tensor.ptr<std::int32_t>()[index] =
                    static_cast<std::int32_t>(value);
                return;
            case DataType::Int64:
                tensor.ptr<std::int64_t>()[index] =
                    static_cast<std::int64_t>(value);
                return;
            case DataType::UInt8:
                tensor.ptr<std::uint8_t>()[index] =
                    static_cast<std::uint8_t>(value);
                return;
            case DataType::Bool:
                tensor.ptr<std::uint8_t>()[index] = value != 0.0;
                return;
            }
            throw std::runtime_error("unsupported tensor dtype");
        }

    } // namespace

    namespace tensor_contract {

        void require_valid(
            const Tensor& tensor, const std::string_view operation,
            const std::string_view role, const SourceSite) {
            if (!tensor.is_valid()) {
                throw std::runtime_error(std::format(
                    "{} requires a valid {} tensor", operation, role));
            }
        }

        void require_same_device(
            const Tensor& reference, const Tensor& other,
            const std::string_view operation,
            const std::string_view reference_role,
            const std::string_view other_role, const SourceSite location) {
            require_valid(reference, operation, reference_role, location);
            require_valid(other, operation, other_role, location);
            if (reference.device() != other.device()) {
                throw std::runtime_error(
                    std::format("{} requires matching tensor devices", operation));
            }
        }

        void require_dtype(
            const Tensor& tensor, const DataType expected,
            const std::string_view operation, const std::string_view role,
            const SourceSite location) {
            require_valid(tensor, operation, role, location);
            if (tensor.dtype() != expected) {
                throw std::runtime_error(
                    std::format("{} received the wrong {} dtype", operation, role));
            }
        }

        void require_dtype(
            const Tensor& tensor,
            const std::initializer_list<DataType> expected,
            const std::string_view operation, const std::string_view role,
            const SourceSite location) {
            require_valid(tensor, operation, role, location);
            if (std::ranges::find(expected, tensor.dtype()) == expected.end()) {
                throw std::runtime_error(
                    std::format("{} received the wrong {} dtype", operation, role));
            }
        }

        void require_shape(
            const Tensor& reference, const Tensor& other,
            const std::string_view operation,
            const std::string_view reference_role,
            const std::string_view other_role, const SourceSite location) {
            require_valid(reference, operation, reference_role, location);
            require_valid(other, operation, other_role, location);
            if (reference.shape() != other.shape()) {
                throw std::runtime_error(
                    std::format("{} requires matching tensor shapes", operation));
            }
        }

        void require_shape(
            const Tensor& tensor, const TensorShape& expected,
            const std::string_view operation, const std::string_view role,
            const SourceSite location) {
            require_valid(tensor, operation, role, location);
            if (tensor.shape() != expected) {
                throw std::runtime_error(
                    std::format("{} received the wrong {} shape", operation, role));
            }
        }

    } // namespace tensor_contract

    Tensor::Tensor(
        void* data, TensorShape shape, const Device device,
        const DataType dtype, const cudaStream_t home_stream)
        : data_(data),
          shape_(std::move(shape)),
          strides_(shape_.strides()),
          device_(device),
          dtype_(dtype),
          is_view_(true),
          id_(next_id_++) {
        if (device != Device::CPU) {
            unsupported_cuda();
        }
        state_->stream = home_stream;
        init_storage_meta();
        compute_alignment();
    }

    Tensor::Tensor(const Tensor& other)
        : data_(other.data_),
          data_owner_(other.data_owner_),
          state_(std::make_shared<TensorState>(*other.state_)),
          shape_(other.shape_),
          strides_(other.strides_),
          storage_offset_(other.storage_offset_),
          is_contiguous_(other.is_contiguous_),
          device_(other.device_),
          dtype_(other.dtype_),
          is_view_(other.is_view_),
          storage_meta_(other.storage_meta_),
          view_generation_snapshot_(other.view_generation_snapshot_),
          id_(next_id_++) {}

    Tensor& Tensor::operator=(const Tensor& other) {
        if (this == &other) {
            return *this;
        }
        data_ = other.data_;
        data_owner_ = other.data_owner_;
        state_ = std::make_shared<TensorState>(*other.state_);
        shape_ = other.shape_;
        strides_ = other.strides_;
        storage_offset_ = other.storage_offset_;
        is_contiguous_ = other.is_contiguous_;
        device_ = other.device_;
        dtype_ = other.dtype_;
        is_view_ = other.is_view_;
        storage_meta_ = other.storage_meta_;
        view_generation_snapshot_ = other.view_generation_snapshot_;
        id_ = next_id_++;
        lazy_ir_registered_ = false;
        return *this;
    }

    Tensor::Tensor(Tensor&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)),
          data_owner_(std::move(other.data_owner_)),
          state_(std::move(other.state_)),
          shape_(std::move(other.shape_)),
          strides_(std::move(other.strides_)),
          storage_offset_(std::exchange(other.storage_offset_, 0)),
          is_contiguous_(std::exchange(other.is_contiguous_, true)),
          device_(other.device_),
          dtype_(other.dtype_),
          is_view_(std::exchange(other.is_view_, false)),
          storage_meta_(std::move(other.storage_meta_)),
          view_generation_snapshot_(other.view_generation_snapshot_),
          id_(std::exchange(other.id_, 0)),
          lazy_ir_registered_(false) {
        if (!state_) {
            state_ = std::make_shared<TensorState>();
        }
        if (!other.state_) {
            other.state_ = std::make_shared<TensorState>();
        }
    }

    Tensor& Tensor::operator=(Tensor&& other) {
        if (this == &other) {
            return *this;
        }
        data_ = std::exchange(other.data_, nullptr);
        data_owner_ = std::move(other.data_owner_);
        state_ = std::move(other.state_);
        shape_ = std::move(other.shape_);
        strides_ = std::move(other.strides_);
        storage_offset_ = std::exchange(other.storage_offset_, 0);
        is_contiguous_ = std::exchange(other.is_contiguous_, true);
        device_ = other.device_;
        dtype_ = other.dtype_;
        is_view_ = std::exchange(other.is_view_, false);
        storage_meta_ = std::move(other.storage_meta_);
        view_generation_snapshot_ = other.view_generation_snapshot_;
        id_ = std::exchange(other.id_, 0);
        lazy_ir_registered_ = false;
        if (!state_) {
            state_ = std::make_shared<TensorState>();
        }
        if (!other.state_) {
            other.state_ = std::make_shared<TensorState>();
        }
        return *this;
    }

    Tensor::~Tensor() = default;

    Tensor Tensor::empty(
        TensorShape shape, const Device device, const DataType dtype,
        const bool) {
        if (device != Device::CPU) {
            unsupported_cuda();
        }
        Tensor result;
        result.shape_ = std::move(shape);
        result.strides_ = result.shape_.strides();
        result.device_ = device;
        result.dtype_ = dtype;
        result.is_view_ = false;
        result.id_ = next_id_++;
        result.state_->capacity =
            result.shape_.rank() == 0 ? 0 : result.shape_[0];
        result.state_->logical_size = result.state_->capacity;
        const std::size_t bytes =
            std::max<std::size_t>(result.shape_.elements() * dtype_size(dtype), 1);
        void* const storage = ::operator new(bytes, std::align_val_t{64});
        result.data_ = storage;
        result.adopt_storage(storage, [](void* pointer) {
            ::operator delete(pointer, std::align_val_t{64});
        });
        result.compute_alignment();
        return result;
    }

    Tensor Tensor::empty_unpinned(TensorShape shape, const DataType dtype) {
        return empty(std::move(shape), Device::CPU, dtype, false);
    }

    Tensor Tensor::zeros(
        TensorShape shape, const Device device, const DataType dtype) {
        auto result = empty(std::move(shape), device, dtype);
        std::memset(result.data_ptr(), 0, result.bytes());
        return result;
    }

    Tensor Tensor::ones(
        TensorShape shape, const Device device, const DataType dtype) {
        return full(std::move(shape), 1.0f, device, dtype);
    }

    Tensor Tensor::full(
        TensorShape shape, const float value, const Device device,
        const DataType dtype) {
        auto result = empty(std::move(shape), device, dtype);
        for (std::size_t index = 0; index < result.numel(); ++index) {
            set_tensor_value(result, index, value);
        }
        return result;
    }

    Tensor Tensor::full_bool(
        TensorShape shape, const bool value, const Device device) {
        return full(std::move(shape), value ? 1.0f : 0.0f,
                    device, DataType::Bool);
    }

    Tensor Tensor::zeros_bool(TensorShape shape, const Device device) {
        return full_bool(std::move(shape), false, device);
    }

    Tensor Tensor::ones_bool(TensorShape shape, const Device device) {
        return full_bool(std::move(shape), true, device);
    }

    Tensor Tensor::from_vector(
        const std::vector<float>& data, TensorShape shape,
        const Device device) {
        if (data.size() != shape.elements()) {
            throw std::runtime_error("tensor data does not match its shape");
        }
        auto result = empty(std::move(shape), device, DataType::Float32);
        std::memcpy(result.data_ptr(), data.data(), result.bytes());
        return result;
    }

    Tensor Tensor::from_vector(
        const std::vector<int>& data, TensorShape shape,
        const Device device) {
        if (data.size() != shape.elements()) {
            throw std::runtime_error("tensor data does not match its shape");
        }
        auto result = empty(std::move(shape), device, DataType::Int32);
        std::memcpy(result.data_ptr(), data.data(), result.bytes());
        return result;
    }

    Tensor Tensor::from_vector(
        const std::vector<bool>& data, TensorShape shape,
        const Device device) {
        if (data.size() != shape.elements()) {
            throw std::runtime_error("tensor data does not match its shape");
        }
        auto result = empty(std::move(shape), device, DataType::Bool);
        for (std::size_t index = 0; index < data.size(); ++index) {
            result.ptr<std::uint8_t>()[index] = data[index];
        }
        return result;
    }

    Tensor Tensor::clone() const {
        tensor_contract::require_valid(
            *this, "clone", "source", LFS_SOURCE_SITE_CURRENT());
        auto result = empty(shape_, device_, dtype_);
        std::memcpy(result.data_ptr(), data_ptr(), bytes());
        return result;
    }

    Tensor Tensor::contiguous() const {
        tensor_contract::require_valid(
            *this, "contiguous", "source", LFS_SOURCE_SITE_CURRENT());
        if (!is_contiguous_) {
            throw std::runtime_error(
                "the format test runtime does not create strided tensors");
        }
        return *this;
    }

    void Tensor::materialize_zero_stride_for_raw_ptr_escape() {
        if (numel() == 0 || is_contiguous_ || !has_zero_stride()) {
            return;
        }
        throw std::runtime_error(
            "the format test runtime does not create zero-stride tensors");
    }

    void Tensor::assert_device_storage_matches_tag() const {
        if (data_ != nullptr && numel() != 0 && device_ != Device::CPU) {
            throw std::runtime_error(
                "the CPU-only format test runtime encountered non-CPU storage");
        }
    }

    Tensor Tensor::to(const Device device, const cudaStream_t) const {
        tensor_contract::require_valid(
            *this, "device transfer", "source", LFS_SOURCE_SITE_CURRENT());
        if (device != Device::CPU) {
            unsupported_cuda();
        }
        return clone();
    }

    Tensor Tensor::to(const DataType dtype) const {
        tensor_contract::require_valid(
            *this, "dtype conversion", "source", LFS_SOURCE_SITE_CURRENT());
        if (dtype == dtype_) {
            return clone();
        }
        auto result = empty(shape_, Device::CPU, dtype);
        for (std::size_t index = 0; index < numel(); ++index) {
            set_tensor_value(result, index, tensor_value(*this, index));
        }
        return result;
    }

    void Tensor::preserve_lazy_snapshots_before_write() {}

    void Tensor::materialize_deferred_slow() {
        throw std::runtime_error(
            "the format test runtime does not support deferred tensors");
    }

    std::string TensorShape::str() const {
        std::ostringstream stream;
        stream << '[';
        for (std::size_t index = 0; index < dims_.size(); ++index) {
            if (index != 0) {
                stream << ", ";
            }
            stream << dims_[index];
        }
        stream << ']';
        return stream.str();
    }

} // namespace lfs::core

namespace lfs::test::licht {

    std::vector<std::byte> one_pixel_png() {
        static constexpr char PNG[] =
            "\x89\x50\x4e\x47\x0d\x0a\x1a\x0a\x00\x00\x00\x0d\x49\x48\x44\x52"
            "\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4"
            "\x89\x00\x00\x00\x0a\x49\x44\x41\x54\x78\x9c\x63\x60\x00\x00\x00"
            "\x02\x00\x01\xe5\x27\xd4\xa2\x00\x00\x00\x00\x49\x45\x4e\x44\xae"
            "\x42\x60\x82";
        const auto bytes = std::as_bytes(std::span(PNG, sizeof(PNG) - 1));
        return {bytes.begin(), bytes.end()};
    }

    std::shared_ptr<core::PointCloud> make_point_cloud(
        const std::size_t count) {
        std::vector<float> means(count * 3, 0.0f);
        std::vector<float> colors(count * 3, 0.5f);
        for (std::size_t index = 0; index < count; ++index) {
            means[index * 3] = static_cast<float>(index + 10);
        }
        return std::make_shared<core::PointCloud>(
            core::Tensor::from_vector(
                means, {count, std::size_t{3}}, core::Device::CPU),
            core::Tensor::from_vector(
                colors, {count, std::size_t{3}}, core::Device::CPU));
    }

} // namespace lfs::test::licht
