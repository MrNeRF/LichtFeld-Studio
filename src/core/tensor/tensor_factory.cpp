/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "internal/tensor_impl.hpp"

namespace lfs::core {

    Tensor Tensor::from_external_owner(void* data,
                                       TensorShape shape,
                                       const Device device,
                                       const DataType dtype,
                                       std::shared_ptr<void> owner) {
        return from_external_owner(
            data, std::move(shape), device, dtype, std::move(owner), 0, nullptr, {});
    }

    Tensor Tensor::from_external_owner(void* data,
                                       TensorShape shape,
                                       const Device device,
                                       const DataType dtype,
                                       std::shared_ptr<void> owner,
                                       const size_t capacity) {
        return from_external_owner(
            data, std::move(shape), device, dtype, std::move(owner), capacity, nullptr, {});
    }

    Tensor Tensor::from_external_owner(void* data,
                                       TensorShape shape,
                                       const Device device,
                                       const DataType dtype,
                                       std::shared_ptr<void> owner,
                                       const size_t capacity,
                                       const cudaStream_t stream) {
        return from_external_owner(
            data, std::move(shape), device, dtype, std::move(owner), capacity, stream, {});
    }

    Tensor Tensor::from_external_owner(void* data,
                                       TensorShape shape,
                                       Device device,
                                       DataType dtype,
                                       std::shared_ptr<void> owner,
                                       size_t capacity,
                                       cudaStream_t stream,
                                       std::string external_kind) {
        LFS_ASSERT_MSG(owner != nullptr,
                       "from_external_owner requires a valid owner");
        LFS_ASSERT_MSG(data != nullptr || shape.elements() == 0,
                       "from_external_owner received null data for a non-empty tensor");
        LFS_ASSERT_MSG(device == Device::CPU || device == Device::CUDA,
                       "from_external_owner received an invalid device");
        LFS_ASSERT_MSG(dtype_size(dtype) != 0,
                       "from_external_owner received an invalid dtype");

        const size_t effective_capacity = capacity == 0 && shape.rank() > 0 ? shape[0] : capacity;
        LFS_ASSERT_MSG(shape.rank() == 0 || effective_capacity >= shape[0],
                       "from_external_owner capacity is smaller than the logical row count");
        const size_t allocation_bytes = storage_allocation_bytes(shape, effective_capacity, dtype);
        auto external_owner = owner;
        auto owner_box = std::make_shared<std::shared_ptr<void>>(std::move(owner));
        record_storage_allocation(StorageAccountingKind::VulkanExternal, allocation_bytes);

        Tensor t;
        t.data_ = data;
        t.data_owner_ = std::shared_ptr<void>(
            data,
            [owner_box = std::move(owner_box), allocation_bytes](void*) mutable {
                owner_box->reset();
                Tensor::record_storage_deallocation(
                    StorageAccountingKind::VulkanExternal,
                    allocation_bytes);
            });
        t.shape_ = std::move(shape);
        t.strides_ = t.shape_.strides();
        t.storage_offset_ = 0;
        t.is_contiguous_ = true;
        t.device_ = device;
        t.dtype_ = dtype;
        t.is_view_ = false;
        t.state_->capacity = effective_capacity;
        t.state_->logical_size = t.shape_.rank() > 0 ? t.shape_[0] : 0;
        t.state_->stream = stream;
        t.id_ = next_id_++;
        t.compute_alignment();
        t.init_storage_meta();
        t.storage_meta_->external_kind = std::move(external_kind);
        t.storage_meta_->external_owner = std::move(external_owner);
        return t;
    }

    Tensor Tensor::empty(TensorShape shape, Device device, DataType dtype, bool use_pinned) {
        LoadArgs args;
        args.shape = shape;
        args.device = device;
        args.dtype = dtype;
        args.use_pinned = use_pinned;
        args.args = std::monostate{};
        return load(LoadOp::Empty, args);
    }

    Tensor Tensor::empty_unpinned(TensorShape shape, DataType dtype) {
        LoadArgs args;
        args.shape = shape;
        args.device = Device::CPU;
        args.dtype = dtype;
        args.use_pinned = false;
        args.args = std::monostate{};
        return load(LoadOp::Empty, args);
    }

    Tensor Tensor::zeros(TensorShape shape, Device device, DataType dtype) {
        LoadArgs args;
        args.shape = shape;
        args.device = device;
        args.dtype = dtype;
        args.args = 0.0f;
        return load(LoadOp::Const, args);
    }

    Tensor Tensor::ones(TensorShape shape, Device device, DataType dtype) {
        LoadArgs args;
        args.shape = shape;
        args.device = device;
        args.dtype = dtype;
        args.args = 1.0f;
        return load(LoadOp::Const, args);
    }

    Tensor Tensor::full(TensorShape shape, float value, Device device, DataType dtype) {
        LoadArgs args;
        args.shape = shape;
        args.device = device;
        args.dtype = dtype;
        args.args = value;
        return load(LoadOp::Const, args);
    }

    Tensor Tensor::full_bool(TensorShape shape, bool value, Device device) {
        LoadArgs args;
        args.shape = shape;
        args.device = device;
        args.dtype = DataType::Bool;
        args.args = value ? 1.0f : 0.0f;
        return load(LoadOp::Const, args);
    }

    Tensor Tensor::zeros_bool(TensorShape shape, Device device) {
        return full_bool(shape, false, device);
    }

    Tensor Tensor::ones_bool(TensorShape shape, Device device) {
        return full_bool(shape, true, device);
    }

    Tensor Tensor::rand(TensorShape shape, Device device, DataType dtype) {
        LoadArgs args;
        args.shape = shape;
        args.device = device;
        args.dtype = dtype;
        args.args = std::pair<float, float>{0.0f, 1.0f};
        return load(LoadOp::Random, args);
    }

    Tensor Tensor::randn(TensorShape shape, Device device, DataType dtype) {
        LoadArgs args;
        args.shape = shape;
        args.device = device;
        args.dtype = dtype;
        args.args = std::pair<float, float>{0.0f, 1.0f};
        return load(LoadOp::Normal, args);
    }

    Tensor Tensor::uniform(TensorShape shape, float low, float high, Device device, DataType dtype) {
        LoadArgs args;
        args.shape = shape;
        args.device = device;
        args.dtype = dtype;
        args.args = std::pair<float, float>{low, high};
        return load(LoadOp::Random, args);
    }

    Tensor Tensor::normal(TensorShape shape, float mean, float std, Device device, DataType dtype) {
        LoadArgs args;
        args.shape = shape;
        args.device = device;
        args.dtype = dtype;
        args.args = std::pair<float, float>{mean, std};
        return load(LoadOp::Normal, args);
    }

    Tensor Tensor::randint(TensorShape shape, int low, int high, Device device, DataType dtype) {
        LoadArgs args;
        args.shape = shape;
        args.device = device;
        args.dtype = dtype;
        args.args = std::pair<int, int>{low, high};
        return load(LoadOp::Randint, args);
    }

    Tensor Tensor::bernoulli(TensorShape shape, float p, Device device, DataType dtype) {
        LoadArgs args;
        args.shape = shape;
        args.device = device;
        args.dtype = dtype;
        args.args = p;
        return load(LoadOp::Bernoulli, args);
    }

    Tensor Tensor::arange(float end) {
        LoadArgs args;
        args.shape = TensorShape{};
        args.device = Device::CUDA;
        args.dtype = DataType::Float32;
        args.args = std::tuple<float, float, float>{0.0f, end, 1.0f};
        return load(LoadOp::Arange, args);
    }

    Tensor Tensor::arange(float start, float end, float step) {
        LoadArgs args;
        args.shape = TensorShape{};
        args.device = Device::CUDA;
        args.dtype = DataType::Float32;
        args.args = std::tuple<float, float, float>{start, end, step};
        return load(LoadOp::Arange, args);
    }

    Tensor Tensor::eye(size_t n, Device device) {
        LoadArgs args;
        args.shape = TensorShape{n, n};
        args.device = device;
        args.dtype = DataType::Float32;
        args.args = std::monostate{};
        return load(LoadOp::Eye, args);
    }

    Tensor Tensor::eye(size_t m, size_t n, Device device) {
        LoadArgs args;
        args.shape = TensorShape{m, n};
        args.device = device;
        args.dtype = DataType::Float32;
        args.args = std::monostate{};
        return load(LoadOp::Eye, args);
    }

} // namespace lfs::core
