/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "py_tensor.hpp"
#include "core/logger.hpp"
#include "python/python_runtime.hpp"

#include <cstring>
#include <cuda_runtime.h>
#include <dlpack/dlpack.h>
#include <nanobind/stl/optional.h>
#include <sstream>

namespace nb = nanobind;

namespace lfs::python {

    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::Tensor;
    using lfs::core::TensorShape;

    namespace {

        constexpr DLDeviceType to_dl_device(const Device d) {
            return d == Device::CUDA ? kDLCUDA : kDLCPU;
        }

        Device from_dl_device(const DLDeviceType t) {
            if (t == kDLCUDA || t == kDLCUDAManaged)
                return Device::CUDA;
            if (t == kDLCPU || t == kDLCUDAHost)
                return Device::CPU;
            throw std::runtime_error("Unsupported DLPack device type");
        }

        DLDataType to_dl_dtype(const DataType dt) {
            DLDataType r{};
            r.lanes = 1;
            switch (dt) {
            case DataType::Float32:
                r.code = kDLFloat;
                r.bits = 32;
                break;
            case DataType::Float16:
                r.code = kDLFloat;
                r.bits = 16;
                break;
            case DataType::Int32:
                r.code = kDLInt;
                r.bits = 32;
                break;
            case DataType::Int64:
                r.code = kDLInt;
                r.bits = 64;
                break;
            case DataType::UInt8:
                r.code = kDLUInt;
                r.bits = 8;
                break;
            case DataType::Bool:
                r.code = kDLUInt;
                r.bits = 8;
                break;
            default: throw std::runtime_error("Unsupported dtype for DLPack");
            }
            return r;
        }

        DataType from_dl_dtype(const DLDataType dt) {
            if (dt.lanes != 1)
                throw std::runtime_error("Vectorized DLPack not supported");
            if (dt.code == kDLFloat && dt.bits == 32)
                return DataType::Float32;
            if (dt.code == kDLFloat && dt.bits == 16)
                return DataType::Float16;
            if (dt.code == kDLInt && dt.bits == 32)
                return DataType::Int32;
            if (dt.code == kDLInt && dt.bits == 64)
                return DataType::Int64;
            if (dt.code == kDLUInt && dt.bits == 8)
                return DataType::UInt8;
            throw std::runtime_error("Unsupported DLPack dtype");
        }

        struct DLPackContext {
            Tensor tensor;
            std::vector<int64_t> shape;
            std::vector<int64_t> strides;

            explicit DLPackContext(Tensor t) : tensor(std::move(t)) {
                shape.reserve(tensor.ndim());
                strides.reserve(tensor.ndim());
                for (const auto d : tensor.shape().dims())
                    shape.push_back(static_cast<int64_t>(d));
                for (const auto s : tensor.strides())
                    strides.push_back(static_cast<int64_t>(s));
            }
        };

        void dlpack_deleter(DLManagedTensor* self) noexcept {
            if (self) {
                delete static_cast<DLPackContext*>(self->manager_ctx);
                delete self;
            }
        }

    } // namespace

    PyTensor::PyTensor(Tensor tensor, bool owns_data)
        : tensor_(std::move(tensor)),
          owns_data_(owns_data) {}

    PyTensor::~PyTensor() = default;

    PyTensor PyTensor::view_of(core::Tensor& t, uint64_t generation) {
        PyTensor pt(t, false);
        pt.source_gen_ = generation;
        return pt;
    }

    void PyTensor::validate() const {
        if (source_gen_ != 0 && source_gen_ != context().scene_generation)
            throw std::runtime_error("Tensor data invalidated - scene changed");
    }

    PyTensor::PyTensor(const PyTensor& other)
        : tensor_(other.tensor_),
          owns_data_(other.owns_data_),
          source_gen_(other.source_gen_),
          dlpack_managed_(other.dlpack_managed_) {}

    PyTensor& PyTensor::operator=(const PyTensor& other) {
        if (this != &other) {
            tensor_ = other.tensor_;
            owns_data_ = other.owns_data_;
            source_gen_ = other.source_gen_;
            dlpack_managed_ = other.dlpack_managed_;
        }
        return *this;
    }

    PyTensor::PyTensor(PyTensor&& other) noexcept
        : tensor_(std::move(other.tensor_)),
          owns_data_(other.owns_data_),
          source_gen_(other.source_gen_),
          dlpack_managed_(std::move(other.dlpack_managed_)) {}

    PyTensor& PyTensor::operator=(PyTensor&& other) noexcept {
        if (this != &other) {
            tensor_ = std::move(other.tensor_);
            owns_data_ = other.owns_data_;
            source_gen_ = other.source_gen_;
            dlpack_managed_ = std::move(other.dlpack_managed_);
        }
        return *this;
    }

    nb::tuple PyTensor::shape() const {
        const auto& dims = tensor_.shape().dims();
        nb::list shape_list;
        for (size_t d : dims) {
            shape_list.append(static_cast<int64_t>(d));
        }
        return nb::tuple(shape_list);
    }

    size_t PyTensor::ndim() const {
        return tensor_.shape().rank();
    }

    size_t PyTensor::numel() const {
        return tensor_.numel();
    }

    std::string PyTensor::device() const {
        return tensor_.device() == Device::CUDA ? "cuda" : "cpu";
    }

    std::string PyTensor::dtype() const {
        switch (tensor_.dtype()) {
        case DataType::Float32: return "float32";
        case DataType::Float16: return "float16";
        case DataType::Int32: return "int32";
        case DataType::Int64: return "int64";
        case DataType::UInt8: return "uint8";
        case DataType::Bool: return "bool";
        default: return "unknown";
        }
    }

    bool PyTensor::is_contiguous() const {
        return tensor_.is_contiguous();
    }

    bool PyTensor::is_cuda() const {
        return tensor_.device() == Device::CUDA;
    }

    size_t PyTensor::size(int dim) const {
        const int resolved = dim < 0 ? static_cast<int>(tensor_.shape().rank()) + dim : dim;
        if (resolved < 0 || resolved >= static_cast<int>(tensor_.shape().rank())) {
            throw std::out_of_range("Dimension out of range");
        }
        return tensor_.shape()[static_cast<size_t>(resolved)];
    }

    PyTensor PyTensor::clone() const {
        return PyTensor(tensor_.clone());
    }

    PyTensor PyTensor::cpu() const {
        validate();
        if (tensor_.device() == Device::CPU) {
            return PyTensor(tensor_);
        }
        return PyTensor(tensor_.cpu());
    }

    PyTensor PyTensor::cuda() const {
        if (tensor_.device() == Device::CUDA) {
            return PyTensor(tensor_);
        }
        return PyTensor(tensor_.cuda());
    }

    PyTensor PyTensor::contiguous() const {
        if (tensor_.is_contiguous()) {
            return PyTensor(tensor_);
        }
        return PyTensor(tensor_.contiguous());
    }

    void PyTensor::sync() const {
        if (tensor_.device() == Device::CUDA) {
            cudaDeviceSynchronize();
        }
    }

    float PyTensor::item() const {
        validate();
        if (tensor_.numel() != 1) {
            throw std::runtime_error("item() requires a tensor with exactly 1 element");
        }
        switch (tensor_.dtype()) {
        case DataType::Float32: return tensor_.item<float>();
        case DataType::Float16: return tensor_.item<float>(); // Tensor handles conversion
        case DataType::Int32: return static_cast<float>(tensor_.item<int>());
        case DataType::Int64: return static_cast<float>(tensor_.item<int64_t>());
        case DataType::Bool: return tensor_.item<unsigned char>() != 0 ? 1.0f : 0.0f;
        default: return tensor_.item<float>();
        }
    }

    int64_t PyTensor::item_int() const {
        validate();
        if (tensor_.numel() != 1) {
            throw std::runtime_error("item() requires a tensor with exactly 1 element");
        }
        if (tensor_.dtype() == DataType::Int32) {
            return static_cast<int64_t>(tensor_.item<int>());
        } else if (tensor_.dtype() == DataType::Int64) {
            return tensor_.item<int64_t>();
        }
        return static_cast<int64_t>(tensor_.item<float>());
    }

    bool PyTensor::item_bool() const {
        validate();
        if (tensor_.numel() != 1) {
            throw std::runtime_error("item() requires a tensor with exactly 1 element");
        }
        if (tensor_.dtype() == DataType::Bool) {
            return tensor_.item<unsigned char>() != 0;
        }
        return tensor_.item<float>() != 0.0f;
    }

    nb::object PyTensor::numpy(bool copy) const {
        validate();
        Tensor cpu_tensor = tensor_.device() == Device::CUDA ? tensor_.cpu() : tensor_;

        // Ensure contiguous
        if (!cpu_tensor.is_contiguous()) {
            cpu_tensor = cpu_tensor.contiguous();
        }

        const auto& dims = cpu_tensor.shape().dims();
        size_t elem_size = 4;
        switch (cpu_tensor.dtype()) {
        case DataType::Float32: elem_size = 4; break;
        case DataType::Float16: elem_size = 2; break;
        case DataType::Int32: elem_size = 4; break;
        case DataType::Int64: elem_size = 8; break;
        case DataType::UInt8:
        case DataType::Bool: elem_size = 1; break;
        }

        if (copy) {
            // Copy mode: allocate new buffer and copy data
            const size_t total_bytes = cpu_tensor.numel() * elem_size;
            void* const buffer = std::malloc(total_bytes);
            if (!buffer) {
                throw std::bad_alloc();
            }
            std::memcpy(buffer, cpu_tensor.data_ptr(), total_bytes);

            // Create owner capsule for memory management
            nb::capsule owner(buffer, [](void* p) noexcept { std::free(p); });

            // Use nb::shape to create proper shape object
            switch (cpu_tensor.dtype()) {
            case DataType::Float32: {
                if (dims.size() == 1) {
                    return nb::cast(nb::ndarray<nb::numpy, float, nb::shape<-1>>(
                        buffer, {dims[0]}, owner));
                } else if (dims.size() == 2) {
                    return nb::cast(nb::ndarray<nb::numpy, float, nb::shape<-1, -1>>(
                        buffer, {dims[0], dims[1]}, owner));
                } else if (dims.size() == 3) {
                    return nb::cast(nb::ndarray<nb::numpy, float, nb::shape<-1, -1, -1>>(
                        buffer, {dims[0], dims[1], dims[2]}, owner));
                } else {
                    // Fallback for higher dimensions - use dynamic ndarray
                    std::vector<size_t> shape_vec(dims.begin(), dims.end());
                    return nb::cast(nb::ndarray<nb::numpy, float>(
                        buffer, dims.size(), shape_vec.data(), owner));
                }
            }
            case DataType::Int32: {
                if (dims.size() == 1) {
                    return nb::cast(nb::ndarray<nb::numpy, int32_t, nb::shape<-1>>(
                        buffer, {dims[0]}, owner));
                } else if (dims.size() == 2) {
                    return nb::cast(nb::ndarray<nb::numpy, int32_t, nb::shape<-1, -1>>(
                        buffer, {dims[0], dims[1]}, owner));
                } else {
                    std::vector<size_t> shape_vec(dims.begin(), dims.end());
                    return nb::cast(nb::ndarray<nb::numpy, int32_t>(
                        buffer, dims.size(), shape_vec.data(), owner));
                }
            }
            case DataType::Int64: {
                if (dims.size() == 1) {
                    return nb::cast(nb::ndarray<nb::numpy, int64_t, nb::shape<-1>>(
                        buffer, {dims[0]}, owner));
                } else if (dims.size() == 2) {
                    return nb::cast(nb::ndarray<nb::numpy, int64_t, nb::shape<-1, -1>>(
                        buffer, {dims[0], dims[1]}, owner));
                } else {
                    std::vector<size_t> shape_vec(dims.begin(), dims.end());
                    return nb::cast(nb::ndarray<nb::numpy, int64_t>(
                        buffer, dims.size(), shape_vec.data(), owner));
                }
            }
            case DataType::UInt8:
            case DataType::Bool: {
                if (dims.size() == 1) {
                    return nb::cast(nb::ndarray<nb::numpy, uint8_t, nb::shape<-1>>(
                        buffer, {dims[0]}, owner));
                } else if (dims.size() == 2) {
                    return nb::cast(nb::ndarray<nb::numpy, uint8_t, nb::shape<-1, -1>>(
                        buffer, {dims[0], dims[1]}, owner));
                } else {
                    std::vector<size_t> shape_vec(dims.begin(), dims.end());
                    return nb::cast(nb::ndarray<nb::numpy, uint8_t>(
                        buffer, dims.size(), shape_vec.data(), owner));
                }
            }
            default:
                std::free(buffer);
                throw std::runtime_error("Unsupported dtype for numpy conversion");
            }
        } else {
            // Zero-copy mode: use capsule to hold tensor reference
            auto* const tensor_copy = new Tensor(cpu_tensor);
            const nb::capsule owner(tensor_copy, [](void* p) noexcept { delete static_cast<Tensor*>(p); });

            void* const data = tensor_copy->data_ptr();

            switch (cpu_tensor.dtype()) {
            case DataType::Float32: {
                if (dims.size() == 1) {
                    return nb::cast(nb::ndarray<nb::numpy, float, nb::shape<-1>>(
                        data, {dims[0]}, owner));
                } else if (dims.size() == 2) {
                    return nb::cast(nb::ndarray<nb::numpy, float, nb::shape<-1, -1>>(
                        data, {dims[0], dims[1]}, owner));
                } else if (dims.size() == 3) {
                    return nb::cast(nb::ndarray<nb::numpy, float, nb::shape<-1, -1, -1>>(
                        data, {dims[0], dims[1], dims[2]}, owner));
                } else {
                    std::vector<size_t> shape_vec(dims.begin(), dims.end());
                    return nb::cast(nb::ndarray<nb::numpy, float>(
                        data, dims.size(), shape_vec.data(), owner));
                }
            }
            case DataType::Int32: {
                if (dims.size() == 1) {
                    return nb::cast(nb::ndarray<nb::numpy, int32_t, nb::shape<-1>>(
                        data, {dims[0]}, owner));
                } else if (dims.size() == 2) {
                    return nb::cast(nb::ndarray<nb::numpy, int32_t, nb::shape<-1, -1>>(
                        data, {dims[0], dims[1]}, owner));
                } else {
                    std::vector<size_t> shape_vec(dims.begin(), dims.end());
                    return nb::cast(nb::ndarray<nb::numpy, int32_t>(
                        data, dims.size(), shape_vec.data(), owner));
                }
            }
            case DataType::Int64: {
                if (dims.size() == 1) {
                    return nb::cast(nb::ndarray<nb::numpy, int64_t, nb::shape<-1>>(
                        data, {dims[0]}, owner));
                } else if (dims.size() == 2) {
                    return nb::cast(nb::ndarray<nb::numpy, int64_t, nb::shape<-1, -1>>(
                        data, {dims[0], dims[1]}, owner));
                } else {
                    std::vector<size_t> shape_vec(dims.begin(), dims.end());
                    return nb::cast(nb::ndarray<nb::numpy, int64_t>(
                        data, dims.size(), shape_vec.data(), owner));
                }
            }
            case DataType::UInt8:
            case DataType::Bool: {
                if (dims.size() == 1) {
                    return nb::cast(nb::ndarray<nb::numpy, uint8_t, nb::shape<-1>>(
                        data, {dims[0]}, owner));
                } else if (dims.size() == 2) {
                    return nb::cast(nb::ndarray<nb::numpy, uint8_t, nb::shape<-1, -1>>(
                        data, {dims[0], dims[1]}, owner));
                } else {
                    std::vector<size_t> shape_vec(dims.begin(), dims.end());
                    return nb::cast(nb::ndarray<nb::numpy, uint8_t>(
                        data, dims.size(), shape_vec.data(), owner));
                }
            }
            default:
                throw std::runtime_error("Unsupported dtype for numpy conversion");
            }
        }
    }

    PyTensor PyTensor::from_numpy(nb::ndarray<> arr, bool copy) {
        // Get shape
        std::vector<size_t> shape_vec;
        for (size_t i = 0; i < arr.ndim(); ++i) {
            shape_vec.push_back(arr.shape(i));
        }
        const TensorShape shape(shape_vec);

        // Determine dtype
        DataType dtype = DataType::Float32;
        const auto nb_dtype = arr.dtype();
        if (nb_dtype == nb::dtype<float>()) {
            dtype = DataType::Float32;
        } else if (nb_dtype == nb::dtype<int32_t>()) {
            dtype = DataType::Int32;
        } else if (nb_dtype == nb::dtype<int64_t>()) {
            dtype = DataType::Int64;
        } else if (nb_dtype == nb::dtype<uint8_t>()) {
            dtype = DataType::UInt8;
        } else if (nb_dtype == nb::dtype<bool>()) {
            dtype = DataType::Bool;
        } else {
            throw std::runtime_error("Unsupported numpy dtype");
        }

        // Create CPU tensor and copy data
        Tensor tensor = Tensor::empty(shape, Device::CPU, dtype, false);

        size_t elem_size = 4;
        switch (dtype) {
        case DataType::Float32: elem_size = 4; break;
        case DataType::Int32: elem_size = 4; break;
        case DataType::Int64: elem_size = 8; break;
        case DataType::UInt8:
        case DataType::Bool: elem_size = 1; break;
        default: break;
        }

        std::memcpy(tensor.data_ptr(), arr.data(), shape.elements() * elem_size);

        return PyTensor(std::move(tensor));
    }

    // Slicing
    PyTensor::SliceInfo PyTensor::parse_slice(const nb::slice& sl, size_t dim_size) const {
        SliceInfo info;
        auto [start, stop, step, count] = sl.compute(dim_size);
        info.start = start;
        info.stop = stop;
        info.step = step;
        return info;
    }

    PyTensor PyTensor::getitem(const nb::object& key) const {
        // Single integer index
        if (nb::isinstance<nb::int_>(key)) {
            int64_t idx = nb::cast<int64_t>(key);
            if (idx < 0) {
                idx += static_cast<int64_t>(tensor_.shape()[0]);
            }
            if (idx < 0 || idx >= static_cast<int64_t>(tensor_.shape()[0])) {
                throw std::out_of_range("Index out of range");
            }
            return PyTensor(tensor_.slice(0, static_cast<size_t>(idx), static_cast<size_t>(idx + 1)).squeeze(0));
        }

        // Single slice
        if (nb::isinstance<nb::slice>(key)) {
            auto sl = nb::cast<nb::slice>(key);
            SliceInfo info = parse_slice(sl, tensor_.shape()[0]);

            if (info.step != 1) {
                throw std::runtime_error("Step != 1 not yet supported");
            }

            return PyTensor(tensor_.slice(0, static_cast<size_t>(info.start), static_cast<size_t>(info.stop)));
        }

        // Tuple of indices/slices
        if (nb::isinstance<nb::tuple>(key)) {
            auto tup = nb::cast<nb::tuple>(key);
            Tensor result = tensor_;

            // Track dimension offset due to squeezed dimensions
            int dim_offset = 0;

            for (size_t i = 0; i < tup.size(); ++i) {
                int current_dim = static_cast<int>(i) - dim_offset;
                nb::object item = tup[i];

                if (nb::isinstance<nb::int_>(item)) {
                    int64_t idx = nb::cast<int64_t>(item);
                    if (idx < 0) {
                        idx += static_cast<int64_t>(result.shape()[current_dim]);
                    }
                    result = result.slice(current_dim, static_cast<size_t>(idx), static_cast<size_t>(idx + 1)).squeeze(current_dim);
                    dim_offset++;
                } else if (nb::isinstance<nb::slice>(item)) {
                    auto sl = nb::cast<nb::slice>(item);
                    SliceInfo info = parse_slice(sl, result.shape()[current_dim]);

                    if (info.step != 1) {
                        throw std::runtime_error("Step != 1 not yet supported");
                    }

                    result = result.slice(current_dim, static_cast<size_t>(info.start), static_cast<size_t>(info.stop));
                }
            }

            return PyTensor(result);
        }

        // Boolean mask
        if (nb::isinstance<PyTensor>(key)) {
            auto mask_tensor = nb::cast<PyTensor>(key);
            auto dt = mask_tensor.tensor().dtype();
            if (dt != DataType::Bool && dt != DataType::UInt8) {
                throw std::runtime_error("Mask must be a boolean tensor");
            }
            return PyTensor(tensor_.masked_select(mask_tensor.tensor()));
        }

        throw std::runtime_error("Unsupported index type");
    }

    void PyTensor::setitem(const nb::object& key, const nb::object& value) {
        bool is_scalar_value = false;
        float scalar_value = 0.0f;
        Tensor val_tensor;

        if (nb::isinstance<PyTensor>(value)) {
            val_tensor = nb::cast<PyTensor>(value).tensor();
        } else if (nb::isinstance<nb::float_>(value) || nb::isinstance<nb::int_>(value)) {
            is_scalar_value = true;
            scalar_value = nb::cast<float>(value);
            val_tensor = Tensor::full({1}, scalar_value, tensor_.device(), tensor_.dtype());
        } else {
            throw std::runtime_error("Unsupported value type for setitem");
        }

        auto assign_to_target = [&](Tensor& target) {
            if (is_scalar_value) {
                target.fill_(scalar_value);
            } else {
                target.copy_from(val_tensor);
            }
        };

        // Single integer index
        if (nb::isinstance<nb::int_>(key)) {
            int64_t idx = nb::cast<int64_t>(key);
            if (idx < 0) {
                idx += static_cast<int64_t>(tensor_.shape()[0]);
            }
            if (idx < 0 || idx >= static_cast<int64_t>(tensor_.shape()[0])) {
                throw std::out_of_range("Index out of range");
            }
            Tensor target = tensor_.slice(0, static_cast<size_t>(idx), static_cast<size_t>(idx + 1));
            assign_to_target(target);
            return;
        }

        // Single slice
        if (nb::isinstance<nb::slice>(key)) {
            auto sl = nb::cast<nb::slice>(key);
            SliceInfo info = parse_slice(sl, tensor_.shape()[0]);

            if (info.step != 1) {
                throw std::runtime_error("Step != 1 not yet supported");
            }

            Tensor target = tensor_.slice(0, static_cast<size_t>(info.start), static_cast<size_t>(info.stop));
            assign_to_target(target);
            return;
        }

        // Tuple of indices/slices
        if (nb::isinstance<nb::tuple>(key)) {
            auto tup = nb::cast<nb::tuple>(key);
            Tensor target = tensor_;

            for (size_t i = 0; i < tup.size(); ++i) {
                int current_dim = static_cast<int>(i);
                nb::object item = tup[i];

                if (nb::isinstance<nb::int_>(item)) {
                    int64_t idx = nb::cast<int64_t>(item);
                    if (idx < 0) {
                        idx += static_cast<int64_t>(target.shape()[current_dim]);
                    }
                    target = target.slice(current_dim, static_cast<size_t>(idx), static_cast<size_t>(idx + 1));
                } else if (nb::isinstance<nb::slice>(item)) {
                    auto sl = nb::cast<nb::slice>(item);
                    SliceInfo info = parse_slice(sl, target.shape()[current_dim]);

                    if (info.step != 1) {
                        throw std::runtime_error("Step != 1 not yet supported");
                    }

                    target = target.slice(current_dim, static_cast<size_t>(info.start), static_cast<size_t>(info.stop));
                }
            }

            assign_to_target(target);
            return;
        }

        // Boolean mask indexing: tensor[bool_mask] = value
        if (nb::isinstance<PyTensor>(key)) {
            auto& mask_py = nb::cast<PyTensor&>(key);
            const auto& mask_t = mask_py.tensor();
            if (mask_t.dtype() == DataType::UInt8 || mask_t.dtype() == DataType::Bool) {
                if (is_scalar_value) {
                    tensor_.masked_fill_(mask_t, scalar_value);
                } else {
                    const auto& ct = static_cast<const Tensor&>(tensor_);
                    auto proxy = ct[mask_t];
                    proxy = val_tensor;
                }
                return;
            }
        }

        throw std::runtime_error("Unsupported index type for setitem");
    }

    // Arithmetic operators
    PyTensor PyTensor::add(const PyTensor& other) const {
        return PyTensor(tensor_.add(other.tensor_));
    }

    PyTensor PyTensor::add_scalar(float scalar) const {
        return PyTensor(tensor_.add(scalar));
    }

    PyTensor PyTensor::sub(const PyTensor& other) const {
        return PyTensor(tensor_.sub(other.tensor_));
    }

    PyTensor PyTensor::sub_scalar(float scalar) const {
        return PyTensor(tensor_.sub(scalar));
    }

    PyTensor PyTensor::rsub_scalar(float scalar) const {
        return PyTensor(Tensor::full(tensor_.shape(), scalar, tensor_.device(), tensor_.dtype()).sub(tensor_));
    }

    PyTensor PyTensor::mul(const PyTensor& other) const {
        return PyTensor(tensor_.mul(other.tensor_));
    }

    PyTensor PyTensor::mul_scalar(float scalar) const {
        return PyTensor(tensor_.mul(scalar));
    }

    PyTensor PyTensor::div(const PyTensor& other) const {
        return PyTensor(tensor_.div(other.tensor_));
    }

    PyTensor PyTensor::div_scalar(float scalar) const {
        return PyTensor(tensor_.div(scalar));
    }

    PyTensor PyTensor::rdiv_scalar(float scalar) const {
        return PyTensor(Tensor::full(tensor_.shape(), scalar, tensor_.device(), tensor_.dtype()).div(tensor_));
    }

    PyTensor PyTensor::neg() const {
        return PyTensor(tensor_.neg());
    }

    PyTensor PyTensor::abs() const {
        return PyTensor(tensor_.abs());
    }

    PyTensor PyTensor::sigmoid() const {
        return PyTensor(tensor_.sigmoid());
    }

    PyTensor PyTensor::exp() const {
        return PyTensor(tensor_.exp());
    }

    PyTensor PyTensor::log() const {
        return PyTensor(tensor_.log());
    }

    PyTensor PyTensor::sqrt() const {
        return PyTensor(tensor_.sqrt());
    }

    PyTensor PyTensor::relu() const {
        return PyTensor(tensor_.relu());
    }

    PyTensor PyTensor::sin() const {
        return PyTensor(tensor_.sin());
    }

    PyTensor PyTensor::cos() const {
        return PyTensor(tensor_.cos());
    }

    PyTensor PyTensor::tan() const {
        return PyTensor(tensor_.tan());
    }

    PyTensor PyTensor::tanh() const {
        return PyTensor(tensor_.tanh());
    }

    PyTensor PyTensor::floor() const {
        return PyTensor(tensor_.floor());
    }

    PyTensor PyTensor::ceil() const {
        return PyTensor(tensor_.ceil());
    }

    PyTensor PyTensor::round() const {
        return PyTensor(tensor_.round());
    }

    // Extended unary operations
    PyTensor PyTensor::log2() const {
        return PyTensor(tensor_.log2());
    }

    PyTensor PyTensor::log10() const {
        return PyTensor(tensor_.log10());
    }

    PyTensor PyTensor::log1p() const {
        return PyTensor(tensor_.log1p());
    }

    PyTensor PyTensor::exp2() const {
        return PyTensor(tensor_.exp2());
    }

    PyTensor PyTensor::rsqrt() const {
        return PyTensor(tensor_.rsqrt());
    }

    PyTensor PyTensor::square() const {
        return PyTensor(tensor_.square());
    }

    PyTensor PyTensor::asin() const {
        return PyTensor(tensor_.asin());
    }

    PyTensor PyTensor::acos() const {
        return PyTensor(tensor_.acos());
    }

    PyTensor PyTensor::atan() const {
        return PyTensor(tensor_.atan());
    }

    PyTensor PyTensor::sinh() const {
        return PyTensor(tensor_.sinh());
    }

    PyTensor PyTensor::cosh() const {
        return PyTensor(tensor_.cosh());
    }

    PyTensor PyTensor::trunc() const {
        return PyTensor(tensor_.trunc());
    }

    PyTensor PyTensor::sign() const {
        return PyTensor(tensor_.sign());
    }

    PyTensor PyTensor::reciprocal() const {
        return PyTensor(tensor_.reciprocal());
    }

    PyTensor PyTensor::gelu() const {
        return PyTensor(tensor_.gelu());
    }

    PyTensor PyTensor::swish() const {
        return PyTensor(tensor_.swish());
    }

    PyTensor PyTensor::isnan() const {
        return PyTensor(tensor_.isnan());
    }

    PyTensor PyTensor::isinf() const {
        return PyTensor(tensor_.isinf());
    }

    PyTensor PyTensor::isfinite() const {
        return PyTensor(tensor_.isfinite());
    }

    // Power operations
    PyTensor PyTensor::pow(float exponent) const {
        return PyTensor(tensor_.pow(exponent));
    }

    PyTensor PyTensor::pow(const PyTensor& exponent) const {
        return PyTensor(tensor_.pow(exponent.tensor_));
    }

    // In-place arithmetic
    PyTensor& PyTensor::iadd(const PyTensor& other) {
        tensor_.add_(other.tensor_);
        return *this;
    }

    PyTensor& PyTensor::iadd_scalar(float scalar) {
        tensor_.add_(scalar);
        return *this;
    }

    PyTensor& PyTensor::isub(const PyTensor& other) {
        tensor_.sub_(other.tensor_);
        return *this;
    }

    PyTensor& PyTensor::isub_scalar(float scalar) {
        tensor_.sub_(scalar);
        return *this;
    }

    PyTensor& PyTensor::imul(const PyTensor& other) {
        tensor_.mul_(other.tensor_);
        return *this;
    }

    PyTensor& PyTensor::imul_scalar(float scalar) {
        tensor_.mul_(scalar);
        return *this;
    }

    PyTensor& PyTensor::idiv(const PyTensor& other) {
        tensor_.div_(other.tensor_);
        return *this;
    }

    PyTensor& PyTensor::idiv_scalar(float scalar) {
        tensor_.div_(scalar);
        return *this;
    }

    PyTensor& PyTensor::fill_(float value) {
        tensor_.fill_(value);
        return *this;
    }

    PyTensor& PyTensor::zero_() {
        tensor_.fill_(0.0f);
        return *this;
    }

    // Comparison operators
    PyTensor PyTensor::eq(const PyTensor& other) const {
        return PyTensor(tensor_.eq(other.tensor_));
    }

    PyTensor PyTensor::eq_scalar(float scalar) const {
        return PyTensor(tensor_.eq(scalar));
    }

    PyTensor PyTensor::ne(const PyTensor& other) const {
        return PyTensor(tensor_.ne(other.tensor_));
    }

    PyTensor PyTensor::ne_scalar(float scalar) const {
        return PyTensor(tensor_.ne(scalar));
    }

    PyTensor PyTensor::lt(const PyTensor& other) const {
        return PyTensor(tensor_.lt(other.tensor_));
    }

    PyTensor PyTensor::lt_scalar(float scalar) const {
        return PyTensor(tensor_.lt(scalar));
    }

    PyTensor PyTensor::le(const PyTensor& other) const {
        return PyTensor(tensor_.le(other.tensor_));
    }

    PyTensor PyTensor::le_scalar(float scalar) const {
        return PyTensor(tensor_.le(scalar));
    }

    PyTensor PyTensor::gt(const PyTensor& other) const {
        return PyTensor(tensor_.gt(other.tensor_));
    }

    PyTensor PyTensor::gt_scalar(float scalar) const {
        return PyTensor(tensor_.gt(scalar));
    }

    PyTensor PyTensor::ge(const PyTensor& other) const {
        return PyTensor(tensor_.ge(other.tensor_));
    }

    PyTensor PyTensor::ge_scalar(float scalar) const {
        return PyTensor(tensor_.ge(scalar));
    }

    // Logical operators
    PyTensor PyTensor::logical_and(const PyTensor& other) const {
        return PyTensor(tensor_.logical_and(other.tensor_));
    }

    PyTensor PyTensor::logical_or(const PyTensor& other) const {
        return PyTensor(tensor_.logical_or(other.tensor_));
    }

    PyTensor PyTensor::logical_not() const {
        return PyTensor(tensor_.logical_not());
    }

    // Reduction operations
    PyTensor PyTensor::sum(std::optional<int> dim, bool keepdim) const {
        if (dim.has_value()) {
            return PyTensor(tensor_.sum(*dim, keepdim));
        }
        return PyTensor(tensor_.sum());
    }

    PyTensor PyTensor::mean(std::optional<int> dim, bool keepdim) const {
        if (dim.has_value()) {
            return PyTensor(tensor_.mean(*dim, keepdim));
        }
        return PyTensor(tensor_.mean());
    }

    PyTensor PyTensor::max(std::optional<int> dim, bool keepdim) const {
        if (dim.has_value()) {
            return PyTensor(tensor_.max(*dim, keepdim));
        }
        return PyTensor(tensor_.max());
    }

    PyTensor PyTensor::min(std::optional<int> dim, bool keepdim) const {
        if (dim.has_value()) {
            return PyTensor(tensor_.min(*dim, keepdim));
        }
        return PyTensor(tensor_.min());
    }

    float PyTensor::sum_scalar() const {
        return tensor_.sum_scalar();
    }

    float PyTensor::mean_scalar() const {
        return tensor_.mean_scalar();
    }

    float PyTensor::max_scalar() const {
        return tensor_.max_scalar();
    }

    float PyTensor::min_scalar() const {
        return tensor_.min_scalar();
    }

    // Extended reductions
    PyTensor PyTensor::prod(std::optional<int> dim, bool keepdim) const {
        if (dim.has_value()) {
            return PyTensor(tensor_.prod(*dim, keepdim));
        }
        return PyTensor(tensor_.prod());
    }

    PyTensor PyTensor::std(std::optional<int> dim, bool keepdim) const {
        if (dim.has_value()) {
            return PyTensor(tensor_.std(*dim, keepdim));
        }
        return PyTensor(tensor_.std());
    }

    PyTensor PyTensor::var(std::optional<int> dim, bool keepdim) const {
        if (dim.has_value()) {
            return PyTensor(tensor_.var(*dim, keepdim));
        }
        return PyTensor(tensor_.var());
    }

    PyTensor PyTensor::argmax(std::optional<int> dim, bool keepdim) const {
        if (dim.has_value()) {
            std::vector<int> axes = {*dim};
            return PyTensor(tensor_.argmax(axes, keepdim));
        }
        return PyTensor(tensor_.argmax());
    }

    PyTensor PyTensor::argmin(std::optional<int> dim, bool keepdim) const {
        if (dim.has_value()) {
            std::vector<int> axes = {*dim};
            return PyTensor(tensor_.argmin(axes, keepdim));
        }
        return PyTensor(tensor_.argmin());
    }

    PyTensor PyTensor::all(std::optional<int> dim, bool keepdim) const {
        if (dim.has_value()) {
            return PyTensor(tensor_.all(*dim, keepdim));
        }
        return PyTensor(tensor_.all());
    }

    PyTensor PyTensor::any(std::optional<int> dim, bool keepdim) const {
        if (dim.has_value()) {
            return PyTensor(tensor_.any(*dim, keepdim));
        }
        return PyTensor(tensor_.any());
    }

    PyTensor PyTensor::norm(float p) const {
        return PyTensor(Tensor::full({1}, tensor_.norm(p), tensor_.device(), tensor_.dtype()));
    }

    float PyTensor::norm_scalar(float p) const {
        return tensor_.norm(p);
    }

    // Shape operations
    PyTensor PyTensor::reshape(const std::vector<int64_t>& new_shape) const {
        std::vector<int> shape_vec;
        shape_vec.reserve(new_shape.size());

        // Handle -1 for inferred dimension
        int64_t infer_idx = -1;
        int known_size = 1;
        for (size_t i = 0; i < new_shape.size(); ++i) {
            if (new_shape[i] == -1) {
                if (infer_idx != -1) {
                    throw std::runtime_error("Only one dimension can be inferred");
                }
                infer_idx = static_cast<int64_t>(i);
                shape_vec.push_back(0); // Placeholder
            } else {
                shape_vec.push_back(static_cast<int>(new_shape[i]));
                known_size *= static_cast<int>(new_shape[i]);
            }
        }

        if (infer_idx != -1) {
            shape_vec[static_cast<size_t>(infer_idx)] = static_cast<int>(tensor_.numel()) / known_size;
        }

        return PyTensor(tensor_.reshape(shape_vec));
    }

    PyTensor PyTensor::view(const std::vector<int64_t>& new_shape) const {
        return reshape(new_shape);
    }

    PyTensor PyTensor::squeeze(std::optional<int> dim) const {
        if (dim.has_value()) {
            return PyTensor(tensor_.squeeze(*dim));
        }
        return PyTensor(tensor_.squeeze());
    }

    PyTensor PyTensor::unsqueeze(int dim) const {
        return PyTensor(tensor_.unsqueeze(dim));
    }

    PyTensor PyTensor::transpose(int dim0, int dim1) const {
        return PyTensor(tensor_.transpose(dim0, dim1));
    }

    PyTensor PyTensor::permute(const std::vector<int>& dims) const {
        return PyTensor(tensor_.permute(dims));
    }

    PyTensor PyTensor::flatten(int start_dim, int end_dim) const {
        return PyTensor(tensor_.flatten(start_dim, end_dim));
    }

    PyTensor PyTensor::expand(const std::vector<int64_t>& sizes) const {
        std::vector<int> int_sizes;
        int_sizes.reserve(sizes.size());
        for (auto s : sizes) {
            int_sizes.push_back(static_cast<int>(s));
        }
        return PyTensor(tensor_.expand(int_sizes));
    }

    PyTensor PyTensor::repeat(const std::vector<int64_t>& repeats) const {
        // Repeat by tiling - expand and then reshape
        std::vector<size_t> result_shape;
        const auto& orig_shape = tensor_.shape().dims();

        // Pad original shape if needed
        size_t ndim = std::max(orig_shape.size(), repeats.size());
        std::vector<size_t> padded_orig(ndim, 1);
        for (size_t i = 0; i < orig_shape.size(); ++i) {
            padded_orig[ndim - orig_shape.size() + i] = orig_shape[i];
        }

        // Calculate result shape
        for (size_t i = 0; i < ndim; ++i) {
            size_t rep = (i < repeats.size()) ? static_cast<size_t>(repeats[i]) : 1;
            result_shape.push_back(padded_orig[i] * rep);
        }

        // Tile using expand and reshape pattern
        Tensor result = tensor_;
        for (size_t i = 0; i < repeats.size(); ++i) {
            if (repeats[i] > 1) {
                result = Tensor::cat({result, result}, static_cast<int>(i));
                // Continue tiling for larger repeats
                for (int64_t j = 2; j < repeats[i]; j *= 2) {
                    result = Tensor::cat({result, result}, static_cast<int>(i));
                }
            }
        }
        return PyTensor(result);
    }

    PyTensor PyTensor::t() const {
        return PyTensor(tensor_.t());
    }

    // Advanced indexing operations
    PyTensor PyTensor::index_select(int dim, const PyTensor& indices) const {
        return PyTensor(tensor_.index_select(dim, indices.tensor_));
    }

    PyTensor PyTensor::gather(int dim, const PyTensor& indices) const {
        return PyTensor(tensor_.gather(dim, indices.tensor_));
    }

    PyTensor PyTensor::masked_select(const PyTensor& mask) const {
        return PyTensor(tensor_.masked_select(mask.tensor_));
    }

    PyTensor PyTensor::masked_fill(const PyTensor& mask, float value) const {
        return PyTensor(tensor_.masked_fill(mask.tensor_, value));
    }

    PyTensor& PyTensor::masked_fill_(const PyTensor& mask, float value) {
        tensor_.masked_fill_(mask.tensor_, value);
        return *this;
    }

    PyTensor PyTensor::nonzero() const {
        return PyTensor(tensor_.nonzero());
    }

    // Linear algebra
    PyTensor PyTensor::matmul(const PyTensor& other) const {
        return PyTensor(tensor_.matmul(other.tensor_));
    }

    PyTensor PyTensor::mm(const PyTensor& other) const {
        return PyTensor(tensor_.mm(other.tensor_));
    }

    PyTensor PyTensor::bmm(const PyTensor& other) const {
        return PyTensor(tensor_.bmm(other.tensor_));
    }

    PyTensor PyTensor::dot(const PyTensor& other) const {
        return PyTensor(tensor_.dot(other.tensor_));
    }

    // Element-wise operations
    PyTensor PyTensor::clamp(float min_val, float max_val) const {
        return PyTensor(tensor_.clamp(min_val, max_val));
    }

    PyTensor PyTensor::maximum(const PyTensor& other) const {
        return PyTensor(tensor_.maximum(other.tensor_));
    }

    PyTensor PyTensor::minimum(const PyTensor& other) const {
        return PyTensor(tensor_.minimum(other.tensor_));
    }

    std::string PyTensor::repr() const {
        std::ostringstream oss;
        oss << "Tensor(shape=" << tensor_.shape().str()
            << ", dtype=" << dtype()
            << ", device=" << device() << ")";
        return oss.str();
    }

    nb::tuple PyTensor::dlpack_device() const {
        const int32_t device_type = tensor_.device() == Device::CUDA ? kDLCUDA : kDLCPU;
        return nb::make_tuple(device_type, 0);
    }

    nb::capsule PyTensor::dlpack(nb::object stream) const {
        if (tensor_.device() == Device::CUDA && stream.is_none()) {
            cudaDeviceSynchronize();
        }

        auto* ctx = new DLPackContext(tensor_);
        auto* managed = new DLManagedTensor{};

        DLTensor& dl = managed->dl_tensor;
        dl.data = const_cast<void*>(tensor_.data_ptr());
        dl.device.device_type = to_dl_device(tensor_.device());
        dl.device.device_id = 0;
        dl.ndim = static_cast<int32_t>(tensor_.ndim());
        dl.dtype = to_dl_dtype(tensor_.dtype());
        dl.shape = ctx->shape.data();
        dl.strides = tensor_.is_contiguous() ? nullptr : ctx->strides.data();
        dl.byte_offset = 0;

        managed->manager_ctx = ctx;
        managed->deleter = dlpack_deleter;

        // Per DLPack spec: the consumer will call managed->deleter themselves,
        // so we set the capsule destructor to nullptr to avoid double-free.
        // If the capsule is never consumed, this causes a leak - but that's
        // rare in practice and better than crashing.
        return nb::capsule(managed, "dltensor");
    }

    PyTensor PyTensor::from_dlpack(nb::object obj) {
        nb::capsule capsule;

        if (nb::hasattr(obj, "__dlpack__")) {
            capsule = nb::cast<nb::capsule>(obj.attr("__dlpack__")());
        } else if (nb::isinstance<nb::capsule>(obj)) {
            capsule = nb::cast<nb::capsule>(obj);
        } else {
            throw std::runtime_error("from_dlpack: requires __dlpack__ method or capsule");
        }

        const char* const name = capsule.name();
        if (!name || std::strcmp(name, "dltensor") != 0) {
            if (name && std::strcmp(name, "used_dltensor") == 0) {
                throw std::runtime_error("from_dlpack: capsule already consumed");
            }
            throw std::runtime_error("from_dlpack: invalid capsule");
        }

        auto* managed = static_cast<DLManagedTensor*>(capsule.data());
        if (!managed) {
            throw std::runtime_error("from_dlpack: null DLManagedTensor");
        }

        // "Consume" the capsule per DLPack spec:
        // 1. Rename to "used_dltensor" so producer knows we took ownership
        // 2. Disable the capsule's destructor - we'll call the deleter ourselves
        PyObject* py_capsule = capsule.ptr();
        PyCapsule_SetName(py_capsule, "used_dltensor");
        PyCapsule_SetDestructor(py_capsule, nullptr);

        const DLTensor& dl = managed->dl_tensor;

        std::vector<size_t> shape_vec;
        shape_vec.reserve(dl.ndim);
        for (int32_t i = 0; i < dl.ndim; ++i) {
            shape_vec.push_back(static_cast<size_t>(dl.shape[i]));
        }

        void* const data = static_cast<char*>(dl.data) + dl.byte_offset;
        const Device device = from_dl_device(dl.device.device_type);
        const DataType dtype = from_dl_dtype(dl.dtype);

        Tensor tensor(data, TensorShape(shape_vec), device, dtype);

        // Store the DLManagedTensor with a custom deleter that calls the DLPack deleter
        auto result = PyTensor(std::move(tensor), false);
        result.dlpack_managed_ = std::shared_ptr<DLManagedTensor>(
            managed,
            [](DLManagedTensor* m) {
                if (m && m->deleter) {
                    m->deleter(m);
                }
            });
        return result;
    }

    namespace {
        Device parse_device(const std::string& device) {
            if (device == "cuda" || device == "gpu")
                return Device::CUDA;
            if (device == "cpu")
                return Device::CPU;
            throw std::runtime_error("Unknown device: " + device);
        }

        DataType parse_dtype(const std::string& dtype) {
            if (dtype == "float32" || dtype == "float")
                return DataType::Float32;
            if (dtype == "float16" || dtype == "half")
                return DataType::Float16;
            if (dtype == "int32" || dtype == "int")
                return DataType::Int32;
            if (dtype == "int64" || dtype == "long")
                return DataType::Int64;
            if (dtype == "uint8" || dtype == "byte")
                return DataType::UInt8;
            if (dtype == "bool")
                return DataType::Bool;
            throw std::runtime_error("Unknown dtype: " + dtype);
        }

        TensorShape to_tensor_shape(const std::vector<int64_t>& shape) {
            std::vector<size_t> dims;
            dims.reserve(shape.size());
            for (auto d : shape) {
                dims.push_back(static_cast<size_t>(d));
            }
            return TensorShape(dims);
        }
    } // namespace

    // Type conversion (needs parse_dtype from anonymous namespace)
    PyTensor PyTensor::to_dtype(const std::string& dtype) const {
        return PyTensor(tensor_.to(parse_dtype(dtype)));
    }

    PyTensor PyTensor::zeros(const std::vector<int64_t>& shape,
                             const std::string& device,
                             const std::string& dtype) {
        return PyTensor(Tensor::zeros(to_tensor_shape(shape), parse_device(device), parse_dtype(dtype)));
    }

    PyTensor PyTensor::ones(const std::vector<int64_t>& shape,
                            const std::string& device,
                            const std::string& dtype) {
        return PyTensor(Tensor::ones(to_tensor_shape(shape), parse_device(device), parse_dtype(dtype)));
    }

    PyTensor PyTensor::full(const std::vector<int64_t>& shape, float value,
                            const std::string& device,
                            const std::string& dtype) {
        return PyTensor(Tensor::full(to_tensor_shape(shape), value, parse_device(device), parse_dtype(dtype)));
    }

    PyTensor PyTensor::arange(float end) {
        return PyTensor(Tensor::arange(end));
    }

    PyTensor PyTensor::arange(float start, float end, float step,
                              const std::string& device,
                              const std::string& dtype) {
        auto t = Tensor::arange(start, end, step);
        if (device != "cuda") {
            t = t.to(parse_device(device));
        }
        if (dtype != "float32") {
            t = t.to(parse_dtype(dtype));
        }
        return PyTensor(std::move(t));
    }

    PyTensor PyTensor::linspace(float start, float end, int64_t steps,
                                const std::string& device,
                                const std::string& dtype) {
        auto t = Tensor::linspace(start, end, static_cast<size_t>(steps), parse_device(device));
        if (dtype != "float32") {
            t = t.to(parse_dtype(dtype));
        }
        return PyTensor(std::move(t));
    }

    PyTensor PyTensor::eye(int64_t n, const std::string& device,
                           const std::string& dtype) {
        auto t = Tensor::eye(static_cast<size_t>(n), parse_device(device));
        if (dtype != "float32") {
            t = t.to(parse_dtype(dtype));
        }
        return PyTensor(std::move(t));
    }

    PyTensor PyTensor::eye(int64_t m, int64_t n, const std::string& device,
                           const std::string& dtype) {
        auto t = Tensor::eye(static_cast<size_t>(m), static_cast<size_t>(n), parse_device(device));
        if (dtype != "float32") {
            t = t.to(parse_dtype(dtype));
        }
        return PyTensor(std::move(t));
    }

    // Random tensor creation
    PyTensor PyTensor::rand(const std::vector<int64_t>& shape,
                            const std::string& device,
                            const std::string& dtype) {
        return PyTensor(Tensor::rand(to_tensor_shape(shape), parse_device(device), parse_dtype(dtype)));
    }

    PyTensor PyTensor::randn(const std::vector<int64_t>& shape,
                             const std::string& device,
                             const std::string& dtype) {
        return PyTensor(Tensor::randn(to_tensor_shape(shape), parse_device(device), parse_dtype(dtype)));
    }

    PyTensor PyTensor::empty(const std::vector<int64_t>& shape,
                             const std::string& device,
                             const std::string& dtype) {
        return PyTensor(Tensor::empty(to_tensor_shape(shape), parse_device(device), parse_dtype(dtype)));
    }

    PyTensor PyTensor::randint(int64_t low, int64_t high,
                               const std::vector<int64_t>& shape,
                               const std::string& device) {
        return PyTensor(Tensor::randint(to_tensor_shape(shape), static_cast<int>(low), static_cast<int>(high),
                                        parse_device(device)));
    }

    // *_like variants
    PyTensor PyTensor::zeros_like(const PyTensor& other) {
        return PyTensor(Tensor::zeros_like(other.tensor_));
    }

    PyTensor PyTensor::ones_like(const PyTensor& other) {
        return PyTensor(Tensor::ones_like(other.tensor_));
    }

    PyTensor PyTensor::rand_like(const PyTensor& other) {
        return PyTensor(Tensor::rand_like(other.tensor_));
    }

    PyTensor PyTensor::randn_like(const PyTensor& other) {
        return PyTensor(Tensor::randn_like(other.tensor_));
    }

    PyTensor PyTensor::empty_like(const PyTensor& other) {
        return PyTensor(Tensor::empty_like(other.tensor_));
    }

    PyTensor PyTensor::full_like(const PyTensor& other, float value) {
        return PyTensor(Tensor::full_like(other.tensor_, value));
    }

    // Tensor combination
    PyTensor PyTensor::cat(const std::vector<PyTensor>& tensors, int dim) {
        std::vector<Tensor> core_tensors;
        core_tensors.reserve(tensors.size());
        for (const auto& t : tensors) {
            core_tensors.push_back(t.tensor_);
        }
        return PyTensor(Tensor::cat(core_tensors, dim));
    }

    PyTensor PyTensor::stack(const std::vector<PyTensor>& tensors, int dim) {
        std::vector<Tensor> core_tensors;
        core_tensors.reserve(tensors.size());
        for (const auto& t : tensors) {
            core_tensors.push_back(t.tensor_);
        }
        return PyTensor(Tensor::stack(core_tensors, dim));
    }

    PyTensor PyTensor::where(const PyTensor& condition, const PyTensor& x, const PyTensor& y) {
        return PyTensor(Tensor::where(condition.tensor_, x.tensor_, y.tensor_));
    }

    void register_tensor(nb::module_& m) {
        nb::class_<PyTensor>(m, "Tensor")
            .def(nb::init<>())

            // Properties
            .def_prop_ro("shape", &PyTensor::shape, "Tensor shape as tuple")
            .def_prop_ro("ndim", &PyTensor::ndim, "Number of dimensions")
            .def_prop_ro("numel", &PyTensor::numel, "Total number of elements")
            .def_prop_ro("device", &PyTensor::device, "Device: 'cpu' or 'cuda'")
            .def_prop_ro("dtype", &PyTensor::dtype, "Data type")
            .def_prop_ro("is_contiguous", &PyTensor::is_contiguous, "Whether memory is contiguous")
            .def_prop_ro("is_cuda", &PyTensor::is_cuda, "Whether tensor is on CUDA")

            // Memory operations
            .def("clone", &PyTensor::clone, "Deep copy of tensor")
            .def("cpu", &PyTensor::cpu, "Move tensor to CPU")
            .def("cuda", &PyTensor::cuda, "Move tensor to CUDA")
            .def("contiguous", &PyTensor::contiguous, "Make tensor contiguous")
            .def("sync", &PyTensor::sync, "Synchronize CUDA stream")
            .def("size", &PyTensor::size, nb::arg("dim"), "Size of dimension")

            // Scalar extraction
            .def("item", &PyTensor::item, "Extract scalar value")
            .def("float_", &PyTensor::item_float, "Extract as float")
            .def("int_", &PyTensor::item_int, "Extract as int")
            .def("bool_", &PyTensor::item_bool, "Extract as bool")

            // NumPy conversion
            .def("numpy", &PyTensor::numpy, nb::arg("copy") = true,
                 "Convert to NumPy array")
            .def_static("from_numpy", &PyTensor::from_numpy,
                        nb::arg("arr"), nb::arg("copy") = true,
                        "Create tensor from NumPy array")

            // Static creation functions
            .def_static("zeros", &PyTensor::zeros,
                        nb::arg("shape"), nb::arg("device") = "cuda", nb::arg("dtype") = "float32",
                        "Create tensor filled with zeros")
            .def_static("ones", &PyTensor::ones,
                        nb::arg("shape"), nb::arg("device") = "cuda", nb::arg("dtype") = "float32",
                        "Create tensor filled with ones")
            .def_static("full", &PyTensor::full,
                        nb::arg("shape"), nb::arg("value"), nb::arg("device") = "cuda", nb::arg("dtype") = "float32",
                        "Create tensor filled with value")
            .def_static("arange", static_cast<PyTensor (*)(float)>(&PyTensor::arange),
                        nb::arg("end"),
                        "Create 1D tensor with values from 0 to end")
            .def_static("arange", static_cast<PyTensor (*)(float, float, float, const std::string&, const std::string&)>(&PyTensor::arange),
                        nb::arg("start"), nb::arg("end"), nb::arg("step") = 1.0f, nb::arg("device") = "cuda", nb::arg("dtype") = "float32",
                        "Create 1D tensor with values from start to end")
            .def_static("linspace", &PyTensor::linspace,
                        nb::arg("start"), nb::arg("end"), nb::arg("steps"), nb::arg("device") = "cuda", nb::arg("dtype") = "float32",
                        "Create 1D tensor with evenly spaced values")
            .def_static("eye", static_cast<PyTensor (*)(int64_t, const std::string&, const std::string&)>(&PyTensor::eye),
                        nb::arg("n"), nb::arg("device") = "cuda", nb::arg("dtype") = "float32",
                        "Create identity matrix")
            .def_static("eye", static_cast<PyTensor (*)(int64_t, int64_t, const std::string&, const std::string&)>(&PyTensor::eye),
                        nb::arg("m"), nb::arg("n"), nb::arg("device") = "cuda", nb::arg("dtype") = "float32",
                        "Create m x n matrix with ones on diagonal")

            // DLPack protocol
            .def("__dlpack__", &PyTensor::dlpack, nb::arg("stream") = nb::none(), "Export as DLPack capsule")
            .def("__dlpack_device__", &PyTensor::dlpack_device, "Get DLPack device tuple")
            .def_static("from_dlpack", &PyTensor::from_dlpack, nb::arg("obj"), "Create tensor from DLPack capsule or object")

            // Indexing
            .def("__getitem__", &PyTensor::getitem, "Get item/slice")
            .def("__setitem__", &PyTensor::setitem, "Set item/slice")

            // Arithmetic operators
            .def("__add__", &PyTensor::add, "Add tensor")
            .def("__add__", &PyTensor::add_scalar, "Add scalar")
            .def("__radd__", &PyTensor::add_scalar, "Reverse add scalar")
            .def(
                "__iadd__", [](PyTensor& self, const PyTensor& other) -> PyTensor& {
                    return self.iadd(other);
                },
                nb::rv_policy::reference, "In-place add tensor")
            .def(
                "__iadd__", [](PyTensor& self, float scalar) -> PyTensor& { return self.iadd_scalar(scalar); }, nb::rv_policy::reference, "In-place add scalar")

            .def("__sub__", &PyTensor::sub, "Subtract tensor")
            .def("__sub__", &PyTensor::sub_scalar, "Subtract scalar")
            .def("__rsub__", &PyTensor::rsub_scalar, "Reverse subtract scalar")
            .def(
                "__isub__", [](PyTensor& self, const PyTensor& other) -> PyTensor& { return self.isub(other); }, nb::rv_policy::reference, "In-place subtract tensor")
            .def(
                "__isub__", [](PyTensor& self, float scalar) -> PyTensor& { return self.isub_scalar(scalar); }, nb::rv_policy::reference, "In-place subtract scalar")

            .def("__mul__", &PyTensor::mul, "Multiply tensor")
            .def("__mul__", &PyTensor::mul_scalar, "Multiply scalar")
            .def("__rmul__", &PyTensor::mul_scalar, "Reverse multiply scalar")
            .def(
                "__imul__", [](PyTensor& self, const PyTensor& other) -> PyTensor& { return self.imul(other); }, nb::rv_policy::reference, "In-place multiply tensor")
            .def(
                "__imul__", [](PyTensor& self, float scalar) -> PyTensor& { return self.imul_scalar(scalar); }, nb::rv_policy::reference, "In-place multiply scalar")

            .def("__truediv__", &PyTensor::div, "Divide tensor")
            .def("__truediv__", &PyTensor::div_scalar, "Divide scalar")
            .def("__rtruediv__", &PyTensor::rdiv_scalar, "Reverse divide scalar")
            .def(
                "__itruediv__", [](PyTensor& self, const PyTensor& other) -> PyTensor& { return self.idiv(other); }, nb::rv_policy::reference, "In-place divide tensor")
            .def(
                "__itruediv__", [](PyTensor& self, float scalar) -> PyTensor& { return self.idiv_scalar(scalar); }, nb::rv_policy::reference, "In-place divide scalar")

            .def("fill_", &PyTensor::fill_, nb::rv_policy::reference, "Fill tensor with value in-place")
            .def("zero_", &PyTensor::zero_, nb::rv_policy::reference, "Zero tensor in-place")

            .def("__neg__", &PyTensor::neg, "Negate")
            .def("__abs__", &PyTensor::abs, "Absolute value")

            // Unary math functions
            .def("sigmoid", &PyTensor::sigmoid, "Sigmoid activation")
            .def("exp", &PyTensor::exp, "Exponential")
            .def("log", &PyTensor::log, "Natural logarithm")
            .def("sqrt", &PyTensor::sqrt, "Square root")
            .def("relu", &PyTensor::relu, "ReLU activation")
            .def("sin", &PyTensor::sin, "Sine")
            .def("cos", &PyTensor::cos, "Cosine")
            .def("tan", &PyTensor::tan, "Tangent")
            .def("tanh", &PyTensor::tanh, "Hyperbolic tangent")
            .def("floor", &PyTensor::floor, "Floor")
            .def("ceil", &PyTensor::ceil, "Ceiling")
            .def("round", &PyTensor::round, "Round to nearest")
            .def("abs", &PyTensor::abs, "Absolute value")

            // Comparison operators
            .def("__eq__", &PyTensor::eq, "Equal tensor")
            .def("__eq__", &PyTensor::eq_scalar, "Equal scalar")
            .def("__ne__", &PyTensor::ne, "Not equal tensor")
            .def("__ne__", &PyTensor::ne_scalar, "Not equal scalar")
            .def("__lt__", &PyTensor::lt, "Less than tensor")
            .def("__lt__", &PyTensor::lt_scalar, "Less than scalar")
            .def("__le__", &PyTensor::le, "Less equal tensor")
            .def("__le__", &PyTensor::le_scalar, "Less equal scalar")
            .def("__gt__", &PyTensor::gt, "Greater than tensor")
            .def("__gt__", &PyTensor::gt_scalar, "Greater than scalar")
            .def("__ge__", &PyTensor::ge, "Greater equal tensor")
            .def("__ge__", &PyTensor::ge_scalar, "Greater equal scalar")

            // Logical operators
            .def("__and__", &PyTensor::logical_and, "Logical AND")
            .def("__or__", &PyTensor::logical_or, "Logical OR")
            .def("__invert__", &PyTensor::logical_not, "Logical NOT")

            // Reduction methods
            .def("sum", &PyTensor::sum, nb::arg("dim") = nb::none(), nb::arg("keepdim") = false, "Sum reduction")
            .def("mean", &PyTensor::mean, nb::arg("dim") = nb::none(), nb::arg("keepdim") = false, "Mean reduction")
            .def("max", &PyTensor::max, nb::arg("dim") = nb::none(), nb::arg("keepdim") = false, "Max reduction")
            .def("min", &PyTensor::min, nb::arg("dim") = nb::none(), nb::arg("keepdim") = false, "Min reduction")
            .def("sum_scalar", &PyTensor::sum_scalar, "Sum all elements to scalar")
            .def("mean_scalar", &PyTensor::mean_scalar, "Mean of all elements as scalar")
            .def("max_scalar", &PyTensor::max_scalar, "Max of all elements as scalar")
            .def("min_scalar", &PyTensor::min_scalar, "Min of all elements as scalar")

            // Shape operations
            .def("reshape", &PyTensor::reshape, nb::arg("shape"), "Reshape tensor")
            .def("view", &PyTensor::view, nb::arg("shape"), "View tensor with new shape")
            .def("squeeze", &PyTensor::squeeze, nb::arg("dim") = nb::none(), "Remove size-1 dimensions")
            .def("unsqueeze", &PyTensor::unsqueeze, nb::arg("dim"), "Add size-1 dimension")
            .def("transpose", &PyTensor::transpose, nb::arg("dim0"), nb::arg("dim1"), "Transpose dimensions")
            .def("permute", &PyTensor::permute, nb::arg("dims"), "Permute dimensions")
            .def("flatten", &PyTensor::flatten, nb::arg("start_dim") = 0, nb::arg("end_dim") = -1, "Flatten dimensions")
            .def("expand", &PyTensor::expand, nb::arg("sizes"), "Expand tensor to larger size")
            .def("repeat", &PyTensor::repeat, nb::arg("repeats"), "Repeat tensor along dimensions")
            .def("t", &PyTensor::t, "Transpose 2D tensor")

            // Extended unary operations
            .def("log2", &PyTensor::log2, "Base-2 logarithm")
            .def("log10", &PyTensor::log10, "Base-10 logarithm")
            .def("log1p", &PyTensor::log1p, "Log(1 + x)")
            .def("exp2", &PyTensor::exp2, "2^x")
            .def("rsqrt", &PyTensor::rsqrt, "Reciprocal square root")
            .def("square", &PyTensor::square, "Element-wise square")
            .def("asin", &PyTensor::asin, "Arc sine")
            .def("acos", &PyTensor::acos, "Arc cosine")
            .def("atan", &PyTensor::atan, "Arc tangent")
            .def("sinh", &PyTensor::sinh, "Hyperbolic sine")
            .def("cosh", &PyTensor::cosh, "Hyperbolic cosine")
            .def("trunc", &PyTensor::trunc, "Truncate to integer")
            .def("sign", &PyTensor::sign, "Sign of elements")
            .def("reciprocal", &PyTensor::reciprocal, "1/x")
            .def("gelu", &PyTensor::gelu, "GELU activation")
            .def("swish", &PyTensor::swish, "Swish activation")
            .def("isnan", &PyTensor::isnan, "Check for NaN")
            .def("isinf", &PyTensor::isinf, "Check for infinity")
            .def("isfinite", &PyTensor::isfinite, "Check for finite values")

            // Power operations
            .def("pow", static_cast<PyTensor (PyTensor::*)(float) const>(&PyTensor::pow), nb::arg("exponent"), "Power with scalar exponent")
            .def("pow", static_cast<PyTensor (PyTensor::*)(const PyTensor&) const>(&PyTensor::pow), nb::arg("exponent"), "Power with tensor exponent")
            .def("__pow__", static_cast<PyTensor (PyTensor::*)(float) const>(&PyTensor::pow), "Power operator")

            // Extended reductions
            .def("prod", &PyTensor::prod, nb::arg("dim") = nb::none(), nb::arg("keepdim") = false, "Product reduction")
            .def("std", &PyTensor::std, nb::arg("dim") = nb::none(), nb::arg("keepdim") = false, "Standard deviation")
            .def("var", &PyTensor::var, nb::arg("dim") = nb::none(), nb::arg("keepdim") = false, "Variance")
            .def("argmax", &PyTensor::argmax, nb::arg("dim") = nb::none(), nb::arg("keepdim") = false, "Index of maximum")
            .def("argmin", &PyTensor::argmin, nb::arg("dim") = nb::none(), nb::arg("keepdim") = false, "Index of minimum")
            .def("all", &PyTensor::all, nb::arg("dim") = nb::none(), nb::arg("keepdim") = false, "Check if all true")
            .def("any", &PyTensor::any, nb::arg("dim") = nb::none(), nb::arg("keepdim") = false, "Check if any true")
            .def("norm", &PyTensor::norm, nb::arg("p") = 2.0f, "Lp norm")
            .def("norm_scalar", &PyTensor::norm_scalar, nb::arg("p") = 2.0f, "Lp norm as scalar")

            // Advanced indexing
            .def("index_select", &PyTensor::index_select, nb::arg("dim"), nb::arg("indices"), "Select along dimension by indices")
            .def("gather", &PyTensor::gather, nb::arg("dim"), nb::arg("indices"), "Gather values along dimension")
            .def("masked_select", &PyTensor::masked_select, nb::arg("mask"), "Select elements where mask is true")
            .def("masked_fill", &PyTensor::masked_fill, nb::arg("mask"), nb::arg("value"), "Fill elements where mask is true")
            .def("masked_fill_", &PyTensor::masked_fill_, nb::arg("mask"), nb::arg("value"), nb::rv_policy::reference, "In-place fill elements where mask is true")
            .def("nonzero", &PyTensor::nonzero, "Indices of non-zero elements")

            // Linear algebra
            .def("matmul", &PyTensor::matmul, nb::arg("other"), "Matrix multiplication")
            .def("mm", &PyTensor::mm, nb::arg("other"), "Matrix multiplication (2D)")
            .def("bmm", &PyTensor::bmm, nb::arg("other"), "Batched matrix multiplication")
            .def("dot", &PyTensor::dot, nb::arg("other"), "Dot product")
            .def("__matmul__", &PyTensor::matmul, "Matrix multiplication operator")

            // Element-wise operations
            .def("clamp", &PyTensor::clamp, nb::arg("min"), nb::arg("max"), "Clamp values to range")
            .def("maximum", &PyTensor::maximum, nb::arg("other"), "Element-wise maximum")
            .def("minimum", &PyTensor::minimum, nb::arg("other"), "Element-wise minimum")

            // Type conversion
            .def("to", &PyTensor::to_dtype, nb::arg("dtype"), "Convert to specified dtype")

            // Random tensor creation
            .def_static("rand", &PyTensor::rand, nb::arg("shape"), nb::arg("device") = "cuda", nb::arg("dtype") = "float32", "Create tensor with uniform random values [0, 1)")
            .def_static("randn", &PyTensor::randn, nb::arg("shape"), nb::arg("device") = "cuda", nb::arg("dtype") = "float32", "Create tensor with normal random values")
            .def_static("empty", &PyTensor::empty, nb::arg("shape"), nb::arg("device") = "cuda", nb::arg("dtype") = "float32", "Create uninitialized tensor")
            .def_static("randint", &PyTensor::randint, nb::arg("low"), nb::arg("high"), nb::arg("shape"), nb::arg("device") = "cuda", "Create tensor with random integers")

            // *_like variants
            .def_static("zeros_like", &PyTensor::zeros_like, nb::arg("other"), "Create zeros tensor like other")
            .def_static("ones_like", &PyTensor::ones_like, nb::arg("other"), "Create ones tensor like other")
            .def_static("rand_like", &PyTensor::rand_like, nb::arg("other"), "Create random tensor like other")
            .def_static("randn_like", &PyTensor::randn_like, nb::arg("other"), "Create normal random tensor like other")
            .def_static("empty_like", &PyTensor::empty_like, nb::arg("other"), "Create uninitialized tensor like other")
            .def_static("full_like", &PyTensor::full_like, nb::arg("other"), nb::arg("value"), "Create filled tensor like other")

            // Tensor combination
            .def_static("cat", &PyTensor::cat, nb::arg("tensors"), nb::arg("dim") = 0, "Concatenate tensors")
            .def_static("stack", &PyTensor::stack, nb::arg("tensors"), nb::arg("dim") = 0, "Stack tensors")
            .def_static("where", &PyTensor::where, nb::arg("condition"), nb::arg("x"), nb::arg("y"), "Conditional select")

            // String representation
            .def("__repr__", &PyTensor::repr, "String representation")

            // __array__ protocol for zero-copy NumPy interop (CPU only)
            // Allows: np.asarray(tensor) for zero-copy when tensor is CPU + contiguous
            .def(
                "__array__", [](PyTensor& self, nb::object dtype) -> nb::object {
                    (void)dtype;              // We return our native dtype, ignore requested dtype
                    return self.numpy(false); // Zero-copy
                },
                nb::arg("dtype") = nb::none(), "Return numpy array view (zero-copy for CPU contiguous tensors)");
    }

} // namespace lfs::python
