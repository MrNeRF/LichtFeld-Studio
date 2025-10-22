/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/logger.hpp"
#include "core_new/pinned_memory_allocator.hpp"
#include "internal/memory_pool.hpp"
#include "internal/tensor_impl.hpp"
#include "internal/tensor_broadcast.hpp"
#include "internal/tensor_functors.hpp"
#include "internal/tensor_ops.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <numeric>

namespace lfs::core {

    // ============= CORE UNIFIED OPERATIONS =============
    constexpr static DataType promote_types(DataType a, DataType b) {
        if (a == b)
            return a;

        if (a == DataType::Bool) {
            if (b == DataType::Float32 || b == DataType::Float16)
                return b;
            if (b == DataType::Int32 || b == DataType::Int64)
                return b;
            return DataType::Float32;
        }
        if (b == DataType::Bool) {
            if (a == DataType::Float32 || a == DataType::Float16)
                return a;
            if (a == DataType::Int32 || a == DataType::Int64)
                return a;
            return DataType::Float32;
        }

        if ((a == DataType::Int32 || a == DataType::Int64) &&
            (b == DataType::Float32 || b == DataType::Float16)) {
            return (b == DataType::Float16) ? DataType::Float16 : DataType::Float32;
        }
        if ((b == DataType::Int32 || b == DataType::Int64) &&
            (a == DataType::Float32 || a == DataType::Float16)) {
            return (a == DataType::Float16) ? DataType::Float16 : DataType::Float32;
        }

        if ((a == DataType::Int32 && b == DataType::Int64) ||
            (a == DataType::Int64 && b == DataType::Int32)) {
            return DataType::Int64;
        }

        if ((a == DataType::Float16 && b == DataType::Float32) ||
            (a == DataType::Float32 && b == DataType::Float16)) {
            return DataType::Float32;
        }

        return DataType::Float32;
    }

    Tensor Tensor::load(LoadOp op, const LoadArgs& args) {
        Tensor result;

        switch (op) {
        case LoadOp::Empty: {
            result.shape_ = args.shape;
            result.strides_ = args.shape.strides(); // Initialize to contiguous strides
            result.storage_offset_ = 0;
            result.is_contiguous_ = true;
            result.device_ = args.device;
            result.dtype_ = args.dtype;
            result.id_ = next_id_++;

            size_t bytes = result.shape_.elements() * dtype_size(result.dtype_);

            if (bytes == 0) {
                // Create a dummy allocation to hold a valid shared_ptr
                // We allocate 1 byte even though we don't need it
                if (result.device_ == Device::CUDA) {
                    void* dummy = CudaMemoryPool::instance().allocate(1, nullptr);
                    result.data_owner_ = std::shared_ptr<void>(dummy, [](void* p) {
                        CudaMemoryPool::instance().deallocate(p, nullptr);
                    });
                } else {
                    // Even dummy allocations use pinned memory
                    void* dummy = PinnedMemoryAllocator::instance().allocate(1);
                    result.data_owner_ = std::shared_ptr<void>(dummy, [](void* p) {
                        if (p)
                            PinnedMemoryAllocator::instance().deallocate(p);
                    });
                }
                result.data_ = nullptr; // Empty tensor has no usable data
                return result;
            }

            if (result.device_ == Device::CUDA) {
                void* ptr = CudaMemoryPool::instance().allocate(bytes, nullptr);
                if (!ptr) {
                    LOG_ERROR("Failed to allocate {} bytes from memory pool", bytes);
                    return Tensor();
                }
                result.data_owner_ = std::shared_ptr<void>(ptr, [](void* p) {
                    CudaMemoryPool::instance().deallocate(p, nullptr);
                });
                result.data_ = result.data_owner_.get();
            } else {
                // Use pinned memory for CPU tensors (2-3x faster PCIe bandwidth)
                void* ptr = PinnedMemoryAllocator::instance().allocate(bytes);
                if (!ptr) {
                    LOG_ERROR("Failed to allocate {} bytes on CPU (pinned memory)", bytes);
                    return Tensor();
                }
                result.data_owner_ = std::shared_ptr<void>(ptr, [](void* p) {
                    if (p)
                        PinnedMemoryAllocator::instance().deallocate(p);
                });
                result.data_ = result.data_owner_.get();
            }
            break;
        }

        case LoadOp::Const: {
            float value = std::get<float>(args.args);
            result = load(LoadOp::Empty, args);
            if (!result.is_valid() || result.numel() == 0)
                return result;

            if (result.device_ == Device::CUDA) {
                if (result.dtype_ == DataType::Float32) {
                    if (value == 0.0f) {
                        cudaMemset(result.data_, 0, result.bytes());
                    } else {
                        tensor_ops::launch_load_op(
                            result.data_,
                            result.shape_.dims().data(),
                            result.shape_.rank(),
                            LoadOp::Const,
                            &value,
                            result.dtype_,
                            nullptr);
                        cudaDeviceSynchronize();
                    }
                } else if (result.dtype_ == DataType::Bool) {
                    unsigned char fill_val = (value != 0.0f) ? 1 : 0;
                    cudaMemset(result.data_, fill_val, result.bytes());
                } else if (result.dtype_ == DataType::Int32) {
                    if (value == 0.0f) {
                        cudaMemset(result.data_, 0, result.bytes());
                    } else {
                        std::vector<int> temp(result.numel(), static_cast<int>(value));
                        cudaMemcpy(result.data_, temp.data(), result.bytes(), cudaMemcpyHostToDevice);
                    }
                }
            } else {
                if (result.dtype_ == DataType::Float32) {
                    float* ptr = static_cast<float*>(result.data_);
                    std::fill_n(ptr, result.numel(), value);
                } else if (result.dtype_ == DataType::Bool) {
                    unsigned char* ptr = static_cast<unsigned char*>(result.data_);
                    std::fill_n(ptr, result.numel(), value != 0 ? 1 : 0);
                } else if (result.dtype_ == DataType::Int32) {
                    int* ptr = static_cast<int*>(result.data_);
                    std::fill_n(ptr, result.numel(), static_cast<int>(value));
                }
            }
            break;
        }

        case LoadOp::Arange: {
            auto [start, end, step] = std::get<std::tuple<float, float, float>>(args.args);

            if (step == 0) {
                LOG_ERROR("Step cannot be zero");
                return Tensor();
            }

            if ((end - start) * step < 0) {
                LOG_ERROR("Invalid range: start={}, end={}, step={}", start, end, step);
                return Tensor();
            }

            size_t count = static_cast<size_t>(std::ceil((end - start) / step));

            result.shape_ = TensorShape{count};
            result.strides_ = result.shape_.strides(); // Initialize to contiguous strides
            result.storage_offset_ = 0;
            result.is_contiguous_ = true;
            result.device_ = args.device;
            result.dtype_ = args.dtype;
            result.id_ = next_id_++;

            size_t bytes = count * dtype_size(result.dtype_);

            if (result.device_ == Device::CUDA) {
                void* ptr = CudaMemoryPool::instance().allocate(bytes, nullptr);
                if (!ptr) {
                    LOG_ERROR("Failed to allocate {} bytes from memory pool", bytes);
                    return Tensor();
                }
                result.data_owner_ = std::shared_ptr<void>(ptr, [](void* p) {
                    CudaMemoryPool::instance().deallocate(p, nullptr);
                });
                result.data_ = result.data_owner_.get();

                if (result.dtype_ == DataType::Float32) {
                    std::vector<float> data(count);
                    for (size_t i = 0; i < count; ++i) {
                        data[i] = start + i * step;
                    }
                    cudaMemcpy(result.data_, data.data(), bytes, cudaMemcpyHostToDevice);
                } else if (result.dtype_ == DataType::Int32) {
                    std::vector<int> data(count);
                    for (size_t i = 0; i < count; ++i) {
                        data[i] = static_cast<int>(start + i * step);
                    }
                    cudaMemcpy(result.data_, data.data(), bytes, cudaMemcpyHostToDevice);
                }
            } else {
                // Use pinned memory for CPU tensors
                void* ptr = PinnedMemoryAllocator::instance().allocate(bytes);
                if (!ptr) {
                    LOG_ERROR("Failed to allocate {} bytes on CPU (pinned memory)", bytes);
                    return Tensor();
                }
                result.data_owner_ = std::shared_ptr<void>(ptr, [](void* p) {
                    if (p)
                        PinnedMemoryAllocator::instance().deallocate(p);
                });
                result.data_ = result.data_owner_.get();

                if (result.dtype_ == DataType::Float32) {
                    float* data_ptr = static_cast<float*>(result.data_);
                    for (size_t i = 0; i < count; ++i) {
                        data_ptr[i] = start + i * step;
                    }
                } else if (result.dtype_ == DataType::Int32) {
                    int* data_ptr = static_cast<int*>(result.data_);
                    for (size_t i = 0; i < count; ++i) {
                        data_ptr[i] = static_cast<int>(start + i * step);
                    }
                }
            }
            break;
        }

        case LoadOp::Random: {
            auto [low, high] = std::get<std::pair<float, float>>(args.args);
            result = load(LoadOp::Empty, args);
            if (!result.is_valid() || result.numel() == 0)
                return result;

            if (result.device_ == Device::CUDA) {
                if (result.dtype_ == DataType::Float32) {
                    tensor_ops::launch_uniform(result.ptr<float>(), result.numel(), low, high,
                                               RandomGenerator::instance().get_next_cuda_seed(), 0);
                    cudaDeviceSynchronize();
                } else if (result.dtype_ == DataType::Int32) {
                    tensor_ops::launch_randint(result.ptr<int>(), result.numel(),
                                               static_cast<int>(low), static_cast<int>(high),
                                               RandomGenerator::instance().get_next_cuda_seed(), 0);
                    cudaDeviceSynchronize();
                }
            } else {
                auto& gen = *static_cast<std::mt19937_64*>(
                    RandomGenerator::instance().get_generator(Device::CPU));

                if (result.dtype_ == DataType::Float32) {
                    std::uniform_real_distribution<float> dist(low, high);
                    float* data = result.ptr<float>();
                    for (size_t i = 0; i < result.numel(); ++i) {
                        data[i] = dist(gen);
                    }
                } else if (result.dtype_ == DataType::Int32) {
                    std::uniform_int_distribution<int> dist(static_cast<int>(low),
                                                            static_cast<int>(high) - 1);
                    int* data = result.ptr<int>();
                    for (size_t i = 0; i < result.numel(); ++i) {
                        data[i] = dist(gen);
                    }
                }
            }
            break;
        }

        case LoadOp::Normal: {
            auto [mean, std] = std::get<std::pair<float, float>>(args.args);
            result = load(LoadOp::Empty, args);
            if (!result.is_valid() || result.numel() == 0)
                return result;

            if (result.device_ == Device::CUDA) {
                tensor_ops::launch_normal(result.ptr<float>(), result.numel(), mean, std,
                                          RandomGenerator::instance().get_next_cuda_seed(), 0);
                cudaDeviceSynchronize();
            } else {
                auto& gen = *static_cast<std::mt19937_64*>(
                    RandomGenerator::instance().get_generator(Device::CPU));
                std::normal_distribution<float> dist(mean, std);
                float* data = result.ptr<float>();
                for (size_t i = 0; i < result.numel(); ++i) {
                    data[i] = dist(gen);
                }
            }
            break;
        }

        case LoadOp::Randint: {
            auto [low, high] = std::get<std::pair<int, int>>(args.args);
            result = load(LoadOp::Empty, args);
            if (!result.is_valid() || result.numel() == 0)
                return result;

            if (result.device_ == Device::CUDA) {
                if (result.dtype_ == DataType::Int32) {
                    tensor_ops::launch_randint(result.ptr<int>(), result.numel(), low, high,
                                               RandomGenerator::instance().get_next_cuda_seed(), 0);
                    cudaDeviceSynchronize();
                } else if (result.dtype_ == DataType::Float32) {
                    int* temp_buffer = static_cast<int*>(
                        CudaMemoryPool::instance().allocate(result.numel() * sizeof(int), nullptr));

                    if (temp_buffer) {
                        tensor_ops::launch_randint(temp_buffer, result.numel(), low, high,
                                                   RandomGenerator::instance().get_next_cuda_seed(), 0);

                        tensor_ops::launch_convert_type<int, float>(temp_buffer, result.ptr<float>(),
                                                                    result.numel(), 0);
                        cudaDeviceSynchronize();

                        CudaMemoryPool::instance().deallocate(temp_buffer, nullptr);
                    } else {
                        LOG_ERROR("Failed to allocate temp buffer from memory pool");
                    }
                }
            } else {
                auto& gen = *static_cast<std::mt19937_64*>(
                    RandomGenerator::instance().get_generator(Device::CPU));
                std::uniform_int_distribution<int> dist(low, high - 1);

                if (result.dtype_ == DataType::Int32) {
                    int* data = result.ptr<int>();
                    for (size_t i = 0; i < result.numel(); ++i) {
                        data[i] = dist(gen);
                    }
                } else if (result.dtype_ == DataType::Float32) {
                    float* data = result.ptr<float>();
                    for (size_t i = 0; i < result.numel(); ++i) {
                        data[i] = static_cast<float>(dist(gen));
                    }
                }
            }
            break;
        }

        case LoadOp::Bernoulli: {
            float p = std::get<float>(args.args);
            result = load(LoadOp::Empty, args);
            if (!result.is_valid() || result.numel() == 0)
                return result;

            if (result.device_ == Device::CUDA) {
                tensor_ops::launch_bernoulli(result.ptr<float>(), result.numel(), p,
                                             RandomGenerator::instance().get_next_cuda_seed(), 0);
                cudaDeviceSynchronize();
            } else {
                auto& gen = *static_cast<std::mt19937_64*>(
                    RandomGenerator::instance().get_generator(Device::CPU));
                std::bernoulli_distribution dist(p);
                float* data = result.ptr<float>();
                for (size_t i = 0; i < result.numel(); ++i) {
                    data[i] = dist(gen) ? 1.0f : 0.0f;
                }
            }
            break;
        }

        case LoadOp::Multinomial: {
            auto [weights_ptr, replacement] = std::get<std::pair<void*, bool>>(args.args);
            const Tensor* weights = static_cast<const Tensor*>(weights_ptr);

            if (!weights->is_valid() || weights->ndim() != 1) {
                LOG_ERROR("Multinomial requires 1D weight tensor");
                return Tensor();
            }

            size_t n = weights->numel();
            size_t num_samples = args.shape.elements();

            result = load(LoadOp::Empty, args);
            if (!result.is_valid())
                return result;

            if (weights->device() == Device::CUDA) {
                tensor_ops::launch_multinomial(weights->ptr<float>(), result.ptr<int>(),
                                               n, num_samples, replacement,
                                               RandomGenerator::instance().get_next_cuda_seed(), 0);
                cudaDeviceSynchronize();
            } else {
                auto weights_data = weights->to_vector();

                float sum = std::accumulate(weights_data.begin(), weights_data.end(), 0.0f);
                if (sum <= 0) {
                    LOG_ERROR("Weights must sum to positive value");
                    return Tensor();
                }

                std::vector<float> cdf(n);
                cdf[0] = weights_data[0] / sum;
                for (size_t i = 1; i < n; ++i) {
                    cdf[i] = cdf[i - 1] + weights_data[i] / sum;
                }

                auto& gen = *static_cast<std::mt19937_64*>(
                    RandomGenerator::instance().get_generator(Device::CPU));
                std::uniform_real_distribution<float> dis(0.0f, 1.0f);

                int* samples = result.ptr<int>();

                if (replacement) {
                    for (size_t i = 0; i < num_samples; ++i) {
                        float u = dis(gen);
                        auto it = std::lower_bound(cdf.begin(), cdf.end(), u);
                        samples[i] = static_cast<int>(std::distance(cdf.begin(), it));
                    }
                } else {
                    std::vector<std::pair<float, int>> keys(n);

                    for (size_t i = 0; i < n; ++i) {
                        float u = dis(gen);
                        u = std::clamp(u, 1e-10f, 1.0f - 1e-10f);
                        float gumbel = -std::log(-std::log(u));
                        float log_weight = std::log(std::max(weights_data[i], 1e-10f));
                        keys[i] = {log_weight + gumbel, static_cast<int>(i)};
                    }

                    std::sort(keys.begin(), keys.end(),
                              [](const auto& a, const auto& b) { return a.first > b.first; });

                    for (size_t i = 0; i < num_samples; ++i) {
                        samples[i] = keys[i].second;
                    }
                }
            }
            break;
        }

        case LoadOp::Eye: {
            result = load(LoadOp::Const, {args.shape, args.device, args.dtype, 0.0f});
            if (!result.is_valid() || args.shape.rank() != 2)
                return result;

            size_t m = args.shape[0];
            size_t n = args.shape[1];
            size_t min_dim = std::min(m, n);

            if (result.device_ == Device::CUDA) {
                tensor_ops::launch_eye(result.ptr<float>(), m, n, 0);
                cudaDeviceSynchronize();
            } else {
                float* data = result.ptr<float>();
                for (size_t i = 0; i < min_dim; ++i) {
                    data[i * n + i] = 1.0f;
                }
            }
            break;
        }

        case LoadOp::FromCPU: {
            void* src_ptr = std::get<void*>(args.args);
            if (!src_ptr) {
                LOG_ERROR("FromCPU requires valid source pointer");
                return Tensor();
            }

            result = Tensor(src_ptr, args.shape, Device::CPU, args.dtype);

            if (args.device == Device::CUDA) {
                result = result.to(Device::CUDA);
            }
            break;
        }

        case LoadOp::FromCUDA: {
            void* src_ptr = std::get<void*>(args.args);
            if (!src_ptr) {
                LOG_ERROR("FromCUDA requires valid source pointer");
                return Tensor();
            }

            result = Tensor(src_ptr, args.shape, Device::CUDA, args.dtype);

            if (args.device == Device::CPU) {
                result = result.to(Device::CPU);
            }
            break;
        }

        default:
            LOG_ERROR("Unknown load operation");
            break;
        }

        return result;
    }

    Tensor Tensor::multinomial(const Tensor& weights, int num_samples, bool replacement) {
        if (!replacement && static_cast<size_t>(num_samples) > weights.numel()) {
            num_samples = static_cast<int>(weights.numel());
        }

        LoadArgs args;
        args.shape = TensorShape({static_cast<size_t>(num_samples)});
        args.device = weights.device();
        args.dtype = DataType::Int32;
        args.args = std::pair<void*, bool>{const_cast<void*>(static_cast<const void*>(&weights)), replacement};
        return load(LoadOp::Multinomial, args);
    }

    Tensor Tensor::reduce(ReduceOp op, const ReduceArgs& args) const {
        if (!validate_unary_op()) {
            return Tensor();
        }

        // Special handling for Std and Var
        if (op == ReduceOp::Std || op == ReduceOp::Var) {
            // Use the dedicated unbiased field from ReduceArgs
            bool unbiased = args.unbiased;

            ReduceArgs mean_args = args;
            mean_args.args = std::monostate{}; // Clear variant args for mean calculation
            auto mean_tensor = reduce(ReduceOp::Mean, mean_args);

            Tensor mean_broadcast = (mean_tensor.shape() == shape_)
                                        ? mean_tensor.clone()
                                        : mean_tensor.broadcast_to(shape_);

            auto diff = this->sub(mean_broadcast);
            auto squared = diff.mul(diff);

            // Compute sum of squared differences
            auto sum_sq = squared.reduce(ReduceOp::Sum, mean_args);

            // Calculate N (number of elements being reduced)
            std::vector<int> axes = args.axes;
            if (axes.empty()) {
                axes.resize(shape_.rank());
                std::iota(axes.begin(), axes.end(), 0);
            }

            size_t reduce_count = 1;
            for (int ax : axes) {
                int resolved = resolve_dim(ax);
                if (resolved >= 0 && resolved < static_cast<int>(shape_.rank())) {
                    reduce_count *= shape_[resolved];
                }
            }

            // Apply Bessel's correction if unbiased and N > 1
            float correction = static_cast<float>(reduce_count);
            if (unbiased && reduce_count > 1) {
                correction = static_cast<float>(reduce_count - 1);
            }

            auto variance = sum_sq.div(correction);

            if (op == ReduceOp::Var) {
                return variance;
            } else {
                return variance.sqrt();
            }
        }

        std::vector<int> axes = args.axes;
        if (axes.empty()) {
            axes.resize(shape_.rank());
            std::iota(axes.begin(), axes.end(), 0);
        }

        // Resolve negative indices to positive indices
        for (auto& ax : axes) {
            ax = resolve_dim(ax);
        }

        std::vector<size_t> out_shape;
        for (size_t i = 0; i < shape_.rank(); ++i) {
            bool is_reduced = std::find(axes.begin(), axes.end(), static_cast<int>(i)) != axes.end();
            if (!is_reduced || args.keepdim) {
                out_shape.push_back(is_reduced ? 1 : shape_[i]);
            }
        }

        DataType out_dtype = dtype_;
        if (op == ReduceOp::Any || op == ReduceOp::All) {
            out_dtype = DataType::Bool;
        } else if (op == ReduceOp::Argmax || op == ReduceOp::Argmin) {
            out_dtype = DataType::Int64;
        }

        auto result = Tensor::empty(TensorShape(out_shape), device_, out_dtype);

        if (numel() == 0) {
            float identity_value = 0.0f;
            switch (op) {
            case ReduceOp::Sum:
            case ReduceOp::Mean:
                identity_value = 0.0f;
                break;
            case ReduceOp::Prod:
                identity_value = 1.0f;
                break;
            case ReduceOp::Max:
                identity_value = -std::numeric_limits<float>::infinity();
                break;
            case ReduceOp::Min:
                identity_value = std::numeric_limits<float>::infinity();
                break;
            case ReduceOp::Any:
                identity_value = 0.0f;
                break;
            case ReduceOp::All:
                identity_value = 1.0f;
                break;
            default:
                identity_value = 0.0f;
                break;
            }

            if (device_ == Device::CUDA) {
                if (out_dtype == DataType::Float32) {
                    std::vector<float> temp(result.numel(), identity_value);
                    cudaMemcpy(result.raw_ptr(), temp.data(),
                               result.bytes(), cudaMemcpyHostToDevice);
                } else if (out_dtype == DataType::Bool) {
                    unsigned char bool_val = (identity_value != 0.0f) ? 1 : 0;
                    cudaMemset(result.raw_ptr(), bool_val, result.bytes());
                }
            } else {
                if (out_dtype == DataType::Float32) {
                    float* ptr = static_cast<float*>(result.raw_ptr());
                    std::fill_n(ptr, result.numel(), identity_value);
                } else if (out_dtype == DataType::Bool) {
                    unsigned char* ptr = static_cast<unsigned char*>(result.raw_ptr());
                    std::fill_n(ptr, result.numel(), identity_value != 0.0f ? 1 : 0);
                }
            }
            return result;
        }

        if (device_ == Device::CUDA) {
            tensor_ops::launch_reduce_op(
                raw_ptr(), result.raw_ptr(),
                shape_.dims().data(), shape_.rank(),
                axes.data(), axes.size(),
                args.keepdim, op,
                dtype_, nullptr);
            cudaDeviceSynchronize();
        } else {
            // CPU implementation
            const float* src = static_cast<const float*>(raw_ptr());
            float* dst = static_cast<float*>(result.raw_ptr());

            // Full reduction to scalar
            if (axes.size() == shape_.rank()) {
                if (op == ReduceOp::Sum) {
                    float sum = 0;
                    for (size_t i = 0; i < numel(); ++i) {
                        sum += src[i];
                    }
                    dst[0] = sum;
                } else if (op == ReduceOp::Mean) {
                    float sum = 0;
                    for (size_t i = 0; i < numel(); ++i) {
                        sum += src[i];
                    }
                    dst[0] = sum / numel();
                } else if (op == ReduceOp::Max) {
                    float max_val = src[0];
                    for (size_t i = 1; i < numel(); ++i) {
                        max_val = std::max(max_val, src[i]);
                    }
                    dst[0] = max_val;
                } else if (op == ReduceOp::Min) {
                    float min_val = src[0];
                    for (size_t i = 1; i < numel(); ++i) {
                        min_val = std::min(min_val, src[i]);
                    }
                    dst[0] = min_val;
                } else if (op == ReduceOp::Prod) {
                    float prod = 1.0f;
                    for (size_t i = 0; i < numel(); ++i) {
                        prod *= src[i];
                    }
                    dst[0] = prod;
                }
                return result;
            }

            // Axis-specific reduction - general implementation
            // Build mask of which dimensions are reduced
            std::vector<bool> is_reduced_dim(shape_.rank(), false);
            for (int ax : axes) {
                int resolved = resolve_dim(ax);
                if (resolved >= 0 && resolved < static_cast<int>(shape_.rank())) {
                    is_reduced_dim[resolved] = true;
                }
            }

            // Calculate input strides
            auto input_strides = shape_.strides();

            // Calculate output strides
            std::vector<size_t> out_shape_vec;
            for (size_t i = 0; i < shape_.rank(); ++i) {
                if (!is_reduced_dim[i]) {
                    out_shape_vec.push_back(shape_[i]);
                }
            }

            std::vector<size_t> output_strides;
            if (!out_shape_vec.empty()) {
                output_strides.resize(out_shape_vec.size());
                output_strides.back() = 1;
                for (int i = static_cast<int>(out_shape_vec.size()) - 2; i >= 0; --i) {
                    output_strides[i] = output_strides[i + 1] * out_shape_vec[i + 1];
                }
            }

            size_t output_elements = result.numel();

            // Calculate how many elements to reduce per output element
            size_t reduce_count = 1;
            std::vector<size_t> reduced_dims;
            for (size_t i = 0; i < shape_.rank(); ++i) {
                if (is_reduced_dim[i]) {
                    reduced_dims.push_back(i);
                    reduce_count *= shape_[i];
                }
            }

            // Perform reduction
            for (size_t out_idx = 0; out_idx < output_elements; ++out_idx) {
                // Convert output linear index to coordinates in output space
                std::vector<size_t> out_coords;
                if (!out_shape_vec.empty()) {
                    out_coords.resize(out_shape_vec.size());
                    size_t temp = out_idx;
                    for (size_t i = 0; i < out_shape_vec.size(); ++i) {
                        out_coords[i] = temp / output_strides[i];
                        temp %= output_strides[i];
                    }
                }

                // Map output coords back to base input coords
                std::vector<size_t> base_input_coords(shape_.rank());
                size_t out_coord_idx = 0;
                for (size_t i = 0; i < shape_.rank(); ++i) {
                    if (!is_reduced_dim[i]) {
                        base_input_coords[i] = out_coords[out_coord_idx++];
                    } else {
                        base_input_coords[i] = 0;
                    }
                }

                // Initialize result with identity value for this output element
                float result_val = 0.0f;
                if (op == ReduceOp::Max) {
                    result_val = -std::numeric_limits<float>::infinity();
                } else if (op == ReduceOp::Min) {
                    result_val = std::numeric_limits<float>::infinity();
                } else if (op == ReduceOp::Prod) {
                    result_val = 1.0f;
                }

                // Iterate through all combinations of reduced dimensions
                for (size_t r = 0; r < reduce_count; ++r) {
                    // Compute coordinates in the reduced dimensions
                    size_t temp_r = r;
                    std::vector<size_t> full_input_coords = base_input_coords;

                    // Fill in reduced dimensions - work backwards for row-major order
                    for (int rd_idx = static_cast<int>(reduced_dims.size()) - 1; rd_idx >= 0; --rd_idx) {
                        size_t dim = reduced_dims[rd_idx];
                        full_input_coords[dim] = temp_r % shape_[dim];
                        temp_r /= shape_[dim];
                    }

                    // Calculate linear input index
                    size_t in_idx = 0;
                    for (size_t i = 0; i < shape_.rank(); ++i) {
                        in_idx += full_input_coords[i] * input_strides[i];
                    }

                    // Apply reduction operation
                    float val = src[in_idx];
                    switch (op) {
                    case ReduceOp::Sum:
                    case ReduceOp::Mean: // Mean accumulates like sum, then divides at end
                        result_val += val;
                        break;
                    case ReduceOp::Max:
                        result_val = std::max(result_val, val);
                        break;
                    case ReduceOp::Min:
                        result_val = std::min(result_val, val);
                        break;
                    case ReduceOp::Prod:
                        result_val *= val;
                        break;
                    default:
                        break;
                    }
                }

                // Store result (apply mean if needed)
                if (op == ReduceOp::Mean) {
                    dst[out_idx] = result_val / reduce_count;
                } else {
                    dst[out_idx] = result_val;
                }
            }
        }

        return result;
    }

    // ============= TERNARY OPERATIONS =============

    Tensor Tensor::ternary(const Tensor& b, const Tensor& c) const {
        if (!validate_ternary_op(b, c)) {
            return Tensor();
        }

        if (numel() == 0 || b.numel() == 0 || c.numel() == 0) {
            auto shape_ab = this->broadcast_shape(b.shape());
            if (shape_ab.rank() == 0) {
                LOG_ERROR("Incompatible shapes for first two tensors in ternary operation with empty tensors");
                return Tensor();
            }

            auto shape_abc_vec = broadcast::shape(shape_ab.dims(), c.shape().dims());
            if (shape_abc_vec.empty()) {
                LOG_ERROR("Incompatible shapes for ternary operation");
                return Tensor();
            }

            DataType out_dtype = promote_types(b.dtype(), c.dtype());
            return empty(TensorShape(shape_abc_vec), device_, out_dtype);
        }

        if (dtype_ != DataType::Bool) {
            LOG_ERROR("Where operation requires boolean condition tensor");
            return Tensor();
        }

        auto shape_ab = this->broadcast_shape(b.shape());
        if (shape_ab.rank() == 0) {
            LOG_ERROR("Incompatible shapes for first two tensors in ternary operation");
            return Tensor();
        }

        auto shape_abc_vec = broadcast::shape(shape_ab.dims(), c.shape().dims());
        if (shape_abc_vec.empty()) {
            LOG_ERROR("Incompatible shapes for ternary operation");
            return Tensor();
        }

        TensorShape shape_abc(shape_abc_vec);

        DataType out_dtype = promote_types(b.dtype(), c.dtype());

        auto result = Tensor::empty(shape_abc, device_, out_dtype);

        Tensor a_broadcast, b_broadcast, c_broadcast;

        if (shape_ == shape_abc) {
            a_broadcast = clone();
        } else {
            a_broadcast = broadcast_to(shape_abc);
        }

        if (b.shape() == shape_abc) {
            b_broadcast = b.clone();
        } else {
            b_broadcast = b.broadcast_to(shape_abc);
        }

        if (c.shape() == shape_abc) {
            c_broadcast = c.clone();
        } else {
            c_broadcast = c.broadcast_to(shape_abc);
        }

        if (device_ == Device::CUDA) {
            tensor_ops::launch_where(
                a_broadcast.ptr<unsigned char>(),
                b_broadcast.ptr<float>(),
                c_broadcast.ptr<float>(),
                result.ptr<float>(),
                a_broadcast.shape().dims().data(),
                b_broadcast.shape().dims().data(),
                c_broadcast.shape().dims().data(),
                result.shape().dims().data(),
                a_broadcast.shape().rank(),
                b_broadcast.shape().rank(),
                c_broadcast.shape().rank(),
                result.shape().rank(),
                result.numel(),
                0);
            cudaDeviceSynchronize();
        } else {
            // CPU implementation of where operation
            const unsigned char* cond = static_cast<const unsigned char*>(a_broadcast.raw_ptr());
            const float* x = static_cast<const float*>(b_broadcast.raw_ptr());
            const float* y = static_cast<const float*>(c_broadcast.raw_ptr());
            float* dst = static_cast<float*>(result.raw_ptr());

            for (size_t i = 0; i < result.numel(); ++i) {
                dst[i] = cond[i] ? x[i] : y[i];
            }
        }

        return result;
    }

    Tensor Tensor::where(const Tensor& condition, const Tensor& x, const Tensor& y) {
        if (!condition.is_valid() || !x.is_valid() || !y.is_valid()) {
            LOG_ERROR("where: invalid input tensors");
            return Tensor();
        }

        if (condition.dtype() != DataType::Bool) {
            LOG_ERROR("where: condition must be boolean tensor");
            return Tensor();
        }

        // Check device compatibility
        if (condition.device() != x.device() || x.device() != y.device()) {
            LOG_ERROR("where: all tensors must be on the same device");
            return Tensor();
        }
        return condition.ternary(x, y);
    }

    float Tensor::norm(float p) const {
        if (!is_valid())
            return 0.0f;

        if (p == 2.0f) {
            auto squared = this->mul(*this);
            return std::sqrt(squared.sum_scalar());
        } else if (p == 1.0f) {
            return this->abs().sum_scalar();
        } else if (std::isinf(p)) {
            return this->abs().max_scalar();
        } else {
            auto abs_vals = this->abs();
            auto powered = abs_vals.pow(p);
            auto sum = powered.sum_scalar();
            return std::pow(sum, 1.0f / p);
        }
    }

    Tensor Tensor::norm(float p, std::span<const int> dims, bool keepdim) const {
        if (!is_valid()) {
            LOG_ERROR("norm() on invalid tensor");
            return Tensor();
        }

        if (numel() == 0) {
            // Return appropriate empty tensor
            std::vector<size_t> out_shape;
            for (size_t i = 0; i < shape_.rank(); ++i) {
                bool is_reduced = false;
                for (int d : dims) {
                    if (resolve_dim(d) == static_cast<int>(i)) {
                        is_reduced = true;
                        break;
                    }
                }
                if (!is_reduced || keepdim) {
                    out_shape.push_back(is_reduced ? 1 : shape_[i]);
                }
            }
            return empty(TensorShape(out_shape), device_, dtype_);
        }

        // Special cases for common norms
        if (p == 2.0f) {
            // L2 norm: sqrt(sum(x^2))
            auto squared = this->mul(*this);
            auto sum = squared.sum(dims, keepdim);
            return sum.sqrt();
        } else if (p == 1.0f) {
            // L1 norm: sum(|x|)
            return this->abs().sum(dims, keepdim);
        } else if (std::isinf(p)) {
            if (p > 0) {
                // L-infinity norm: max(|x|)
                return this->abs().max(dims, keepdim);
            } else {
                // L-negative-infinity norm: min(|x|)
                return this->abs().min(dims, keepdim);
            }
        } else if (p == 0.0f) {
            // L0 "norm": count of non-zero elements
            // This isn't a true norm, but often used
            auto nonzero_mask = this->ne(0.0f).to(DataType::Float32);
            return nonzero_mask.sum(dims, keepdim);
        } else {
            // General Lp norm: (sum(|x|^p))^(1/p)
            auto abs_vals = this->abs();
            auto powered = abs_vals.pow(p);
            auto sum = powered.sum(dims, keepdim);
            return sum.pow(1.0f / p);
        }
    }

    std::pair<Tensor, Tensor> Tensor::_broadcasted(const Tensor& other, bool match_dtype) const {
        if (!is_valid() || !other.is_valid()) {
            return {Tensor(), Tensor()};
        }

        auto bcast_shape = this->broadcast_shape(other.shape());
        if (bcast_shape.rank() == 0) {
            LOG_ERROR("Incompatible shapes for broadcasting");
            return {Tensor(), Tensor()};
        }

        Tensor a_broadcast = (shape_ == bcast_shape) ? this->clone() : broadcast_to(bcast_shape);
        Tensor b_broadcast = (other.shape() == bcast_shape) ? other.clone() : other.broadcast_to(bcast_shape);

        if (match_dtype && dtype_ != other.dtype()) {
            auto common_dtype = promote_types(dtype_, other.dtype());
            if (a_broadcast.dtype() != common_dtype) {
                a_broadcast = a_broadcast.to(common_dtype);
            }
            if (b_broadcast.dtype() != common_dtype) {
                b_broadcast = b_broadcast.to(common_dtype);
            }
        }

        return {std::move(a_broadcast), std::move(b_broadcast)};
    }

    // ============= STATIC CAT OPERATION =============

    Tensor Tensor::cat(const std::vector<Tensor>& tensors, int dim) {
        if (tensors.empty()) {
            LOG_ERROR("Cannot concatenate empty vector of tensors");
            return Tensor();
        }

        if (tensors.size() == 1) {
            return tensors[0].clone();
        }

        int resolved_dim = dim;
        if (resolved_dim < 0) {
            resolved_dim = tensors[0].shape().rank() + resolved_dim;
        }

        if (resolved_dim < 0 || resolved_dim >= static_cast<int>(tensors[0].shape().rank())) {
            LOG_ERROR("Invalid dimension for cat: {}", dim);
            return Tensor();
        }

        const auto& first_shape = tensors[0].shape();
        const auto first_device = tensors[0].device();
        const auto first_dtype = tensors[0].dtype();

        size_t total_size_along_dim = first_shape[resolved_dim];

        // Validate all tensors
        for (size_t i = 1; i < tensors.size(); ++i) {
            const auto& shape = tensors[i].shape();

            if (shape.rank() != first_shape.rank()) {
                LOG_ERROR("All tensors must have the same number of dimensions");
                return Tensor();
            }

            for (size_t d = 0; d < shape.rank(); ++d) {
                if (d != static_cast<size_t>(resolved_dim) && shape[d] != first_shape[d]) {
                    LOG_ERROR("All dimensions except dim={} must match", dim);
                    return Tensor();
                }
            }

            if (tensors[i].device() != first_device) {
                LOG_ERROR("All tensors must be on the same device");
                return Tensor();
            }

            if (tensors[i].dtype() != first_dtype) {
                LOG_ERROR("All tensors must have the same dtype");
                return Tensor();
            }

            total_size_along_dim += shape[resolved_dim];
        }

        // Build result shape
        std::vector<size_t> result_dims = first_shape.dims();
        result_dims[resolved_dim] = total_size_along_dim;
        auto result = Tensor::empty(TensorShape(result_dims), first_device, first_dtype);

        size_t element_size = dtype_size(first_dtype);

        // ============= OPTIMIZED PATH: First dimension =============
        if (resolved_dim == 0) {
            // Concatenating along first dimension - completely contiguous
            if (first_device == Device::CUDA) {
                size_t offset = 0;
                for (const auto& t : tensors) {
                    size_t bytes = t.bytes();
                    cudaMemcpy(
                        static_cast<char*>(result.raw_ptr()) + offset,
                        t.raw_ptr(),
                        bytes,
                        cudaMemcpyDeviceToDevice);
                    offset += bytes;
                }
            } else {
                size_t offset = 0;
                for (const auto& t : tensors) {
                    size_t bytes = t.bytes();
                    std::memcpy(
                        static_cast<char*>(result.raw_ptr()) + offset,
                        t.raw_ptr(),
                        bytes);
                    offset += bytes;
                }
            }

            return result;
        }

        // ============= OPTIMIZED PATH: Last dimension (most common case) =============
        if (resolved_dim == static_cast<int>(first_shape.rank()) - 1) {
            // Concatenating along last dimension - can do bulk copies per "row"
            size_t row_size = total_size_along_dim;
            size_t num_rows = 1;
            for (int i = 0; i < resolved_dim; ++i) {
                num_rows *= first_shape[i];
            }

            if (first_device == Device::CUDA) {
                tensor_ops::launch_cat_last_dim(
                    result.raw_ptr(),
                    tensors,
                    num_rows,
                    row_size,
                    element_size,
                    nullptr);
                cudaDeviceSynchronize();
            } else {
                // CPU: Simple memcpy per row
                size_t result_offset = 0;
                for (const auto& t : tensors) {
                    size_t tensor_dim_size = t.shape()[resolved_dim];

                    for (size_t row = 0; row < num_rows; ++row) {
                        const void* src = static_cast<const char*>(t.raw_ptr()) +
                                          row * tensor_dim_size * element_size;
                        void* dst = static_cast<char*>(result.raw_ptr()) +
                                    row * row_size * element_size + result_offset * element_size;

                        std::memcpy(dst, src, tensor_dim_size * element_size);
                    }

                    result_offset += tensor_dim_size;
                }
            }

            return result;
        }

        // ============= GENERAL PATH: Middle dimensions =============
        size_t outer_size = 1;
        for (int i = 0; i < resolved_dim; ++i) {
            outer_size *= first_shape[i];
        }

        size_t inner_size = 1;
        for (size_t i = resolved_dim + 1; i < first_shape.rank(); ++i) {
            inner_size *= first_shape[i];
        }

        if (first_device == Device::CUDA) {
            tensor_ops::launch_cat_middle_dim(
                result.raw_ptr(),
                tensors,
                outer_size,
                inner_size,
                resolved_dim,
                element_size,
                nullptr);
            cudaDeviceSynchronize();
        } else {
            // CPU fallback
            for (size_t outer = 0; outer < outer_size; ++outer) {
                size_t result_offset = 0;

                for (const auto& t : tensors) {
                    size_t tensor_dim_size = t.shape()[resolved_dim];
                    size_t copy_size = tensor_dim_size * inner_size * element_size;

                    const void* src = static_cast<const char*>(t.raw_ptr()) +
                                      outer * tensor_dim_size * inner_size * element_size;
                    void* dst = static_cast<char*>(result.raw_ptr()) +
                                (outer * total_size_along_dim * inner_size + result_offset) * element_size;

                    std::memcpy(dst, src, copy_size);
                    result_offset += tensor_dim_size * inner_size;
                }
            }
        }

        return result;
    }

    // ============= STATIC STACK OPERATION =============

    Tensor Tensor::stack(const std::vector<Tensor>& tensors, int dim) {
        if (tensors.empty()) {
            LOG_ERROR("Cannot stack empty vector of tensors");
            return Tensor();
        }

        const auto& first_shape = tensors[0].shape();
        const auto first_device = tensors[0].device();
        const auto first_dtype = tensors[0].dtype();

        // Validate all tensors have same shape, device, and dtype
        for (size_t i = 1; i < tensors.size(); ++i) {
            if (tensors[i].shape() != first_shape) {
                LOG_ERROR("All tensors must have the same shape for stack");
                return Tensor();
            }
            if (tensors[i].device() != first_device) {
                LOG_ERROR("All tensors must be on the same device");
                return Tensor();
            }
            if (tensors[i].dtype() != first_dtype) {
                LOG_ERROR("All tensors must have the same dtype");
                return Tensor();
            }
        }

        // Build output shape with new dimension inserted at 'dim'
        std::vector<size_t> new_dims = first_shape.dims();

        // Handle negative dimension
        if (dim < 0) {
            dim = first_shape.rank() + dim + 1;
        }

        if (dim < 0 || dim > static_cast<int>(first_shape.rank())) {
            LOG_ERROR("Invalid dimension for stack: {}", dim);
            return Tensor();
        }

        // Insert new dimension of size tensors.size() at position 'dim'
        new_dims.insert(new_dims.begin() + dim, tensors.size());

        auto result = Tensor::empty(TensorShape(new_dims), first_device, first_dtype);

        size_t elements_per_tensor = first_shape.elements();
        size_t bytes_per_tensor = elements_per_tensor * dtype_size(first_dtype);

        // Compute strides for the output tensor
        auto result_strides = result.shape().strides();

        // Size of one "slice" along the stacked dimension
        size_t stride_at_dim = result_strides[dim];

        // For each input tensor, we need to copy it to the right location in output
        if (first_device == Device::CUDA) {
            for (size_t i = 0; i < tensors.size(); ++i) {
                if (dim == 0) {
                    // Contiguous copy for dim=0 case (optimized path)
                    void* dst = static_cast<char*>(result.raw_ptr()) + i * bytes_per_tensor;
                    cudaMemcpy(dst, tensors[i].raw_ptr(), bytes_per_tensor,
                               cudaMemcpyDeviceToDevice);
                } else {
                    // For non-zero dimensions, we need to scatter the data properly
                    size_t outer_size = 1;
                    for (int d = 0; d < dim; ++d) {
                        outer_size *= result.shape()[d];
                    }

                    size_t inner_size = elements_per_tensor / outer_size;

                    // Copy each outer slice
                    for (size_t outer = 0; outer < outer_size; ++outer) {
                        const void* src_ptr = static_cast<const char*>(tensors[i].raw_ptr()) +
                                              outer * inner_size * dtype_size(first_dtype);
                        void* dst_ptr = static_cast<char*>(result.raw_ptr()) +
                                        (outer * result_strides[dim == 0 ? 1 : 0] +
                                         i * stride_at_dim) *
                                            dtype_size(first_dtype);

                        cudaMemcpy(dst_ptr, src_ptr, inner_size * dtype_size(first_dtype),
                                   cudaMemcpyDeviceToDevice);
                    }
                }
            }
        } else {
            // CPU implementation
            for (size_t i = 0; i < tensors.size(); ++i) {
                if (dim == 0) {
                    // Contiguous copy for dim=0 case
                    void* dst = static_cast<char*>(result.raw_ptr()) + i * bytes_per_tensor;
                    std::memcpy(dst, tensors[i].raw_ptr(), bytes_per_tensor);
                } else {
                    // For non-zero dimensions, scatter properly
                    size_t outer_size = 1;
                    for (int d = 0; d < dim; ++d) {
                        outer_size *= result.shape()[d];
                    }

                    size_t inner_size = elements_per_tensor / outer_size;

                    for (size_t outer = 0; outer < outer_size; ++outer) {
                        const void* src_ptr = static_cast<const char*>(tensors[i].raw_ptr()) +
                                              outer * inner_size * dtype_size(first_dtype);
                        void* dst_ptr = static_cast<char*>(result.raw_ptr()) +
                                        (outer * result_strides[dim == 0 ? 1 : 0] +
                                         i * stride_at_dim) *
                                            dtype_size(first_dtype);

                        std::memcpy(dst_ptr, src_ptr, inner_size * dtype_size(first_dtype));
                    }
                }
            }
        }

        return result;
    }

    // ============= OPTIMIZED CLAMP (FUSED VERSION) =============

    Tensor Tensor::clamp(float min_val, float max_val) const {
        if (!is_valid()) {
            LOG_ERROR("clamp() on invalid tensor");
            return Tensor();
        }

        if (numel() == 0) {
            return empty(shape_, device_, dtype_);
        }

        // FUSED VERSION: Allocate output + clamp in one pass (avoids separate clone)
        auto result = empty(shape_, device_, dtype_);

        if (device_ == Device::CUDA) {
            if (dtype_ == DataType::Float32) {
                // Single-pass: read from source, write clamped to destination
                const float* src = ptr<float>();
                float* dst = result.ptr<float>();

                // Use our optimized kernel
                tensor_ops::launch_clamp_fused(src, dst, min_val, max_val, numel(), 0);
            } else if (dtype_ == DataType::Int32) {
                // Fallback: copy then clamp for int
                cudaMemcpy(result.data_, data_, bytes(), cudaMemcpyDeviceToDevice);
                tensor_ops::launch_clamp_scalar_int(result.ptr<int>(),
                                                    static_cast<int>(min_val),
                                                    static_cast<int>(max_val),
                                                    numel(), 0);
            }
        } else {
            // CPU: simple loop
            if (dtype_ == DataType::Float32) {
                const float* src = ptr<float>();
                float* dst = result.ptr<float>();
                for (size_t i = 0; i < numel(); ++i) {
                    dst[i] = std::isnan(src[i]) ? src[i] : std::clamp(src[i], min_val, max_val);
                }
            } else if (dtype_ == DataType::Int32) {
                const int* src = ptr<int>();
                int* dst = result.ptr<int>();
                int min_int = static_cast<int>(min_val);
                int max_int = static_cast<int>(max_val);
                for (size_t i = 0; i < numel(); ++i) {
                    dst[i] = std::clamp(src[i], min_int, max_int);
                }
            }
        }

        return result;
    }

} // namespace lfs::core
