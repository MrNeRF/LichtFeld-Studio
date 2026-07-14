/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda_error.hpp"
#include "core/logger.hpp"
#include "core/pinned_memory_allocator.hpp"
#include "core/tensor_trace.hpp"
#include "internal/cuda_stream_context.hpp"
#include "internal/lazy_config.hpp"
#include "internal/lazy_executor.hpp"
#include "internal/lazy_ir.hpp"
#include "internal/memory_pool.hpp"
#include "internal/tensor_broadcast.hpp"
#include "internal/tensor_functors.hpp"
#include "internal/tensor_impl.hpp"
#include "internal/tensor_ops.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <curand.h>
#include <format>
#include <numeric>
#include <optional>
#include <string_view>

namespace lfs::core {

    namespace {

        cudaStream_t resolve_cuda_execution_stream(const Tensor& tensor) {
            cudaStream_t execution_stream = getCurrentCUDAStream();
            if (execution_stream == nullptr && tensor.device() == Device::CUDA) {
                execution_stream = tensor.stream();
            }
            return execution_stream;
        }

        Tensor empty_on_tensor_stream(const TensorShape& shape, Device device, DataType dtype, const Tensor& tensor) {
            if (device != Device::CUDA) {
                return Tensor::empty(shape, device, dtype);
            }

            const cudaStream_t execution_stream = resolve_cuda_execution_stream(tensor);
            tensor.sync_to_stream(execution_stream);
            CUDAStreamGuard guard(execution_stream);
            return Tensor::empty(shape, device, dtype);
        }

    } // namespace

    // ============= CORE UNIFIED OPERATIONS =============
    constexpr static int promotion_rank(DataType dtype) {
        switch (dtype) {
        case DataType::Bool: return 0;
        case DataType::UInt8: return 1;
        case DataType::Int32: return 2;
        case DataType::Int64: return 3;
        case DataType::Float16: return 4;
        case DataType::Float32: return 5;
        }
        return -1;
    }

    constexpr static DataType promote_types(DataType a, DataType b) {
        return promotion_rank(a) >= promotion_rank(b) ? a : b;
    }

    Tensor Tensor::load(LoadOp op, const LoadArgs& args) {
        Tensor result;
        LFS_ASSERT_MSG(args.device == Device::CPU || args.device == Device::CUDA,
                       "tensor load received an invalid device");
        LFS_ASSERT_MSG(dtype_size(args.dtype) != 0,
                       "tensor load received an invalid dtype");
        LFS_ASSERT_MSG(static_cast<int>(op) >= static_cast<int>(LoadOp::Empty) &&
                           static_cast<int>(op) <= static_cast<int>(LoadOp::Multinomial),
                       "tensor load received an unknown operation");

        switch (op) {
        case LoadOp::Empty: {
            result.shape_ = args.shape;
            result.strides_ = args.shape.strides();
            result.storage_offset_ = 0;
            result.is_contiguous_ = true;
            result.device_ = args.device;
            result.dtype_ = args.dtype;
            result.id_ = next_id_++;
            result.state_->stream = getCurrentCUDAStream();

            LFS_ASSERT_MSG(result.shape_.elements() == 0 ||
                               dtype_size(result.dtype_) <=
                                   std::numeric_limits<size_t>::max() / result.shape_.elements(),
                           "tensor allocation byte count overflow");
            size_t bytes = result.shape_.elements() * dtype_size(result.dtype_);
            internal::telemetry_record_materialization(bytes);

            if (bytes == 0) {
                if (result.device_ == Device::CUDA) {
                    cudaStream_t s = result.stream();
                    void* dummy = allocate_cuda_storage(1, s);
                    LFS_ASSERT_MSG(dummy != nullptr,
                                   "failed to allocate CUDA sentinel storage for an empty tensor");
                    result.data_owner_ = std::shared_ptr<void>(dummy, [s](void* p) {
                        CudaMemoryPool::instance().deallocate(p, s);
                    });
                } else {
                    void* dummy = nullptr;
                    if (args.use_pinned) {
                        dummy = PinnedMemoryAllocator::instance().allocate(1);
                        LFS_ASSERT_MSG(dummy != nullptr,
                                       "failed to allocate pinned sentinel storage for an empty tensor");
                        cudaStream_t s = result.stream();
                        result.data_owner_ = std::shared_ptr<void>(dummy, [s](void* p) {
                            if (p)
                                PinnedMemoryAllocator::instance().deallocate(p, s);
                        });
                    } else {
                        dummy = std::malloc(1);
                        LFS_ASSERT_MSG(dummy != nullptr,
                                       "failed to allocate sentinel storage for an empty tensor");
                        result.data_owner_ = std::shared_ptr<void>(dummy, [](void* p) {
                            std::free(p);
                        });
                    }
                }
                result.data_ = nullptr;
                return result;
            }

            if (result.device_ == Device::CUDA) {
                cudaStream_t s = result.stream();
                void* ptr = allocate_cuda_storage(bytes, s);
                result.data_owner_ = std::shared_ptr<void>(ptr, [s](void* p) {
                    CudaMemoryPool::instance().deallocate(p, s);
                });
                result.data_ = result.data_owner_.get();
                result.compute_alignment(); // Compute alignment flags once

                // Record tensor allocation for profiling
                CudaMemoryPool::instance().record_tensor(
                    result.data_,
                    result.shape().dims(),
                    bytes,
                    dtype_name(result.dtype_));
            } else {
                // CPU tensor allocation - choose between pinned and regular memory
                void* ptr = nullptr;

                if (args.use_pinned) {
                    ptr = PinnedMemoryAllocator::instance().allocate(bytes);
                    if (!ptr) {
                        throw std::runtime_error(std::format(
                            "Out of memory: failed to allocate {} bytes ({:.2f} GB) of pinned memory.",
                            bytes, bytes / (1024.0 * 1024.0 * 1024.0)));
                    }
                    cudaStream_t s = result.stream();
                    result.data_owner_ = std::shared_ptr<void>(ptr, [s](void* p) {
                        if (p)
                            PinnedMemoryAllocator::instance().deallocate(p, s);
                    });
                } else {
                    // Use regular malloc for CPU memory
                    ptr = std::malloc(bytes);
                    if (!ptr) {
                        throw std::runtime_error(std::format(
                            "Out of memory: failed to allocate {} bytes ({:.2f} GB) of CPU memory.",
                            bytes, bytes / (1024.0 * 1024.0 * 1024.0)));
                    }
                    result.data_owner_ = std::shared_ptr<void>(ptr, [](void* p) {
                        std::free(p);
                    });
                }

                result.data_ = result.data_owner_.get();
                result.compute_alignment(); // Compute alignment flags once
            }
            break;
        }

        case LoadOp::Const: {
            LFS_ASSERT_MSG(std::holds_alternative<float>(args.args),
                           "constant tensor load requires a float value");
            float value = std::get<float>(args.args);
            const bool floating_dtype =
                args.dtype == DataType::Float32 || args.dtype == DataType::Float16;
            LFS_ASSERT_MSG(floating_dtype || std::isfinite(value),
                           "non-finite full/constant values require a floating-point dtype");
            LFS_ASSERT_MSG(args.dtype != DataType::Float16 ||
                               !std::isfinite(value) || std::abs(value) <= 65504.0f,
                           "Float16 constant is outside the finite representable range");
            LFS_ASSERT_MSG(args.dtype != DataType::Int32 ||
                               (value >= static_cast<float>(std::numeric_limits<int32_t>::lowest()) &&
                                value <= std::nextafter(
                                             static_cast<float>(std::numeric_limits<int32_t>::max()),
                                             0.0f)),
                           "Int32 constant is outside the representable range");
            LFS_ASSERT_MSG(args.dtype != DataType::Int64 ||
                               (value >= static_cast<float>(std::numeric_limits<int64_t>::lowest()) &&
                                value <= std::nextafter(
                                             static_cast<float>(std::numeric_limits<int64_t>::max()),
                                             0.0f)),
                           "Int64 constant is outside the representable range");
            LFS_ASSERT_MSG(args.dtype != DataType::UInt8 ||
                               (value >= 0.0f && value <= 255.0f),
                           "UInt8 constant must be in [0, 255]");
            result = load(LoadOp::Empty, args);
            if (!result.is_valid() || result.numel() == 0)
                return result;

            if (result.device_ == Device::CUDA) {
                if (result.dtype_ == DataType::Float32) {
                    if (value == 0.0f) {
                        LFS_CUDA_CHECK_MSG(cudaMemset(result.data_, 0, result.bytes()),
                                           "constant Float32 CUDA memset");
                    } else {
                        tensor_ops::launch_load_op(
                            result.data_,
                            result.shape_.dims().data(),
                            result.shape_.rank(),
                            LoadOp::Const,
                            &value,
                            result.dtype_,
                            result.stream());
                        // No sync - tensor operation
                    }
                } else if (result.dtype_ == DataType::Float16) {
                    if (value == 0.0f) {
                        LFS_CUDA_CHECK_MSG(cudaMemset(result.data_, 0, result.bytes()),
                                           "constant Float16 CUDA memset");
                    } else {
                        // Create Float16 values on CPU, then copy to GPU
                        std::vector<__half> temp(result.numel(), __float2half(value));
                        LFS_CUDA_CHECK_MSG(
                            cudaMemcpy(result.data_, temp.data(), result.bytes(),
                                       cudaMemcpyHostToDevice),
                            "constant Float16 CUDA copy");
                    }
                } else if (result.dtype_ == DataType::Bool) {
                    unsigned char fill_val = (value != 0.0f) ? 1 : 0;
                    LFS_CUDA_CHECK_MSG(cudaMemset(result.data_, fill_val, result.bytes()),
                                       "constant Bool CUDA memset");
                } else if (result.dtype_ == DataType::Int32) {
                    if (value == 0.0f) {
                        LFS_CUDA_CHECK_MSG(cudaMemset(result.data_, 0, result.bytes()),
                                           "constant Int32 CUDA memset");
                    } else {
                        std::vector<int> temp(result.numel(), static_cast<int>(value));
                        LFS_CUDA_CHECK_MSG(
                            cudaMemcpy(result.data_, temp.data(), result.bytes(),
                                       cudaMemcpyHostToDevice),
                            "constant Int32 CUDA copy");
                    }
                } else if (result.dtype_ == DataType::Int64) {
                    if (value == 0.0f) {
                        LFS_CUDA_CHECK_MSG(cudaMemset(result.data_, 0, result.bytes()),
                                           "constant Int64 CUDA memset");
                    } else {
                        std::vector<int64_t> temp(result.numel(), static_cast<int64_t>(value));
                        LFS_CUDA_CHECK_MSG(
                            cudaMemcpy(result.data_, temp.data(), result.bytes(),
                                       cudaMemcpyHostToDevice),
                            "constant Int64 CUDA copy");
                    }
                } else if (result.dtype_ == DataType::UInt8) {
                    const uint8_t fill_val = static_cast<uint8_t>(std::clamp(value, 0.0f, 255.0f));
                    LFS_CUDA_CHECK_MSG(cudaMemset(result.data_, fill_val, result.bytes()),
                                       "constant UInt8 CUDA memset");
                }
            } else {
                if (result.dtype_ == DataType::Float32) {
                    float* ptr = static_cast<float*>(result.data_);
                    std::fill_n(ptr, result.numel(), value);
                } else if (result.dtype_ == DataType::Float16) {
                    __half* ptr = static_cast<__half*>(result.data_);
                    std::fill_n(ptr, result.numel(), __float2half(value));
                } else if (result.dtype_ == DataType::Bool) {
                    unsigned char* ptr = static_cast<unsigned char*>(result.data_);
                    std::fill_n(ptr, result.numel(), value != 0 ? 1 : 0);
                } else if (result.dtype_ == DataType::Int32) {
                    int* ptr = static_cast<int*>(result.data_);
                    std::fill_n(ptr, result.numel(), static_cast<int>(value));
                } else if (result.dtype_ == DataType::Int64) {
                    int64_t* ptr = static_cast<int64_t*>(result.data_);
                    std::fill_n(ptr, result.numel(), static_cast<int64_t>(value));
                } else if (result.dtype_ == DataType::UInt8) {
                    uint8_t* ptr = static_cast<uint8_t*>(result.data_);
                    std::fill_n(ptr, result.numel(), static_cast<uint8_t>(std::clamp(value, 0.0f, 255.0f)));
                }
            }
            break;
        }

        case LoadOp::Arange: {
            LFS_ASSERT_MSG((std::holds_alternative<std::tuple<float, float, float>>(args.args)),
                           "arange load requires start, end, and step values");
            auto [start, end, step] = std::get<std::tuple<float, float, float>>(args.args);
            LFS_ASSERT_MSG(args.dtype == DataType::Float32 || args.dtype == DataType::Int32,
                           "arange supports only Float32 and Int32");
            LFS_ASSERT_MSG(std::isfinite(start) && std::isfinite(end) && std::isfinite(step),
                           "arange bounds and step must be finite");
            LFS_ASSERT_MSG(step != 0.0f,
                           "arange step cannot be zero");
            const long double extent = static_cast<long double>(end) -
                                       static_cast<long double>(start);
            LFS_ASSERT_MSG(extent * static_cast<long double>(step) >= 0.0L,
                           "arange step points away from the end value");

            const long double count_value =
                std::ceil(extent / static_cast<long double>(step));
            LFS_ASSERT_MSG(std::isfinite(count_value) && count_value >= 0.0L &&
                               count_value <=
                                   static_cast<long double>(std::numeric_limits<size_t>::max()),
                           "arange element count is not representable");
            const size_t count = static_cast<size_t>(count_value);
            LFS_ASSERT_MSG(count == 0 ||
                               dtype_size(args.dtype) <=
                                   std::numeric_limits<size_t>::max() / count,
                           "arange allocation byte count overflow");
            if (args.dtype == DataType::Int32 && count > 0) {
                const double last = static_cast<double>(start) +
                                    static_cast<double>(count - 1) * static_cast<double>(step);
                const double minimum = std::min(static_cast<double>(start), last);
                const double maximum = std::max(static_cast<double>(start), last);
                LFS_ASSERT_MSG(minimum >= static_cast<double>(std::numeric_limits<int32_t>::lowest()) &&
                                   maximum <= static_cast<double>(std::numeric_limits<int32_t>::max()),
                               "Int32 arange values are outside the representable range");
            }
            if (count == 0) {
                LoadArgs empty_args = args;
                empty_args.shape = TensorShape{0};
                empty_args.args = std::monostate{};
                return load(LoadOp::Empty, empty_args);
            }

            result.shape_ = TensorShape{count};
            result.strides_ = result.shape_.strides(); // Initialize to contiguous strides
            result.storage_offset_ = 0;
            result.is_contiguous_ = true;
            result.device_ = args.device;
            result.dtype_ = args.dtype;
            result.id_ = next_id_++;
            result.state_->stream = getCurrentCUDAStream();

            size_t bytes = count * dtype_size(result.dtype_);

            if (result.device_ == Device::CUDA) {
                cudaStream_t s = result.stream();
                void* ptr = allocate_cuda_storage(bytes, s);
                result.data_owner_ = std::shared_ptr<void>(ptr, [s](void* p) {
                    CudaMemoryPool::instance().deallocate(p, s);
                });
                result.data_ = result.data_owner_.get();

                CudaMemoryPool::instance().record_tensor(
                    result.data_,
                    result.shape().dims(),
                    bytes,
                    dtype_name(result.dtype_));

                if (result.dtype_ == DataType::Float32) {
                    std::vector<float> data(count);
                    for (size_t i = 0; i < count; ++i) {
                        data[i] = start + i * step;
                    }
                    LFS_CUDA_CHECK_MSG(
                        cudaMemcpy(result.data_, data.data(), bytes, cudaMemcpyHostToDevice),
                        "Float32 arange CUDA copy");
                } else if (result.dtype_ == DataType::Int32) {
                    std::vector<int> data(count);
                    for (size_t i = 0; i < count; ++i) {
                        data[i] = static_cast<int>(start + i * step);
                    }
                    LFS_CUDA_CHECK_MSG(
                        cudaMemcpy(result.data_, data.data(), bytes, cudaMemcpyHostToDevice),
                        "Int32 arange CUDA copy");
                }
            } else {
                void* ptr = PinnedMemoryAllocator::instance().allocate(bytes);
                if (!ptr) {
                    throw std::runtime_error(std::format(
                        "Out of memory: failed to allocate {} bytes ({:.2f} GB) of pinned memory.",
                        bytes, bytes / (1024.0 * 1024.0 * 1024.0)));
                }
                cudaStream_t s = result.stream();
                result.data_owner_ = std::shared_ptr<void>(ptr, [s](void* p) {
                    if (p)
                        PinnedMemoryAllocator::instance().deallocate(p, s);
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
            LFS_ASSERT_MSG((std::holds_alternative<std::pair<float, float>>(args.args)),
                           "uniform/rand load requires lower and upper bounds");
            auto [low, high] = std::get<std::pair<float, float>>(args.args);
            LFS_ASSERT_MSG(args.dtype == DataType::Float32 || args.dtype == DataType::Int32,
                           "uniform/rand supports only Float32 and Int32");
            LFS_ASSERT_MSG(std::isfinite(low) && std::isfinite(high) && low < high,
                           "uniform/rand bounds must be finite and low < high");
            LFS_ASSERT_MSG(args.dtype != DataType::Int32 ||
                               (low >= static_cast<float>(std::numeric_limits<int32_t>::lowest()) &&
                                high <= std::nextafter(
                                            static_cast<float>(std::numeric_limits<int32_t>::max()),
                                            0.0f)),
                           "Int32 uniform/rand bounds are outside the representable range");
            result = load(LoadOp::Empty, args);
            if (!result.is_valid() || result.numel() == 0)
                return result;

            if (result.device_ == Device::CUDA) {
                const cudaStream_t stream = result.stream();
                if (result.dtype_ == DataType::Float32) {
                    tensor_ops::launch_uniform(result.ptr<float>(), result.numel(), low, high,
                                               RandomGenerator::instance().get_next_cuda_seed(), stream);
                    // No sync - tensor operation
                } else if (result.dtype_ == DataType::Int32) {
                    tensor_ops::launch_randint(result.ptr<int>(), result.numel(),
                                               static_cast<int>(low), static_cast<int>(high),
                                               RandomGenerator::instance().get_next_cuda_seed(), stream);
                    // No sync - tensor operation
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
            LFS_ASSERT_MSG((std::holds_alternative<std::pair<float, float>>(args.args)),
                           "normal/randn load requires mean and standard deviation");
            auto [mean, std] = std::get<std::pair<float, float>>(args.args);
            LFS_ASSERT_MSG(args.dtype == DataType::Float32,
                           "normal/randn supports only Float32");
            LFS_ASSERT_MSG(std::isfinite(mean) && std::isfinite(std) && std > 0.0f,
                           "normal/randn requires finite mean and std > 0");
            result = load(LoadOp::Empty, args);
            if (!result.is_valid() || result.numel() == 0)
                return result;

            if (result.device_ == Device::CUDA) {
                // Use Philox RNG without resetting offset (stateful like PyTorch - much faster!)
                curandGenerator_t* gen = static_cast<curandGenerator_t*>(
                    RandomGenerator::instance().get_generator(Device::CUDA));

                size_t n = result.numel();
                if (n % 2 == 1) {
                    auto scratch = Tensor::empty({n + 1}, Device::CUDA, DataType::Float32);
                    const auto status = curandGenerateNormal(*gen, scratch.ptr<float>(), n + 1, mean, std);
                    LFS_ASSERT_MSG(status == CURAND_STATUS_SUCCESS,
                                   "normal/randn cuRAND generation failed");
                    LFS_CUDA_CHECK_MSG(
                        cudaMemcpy(result.ptr<float>(), scratch.ptr<float>(), n * sizeof(float),
                                   cudaMemcpyDeviceToDevice),
                        "normal/randn scratch copy (bytes={}, requested_count={})",
                        n * sizeof(float), n);
                } else {
                    const auto status = curandGenerateNormal(*gen, result.ptr<float>(), n, mean, std);
                    LFS_ASSERT_MSG(status == CURAND_STATUS_SUCCESS,
                                   "normal/randn cuRAND generation failed");
                }
                // curandGenerateNormal is blocking, no need for explicit sync
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
            LFS_ASSERT_MSG((std::holds_alternative<std::pair<int, int>>(args.args)),
                           "randint load requires integer lower and upper bounds");
            auto [low, high] = std::get<std::pair<int, int>>(args.args);
            LFS_ASSERT_MSG(args.dtype == DataType::Int32 || args.dtype == DataType::Float32 ||
                               args.dtype == DataType::UInt8,
                           "randint encountered an unsupported dtype");
            LFS_ASSERT_MSG(low < high,
                           "randint requires low < high");
            LFS_ASSERT_MSG(args.dtype != DataType::UInt8 || (low >= 0 && high <= 256),
                           "UInt8 randint bounds must be within [0, 256]");
            result = load(LoadOp::Empty, args);
            if (!result.is_valid() || result.numel() == 0)
                return result;

            if (result.device_ == Device::CUDA) {
                const cudaStream_t stream = result.stream();
                if (result.dtype_ == DataType::Int32) {
                    tensor_ops::launch_randint(result.ptr<int>(), result.numel(), low, high,
                                               RandomGenerator::instance().get_next_cuda_seed(), stream);
                    // No sync - tensor operation
                } else if (result.dtype_ == DataType::Float32) {
                    int* temp_buffer = static_cast<int*>(
                        CudaMemoryPool::instance().allocate(result.numel() * sizeof(int), stream));
                    LFS_ASSERT_MSG(temp_buffer != nullptr,
                                   "randint failed to allocate its Float32 CUDA conversion buffer");
                    tensor_ops::launch_randint(temp_buffer, result.numel(), low, high,
                                               RandomGenerator::instance().get_next_cuda_seed(), stream);

                    tensor_ops::launch_convert_type<int, float>(temp_buffer, result.ptr<float>(),
                                                                result.numel(), stream);
                    // No sync - tensor operation

                    CudaMemoryPool::instance().deallocate(temp_buffer, stream);
                } else if (result.dtype_ == DataType::UInt8) {
                    int* temp_buffer = static_cast<int*>(
                        CudaMemoryPool::instance().allocate(result.numel() * sizeof(int), stream));
                    LFS_ASSERT_MSG(temp_buffer != nullptr,
                                   "randint failed to allocate its UInt8 CUDA conversion buffer");
                    tensor_ops::launch_randint(temp_buffer, result.numel(), low, high,
                                               RandomGenerator::instance().get_next_cuda_seed(), stream);

                    tensor_ops::launch_convert_type<int, uint8_t>(temp_buffer, result.ptr<uint8_t>(),
                                                                  result.numel(), stream);
                    // No sync - tensor operation

                    CudaMemoryPool::instance().deallocate(temp_buffer, stream);
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
                } else if (result.dtype_ == DataType::UInt8) {
                    uint8_t* data = result.ptr<uint8_t>();
                    for (size_t i = 0; i < result.numel(); ++i) {
                        data[i] = static_cast<uint8_t>(dist(gen));
                    }
                }
            }
            break;
        }

        case LoadOp::Bernoulli: {
            LFS_ASSERT_MSG(std::holds_alternative<float>(args.args),
                           "bernoulli load requires a probability");
            float p = std::get<float>(args.args);
            LFS_ASSERT_MSG(args.dtype == DataType::Float32,
                           "bernoulli currently supports only Float32");
            LFS_ASSERT_MSG(std::isfinite(p) && p >= 0.0f && p <= 1.0f,
                           "bernoulli probability must be in [0, 1]");
            result = load(LoadOp::Empty, args);
            if (!result.is_valid() || result.numel() == 0)
                return result;

            if (result.device_ == Device::CUDA) {
                tensor_ops::launch_bernoulli(result.ptr<float>(), result.numel(), p,
                                             RandomGenerator::instance().get_next_cuda_seed(), result.stream());
                // No sync - tensor operation
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
            LFS_ASSERT_MSG((std::holds_alternative<std::pair<void*, bool>>(args.args)),
                           "multinomial load requires weights and replacement mode");
            auto [weights_ptr, replacement] = std::get<std::pair<void*, bool>>(args.args);
            const Tensor* weights = static_cast<const Tensor*>(weights_ptr);

            LFS_ASSERT_MSG(weights != nullptr && weights->is_valid() && weights->ndim() == 1,
                           "multinomial requires a valid rank-1 weight tensor");
            LFS_ASSERT_MSG(weights->dtype() == DataType::Float32,
                           "multinomial weights must be Float32");
            LFS_ASSERT_MSG(args.device == weights->device(),
                           "multinomial output and weights must use the same device");
            LFS_ASSERT_MSG(args.dtype == DataType::Int64,
                           "multinomial output must use Int64 dtype");

            size_t n = weights->numel();
            size_t num_samples = args.shape.elements();
            LFS_ASSERT_MSG(n > 0 && num_samples > 0,
                           "multinomial requires non-empty weights and output");
            LFS_ASSERT_MSG(replacement || num_samples <= n,
                           "multinomial sample count exceeds weights without replacement");

            result = load(LoadOp::Empty, args);
            if (!result.is_valid())
                return result;

            if (weights->device() == Device::CUDA) {
                tensor_ops::launch_multinomial(weights->ptr<float>(), result.ptr<int64_t>(),
                                               n, num_samples, replacement,
                                               RandomGenerator::instance().get_next_cuda_seed(), result.stream());
                // No sync - tensor operation
            } else {
                auto weights_data = weights->to_vector();

                float sum = std::accumulate(weights_data.begin(), weights_data.end(), 0.0f);
                LFS_ASSERT_MSG(std::isfinite(sum) && sum > 0.0f,
                               "multinomial weights must have a positive finite sum");

                std::vector<float> cdf(n);
                cdf[0] = weights_data[0] / sum;
                for (size_t i = 1; i < n; ++i) {
                    cdf[i] = cdf[i - 1] + weights_data[i] / sum;
                }

                auto& gen = *static_cast<std::mt19937_64*>(
                    RandomGenerator::instance().get_generator(Device::CPU));
                std::uniform_real_distribution<float> dis(0.0f, 1.0f);

                int64_t* samples = result.ptr<int64_t>();

                if (replacement) {
                    for (size_t i = 0; i < num_samples; ++i) {
                        float u = dis(gen);
                        auto it = std::lower_bound(cdf.begin(), cdf.end(), u);
                        samples[i] = static_cast<int64_t>(std::distance(cdf.begin(), it));
                    }
                } else {
                    std::vector<std::pair<float, int64_t>> keys(n);

                    for (size_t i = 0; i < n; ++i) {
                        float u = dis(gen);
                        u = std::clamp(u, 1e-10f, 1.0f - 1e-10f);
                        float gumbel = -std::log(-std::log(u));
                        float log_weight = std::log(std::max(weights_data[i], 1e-10f));
                        keys[i] = {log_weight + gumbel, static_cast<int64_t>(i)};
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
            LFS_ASSERT_MSG(args.shape.rank() == 2,
                           "eye requires a rank-2 output shape");
            LFS_ASSERT_MSG(args.dtype == DataType::Float32,
                           "eye currently supports only Float32");
            result = load(LoadOp::Const, {args.shape, args.device, args.dtype, args.use_pinned, 0.0f});
            if (!result.is_valid())
                return result;
            if (result.numel() == 0)
                return result;

            size_t m = args.shape[0];
            size_t n = args.shape[1];
            size_t min_dim = std::min(m, n);

            if (result.device_ == Device::CUDA) {
                LFS_CUDA_CHECK_MSG(cudaGetLastError(),
                                   "eye pending CUDA state check");
                tensor_ops::launch_eye(result.ptr<float>(), m, n, result.stream());
                LFS_CUDA_CHECK_MSG(cudaGetLastError(),
                                   "eye CUDA kernel launch");
                // No sync - tensor operation
            } else {
                float* data = result.ptr<float>();
                for (size_t i = 0; i < min_dim; ++i) {
                    data[i * n + i] = 1.0f;
                }
            }
            break;
        }

        case LoadOp::FromCPU: {
            LFS_ASSERT_MSG(std::holds_alternative<void*>(args.args),
                           "FromCPU load requires a source pointer");
            void* src_ptr = std::get<void*>(args.args);
            LFS_ASSERT_MSG(src_ptr != nullptr,
                           "FromCPU requires a non-null source pointer");

            result = Tensor(src_ptr, args.shape, Device::CPU, args.dtype);

            if (args.device == Device::CUDA) {
                result = result.to(Device::CUDA);
            }
            break;
        }

        case LoadOp::FromCUDA: {
            LFS_ASSERT_MSG(std::holds_alternative<void*>(args.args),
                           "FromCUDA load requires a source pointer");
            void* src_ptr = std::get<void*>(args.args);
            LFS_ASSERT_MSG(src_ptr != nullptr,
                           "FromCUDA requires a non-null source pointer");

            result = Tensor(src_ptr, args.shape, Device::CUDA, args.dtype);

            if (args.device == Device::CPU) {
                result = result.to(Device::CPU);
            }
            break;
        }

        default:
            LFS_ASSERT_MSG(false,
                           "unknown tensor load operation");
        }

        return result;
    }

    Tensor Tensor::multinomial(const Tensor& weights, int num_samples, bool replacement) {
        LFS_ASSERT_MSG(weights.is_valid() && weights.ndim() == 1,
                       "multinomial requires valid rank-1 weights");
        LFS_ASSERT_MSG(weights.dtype() == DataType::Float32,
                       "multinomial weights must be Float32");
        LFS_ASSERT_MSG(num_samples > 0,
                       "multinomial sample count must be positive");
        LFS_ASSERT_MSG(replacement || static_cast<size_t>(num_samples) <= weights.numel(),
                       "multinomial cannot sample more entries than weights without replacement");
        const auto host_weights = weights.to_vector();
        double weight_sum = 0.0;
        for (size_t index = 0; index < host_weights.size(); ++index) {
            const float weight = host_weights[index];
            LFS_ASSERT_MSG(std::isfinite(weight) && weight >= 0.0f,
                           "multinomial weights must be finite and non-negative");
            weight_sum += weight;
        }
        LFS_ASSERT_MSG(std::isfinite(weight_sum) && weight_sum > 0.0,
                       "multinomial weights must have a positive finite sum");

        LoadArgs args;
        args.shape = TensorShape({static_cast<size_t>(num_samples)});
        args.device = weights.device();
        args.dtype = DataType::Int64; // Must be Int64 for MCMC compatibility (nonzero() returns Int64)
        args.args = std::pair<void*, bool>{const_cast<void*>(static_cast<const void*>(&weights)), replacement};
        return load(LoadOp::Multinomial, args);
    }

    Tensor Tensor::reduce(const ReduceOp op) const {
        return reduce(op, ReduceArgs{});
    }

    Tensor Tensor::reduce(ReduceOp op, const ReduceArgs& args) const {
        LFS_CUDA_BREADCRUMB_STREAM("tensor.reduce", stream());
        constexpr std::array op_names = {
            "sum", "mean", "max", "min", "prod", "any", "all",
            "std", "var", "argmax", "argmin", "count_nonzero", "norm"};
        const int op_value = static_cast<int>(op);
        LFS_ASSERT_MSG(op_value >= static_cast<int>(ReduceOp::Sum) &&
                           op_value <= static_cast<int>(ReduceOp::Norm),
                       "reduce received an unknown operation");
        const char* op_name = op_names[static_cast<size_t>(op_value)];
        debug::OpTraceGuard trace(op_name, *this, LFS_SOURCE_SITE_CURRENT());

        validate_unary_op();
        LFS_ASSERT_MSG(op != ReduceOp::Argmax && op != ReduceOp::Argmin,
                       "argmax and argmin are not implemented by the reduction backend");
        LFS_ASSERT_MSG(op != ReduceOp::CountNonzero && op != ReduceOp::Norm,
                       "count_nonzero and norm must use their dedicated tensor operations");
        LFS_ASSERT_MSG(dtype_ == DataType::Float32 || dtype_ == DataType::Int32 ||
                           dtype_ == DataType::Bool,
                       "reduce currently supports only Float32, Int32, and Bool");
        std::vector<bool> seen_axes(shape_.rank(), false);
        for (const int axis : args.axes) {
            const int resolved = resolve_dim(axis);
            LFS_ASSERT_MSG(resolved >= 0 && resolved < static_cast<int>(shape_.rank()),
                           std::format("reduce axis {} is out of range for rank {}", axis, shape_.rank()));
            LFS_ASSERT_MSG(!seen_axes[resolved],
                           "reduce axes must be unique");
            seen_axes[resolved] = true;
        }
        if ((op == ReduceOp::Std || op == ReduceOp::Var || op == ReduceOp::Norm)) {
            LFS_ASSERT_MSG(dtype_ == DataType::Float32,
                           "std, var, and norm reductions currently require Float32");
        }
        if (dtype_ == DataType::Int32 && !args.axes.empty()) {
            LFS_ASSERT_MSG(args.axes.size() == shape_.rank(),
                           "Int32 partial reductions are unsupported");
        }
        if (dtype_ == DataType::Int32) {
            LFS_ASSERT_MSG(op == ReduceOp::Sum || op == ReduceOp::Mean ||
                               op == ReduceOp::Max || op == ReduceOp::Min ||
                               op == ReduceOp::Prod,
                           "Int32 reduction encountered an unsupported operation");
        }
        if (dtype_ == DataType::Float32) {
            LFS_ASSERT_MSG(op != ReduceOp::Any && op != ReduceOp::All,
                           "Float32 any/all reductions are unsupported");
        }
        if (dtype_ == DataType::Bool && device_ == Device::CPU &&
            !args.axes.empty() && args.axes.size() != shape_.rank()) {
            LFS_ASSERT_MSG(op == ReduceOp::Any || op == ReduceOp::All,
                           "partial CPU Bool reductions currently support only any and all");
        }
        if (dtype_ == DataType::Bool && device_ == Device::CUDA &&
            !args.axes.empty() && args.axes.size() != shape_.rank()) {
            LFS_ASSERT_MSG(op == ReduceOp::Any || op == ReduceOp::All,
                           "partial CUDA Bool reductions currently support only any and all");
            std::vector<int> sorted_axes = args.axes;
            for (int& axis : sorted_axes) {
                axis = resolve_dim(axis);
            }
            std::sort(sorted_axes.begin(), sorted_axes.end());
            for (size_t i = 1; i < sorted_axes.size(); ++i) {
                LFS_ASSERT_MSG(sorted_axes[i] == sorted_axes[i - 1] + 1,
                               "multi-axis CUDA Bool reductions require contiguous axes");
            }
        }

        // Fused transform-reduce: consume pending pointwise chain
        if (dtype_ == DataType::Float32 &&
            device_ == Device::CUDA && has_lazy_expr() &&
            (op == ReduceOp::Sum || op == ReduceOp::Mean ||
             op == ReduceOp::Max || op == ReduceOp::Min ||
             op == ReduceOp::Prod)) {
            bool is_full_reduce = args.axes.empty();
            if (!is_full_reduce) {
                std::vector<int> sorted_axes = args.axes;
                for (auto& ax : sorted_axes) {
                    if (ax < 0)
                        ax += static_cast<int>(shape_.rank());
                }
                std::sort(sorted_axes.begin(), sorted_axes.end());
                sorted_axes.erase(std::unique(sorted_axes.begin(), sorted_axes.end()), sorted_axes.end());
                is_full_reduce = sorted_axes.size() == shape_.rank();
            }
            if (is_full_reduce) {
                Tensor fused_source;
                std::vector<internal::LazyPointwiseOp> fused_ops;
                if (internal::lazy_executor_try_consume_pointwise_fusion(
                        lazy_expr_id(), &fused_source, &fused_ops) &&
                    fused_source.is_valid() &&
                    fused_source.device() == Device::CUDA &&
                    fused_source.is_contiguous() &&
                    fused_source.dtype() == DataType::Float32 &&
                    static_cast<int>(fused_ops.size()) <= tensor_ops::FUSED_POINTWISE_MAX_OPS) {
                    tensor_ops::FusedPointwiseOpChain chain{};
                    chain.num_ops = static_cast<int>(fused_ops.size());
                    for (int i = 0; i < chain.num_ops; ++i) {
                        chain.ops[i].kind = static_cast<uint8_t>(fused_ops[i].kind);
                        chain.ops[i].scalar = fused_ops[i].scalar;
                    }

                    const size_t n = fused_source.numel();
                    Tensor result;
                    {
                        const cudaStream_t execution_stream = resolve_cuda_execution_stream(fused_source);
                        fused_source.sync_to_stream(execution_stream);
                        CUDAStreamGuard guard(execution_stream);
                        result = Tensor::empty(
                            TensorShape(args.keepdim
                                            ? std::vector<size_t>(shape_.rank(), 1)
                                            : std::vector<size_t>{}),
                            Device::CUDA, DataType::Float32);
                        tensor_ops::launch_fused_transform_reduce(
                            fused_source.ptr<float>(), result.ptr<float>(), n,
                            chain, op, result.stream());
                    }

                    internal::lazy_executor_diagnostics_counters_increment_fused();
                    internal::lazy_executor_diagnostics_counters_increment_fused_reduce();
                    internal::lazy_ir_record_reduce(*this, result, op_name);
                    return result;
                }
            }
        }

        // Fused segmented transform-reduce: last-dim reduction with producer pointwise chain
        if (dtype_ == DataType::Float32 &&
            device_ == Device::CUDA && has_lazy_expr() &&
            (op == ReduceOp::Sum || op == ReduceOp::Mean ||
             op == ReduceOp::Max || op == ReduceOp::Min ||
             op == ReduceOp::Prod) &&
            args.axes.size() == 1 && shape_.rank() >= 2) {
            int dim = args.axes[0];
            if (dim < 0)
                dim += static_cast<int>(shape_.rank());
            if (dim == static_cast<int>(shape_.rank()) - 1) {
                Tensor fused_source;
                std::vector<internal::LazyPointwiseOp> fused_ops;
                if (internal::lazy_executor_try_consume_pointwise_fusion(
                        lazy_expr_id(), &fused_source, &fused_ops) &&
                    fused_source.is_valid() &&
                    fused_source.device() == Device::CUDA &&
                    fused_source.is_contiguous() &&
                    fused_source.dtype() == DataType::Float32 &&
                    static_cast<int>(fused_ops.size()) <= tensor_ops::FUSED_POINTWISE_MAX_OPS) {
                    tensor_ops::FusedPointwiseOpChain chain{};
                    chain.num_ops = static_cast<int>(fused_ops.size());
                    for (int i = 0; i < chain.num_ops; ++i) {
                        chain.ops[i].kind = static_cast<uint8_t>(fused_ops[i].kind);
                        chain.ops[i].scalar = fused_ops[i].scalar;
                    }

                    LFS_ASSERT_MSG(fused_source.shape().rank() > 0,
                                   std::format("fused segmented reduction source must have at least "
                                               "one dimension before reading the last dimension "
                                               "(source_shape={}, source_rank={}, source_numel={}, "
                                               "reduction_op={})",
                                               fused_source.shape().str(), fused_source.shape().rank(),
                                               fused_source.numel(), op_name));
                    const size_t segment_size = fused_source.shape()[fused_source.shape().rank() - 1];
                    LFS_ASSERT_MSG(segment_size > 0,
                                   std::format("fused segmented reduction requires a non-empty "
                                               "last dimension before division "
                                               "(segment_size={}, source_shape={}, source_numel={}, "
                                               "reduction_op={})",
                                               segment_size, fused_source.shape().str(),
                                               fused_source.numel(), op_name));
                    LFS_ASSERT_MSG(fused_source.numel() % segment_size == 0,
                                   std::format("fused segmented reduction source size must be an "
                                               "exact multiple of the segment size "
                                               "(source_numel={}, segment_size={}, source_shape={}, "
                                               "reduction_op={})",
                                               fused_source.numel(), segment_size,
                                               fused_source.shape().str(), op_name));
                    const size_t num_segments = fused_source.numel() / segment_size;
                    LFS_ASSERT_MSG(num_segments > 0,
                                   std::format("fused segmented reduction requires at least one segment "
                                               "(num_segments={}, source_numel={}, segment_size={}, "
                                               "source_shape={}, reduction_op={})",
                                               num_segments, fused_source.numel(), segment_size,
                                               fused_source.shape().str(), op_name));

                    std::vector<size_t> out_shape;
                    for (size_t i = 0; i < shape_.rank() - 1; ++i) {
                        out_shape.push_back(shape_[i]);
                    }
                    if (args.keepdim) {
                        out_shape.push_back(1);
                    }

                    Tensor result;
                    {
                        const cudaStream_t execution_stream = resolve_cuda_execution_stream(fused_source);
                        fused_source.sync_to_stream(execution_stream);
                        CUDAStreamGuard guard(execution_stream);
                        result = Tensor::empty(TensorShape(out_shape), Device::CUDA, DataType::Float32);
                        tensor_ops::launch_fused_segmented_transform_reduce(
                            fused_source.ptr<float>(), result.ptr<float>(),
                            num_segments, segment_size, chain, op, result.stream());
                    }

                    internal::lazy_executor_diagnostics_counters_increment_fused();
                    internal::lazy_executor_diagnostics_counters_increment_fused_reduce();
                    internal::lazy_ir_record_reduce(*this, result, op_name);
                    return result;
                }
            }
        }

        // Materialize deferred tensors before the reduce kernels capture raw shape/data pointers.
        // data_ptr() triggers materialization which std::moves internal state; if shape pointers
        // were captured before that move, they dangle. Materializing up front is safe and cheap
        // (no-op for already-materialized tensors).
        const_cast<Tensor*>(this)->materialize_if_deferred();

        // FAST PATH: 2D dim=0 reduction (column sums) - use specialized kernel
        // This is faster than transpose+contiguous+reduce because it avoids the copy
        if (args.axes.size() == 1 && device_ == Device::CUDA && shape_.rank() == 2 &&
            dtype_ == DataType::Float32 && is_contiguous_) {
            int dim = args.axes[0];
            if (dim < 0)
                dim += 2;
            if (dim == 0 && (op == ReduceOp::Sum || op == ReduceOp::Mean ||
                             op == ReduceOp::Max || op == ReduceOp::Min)) {
                size_t M = shape_[0]; // rows (reduction dim)
                size_t N = shape_[1]; // cols (output size)

                std::vector<size_t> out_shape = args.keepdim ? std::vector<size_t>{1, N} : std::vector<size_t>{N};
                auto result = empty_on_tensor_stream(TensorShape(out_shape), device_, dtype_, *this);

                LOG_DEBUG("[COLUMN REDUCE] M={}, N={}, op={}", M, N, static_cast<int>(op));
                tensor_ops::launch_column_reduce(ptr<float>(), result.ptr<float>(), M, N, op, result.stream());
                internal::lazy_ir_record_reduce(*this, result, op_name);
                return result;
            }
        }

        // OPTIMIZATION: For single-axis reduction where the reduction dimension is NOT the last,
        // it's faster to transpose the tensor so the reduction dim becomes contiguous, then reduce.
        // This trades a memory copy for much better memory coalescing in the reduction kernel.
        //
        // Example: [1024, 1024].sum({0}) with row-major layout:
        //   - Strided: Each output element reads 1024 values with stride=1024 → ~74 us
        //   - Transposed: Copy to column-major, then contiguous reduce → ~15 us
        //
        // Threshold: Only use this optimization when inner_size >= 256 (strided access hurts)
        if (args.axes.size() == 1 && device_ == Device::CUDA && shape_.rank() >= 2) {
            int dim = args.axes[0];
            if (dim < 0)
                dim += static_cast<int>(shape_.rank());

            if (dim >= 0 && dim < static_cast<int>(shape_.rank()) - 1) {
                // Calculate inner_size (product of dims after the reduction dim)
                size_t inner_size = 1;
                for (size_t i = dim + 1; i < shape_.rank(); ++i) {
                    inner_size *= shape_[i];
                }

                // Transpose+contiguous+reduce is faster than strided segmented reduce
                // The copy overhead is offset by better memory coalescing in reduction
                if (inner_size >= 256) {
                    // Build permutation to move dim to the last position
                    // e.g., for dim=0, rank=2: [0,1] → [1,0]
                    // e.g., for dim=1, rank=3: [0,1,2] → [0,2,1]
                    std::vector<int> perm;
                    for (size_t i = 0; i < shape_.rank(); ++i) {
                        if (static_cast<int>(i) != dim) {
                            perm.push_back(static_cast<int>(i));
                        }
                    }
                    perm.push_back(dim); // dim goes to the last position

                    LOG_DEBUG("[REDUCE TRANSPOSE] dim={}, inner_size={}, perm=[{}], shape=[{}]",
                              dim, inner_size,
                              perm.size() > 0 ? std::to_string(perm[0]) + (perm.size() > 1 ? "," + std::to_string(perm[1]) : "") + (perm.size() > 2 ? "," + std::to_string(perm[2]) : "") : "",
                              shape_.rank() > 0 ? std::to_string(shape_[0]) + (shape_.rank() > 1 ? "," + std::to_string(shape_[1]) : "") + (shape_.rank() > 2 ? "," + std::to_string(shape_[2]) : "") : "");

                    // Permute and make contiguous (this does the transpose copy)
                    Tensor transposed = this->permute(perm).contiguous();

                    LOG_DEBUG("[REDUCE TRANSPOSE] transposed shape=[{}], is_contiguous={}",
                              transposed.shape().rank() > 0 ? std::to_string(transposed.shape()[0]) + (transposed.shape().rank() > 1 ? "," + std::to_string(transposed.shape()[1]) : "") + (transposed.shape().rank() > 2 ? "," + std::to_string(transposed.shape()[2]) : "") : "",
                              transposed.is_contiguous() ? "true" : "false");

                    // Verify transposed tensor has expected number of elements
                    LOG_DEBUG("[REDUCE TRANSPOSE] orig numel={}, transposed numel={}", numel(), transposed.numel());

                    // Now reduce along the LAST dimension (which is contiguous)
                    ReduceArgs new_args = args;
                    new_args.axes = {static_cast<int>(transposed.shape().rank()) - 1}; // Use transposed.shape()!

                    return transposed.reduce(op, new_args);
                }
            }
        }

        // Reduce kernel expects contiguous memory
        const Tensor* input = this;
        Tensor contiguous_copy;
        if (!is_contiguous()) {
            contiguous_copy = this->contiguous();
            input = &contiguous_copy;
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
            axes.resize(input->shape_.rank());
            std::iota(axes.begin(), axes.end(), 0);
        }

        // Resolve negative indices to positive indices
        for (auto& ax : axes) {
            ax = input->resolve_dim(ax);
        }

        std::vector<size_t> out_shape;
        for (size_t i = 0; i < input->shape_.rank(); ++i) {
            bool is_reduced = std::find(axes.begin(), axes.end(), static_cast<int>(i)) != axes.end();
            if (!is_reduced || args.keepdim) {
                out_shape.push_back(is_reduced ? 1 : input->shape_[i]);
            }
        }

        DataType out_dtype = input->dtype_;
        if (op == ReduceOp::Any || op == ReduceOp::All) {
            out_dtype = DataType::Bool;
        } else if (op == ReduceOp::Argmax || op == ReduceOp::Argmin) {
            out_dtype = DataType::Int64;
        } else if (input->dtype_ == DataType::Bool) {
            // Bool reductions return Int64 (PyTorch behavior)
            out_dtype = DataType::Int64;
        }

        std::optional<CUDAStreamGuard> execution_guard;
        if (input->device_ == Device::CUDA) {
            const cudaStream_t execution_stream = resolve_cuda_execution_stream(*input);
            input->sync_to_stream(execution_stream);
            execution_guard.emplace(execution_stream);
        }

        auto result = Tensor::empty(TensorShape(out_shape), input->device_, out_dtype);

        if (input->numel() == 0) {
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
                identity_value = input->dtype_ == DataType::Bool
                                     ? 0.0f
                                     : -std::numeric_limits<float>::infinity();
                break;
            case ReduceOp::Min:
                identity_value = input->dtype_ == DataType::Bool
                                     ? 1.0f
                                     : std::numeric_limits<float>::infinity();
                break;
            case ReduceOp::Any:
                identity_value = 0.0f;
                break;
            case ReduceOp::All:
                identity_value = 1.0f;
                break;
            default:
                LFS_ASSERT_MSG(false,
                               "empty reduction encountered an unsupported operation");
            }

            if (input->dtype_ == DataType::Int32 &&
                (op == ReduceOp::Max || op == ReduceOp::Min)) {
                LFS_ASSERT_MSG(false,
                               "empty Int32 max/min reductions have no identity value");
            }

            if (result.numel() == 0) {
                return result;
            }

            if (input->device_ == Device::CUDA) {
                if (out_dtype == DataType::Float32) {
                    std::vector<float> temp(result.numel(), identity_value);
                    LFS_CUDA_CHECK_MSG(
                        cudaMemcpy(result.data_ptr(), temp.data(), result.bytes(),
                                   cudaMemcpyHostToDevice),
                        "empty reduction CUDA copy");
                } else if (out_dtype == DataType::Bool) {
                    unsigned char bool_val = (identity_value != 0.0f) ? 1 : 0;
                    LFS_CUDA_CHECK_MSG(
                        cudaMemset(result.data_ptr(), bool_val, result.bytes()),
                        "empty reduction CUDA memset");
                } else if (out_dtype == DataType::Int32) {
                    if (identity_value == 0.0f) {
                        LFS_CUDA_CHECK_MSG(cudaMemset(result.data_ptr(), 0, result.bytes()),
                                           "empty Int32 reduction CUDA memset");
                    } else {
                        std::vector<int32_t> temp(
                            result.numel(), static_cast<int32_t>(identity_value));
                        LFS_CUDA_CHECK_MSG(
                            cudaMemcpy(result.data_ptr(), temp.data(), result.bytes(),
                                       cudaMemcpyHostToDevice),
                            "empty Int32 reduction CUDA copy");
                    }
                } else if (out_dtype == DataType::Int64) {
                    if (identity_value == 0.0f) {
                        LFS_CUDA_CHECK_MSG(cudaMemset(result.data_ptr(), 0, result.bytes()),
                                           "empty Int64 reduction CUDA memset");
                    } else {
                        std::vector<int64_t> temp(
                            result.numel(), static_cast<int64_t>(identity_value));
                        LFS_CUDA_CHECK_MSG(
                            cudaMemcpy(result.data_ptr(), temp.data(), result.bytes(),
                                       cudaMemcpyHostToDevice),
                            "empty Int64 reduction CUDA copy");
                    }
                } else {
                    LFS_ASSERT_MSG(false,
                                   "empty CUDA reduction encountered an unsupported output dtype");
                }
            } else {
                if (out_dtype == DataType::Float32) {
                    float* ptr = static_cast<float*>(result.data_ptr());
                    std::fill_n(ptr, result.numel(), identity_value);
                } else if (out_dtype == DataType::Bool) {
                    unsigned char* ptr = static_cast<unsigned char*>(result.data_ptr());
                    std::fill_n(ptr, result.numel(), identity_value != 0.0f ? 1 : 0);
                } else if (out_dtype == DataType::Int32) {
                    int32_t* ptr = static_cast<int32_t*>(result.data_ptr());
                    std::fill_n(ptr, result.numel(), static_cast<int32_t>(identity_value));
                } else if (out_dtype == DataType::Int64) {
                    int64_t* ptr = static_cast<int64_t*>(result.data_ptr());
                    std::fill_n(ptr, result.numel(), static_cast<int64_t>(identity_value));
                } else {
                    LFS_ASSERT_MSG(false,
                                   "empty CPU reduction encountered an unsupported output dtype");
                }
            }
            return result;
        }

        if (input->device_ == Device::CUDA) {
            tensor_ops::launch_reduce_op(
                input->data_ptr(), result.data_ptr(),
                input->shape_.dims().data(), input->shape_.rank(),
                axes.data(), axes.size(),
                args.keepdim, op,
                input->dtype_, result.stream());
            // No sync - tensor operation
        } else {
            // CPU implementation

            // Handle Int32 dtype
            if (input->dtype_ == DataType::Int32) {
                const int* src = static_cast<const int*>(input->data_ptr());
                int* dst = static_cast<int*>(result.data_ptr());

                // Full reduction to scalar (only mode supported for Int32)
                if (axes.size() == input->shape_.rank()) {
                    if (op == ReduceOp::Sum) {
                        int sum = 0;
                        for (size_t i = 0; i < input->numel(); ++i) {
                            sum += src[i];
                        }
                        dst[0] = sum;
                    } else if (op == ReduceOp::Mean) {
                        int sum = 0;
                        for (size_t i = 0; i < input->numel(); ++i) {
                            sum += src[i];
                        }
                        dst[0] = sum / static_cast<int>(input->numel());
                    } else if (op == ReduceOp::Max) {
                        int max_val = src[0];
                        for (size_t i = 1; i < input->numel(); ++i) {
                            max_val = std::max(max_val, src[i]);
                        }
                        dst[0] = max_val;
                    } else if (op == ReduceOp::Min) {
                        int min_val = src[0];
                        for (size_t i = 1; i < input->numel(); ++i) {
                            min_val = std::min(min_val, src[i]);
                        }
                        dst[0] = min_val;
                    } else if (op == ReduceOp::Prod) {
                        int prod = 1;
                        for (size_t i = 0; i < input->numel(); ++i) {
                            prod *= src[i];
                        }
                        dst[0] = prod;
                    }
                    return result;
                }
                // Partial reductions not supported for Int32
                return result;
            }

            // Full Bool arithmetic reductions produce Int64 results.
            if (input->dtype_ == DataType::Bool &&
                axes.size() == input->shape_.rank() &&
                op != ReduceOp::Any && op != ReduceOp::All) {
                const unsigned char* src =
                    static_cast<const unsigned char*>(input->data_ptr());
                int64_t* dst = static_cast<int64_t*>(result.data_ptr());
                int64_t true_count = 0;
                for (size_t i = 0; i < input->numel(); ++i) {
                    true_count += src[i] != 0 ? 1 : 0;
                }

                switch (op) {
                case ReduceOp::Sum:
                    dst[0] = true_count;
                    break;
                case ReduceOp::Mean:
                    dst[0] = true_count / static_cast<int64_t>(input->numel());
                    break;
                case ReduceOp::Max:
                    dst[0] = true_count != 0 ? 1 : 0;
                    break;
                case ReduceOp::Min:
                case ReduceOp::Prod:
                    dst[0] = true_count == static_cast<int64_t>(input->numel()) ? 1 : 0;
                    break;
                default:
                    LFS_ASSERT_MSG(false,
                                   "CPU Bool reduction encountered an unsupported operation");
                }
                return result;
            }

            // Handle Bool dtype for Any/All operations
            if (input->dtype_ == DataType::Bool && (op == ReduceOp::Any || op == ReduceOp::All)) {
                // For non-contiguous tensors, make contiguous first for correct linear access
                const Tensor* src_tensor = input;
                Tensor contiguous_copy;
                if (!input->is_contiguous_) {
                    contiguous_copy = input->contiguous();
                    src_tensor = &contiguous_copy;
                }

                const unsigned char* src = static_cast<const unsigned char*>(src_tensor->data_ptr());
                unsigned char* dst = static_cast<unsigned char*>(result.data_ptr());

                // Full reduction to scalar
                if (axes.size() == input->shape_.rank()) {
                    if (op == ReduceOp::Any) {
                        bool any_true = false;
                        for (size_t i = 0; i < src_tensor->numel(); ++i) {
                            if (src[i]) {
                                any_true = true;
                                break;
                            }
                        }
                        dst[0] = any_true ? 1 : 0;
                    } else { // ReduceOp::All
                        bool all_true = true;
                        for (size_t i = 0; i < src_tensor->numel(); ++i) {
                            if (!src[i]) {
                                all_true = false;
                                break;
                            }
                        }
                        dst[0] = all_true ? 1 : 0;
                    }
                    return result;
                }

                // Axis-specific reduction for Bool
                // Build mask of which dimensions are reduced
                std::vector<bool> is_reduced_dim(input->shape_.rank(), false);
                for (int ax : axes) {
                    int resolved = input->resolve_dim(ax);
                    if (resolved >= 0 && resolved < static_cast<int>(input->shape_.rank())) {
                        is_reduced_dim[resolved] = true;
                    }
                }

                const auto& input_strides = input->strides_;
                std::vector<size_t> out_shape_vec;
                for (size_t i = 0; i < input->shape_.rank(); ++i) {
                    if (!is_reduced_dim[i]) {
                        out_shape_vec.push_back(input->shape_[i]);
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
                for (size_t i = 0; i < input->shape_.rank(); ++i) {
                    if (is_reduced_dim[i]) {
                        reduced_dims.push_back(i);
                        reduce_count *= input->shape_[i];
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
                    std::vector<size_t> base_input_coords(input->shape_.rank());
                    size_t out_coord_idx = 0;
                    for (size_t i = 0; i < input->shape_.rank(); ++i) {
                        if (!is_reduced_dim[i]) {
                            base_input_coords[i] = out_coords[out_coord_idx++];
                        } else {
                            base_input_coords[i] = 0;
                        }
                    }

                    // Initialize result with identity value
                    bool result_val = (op == ReduceOp::All); // All starts true, Any starts false

                    // Iterate through all combinations of reduced dimensions
                    for (size_t r = 0; r < reduce_count; ++r) {
                        // Compute coordinates in the reduced dimensions
                        size_t temp_r = r;
                        std::vector<size_t> full_input_coords = base_input_coords;

                        // Fill in reduced dimensions - work backwards for row-major order
                        for (int rd_idx = static_cast<int>(reduced_dims.size()) - 1; rd_idx >= 0; --rd_idx) {
                            size_t dim = reduced_dims[rd_idx];
                            full_input_coords[dim] = temp_r % input->shape_[dim];
                            temp_r /= input->shape_[dim];
                        }

                        // Calculate linear input index
                        size_t in_idx = 0;
                        for (size_t i = 0; i < input->shape_.rank(); ++i) {
                            in_idx += full_input_coords[i] * input_strides[i];
                        }

                        // Apply reduction operation
                        bool val = src[in_idx] != 0;
                        if (op == ReduceOp::Any) {
                            if (val) {
                                result_val = true;
                                break; // Short-circuit: found a true value
                            }
                        } else { // ReduceOp::All
                            if (!val) {
                                result_val = false;
                                break; // Short-circuit: found a false value
                            }
                        }
                    }

                    dst[out_idx] = result_val ? 1 : 0;
                }

                return result;
            }

            // Float32 implementation
            const float* src = static_cast<const float*>(input->data_ptr());
            float* dst = static_cast<float*>(result.data_ptr());

            // Full reduction to scalar
            if (axes.size() == input->shape_.rank()) {
                if (op == ReduceOp::Sum) {
                    // Use double accumulation to avoid FP32 precision loss
                    double sum = 0.0;
                    for (size_t i = 0; i < input->numel(); ++i) {
                        sum += src[i];
                    }
                    dst[0] = static_cast<float>(sum);
                } else if (op == ReduceOp::Mean) {
                    // Use double accumulation to avoid FP32 precision loss
                    double sum = 0.0;
                    for (size_t i = 0; i < input->numel(); ++i) {
                        sum += src[i];
                    }
                    dst[0] = static_cast<float>(sum / input->numel());
                } else if (op == ReduceOp::Max) {
                    float max_val = src[0];
                    for (size_t i = 1; i < input->numel(); ++i) {
                        max_val = std::max(max_val, src[i]);
                    }
                    dst[0] = max_val;
                } else if (op == ReduceOp::Min) {
                    float min_val = src[0];
                    for (size_t i = 1; i < input->numel(); ++i) {
                        min_val = std::min(min_val, src[i]);
                    }
                    dst[0] = min_val;
                } else if (op == ReduceOp::Prod) {
                    float prod = 1.0f;
                    for (size_t i = 0; i < input->numel(); ++i) {
                        prod *= src[i];
                    }
                    dst[0] = prod;
                }
                return result;
            }

            // Axis-specific reduction - general implementation
            // Build mask of which dimensions are reduced
            std::vector<bool> is_reduced_dim(input->shape_.rank(), false);
            for (int ax : axes) {
                int resolved = input->resolve_dim(ax);
                if (resolved >= 0 && resolved < static_cast<int>(input->shape_.rank())) {
                    is_reduced_dim[resolved] = true;
                }
            }

            const auto& input_strides = input->strides_;

            std::vector<size_t> out_shape_vec;
            for (size_t i = 0; i < input->shape_.rank(); ++i) {
                if (!is_reduced_dim[i]) {
                    out_shape_vec.push_back(input->shape_[i]);
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
            for (size_t i = 0; i < input->shape_.rank(); ++i) {
                if (is_reduced_dim[i]) {
                    reduced_dims.push_back(i);
                    reduce_count *= input->shape_[i];
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
                std::vector<size_t> base_input_coords(input->shape_.rank());
                size_t out_coord_idx = 0;
                for (size_t i = 0; i < input->shape_.rank(); ++i) {
                    if (!is_reduced_dim[i]) {
                        base_input_coords[i] = out_coords[out_coord_idx++];
                    } else {
                        base_input_coords[i] = 0;
                    }
                }

                // Initialize result with identity value for this output element
                // Use double for sum/mean to avoid FP32 precision loss
                double result_val_double = 0.0;
                float result_val_float = 0.0f;

                if (op == ReduceOp::Max) {
                    result_val_float = -std::numeric_limits<float>::infinity();
                } else if (op == ReduceOp::Min) {
                    result_val_float = std::numeric_limits<float>::infinity();
                } else if (op == ReduceOp::Prod) {
                    result_val_float = 1.0f;
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
                    case ReduceOp::Mean:          // Mean accumulates like sum, then divides at end
                        result_val_double += val; // Use double accumulation
                        break;
                    case ReduceOp::Max:
                        result_val_float = std::max(result_val_float, val);
                        break;
                    case ReduceOp::Min:
                        result_val_float = std::min(result_val_float, val);
                        break;
                    case ReduceOp::Prod:
                        result_val_float *= val;
                        break;
                    default:
                        break;
                    }
                }

                // Store result (apply mean if needed)
                if (op == ReduceOp::Sum) {
                    dst[out_idx] = static_cast<float>(result_val_double);
                } else if (op == ReduceOp::Mean) {
                    dst[out_idx] = static_cast<float>(result_val_double / reduce_count);
                } else {
                    dst[out_idx] = result_val_float;
                }
            }
        }

        internal::lazy_ir_record_reduce(*input, result, op_name);
        return result;
    }

    // ============= TERNARY OPERATIONS =============

    Tensor Tensor::ternary(const Tensor& b, const Tensor& c) const {
        validate_ternary_op(b, c);
        LFS_ASSERT_MSG(dtype_ == DataType::Bool,
                       "where condition must have Bool dtype");

        if (numel() == 0 || b.numel() == 0 || c.numel() == 0) {
            auto shape_ab = this->broadcast_shape(b.shape());
            LFS_ASSERT_MSG(shape_ab.rank() != 0,
                           "where inputs have incompatible broadcast shapes");

            auto shape_abc_vec = broadcast::shape(shape_ab.dims(), c.shape().dims());
            LFS_ASSERT_MSG(!shape_abc_vec.empty(),
                           "where inputs have incompatible broadcast shapes");

            DataType out_dtype = promote_types(b.dtype(), c.dtype());
            return empty(TensorShape(shape_abc_vec), device_, out_dtype);
        }

        auto shape_ab = this->broadcast_shape(b.shape());
        LFS_ASSERT_MSG(shape_ab.rank() != 0,
                       "where condition and x shapes are incompatible");

        auto shape_abc_vec = broadcast::shape(shape_ab.dims(), c.shape().dims());
        LFS_ASSERT_MSG(!shape_abc_vec.empty(),
                       "where input shapes are incompatible");

        TensorShape shape_abc(shape_abc_vec);

        DataType out_dtype = promote_types(b.dtype(), c.dtype());

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

        Tensor b_cast = (b_broadcast.dtype() == out_dtype) ? b_broadcast : b_broadcast.to(out_dtype);
        Tensor c_cast = (c_broadcast.dtype() == out_dtype) ? c_broadcast : c_broadcast.to(out_dtype);
        LFS_ASSERT_MSG(b_cast.is_valid() && c_cast.is_valid(),
                       std::format("where failed to cast inputs to output dtype {}",
                                   dtype_name(out_dtype)));

        if (device_ == Device::CUDA && out_dtype == DataType::Float32) {
            auto result = Tensor::empty(shape_abc, device_, out_dtype);
            tensor_ops::launch_where(
                a_broadcast.ptr<unsigned char>(),
                b_cast.ptr<float>(),
                c_cast.ptr<float>(),
                result.ptr<float>(),
                a_broadcast.shape().dims().data(),
                b_cast.shape().dims().data(),
                c_cast.shape().dims().data(),
                result.shape().dims().data(),
                a_broadcast.shape().rank(),
                b_cast.shape().rank(),
                c_cast.shape().rank(),
                result.shape().rank(),
                result.numel(),
                result.stream());
            // No sync - tensor operation
            return result;
        }

        Tensor cond_cpu = (a_broadcast.device() == Device::CUDA) ? a_broadcast.to(Device::CPU) : a_broadcast;
        Tensor x_cpu = (b_cast.device() == Device::CUDA) ? b_cast.to(Device::CPU) : b_cast;
        Tensor y_cpu = (c_cast.device() == Device::CUDA) ? c_cast.to(Device::CPU) : c_cast;
        LFS_ASSERT_MSG(cond_cpu.is_valid() && x_cpu.is_valid() && y_cpu.is_valid(),
                       std::format("where failed to materialize host tensors for dtype {}",
                                   dtype_name(out_dtype)));

        Tensor result_cpu = Tensor::empty(shape_abc, Device::CPU, out_dtype);
        const unsigned char* cond = cond_cpu.ptr<unsigned char>();
        const char* x = static_cast<const char*>(x_cpu.data_ptr());
        const char* y = static_cast<const char*>(y_cpu.data_ptr());
        char* dst = static_cast<char*>(result_cpu.data_ptr());
        const size_t elem_size = dtype_size(out_dtype);

        for (size_t i = 0; i < result_cpu.numel(); ++i) {
            const char* src = cond[i] ? (x + i * elem_size) : (y + i * elem_size);
            std::memcpy(dst + i * elem_size, src, elem_size);
        }

        if (device_ == Device::CUDA) {
            return result_cpu.to(Device::CUDA);
        }
        return result_cpu;
    }

    Tensor Tensor::where(const Tensor& condition, const Tensor& x, const Tensor& y) {
        tensor_contract::require_valid(
            condition, "where", "condition", LFS_SOURCE_SITE_CURRENT());
        tensor_contract::require_valid(
            x, "where", "true value", LFS_SOURCE_SITE_CURRENT());
        tensor_contract::require_valid(
            y, "where", "false value", LFS_SOURCE_SITE_CURRENT());
        tensor_contract::require_dtype(
            condition, DataType::Bool, "where", "condition", LFS_SOURCE_SITE_CURRENT());
        tensor_contract::require_same_device(
            condition, x, "where", "condition", "true value", LFS_SOURCE_SITE_CURRENT());
        tensor_contract::require_same_device(
            condition, y, "where", "condition", "false value", LFS_SOURCE_SITE_CURRENT());
        return condition.ternary(x, y);
    }

    float Tensor::norm(float p) const {
        LFS_ASSERT_MSG(is_valid(),
                       "norm requires a valid tensor");
        LFS_ASSERT_MSG(dtype_ == DataType::Float32,
                       "norm currently supports only Float32");
        LFS_ASSERT_MSG(!std::isnan(p),
                       "norm order cannot be NaN");

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
        LFS_ASSERT_MSG(is_valid(),
                       "norm requires a valid tensor");
        LFS_ASSERT_MSG(dtype_ == DataType::Float32,
                       "norm currently supports only Float32");
        LFS_ASSERT_MSG(!std::isnan(p),
                       "norm order cannot be NaN");
        (void)resolve_dims(dims);

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
        LFS_ASSERT_MSG(is_valid() && other.is_valid(),
                       "broadcast requires valid tensors");
        LFS_ASSERT_MSG(device_ == other.device(),
                       "broadcast operands must be on the same device");

        auto bcast_shape = this->broadcast_shape(other.shape());
        LFS_ASSERT_MSG(bcast_shape.rank() != 0,
                       "broadcast shapes are incompatible");

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
        LFS_CUDA_BREADCRUMB("tensor.cat");
        if (tensors.empty()) {
            throw std::invalid_argument("Cannot concatenate empty vector of tensors");
        }
        for (const auto& tensor : tensors) {
            tensor_contract::require_valid(
                tensor, "cat", "input", LFS_SOURCE_SITE_CURRENT());
        }

        if (tensors.size() == 1) {
            return tensors[0].clone();
        }

        int resolved_dim = dim;
        if (resolved_dim < 0) {
            resolved_dim = tensors[0].shape().rank() + resolved_dim;
        }

        if (resolved_dim < 0 || resolved_dim >= static_cast<int>(tensors[0].shape().rank())) {
            throw std::invalid_argument(std::format(
                "Invalid dimension for cat: dim={}, rank={}", dim, tensors[0].shape().rank()));
        }

        const auto& first_shape = tensors[0].shape();
        const auto first_device = tensors[0].device();
        const auto first_dtype = tensors[0].dtype();

        // Early detection of rank-0 tensors
        if (first_shape.rank() == 0) {
            LOG_ERROR("cat(): First tensor is rank-0 (scalar)! Cannot concatenate scalars.");
            LOG_ERROR("  Tensor 0: valid={}, shape={}", tensors[0].is_valid(), first_shape.str());
            throw std::runtime_error("cat() called with rank-0 first tensor");
        }

        size_t total_size_along_dim = first_shape[resolved_dim];

        // Validate all tensors
        for (size_t i = 1; i < tensors.size(); ++i) {
            const auto& shape = tensors[i].shape();

            if (shape.rank() != first_shape.rank()) {
                LOG_ERROR("================================================================");
                LOG_ERROR("CRITICAL: cat() rank mismatch detected!");
                LOG_ERROR("================================================================");
                LOG_ERROR("Attempting to concatenate tensors with different ranks:");
                LOG_ERROR("  Tensor 0: rank={}, shape={}, valid={}", first_shape.rank(), first_shape.str(), tensors[0].is_valid());
                LOG_ERROR("  Tensor {}: rank={}, shape={}, valid={}", i, shape.rank(), shape.str(), tensors[i].is_valid());
                LOG_ERROR("  Concatenation dimension: {}", dim);
                LOG_ERROR("  Number of tensors being concatenated: {}", tensors.size());

                // Check if tensor 1 is invalid or scalar
                if (i == 1 && shape.rank() == 0) {
                    LOG_ERROR("  Tensor 1 is SCALAR/RANK-0! This usually means:");
                    LOG_ERROR("    - Tensor was created with empty shape vector");
                    LOG_ERROR("    - Tensor is invalid (is_valid={})", tensors[i].is_valid());
                    LOG_ERROR("    - Bug in tensor creation code (e.g., zeros_dims empty)");
                }
                LOG_ERROR("================================================================");
                throw std::runtime_error(std::format(
                    "cat() rank mismatch: tensor 0 has rank {} (shape {}), tensor {} has rank {} (shape {})",
                    first_shape.rank(), first_shape.str(), i, shape.rank(), shape.str()));
            }

            for (size_t d = 0; d < shape.rank(); ++d) {
                if (d != static_cast<size_t>(resolved_dim) && shape[d] != first_shape[d]) {
                    throw std::invalid_argument(std::format(
                        "cat() dimension mismatch: all dimensions except dim={} must match. "
                        "Tensor 0 shape: {}, Tensor {} shape: {}, mismatch at dimension {}",
                        dim, first_shape.str(), i, shape.str(), d));
                }
            }

            if (tensors[i].device() != first_device) {
                throw std::invalid_argument(std::format(
                    "cat() device mismatch: tensor 0 is on device {}, tensor {} is on device {}",
                    static_cast<int>(first_device), i, static_cast<int>(tensors[i].device())));
            }

            if (tensors[i].dtype() != first_dtype) {
                throw std::invalid_argument(std::format(
                    "cat() dtype mismatch: tensor 0 has dtype {}, tensor {} has dtype {}",
                    static_cast<int>(first_dtype), i, static_cast<int>(tensors[i].dtype())));
            }

            total_size_along_dim += shape[resolved_dim];
        }

        // Build result shape (needed for both in-place and standard paths)
        std::vector<size_t> result_dims = first_shape.dims();
        result_dims[resolved_dim] = total_size_along_dim;

        // ============= IN-PLACE OPTIMIZATION CHECK =============
        // Check if we can grow the first tensor in-place using reserved capacity
        // Conditions:
        // 1. First tensor must own its data (not a view)
        // 2. First tensor must have reserved capacity
        // 3. Concatenation must be along dimension 0 (first dimension)
        // 4. First tensor must have enough capacity for the total size
        // Check if in-place optimization is possible
        LOG_DEBUG("  In-place check: tensors[0] id={}, data_ptr={}, capacity={}, shape[0]={}, total_needed={}",
                  tensors[0].id_, tensors[0].data_, tensors[0].capacity(), tensors[0].shape()[0], total_size_along_dim);
        if (tensors.size() > 1) {
            LOG_DEBUG("  tensors[1] id={}, data_ptr={}, capacity={}, shape[0]={}",
                      tensors[1].id_, tensors[1].data_, tensors[1].capacity(), tensors[1].shape()[0]);
        }

        // IN-PLACE OPTIMIZATION: Reuse pre-allocated capacity when available
        // FIXED: Move assignment operator now properly transfers capacity_ and logical_size_
        if (resolved_dim == 0 &&
            tensors[0].data_owner_ &&
            tensors[0].capacity() > 0 &&
            tensors[0].capacity() >= total_size_along_dim) {

            LOG_DEBUG("  ✓ IN-PLACE OPTIMIZATION: Reusing buffer");
            // IN-PLACE PATH: Reuse first tensor's pre-allocated buffer
            // IMPORTANT: Use logical_size_ (actual current size) not shape_[0] which may be stale after reserve()
            const size_t first_size = (tensors[0].capacity() > 0 && tensors[0].logical_size() > 0)
                                          ? tensors[0].logical_size()
                                          : first_shape[0];
            const size_t row_size = tensors[0].numel() / first_shape[0]; // elements per "row" based on CURRENT shape
            const size_t element_size = dtype_size(first_dtype);

            LOG_DEBUG("Tensor::cat() IN-PLACE: growing tensor #{} from {} to {} rows (capacity {})",
                      tensors[0].id_, first_size, total_size_along_dim, tensors[0].capacity());
            LOG_DEBUG("  first_shape[0]={}, logical_size={}, numel={}, row_size={}, element_size={}",
                      first_shape[0], tensors[0].logical_size(), tensors[0].numel(), row_size, element_size);
            LOG_DEBUG("  Buffer offset calculation: first_size={} * row_size={} * element_size={} = {} bytes",
                      first_size, row_size, element_size, first_size * row_size * element_size);

            // Create result tensor that shares the first tensor's buffer
            Tensor result;
            result.shape_ = TensorShape(result_dims);
            result.strides_ = result.shape_.strides();
            result.storage_offset_ = 0;
            result.is_contiguous_ = true;
            result.device_ = first_device;
            result.dtype_ = first_dtype;
            result.data_ = tensors[0].data_;
            result.data_owner_ = tensors[0].data_owner_; // Share ownership
            result.state_ = std::make_shared<Tensor::TensorState>(*tensors[0].state_);
            result.state_->capacity = tensors[0].capacity();
            result.state_->logical_size = total_size_along_dim;
            result.is_view_ = false;                // Not a view, it owns the data (via shared_ptr)
            result.set_stream(tensors[0].stream()); // Inherit stream from first tensor
            result.compute_alignment();             // Compute alignment flags
            result.id_ = Tensor::next_id_++;

            LOG_DEBUG("  Result tensor: id={}, data_ptr={}, capacity={}, logical_size={}",
                      result.id_, result.data_, result.capacity(), result.logical_size());

            // Copy additional tensors into the reserved space
            if (first_device == Device::CUDA) {
                size_t offset = first_size * row_size * element_size;
                LOG_DEBUG("  Starting CUDA memcpy for {} additional tensors, initial offset={} bytes",
                          tensors.size() - 1, offset);

                LFS_VALIDATE_CUDA_DEVICE_POINTER(result.data_, "in-place cat destination");

                for (size_t i = 1; i < tensors.size(); ++i) {
                    const size_t bytes = tensors[i].bytes();
                    const size_t tensor_rows = tensors[i].shape()[0];
                    const void* src_ptr = tensors[i].data_ptr();
                    LOG_DEBUG("  Copying tensor[{}]: shape_[0]={}, numel={}, {} bytes from src={} at offset {}",
                              i, tensor_rows, tensors[i].numel(), bytes, src_ptr, offset);

                    LFS_VALIDATE_CUDA_DEVICE_POINTER(src_ptr, "in-place cat source");

                    LFS_CUDA_CHECK_MSG(
                        cudaMemcpy(static_cast<char*>(result.data_) + offset, src_ptr, bytes,
                                   cudaMemcpyDeviceToDevice),
                        "in-place cat copy (tensor_index={}, source_pointer={}, "
                        "source_device={}, source_contiguous={}, source_is_view={}, "
                        "destination_pointer={}, destination_offset={}, bytes={}, "
                        "destination_end={})",
                        i, src_ptr, static_cast<int>(tensors[i].device()),
                        tensors[i].is_contiguous(), tensors[i].is_view(), result.data_, offset,
                        bytes, offset + bytes);
                    offset += bytes;
                }
                LOG_DEBUG("  CUDA memcpy complete, final offset={} bytes", offset);
            } else {
                size_t offset = first_size * row_size * element_size;
                for (size_t i = 1; i < tensors.size(); ++i) {
                    const size_t bytes = tensors[i].bytes();
                    std::memcpy(
                        static_cast<char*>(result.data_) + offset,
                        tensors[i].data_ptr(),
                        bytes);
                    offset += bytes;
                }
            }

            LOG_DEBUG("  ← Returning IN-PLACE result: id={}, data_ptr={}, capacity={}",
                      result.id_, result.data_ptr(), result.capacity());
            return result;
        }

        // ============= FALLBACK: Standard allocation path =============
        LOG_DEBUG("  → SLOW PATH: Allocating new buffer");
        auto result = Tensor::empty(TensorShape(result_dims), first_device, first_dtype);
        LOG_DEBUG("  Created new tensor: id={}, data_ptr={}, capacity={}",
                  result.id_, result.data_ptr(), result.capacity());

        size_t element_size = dtype_size(first_dtype);

        // ============= OPTIMIZED PATH: First dimension =============
        if (resolved_dim == 0) {
            // Concatenating along first dimension - completely contiguous
            if (first_device == Device::CUDA) {
                size_t offset = 0;
                for (const auto& t : tensors) {
                    size_t bytes = t.bytes();
                    LFS_CUDA_CHECK_MSG(
                        cudaMemcpy(static_cast<char*>(result.data_ptr()) + offset,
                                   t.data_ptr(), bytes, cudaMemcpyDeviceToDevice),
                        "cat CUDA copy");
                    offset += bytes;
                }
            } else {
                size_t offset = 0;
                for (const auto& t : tensors) {
                    size_t bytes = t.bytes();
                    std::memcpy(
                        static_cast<char*>(result.data_ptr()) + offset,
                        t.data_ptr(),
                        bytes);
                    offset += bytes;
                }
            }

            LOG_DEBUG("  ← Returning SLOW PATH result: id={}, data_ptr={}, capacity={}",
                      result.id_, result.data_ptr(), result.capacity());
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
                    result.data_ptr(),
                    tensors,
                    num_rows,
                    row_size,
                    element_size,
                    result.stream());
                // No sync - tensor operation
            } else {
                // CPU: Simple memcpy per row
                size_t result_offset = 0;
                for (const auto& t : tensors) {
                    size_t tensor_dim_size = t.shape()[resolved_dim];

                    for (size_t row = 0; row < num_rows; ++row) {
                        const void* src = static_cast<const char*>(t.data_ptr()) +
                                          row * tensor_dim_size * element_size;
                        void* dst = static_cast<char*>(result.data_ptr()) +
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
                result.data_ptr(),
                tensors,
                outer_size,
                inner_size,
                resolved_dim,
                element_size,
                result.stream());
            // No sync - tensor operation
        } else {
            // CPU fallback
            for (size_t outer = 0; outer < outer_size; ++outer) {
                size_t result_offset = 0;

                for (const auto& t : tensors) {
                    size_t tensor_dim_size = t.shape()[resolved_dim];
                    size_t copy_size = tensor_dim_size * inner_size * element_size;

                    const void* src = static_cast<const char*>(t.data_ptr()) +
                                      outer * tensor_dim_size * inner_size * element_size;
                    void* dst = static_cast<char*>(result.data_ptr()) +
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
        LFS_ASSERT_MSG(!tensors.empty(),
                       "stack requires at least one tensor");
        tensor_contract::require_valid(
            tensors[0], "stack", "reference", LFS_SOURCE_SITE_CURRENT());

        const auto& first_shape = tensors[0].shape();
        const auto first_device = tensors[0].device();
        const auto first_dtype = tensors[0].dtype();

        // Validate all tensors have same shape, device, and dtype
        for (size_t i = 1; i < tensors.size(); ++i) {
            tensor_contract::require_valid(
                tensors[i], "stack", "input", LFS_SOURCE_SITE_CURRENT());
            tensor_contract::require_shape(
                tensors[0], tensors[i], "stack", "reference", "input",
                LFS_SOURCE_SITE_CURRENT());
            tensor_contract::require_same_device(
                tensors[0], tensors[i], "stack", "reference", "input",
                LFS_SOURCE_SITE_CURRENT());
            tensor_contract::require_dtype(
                tensors[i], first_dtype, "stack", "input", LFS_SOURCE_SITE_CURRENT());
        }

        // Build output shape with new dimension inserted at 'dim'
        std::vector<size_t> new_dims = first_shape.dims();

        // Handle negative dimension
        if (dim < 0) {
            dim = first_shape.rank() + dim + 1;
        }

        LFS_ASSERT_MSG(dim >= 0 && dim <= static_cast<int>(first_shape.rank()),
                       "stack dimension is out of range");

        // Insert new dimension of size tensors.size() at position 'dim'
        new_dims.insert(new_dims.begin() + dim, tensors.size());

        auto result = Tensor::empty(TensorShape(new_dims), first_device, first_dtype);

        size_t elements_per_tensor = first_shape.elements();
        size_t bytes_per_tensor = elements_per_tensor * dtype_size(first_dtype);

        // Copy each input tensor into the corresponding slice of the output
        for (size_t i = 0; i < tensors.size(); ++i) {
            const int si = static_cast<int>(i);
            if (dim == 0 && first_device == Device::CUDA) {
                // dim=0: output slices are contiguous, use direct memcpy
                void* dst = static_cast<char*>(result.data_ptr()) + i * bytes_per_tensor;
                LFS_CUDA_CHECK_MSG(
                    cudaMemcpy(dst, tensors[i].data_ptr(), bytes_per_tensor,
                               cudaMemcpyDeviceToDevice),
                    "stack CUDA copy");
            } else if (dim == 0) {
                void* dst = static_cast<char*>(result.data_ptr()) + i * bytes_per_tensor;
                std::memcpy(dst, tensors[i].data_ptr(), bytes_per_tensor);
            } else {
                // dim>0: use strided scatter via slice (single kernel launch per tensor)
                auto dst_slice = result.slice(dim, si, si + 1);
                dst_slice.copy_from(tensors[i].unsqueeze(dim));
            }
        }

        return result;
    }

    // ============= OPTIMIZED CLAMP (FUSED VERSION) =============

    Tensor Tensor::clamp(float min_val, float max_val) const {
        LFS_ASSERT_MSG(is_valid(),
                       "clamp requires a valid tensor");
        LFS_ASSERT_MSG(dtype_ == DataType::Float32 || dtype_ == DataType::Int32,
                       "clamp currently supports only Float32 and Int32");
        LFS_ASSERT_MSG(!std::isnan(min_val) && !std::isnan(max_val) && min_val <= max_val,
                       "clamp bounds must not be NaN and must be ordered");

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
                tensor_ops::launch_clamp_fused(src, dst, min_val, max_val, numel(), result.stream());
            } else if (dtype_ == DataType::Int32) {
                // Fallback: copy then clamp for int
                LFS_CUDA_CHECK_MSG(
                    cudaMemcpy(result.data_, data_ptr(), bytes(), cudaMemcpyDeviceToDevice),
                    "Int32 clamp CUDA copy");
                tensor_ops::launch_clamp_scalar_int(result.ptr<int>(),
                                                    static_cast<int>(min_val),
                                                    static_cast<int>(max_val),
                                                    numel(), result.stream());
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
