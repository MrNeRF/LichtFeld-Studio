/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor_cuda_interop.hpp"

#include "core/cuda_vulkan_interop.hpp"
#include "core/tensor_backend.hpp"
#include "../../internal/tensor_impl.hpp"
#include "../native_pressure.hpp"
#include "../random_state.hpp"
#include "../reduction_tuning.hpp"
#include "../target_lifetime.hpp"
#include "core/pinned_allocator_stats.hpp"

#include <stdexcept>

namespace lfs::core {
    int selected_cuda_device() { return 0; }

    namespace {
        [[noreturn]] void throw_cuda_unavailable() {
            throw std::runtime_error("CUDA support is not compiled into this build");
        }
    } // namespace

    cudaStream_t getCurrentCUDAStream() { return nullptr; }

    void setCurrentCUDAStream(const cudaStream_t stream) {
        if (stream)
            throw_cuda_unavailable();
    }

    void waitForCUDAStream(const cudaStream_t execution_stream, const cudaStream_t dependency_stream) {
        if (execution_stream || dependency_stream)
            throw_cuda_unavailable();
    }

    cudaStream_t prepare_inputs_for_stream(
        const std::initializer_list<const Tensor*> inputs,
        const std::optional<cudaStream_t> execution_stream) {
        if (execution_stream.value_or(nullptr))
            throw_cuda_unavailable();
        for (const Tensor* input : inputs) {
            if (!input || !input->is_valid())
                throw TensorError("stream preparation requires valid tensor inputs");
            if (input->device() == Device::GPU &&
                internal::gpu_backend_tag(*input) == GpuBackend::CUDA)
                throw_cuda_unavailable();
        }
        return nullptr;
    }

    PinnedAllocatorStats pinned_allocator_stats() { return {}; }
    void* allocate_cuda_storage(size_t, cudaStream_t, CudaStorageMode, const char*, const char*) { throw_cuda_unavailable(); }
    bool is_stream_retired(cudaStream_t) noexcept { return false; }
    size_t cuda_allocation_size(size_t) { return 0; }
    void trim_cuda_memory_pool() {}
    size_t cuda_bucket_cached_bytes() noexcept { return 0; }
    void safe_cuda_pool_deallocate(void*, cudaStream_t) noexcept {}
    bool tensor_uses_cuda_storage(const Tensor&) { return false; }

    TensorCudaStream::TensorCudaStream() { throw_cuda_unavailable(); }
    TensorCudaStream::~TensorCudaStream() = default;
} // namespace lfs::core

namespace lfs::core::cuda {
    ExternalMemoryImportScope::ExternalMemoryImportScope() noexcept {}
    ExternalMemoryImportScope::~ExternalMemoryImportScope() noexcept {}
    bool ExternalMemoryImportScope::active() noexcept { return false; }
} // namespace lfs::core::cuda

namespace lfs::core::internal {
    void record_pressure_breadcrumb() {}
    std::unique_ptr<NativeRandomState> make_native_random_state(uint64_t, const char* message) { throw TensorError(message); }
    void observe_native_target(cudaStream_t) noexcept {}
    ReductionTuningPath reduction_tuning_override() noexcept { return ReductionTuningPath::None; }
    bool prefer_strided_reduction(size_t, size_t, size_t) noexcept { return true; }
    void record_reduction_tuning_path(ReductionTuningPath) noexcept {}
} // namespace lfs::core::internal
