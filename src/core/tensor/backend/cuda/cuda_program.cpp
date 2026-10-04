/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../../internal/expression_runtime.hpp"
#include "../gpu_program.hpp"
#include "core/logger.hpp"
#include <cuda.h>
#include <format>
#include <map>
#include <mutex>

namespace lfs::core::internal {
    namespace {
        struct Driver {
#ifdef _WIN32
            RuntimeLibrary library{"nvcuda.dll"};
#else
            RuntimeLibrary library{"libcuda.so.1"};
#endif
#define LFS_PROGRAM_DRIVER(name) decltype(&::name) name = reinterpret_cast<decltype(&::name)>(library.symbol(#name))
            LFS_PROGRAM_DRIVER(cuModuleLoadDataEx);
            LFS_PROGRAM_DRIVER(cuModuleGetFunction);
            LFS_PROGRAM_DRIVER(cuModuleGetGlobal_v2);
            LFS_PROGRAM_DRIVER(cuModuleUnload);
            LFS_PROGRAM_DRIVER(cuMemcpyDtoDAsync_v2);
            LFS_PROGRAM_DRIVER(cuLaunchKernel);
            LFS_PROGRAM_DRIVER(cuEventCreate);
            LFS_PROGRAM_DRIVER(cuEventRecord);
            LFS_PROGRAM_DRIVER(cuEventSynchronize);
            LFS_PROGRAM_DRIVER(cuEventDestroy_v2);
            LFS_PROGRAM_DRIVER(cuStreamWaitEvent);
            LFS_PROGRAM_DRIVER(cuCtxGetCurrent);
            LFS_PROGRAM_DRIVER(cuCtxPushCurrent_v2);
            LFS_PROGRAM_DRIVER(cuCtxPopCurrent_v2);
            LFS_PROGRAM_DRIVER(cuGetErrorString);
#undef LFS_PROGRAM_DRIVER
            void check(CUresult status) const {
                if (status == CUDA_SUCCESS)
                    return;
                const char* message = nullptr;
                cuGetErrorString(status, &message);
                throw Exception(make_error({.code = ErrorCode::Internal, .domain = ErrorDomain::CUDA, .detail = std::format("CUDA program: {} ({})", message ? message : "driver error", int(status)), .detection = LFS_SOURCE_SITE_CURRENT(), .native = NativeError{ErrorDomain::CUDA, int(status), message ? message : "CUDA_ERROR"}}));
            }
        };

        struct Kernel {
            Driver& driver;
            CUcontext context = nullptr;
            CUmodule module = nullptr;
            CUfunction function = nullptr;
            CUdeviceptr globals = 0;
            CUevent last = nullptr;
            bool submitted = false;
            ~Kernel() {
                if (!context || driver.cuCtxPushCurrent_v2(context) != CUDA_SUCCESS)
                    return;
                if (submitted) {
                    const auto status = driver.cuEventSynchronize(last);
                    if (status != CUDA_SUCCESS)
                        LOG_ERROR("CUDA program retirement failed: {}", int(status));
                }
                if (last)
                    driver.cuEventDestroy_v2(last);
                if (module)
                    driver.cuModuleUnload(module);
                CUcontext previous = nullptr;
                driver.cuCtxPopCurrent_v2(&previous);
            }
        };

        class Program final : public GpuProgram {
        public:
            explicit Program(std::span<const GpuKernelModule::Entry> entries) {
                const GpuBackendScope scope(GpuBackend::CUDA);
                // Loading a module is valid before the caller's first allocation.
                const auto initialize_device = Tensor::empty({1}, Device::GPU);
                for (const auto& entry : entries) {
                    if (entry.backend != GpuBackend::CUDA || entry.stage != GpuKernelModule::Stage::Compute)
                        continue;
                    auto kernel = std::make_unique<Kernel>(driver_);
                    driver_.check(driver_.cuCtxGetCurrent(&kernel->context));
                    if (!kernel->context)
                        throw Exception(make_error({.code = ErrorCode::FailedPrecondition, .domain = ErrorDomain::Tensor, .detail = "CUDA program loading requires an initialized tensor device (current context=null)", .detection = LFS_SOURCE_SITE_CURRENT()}));
                    driver_.check(driver_.cuModuleLoadDataEx(&kernel->module, entry.code.data(), 0, nullptr, nullptr));
                    const std::string name(entry.name);
                    driver_.check(driver_.cuModuleGetFunction(&kernel->function, kernel->module, name.c_str()));
                    size_t global_bytes = 0;
                    driver_.check(driver_.cuModuleGetGlobal_v2(&kernel->globals, &global_bytes, kernel->module, "SLANG_globalParams"));
                    LFS_ASSERT_MSG(global_bytes == sizeof(uint64_t), std::format("Slang CUDA global block must contain one parameter pointer: got {} bytes", global_bytes));
                    driver_.check(driver_.cuEventCreate(&kernel->last, CU_EVENT_DISABLE_TIMING));
                    kernels_.emplace(name, std::move(kernel));
                }
            }
            uint64_t address(const Tensor& tensor) override { return reinterpret_cast<uint64_t>(tensor.ptr<void>()); }
            bool supports_raster() const override { return false; }

            void dispatch(const GpuKernelModule::Dispatch& launch, const ProgramArguments& arguments) override {
                std::lock_guard lock(mutex_);
                auto& kernel = *kernels_.at(std::string(launch.function));
                const GpuBackendScope backend(GpuBackend::CUDA);
                const auto stream = getCurrentCUDAStream();
                CUcontext current = nullptr;
                driver_.check(driver_.cuCtxGetCurrent(&current));
                LFS_ASSERT_MSG(current == kernel.context, std::format("CUDA program context mismatch: current={}, owner={}", static_cast<void*>(current), static_cast<void*>(kernel.context)));
                for (const auto* tensor : arguments.reads)
                    tensor->sync_to_stream(stream);
                for (const auto* tensor : arguments.writes)
                    tensor->sync_to_stream(stream);
                // Slang emits one module-global constant pointer. Order updates
                // after the previous launch, including launches on other streams.
                if (kernel.submitted)
                    driver_.check(driver_.cuStreamWaitEvent(stream, kernel.last, 0));
                auto params = Tensor::from_blob(const_cast<std::byte*>(arguments.parameters.data()),
                                                {arguments.parameters.size()}, Device::CPU, DataType::UInt8)
                                  .to(Device::GPU);
                uint64_t pointer = address(params);
                auto global = Tensor::from_blob(&pointer, {sizeof(pointer)}, Device::CPU, DataType::UInt8).to(Device::GPU);
                driver_.check(driver_.cuMemcpyDtoDAsync_v2(kernel.globals, address(global), sizeof(pointer), stream));
                driver_.check(driver_.cuLaunchKernel(kernel.function, launch.groups[0], launch.groups[1], launch.groups[2],
                                                     launch.group[0], launch.group[1], launch.group[2], 0, stream, nullptr, nullptr));
                driver_.check(driver_.cuEventRecord(kernel.last, stream));
                kernel.submitted = true;
                for (const auto* tensor : arguments.reads)
                    tensor->record_stream(stream);
                for (const auto* tensor : arguments.writes)
                    const_cast<Tensor*>(tensor)->set_stream(stream);
                params.record_stream(stream);
                global.record_stream(stream);
            }
            void draw(std::span<const GpuKernelModule::Draw>, std::span<const ProgramArguments>) override {
                throw Exception(make_error({.code = ErrorCode::Unsupported, .domain = ErrorDomain::Tensor, .detail = "CUDA tensor programs support compute, not raster", .detection = LFS_SOURCE_SITE_CURRENT()}));
            }

        private:
            Driver driver_;
            std::mutex mutex_;
            std::map<std::string, std::unique_ptr<Kernel>> kernels_;
        };
    } // namespace
    std::unique_ptr<GpuProgram> make_cuda_program(std::span<const GpuKernelModule::Entry> entries) {
        return std::make_unique<Program>(entries);
    }
} // namespace lfs::core::internal
