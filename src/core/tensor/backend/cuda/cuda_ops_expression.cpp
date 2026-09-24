/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../../internal/expression_emitter.hpp"
#include "../facade_trace.hpp"
#include "../gpu_backend_ops.hpp"
#include "core/logger.hpp"

#include <algorithm>
#include <cuda.h>
#include <format>
#include <stdexcept>

namespace lfs::core::internal {
    namespace {
        struct CudaDriver {
#ifdef _WIN32
            RuntimeLibrary library{"nvcuda.dll"};
#else
            RuntimeLibrary library{"libcuda.so.1"};
#endif
#define LFS_DRIVER(name) decltype(&::name) name = reinterpret_cast<decltype(&::name)>(library.symbol(#name))
            LFS_DRIVER(cuCtxGetCurrent);
            LFS_DRIVER(cuCtxGetDevice);
            LFS_DRIVER(cuCtxPushCurrent_v2);
            LFS_DRIVER(cuCtxPopCurrent_v2);
            LFS_DRIVER(cuDeviceGetAttribute);
            LFS_DRIVER(cuDeviceGetUuid);
            LFS_DRIVER(cuDriverGetVersion);
            LFS_DRIVER(cuModuleLoadDataEx);
            LFS_DRIVER(cuLinkCreate_v2);
            LFS_DRIVER(cuLinkAddData_v2);
            LFS_DRIVER(cuLinkComplete);
            LFS_DRIVER(cuLinkDestroy);
            LFS_DRIVER(cuModuleGetFunction);
            LFS_DRIVER(cuModuleUnload);
            LFS_DRIVER(cuLaunchKernel);
            LFS_DRIVER(cuStreamIsCapturing);
            LFS_DRIVER(cuGetErrorString);
#undef LFS_DRIVER

            void check(const CUresult result) const {
                if (result == CUDA_SUCCESS)
                    return;
                const char* message = nullptr;
                cuGetErrorString(result, &message);
                throw std::runtime_error(std::string("CUDA expression: ") + (message ? message : "driver failure"));
            }
        };

        std::vector<char> link_ptx(CudaDriver& driver, std::string ptx) {
            char error[8192]{};
            CUjit_option options[]{CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES};
            void* values[]{error, reinterpret_cast<void*>(sizeof(error))};
            struct Link {
                CudaDriver& driver;
                CUlinkState state = nullptr;
                ~Link() {
                    if (state)
                        driver.cuLinkDestroy(state);
                }
            } link{driver};
            driver.check(driver.cuLinkCreate_v2(2, options, values, &link.state));
            const auto added = driver.cuLinkAddData_v2(link.state, CU_JIT_INPUT_PTX, ptx.data(), ptx.size() + 1,
                                                       "tensor-expression", 0, nullptr, nullptr);
            if (added != CUDA_SUCCESS)
                throw std::runtime_error(std::string("PTX expression link: ") + error);
            void* cubin = nullptr;
            size_t bytes = 0;
            driver.check(driver.cuLinkComplete(link.state, &cubin, &bytes));
            const auto* first = static_cast<const char*>(cubin);
            return {first, first + bytes};
        }

        // Evicted modules may still have queued launches on any stream. They are
        // unloaded only after a later device-wide synchronization.
        struct RetiredModules {
            std::mutex mutex;
            std::vector<std::pair<CUcontext, CUmodule>> modules;
        };

        struct CudaKernel final : CompiledExpression {
            CUcontext context;
            std::shared_ptr<RetiredModules> retired;
            CUmodule module = nullptr;
            CUfunction function = nullptr;
            CudaKernel(CUcontext owner, std::shared_ptr<RetiredModules> retired_modules)
                : context(owner),
                  retired(std::move(retired_modules)) {}
            ~CudaKernel() override {
                if (!module)
                    return;
                std::lock_guard lock(retired->mutex);
                retired->modules.emplace_back(context, module);
            }
        };

        struct DeviceCache {
            CUcontext context = nullptr;
            int architecture = 0;
            std::string identity;
            ExpressionCache cache;
        };

        struct CudaCompiler {
            CudaDriver driver;
            std::shared_ptr<RetiredModules> retired = std::make_shared<RetiredModules>();
            std::mutex mutex;
            std::unordered_map<CUcontext, std::unique_ptr<DeviceCache>> devices;

            DeviceCache& device() {
                CUcontext context = nullptr;
                driver.check(driver.cuCtxGetCurrent(&context));
                if (!context)
                    throw std::runtime_error("CUDA expressions require an initialized tensor device");
                std::lock_guard lock(mutex);
                auto& result = devices[context];
                if (!result) {
                    auto created = std::make_unique<DeviceCache>();
                    CUdevice device;
                    int major, minor, driver_version;
                    CUuuid uuid;
                    driver.check(driver.cuCtxGetDevice(&device));
                    driver.check(driver.cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
                    driver.check(driver.cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
                    driver.check(driver.cuDeviceGetUuid(&uuid, device));
                    driver.check(driver.cuDriverGetVersion(&driver_version));
                    created->context = context;
                    created->architecture = major * 10 + minor;
                    created->identity = std::format("cuda-ptx:{}:{}:", created->architecture, driver_version);
                    created->identity.append(uuid.bytes, sizeof(uuid.bytes));
                    result = std::move(created);
                }
                return *result;
            }
        };

        CudaCompiler& compiler() {
            static CudaCompiler instance;
            return instance;
        }
        std::shared_ptr<CudaKernel> compiled_kernel(CudaCompiler& runtime, DeviceCache& device,
                                                    const ExpressionProgram& program,
                                                    const ExpressionSignature& signature) {
            const auto key = device.identity + expression_key(program, signature);
            return std::static_pointer_cast<CudaKernel>(device.cache.get(key, [&] { return link_ptx(runtime.driver, expression_ptx(program, signature)); }, [&](const std::span<const char> artifact) {
                auto result = std::make_shared<CudaKernel>(device.context, runtime.retired);
                runtime.driver.check(runtime.driver.cuModuleLoadDataEx(&result->module, artifact.data(), 0, nullptr, nullptr));
                runtime.driver.check(runtime.driver.cuModuleGetFunction(&result->function, result->module, "expression_main"));
                return result; }));
        }
    } // namespace

    void synchronize_and_unload_cuda_expressions(const std::function<void()>& synchronize) {
        auto& runtime = compiler();
        std::vector<std::pair<CUcontext, CUmodule>> modules;
        {
            std::lock_guard lock(runtime.retired->mutex);
            modules.swap(runtime.retired->modules);
        }
        synchronize();
        for (const auto& [context, module] : modules) {
            if (runtime.driver.cuCtxPushCurrent_v2(context) != CUDA_SUCCESS)
                continue;
            const auto unloaded = runtime.driver.cuModuleUnload(module);
            if (unloaded != CUDA_SUCCESS && unloaded != CUDA_ERROR_DEINITIALIZED)
                LOG_WARN("CUDA expression module unload failed: {}", int(unloaded));
            CUcontext popped = nullptr;
            runtime.driver.cuCtxPopCurrent_v2(&popped);
        }
    }

    ExpressionCache& cuda_expression_cache() {
        return compiler().device().cache;
    }

    void CudaBackendOps::compiled_expression(const ExpressionLaunch& launch, const ExecContext context) {
        LFS_FACADE_TRACE(compiled_expression);
        auto& runtime = compiler();
        auto& device = runtime.device();
        if (!launch.prepare_only) {
            CUstreamCaptureStatus capture;
            runtime.driver.check(runtime.driver.cuStreamIsCapturing(context.cuda_stream, &capture));
            if (capture != CU_STREAM_CAPTURE_STATUS_NONE)
                throw std::runtime_error("Compiled tensor expressions cannot be recorded in a CUDA graph");
        }
        auto kernel = compiled_kernel(runtime, device, *launch.program, launch.signature);
        if (launch.prepare_only)
            return;
        const uint32_t work = launch.count;
        void* args[]{const_cast<uint32_t*>(launch.arguments.data())};
        runtime.driver.check(runtime.driver.cuLaunchKernel(kernel->function, std::min(65535u, (work + 255) / 256),
                                                           1, 1, 256, 1, 1, 0, context.cuda_stream, args, nullptr));
    }
} // namespace lfs::core::internal
