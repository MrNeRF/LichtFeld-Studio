/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "session.hpp"
#include <cstdio>
#include <mutex>
#include <shared_mutex>
#include <string>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace lfs::training::hip {
    namespace {
        void check(int status, const LfsHipError& error) {
            if (status != LFS_HIP_TRAIN_OK)
                throw TrainingError(status, error);
        }
        struct Module {
#ifdef _WIN32
            HMODULE handle = nullptr;
#else
            void* handle = nullptr;
#endif
            explicit Module(const std::filesystem::path& path) {
                // Never resolve an optional backend DLL through the current directory.
                if (!path.is_absolute())
                    throw std::invalid_argument("HIP training module requires an absolute path");
#ifdef _WIN32
                handle = LoadLibraryExW(path.c_str(), nullptr,
                                        LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
                if (!handle)
                    throw std::runtime_error("Cannot load HIP training module; Windows error " + std::to_string(GetLastError()));
#else
                handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
                if (!handle)
                    throw std::runtime_error(std::string("Cannot load HIP training module: ") + dlerror());
#endif
            }
            ~Module() {
                if (!handle)
                    return;
#ifdef _WIN32
                FreeLibrary(handle);
#else
                dlclose(handle);
#endif
            }
            template <class F>
            F symbol(const char* name) {
#ifdef _WIN32
                auto address = GetProcAddress(handle, name);
#else
                auto address = dlsym(handle, name);
#endif
                if (!address)
                    throw std::runtime_error(std::string("Missing HIP training entry point: ") + name);
                return reinterpret_cast<F>(address);
            }
        };
    } // namespace

    TrainingError::TrainingError(int status, const LfsHipError& error)
        : std::runtime_error(error.message), status(status), native_code(error.native_code) {}

    struct HostTargets::Impl {
        Module module;
        decltype(&::lfs_hip_train_host_alloc) allocate;
        decltype(&::lfs_hip_train_host_free) release;
        int device;
        size_t budget, used = 0;
        mutable std::recursive_mutex mutex;
        Impl(const std::filesystem::path& path, int ordinal, size_t limit)
            : module(path),
              allocate(module.symbol<decltype(allocate)>("lfs_hip_train_host_alloc")),
              release(module.symbol<decltype(release)>("lfs_hip_train_host_free")),
              device(ordinal), budget(limit) {}
    };
    HostTargets::HostTargets(const std::filesystem::path& module, int device, size_t budget)
        : impl_(std::make_shared<Impl>(module, device, budget)) {}
    size_t HostTargets::allocated_bytes() const {
        std::lock_guard lock(impl_->mutex);
        return impl_->used;
    }
    std::shared_ptr<void> HostTargets::allocate(size_t bytes) {
        auto state = impl_;
        std::lock_guard lock(state->mutex);
        if (!bytes || bytes > state->budget - state->used) {
            std::fprintf(stderr, "[warning] HIP host target fallback to ordinary RAM: pinned budget exceeded (%zu bytes requested).\n", bytes);
            return {};
        }
        void* pointer = nullptr;
        LfsHipError error{};
        if (state->allocate(state->device, bytes, &pointer, &error) != LFS_HIP_TRAIN_OK) {
            std::fprintf(stderr, "[warning] HIP host target fallback to ordinary RAM: %s.\n", error.message);
            return {};
        }
        state->used += bytes;
        // shared_ptr invokes the deleter if its control-block allocation fails.
        // A recursive reservation lock also covers that failure path.
        try {
            auto owner = std::shared_ptr<void>(pointer, [state, bytes](void* data) {
                std::lock_guard lock(state->mutex);
                LfsHipError error{};
                if (state->release(state->device, data, &error) == LFS_HIP_TRAIN_OK) {
                    state->used -= bytes;
                } else {
                    std::fprintf(stderr, "[error] HIP host target release failed; retaining module: %s\n", error.message);
                    state->module.handle = nullptr;
                }
            });
            return owner;
        } catch (const std::bad_alloc&) {
            // shared_ptr already invoked the deleter and restored the budget.
            std::fprintf(stderr, "[warning] HIP host target fallback to ordinary RAM: owner allocation failed.\n");
            return {};
        }
    }

    struct Session::Impl {
        Module module;
#define ENTRY(name) decltype(&::name) name = module.symbol<decltype(&::name)>(#name)
        ENTRY(lfs_hip_train_abi);
        ENTRY(lfs_hip_train_create);
        ENTRY(lfs_hip_train_create_shared);
        ENTRY(lfs_hip_train_create_shared_layout);
        ENTRY(lfs_hip_train_attach_shared_layout);
        ENTRY(lfs_hip_train_attach_device_layout);
        ENTRY(lfs_hip_train_destroy);
        ENTRY(lfs_hip_train_set_frame);
        ENTRY(lfs_hip_train_set_rasterizer);
        ENTRY(lfs_hip_train_step);
        ENTRY(lfs_hip_train_render);
        ENTRY(lfs_hip_train_read_parameter);
        ENTRY(lfs_hip_train_info);
        ENTRY(lfs_hip_train_set_budget);
        ENTRY(lfs_hip_train_set_loss);
        ENTRY(lfs_hip_train_loss);
        ENTRY(lfs_hip_train_set_sh);
        ENTRY(lfs_hip_train_set_sh_degree);
        ENTRY(lfs_hip_train_set_cancel);
#undef ENTRY
        LfsHipTraining* handle = nullptr;
        std::shared_ptr<void> device_owner;
        std::function<void()> validate_model;
        std::function<void(int)> invalidate_model;
        std::shared_mutex lifetime;
        std::mutex operations;

        Impl(const std::filesystem::path& path, const LfsHipTrainConfig& config, const LfsHipTrainModel& model,
             const LfsHipTrainSharedModel* shared = nullptr, const LfsHipTrainModelLayout* layout = nullptr, bool attach = false, const LfsHipTrainDeviceModel* device = nullptr, std::shared_ptr<void> owner = {}, std::function<void()> validate = {}, std::function<void(int)> invalidate = {})
            : module(path), device_owner(std::move(owner)), validate_model(std::move(validate)), invalidate_model(std::move(invalidate)) {
            if (lfs_hip_train_abi() != 1)
                throw std::runtime_error("Incompatible HIP training ABI");
            if (device && !device_owner)
                throw std::invalid_argument("Native HIP model attachment requires an allocation owner");
            if (validate_model)
                validate_model();
            LfsHipError error{};
            const int status = device   ? lfs_hip_train_attach_device_layout(&config, device, layout, &handle, &error)
                               : attach ? lfs_hip_train_attach_shared_layout(&config, shared, layout, &handle, &error)
                               : layout ? lfs_hip_train_create_shared_layout(&config, &model, shared, layout, &handle, &error)
                               : shared ? lfs_hip_train_create_shared(&config, &model, shared, &handle, &error)
                                        : lfs_hip_train_create(&config, &model, &handle, &error);
            if (status != LFS_HIP_TRAIN_OK && invalidate_model)
                invalidate_model(status);
            check(status, error);
        }
        ~Impl() {
            if (!handle)
                return;
            LfsHipError error{};
            if (lfs_hip_train_destroy(handle, &error) != LFS_HIP_TRAIN_OK) {
                std::fprintf(stderr, "[error] HIP training teardown failed; module retained: %s\n", error.message);
                module.handle = nullptr; // Retain the OS reference with the live session.
            }
        }
        template <class F>
        void invoke(F&& operation) {
            std::shared_lock alive(lifetime);
            std::lock_guard serialized(operations);
            require_open();
            if (validate_model)
                validate_model();
            LfsHipError error{};
            const int status = operation(&error);
            if (status != LFS_HIP_TRAIN_OK && invalidate_model)
                invalidate_model(status);
            check(status, error);
        }
        void require_open() const {
            if (!handle)
                throw std::logic_error("HIP training session is closed");
        }
    };

    Session::Session(const std::filesystem::path& path, const LfsHipTrainConfig& config, const LfsHipTrainModel& model)
        : impl_(std::make_unique<Impl>(path, config, model)) {}
    Session::Session(const std::filesystem::path& path, const LfsHipTrainConfig& config, const LfsHipTrainModel& model,
                     const LfsHipTrainSharedModel& shared)
        : impl_(std::make_unique<Impl>(path, config, model, &shared)) {}
    Session::Session(const std::filesystem::path& path, const LfsHipTrainConfig& config, const LfsHipTrainModel& model,
                     const LfsHipTrainSharedModel& shared, const LfsHipTrainModelLayout& layout)
        : impl_(std::make_unique<Impl>(path, config, model, &shared, &layout)) {}
    Session::Session(const std::filesystem::path& path, const LfsHipTrainConfig& config,
                     const LfsHipTrainSharedModel& shared, const LfsHipTrainModelLayout& layout)
        : impl_(std::make_unique<Impl>(path, config, LfsHipTrainModel{}, &shared, &layout, true)) {}
    Session::Session(const std::filesystem::path& path, const LfsHipTrainConfig& config,
                     const LfsHipTrainDeviceModel& device, const LfsHipTrainModelLayout& layout, std::shared_ptr<void> owner, std::function<void()> validate, std::function<void(int)> invalidate)
        : impl_(std::make_unique<Impl>(path, config, LfsHipTrainModel{}, nullptr, &layout, true, &device, std::move(owner), std::move(validate), std::move(invalidate))) {}
    LfsHipTrainModelLayout Session::model_layout(const std::filesystem::path& path, uint64_t count) {
        Module module(path);
        if (module.symbol<decltype(&::lfs_hip_train_abi)>("lfs_hip_train_abi")() != 1)
            throw std::runtime_error("Incompatible HIP training ABI");
        LfsHipTrainModelLayout result{};
        result.struct_bytes = sizeof(result);
        LfsHipError error{};
        const int status = module.symbol<decltype(&::lfs_hip_train_model_layout)>("lfs_hip_train_model_layout")(count, &result, &error);
        check(status, error);
        return result;
    }
    Session::~Session() {
        try {
            close();
        } catch (const std::exception& error) { /* LFS-CENSUS-OK(empty-catch): Report failed teardown and retain unsafe native owners. */
            std::fprintf(stderr, "[error] HIP training teardown failed; session, module and model owner retained: %s\n", error.what());
            (void)impl_.release();
        }
    }
    void Session::set_frame(const LfsHipTrainFrame& frame) {
        impl_->invoke([&](auto* error) { return impl_->lfs_hip_train_set_frame(impl_->handle, &frame, error); });
    }
    void Session::set_rasterizer(uint32_t kind) {
        impl_->invoke([&](auto* error) { return impl_->lfs_hip_train_set_rasterizer(impl_->handle, kind, error); });
    }
    void Session::attach_resident_sh(const LfsHipTrainResidentSh& descriptor) {
        impl_->invoke([&](auto* error) {
            const auto function = impl_->module.symbol<decltype(&::lfs_hip_train_attach_resident_sh)>("lfs_hip_train_attach_resident_sh");
            return function(impl_->handle, &descriptor, error);
        });
    }
    void Session::set_optimizer(uint32_t kind) {
        impl_->invoke([&](auto* error) {
            const auto function = impl_->module.symbol<decltype(&::lfs_hip_train_set_optimizer)>("lfs_hip_train_set_optimizer");
            return function(impl_->handle, kind, error);
        });
    }
    void Session::step(const LfsHipTrainRates& rates) {
        impl_->invoke([&](auto* error) { return impl_->lfs_hip_train_step(impl_->handle, &rates, error); });
    }
    void Session::render(std::span<float> output) {
        impl_->invoke([&](auto* error) { return impl_->lfs_hip_train_render(impl_->handle, output.data(), output.size(), error); });
    }
    void Session::read_parameter(uint32_t family, std::span<float> output) {
        impl_->invoke([&](auto* error) { return impl_->lfs_hip_train_read_parameter(impl_->handle, family, output.data(), output.size(), error); });
    }
    LfsHipTrainInfo Session::info() {
        LfsHipTrainInfo result{};
        impl_->invoke([&](auto* error) { return impl_->lfs_hip_train_info(impl_->handle, &result, error); });
        return result;
    }
    void Session::set_budget(uint64_t bytes) {
        impl_->invoke([&](auto* error) { return impl_->lfs_hip_train_set_budget(impl_->handle, bytes, error); });
    }
    void Session::set_cancel(bool cancelled) {
        std::shared_lock alive(impl_->lifetime);
        impl_->require_open();
        impl_->lfs_hip_train_set_cancel(impl_->handle, cancelled);
    }
    void Session::set_loss(const LfsHipTrainLoss& loss) {
        impl_->invoke([&](auto* error) { return impl_->lfs_hip_train_set_loss(impl_->handle, &loss, error); });
    }
    float Session::loss() {
        float result = 0;
        impl_->invoke([&](auto* error) { return impl_->lfs_hip_train_loss(impl_->handle, &result, error); });
        return result;
    }
    void Session::set_sh(const LfsHipTrainSh& config) {
        impl_->invoke([&](auto* error) { return impl_->lfs_hip_train_set_sh(impl_->handle, &config, error); });
    }
    void Session::set_sh_degree(uint32_t degree, float rate) {
        impl_->invoke([&](auto* error) { return impl_->lfs_hip_train_set_sh_degree(impl_->handle, degree, rate, error); });
    }
    void Session::close() {
        std::unique_lock alive(impl_->lifetime);
        if (!impl_->handle)
            return;
        LfsHipError error{};
        check(impl_->lfs_hip_train_destroy(impl_->handle, &error), error);
        impl_->handle = nullptr;
        impl_->invalidate_model = {};
        impl_->validate_model = {};
        impl_->device_owner.reset();
    }
} // namespace lfs::training::hip
