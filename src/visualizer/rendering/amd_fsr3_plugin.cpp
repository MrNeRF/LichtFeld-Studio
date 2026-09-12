/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "amd_fsr3_plugin.hpp"

#include "core/executable_path.hpp"
#include "core/logger.hpp"
#include "core/user_paths.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <format>
#include <memory>
#include <optional>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace lfs::vis {
    namespace {
        constexpr std::string_view PLUGIN_ID = "amd-fsr3";
        constexpr std::string_view PROJECT_ID = "7fc73d74-f126-4146-b028-4bc1026e5c3b";
        constexpr std::string_view ENGINE_VERSION = "LichtFeld Studio";

#ifdef _WIN32
        constexpr const wchar_t* PLUGIN_FILENAME = L"lfs_scene_upscaler_amd_fsr3.dll";
        using NativeLibrary = HMODULE;
#else
        constexpr const char* PLUGIN_FILENAME = "liblfs_scene_upscaler_amd_fsr3.so";
        using NativeLibrary = void*;
#endif

        [[nodiscard]] std::vector<std::filesystem::path> pluginCandidates() {
            std::vector<std::filesystem::path> result;
            const auto append = [&result](const std::filesystem::path& root) {
                if (root.empty())
                    return;
                const auto candidate = root / "scene_upscalers" / "amd" / PLUGIN_FILENAME;
                if (std::ranges::find(result, candidate) == result.end())
                    result.push_back(candidate);
            };
            append(lfs::core::getExecutableDir());
            append(lfs::core::getLibDir());
            return result;
        }

        [[nodiscard]] std::string nativeLoadError() {
#ifdef _WIN32
            return std::format("Windows error {}", static_cast<unsigned long>(GetLastError()));
#else
            const char* const error = dlerror();
            return error != nullptr ? std::string(error) : std::string("unknown dlopen error");
#endif
        }

        [[nodiscard]] NativeLibrary loadLibrary(const std::filesystem::path& path) {
#ifdef _WIN32
            return LoadLibraryExW(path.c_str(),
                                  nullptr,
                                  LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR |
                                      LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
#else
            return dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
        }

        void unloadLibrary(const NativeLibrary library) {
            if (library == nullptr)
                return;
#ifdef _WIN32
            FreeLibrary(library);
#else
            dlclose(library);
#endif
        }

        [[nodiscard]] void* loadSymbol(const NativeLibrary library, const char* const name) {
#ifdef _WIN32
            return reinterpret_cast<void*>(GetProcAddress(library, name));
#else
            return dlsym(library, name);
#endif
        }

        int appendExtension(void* const user, const char* const name) {
            if (user == nullptr || name == nullptr || *name == '\0')
                return 0;
            try {
                auto& extensions = *static_cast<std::vector<std::string>*>(user);
                if (std::ranges::find(extensions, name) == extensions.end())
                    extensions.emplace_back(name);
                return 1;
            } catch (...) {
                // LFS-CENSUS-OK(empty-catch): the extension sink is a C ABI callback;
                // report rejection to the plugin instead of unwinding across the DLL.
                return 0;
            }
        }
    } // namespace

    struct AmdFsr3Plugin::Impl {
        struct OptimalSettingsCache {
            std::uint32_t output_width = 0;
            std::uint32_t output_height = 0;
            std::uint32_t quality = 0;
            LfsSceneUpscalerOptimalSettingsV1 settings{};
        };

        mutable std::mutex mutex;
        NativeLibrary library = nullptr;
        const LfsSceneUpscalerPluginApiV1* api = nullptr;
        void* plugin = nullptr;
        AmdFsr3PluginState state = AmdFsr3PluginState::Unprobed;
        std::filesystem::path library_path;
        std::wstring application_data_path;
        std::wstring plugin_directory;
        std::string diagnostic;
        std::optional<OptimalSettingsCache> optimal_settings_cache;
        bool loading_enabled = true;
        bool runtime_initialized = false;
        std::optional<std::thread::id> fidelityfx_thread_id;
        bool fidelityfx_thread_mismatch_warned = false;

        void noteFidelityFxCallerThreadLocked() {
            const auto thread_id = std::this_thread::get_id();
            if (!fidelityfx_thread_id) {
                fidelityfx_thread_id = thread_id;
                return;
            }
            if (*fidelityfx_thread_id != thread_id && !fidelityfx_thread_mismatch_warned) {
                fidelityfx_thread_mismatch_warned = true;
                LOG_WARN("AMD FidelityFX called from a different thread than the first "
                         "initializeRuntime/evaluate/createFeature caller; vendor calls are "
                         "serialized and expected to remain on one render thread");
            }
        }

        [[nodiscard]] std::string pluginErrorLocked() const {
            constexpr std::size_t MAX_ERROR_SIZE = 64 * 1024;
            if (api == nullptr || plugin == nullptr || api->last_error == nullptr)
                return {};
            const std::size_t required = api->last_error(plugin, nullptr, 0);
            if (required == 0 || required > MAX_ERROR_SIZE)
                return {};
            std::string result(required, '\0');
            const std::size_t written = api->last_error(plugin, result.data(), result.size());
            result.resize(std::min(written, result.size()));
            while (!result.empty() && result.back() == '\0')
                result.pop_back();
            return result;
        }

        void failLocked(const AmdFsr3PluginState failed_state, std::string reason) {
            state = failed_state;
            diagnostic = std::move(reason);
        }

        void destroyLocked() {
            if (api != nullptr && plugin != nullptr) {
                if (runtime_initialized)
                    api->shutdown_runtime(plugin);
                api->destroy(plugin);
            }
            plugin = nullptr;
            api = nullptr;
            runtime_initialized = false;
            optimal_settings_cache.reset();
            unloadLibrary(library);
            library = nullptr;
            library_path.clear();
            application_data_path.clear();
            plugin_directory.clear();
        }

        [[nodiscard]] bool probeLocked() {
            if (!loading_enabled) {
                state = AmdFsr3PluginState::DisabledBySafeMode;
                diagnostic = "optional scene-reconstruction plugins are disabled in safe mode";
                return false;
            }
            if (state == AmdFsr3PluginState::BootstrapReady ||
                state == AmdFsr3PluginState::RuntimeReady ||
                state == AmdFsr3PluginState::RuntimeMissing ||
                state == AmdFsr3PluginState::UnsupportedEnvironment ||
                state == AmdFsr3PluginState::RuntimeFailed) {
                return plugin != nullptr;
            }
            if (state == AmdFsr3PluginState::DisabledBySafeMode ||
                state == AmdFsr3PluginState::NotInstalled ||
                state == AmdFsr3PluginState::InvalidPlugin ||
                state == AmdFsr3PluginState::BootstrapFailed)
                return false;
            destroyLocked();

            std::error_code error;
            std::filesystem::path candidate;
            for (const auto& path : pluginCandidates()) {
                if (std::filesystem::is_regular_file(path, error) && !error) {
                    candidate = path;
                    break;
                }
                error.clear();
            }
            if (candidate.empty()) {
                state = AmdFsr3PluginState::NotInstalled;
                diagnostic = "AMD FSR 3.1 plugin is not installed";
                return false;
            }

            library = loadLibrary(candidate);
            if (library == nullptr) {
                failLocked(AmdFsr3PluginState::InvalidPlugin,
                           std::format("failed to load '{}': {}",
                                       candidate.string(),
                                       nativeLoadError()));
                LOG_WARN("Optional AMD FSR 3.1 plugin is invalid: {}", diagnostic);
                return false;
            }
            library_path = candidate;

            const auto get_api = reinterpret_cast<LfsSceneUpscalerGetPluginApiV1Fn>(
                loadSymbol(library, LFS_SCENE_UPSCALER_PLUGIN_ENTRY_V1));
            if (get_api == nullptr) {
                failLocked(AmdFsr3PluginState::InvalidPlugin,
                           "plugin entry point is missing");
                LOG_WARN("Optional AMD FSR 3.1 plugin '{}' is invalid: {}",
                         candidate.string(),
                         diagnostic);
                destroyLocked();
                return false;
            }
            api = get_api();
            if (!lfs_scene_upscaler_plugin_api_v1_complete(api) ||
                std::string_view(api->plugin_id) != PLUGIN_ID) {
                failLocked(AmdFsr3PluginState::InvalidPlugin,
                           "plugin ABI or identifier is incompatible");
                LOG_WARN("Optional AMD FSR 3.1 plugin '{}' is invalid: {}",
                         candidate.string(),
                         diagnostic);
                destroyLocked();
                return false;
            }

            const auto paths = lfs::core::UserPaths::resolve();
            if (!paths) {
                failLocked(AmdFsr3PluginState::InvalidPlugin,
                           "cannot resolve the FidelityFX cache directory");
                LOG_WARN("Optional AMD FSR 3.1 plugin '{}' cannot be initialized: {}",
                         candidate.string(),
                         diagnostic);
                destroyLocked();
                return false;
            }
            const auto data_path = paths->cacheDir() / "fidelityfx";
            application_data_path = data_path.wstring();
            plugin_directory = candidate.parent_path().wstring();
            const LfsSceneUpscalerBootstrapConfigV1 config{
                .struct_size = sizeof(LfsSceneUpscalerBootstrapConfigV1),
                .project_id = PROJECT_ID.data(),
                .engine_version = ENGINE_VERSION.data(),
                .application_data_path = application_data_path.c_str(),
                .plugin_directory = plugin_directory.c_str(),
            };
            plugin = api->create(&config);
            if (plugin == nullptr) {
                failLocked(AmdFsr3PluginState::InvalidPlugin,
                           "plugin bootstrap context creation failed");
                LOG_WARN("Optional AMD FSR 3.1 plugin '{}' cannot be initialized: {}",
                         candidate.string(),
                         diagnostic);
                destroyLocked();
                return false;
            }
            state = AmdFsr3PluginState::BootstrapReady;
            diagnostic.clear();
            LOG_INFO("Discovered optional scene-reconstruction plugin '{}' at {}",
                     api->display_name,
                     candidate.string());
            return true;
        }

        [[nodiscard]] std::vector<std::string> extensionsLocked(
            const bool device_extensions,
            const VkInstance instance,
            const VkPhysicalDevice physical_device) {
            std::vector<std::string> extensions;
            if (!probeLocked())
                return extensions;
            const LfsSceneUpscalerExtensionSink sink{
                .struct_size = sizeof(LfsSceneUpscalerExtensionSink),
                .user = &extensions,
                .append = &appendExtension,
            };
            const auto result = device_extensions
                                    ? api->required_device_extensions(
                                          plugin, instance, physical_device, &sink)
                                    : api->required_instance_extensions(plugin, &sink);
            if (result != LFS_SCENE_UPSCALER_PLUGIN_OK) {
                const auto plugin_error = pluginErrorLocked();
                failLocked(AmdFsr3PluginState::BootstrapFailed,
                           plugin_error.empty()
                               ? "plugin could not report required Vulkan extensions"
                               : plugin_error);
                LOG_WARN("AMD FSR 3.1 bootstrap unavailable: {}", diagnostic);
                extensions.clear();
            }
            return extensions;
        }
    };

    AmdFsr3Plugin& AmdFsr3Plugin::instance() {
        static AmdFsr3Plugin plugin;
        return plugin;
    }

    AmdFsr3Plugin::AmdFsr3Plugin() : impl_(new Impl) {}

    AmdFsr3Plugin::~AmdFsr3Plugin() {
        shutdown();
        delete impl_;
    }

    void AmdFsr3Plugin::configure(const bool loading_enabled) {
        std::scoped_lock lock(impl_->mutex);
        if (impl_->loading_enabled == loading_enabled &&
            impl_->state != AmdFsr3PluginState::Unprobed)
            return;
        impl_->destroyLocked();
        impl_->loading_enabled = loading_enabled;
        impl_->state = loading_enabled ? AmdFsr3PluginState::Unprobed
                                       : AmdFsr3PluginState::DisabledBySafeMode;
        impl_->diagnostic = loading_enabled
                                ? std::string{}
                                : "optional scene-reconstruction plugins are disabled in safe mode";
    }

    bool AmdFsr3Plugin::probe() {
        std::scoped_lock lock(impl_->mutex);
        return impl_->probeLocked();
    }

    bool AmdFsr3Plugin::available() { return probe(); }

    AmdFsr3PluginState AmdFsr3Plugin::state() const {
        std::scoped_lock lock(impl_->mutex);
        return impl_->state;
    }

    std::string AmdFsr3Plugin::diagnostic() const {
        std::scoped_lock lock(impl_->mutex);
        return impl_->diagnostic;
    }

    std::filesystem::path AmdFsr3Plugin::libraryPath() const {
        std::scoped_lock lock(impl_->mutex);
        return impl_->library_path;
    }

    std::vector<std::string> AmdFsr3Plugin::requiredInstanceExtensions() {
        std::scoped_lock lock(impl_->mutex);
        return impl_->extensionsLocked(false, VK_NULL_HANDLE, VK_NULL_HANDLE);
    }

    std::vector<std::string> AmdFsr3Plugin::requiredDeviceExtensions(
        const VkInstance instance,
        const VkPhysicalDevice physical_device) {
        std::scoped_lock lock(impl_->mutex);
        return impl_->extensionsLocked(true, instance, physical_device);
    }

    void AmdFsr3Plugin::markBootstrapFailed(std::string reason) {
        std::scoped_lock lock(impl_->mutex);
        impl_->failLocked(AmdFsr3PluginState::BootstrapFailed, std::move(reason));
        LOG_WARN("AMD FSR 3.1 bootstrap unavailable: {}", impl_->diagnostic);
    }

    bool AmdFsr3Plugin::initializeRuntime(const LfsSceneUpscalerRuntimeConfigV1& config) {
        std::scoped_lock lock(impl_->mutex);
        impl_->noteFidelityFxCallerThreadLocked();
        if (!impl_->probeLocked())
            return false;
        if (impl_->runtime_initialized)
            return true;
        std::error_code error;
        std::filesystem::create_directories(
            std::filesystem::path(impl_->application_data_path), error);
        if (error) {
            impl_->failLocked(
                AmdFsr3PluginState::RuntimeFailed,
                std::format("cannot create the FidelityFX cache directory: {}",
                            error.message()));
            return false;
        }
        const auto result = impl_->api->initialize_runtime(impl_->plugin, &config);
        if (result != LFS_SCENE_UPSCALER_PLUGIN_OK) {
            const auto failed_state = [&] {
                switch (result) {
                case LFS_SCENE_UPSCALER_PLUGIN_UNAVAILABLE:
                    return AmdFsr3PluginState::RuntimeMissing;
                case LFS_SCENE_UPSCALER_PLUGIN_UNSUPPORTED_DEVICE:
                    return AmdFsr3PluginState::UnsupportedEnvironment;
                default:
                    return AmdFsr3PluginState::RuntimeFailed;
                }
            }();
            impl_->failLocked(failed_state, impl_->pluginErrorLocked());
            if (impl_->diagnostic.empty())
                impl_->diagnostic = "FidelityFX runtime initialization failed";
            return false;
        }
        impl_->runtime_initialized = true;
        impl_->optimal_settings_cache.reset();
        impl_->state = AmdFsr3PluginState::RuntimeReady;
        impl_->diagnostic.clear();
        return true;
    }

    std::optional<LfsSceneUpscalerOptimalSettingsV1> AmdFsr3Plugin::optimalSettings(
        const std::uint32_t output_width,
        const std::uint32_t output_height,
        const std::uint32_t quality) {
        std::scoped_lock lock(impl_->mutex);
        if (!impl_->runtime_initialized)
            return std::nullopt;
        if (impl_->optimal_settings_cache &&
            impl_->optimal_settings_cache->output_width == output_width &&
            impl_->optimal_settings_cache->output_height == output_height &&
            impl_->optimal_settings_cache->quality == quality) {
            return impl_->optimal_settings_cache->settings;
        }
        LfsSceneUpscalerOptimalSettingsV1 settings{
            .struct_size = sizeof(LfsSceneUpscalerOptimalSettingsV1)};
        if (impl_->api->optimal_settings(
                impl_->plugin, output_width, output_height, quality, &settings) !=
            LFS_SCENE_UPSCALER_PLUGIN_OK) {
            impl_->diagnostic = impl_->pluginErrorLocked();
            return std::nullopt;
        }
        impl_->optimal_settings_cache = Impl::OptimalSettingsCache{
            .output_width = output_width,
            .output_height = output_height,
            .quality = quality,
            .settings = settings,
        };
        return settings;
    }

    bool AmdFsr3Plugin::createFeature(
        const VkCommandBuffer command_buffer,
        const LfsSceneUpscalerFeatureConfigV1& config) {
        std::scoped_lock lock(impl_->mutex);
        impl_->noteFidelityFxCallerThreadLocked();
        if (!impl_->runtime_initialized)
            return false;
        if (impl_->api->create_feature(impl_->plugin, command_buffer, &config) !=
            LFS_SCENE_UPSCALER_PLUGIN_OK) {
            impl_->diagnostic = impl_->pluginErrorLocked();
            return false;
        }
        return true;
    }

    bool AmdFsr3Plugin::evaluate(const LfsSceneUpscalerEvaluateV1& evaluation) {
        std::scoped_lock lock(impl_->mutex);
        impl_->noteFidelityFxCallerThreadLocked();
        if (!impl_->runtime_initialized)
            return false;
        if (impl_->api->evaluate(impl_->plugin, &evaluation) !=
            LFS_SCENE_UPSCALER_PLUGIN_OK) {
            impl_->diagnostic = impl_->pluginErrorLocked();
            return false;
        }
        return true;
    }

    void AmdFsr3Plugin::releaseFeature(const std::uint32_t view) {
        std::scoped_lock lock(impl_->mutex);
        if (impl_->runtime_initialized)
            impl_->api->release_feature(impl_->plugin, view);
    }

    void AmdFsr3Plugin::shutdownRuntime() {
        std::scoped_lock lock(impl_->mutex);
        if (!impl_->runtime_initialized)
            return;
        impl_->api->shutdown_runtime(impl_->plugin);
        impl_->runtime_initialized = false;
        impl_->optimal_settings_cache.reset();
        impl_->state = AmdFsr3PluginState::BootstrapReady;
    }

    void AmdFsr3Plugin::shutdown() {
        std::scoped_lock lock(impl_->mutex);
        impl_->destroyLocked();
        impl_->state = impl_->loading_enabled ? AmdFsr3PluginState::Unprobed
                                              : AmdFsr3PluginState::DisabledBySafeMode;
    }

    void configureAmdFsr3PluginLoading(const bool enabled) {
        AmdFsr3Plugin::instance().configure(enabled);
    }

    bool amdFsr3PluginAvailable() { return AmdFsr3Plugin::instance().available(); }

} // namespace lfs::vis
