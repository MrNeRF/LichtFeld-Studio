/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "scene_upscaler_plugin.hpp"
#include "scene_upscaler_plugin_metadata.hpp"

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

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace lfs::vis {
    namespace {
        constexpr std::string_view PROJECT_ID = "7fc73d74-f126-4146-b028-4bc1026e5c3b";
        constexpr std::string_view ENGINE_VERSION = "LichtFeld Studio";

#ifdef _WIN32
        using NativeLibrary = HMODULE;
#else
        using NativeLibrary = void*;
#endif

        [[nodiscard]] std::string pluginFilename(const std::string_view library) {
#if defined(_WIN32)
            return std::format("{}.dll", library);
#elif defined(__APPLE__)
            return std::format("lib{}.dylib", library);
#else
            return std::format("lib{}.so", library);
#endif
        }

        [[nodiscard]] std::vector<std::filesystem::path> pluginCandidates(
            const SceneUpscalerPluginInfo& info) {
            std::vector<std::filesystem::path> result;
            const auto filename = pluginFilename(info.library);
            const auto append = [&](const std::filesystem::path& root) {
                if (root.empty())
                    return;
                const auto candidate =
                    root / "scene_upscalers" / info.directory / filename;
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

    struct SceneUpscalerPlugin::Impl {
        explicit Impl(const SceneUpscalerPluginInfo& plugin_info) : info(plugin_info) {}

        struct OptimalSettingsCache {
            std::uint32_t output_width = 0;
            std::uint32_t output_height = 0;
            std::uint32_t quality = 0;
            LfsSceneUpscalerOptimalSettingsV1 settings{};
        };

        const SceneUpscalerPluginInfo& info;
        mutable std::mutex mutex;
        NativeLibrary library = nullptr;
        const LfsSceneUpscalerPluginApiV1* api = nullptr;
        void* plugin = nullptr;
        SceneUpscalerPluginState state = SceneUpscalerPluginState::Unprobed;
        std::wstring application_data_path;
        std::wstring plugin_directory;
        std::string diagnostic;
        std::string display_name;
        std::optional<OptimalSettingsCache> optimal_settings_cache;
        bool loading_enabled = true;
        bool runtime_initialized = false;
        std::optional<SceneUpscalerPluginViewIdentityAllocator> view_identity_allocator;
        std::optional<std::thread::id> vendor_thread_id;
        bool vendor_thread_mismatch_warned = false;

        void noteVendorCallerThreadLocked() {
            const auto thread_id = std::this_thread::get_id();
            if (!vendor_thread_id) {
                vendor_thread_id = thread_id;
                return;
            }
            if (*vendor_thread_id != thread_id && !vendor_thread_mismatch_warned) {
                vendor_thread_mismatch_warned = true;
                LOG_WARN("{} called from a different thread than the first "
                         "initializeRuntime/evaluate/createFeature caller; vendor runtimes "
                         "are not thread-safe",
                         info.name);
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

        void failLocked(const SceneUpscalerPluginState failed_state, std::string reason) {
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
            application_data_path.clear();
            plugin_directory.clear();
        }

        [[nodiscard]] bool probeLocked() {
            if (!loading_enabled) {
                state = SceneUpscalerPluginState::DisabledBySafeMode;
                diagnostic = "optional scene-reconstruction plugins are disabled in safe mode";
                return false;
            }
            if (state == SceneUpscalerPluginState::BootstrapReady ||
                state == SceneUpscalerPluginState::RuntimeReady ||
                state == SceneUpscalerPluginState::RuntimeMissing ||
                state == SceneUpscalerPluginState::UnsupportedEnvironment ||
                state == SceneUpscalerPluginState::RuntimeFailed) {
                return plugin != nullptr;
            }
            if (state == SceneUpscalerPluginState::DisabledBySafeMode ||
                state == SceneUpscalerPluginState::NotInstalled ||
                state == SceneUpscalerPluginState::InvalidPlugin ||
                state == SceneUpscalerPluginState::BootstrapFailed)
                return false;
            destroyLocked();

            std::error_code error;
            std::filesystem::path candidate;
            for (const auto& path : pluginCandidates(info)) {
                if (std::filesystem::is_regular_file(path, error) && !error) {
                    candidate = path;
                    break;
                }
                error.clear();
            }
            if (candidate.empty()) {
                state = SceneUpscalerPluginState::NotInstalled;
                diagnostic = std::format("{} plugin is not installed", info.name);
                return false;
            }

            library = loadLibrary(candidate);
            if (library == nullptr) {
                failLocked(SceneUpscalerPluginState::InvalidPlugin,
                           std::format("failed to load '{}': {}",
                                       candidate.string(),
                                       nativeLoadError()));
                LOG_WARN("Optional {} plugin is invalid: {}", info.name, diagnostic);
                return false;
            }

            const auto get_api = reinterpret_cast<LfsSceneUpscalerGetPluginApiV1Fn>(
                loadSymbol(library, LFS_SCENE_UPSCALER_PLUGIN_ENTRY_V1));
            if (get_api == nullptr) {
                failLocked(SceneUpscalerPluginState::InvalidPlugin,
                           "plugin entry point is missing");
                LOG_WARN("Optional {} plugin '{}' is invalid: {}",
                         info.name,
                         candidate.string(),
                         diagnostic);
                destroyLocked();
                return false;
            }
            api = get_api();
            if (!lfs_scene_upscaler_plugin_api_v1_complete(api) ||
                std::string_view(api->plugin_id) != info.id) {
                failLocked(SceneUpscalerPluginState::InvalidPlugin,
                           "plugin ABI or identifier is incompatible");
                LOG_WARN("Optional {} plugin '{}' is invalid: {}",
                         info.name,
                         candidate.string(),
                         diagnostic);
                destroyLocked();
                return false;
            }
            // Copy the provider's name before it can be unloaded. Metadata and
            // ABI identity agree, so settings stay keyed by the real provider ID.
            std::size_t name_length = 0;
            while (name_length <= 256 && api->display_name[name_length] != '\0')
                ++name_length;
            if (name_length == 0 || name_length > 256) {
                failLocked(SceneUpscalerPluginState::InvalidPlugin, "plugin display name is invalid");
                destroyLocked();
                return false;
            }
            display_name.assign(api->display_name, name_length);

            const auto paths = lfs::core::UserPaths::resolve();
            if (!paths) {
                failLocked(SceneUpscalerPluginState::InvalidPlugin,
                           "cannot resolve the plugin cache directory");
                LOG_WARN("Optional {} plugin '{}' cannot be initialized: {}",
                         info.name,
                         candidate.string(),
                         diagnostic);
                destroyLocked();
                return false;
            }
            const auto data_path = paths->cacheDir() / info.cache_dir;
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
                failLocked(SceneUpscalerPluginState::InvalidPlugin,
                           "plugin bootstrap context creation failed");
                LOG_WARN("Optional {} plugin '{}' cannot be initialized: {}",
                         info.name,
                         candidate.string(),
                         diagnostic);
                destroyLocked();
                return false;
            }
            state = SceneUpscalerPluginState::BootstrapReady;
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
                failLocked(SceneUpscalerPluginState::BootstrapFailed,
                           plugin_error.empty()
                               ? "plugin could not report required Vulkan extensions"
                               : plugin_error);
                LOG_WARN("{} bootstrap unavailable: {}", info.name, diagnostic);
                extensions.clear();
            }
            return extensions;
        }
    };

    SceneUpscalerPlugin::SceneUpscalerPlugin(const SceneUpscalerPluginInfo& info)
        : info_(info),
          impl_(new Impl(info_)) {}

    SceneUpscalerPlugin::~SceneUpscalerPlugin() {
        shutdown();
        delete impl_;
    }

    void SceneUpscalerPlugin::configure(const bool loading_enabled) {
        std::scoped_lock lock(impl_->mutex);
        if (impl_->loading_enabled == loading_enabled &&
            impl_->state != SceneUpscalerPluginState::Unprobed)
            return;
        impl_->destroyLocked();
        impl_->loading_enabled = loading_enabled;
        impl_->state = loading_enabled ? SceneUpscalerPluginState::Unprobed
                                       : SceneUpscalerPluginState::DisabledBySafeMode;
        impl_->diagnostic = loading_enabled
                                ? std::string{}
                                : "optional scene-reconstruction plugins are disabled in safe mode";
    }

    bool SceneUpscalerPlugin::probe() {
        std::scoped_lock lock(impl_->mutex);
        return impl_->probeLocked();
    }

    bool SceneUpscalerPlugin::available() { return probe(); }

    std::string SceneUpscalerPlugin::displayName() const {
        std::scoped_lock lock(impl_->mutex);
        return impl_->display_name.empty() ? info_.name : impl_->display_name;
    }

    std::string SceneUpscalerPlugin::diagnostic() const {
        std::scoped_lock lock(impl_->mutex);
        return impl_->diagnostic;
    }

    std::vector<std::string> SceneUpscalerPlugin::requiredInstanceExtensions() {
        std::scoped_lock lock(impl_->mutex);
        return impl_->extensionsLocked(false, VK_NULL_HANDLE, VK_NULL_HANDLE);
    }

    std::vector<std::string> SceneUpscalerPlugin::requiredDeviceExtensions(
        const VkInstance instance,
        const VkPhysicalDevice physical_device) {
        std::scoped_lock lock(impl_->mutex);
        return impl_->extensionsLocked(true, instance, physical_device);
    }

    void SceneUpscalerPlugin::markBootstrapFailed(std::string reason) {
        std::scoped_lock lock(impl_->mutex);
        impl_->failLocked(SceneUpscalerPluginState::BootstrapFailed, std::move(reason));
        LOG_WARN("{} bootstrap unavailable: {}", info_.name, impl_->diagnostic);
    }

    bool SceneUpscalerPlugin::initializeRuntime(const LfsSceneUpscalerRuntimeConfigV1& config) {
        std::scoped_lock lock(impl_->mutex);
        impl_->noteVendorCallerThreadLocked();
        if (!impl_->probeLocked())
            return false;
        if (impl_->runtime_initialized)
            return true;
        std::error_code error;
        std::filesystem::create_directories(
            std::filesystem::path(impl_->application_data_path), error);
        if (error) {
            impl_->failLocked(
                SceneUpscalerPluginState::RuntimeFailed,
                std::format("cannot create the plugin cache directory: {}",
                            error.message()));
            return false;
        }
        const auto result = impl_->api->initialize_runtime(impl_->plugin, &config);
        if (result != LFS_SCENE_UPSCALER_PLUGIN_OK) {
            const auto failed_state = [&] {
                switch (result) {
                case LFS_SCENE_UPSCALER_PLUGIN_UNAVAILABLE:
                    return SceneUpscalerPluginState::RuntimeMissing;
                case LFS_SCENE_UPSCALER_PLUGIN_UNSUPPORTED_DEVICE:
                    return SceneUpscalerPluginState::UnsupportedEnvironment;
                default:
                    return SceneUpscalerPluginState::RuntimeFailed;
                }
            }();
            impl_->failLocked(failed_state, impl_->pluginErrorLocked());
            if (impl_->diagnostic.empty())
                impl_->diagnostic = "plugin runtime initialization failed";
            return false;
        }
        impl_->runtime_initialized = true;
        impl_->optimal_settings_cache.reset();
        impl_->state = SceneUpscalerPluginState::RuntimeReady;
        impl_->diagnostic.clear();
        return true;
    }

    std::optional<LfsSceneUpscalerOptimalSettingsV1> SceneUpscalerPlugin::optimalSettings(
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

    std::optional<std::uint32_t> SceneUpscalerPlugin::acquireViewIdentity() {
        std::scoped_lock lock(impl_->mutex);
        if (!impl_->probeLocked())
            return std::nullopt;
        if (!impl_->view_identity_allocator) {
            impl_->view_identity_allocator.emplace(
                lfs_scene_upscaler_plugin_api_v1_supports_dynamic_view_ids(impl_->api) != 0);
        }
        return impl_->view_identity_allocator->acquire();
    }

    void SceneUpscalerPlugin::releaseViewIdentity(const std::uint32_t view) {
        std::scoped_lock lock(impl_->mutex);
        if (!impl_->view_identity_allocator ||
            !impl_->view_identity_allocator->owns(view))
            return;
        if (impl_->runtime_initialized && impl_->api != nullptr && impl_->plugin != nullptr)
            impl_->api->release_feature(impl_->plugin, view);
        impl_->view_identity_allocator->release(view);
    }

    bool SceneUpscalerPlugin::createFeature(
        const VkCommandBuffer command_buffer,
        const LfsSceneUpscalerFeatureConfigV1& config) {
        std::scoped_lock lock(impl_->mutex);
        impl_->noteVendorCallerThreadLocked();
        if (!impl_->runtime_initialized)
            return false;
        if (!impl_->view_identity_allocator ||
            !impl_->view_identity_allocator->owns(config.view)) {
            impl_->diagnostic = "feature identity was not allocated by the host";
            return false;
        }
        if (impl_->api->create_feature(impl_->plugin, command_buffer, &config) !=
            LFS_SCENE_UPSCALER_PLUGIN_OK) {
            impl_->diagnostic = impl_->pluginErrorLocked();
            return false;
        }
        return true;
    }

    bool SceneUpscalerPlugin::evaluate(const LfsSceneUpscalerEvaluateV1& evaluation) {
        std::scoped_lock lock(impl_->mutex);
        impl_->noteVendorCallerThreadLocked();
        if (!impl_->runtime_initialized)
            return false;
        if (!impl_->view_identity_allocator ||
            !impl_->view_identity_allocator->owns(evaluation.view)) {
            impl_->diagnostic = "evaluation identity was not allocated by the host";
            return false;
        }
        if (impl_->api->evaluate(impl_->plugin, &evaluation) !=
            LFS_SCENE_UPSCALER_PLUGIN_OK) {
            impl_->diagnostic = impl_->pluginErrorLocked();
            return false;
        }
        return true;
    }

    void SceneUpscalerPlugin::shutdownRuntime() {
        std::scoped_lock lock(impl_->mutex);
        if (!impl_->runtime_initialized)
            return;
        impl_->api->shutdown_runtime(impl_->plugin);
        impl_->runtime_initialized = false;
        impl_->optimal_settings_cache.reset();
        impl_->state = SceneUpscalerPluginState::BootstrapReady;
    }

    void SceneUpscalerPlugin::shutdown() {
        std::scoped_lock lock(impl_->mutex);
        impl_->destroyLocked();
        impl_->state = impl_->loading_enabled ? SceneUpscalerPluginState::Unprobed
                                              : SceneUpscalerPluginState::DisabledBySafeMode;
    }

    bool SceneUpscalerPlugin::hasCapability(const LfsSceneUpscalerPluginCapability capability) {
        std::scoped_lock lock(impl_->mutex);
        return impl_->probeLocked() &&
               lfs_scene_upscaler_plugin_api_v1_has_capability(impl_->api, capability) != 0;
    }

    std::span<SceneUpscalerPlugin* const> sceneUpscalerPlugins() {
        static const auto providers = [] {
            std::vector<std::unique_ptr<SceneUpscalerPlugin>> result;
            const std::array roots{lfs::core::getExecutableDir(), lfs::core::getLibDir()};
            const auto folders = discoverSceneUpscalerProviderFolders(roots);
            auto identity = static_cast<std::uint32_t>(SceneUpscalerBackend::FirstExternal);
            for (const auto& folder : folders) {
                result.push_back(std::make_unique<SceneUpscalerPlugin>(SceneUpscalerPluginInfo{
                    .backend = static_cast<SceneUpscalerBackend>(identity++),
                    .id = folder.id,
                    .name = folder.id,
                    .directory = folder.directory,
                    .library = folder.library,
                    .cache_dir = "providers/" + folder.id,
                    .presets = {{
                        {"quality", "preferences.scene_reconstruction_quality", folder.input_scales[0]},
                        {"balanced", "preferences.scene_reconstruction_balanced", folder.input_scales[1]},
                        {"performance", "preferences.scene_reconstruction_performance", folder.input_scales[2]},
                    }},
                }));
            }
            return result;
        }();
        static const auto plugins = [&] {
            std::vector<SceneUpscalerPlugin*> result;
            for (const auto& provider : providers)
                result.push_back(provider.get());
            return result;
        }();
        return plugins;
    }

    SceneUpscalerPlugin* sceneUpscalerPlugin(const SceneUpscalerBackend backend) {
        for (auto* const plugin : sceneUpscalerPlugins()) {
            if (plugin->info().backend == backend)
                return plugin;
        }
        return nullptr;
    }

    void configureSceneUpscalerPluginLoading(const bool enabled) {
        for (auto* const plugin : sceneUpscalerPlugins())
            plugin->configure(enabled);
    }

} // namespace lfs::vis
