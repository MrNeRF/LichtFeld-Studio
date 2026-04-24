/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// clang-format off
#include <glad/glad.h>
// clang-format on

#include <glm/mat4x4.hpp>

#include "gui/rmlui/rmlui_render_interface.hpp"

#include "core/camera.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "gui/rmlui/rml_path_utils.hpp"
#include "internal/resource_paths.hpp"
#include "io/pipelined_image_loader.hpp"
#include "rendering/gl_state_guard.hpp"
#include "scene/scene_manager.hpp"
#include "training/training_manager.hpp"

#include <stb_image.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <charconv>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <filesystem>
#include <mutex>
#include <optional>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

namespace lfs::vis::gui {

    struct PreviewTextureEntry {
        GLuint texture_id = 0;
        int width = 1;
        int height = 1;
        int ref_count = 0;
        bool loading = false;
        bool failed = false;
        bool high_priority = false;
        size_t approx_bytes = 4;
        std::chrono::steady_clock::time_point last_access = std::chrono::steady_clock::now();
        std::string source;
    };

    struct PreviewLoadRequest {
        Rml::TextureHandle handle = 0;
        std::filesystem::path path;
        lfs::io::LoadParams display_params;
        lfs::io::PipelinedImageLoader* preview_loader = nullptr;
        std::shared_ptr<lfs::io::PipelinedImageLoader> active_loader;
        bool high_priority = false;
        uint64_t navigation_epoch = 0;
    };

    struct LoadedPreview {
        Rml::TextureHandle handle = 0;
        std::vector<uint8_t> pixel_data;
        int width = 0;
        int height = 0;
        int channels = 0;
        bool failed = false;
    };

    struct ExternalTextureEntry {
        GLuint texture_id = 0;
        int width = 1;
        int height = 1;
        int ref_count = 0;
        bool owns_texture = false;
        std::string source;
    };

    struct PreviewTextureCache {
        static constexpr int NUM_WORKERS = 4;
        static constexpr int NUM_HIGH_PRIORITY_WORKERS = 1;

        std::unique_ptr<lfs::io::PipelinedImageLoader> preview_loader;
        std::unordered_map<Rml::TextureHandle, PreviewTextureEntry> preview_entries;
        std::unordered_map<std::string, Rml::TextureHandle> source_to_handle;
        std::unordered_map<Rml::TextureHandle, ExternalTextureEntry> external_entries;
        std::unordered_map<std::string, Rml::TextureHandle> external_source_to_handle;
        std::deque<PreviewLoadRequest> high_priority_queue;
        std::deque<PreviewLoadRequest> low_priority_queue;
        std::mutex load_queue_mutex;
        std::condition_variable load_queue_cv;
        std::queue<LoadedPreview> ready_queue;
        std::mutex ready_queue_mutex;
        std::atomic<bool> worker_running{false};
        std::vector<std::thread> workers;
        std::atomic<uint64_t> navigation_epoch{0};
        size_t cached_texture_bytes = 0;

        PreviewTextureCache();
        ~PreviewTextureCache();

        void worker_loop(bool allow_low_priority);
        uint64_t begin_navigation_epoch();
        void invalidate_loading_entries(bool high_priority_only);
        void detach_entry_source(Rml::TextureHandle handle);
        void erase_entry(Rml::TextureHandle handle);
        bool source_maps_to_handle(const std::string& source, Rml::TextureHandle handle) const;
        void evict_unused_textures();
        void clear_pending_loads();
    };

    namespace {
        constexpr size_t PREVIEW_CACHE_BYTES = 512ULL * 1024 * 1024;
        constexpr size_t PREVIEW_TEXTURE_CACHE_BYTES = 512ULL * 1024 * 1024;
        constexpr size_t MAX_PREVIEW_TEXTURES = 96;
        constexpr int PLACEHOLDER_DIM = 1;

        struct PreviewParams {
            int cam_uid = -1;
            int thumb = 0;
            int rf = 1;
            int mw = 0;
            int pmw = 0;
            bool ud = false;
            std::string path;
        };

        struct PreviewPixels {
            std::vector<uint8_t> data;
            int width = 0;
            int height = 0;
            int channels = 0;
        };

        int parse_int(const std::string_view sv, const int fallback = 0) {
            int value = fallback;
            std::from_chars(sv.data(), sv.data() + sv.size(), value);
            return value;
        }

        int hex_value(const char ch) {
            if (ch >= '0' && ch <= '9')
                return ch - '0';
            if (ch >= 'a' && ch <= 'f')
                return ch - 'a' + 10;
            if (ch >= 'A' && ch <= 'F')
                return ch - 'A' + 10;
            return -1;
        }

        std::string percent_decode(const std::string_view text) {
            std::string decoded;
            decoded.reserve(text.size());
            for (size_t i = 0; i < text.size(); ++i) {
                if (text[i] != '%' || i + 2 >= text.size()) {
                    decoded.push_back(text[i]);
                    continue;
                }

                const int hi = hex_value(text[i + 1]);
                const int lo = hex_value(text[i + 2]);
                if (hi < 0 || lo < 0) {
                    decoded.push_back(text[i]);
                    continue;
                }

                decoded.push_back(static_cast<char>((hi << 4) | lo));
                i += 2;
            }
            return decoded;
        }

        std::optional<std::filesystem::path> resolve_asset_texture_fallback(
            const std::string_view original_source,
            const std::string_view attempted_source) {
            const auto try_asset = [](std::string asset_name) -> std::optional<std::filesystem::path> {
                while (asset_name.rfind("../", 0) == 0)
                    asset_name.erase(0, 3);
                while (asset_name.rfind("./", 0) == 0)
                    asset_name.erase(0, 2);
                if (asset_name.empty())
                    return std::nullopt;

                try {
                    const auto path = lfs::vis::getAssetPath(asset_name);
                    if (std::filesystem::exists(path))
                        return path;
                } catch (const std::exception& e) {
                    LOG_DEBUG("RmlUI texture fallback could not resolve asset '{}': {}",
                              asset_name,
                              e.what());
                }
                return std::nullopt;
            };

            for (const std::string_view source : {original_source, attempted_source}) {
                if (source.empty())
                    continue;

                if (auto path = try_asset(std::string(source)))
                    return path;

                constexpr std::string_view rmlui_icon_segment = "rmlui/icon/";
                if (const auto pos = source.find(rmlui_icon_segment); pos != std::string_view::npos) {
                    if (auto path = try_asset("icon/" + std::string(source.substr(pos + rmlui_icon_segment.size()))))
                        return path;
                }

                constexpr std::string_view icon_segment = "/icon/";
                if (const auto pos = source.find(icon_segment); pos != std::string_view::npos) {
                    if (auto path = try_asset("icon/" + std::string(source.substr(pos + icon_segment.size()))))
                        return path;
                }
            }

            return std::nullopt;
        }

        PreviewParams parse_preview_url(const Rml::String& source) {
            PreviewParams p;
            constexpr std::string_view PREFIX = "preview://";
            assert(source.substr(0, PREFIX.size()) == PREFIX);

            const auto query = source.substr(PREFIX.size());

            const auto path_pos = query.find("path=");
            if (path_pos != std::string::npos)
                p.path = percent_decode(query.substr(path_pos + 5));

            const auto params_end = (path_pos != std::string::npos && path_pos > 0) ? path_pos - 1 : query.size();
            const auto params_str = query.substr(0, params_end);
            std::istringstream stream(std::string{params_str});
            std::string pair;
            while (std::getline(stream, pair, '&')) {
                const auto eq = pair.find('=');
                if (eq == std::string::npos)
                    continue;
                const auto key = std::string_view{pair}.substr(0, eq);
                const auto val = std::string_view{pair}.substr(eq + 1);
                if (key == "cam")
                    p.cam_uid = parse_int(val, -1);
                else if (key == "thumb")
                    p.thumb = parse_int(val);
                else if (key == "rf")
                    p.rf = parse_int(val, 1);
                else if (key == "mw")
                    p.mw = parse_int(val);
                else if (key == "pmw")
                    p.pmw = parse_int(val);
                else if (key == "ud")
                    p.ud = (val == "1");
            }
            return p;
        }

        lfs::io::PipelinedLoaderConfig make_preview_loader_config() {
            lfs::io::PipelinedLoaderConfig config;
            config.io_threads = 0;
            config.cold_process_threads = 0;
            config.max_cache_bytes = PREVIEW_CACHE_BYTES;
            return config;
        }

        lfs::io::PipelinedImageLoader& get_preview_loader(PreviewTextureCache& preview_cache) {
            if (!preview_cache.preview_loader) {
                preview_cache.preview_loader =
                    std::make_unique<lfs::io::PipelinedImageLoader>(make_preview_loader_config());
            }
            return *preview_cache.preview_loader;
        }

        std::shared_ptr<lfs::io::PipelinedImageLoader> get_active_preview_loader(
            lfs::vis::SceneManager* scene_manager) {
            if (!scene_manager)
                return {};

            auto* trainer_manager = scene_manager->getTrainerManager();
            if (!trainer_manager || !trainer_manager->hasTrainer())
                return {};

            const auto* trainer = trainer_manager->getTrainer();
            if (!trainer)
                return {};

            return trainer->getActiveImageLoader();
        }

        GLuint create_placeholder_texture() {
            GLuint texture = 0;
            constexpr std::array<unsigned char, 4> PLACEHOLDER_RGBA = {0, 0, 0, 0};

            glGenTextures(1, &texture);
            glBindTexture(GL_TEXTURE_2D, texture);
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, PLACEHOLDER_DIM, PLACEHOLDER_DIM, 0,
                         GL_RGBA, GL_UNSIGNED_BYTE, PLACEHOLDER_RGBA.data());
            glBindTexture(GL_TEXTURE_2D, 0);

            if (const GLenum gl_err = glGetError(); gl_err != GL_NO_ERROR) {
                if (texture != 0)
                    glDeleteTextures(1, &texture);
                LOG_WARN("Failed to create preview placeholder texture: {}", static_cast<int>(gl_err));
                return 0;
            }

            return texture;
        }

        GLuint create_external_texture(const int width, const int height) {
            GLuint texture = 0;
            glGenTextures(1, &texture);
            glBindTexture(GL_TEXTURE_2D, texture);
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, std::max(width, 1), std::max(height, 1), 0,
                         GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
            glBindTexture(GL_TEXTURE_2D, 0);

            if (const GLenum gl_err = glGetError(); gl_err != GL_NO_ERROR) {
                if (texture != 0)
                    glDeleteTextures(1, &texture);
                LOG_WARN("Failed to create external presentation texture: {}", static_cast<int>(gl_err));
                return 0;
            }

            return texture;
        }

        void ensure_external_texture_storage(const GLuint texture_id, const int width, const int height) {
            glBindTexture(GL_TEXTURE_2D, texture_id);
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, std::max(width, 1), std::max(height, 1), 0,
                         GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
            glBindTexture(GL_TEXTURE_2D, 0);
        }

        bool copy_external_texture(const GLuint source_texture_id,
                                   const GLuint presentation_texture_id,
                                   const int width,
                                   const int height,
                                   const bool flip_y) {
            if (source_texture_id == 0 || presentation_texture_id == 0 || width <= 0 || height <= 0) {
                return false;
            }

            const lfs::rendering::GLStateGuard state_guard;

            GLuint read_fbo = 0;
            GLuint draw_fbo = 0;
            glGenFramebuffers(1, &read_fbo);
            glGenFramebuffers(1, &draw_fbo);

            glBindFramebuffer(GL_READ_FRAMEBUFFER, read_fbo);
            glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                   GL_TEXTURE_2D, source_texture_id, 0);
            glReadBuffer(GL_COLOR_ATTACHMENT0);

            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, draw_fbo);
            glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                   GL_TEXTURE_2D, presentation_texture_id, 0);
            glDrawBuffer(GL_COLOR_ATTACHMENT0);

            const bool complete =
                glCheckFramebufferStatus(GL_READ_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE &&
                glCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
            if (!complete) {
                LOG_WARN("External texture presentation FBO incomplete");
                glDeleteFramebuffers(1, &read_fbo);
                glDeleteFramebuffers(1, &draw_fbo);
                return false;
            }

            const GLint dst_y0 = flip_y ? height : 0;
            const GLint dst_y1 = flip_y ? 0 : height;
            glBlitFramebuffer(0, 0, width, height,
                              0, dst_y0, width, dst_y1,
                              GL_COLOR_BUFFER_BIT, GL_NEAREST);

            const GLenum gl_err = glGetError();
            glDeleteFramebuffers(1, &read_fbo);
            glDeleteFramebuffers(1, &draw_fbo);
            if (gl_err != GL_NO_ERROR) {
                LOG_WARN("Failed to update external presentation texture: {}", static_cast<int>(gl_err));
                return false;
            }

            return true;
        }

        std::optional<PreviewPixels> tensor_to_preview_pixels(const lfs::core::Tensor& tensor,
                                                              const std::string_view log_context) {
            if (!tensor.is_valid() || tensor.ndim() != 3 || tensor.numel() == 0) {
                return std::nullopt;
            }

            lfs::core::Tensor formatted = tensor;
            int channels = 0;
            int width = 0;
            int height = 0;

            const auto& shape = tensor.shape();
            const int first_dim = static_cast<int>(shape[0]);
            const int last_dim = static_cast<int>(shape[2]);

            if (first_dim == 1 || first_dim == 3 || first_dim == 4) {
                channels = first_dim;
                height = static_cast<int>(shape[1]);
                width = static_cast<int>(shape[2]);
                formatted = tensor.permute({1, 2, 0}).contiguous();
            } else if (last_dim == 1 || last_dim == 3 || last_dim == 4) {
                channels = last_dim;
                height = static_cast<int>(shape[0]);
                width = static_cast<int>(shape[1]);
                formatted = tensor.contiguous();
            } else {
                LOG_WARN("Unsupported {} tensor shape: [{}, {}, {}]",
                         log_context,
                         static_cast<int>(shape[0]),
                         static_cast<int>(shape[1]),
                         static_cast<int>(shape[2]));
                return std::nullopt;
            }

            if (formatted.dtype() != lfs::core::DataType::UInt8) {
                formatted = (formatted.clamp(0.0f, 1.0f) * 255.0f)
                                .to(lfs::core::DataType::UInt8)
                                .contiguous();
            }

            if (formatted.device() == lfs::core::Device::CUDA) {
                formatted = formatted.cpu();
            }
            formatted = formatted.contiguous();

            PreviewPixels pixels;
            pixels.width = width;
            pixels.height = height;
            pixels.channels = channels;
            pixels.data.resize(static_cast<size_t>(width) * height * channels);
            std::memcpy(pixels.data.data(),
                        formatted.ptr<unsigned char>(),
                        pixels.data.size());
            return pixels;
        }

        bool upload_preview_pixels(const GLuint texture_id,
                                   const LoadedPreview& loaded) {
            glBindTexture(GL_TEXTURE_2D, texture_id);
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

            const GLenum format = (loaded.channels == 1)   ? GL_RED
                                  : (loaded.channels == 4) ? GL_RGBA
                                                           : GL_RGB;
            const GLenum internal = (loaded.channels == 1)   ? GL_R8
                                    : (loaded.channels == 4) ? GL_RGBA8
                                                             : GL_RGB8;

            glTexImage2D(GL_TEXTURE_2D, 0, internal, loaded.width, loaded.height, 0,
                         format, GL_UNSIGNED_BYTE, loaded.pixel_data.data());
            glBindTexture(GL_TEXTURE_2D, 0);

            if (const GLenum gl_err = glGetError(); gl_err != GL_NO_ERROR) {
                LOG_WARN("Failed to upload preview texture: {}", static_cast<int>(gl_err));
                return false;
            }

            return true;
        }

    } // namespace

    PreviewTextureCache::PreviewTextureCache()
        : worker_running(true) {
        workers.reserve(NUM_WORKERS);
        for (int i = 0; i < NUM_WORKERS; ++i) {
            const bool allow_low_priority = (i >= NUM_HIGH_PRIORITY_WORKERS);
            workers.emplace_back([this, allow_low_priority] { worker_loop(allow_low_priority); });
        }
    }

    PreviewTextureCache::~PreviewTextureCache() {
        worker_running = false;
        load_queue_cv.notify_all();
        for (auto& w : workers) {
            if (w.joinable())
                w.join();
        }
    }

    bool PreviewTextureCache::source_maps_to_handle(const std::string& source,
                                                    const Rml::TextureHandle handle) const {
        if (const auto it = source_to_handle.find(source); it != source_to_handle.end())
            return it->second == handle;
        return false;
    }

    void PreviewTextureCache::detach_entry_source(const Rml::TextureHandle handle) {
        const auto entry_it = preview_entries.find(handle);
        if (entry_it == preview_entries.end())
            return;

        if (const auto source_it = source_to_handle.find(entry_it->second.source);
            source_it != source_to_handle.end() && source_it->second == handle) {
            source_to_handle.erase(source_it);
        }
    }

    void PreviewTextureCache::erase_entry(const Rml::TextureHandle handle) {
        const auto entry_it = preview_entries.find(handle);
        if (entry_it == preview_entries.end())
            return;

        detach_entry_source(handle);
        if (entry_it->second.texture_id != 0) {
            glDeleteTextures(1, &entry_it->second.texture_id);
        }
        cached_texture_bytes =
            (cached_texture_bytes > entry_it->second.approx_bytes)
                ? cached_texture_bytes - entry_it->second.approx_bytes
                : 0;
        preview_entries.erase(entry_it);
    }

    void PreviewTextureCache::invalidate_loading_entries(const bool high_priority_only) {
        const auto now = std::chrono::steady_clock::now();
        for (auto& [handle, entry] : preview_entries) {
            if (!entry.loading)
                continue;
            if (high_priority_only && !entry.high_priority)
                continue;

            detach_entry_source(handle);
            entry.loading = false;
            entry.failed = true;
            entry.last_access = now;
        }
    }

    uint64_t PreviewTextureCache::begin_navigation_epoch() {
        const auto epoch = navigation_epoch.fetch_add(1, std::memory_order_acq_rel) + 1;
        {
            std::lock_guard lock(load_queue_mutex);
            high_priority_queue.clear();
        }
        invalidate_loading_entries(true);
        return epoch;
    }

    void PreviewTextureCache::evict_unused_textures() {
        while ((preview_entries.size() > MAX_PREVIEW_TEXTURES ||
                cached_texture_bytes > PREVIEW_TEXTURE_CACHE_BYTES)) {
            auto oldest_it = preview_entries.end();
            for (auto it = preview_entries.begin(); it != preview_entries.end(); ++it) {
                if (it->second.ref_count > 0 || it->second.loading)
                    continue;
                if (oldest_it == preview_entries.end() ||
                    it->second.last_access < oldest_it->second.last_access) {
                    oldest_it = it;
                }
            }

            if (oldest_it == preview_entries.end())
                break;

            erase_entry(oldest_it->first);
        }
    }

    void PreviewTextureCache::clear_pending_loads() {
        navigation_epoch.fetch_add(1, std::memory_order_release);
        {
            std::lock_guard lock(load_queue_mutex);
            high_priority_queue.clear();
            low_priority_queue.clear();
        }
        invalidate_loading_entries(false);
    }

    void PreviewTextureCache::worker_loop(const bool allow_low_priority) {
        while (worker_running) {
            PreviewLoadRequest request;

            {
                std::unique_lock lock(load_queue_mutex);
                load_queue_cv.wait(lock, [this, allow_low_priority] {
                    if (!worker_running.load())
                        return true;
                    if (!high_priority_queue.empty())
                        return true;
                    if (!allow_low_priority)
                        return false;
                    return !low_priority_queue.empty();
                });

                if (!worker_running && high_priority_queue.empty() && low_priority_queue.empty())
                    return;

                if (!high_priority_queue.empty()) {
                    request = std::move(high_priority_queue.front());
                    high_priority_queue.pop_front();
                } else {
                    if (!allow_low_priority || low_priority_queue.empty())
                        continue;
                    request = std::move(low_priority_queue.front());
                    low_priority_queue.pop_front();
                }
            }

            if (request.high_priority &&
                request.navigation_epoch < navigation_epoch.load(std::memory_order_acquire)) {
                continue;
            }

            try {
                auto* loader = request.active_loader
                                   ? request.active_loader.get()
                                   : request.preview_loader;
                assert(loader);

                auto tensor = loader->load_image_immediate(request.path, request.display_params);
                auto pixels = tensor_to_preview_pixels(tensor, "preview");

                if (!pixels) {
                    throw std::runtime_error("Failed to decode preview pixels");
                }

                if (request.high_priority &&
                    request.navigation_epoch < navigation_epoch.load(std::memory_order_acquire)) {
                    continue;
                }

                {
                    std::lock_guard lock(ready_queue_mutex);
                    ready_queue.push({.handle = request.handle,
                                      .pixel_data = std::move(pixels->data),
                                      .width = pixels->width,
                                      .height = pixels->height,
                                      .channels = pixels->channels,
                                      .failed = false});
                }

            } catch (const std::exception& e) {
                LOG_WARN("Preview load failed for {}: {}",
                         lfs::core::path_to_utf8(request.path),
                         e.what());
                std::lock_guard lock(ready_queue_mutex);
                ready_queue.push(LoadedPreview{
                    .handle = request.handle,
                    .pixel_data = {},
                    .width = 0,
                    .height = 0,
                    .channels = 0,
                    .failed = true});
            }
        }
    }

    RmlRenderInterface::RmlRenderInterface()
        : preview_cache_(std::make_unique<PreviewTextureCache>()) {
        if (!*this)
            LOG_ERROR("RmlUI GL3 render interface failed to initialize");
    }

    RmlRenderInterface::~RmlRenderInterface() {
        if (!preview_cache_)
            return;

        for (const auto& [handle, entry] : preview_cache_->preview_entries) {
            if (entry.texture_id != 0) {
                glDeleteTextures(1, &entry.texture_id);
            }
        }
    }

    void RmlRenderInterface::set_scene_manager(lfs::vis::SceneManager* scene_manager) { scene_manager_ = scene_manager; }

    void RmlRenderInterface::clear_pending_preview_loads() {
        if (preview_cache_)
            preview_cache_->clear_pending_loads();
    }

    void RmlRenderInterface::process_pending_preview_uploads() {
        if (!preview_cache_)
            return;

        while (true) {
            LoadedPreview loaded;
            {
                std::lock_guard lock(preview_cache_->ready_queue_mutex);
                if (preview_cache_->ready_queue.empty())
                    break;
                loaded = std::move(preview_cache_->ready_queue.front());
                preview_cache_->ready_queue.pop();
            }

            const auto entry_it = preview_cache_->preview_entries.find(loaded.handle);
            if (entry_it == preview_cache_->preview_entries.end())
                continue;

            auto& entry = entry_it->second;
            if (!entry.loading)
                continue;

            entry.loading = false;
            entry.last_access = std::chrono::steady_clock::now();

            if (loaded.failed) {
                entry.failed = true;
                preview_cache_->detach_entry_source(loaded.handle);
                if (entry.ref_count <= 0) {
                    preview_cache_->erase_entry(loaded.handle);
                }
                continue;
            }

            if (!upload_preview_pixels(entry.texture_id, loaded)) {
                entry.failed = true;
                preview_cache_->detach_entry_source(loaded.handle);
                if (entry.ref_count <= 0) {
                    preview_cache_->erase_entry(loaded.handle);
                }
                continue;
            }

            entry.width = loaded.width;
            entry.height = loaded.height;
            preview_cache_->cached_texture_bytes =
                (preview_cache_->cached_texture_bytes > entry.approx_bytes)
                    ? preview_cache_->cached_texture_bytes - entry.approx_bytes
                    : 0;
            entry.approx_bytes = static_cast<size_t>(loaded.width) * loaded.height *
                                 std::max(1, loaded.channels);
            preview_cache_->cached_texture_bytes += entry.approx_bytes;
            entry.failed = false;
        }

        preview_cache_->evict_unused_textures();
    }

    void RmlRenderInterface::register_external_texture(const Rml::String& source,
                                                       const unsigned int texture_id,
                                                       const int width,
                                                       const int height,
                                                       const bool flip_y) {
        if (!preview_cache_ || source.empty() || texture_id == 0)
            return;

        if (!flip_y) {
            const auto handle = static_cast<Rml::TextureHandle>(texture_id);

            if (const auto source_it = preview_cache_->external_source_to_handle.find(source);
                source_it != preview_cache_->external_source_to_handle.end() && source_it->second != handle) {
                if (auto old_entry_it = preview_cache_->external_entries.find(source_it->second);
                    old_entry_it != preview_cache_->external_entries.end()) {
                    if (old_entry_it->second.ref_count <= 0) {
                        if (old_entry_it->second.owns_texture && old_entry_it->second.texture_id != 0) {
                            GLuint texture = old_entry_it->second.texture_id;
                            glDeleteTextures(1, &texture);
                        }
                        preview_cache_->external_entries.erase(old_entry_it);
                    } else {
                        old_entry_it->second.source.clear();
                    }
                }
            }

            auto& entry = preview_cache_->external_entries[handle];
            if (!entry.source.empty() && entry.source != source) {
                if (const auto old_source_it = preview_cache_->external_source_to_handle.find(entry.source);
                    old_source_it != preview_cache_->external_source_to_handle.end() && old_source_it->second == handle) {
                    preview_cache_->external_source_to_handle.erase(old_source_it);
                }
            }

            entry.texture_id = texture_id;
            entry.width = width;
            entry.height = height;
            entry.owns_texture = false;
            entry.source = std::string{source};
            preview_cache_->external_source_to_handle[entry.source] = handle;
            return;
        }

        Rml::TextureHandle handle = 0;
        if (const auto source_it = preview_cache_->external_source_to_handle.find(source);
            source_it != preview_cache_->external_source_to_handle.end()) {
            handle = source_it->second;
            if (const auto entry_it = preview_cache_->external_entries.find(handle);
                entry_it != preview_cache_->external_entries.end() && !entry_it->second.owns_texture) {
                if (entry_it->second.ref_count <= 0) {
                    preview_cache_->external_entries.erase(entry_it);
                } else {
                    entry_it->second.source.clear();
                }
                preview_cache_->external_source_to_handle.erase(source_it);
                handle = 0;
            }
        }
        if (handle == 0) {
            const GLuint presentation_texture = create_external_texture(width, height);
            if (presentation_texture == 0) {
                return;
            }
            handle = static_cast<Rml::TextureHandle>(presentation_texture);
            preview_cache_->external_entries.emplace(
                handle,
                ExternalTextureEntry{
                    .texture_id = presentation_texture,
                    .width = width,
                    .height = height,
                    .ref_count = 0,
                    .owns_texture = true,
                    .source = {}});
        }

        auto entry_it = preview_cache_->external_entries.find(handle);
        if (entry_it == preview_cache_->external_entries.end()) {
            return;
        }

        auto& entry = entry_it->second;
        if ((entry.width != width || entry.height != height) && entry.texture_id != 0) {
            ensure_external_texture_storage(entry.texture_id, width, height);
        }
        entry.width = width;
        entry.height = height;
        entry.owns_texture = true;
        entry.source = std::string{source};
        preview_cache_->external_source_to_handle[entry.source] = handle;

        if (!copy_external_texture(texture_id, entry.texture_id, width, height, true)) {
            LOG_WARN("Failed to copy external texture '{}'", entry.source);
        }
    }

    void RmlRenderInterface::unregister_external_texture(const Rml::String& source) {
        if (!preview_cache_ || source.empty())
            return;

        const auto source_it = preview_cache_->external_source_to_handle.find(source);
        if (source_it == preview_cache_->external_source_to_handle.end())
            return;

        const auto handle = source_it->second;
        preview_cache_->external_source_to_handle.erase(source_it);

        if (auto entry_it = preview_cache_->external_entries.find(handle);
            entry_it != preview_cache_->external_entries.end()) {
            if (entry_it->second.ref_count <= 0) {
                if (entry_it->second.owns_texture && entry_it->second.texture_id != 0) {
                    GLuint texture = entry_it->second.texture_id;
                    glDeleteTextures(1, &texture);
                }
                preview_cache_->external_entries.erase(entry_it);
            } else
                entry_it->second.source.clear();
        }
    }

    bool RmlRenderInterface::is_external_texture(const Rml::TextureHandle texture_handle) const {
        return preview_cache_ && preview_cache_->external_entries.contains(texture_handle);
    }

    void RmlRenderInterface::RenderGeometry(const Rml::CompiledGeometryHandle handle,
                                            const Rml::Vector2f translation,
                                            const Rml::TextureHandle texture) {
        const bool external_texture = is_external_texture(texture);
        if (external_texture) {
            glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
                                GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
        }

        RenderInterface_GL3::RenderGeometry(handle, translation, texture);

        if (external_texture)
            glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    }

    Rml::TextureHandle RmlRenderInterface::LoadTexture(Rml::Vector2i& dimensions,
                                                       const Rml::String& source) {
        if (const auto external_handle = load_external_texture(dimensions, source); external_handle)
            return external_handle;

        constexpr std::string_view PREVIEW_PREFIX = "preview://";
        if (source.size() > PREVIEW_PREFIX.size() &&
            source.substr(0, PREVIEW_PREFIX.size()) == PREVIEW_PREFIX) {
            auto handle = load_preview_texture(dimensions, source);
            if (handle)
                return handle;
        }

        int w = 0, h = 0, channels = 0;
        std::string decoded_source;
        std::string resolved_source = source;
        if (const auto file_uri_path = rml_paths::fileUriToPath(source)) {
            resolved_source = rml_paths::normalizeFilesystemPath(*file_uri_path);
        }

        std::string attempted_source = resolved_source;
        unsigned char* data = stbi_load(resolved_source.c_str(), &w, &h, &channels, 4);
        if (!data && resolved_source.find('%') != std::string::npos) {
            decoded_source = percent_decode(resolved_source);
            if (decoded_source != resolved_source) {
                attempted_source = decoded_source;
                data = stbi_load(decoded_source.c_str(), &w, &h, &channels, 4);
            }
        }
#ifndef _WIN32
        if (!data && !attempted_source.empty() && attempted_source[0] != '/' &&
            attempted_source.find("://") == std::string::npos) {
            const std::string absolute_source = "/" + attempted_source;
            if (std::filesystem::exists(lfs::core::utf8_to_path(absolute_source))) {
                attempted_source = absolute_source;
                data = stbi_load(absolute_source.c_str(), &w, &h, &channels, 4);
            }
        }
#endif
        if (!data) {
            if (const auto fallback_path = resolve_asset_texture_fallback(source, attempted_source)) {
                attempted_source = rml_paths::normalizeFilesystemPath(*fallback_path);
                data = stbi_load(attempted_source.c_str(), &w, &h, &channels, 4);
            }
        }

        if (!data) {
            LOG_WARN("RmlUI LoadTexture failed: {}", attempted_source);
            return 0;
        }

        dimensions.x = w;
        dimensions.y = h;

        const int pixel_count = w * h;
        for (int i = 0; i < pixel_count; ++i) {
            unsigned char* p = data + i * 4;
            const unsigned int a = p[3];
            p[0] = static_cast<unsigned char>((p[0] * a + 127) / 255);
            p[1] = static_cast<unsigned char>((p[1] * a + 127) / 255);
            p[2] = static_cast<unsigned char>((p[2] * a + 127) / 255);
        }

        GLuint tex = 0;
        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
        glBindTexture(GL_TEXTURE_2D, 0);

        stbi_image_free(data);
        return static_cast<Rml::TextureHandle>(tex);
    }

    void RmlRenderInterface::ReleaseTexture(Rml::TextureHandle texture_handle) {
        if (preview_cache_) {
            auto external_it = preview_cache_->external_entries.find(texture_handle);
            if (external_it != preview_cache_->external_entries.end()) {
                auto& entry = external_it->second;
                entry.ref_count = std::max(0, entry.ref_count - 1);
                if (entry.ref_count == 0 && entry.source.empty()) {
                    if (entry.owns_texture && entry.texture_id != 0) {
                        GLuint texture = entry.texture_id;
                        glDeleteTextures(1, &texture);
                    }
                    preview_cache_->external_entries.erase(external_it);
                }
                return;
            }
        }

        if (preview_cache_) {
            auto it = preview_cache_->preview_entries.find(texture_handle);
            if (it != preview_cache_->preview_entries.end()) {
                auto& entry = it->second;
                entry.ref_count = std::max(0, entry.ref_count - 1);
                entry.last_access = std::chrono::steady_clock::now();
                if (entry.ref_count == 0 &&
                    !preview_cache_->source_maps_to_handle(entry.source, texture_handle)) {
                    preview_cache_->erase_entry(texture_handle);
                    return;
                }
                preview_cache_->evict_unused_textures();
                return;
            }
        }
        GLuint tex = static_cast<GLuint>(texture_handle);
        glDeleteTextures(1, &tex);
    }

    Rml::TextureHandle RmlRenderInterface::load_external_texture(Rml::Vector2i& dimensions,
                                                                 const Rml::String& source) {
        if (!preview_cache_)
            return 0;

        const auto source_it = preview_cache_->external_source_to_handle.find(source);
        if (source_it == preview_cache_->external_source_to_handle.end())
            return 0;

        const auto handle = source_it->second;
        const auto entry_it = preview_cache_->external_entries.find(handle);
        if (entry_it == preview_cache_->external_entries.end())
            return 0;

        auto& entry = entry_it->second;
        ++entry.ref_count;
        dimensions.x = entry.width;
        dimensions.y = entry.height;
        return handle;
    }

    Rml::TextureHandle RmlRenderInterface::load_preview_texture(Rml::Vector2i& dimensions,
                                                                const Rml::String& source) {
        assert(preview_cache_);
        const auto params = parse_preview_url(source);
        if (params.path.empty())
            return 0;

        const auto path = std::filesystem::path(params.path);
        if (!std::filesystem::exists(path))
            return 0;

        if (const auto source_it = preview_cache_->source_to_handle.find(source);
            source_it != preview_cache_->source_to_handle.end()) {
            const auto existing_handle = source_it->second;
            if (auto entry_it = preview_cache_->preview_entries.find(existing_handle);
                entry_it != preview_cache_->preview_entries.end()) {
                auto& entry = entry_it->second;
                if (entry.failed) {
                    preview_cache_->detach_entry_source(existing_handle);
                    if (entry.ref_count <= 0) {
                        preview_cache_->erase_entry(existing_handle);
                    }
                } else {
                    ++entry.ref_count;
                    entry.last_access = std::chrono::steady_clock::now();
                    dimensions.x = entry.width;
                    dimensions.y = entry.height;
                    return existing_handle;
                }
            }
            if (preview_cache_->source_maps_to_handle(std::string{source}, existing_handle)) {
                preview_cache_->source_to_handle.erase(source_it);
            }
        }

        const int rf = params.thumb > 0 ? 1 : params.rf;
        const int mw = params.thumb > 0 ? params.thumb : params.mw;

        lfs::io::LoadParams display_params;
        display_params.resize_factor = rf;
        display_params.max_width = mw;

        if (params.thumb == 0 && params.pmw > 0 &&
            (display_params.max_width <= 0 || params.pmw < display_params.max_width)) {
            display_params.max_width = params.pmw;
        }

        if (params.ud && params.cam_uid >= 0 && scene_manager_) {
            auto& scene = scene_manager_->getScene();
            auto cameras = scene.getAllCameras();
            const auto it = std::find_if(cameras.begin(), cameras.end(),
                                         [&](const auto& c) { return c->uid() == params.cam_uid; });
            if (it != cameras.end() && (*it)->has_distortion()) {
                auto& camera = *it;
                if (!camera->is_undistort_precomputed())
                    camera->precompute_undistortion();
                display_params.undistort = &camera->undistort_params();
            }
        }

        auto active_loader = get_active_preview_loader(scene_manager_);
        lfs::io::PipelinedImageLoader* preview_loader = nullptr;
        try {
            preview_loader = &get_preview_loader(*preview_cache_);
        } catch (const std::exception& e) {
            LOG_WARN("Preview loader init failed: {}", e.what());
            return 0;
        }

        const bool high_priority = (params.thumb == 0);
        const uint64_t epoch =
            high_priority ? preview_cache_->begin_navigation_epoch() : 0;

        const GLuint texture_id = create_placeholder_texture();
        if (texture_id == 0)
            return 0;

        const auto handle = static_cast<Rml::TextureHandle>(texture_id);
        preview_cache_->source_to_handle[source] = handle;
        preview_cache_->preview_entries[handle] = PreviewTextureEntry{
            .texture_id = texture_id,
            .width = PLACEHOLDER_DIM,
            .height = PLACEHOLDER_DIM,
            .ref_count = 1,
            .loading = true,
            .failed = false,
            .high_priority = high_priority,
            .approx_bytes = 4,
            .last_access = std::chrono::steady_clock::now(),
            .source = std::string{source}};
        preview_cache_->cached_texture_bytes += 4;

        {
            std::lock_guard lock(preview_cache_->load_queue_mutex);
            PreviewLoadRequest request{
                .handle = handle,
                .path = path,
                .display_params = display_params,
                .preview_loader = preview_loader,
                .active_loader = std::move(active_loader),
                .high_priority = high_priority,
                .navigation_epoch = epoch};
            if (request.high_priority) {
                preview_cache_->high_priority_queue.push_back(std::move(request));
            } else {
                preview_cache_->low_priority_queue.push_back(std::move(request));
            }
        }
        preview_cache_->load_queue_cv.notify_all();

        dimensions.x = PLACEHOLDER_DIM;
        dimensions.y = PLACEHOLDER_DIM;
        return handle;
    }

} // namespace lfs::vis::gui
