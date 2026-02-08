/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rendering_manager.hpp"
#include "core/camera.hpp"
#include "core/image_io.hpp" // Use existing image_io utilities
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/splat_data.hpp"
#include "core/tensor/internal/memory_pool.hpp"
#include "geometry/euclidean_transform.hpp"
#include "rendering/cuda_kernels.hpp"
#include "rendering/rasterizer/rasterization/include/rasterization_api_tensor.h"
#include "rendering/rasterizer/rasterization/include/rasterization_config.h"
#include "rendering/rendering.hpp"
#include "rendering/rendering_pipeline.hpp"
#include "scene/scene_manager.hpp"
#include "training/components/ppisp.hpp"
#include "training/components/ppisp_controller.hpp"
#include "training/trainer.hpp"
#include "training/training_manager.hpp"
#include <cuda_runtime.h>
#include <glad/glad.h>
#include <glm/gtc/type_ptr.hpp>
#include <shared_mutex>
#include <stdexcept>

namespace lfs::vis {

    namespace {
        constexpr int GPU_ALIGNMENT = 16; // 16-pixel alignment for GPU texture efficiency

        lfs::training::PPISPRenderOverrides toRenderOverrides(const PPISPOverrides& ov) {
            lfs::training::PPISPRenderOverrides r;
            r.exposure_offset = ov.exposure_offset;
            r.vignette_enabled = ov.vignette_enabled;
            r.vignette_strength = ov.vignette_strength;
            r.wb_temperature = ov.wb_temperature;
            r.wb_tint = ov.wb_tint;
            r.color_red_x = ov.color_red_x;
            r.color_red_y = ov.color_red_y;
            r.color_green_x = ov.color_green_x;
            r.color_green_y = ov.color_green_y;
            r.color_blue_x = ov.color_blue_x;
            r.color_blue_y = ov.color_blue_y;
            r.gamma_multiplier = ov.gamma_multiplier;
            r.gamma_red = ov.gamma_red;
            r.gamma_green = ov.gamma_green;
            r.gamma_blue = ov.gamma_blue;
            r.crf_toe = ov.crf_toe;
            r.crf_shoulder = ov.crf_shoulder;
            return r;
        }

        lfs::core::Tensor applyStandaloneAppearance(const lfs::core::Tensor& rgb, Scene& scene,
                                                    const int camera_uid, const PPISPOverrides& overrides,
                                                    const bool use_controller = true) {
            auto* ppisp = scene.getAppearancePPISP();
            if (!ppisp) {
                return rgb;
            }

            const bool was_hwc = (rgb.ndim() == 3 && rgb.shape()[2] == 3);
            const auto input = was_hwc ? rgb.permute({2, 0, 1}).contiguous() : rgb;
            const bool is_training_camera = (camera_uid >= 0 && camera_uid < ppisp->num_frames());
            const bool has_controller = use_controller && scene.hasAppearanceController();

            lfs::core::Tensor result;

            if (has_controller) {
                auto* pool = scene.getAppearanceControllerPool();
                const int controller_idx = camera_uid >= 0 ? camera_uid % pool->num_cameras() : 0;
                const auto params = pool->predict(controller_idx, input.unsqueeze(0), 1.0f);
                result = overrides.isIdentity()
                             ? ppisp->apply_with_controller_params(input, params, 0)
                             : ppisp->apply_with_controller_params_and_overrides(input, params, 0,
                                                                                 toRenderOverrides(overrides));
            } else if (is_training_camera) {
                result = overrides.isIdentity() ? ppisp->apply(input, camera_uid, camera_uid)
                                                : ppisp->apply_with_overrides(input, camera_uid, camera_uid,
                                                                              toRenderOverrides(overrides));
            } else {
                const int fallback_camera = ppisp->any_camera_id();
                const int fallback_frame = ppisp->any_frame_uid();
                result = overrides.isIdentity() ? ppisp->apply(input, fallback_camera, fallback_frame)
                                                : ppisp->apply_with_overrides(input, fallback_camera, fallback_frame,
                                                                              toRenderOverrides(overrides));
            }

            return (was_hwc && result.is_valid()) ? result.permute({1, 2, 0}).contiguous() : result;
        }
    } // namespace

    using namespace lfs::core::events;

    GTTextureCache::GTTextureCache() {
        try {
            constexpr lfs::io::NvCodecImageLoader::Options OPTS{.device_id = 0, .decoder_pool_size = 2};
            nvcodec_loader_ = std::make_unique<lfs::io::NvCodecImageLoader>(OPTS);
        } catch (...) {
            nvcodec_loader_ = nullptr;
        }
    }

    GTTextureCache::~GTTextureCache() {
        clear();
    }

    void GTTextureCache::clear() {
        for (const auto& [id, entry] : texture_cache_) {
            if (!entry.interop_texture && entry.texture_id > 0) {
                glDeleteTextures(1, &entry.texture_id);
            }
        }
        glFinish();
        texture_cache_.clear();
    }

    GTTextureCache::TextureInfo GTTextureCache::getGTTexture(const int cam_id, const std::filesystem::path& image_path) {
        if (const auto it = texture_cache_.find(cam_id); it != texture_cache_.end()) {
            it->second.last_access = std::chrono::steady_clock::now();
            const auto& entry = it->second;
            const unsigned int tex_id = entry.interop_texture ? entry.interop_texture->getTextureID() : entry.texture_id;
            const glm::vec2 tex_scale = entry.interop_texture
                                            ? glm::vec2(entry.interop_texture->getTexcoordScaleX(),
                                                        entry.interop_texture->getTexcoordScaleY())
                                            : glm::vec2(1.0f);
            return {tex_id, entry.width, entry.height, entry.needs_flip, tex_scale};
        }

        if (texture_cache_.size() >= MAX_CACHE_SIZE)
            evictOldest();

        const auto ext = image_path.extension().string();
        const bool is_jpeg = (ext == ".jpg" || ext == ".jpeg" || ext == ".JPG" || ext == ".JPEG");

        auto& entry = texture_cache_[cam_id];
        entry.last_access = std::chrono::steady_clock::now();

        TextureInfo info{};
        if (nvcodec_loader_ && is_jpeg) {
            info = loadTextureGPU(image_path, entry);
        }

        if (info.texture_id == 0) {
            info = loadTexture(image_path);
            if (info.texture_id != 0) {
                entry.texture_id = info.texture_id;
                entry.width = info.width;
                entry.height = info.height;
                entry.needs_flip = false;
            }
        }

        if (info.texture_id == 0) {
            texture_cache_.erase(cam_id);
            return {};
        }

        return info;
    }

    void GTTextureCache::evictOldest() {
        if (texture_cache_.empty()) {
            return;
        }

        const auto oldest = std::min_element(texture_cache_.begin(), texture_cache_.end(),
                                             [](const auto& a, const auto& b) { return a.second.last_access < b.second.last_access; });

        if (oldest->second.interop_texture) {
            glFinish();
        } else if (oldest->second.texture_id != 0) {
            glDeleteTextures(1, &oldest->second.texture_id);
        }
        texture_cache_.erase(oldest);
    }

    GTTextureCache::TextureInfo GTTextureCache::loadTexture(const std::filesystem::path& path) {
        if (!std::filesystem::exists(path))
            return {};

        try {
            auto [data, width, height, channels] = lfs::core::load_image(path);
            if (!data)
                return {};

            int out_width = width;
            int out_height = height;
            int scale = 1;
            while (out_width > MAX_TEXTURE_DIM || out_height > MAX_TEXTURE_DIM) {
                out_width /= 2;
                out_height /= 2;
                scale *= 2;
            }

            std::vector<unsigned char> final_data(out_width * out_height * channels);
            const int scale_sq = scale * scale;

            if (scale > 1) {
                for (int y = 0; y < out_height; ++y) {
                    const int src_y = (out_height - 1 - y) * scale;
                    for (int x = 0; x < out_width; ++x) {
                        const int src_x = x * scale;
                        for (int c = 0; c < channels; ++c) {
                            int sum = 0;
                            for (int sy = 0; sy < scale; ++sy)
                                for (int sx = 0; sx < scale; ++sx)
                                    sum += data[((src_y + sy) * width + src_x + sx) * channels + c];
                            final_data[(y * out_width + x) * channels + c] =
                                static_cast<unsigned char>(sum / scale_sq);
                        }
                    }
                }
            } else {
                const size_t row_size = width * channels;
                for (int y = 0; y < height; ++y)
                    std::memcpy(final_data.data() + y * row_size,
                                data + (height - 1 - y) * row_size, row_size);
            }

            lfs::core::free_image(data);

            unsigned int texture;
            glGenTextures(1, &texture);
            glBindTexture(GL_TEXTURE_2D, texture);

            const GLenum format = (channels == 1) ? GL_RED : (channels == 4) ? GL_RGBA
                                                                             : GL_RGB;
            const GLenum internal = (channels == 1) ? GL_R8 : (channels == 4) ? GL_RGBA8
                                                                              : GL_RGB8;

            glTexImage2D(GL_TEXTURE_2D, 0, internal, out_width, out_height, 0,
                         format, GL_UNSIGNED_BYTE, final_data.data());
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

            return {texture, out_width, out_height};
        } catch (...) {
            return {};
        }
    }

    GTTextureCache::TextureInfo GTTextureCache::loadTextureGPU(const std::filesystem::path& path, CacheEntry& entry) {
        if (!nvcodec_loader_)
            return {};

        try {
            const auto tensor = nvcodec_loader_->load_image_gpu(path, 1, MAX_TEXTURE_DIM);
            if (tensor.numel() == 0)
                return {};

            const auto& shape = tensor.shape();
            const int height = static_cast<int>(shape[1]);
            const int width = static_cast<int>(shape[2]);

            const auto hwc = tensor.permute({1, 2, 0}).contiguous();

            entry.interop_texture = std::make_unique<lfs::rendering::CudaGLInteropTexture>();
            if (auto result = entry.interop_texture->init(width, height); !result) {
                LOG_WARN("Failed to init interop texture: {}", result.error());
                entry.interop_texture.reset();
                return {};
            }

            if (auto result = entry.interop_texture->updateFromTensor(hwc); !result) {
                LOG_WARN("Failed to upload to interop texture: {}", result.error());
                entry.interop_texture.reset();
                return {};
            }

            entry.width = width;
            entry.height = height;
            entry.needs_flip = true;

            const glm::vec2 tex_scale(
                entry.interop_texture->getTexcoordScaleX(),
                entry.interop_texture->getTexcoordScaleY());

            return {entry.interop_texture->getTextureID(), width, height, true, tex_scale};
        } catch (const std::exception& e) {
            LOG_WARN("GPU texture load failed: {}", e.what());
            entry.interop_texture.reset();
            return {};
        }
    }

    // RenderingManager Implementation
    RenderingManager::RenderingManager() {
        setupEventHandlers();
    }

    RenderingManager::~RenderingManager() {
        if (cached_render_texture_ > 0) {
            glDeleteTextures(1, &cached_render_texture_);
        }
        if (d_hovered_depth_id_ != nullptr) {
            cudaFree(d_hovered_depth_id_);
            d_hovered_depth_id_ = nullptr;
        }
    }

    void RenderingManager::initialize() {
        if (initialized_)
            return;

        LOG_TIMER("RenderingEngine initialization");

        engine_ = lfs::rendering::RenderingEngine::create();
        auto init_result = engine_->initialize();
        if (!init_result) {
            LOG_ERROR("Failed to initialize rendering engine: {}", init_result.error());
            throw std::runtime_error("Failed to initialize rendering engine: " + init_result.error());
        }

        // Create cached render texture
        glGenTextures(1, &cached_render_texture_);
        glBindTexture(GL_TEXTURE_2D, cached_render_texture_);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        initialized_ = true;
        LOG_INFO("Rendering engine initialized successfully");
    }

    void RenderingManager::setupEventHandlers() {
        // Listen for split view toggle
        cmd::ToggleSplitView::when([this](const auto&) {
            std::lock_guard<std::mutex> lock(settings_mutex_);

            // V key toggles between Disabled and PLYComparison only
            if (settings_.split_view_mode == SplitViewMode::PLYComparison) {
                settings_.split_view_mode = SplitViewMode::Disabled;
                LOG_INFO("Split view: disabled");
            } else {
                // From Disabled or GTComparison, go to PLYComparison
                settings_.split_view_mode = SplitViewMode::PLYComparison;
                LOG_INFO("Split view: PLY comparison mode");
            }

            settings_.split_view_offset = 0; // Reset when toggling
            markDirty();
        });

        cmd::ToggleGTComparison::when([this](const auto&) {
            bool is_now_enabled = false;
            std::optional<bool> restore_equirectangular;

            {
                std::lock_guard<std::mutex> lock(settings_mutex_);

                if (settings_.split_view_mode == SplitViewMode::GTComparison) {
                    settings_.split_view_mode = SplitViewMode::Disabled;
                    settings_.equirectangular = pre_gt_equirectangular_;
                    restore_equirectangular = pre_gt_equirectangular_;
                } else {
                    if (current_camera_id_ < 0)
                        return;
                    pre_gt_equirectangular_ = settings_.equirectangular;
                    settings_.split_view_mode = SplitViewMode::GTComparison;
                    is_now_enabled = true;
                }
                markDirty();
            }

            // Emit events outside the lock to avoid deadlock
            if (restore_equirectangular) {
                ui::RenderSettingsChanged{.equirectangular = *restore_equirectangular}.emit();
            }
            ui::GTComparisonModeChanged{.enabled = is_now_enabled}.emit();
        });

        // Listen for camera view changes
        cmd::GoToCamView::when([this](const auto& event) {
            setCurrentCameraId(event.cam_id);
            LOG_DEBUG("Current camera ID set to: {}", event.cam_id);

            if (settings_.split_view_mode == SplitViewMode::GTComparison && event.cam_id >= 0) {
                markDirty();
            }
        });

        // Listen for split position changes
        ui::SplitPositionChanged::when([this](const auto& event) {
            std::lock_guard<std::mutex> lock(settings_mutex_);
            settings_.split_position = event.position;
            LOG_TRACE("Split position changed to: {}", event.position);
            markDirty();
        });

        // Listen for settings changes
        ui::RenderSettingsChanged::when([this](const auto& event) {
            std::lock_guard<std::mutex> lock(settings_mutex_);
            if (event.sh_degree) {
                settings_.sh_degree = *event.sh_degree;
                LOG_TRACE("SH_DEGREE changed to: {}", settings_.sh_degree);
            }
            if (event.focal_length_mm) {
                settings_.focal_length_mm = *event.focal_length_mm;
                LOG_TRACE("Focal length changed to: {} mm", settings_.focal_length_mm);
            }
            if (event.scaling_modifier) {
                settings_.scaling_modifier = *event.scaling_modifier;
                LOG_TRACE("Scaling modifier changed to: {}", settings_.scaling_modifier);
            }
            if (event.antialiasing) {
                settings_.antialiasing = *event.antialiasing;
                LOG_TRACE("Antialiasing: {}", settings_.antialiasing ? "enabled" : "disabled");
            }
            if (event.background_color) {
                settings_.background_color = *event.background_color;
                LOG_TRACE("Background color changed");
            }
            if (event.equirectangular) {
                settings_.equirectangular = *event.equirectangular;
                LOG_TRACE("Equirectangular rendering: {}", settings_.equirectangular ? "enabled" : "disabled");
            }
            markDirty();
        });

        // Window resize
        ui::WindowResized::when([this](const auto&) {
            LOG_DEBUG("Window resized, clearing render cache");
            markDirty();
            cached_result_ = {};
            last_viewport_size_ = glm::ivec2(0, 0);
            render_texture_valid_ = false;
            gt_texture_cache_.clear();
        });

        // Grid settings
        ui::GridSettingsChanged::when([this](const auto& event) {
            std::lock_guard<std::mutex> lock(settings_mutex_);
            settings_.show_grid = event.enabled;
            settings_.grid_plane = event.plane;
            settings_.grid_opacity = event.opacity;
            LOG_TRACE("Grid settings updated - enabled: {}, plane: {}, opacity: {}",
                      event.enabled, event.plane, event.opacity);
            markDirty();
        });

        ui::NodeSelected::when([this](const auto&) { triggerSelectionFlash(); });

        // Scene changes
        state::SceneLoaded::when([this](const auto&) {
            LOG_DEBUG("Scene loaded, marking render dirty");
            markDirty();
            gt_texture_cache_.clear(); // Clear GT cache when scene changes

            // Reset current camera ID when loading a new scene
            current_camera_id_ = -1;

            // If GT comparison is enabled but we lost the camera, disable it
            if (settings_.split_view_mode == SplitViewMode::GTComparison) {
                LOG_INFO("Scene loaded, disabling GT comparison (camera selection reset)");
                settings_.split_view_mode = SplitViewMode::Disabled;
            }
        });

        state::SceneChanged::when([this](const auto&) {
            cached_filtered_point_cloud_.reset();
            cached_source_point_cloud_ = nullptr;
            markDirty();
        });

        state::SceneCleared::when([this](const auto&) {
            cached_result_ = {};
            cached_filtered_point_cloud_.reset();
            cached_source_point_cloud_ = nullptr;
            render_texture_valid_ = false;
            gt_texture_cache_.clear();
            if (engine_) {
                engine_->clearFrustumCache();
            }
            current_camera_id_ = -1;
            last_model_ptr_ = 0;
            markDirty();
        });

        // PLY visibility changes
        cmd::SetPLYVisibility::when([this](const auto&) {
            markDirty();
        });

        // PLY added/removed
        state::PLYAdded::when([this](const auto&) {
            LOG_DEBUG("PLY added, marking render dirty");
            markDirty();
        });

        state::PLYRemoved::when([this](const auto&) {
            std::lock_guard<std::mutex> lock(settings_mutex_);

            // If in PLY comparison mode, check if we still have enough nodes
            if (settings_.split_view_mode == SplitViewMode::PLYComparison) {
                auto* scene_manager = services().sceneOrNull();
                if (scene_manager) {
                    auto visible_nodes = scene_manager->getScene().getVisibleNodes();
                    if (visible_nodes.size() < 2) {
                        LOG_DEBUG("PLY removed, disabling split view (not enough PLYs)");
                        settings_.split_view_mode = SplitViewMode::Disabled;
                        settings_.split_view_offset = 0;
                    }
                }
            }

            markDirty();
        });

        // Crop box changes (scene graph is source of truth, this just handles enable flag)
        ui::CropBoxChanged::when([this](const auto& event) {
            std::lock_guard<std::mutex> lock(settings_mutex_);
            settings_.use_crop_box = event.enabled;
            markDirty();
        });

        // Ellipsoid changes (scene graph is source of truth, this just handles enable flag)
        ui::EllipsoidChanged::when([this](const auto& event) {
            std::lock_guard<std::mutex> lock(settings_mutex_);
            settings_.use_ellipsoid = event.enabled;
            markDirty();
        });

        // Point cloud mode changes
        ui::PointCloudModeChanged::when([this](const auto& event) {
            std::lock_guard<std::mutex> lock(settings_mutex_);
            settings_.point_cloud_mode = event.enabled;
            settings_.voxel_size = event.voxel_size;
            LOG_DEBUG("Point cloud mode: {}, voxel size: {}",
                      event.enabled ? "enabled" : "disabled", event.voxel_size);
            cached_result_ = {};
            markDirty();
        });
    }

    void RenderingManager::markDirty() {
        needs_render_ = true;
        render_texture_valid_ = false;
        LOG_TRACE("Render marked dirty");
    }

    void RenderingManager::updateSettings(const RenderSettings& new_settings) {
        std::lock_guard<std::mutex> lock(settings_mutex_);

        // Update preview color if changed
        if (settings_.selection_color_preview != new_settings.selection_color_preview) {
            const auto& p = new_settings.selection_color_preview;
            lfs::rendering::config::setSelectionPreviewColor(make_float3(p.x, p.y, p.z));
        }

        // Update center marker color (group 0) if changed
        if (settings_.selection_color_center_marker != new_settings.selection_color_center_marker) {
            const auto& m = new_settings.selection_color_center_marker;
            lfs::rendering::config::setSelectionGroupColor(0, make_float3(m.x, m.y, m.z));
        }

        settings_ = new_settings;
        markDirty();
    }

    RenderSettings RenderingManager::getSettings() const {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        return settings_;
    }

    void RenderingManager::setOrthographic(const bool enabled, const float viewport_height, const float distance_to_pivot) {
        std::lock_guard<std::mutex> lock(settings_mutex_);

        // Calculate ortho_scale to preserve apparent size at pivot distance
        if (enabled && !settings_.orthographic) {
            constexpr float MIN_DISTANCE = 0.01f;
            constexpr float MIN_SCALE = 1.0f;
            constexpr float MAX_SCALE = 10000.0f;
            constexpr float DEFAULT_SCALE = 100.0f;

            if (viewport_height <= 0.0f || distance_to_pivot <= MIN_DISTANCE) {
                LOG_WARN("setOrthographic: invalid viewport_height={} or distance={}", viewport_height, distance_to_pivot);
                settings_.ortho_scale = DEFAULT_SCALE;
            } else {
                const float vfov = lfs::rendering::focalLengthToVFov(settings_.focal_length_mm);
                const float half_tan_fov = std::tan(glm::radians(vfov) * 0.5f);
                settings_.ortho_scale = std::clamp(
                    viewport_height / (2.0f * distance_to_pivot * half_tan_fov),
                    MIN_SCALE, MAX_SCALE);
            }
        }

        settings_.orthographic = enabled;
        needs_render_.store(true);
    }

    float RenderingManager::getFovDegrees() const {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        return lfs::rendering::focalLengthToVFov(settings_.focal_length_mm);
    }

    float RenderingManager::getFocalLengthMm() const {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        return settings_.focal_length_mm;
    }

    void RenderingManager::setFocalLength(const float focal_mm) {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        settings_.focal_length_mm = std::clamp(focal_mm,
                                               lfs::rendering::MIN_FOCAL_LENGTH_MM,
                                               lfs::rendering::MAX_FOCAL_LENGTH_MM);
        markDirty();
    }

    float RenderingManager::getScalingModifier() const {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        return settings_.scaling_modifier;
    }

    void RenderingManager::setScalingModifier(const float s) {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        settings_.scaling_modifier = s;
        markDirty();
    }

    void RenderingManager::syncSelectionGroupColor(const int group_id, const glm::vec3& color) {
        lfs::rendering::config::setSelectionGroupColor(group_id, make_float3(color.x, color.y, color.z));
        markDirty();
    }

    void RenderingManager::advanceSplitOffset() {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        settings_.split_view_offset++;
        markDirty();
    }

    SplitViewInfo RenderingManager::getSplitViewInfo() const {
        std::lock_guard<std::mutex> lock(split_info_mutex_);
        return current_split_info_;
    }

    RenderingManager::ContentBounds RenderingManager::getContentBounds(const glm::ivec2& viewport_size) const {
        ContentBounds bounds{0.0f, 0.0f, static_cast<float>(viewport_size.x), static_cast<float>(viewport_size.y), false};

        if (settings_.split_view_mode == SplitViewMode::GTComparison && gt_context_ && gt_context_->valid()) {
            const float content_aspect = static_cast<float>(gt_context_->dimensions.x) / gt_context_->dimensions.y;
            const float viewport_aspect = static_cast<float>(viewport_size.x) / viewport_size.y;

            if (content_aspect > viewport_aspect) {
                bounds.width = static_cast<float>(viewport_size.x);
                bounds.height = viewport_size.x / content_aspect;
                bounds.x = 0.0f;
                bounds.y = (viewport_size.y - bounds.height) / 2.0f;
            } else {
                bounds.height = static_cast<float>(viewport_size.y);
                bounds.width = viewport_size.y * content_aspect;
                bounds.x = (viewport_size.x - bounds.width) / 2.0f;
                bounds.y = 0.0f;
            }
            bounds.letterboxed = true;
        }
        return bounds;
    }

    lfs::rendering::RenderingEngine* RenderingManager::getRenderingEngine() {
        if (!initialized_) {
            initialize();
        }
        return engine_.get();
    }

    int RenderingManager::pickCameraFrustum(const glm::vec2& mouse_pos) {
        // Throttle picking to avoid excessive calls
        auto now = std::chrono::steady_clock::now();
        if (now - last_pick_time_ < pick_throttle_interval_) {
            return hovered_camera_id_; // Return cached value
        }
        last_pick_time_ = now;

        pending_pick_pos_ = mouse_pos;
        pick_requested_ = true;

        pick_count_++;
        LOG_TRACE("Pick #{} requested at ({}, {}), current hover: {}",
                  pick_count_, mouse_pos.x, mouse_pos.y, hovered_camera_id_);

        return hovered_camera_id_; // Return current value
    }

    void RenderingManager::renderToTexture(const RenderContext& context, SceneManager* scene_manager, const lfs::core::SplatData* model) {
        LOG_TIMER_TRACE("RenderingManager::renderToTexture");
        if (!model || model->size() == 0) {
            render_texture_valid_ = false;
            return;
        }

        glm::ivec2 viewport_size = context.viewport.windowSize;
        if (context.viewport_region) {
            viewport_size = glm::ivec2(
                static_cast<int>(context.viewport_region->width),
                static_cast<int>(context.viewport_region->height));
        }

        const float scale = std::clamp(settings_.render_scale, 0.25f, 1.0f);
        glm::ivec2 render_size(
            static_cast<int>(viewport_size.x * scale),
            static_cast<int>(viewport_size.y * scale));

        if (settings_.split_view_mode == SplitViewMode::GTComparison && gt_context_ && gt_context_->valid()) {
            render_size = gt_context_->dimensions;
        }

        const glm::ivec2 alloc_size(
            ((render_size.x + GPU_ALIGNMENT - 1) / GPU_ALIGNMENT) * GPU_ALIGNMENT,
            ((render_size.y + GPU_ALIGNMENT - 1) / GPU_ALIGNMENT) * GPU_ALIGNMENT);

        static glm::ivec2 texture_size{0, 0};
        if (alloc_size != texture_size) {
            glBindTexture(GL_TEXTURE_2D, cached_render_texture_);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, alloc_size.x, alloc_size.y,
                         0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
            LOG_DEBUG("Render texture resize: {}x{} -> {}x{}", texture_size.x, texture_size.y, alloc_size.x, alloc_size.y);
            texture_size = alloc_size;
        }

        static GLuint render_fbo = 0;
        static GLuint render_depth_rbo = 0;
        static glm::ivec2 depth_buffer_size{0, 0};

        if (render_fbo == 0) {
            glGenFramebuffers(1, &render_fbo);
            glGenRenderbuffers(1, &render_depth_rbo);
        }

        GLint current_fbo;
        glGetIntegerv(GL_FRAMEBUFFER_BINDING, &current_fbo);

        glBindFramebuffer(GL_FRAMEBUFFER, render_fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, cached_render_texture_, 0);

        if (alloc_size != depth_buffer_size) {
            glBindRenderbuffer(GL_RENDERBUFFER, render_depth_rbo);
            glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, alloc_size.x, alloc_size.y);
            LOG_DEBUG("Depth buffer resize: {}x{}", alloc_size.x, alloc_size.y);
            depth_buffer_size = alloc_size;
        }
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, render_depth_rbo);

        const GLenum fb_status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
        if (fb_status != GL_FRAMEBUFFER_COMPLETE) {
            LOG_ERROR("FBO incomplete: 0x{:x}", fb_status);
            glBindFramebuffer(GL_FRAMEBUFFER, current_fbo);
            render_texture_valid_ = false;
            return;
        }

        glViewport(0, 0, render_size.x, render_size.y);
        glClearColor(settings_.background_color.r, settings_.background_color.g, settings_.background_color.b, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        lfs::rendering::ViewportData viewport_data{
            .rotation = context.viewport.getRotationMatrix(),
            .translation = context.viewport.getTranslation(),
            .size = render_size,
            .focal_length_mm = settings_.focal_length_mm,
            .orthographic = settings_.orthographic,
            .ortho_scale = settings_.ortho_scale};

        // Build render state from scene (single source of truth)
        lfs::vis::SceneRenderState scene_state;
        if (scene_manager) {
            scene_state = scene_manager->buildRenderState();
        }

        lfs::rendering::RenderRequest request{
            .viewport = viewport_data,
            .scaling_modifier = settings_.scaling_modifier,
            .antialiasing = settings_.antialiasing,
            .mip_filter = settings_.mip_filter,
            .sh_degree = settings_.sh_degree,
            .background_color = settings_.background_color,
            .crop_box = std::nullopt,
            .point_cloud_mode = settings_.point_cloud_mode,
            .voxel_size = settings_.voxel_size,
            .gut = settings_.gut,
            .equirectangular = settings_.equirectangular,
            .show_rings = settings_.show_rings,
            .ring_width = settings_.ring_width,
            .show_center_markers = settings_.show_center_markers,
            .model_transforms = std::move(scene_state.model_transforms),
            .transform_indices = scene_state.transform_indices,
            .selection_mask = scene_state.selection_mask,
            .output_screen_positions = output_screen_positions_,
            .brush_active = brush_active_,
            .brush_x = brush_x_,
            .brush_y = brush_y_,
            .brush_radius = brush_radius_,
            .brush_add_mode = brush_add_mode_,
            .brush_selection_tensor = preview_selection_ ? preview_selection_ : brush_selection_tensor_,
            .brush_saturation_mode = brush_saturation_mode_,
            .brush_saturation_amount = brush_saturation_amount_,
            .selection_mode_rings = (selection_mode_ == lfs::rendering::SelectionMode::Rings),
            .selected_node_mask = (settings_.desaturate_unselected || getSelectionFlashIntensity() > 0.0f)
                                      ? std::move(scene_state.selected_node_mask)
                                      : std::vector<bool>{},
            .node_visibility_mask = std::move(scene_state.node_visibility_mask),
            .desaturate_unselected = settings_.desaturate_unselected,
            .selection_flash_intensity = getSelectionFlashIntensity(),
            .hovered_depth_id = nullptr,
            .highlight_gaussian_id = (selection_mode_ == lfs::rendering::SelectionMode::Rings) ? hovered_gaussian_id_ : -1,
            .far_plane = settings_.depth_clip_enabled ? settings_.depth_clip_far : lfs::rendering::DEFAULT_FAR_PLANE,
            .orthographic = settings_.orthographic,
            .ortho_scale = settings_.ortho_scale};

        // Ring mode hover preview: allocate device buffer if needed
        const bool need_hovered_output = (selection_mode_ == lfs::rendering::SelectionMode::Rings) && brush_active_;
        if (need_hovered_output) {
            if (d_hovered_depth_id_ == nullptr) {
                cudaMalloc(&d_hovered_depth_id_, sizeof(unsigned long long));
            }
            // Initialize to max value (atomicMin finds minimum)
            constexpr unsigned long long init_val = 0xFFFFFFFFFFFFFFFFULL;
            cudaMemcpy(d_hovered_depth_id_, &init_val, sizeof(unsigned long long), cudaMemcpyHostToDevice);
            request.hovered_depth_id = d_hovered_depth_id_;
        }

        // Crop box from scene graph (single source of truth)
        if (settings_.use_crop_box || settings_.show_crop_box) {
            const auto& cropboxes = scene_state.cropboxes;
            const size_t idx = (scene_state.selected_cropbox_index >= 0)
                                   ? static_cast<size_t>(scene_state.selected_cropbox_index)
                                   : 0;

            if (idx < cropboxes.size() && cropboxes[idx].data) {
                const auto& cb = cropboxes[idx];
                request.crop_box = lfs::rendering::BoundingBox{
                    .min = cb.data->min,
                    .max = cb.data->max,
                    .transform = glm::inverse(cb.world_transform)};
                request.crop_inverse = cb.data->inverse;
                request.crop_desaturate = settings_.show_crop_box && !settings_.use_crop_box && settings_.desaturate_cropping;
                request.crop_parent_node_index = scene_manager->getScene().getVisibleNodeIndex(cb.parent_splat_id);
            }
        }

        // Ellipsoid from scene graph
        if (settings_.use_ellipsoid || settings_.show_ellipsoid) {
            const auto& scene = scene_manager->getScene();
            const auto visible_ellipsoids = scene.getVisibleEllipsoids();
            const NodeId selected_ellipsoid_id = scene_manager->getSelectedNodeEllipsoidId();
            for (const auto& el : visible_ellipsoids) {
                if (!el.data)
                    continue;
                if (selected_ellipsoid_id != NULL_NODE && el.node_id != selected_ellipsoid_id)
                    continue;
                request.ellipsoid = lfs::rendering::Ellipsoid{
                    .radii = el.data->radii,
                    .transform = glm::inverse(el.world_transform)};
                request.ellipsoid_inverse = el.data->inverse;
                request.ellipsoid_desaturate = settings_.show_ellipsoid && !settings_.use_ellipsoid && settings_.desaturate_cropping;
                request.ellipsoid_parent_node_index = scene.getVisibleNodeIndex(el.parent_splat_id);
                break;
            }
        }

        // Add depth filter (Selection tool only - separate from crop box)
        // Depth filter always desaturates outside, never actually filters
        if (settings_.depth_filter_enabled) {
            request.depth_filter = lfs::rendering::BoundingBox{
                .min = settings_.depth_filter_min,
                .max = settings_.depth_filter_max,
                .transform = settings_.depth_filter_transform.inv().toMat4()};
        }

        // Lock only during CUDA rasterization to minimize training blocking
        std::optional<std::shared_lock<std::shared_mutex>> render_lock;
        if (const auto* tm = scene_manager ? scene_manager->getTrainerManager() : nullptr) {
            if (const auto* trainer = tm->getTrainer()) {
                render_lock.emplace(trainer->getRenderMutex());
            }
        }

        auto render_result = engine_->renderGaussians(*model, request);

        // Apply PPISP correction if enabled via checkbox
        if (render_result && render_result->image && settings_.apply_appearance_correction) {
            bool applied = false;

            // Try trainer's PPISP first (has per-frame params and knows training cameras)
            if (const auto* tm = scene_manager ? scene_manager->getTrainerManager() : nullptr) {
                if (const auto* trainer = tm->getTrainer(); trainer && trainer->hasPPISP()) {
                    lfs::training::PPISPViewportOverrides trainer_overrides{};
                    if (settings_.ppisp_mode == RenderSettings::PPISPMode::MANUAL) {
                        trainer_overrides.exposure_offset = settings_.ppisp_overrides.exposure_offset;
                        trainer_overrides.vignette_enabled = settings_.ppisp_overrides.vignette_enabled;
                        trainer_overrides.vignette_strength = settings_.ppisp_overrides.vignette_strength;
                        trainer_overrides.wb_temperature = settings_.ppisp_overrides.wb_temperature;
                        trainer_overrides.wb_tint = settings_.ppisp_overrides.wb_tint;
                        trainer_overrides.gamma_multiplier = settings_.ppisp_overrides.gamma_multiplier;
                    }
                    const bool use_controller = (settings_.ppisp_mode == RenderSettings::PPISPMode::AUTO);
                    auto corrected = trainer->applyPPISPForViewport(
                        *render_result->image, current_camera_id_, trainer_overrides, use_controller);
                    render_result->image = std::make_shared<lfs::core::Tensor>(std::move(corrected));
                    applied = true;
                }
            }

            if (!applied && scene_manager) {
                auto& scene = scene_manager->getScene();
                if (scene.hasAppearanceModel()) {
                    const auto& overrides = (settings_.ppisp_mode == RenderSettings::PPISPMode::MANUAL)
                                                ? settings_.ppisp_overrides
                                                : PPISPOverrides{};
                    const bool use_controller = (settings_.ppisp_mode == RenderSettings::PPISPMode::AUTO);
                    auto corrected = applyStandaloneAppearance(
                        *render_result->image, scene, current_camera_id_, overrides, use_controller);
                    if (corrected.is_valid()) {
                        render_result->image = std::make_shared<lfs::core::Tensor>(std::move(corrected));
                    }
                }
            }
        }

        render_lock.reset();

        if (render_result) {
            cached_result_ = *render_result;

            // Copy packed depth+id back and extract gaussian ID
            if (need_hovered_output) {
                cudaMemcpy(&hovered_depth_id_, d_hovered_depth_id_, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
                // Extract gaussian ID from lower 32 bits; -1 if no hit (max value)
                if (hovered_depth_id_ == 0xFFFFFFFFFFFFFFFFULL) {
                    hovered_gaussian_id_ = -1;
                } else {
                    hovered_gaussian_id_ = static_cast<int>(hovered_depth_id_ & 0xFFFFFFFF);
                }
            }

            // Store the actual size at which this result was rendered
            cached_result_size_ = render_size;

            // For GT comparison, present to the bound FBO to fill cached_render_texture_
            if (settings_.split_view_mode == SplitViewMode::GTComparison) {
                const auto present_result = engine_->presentToScreen(cached_result_, glm::ivec2(0), render_size);
                render_texture_valid_ = present_result.has_value();
            } else {
                render_texture_valid_ = true;
            }
        } else {
            LOG_ERROR("Failed to render gaussians: {}", render_result.error());
            render_texture_valid_ = false;
            cached_result_size_ = {0, 0};
        }

        glBindFramebuffer(GL_FRAMEBUFFER, current_fbo);
    }

    bool RenderingManager::renderPreviewFrame(SceneManager* const scene_manager,
                                              const glm::mat3& rotation,
                                              const glm::vec3& position,
                                              const float fov,
                                              const unsigned int fbo,
                                              [[maybe_unused]] const unsigned int texture,
                                              const int width, const int height) {
        if (!initialized_ || !engine_)
            return false;

        const auto* const model = scene_manager ? scene_manager->getModelForRendering() : nullptr;
        if (!model || model->size() == 0)
            return false;

        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glViewport(0, 0, width, height);
        const auto& bg = settings_.background_color;
        glClearColor(bg.r, bg.g, bg.b, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        const lfs::rendering::RenderRequest request{
            .viewport = {rotation, position, {width, height}, fov, false, 1.0f},
            .scaling_modifier = settings_.scaling_modifier,
            .antialiasing = false,
            .sh_degree = 0,
            .background_color = bg,
            .crop_box = std::nullopt,
            .point_cloud_mode = settings_.point_cloud_mode,
            .voxel_size = settings_.voxel_size,
            .gut = false,
            .show_rings = false,
            .ring_width = 0.0f,
            .show_center_markers = false};

        if (const auto result = engine_->renderGaussians(*model, request)) {
            engine_->presentToScreen(*result, {0, 0}, {width, height});
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            return true;
        }

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        return false;
    }

    void RenderingManager::renderFrame(const RenderContext& context, SceneManager* scene_manager) {
        if (!initialized_) {
            initialize();
        }

        // Sync selection group colors to GPU constant memory
        if (scene_manager) {
            for (const auto& group : scene_manager->getScene().getSelectionGroups()) {
                lfs::rendering::config::setSelectionGroupColor(
                    group.id, make_float3(group.color.x, group.color.y, group.color.z));
            }
        }

        // Calculate current render size
        glm::ivec2 current_size = context.viewport.windowSize;
        if (context.viewport_region) {
            current_size = glm::ivec2(
                static_cast<int>(context.viewport_region->width),
                static_cast<int>(context.viewport_region->height));
        }

        // SAFETY CHECK: Don't render with invalid viewport dimensions
        if (current_size.x <= 0 || current_size.y <= 0) {
            LOG_TRACE("Skipping render - invalid viewport size: {}x{}", current_size.x, current_size.y);
            // Still clear to prevent trails
            glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            return;
        }

        // Track viewport size changes
        if (current_size != last_viewport_size_) {
            LOG_DEBUG("Viewport resize: {}x{} -> {}x{}", last_viewport_size_.x, last_viewport_size_.y,
                      current_size.x, current_size.y);
            needs_render_ = true;
            last_viewport_size_ = current_size;
        }

        const lfs::core::SplatData* const model = scene_manager ? scene_manager->getModelForRendering() : nullptr;
        const size_t model_ptr = reinterpret_cast<size_t>(model);

        // Render mutex acquired in renderToTexture() during CUDA rasterization only

        if (model_ptr != last_model_ptr_) {
            LOG_DEBUG("Model ptr changed: {} -> {}, size={}", last_model_ptr_, model_ptr, model ? model->size() : 0);
            needs_render_ = true;
            render_texture_valid_ = false;
            last_model_ptr_ = model_ptr;
            cached_result_ = {};
        }

        bool split_view_active = settings_.split_view_mode != SplitViewMode::Disabled;
        bool should_render = false;
        const bool needs_render_now = needs_render_.load();
        const bool is_training = scene_manager && scene_manager->hasDataset() &&
                                 scene_manager->getTrainerManager() &&
                                 scene_manager->getTrainerManager()->isRunning();

        // Invalidate render cache periodically during training
        if (is_training) {
            const auto now = std::chrono::steady_clock::now();
            const auto interval = std::chrono::duration<float>(
                framerate_controller_.getSettings().training_frame_refresh_time_sec);
            if (now - last_training_render_ > interval) {
                should_render = true;
                render_texture_valid_ = false;
                last_training_render_ = now;
            }
        }

        if (settings_.split_view_mode == SplitViewMode::GTComparison) {
            if (current_camera_id_ < 0) {
                split_view_active = false;
                gt_context_.reset();
                gt_context_camera_id_ = -1;
            } else if (model) {
                if (gt_context_camera_id_ != current_camera_id_ || !gt_context_) {
                    gt_context_.reset();
                    gt_context_camera_id_ = -1;

                    if (auto* trainer_manager = scene_manager->getTrainerManager()) {
                        if (trainer_manager->hasTrainer()) {
                            if (const auto cam = trainer_manager->getCamById(current_camera_id_)) {
                                const auto gt_info = gt_texture_cache_.getGTTexture(current_camera_id_, cam->image_path());
                                if (gt_info.texture_id != 0) {
                                    const glm::ivec2 dims(gt_info.width, gt_info.height);
                                    const glm::ivec2 aligned(
                                        ((dims.x + GPU_ALIGNMENT - 1) / GPU_ALIGNMENT) * GPU_ALIGNMENT,
                                        ((dims.y + GPU_ALIGNMENT - 1) / GPU_ALIGNMENT) * GPU_ALIGNMENT);

                                    gt_context_ = GTComparisonContext{
                                        .gt_texture_id = gt_info.texture_id,
                                        .dimensions = dims,
                                        .gpu_aligned_dims = aligned,
                                        .render_texcoord_scale = glm::vec2(dims) / glm::vec2(aligned),
                                        .gt_texcoord_scale = gt_info.texcoord_scale,
                                        .gt_needs_flip = gt_info.needs_flip};
                                    gt_context_camera_id_ = current_camera_id_;
                                }
                            }
                        }
                    }
                }

                if (gt_context_ && !render_texture_valid_) {
                    renderToTexture(context, scene_manager, model);
                }
            }
        } else {
            if (gt_context_) {
                gt_context_.reset();
                gt_context_camera_id_ = -1;
            }
        }

        if (!cached_result_.image || needs_render_now || split_view_active) {
            should_render = true;
            needs_render_ = false;
        }

        glViewport(0, 0, context.viewport.frameBufferSize.x, context.viewport.frameBufferSize.y);

        // Clear full framebuffer first to avoid artifacts in gaps between UI elements
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // Set viewport region with scissor clipping (Y flipped for OpenGL)
        if (context.viewport_region) {
            const GLint x = static_cast<GLint>(context.viewport_region->x);
            const GLint y = context.viewport.frameBufferSize.y - static_cast<GLint>(context.viewport_region->y + context.viewport_region->height);
            const GLsizei w = static_cast<GLsizei>(context.viewport_region->width);
            const GLsizei h = static_cast<GLsizei>(context.viewport_region->height);
            glViewport(x, y, w, h);
            glScissor(x, y, w, h);
            glEnable(GL_SCISSOR_TEST);
        }

        if (should_render || !model) {
            doFullRender(context, scene_manager, model);
        } else if (cached_result_.image && cached_result_size_.x > 0 && cached_result_size_.y > 0) {
            // Use cached result - display at current viewport size (upscaling if needed)
            glm::ivec2 viewport_pos(0, 0);
            glm::ivec2 display_size = current_size;

            if (context.viewport_region) {
                const int gl_y = context.viewport.frameBufferSize.y - static_cast<int>(context.viewport_region->y) - static_cast<int>(context.viewport_region->height);
                viewport_pos = glm::ivec2(static_cast<int>(context.viewport_region->x), gl_y);
                display_size = glm::ivec2(static_cast<int>(context.viewport_region->width),
                                          static_cast<int>(context.viewport_region->height));
            } else {
                glViewport(viewport_pos.x, viewport_pos.y, display_size.x, display_size.y);
            }
            glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            // Pass IMAGE size for upload validation
            engine_->presentToScreen(cached_result_, viewport_pos, cached_result_size_);
            renderOverlays(context);
        }

        if (context.viewport_region) {
            glDisable(GL_SCISSOR_TEST);
        }
    }

    void RenderingManager::doFullRender(const RenderContext& context, SceneManager* scene_manager, const lfs::core::SplatData* model) {
        const bool count_frame = model != nullptr;
        if (count_frame) {
            framerate_controller_.beginFrame();
        }
        LOG_TIMER_TRACE("RenderingManager::doFullRender");

        render_count_++;
        LOG_TRACE("Render #{}, pick_requested: {}", render_count_, pick_requested_);

        glm::ivec2 render_size = context.viewport.windowSize;
        if (context.viewport_region) {
            render_size = glm::ivec2(
                static_cast<int>(context.viewport_region->width),
                static_cast<int>(context.viewport_region->height));
        }

        glClearColor(settings_.background_color.r, settings_.background_color.g,
                     settings_.background_color.b, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Check for split view
        if (auto split_request = createSplitViewRequest(context, scene_manager)) {
            // Update split info
            {
                std::lock_guard<std::mutex> lock(split_info_mutex_);
                current_split_info_.enabled = true;
                if (split_request->panels.size() >= 2) {
                    current_split_info_.left_name = split_request->panels[0].label;
                    current_split_info_.right_name = split_request->panels[1].label;
                }
            }

            std::optional<std::shared_lock<std::shared_mutex>> render_lock;
            if (const auto* tm = scene_manager ? scene_manager->getTrainerManager() : nullptr) {
                if (const auto* trainer = tm->getTrainer()) {
                    render_lock.emplace(trainer->getRenderMutex());
                }
            }

            auto result = engine_->renderSplitView(*split_request);
            render_lock.reset();

            if (result) {
                cached_result_ = *result;
                // Store viewport size for coordinate calculations in getDepthAtPixel
                cached_result_size_ = render_size;
            } else {
                LOG_ERROR("Failed to render split view: {}", result.error());
                cached_result_size_ = {0, 0};
            }

            renderOverlays(context);
            if (count_frame) {
                framerate_controller_.endFrame();
            }
            return;
        }

        // Clear split info if not in split view
        {
            std::lock_guard<std::mutex> lock(split_info_mutex_);
            current_split_info_ = SplitViewInfo{};
        }

        // For non-split view, render to texture first (for potential reuse)
        if (model && model->size() > 0) {
            renderToTexture(context, scene_manager, model);

            if (render_texture_valid_ && cached_result_size_.x > 0 && cached_result_size_.y > 0) {
                glm::ivec2 viewport_pos(0, 0);
                if (context.viewport_region) {
                    const int gl_y = context.viewport.frameBufferSize.y - static_cast<int>(context.viewport_region->y) - static_cast<int>(context.viewport_region->height);
                    viewport_pos = glm::ivec2(static_cast<int>(context.viewport_region->x), gl_y);
                }

                glViewport(viewport_pos.x, viewport_pos.y, render_size.x, render_size.y);
                glClearColor(settings_.background_color.r, settings_.background_color.g,
                             settings_.background_color.b, 1.0f);
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

                const auto present_result = engine_->presentToScreen(
                    cached_result_,
                    viewport_pos,
                    cached_result_size_);
                if (!present_result) {
                    LOG_ERROR("Failed to present render result: {}", present_result.error());
                }
            }
        } else if (scene_manager) {
            // No splat model - try to render point cloud (pre-training mode only)
            auto scene_state = scene_manager->buildRenderState();

            // Invalidate point cloud cache if source removed
            if (!scene_state.point_cloud && cached_source_point_cloud_) {
                cached_filtered_point_cloud_.reset();
                cached_source_point_cloud_ = nullptr;
            }

            if (scene_state.point_cloud && scene_state.point_cloud->size() > 0) {
                const lfs::core::PointCloud* point_cloud_to_render = scene_state.point_cloud;

                // Apply cropbox filter (GPU-accelerated, cached)
                for (const auto& cb : scene_state.cropboxes) {
                    if (!cb.data || (!cb.data->enabled && !settings_.show_crop_box))
                        continue;

                    const bool cache_valid = cached_filtered_point_cloud_ &&
                                             cached_source_point_cloud_ == scene_state.point_cloud &&
                                             cached_cropbox_transform_ == cb.world_transform &&
                                             cached_cropbox_min_ == cb.data->min &&
                                             cached_cropbox_max_ == cb.data->max &&
                                             cached_cropbox_inverse_ == cb.data->inverse;

                    if (!cache_valid) {
                        const auto& means = scene_state.point_cloud->means;
                        const auto& colors = scene_state.point_cloud->colors;
                        const size_t num_points = scene_state.point_cloud->size();
                        const glm::mat4 m = glm::inverse(cb.world_transform);
                        const auto device = means.device();

                        // GLM column-major -> row-major for tensor matmul
                        const auto transform = lfs::core::Tensor::from_vector({m[0][0], m[1][0], m[2][0], m[3][0],
                                                                               m[0][1], m[1][1], m[2][1], m[3][1],
                                                                               m[0][2], m[1][2], m[2][2], m[3][2],
                                                                               m[0][3], m[1][3], m[2][3], m[3][3]},
                                                                              {4, 4}, device);

                        // Transform and filter on GPU
                        const auto ones = lfs::core::Tensor::ones({num_points, 1}, device);
                        const auto local_pos = transform.mm(means.cat(ones, 1).t()).t();

                        const auto x = local_pos.slice(1, 0, 1).squeeze(1);
                        const auto y = local_pos.slice(1, 1, 2).squeeze(1);
                        const auto z = local_pos.slice(1, 2, 3).squeeze(1);

                        auto mask = (x >= cb.data->min.x) && (x <= cb.data->max.x) &&
                                    (y >= cb.data->min.y) && (y <= cb.data->max.y) &&
                                    (z >= cb.data->min.z) && (z <= cb.data->max.z);
                        if (cb.data->inverse)
                            mask = mask.logical_not();

                        const auto indices = mask.nonzero().squeeze(1);
                        if (indices.size(0) > 0) {
                            cached_filtered_point_cloud_ = std::make_unique<lfs::core::PointCloud>(
                                means.index_select(0, indices), colors.index_select(0, indices));
                        } else {
                            cached_filtered_point_cloud_.reset();
                        }

                        cached_source_point_cloud_ = scene_state.point_cloud;
                        cached_cropbox_transform_ = cb.world_transform;
                        cached_cropbox_min_ = cb.data->min;
                        cached_cropbox_max_ = cb.data->max;
                        cached_cropbox_inverse_ = cb.data->inverse;
                    }

                    if (cached_filtered_point_cloud_) {
                        point_cloud_to_render = cached_filtered_point_cloud_.get();
                    } else {
                        if (count_frame) {
                            framerate_controller_.endFrame();
                        }
                        return;
                    }
                    break;
                }

                LOG_TRACE("Rendering point cloud with {} points", point_cloud_to_render->size());

                // Get point cloud transform from scene state
                glm::mat4 point_cloud_transform(1.0f);
                if (!scene_state.model_transforms.empty()) {
                    point_cloud_transform = scene_state.model_transforms[0];
                }

                lfs::rendering::ViewportData viewport_data{
                    .rotation = context.viewport.getRotationMatrix(),
                    .translation = context.viewport.getTranslation(),
                    .size = render_size,
                    .focal_length_mm = settings_.focal_length_mm,
                    .orthographic = settings_.orthographic,
                    .ortho_scale = settings_.ortho_scale};

                // Build crop box from scene state for GPU-based desaturation
                std::optional<lfs::rendering::BoundingBox> crop_box;
                bool crop_inverse = false;
                bool crop_desaturate = false;
                for (const auto& cb : scene_state.cropboxes) {
                    if (!cb.data || (!cb.data->enabled && !settings_.show_crop_box))
                        continue;

                    crop_box = lfs::rendering::BoundingBox{
                        .min = cb.data->min,
                        .max = cb.data->max,
                        .transform = glm::inverse(cb.world_transform)};
                    crop_inverse = cb.data->inverse;
                    crop_desaturate = settings_.show_crop_box && settings_.desaturate_cropping;
                    break;
                }

                lfs::rendering::RenderRequest pc_request{
                    .viewport = viewport_data,
                    .scaling_modifier = settings_.scaling_modifier,
                    .antialiasing = false,
                    .mip_filter = settings_.mip_filter,
                    .sh_degree = 0,
                    .background_color = settings_.background_color,
                    .crop_box = crop_box,
                    .point_cloud_mode = true,
                    .voxel_size = settings_.voxel_size,
                    .gut = false,
                    .equirectangular = settings_.equirectangular,
                    .show_rings = false,
                    .ring_width = 0.0f,
                    .show_center_markers = false,
                    .model_transforms = {point_cloud_transform},
                    .transform_indices = nullptr,
                    .selection_mask = nullptr,
                    .output_screen_positions = false,
                    .brush_active = false,
                    .brush_x = 0.0f,
                    .brush_y = 0.0f,
                    .brush_radius = 0.0f,
                    .brush_add_mode = true,
                    .brush_selection_tensor = nullptr,
                    .brush_saturation_mode = false,
                    .brush_saturation_amount = 0.0f,
                    .selection_mode_rings = false,
                    .crop_inverse = crop_inverse,
                    .crop_desaturate = crop_desaturate,
                    .selected_node_mask = {},
                    .hovered_depth_id = nullptr,
                    .highlight_gaussian_id = -1,
                    .far_plane = lfs::rendering::DEFAULT_FAR_PLANE};

                auto render_result = engine_->renderPointCloud(*scene_state.point_cloud, pc_request);
                if (render_result) {
                    cached_result_ = *render_result;

                    glm::ivec2 actual_image_size(0, 0);
                    if (cached_result_.image) {
                        const auto& img = *cached_result_.image;
                        actual_image_size = glm::ivec2(img.size(2), img.size(1)); // [C, H, W] -> (W, H)
                    }

                    glm::ivec2 viewport_pos(0, 0);
                    if (context.viewport_region) {
                        const int gl_y = context.viewport.frameBufferSize.y - static_cast<int>(context.viewport_region->y) - static_cast<int>(context.viewport_region->height);
                        viewport_pos = glm::ivec2(static_cast<int>(context.viewport_region->x), gl_y);
                    }

                    glViewport(viewport_pos.x, viewport_pos.y, render_size.x, render_size.y);
                    glClearColor(settings_.background_color.r, settings_.background_color.g,
                                 settings_.background_color.b, 1.0f);
                    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

                    const auto present_result = engine_->presentToScreen(cached_result_, viewport_pos, actual_image_size);
                    if (!present_result) {
                        LOG_ERROR("Failed to present point cloud: {}", present_result.error());
                    }
                } else {
                    LOG_ERROR("Failed to render point cloud: {}", render_result.error());
                }
            }
        }

        // Always render overlays
        renderOverlays(context);
        if (count_frame) {
            framerate_controller_.endFrame();
        }
    }

    std::optional<lfs::rendering::SplitViewRequest>
    RenderingManager::createSplitViewRequest(const RenderContext& context, SceneManager* scene_manager) {
        if (settings_.split_view_mode == SplitViewMode::Disabled || !scene_manager) {
            return std::nullopt;
        }

        // Get render size
        glm::ivec2 render_size = context.viewport.windowSize;
        if (context.viewport_region) {
            render_size = glm::ivec2(
                static_cast<int>(context.viewport_region->width),
                static_cast<int>(context.viewport_region->height));
        }

        lfs::rendering::ViewportData viewport_data{
            .rotation = context.viewport.getRotationMatrix(),
            .translation = context.viewport.getTranslation(),
            .size = render_size,
            .focal_length_mm = settings_.focal_length_mm,
            .orthographic = settings_.orthographic,
            .ortho_scale = settings_.ortho_scale};

        // Crop box from scene graph (single source of truth)
        std::optional<lfs::rendering::BoundingBox> crop_box;
        if (settings_.use_crop_box || settings_.show_crop_box) {
            const auto& cropboxes = scene_manager->getScene().getVisibleCropBoxes();
            if (!cropboxes.empty() && cropboxes[0].data) {
                const auto& cb = cropboxes[0];
                crop_box = lfs::rendering::BoundingBox{
                    .min = cb.data->min,
                    .max = cb.data->max,
                    .transform = glm::inverse(cb.world_transform)};
            }
        }

        if (settings_.split_view_mode == SplitViewMode::GTComparison) {
            if (!gt_context_ || !gt_context_->valid() || !render_texture_valid_) {
                return std::nullopt;
            }

            auto letterbox_viewport = viewport_data;
            letterbox_viewport.size = render_size;

            const auto disabled_uids = scene_manager->getScene().getTrainingDisabledCameraUids();
            const bool cam_disabled = current_camera_id_ >= 0 && disabled_uids.count(current_camera_id_) > 0;
            std::string gt_label = cam_disabled ? "Ground Truth (Excluded from Training)" : "Ground Truth";

            return lfs::rendering::SplitViewRequest{
                .panels = {{.content_type = lfs::rendering::PanelContentType::Image2D,
                            .texture_id = gt_context_->gt_texture_id,
                            .label = std::move(gt_label),
                            .start_position = 0.0f,
                            .end_position = settings_.split_position},
                           {.content_type = lfs::rendering::PanelContentType::CachedRender,
                            .texture_id = cached_render_texture_,
                            .label = "Rendered",
                            .start_position = settings_.split_position,
                            .end_position = 1.0f}},
                .viewport = letterbox_viewport,
                .scaling_modifier = settings_.scaling_modifier,
                .antialiasing = settings_.antialiasing,
                .mip_filter = settings_.mip_filter,
                .sh_degree = settings_.sh_degree,
                .background_color = settings_.background_color,
                .crop_box = crop_box,
                .point_cloud_mode = settings_.point_cloud_mode,
                .voxel_size = settings_.voxel_size,
                .gut = settings_.gut,
                .equirectangular = settings_.equirectangular,
                .show_rings = settings_.show_rings,
                .ring_width = settings_.ring_width,
                .show_dividers = true,
                .divider_color = glm::vec4(1.0f, 0.85f, 0.0f, 1.0f),
                .show_labels = true,
                .left_texcoord_scale = gt_context_->gt_texcoord_scale,
                .right_texcoord_scale = gt_context_->render_texcoord_scale,
                .flip_left_y = gt_context_->gt_needs_flip,
                .letterbox = true,
                .content_size = gt_context_->dimensions};
        }

        if (settings_.split_view_mode == SplitViewMode::PLYComparison) {
            const auto visible_nodes = scene_manager->getScene().getVisibleNodes();
            if (visible_nodes.size() < 2) {
                LOG_TRACE("PLY comparison needs at least 2 visible nodes, have {}", visible_nodes.size());
                return std::nullopt;
            }

            // Calculate which pair to show
            size_t left_idx = settings_.split_view_offset % visible_nodes.size();
            size_t right_idx = (settings_.split_view_offset + 1) % visible_nodes.size();

            LOG_TRACE("Creating PLY comparison split view: {} vs {}",
                      visible_nodes[left_idx]->name, visible_nodes[right_idx]->name);

            // PLY comparison uses exact viewport-sized framebuffers, so scale is 1.0
            const glm::vec2 texcoord_scale(1.0f, 1.0f);

            return lfs::rendering::SplitViewRequest{
                .panels = {
                    {.content_type = lfs::rendering::PanelContentType::Model3D,
                     .model = visible_nodes[left_idx]->model.get(),
                     .texture_id = 0,
                     .label = visible_nodes[left_idx]->name,
                     .start_position = 0.0f,
                     .end_position = settings_.split_position},
                    {.content_type = lfs::rendering::PanelContentType::Model3D,
                     .model = visible_nodes[right_idx]->model.get(),
                     .texture_id = 0,
                     .label = visible_nodes[right_idx]->name,
                     .start_position = settings_.split_position,
                     .end_position = 1.0f}},
                .viewport = viewport_data,
                .scaling_modifier = settings_.scaling_modifier,
                .antialiasing = settings_.antialiasing,
                .mip_filter = settings_.mip_filter,
                .sh_degree = settings_.sh_degree,
                .background_color = settings_.background_color,
                .crop_box = crop_box,
                .point_cloud_mode = settings_.point_cloud_mode,
                .voxel_size = settings_.voxel_size,
                .gut = settings_.gut,
                .equirectangular = settings_.equirectangular,
                .show_rings = settings_.show_rings,
                .ring_width = settings_.ring_width,
                .show_dividers = true,
                .divider_color = glm::vec4(1.0f, 0.85f, 0.0f, 1.0f),
                .show_labels = true,
                .left_texcoord_scale = texcoord_scale,
                .right_texcoord_scale = texcoord_scale};
        }

        return std::nullopt;
    }

    void RenderingManager::renderOverlays(const RenderContext& context) {
        glm::ivec2 render_size = context.viewport.windowSize;
        if (context.viewport_region) {
            render_size = glm::ivec2(
                static_cast<int>(context.viewport_region->width),
                static_cast<int>(context.viewport_region->height));
        }

        if (render_size.x <= 0 || render_size.y <= 0) {
            return;
        }

        lfs::rendering::ViewportData viewport{
            .rotation = context.viewport.getRotationMatrix(),
            .translation = context.viewport.getTranslation(),
            .size = render_size,
            .focal_length_mm = settings_.focal_length_mm,
            .orthographic = settings_.orthographic,
            .ortho_scale = settings_.ortho_scale};

        // Render wireframe overlays before grid
        if (settings_.show_crop_box && engine_ && context.scene_manager) {
            const auto visible_cropboxes = context.scene_manager->getScene().getVisibleCropBoxes();
            const NodeId selected_cropbox_id = context.scene_manager->getSelectedNodeCropBoxId();

            for (const auto& cb : visible_cropboxes) {
                if (!cb.data)
                    continue;

                const bool is_selected = (cb.node_id == selected_cropbox_id);

                // Use pending state for selected cropbox during gizmo manipulation
                const bool use_pending = is_selected && cropbox_gizmo_active_;
                const glm::vec3 box_min = use_pending ? pending_cropbox_min_ : cb.data->min;
                const glm::vec3 box_max = use_pending ? pending_cropbox_max_ : cb.data->max;
                const glm::mat4 box_transform = use_pending ? pending_cropbox_transform_ : cb.world_transform;

                const lfs::rendering::BoundingBox box{
                    .min = box_min,
                    .max = box_max,
                    .transform = glm::inverse(box_transform)};

                const glm::vec3 base_color = cb.data->inverse
                                                 ? glm::vec3(1.0f, 0.2f, 0.2f)
                                                 : cb.data->color;
                const float flash = is_selected ? cb.data->flash_intensity : 0.0f;
                constexpr float FLASH_LINE_BOOST = 4.0f;
                const glm::vec3 color = glm::mix(base_color, glm::vec3(1.0f), flash);
                const float line_width = cb.data->line_width + flash * FLASH_LINE_BOOST;

                auto bbox_result = engine_->renderBoundingBox(box, viewport, color, line_width);
                if (!bbox_result) {
                    LOG_WARN("Failed to render bounding box: {}", bbox_result.error());
                }
            }
        }

        // Render ellipsoid wireframe overlays
        if (settings_.show_ellipsoid && engine_ && context.scene_manager) {
            const auto visible_ellipsoids = context.scene_manager->getScene().getVisibleEllipsoids();
            const NodeId selected_ellipsoid_id = context.scene_manager->getSelectedNodeEllipsoidId();

            for (const auto& el : visible_ellipsoids) {
                if (!el.data)
                    continue;

                const bool is_selected = (el.node_id == selected_ellipsoid_id);

                // Use pending state for selected ellipsoid during gizmo manipulation
                const glm::vec3 radii = (is_selected && ellipsoid_gizmo_active_)
                                            ? pending_ellipsoid_radii_
                                            : el.data->radii;
                const glm::mat4 transform = (is_selected && ellipsoid_gizmo_active_)
                                                ? pending_ellipsoid_transform_
                                                : el.world_transform;

                const lfs::rendering::Ellipsoid ellipsoid{
                    .radii = radii,
                    .transform = transform};

                const glm::vec3 base_color = el.data->inverse
                                                 ? glm::vec3(1.0f, 0.2f, 0.2f)
                                                 : el.data->color;
                const float flash = is_selected ? el.data->flash_intensity : 0.0f;
                constexpr float FLASH_LINE_BOOST = 4.0f;
                const glm::vec3 color = glm::mix(base_color, glm::vec3(1.0f), flash);
                const float line_width = el.data->line_width + flash * FLASH_LINE_BOOST;

                auto ellipsoid_result = engine_->renderEllipsoid(ellipsoid, viewport, color, line_width);
                if (!ellipsoid_result) {
                    LOG_WARN("Failed to render ellipsoid: {}", ellipsoid_result.error());
                }
            }
        }

        // Coordinate axes
        if (settings_.show_coord_axes && engine_) {
            auto axes_result = engine_->renderCoordinateAxes(viewport, settings_.axes_size, settings_.axes_visibility, settings_.equirectangular);
            if (!axes_result) {
                LOG_WARN("Failed to render coordinate axes: {}", axes_result.error());
            }
        }

        // Pivot point ripple animation
        if (engine_) {
            constexpr float PIVOT_DURATION_SEC = 0.5f;
            constexpr float PIVOT_SIZE_PX = 50.0f;

            const float time_since_set = context.viewport.camera.getSecondsSincePivotSet();
            const bool animation_active = time_since_set < PIVOT_DURATION_SEC;

            if (animation_active) {
                const auto remaining_ms = static_cast<int>((PIVOT_DURATION_SEC - time_since_set) * 1000.0f);
                setPivotAnimationEndTime(std::chrono::steady_clock::now() +
                                         std::chrono::milliseconds(remaining_ms));
            }

            if (settings_.show_pivot || animation_active) {
                const float opacity = settings_.show_pivot ? 1.0f : 1.0f - std::clamp(time_since_set / PIVOT_DURATION_SEC, 0.0f, 1.0f);

                if (auto result = engine_->renderPivot(viewport, context.viewport.camera.getPivot(),
                                                       PIVOT_SIZE_PX, opacity);
                    !result) {
                    LOG_WARN("Pivot render failed: {}", result.error());
                }
            }
        }

        // Camera frustums - read directly from scene nodes (no trainer dependency)
        if (settings_.show_camera_frustums && engine_ && context.scene_manager) {
            auto cameras = context.scene_manager->getScene().getVisibleCameras();

            if (!cameras.empty()) {
                int highlight_index = -1;
                if (hovered_camera_id_ >= 0) {
                    for (size_t i = 0; i < cameras.size(); ++i) {
                        if (cameras[i]->uid() == hovered_camera_id_) {
                            highlight_index = static_cast<int>(i);
                            break;
                        }
                    }
                }

                glm::mat4 scene_transform(1.0f);
                auto visible_transforms = context.scene_manager->getScene().getVisibleNodeTransforms();
                if (!visible_transforms.empty()) {
                    scene_transform = visible_transforms[0];
                }

                LOG_TRACE("Rendering {} camera frustums with scale {}, highlighted index: {} (ID: {})",
                          cameras.size(), settings_.camera_frustum_scale, highlight_index, hovered_camera_id_);

                auto disabled_uids = context.scene_manager->getScene().getTrainingDisabledCameraUids();

                std::unordered_set<int> selected_uids;
                for (const auto& name : context.scene_manager->getSelectedNodeNames()) {
                    const auto* node = context.scene_manager->getScene().getNode(name);
                    if (node && node->type == NodeType::CAMERA && node->camera_uid >= 0)
                        selected_uids.insert(node->camera_uid);
                }

                auto frustum_result = engine_->renderCameraFrustumsWithHighlight(
                    cameras, viewport,
                    settings_.camera_frustum_scale,
                    settings_.train_camera_color,
                    settings_.eval_camera_color,
                    highlight_index,
                    scene_transform,
                    settings_.equirectangular,
                    disabled_uids,
                    selected_uids);

                if (!frustum_result) {
                    LOG_ERROR("Failed to render camera frustums: {}", frustum_result.error());
                }

                if (pick_requested_ && context.viewport_region) {
                    pick_requested_ = false;

                    auto pick_result = engine_->pickCameraFrustum(
                        cameras,
                        pending_pick_pos_,
                        glm::vec2(context.viewport_region->x, context.viewport_region->y),
                        glm::vec2(context.viewport_region->width, context.viewport_region->height),
                        viewport,
                        settings_.camera_frustum_scale,
                        scene_transform);

                    if (pick_result) {
                        int cam_id = *pick_result;
                        if (cam_id != hovered_camera_id_) {
                            int old_hover = hovered_camera_id_;
                            hovered_camera_id_ = cam_id;
                            markDirty();
                            LOG_DEBUG("Camera hover changed: {} -> {}", old_hover, cam_id);
                        }
                    } else if (hovered_camera_id_ != -1) {
                        int old_hover = hovered_camera_id_;
                        hovered_camera_id_ = -1;
                        markDirty();
                        LOG_DEBUG("Camera hover lost (was ID: {})", old_hover);
                    }
                }
            }
        }

        // Grid - disabled in split view and equirectangular modes
        if (settings_.show_grid && engine_ && settings_.split_view_mode == SplitViewMode::Disabled && !settings_.equirectangular) {
            if (const auto result = engine_->renderGrid(
                    viewport,
                    static_cast<lfs::rendering::GridPlane>(settings_.grid_plane),
                    settings_.grid_opacity);
                !result) {
                LOG_WARN("Grid render failed: {}", result.error());
            }
        }
    }

    float RenderingManager::getDepthAtPixel(int x, int y) const {
        if (!cached_result_.valid) {
            return -1.0f;
        }

        const lfs::core::Tensor* depth_ptr = nullptr;
        const int viewport_width = cached_result_size_.x;
        const int viewport_height = cached_result_size_.y;

        if (viewport_width <= 0 || viewport_height <= 0) {
            return -1.0f;
        }

        if (cached_result_.split_position > 0.0f && cached_result_.depth && cached_result_.depth->is_valid()) {
            const float normalized_x = static_cast<float>(x) / static_cast<float>(viewport_width);

            if (normalized_x >= cached_result_.split_position &&
                cached_result_.depth_right && cached_result_.depth_right->is_valid()) {
                depth_ptr = cached_result_.depth_right.get();
            } else {
                depth_ptr = cached_result_.depth.get();
            }
        } else if (cached_result_.depth && cached_result_.depth->is_valid()) {
            depth_ptr = cached_result_.depth.get();
        }

        if (!depth_ptr) {
            return -1.0f;
        }

        const auto& depth = *depth_ptr;
        if (depth.ndim() != 3) {
            return -1.0f;
        }

        const int depth_height = static_cast<int>(depth.size(1));
        const int depth_width = static_cast<int>(depth.size(2));

        int scaled_x = x;
        int scaled_y = y;
        if (viewport_width > 0 && viewport_height > 0 &&
            (depth_width != viewport_width || depth_height != viewport_height)) {
            scaled_x = static_cast<int>(static_cast<float>(x) * depth_width / viewport_width);
            scaled_y = static_cast<int>(static_cast<float>(y) * depth_height / viewport_height);
        }

        if (scaled_x < 0 || scaled_x >= depth_width || scaled_y < 0 || scaled_y >= depth_height) {
            return -1.0f;
        }

        auto depth_cpu = depth.cpu();
        const float* data = depth_cpu.ptr<float>();
        const float d = data[scaled_y * depth_width + scaled_x];

        if (d > 1e9f) {
            return -1.0f;
        }

        return d;
    }

    void RenderingManager::brushSelect(float mouse_x, float mouse_y, float radius, lfs::core::Tensor& selection_out) {
        if (!cached_result_.screen_positions || !cached_result_.screen_positions->is_valid()) {
            return;
        }
        lfs::rendering::brush_select_tensor(*cached_result_.screen_positions, mouse_x, mouse_y, radius, selection_out);
    }

    void RenderingManager::applyCropFilter(lfs::core::Tensor& selection) {
        if (!selection.is_valid())
            return;

        auto* const sm = services().sceneOrNull();
        if (!sm)
            return;

        const auto* const model = sm->getModelForRendering();
        if (!model || model->size() == 0)
            return;

        const auto& means = model->means();
        if (!means.is_valid() || means.size(0) != selection.size(0))
            return;

        lfs::core::Tensor crop_t, crop_min, crop_max;
        bool crop_inverse = false;

        const auto& cropboxes = sm->buildRenderState().cropboxes;
        if (!cropboxes.empty() && cropboxes[0].data) {
            const auto& cb = cropboxes[0];
            const glm::mat4 inv_transform = glm::inverse(cb.world_transform);
            const float* const t_ptr = glm::value_ptr(inv_transform);
            crop_t = lfs::core::Tensor::from_vector(std::vector<float>(t_ptr, t_ptr + 16), {4, 4}, lfs::core::Device::CPU).cuda();
            crop_min = lfs::core::Tensor::from_vector(
                           {cb.data->min.x, cb.data->min.y, cb.data->min.z}, {3}, lfs::core::Device::CPU)
                           .cuda();
            crop_max = lfs::core::Tensor::from_vector(
                           {cb.data->max.x, cb.data->max.y, cb.data->max.z}, {3}, lfs::core::Device::CPU)
                           .cuda();
            crop_inverse = cb.data->inverse;
        }

        lfs::core::Tensor ellip_t, ellip_radii;
        bool ellipsoid_inverse = false;

        const auto& ellipsoids = sm->getScene().getVisibleEllipsoids();
        if (!ellipsoids.empty() && ellipsoids[0].data) {
            const auto& el = ellipsoids[0];
            const glm::mat4 inv_transform = glm::inverse(el.world_transform);
            const float* const t_ptr = glm::value_ptr(inv_transform);
            ellip_t = lfs::core::Tensor::from_vector(std::vector<float>(t_ptr, t_ptr + 16), {4, 4}, lfs::core::Device::CPU).cuda();
            ellip_radii = lfs::core::Tensor::from_vector(
                              {el.data->radii.x, el.data->radii.y, el.data->radii.z}, {3}, lfs::core::Device::CPU)
                              .cuda();
            ellipsoid_inverse = el.data->inverse;
        }

        lfs::rendering::filter_selection_by_crop(
            selection, means,
            crop_t.is_valid() ? &crop_t : nullptr,
            crop_min.is_valid() ? &crop_min : nullptr,
            crop_max.is_valid() ? &crop_max : nullptr,
            crop_inverse,
            ellip_t.is_valid() ? &ellip_t : nullptr,
            ellip_radii.is_valid() ? &ellip_radii : nullptr,
            ellipsoid_inverse);
    }

    void RenderingManager::setBrushState(const bool active, const float x, const float y, const float radius,
                                         const bool add_mode, lfs::core::Tensor* selection_tensor,
                                         const bool saturation_mode, const float saturation_amount) {
        brush_active_ = active;
        brush_x_ = x;
        brush_y_ = y;
        brush_radius_ = radius;
        brush_add_mode_ = add_mode;
        brush_selection_tensor_ = selection_tensor;
        brush_saturation_mode_ = saturation_mode;
        brush_saturation_amount_ = saturation_amount;
        markDirty();
    }

    void RenderingManager::clearBrushState() {
        brush_active_ = false;
        brush_x_ = 0.0f;
        brush_y_ = 0.0f;
        brush_radius_ = 0.0f;
        brush_selection_tensor_ = nullptr;
        brush_saturation_mode_ = false;
        brush_saturation_amount_ = 0.0f;
        hovered_gaussian_id_ = -1;
        preview_selection_ = nullptr;
        markDirty();
    }

    void RenderingManager::setRectPreview(float x0, float y0, float x1, float y1, bool add_mode) {
        rect_preview_active_ = true;
        rect_x0_ = x0;
        rect_y0_ = y0;
        rect_x1_ = x1;
        rect_y1_ = y1;
        rect_add_mode_ = add_mode;
    }

    void RenderingManager::clearRectPreview() {
        rect_preview_active_ = false;
    }

    void RenderingManager::setPolygonPreview(const std::vector<std::pair<float, float>>& points, bool closed, bool add_mode) {
        polygon_preview_active_ = true;
        polygon_points_ = points;
        polygon_closed_ = closed;
        polygon_add_mode_ = add_mode;
    }

    void RenderingManager::clearPolygonPreview() {
        polygon_preview_active_ = false;
        polygon_points_.clear();
        polygon_closed_ = false;
    }

    void RenderingManager::setLassoPreview(const std::vector<std::pair<float, float>>& points, bool add_mode) {
        lasso_preview_active_ = true;
        lasso_points_ = points;
        lasso_add_mode_ = add_mode;
    }

    void RenderingManager::clearLassoPreview() {
        lasso_preview_active_ = false;
        lasso_points_.clear();
    }

    void RenderingManager::adjustSaturation(const float mouse_x, const float mouse_y, const float radius,
                                            const float saturation_delta, lfs::core::Tensor& sh0_tensor) {
        const auto& screen_pos = cached_result_.screen_positions;
        if (!screen_pos || !screen_pos->is_valid())
            return;
        if (!sh0_tensor.is_valid() || sh0_tensor.device() != lfs::core::Device::CUDA)
            return;

        const int num_gaussians = static_cast<int>(screen_pos->size(0));
        if (num_gaussians == 0)
            return;

        lfs::launchAdjustSaturation(
            sh0_tensor.ptr<float>(),
            screen_pos->ptr<float>(),
            mouse_x, mouse_y, radius,
            saturation_delta,
            num_gaussians,
            nullptr);

        markDirty();
    }

} // namespace lfs::vis