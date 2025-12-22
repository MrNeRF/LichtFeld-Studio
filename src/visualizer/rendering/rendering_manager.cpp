/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rendering_manager.hpp"
#include "core/camera.hpp"
#include "core/image_io.hpp" // Use existing image_io utilities
#include "core/logger.hpp"
#include "core/splat_data.hpp"
#include "geometry/euclidean_transform.hpp"
#include "rendering/cuda_kernels.hpp"
#include "rendering/rasterizer/rasterization/include/rasterization_api_tensor.h"
#include "rendering/rasterizer/rasterization/include/rasterization_config.h"
#include "rendering/rendering.hpp"
#include "scene/scene_manager.hpp"
#include "training/trainer.hpp"
#include "training/training_manager.hpp"
#include <cuda_runtime.h>
#include <glad/glad.h>
#include <shared_mutex>
#include <stdexcept>

namespace lfs::vis {

    using namespace lfs::core::events;

    // GTTextureCache Implementation
    GTTextureCache::GTTextureCache() {
        LOG_DEBUG("GTTextureCache created");
    }

    GTTextureCache::~GTTextureCache() {
        clear();
    }

    void GTTextureCache::clear() {
        for (auto& [id, entry] : texture_cache_) {
            if (entry.texture_id > 0) {
                glDeleteTextures(1, &entry.texture_id);
            }
        }
        texture_cache_.clear();
        LOG_DEBUG("GTTextureCache cleared");
    }

    unsigned int GTTextureCache::getGTTexture(const int cam_id, const std::filesystem::path& image_path) {
        if (const auto it = texture_cache_.find(cam_id); it != texture_cache_.end()) {
            it->second.last_access = std::chrono::steady_clock::now();
            return it->second.texture_id;
        }

        const unsigned int texture_id = loadTexture(image_path);
        if (texture_id == 0) {
            return 0;
        }

        if (texture_cache_.size() >= MAX_CACHE_SIZE) {
            evictOldest();
        }

        texture_cache_[cam_id] = {texture_id, std::chrono::steady_clock::now()};
        return texture_id;
    }

    void GTTextureCache::evictOldest() {
        if (texture_cache_.empty())
            return;

        auto oldest = texture_cache_.begin();
        auto oldest_time = oldest->second.last_access;

        for (auto it = texture_cache_.begin(); it != texture_cache_.end(); ++it) {
            if (it->second.last_access < oldest_time) {
                oldest = it;
                oldest_time = it->second.last_access;
            }
        }

        LOG_TRACE("Evicting GT texture for camera {} from cache", oldest->first);
        glDeleteTextures(1, &oldest->second.texture_id);
        texture_cache_.erase(oldest);
    }

    unsigned int GTTextureCache::loadTexture(const std::filesystem::path& path) {
        if (!std::filesystem::exists(path)) {
            LOG_ERROR("GT image file does not exist: {}", path.string());
            return 0;
        }

        try {
            // Use image_io to load the image
            auto [data, width, height, channels] = lfs::core::load_image(path);

            if (!data) {
                LOG_ERROR("Failed to load image data: {}", path.string());
                return 0;
            }

            LOG_TRACE("Loaded GT image: {}x{} with {} channels", width, height, channels);

            // FLIP vertically: OpenGL expects origin at bottom-left, images have origin at top-left
            // This matches what the renderer produces
            std::vector<unsigned char> flipped_data(width * height * channels);
            size_t row_size = width * channels;
            for (int y = 0; y < height; ++y) {
                std::memcpy(
                    flipped_data.data() + y * row_size,
                    data + (height - 1 - y) * row_size,
                    row_size);
            }

            // Create OpenGL texture
            unsigned int texture;
            glGenTextures(1, &texture);
            glBindTexture(GL_TEXTURE_2D, texture);

            // Determine format based on channels
            GLenum format = GL_RGB;
            GLenum internal_format = GL_RGB8;

            if (channels == 1) {
                format = GL_RED;
                internal_format = GL_R8;
            } else if (channels == 2) {
                format = GL_RG;
                internal_format = GL_RG8;
            } else if (channels == 3) {
                format = GL_RGB;
                internal_format = GL_RGB8;
            } else if (channels == 4) {
                format = GL_RGBA;
                internal_format = GL_RGBA8;
            }

            // Upload flipped texture data
            glTexImage2D(GL_TEXTURE_2D, 0, internal_format, width, height, 0,
                         format, GL_UNSIGNED_BYTE, flipped_data.data());

            // Set texture parameters
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

            // Generate mipmaps for better quality when scaled
            glGenerateMipmap(GL_TEXTURE_2D);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);

            // Free image data
            lfs::core::free_image(data);

            LOG_DEBUG("Created GL texture {} for image: {} ({}x{})",
                      texture, path.filename().string(), width, height);
            return texture;

        } catch (const std::exception& e) {
            LOG_ERROR("Exception loading image {}: {}", path.string(), e.what());
            return 0;
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

        // Listen for GT comparison toggle (G key - for camera/GT comparison)
        cmd::ToggleGTComparison::when([this](const auto&) {
            std::lock_guard<std::mutex> lock(settings_mutex_);

            // G key toggles between Disabled and GTComparison only
            if (settings_.split_view_mode == SplitViewMode::GTComparison) {
                settings_.split_view_mode = SplitViewMode::Disabled;
                LOG_INFO("GT comparison disabled");
            } else {
                // Check if we can actually do GT comparison
                if (current_camera_id_ < 0) {
                    LOG_WARN("Cannot enable GT comparison: no camera selected. Use arrow keys or click a camera to select one.");
                    // Don't change the mode
                    return;
                }

                // From Disabled or PLYComparison, go to GTComparison
                settings_.split_view_mode = SplitViewMode::GTComparison;
                LOG_INFO("GT comparison enabled for camera {}", current_camera_id_);
            }

            markDirty();

            // Emit UI event
            ui::GTComparisonModeChanged{
                .enabled = (settings_.split_view_mode == SplitViewMode::GTComparison)}
                .emit();
        });

        // Listen for camera view changes
        cmd::GoToCamView::when([this](const auto& event) {
            setCurrentCameraId(event.cam_id);
            LOG_DEBUG("Current camera ID set to: {}", event.cam_id);

            // If GT comparison was waiting for a camera, re-enable rendering
            if (settings_.split_view_mode == SplitViewMode::GTComparison && event.cam_id >= 0) {
                LOG_INFO("Camera {} selected, GT comparison now active", event.cam_id);
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
            if (event.fov) {
                settings_.fov = *event.fov;
                LOG_TRACE("FOV changed to: {}", settings_.fov);
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
            last_render_size_ = glm::ivec2(0, 0);
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
            LOG_DEBUG("PLY removed, marking render dirty");
            markDirty();
        });

        // Crop box changes (scene graph is source of truth, this just handles enable flag)
        ui::CropBoxChanged::when([this](const auto& event) {
            std::lock_guard<std::mutex> lock(settings_mutex_);
            settings_.use_crop_box = event.enabled;
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
                const float half_tan_fov = std::tan(glm::radians(settings_.fov) * 0.5f);
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
        return settings_.fov;
    }

    float RenderingManager::getScalingModifier() const {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        return settings_.scaling_modifier;
    }

    void RenderingManager::setFov(float f) {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        settings_.fov = f;
        markDirty();
    }

    void RenderingManager::setScalingModifier(float s) {
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

        glm::ivec2 render_size = context.viewport.windowSize;
        if (context.viewport_region) {
            render_size = glm::ivec2(
                static_cast<int>(context.viewport_region->width),
                static_cast<int>(context.viewport_region->height));
        }

        // For GT comparison mode, get the actual camera dimensions
        if (settings_.split_view_mode == SplitViewMode::GTComparison && current_camera_id_ >= 0) {
            auto* trainer_manager = scene_manager->getTrainerManager();
            if (trainer_manager && trainer_manager->hasTrainer()) {
                auto cam = trainer_manager->getCamById(current_camera_id_);
                if (cam) {
                    // Use the GT camera's image dimensions for rendering
                    render_size.x = cam->image_width();
                    render_size.y = cam->image_height();
                    LOG_TRACE("Using GT camera dimensions for rendering: {}x{}", render_size.x, render_size.y);
                }
            }
        }

        // Resize texture if needed
        static glm::ivec2 texture_size{0, 0};
        if (render_size != texture_size) {
            glBindTexture(GL_TEXTURE_2D, cached_render_texture_);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, render_size.x, render_size.y,
                         0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
            texture_size = render_size;
            LOG_DEBUG("Resized cached render texture to {}x{}", render_size.x, render_size.y);
        }

        // Create framebuffer for offscreen rendering
        static GLuint render_fbo = 0;
        static GLuint render_depth_rbo = 0;

        if (render_fbo == 0) {
            glGenFramebuffers(1, &render_fbo);
            glGenRenderbuffers(1, &render_depth_rbo);
        }

        GLint current_fbo;
        glGetIntegerv(GL_FRAMEBUFFER_BINDING, &current_fbo);

        glBindFramebuffer(GL_FRAMEBUFFER, render_fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, cached_render_texture_, 0);

        // Update depth buffer size if needed
        glBindRenderbuffer(GL_RENDERBUFFER, render_depth_rbo);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, render_size.x, render_size.y);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, render_depth_rbo);

        // Check framebuffer completeness
        GLenum fb_status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
        if (fb_status != GL_FRAMEBUFFER_COMPLETE) {
            LOG_ERROR("Framebuffer incomplete: 0x{:x}", fb_status);
            glBindFramebuffer(GL_FRAMEBUFFER, current_fbo);
            render_texture_valid_ = false;
            return;
        }

        // Render model to texture
        glViewport(0, 0, render_size.x, render_size.y);
        glClearColor(settings_.background_color.r, settings_.background_color.g, settings_.background_color.b, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Create viewport data
        lfs::rendering::ViewportData viewport_data{
            .rotation = context.viewport.getRotationMatrix(),
            .translation = context.viewport.getTranslation(),
            .size = render_size,
            .fov = settings_.fov,
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
            .sh_degree = settings_.sh_degree,
            .background_color = settings_.background_color,
            .crop_box = std::nullopt,
            .point_cloud_mode = settings_.point_cloud_mode,
            .voxel_size = settings_.voxel_size,
            .gut = settings_.gut,
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
                request.crop_desaturate = settings_.show_crop_box && !settings_.use_crop_box;
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

        const auto render_result = engine_->renderGaussians(*model, request);
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
        }

        // Restore framebuffer
        glBindFramebuffer(GL_FRAMEBUFFER, current_fbo);
    }

    void RenderingManager::renderFrame(const RenderContext& context, SceneManager* scene_manager) {
        framerate_controller_.beginFrame();

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
            framerate_controller_.endFrame();
            return;
        }

        if (current_size != last_render_size_) {
            LOG_DEBUG("Viewport resize: {}x{} -> {}x{}", last_render_size_.x, last_render_size_.y,
                      current_size.x, current_size.y);
            needs_render_ = true;
            last_render_size_ = current_size;
        }

        const lfs::core::SplatData* const model = scene_manager ? scene_manager->getModelForRendering() : nullptr;
        const size_t model_ptr = reinterpret_cast<size_t>(model);

        // Lock model during render to prevent tensor replacement
        std::shared_lock<std::shared_mutex> render_lock;
        if (const auto* tm = scene_manager ? scene_manager->getTrainerManager() : nullptr) {
            if (const auto* trainer = tm->getTrainer()) {
                render_lock = std::shared_lock{trainer->getRenderMutex()};
            }
        }

        // Detect model switch
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

        // GT comparison requires valid render texture
        if (settings_.split_view_mode == SplitViewMode::GTComparison) {
            if (current_camera_id_ < 0) {
                split_view_active = false;
            } else if (!render_texture_valid_ && model) {
                renderToTexture(context, scene_manager, model);
            }
        }

        if (!cached_result_.image || needs_render_now || split_view_active) {
            should_render = true;
            needs_render_ = false;
        }

        glViewport(0, 0, context.viewport.frameBufferSize.x, context.viewport.frameBufferSize.y);

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
        } else if (cached_result_.image) {
            glm::ivec2 viewport_pos(0, 0);
            const glm::ivec2 render_size = current_size;

            if (context.viewport_region) {
                const int gl_y = context.viewport.frameBufferSize.y - static_cast<int>(context.viewport_region->y) - static_cast<int>(context.viewport_region->height);
                viewport_pos = glm::ivec2(static_cast<int>(context.viewport_region->x), gl_y);
            }

            glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            engine_->presentToScreen(cached_result_, viewport_pos, render_size);
            renderOverlays(context);
        }

        if (context.viewport_region) {
            glDisable(GL_SCISSOR_TEST);
        }
        framerate_controller_.endFrame();
    }

    void RenderingManager::doFullRender(const RenderContext& context, SceneManager* scene_manager, const lfs::core::SplatData* model) {
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

            auto result = engine_->renderSplitView(*split_request);
            if (result) {
                // Split view already composites to screen, no need to present again
                // Just store a dummy result to prevent re-rendering every frame
                cached_result_ = *result;
            } else {
                LOG_ERROR("Failed to render split view: {}", result.error());
            }

            renderOverlays(context);
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

            if (render_texture_valid_) {
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
                    render_size);
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
                                             cached_cropbox_transform_ == cb.local_transform &&
                                             cached_cropbox_min_ == cb.data->min &&
                                             cached_cropbox_max_ == cb.data->max &&
                                             cached_cropbox_inverse_ == cb.data->inverse;

                    if (!cache_valid) {
                        const auto& means = scene_state.point_cloud->means;
                        const auto& colors = scene_state.point_cloud->colors;
                        const size_t num_points = scene_state.point_cloud->size();
                        const glm::mat4 m = glm::inverse(cb.local_transform);
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
                        cached_cropbox_transform_ = cb.local_transform;
                        cached_cropbox_min_ = cb.data->min;
                        cached_cropbox_max_ = cb.data->max;
                        cached_cropbox_inverse_ = cb.data->inverse;
                    }

                    if (cached_filtered_point_cloud_) {
                        point_cloud_to_render = cached_filtered_point_cloud_.get();
                    } else {
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

                // Create viewport data
                lfs::rendering::ViewportData viewport_data{
                    .rotation = context.viewport.getRotationMatrix(),
                    .translation = context.viewport.getTranslation(),
                    .size = render_size,
                    .fov = settings_.fov,
                    .orthographic = settings_.orthographic,
                    .ortho_scale = settings_.ortho_scale};

                lfs::rendering::RenderRequest pc_request{
                    .viewport = viewport_data,
                    .scaling_modifier = settings_.scaling_modifier,
                    .antialiasing = false,
                    .sh_degree = 0,
                    .background_color = settings_.background_color,
                    .crop_box = std::nullopt,
                    .point_cloud_mode = true,
                    .voxel_size = settings_.voxel_size,
                    .gut = false,
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
                    .selected_node_mask = {},
                    .hovered_depth_id = nullptr,
                    .highlight_gaussian_id = -1,
                    .far_plane = lfs::rendering::DEFAULT_FAR_PLANE};

                auto render_result = engine_->renderPointCloud(*point_cloud_to_render, pc_request);
                if (render_result) {
                    cached_result_ = *render_result;

                    glm::ivec2 viewport_pos(0, 0);
                    if (context.viewport_region) {
                        const int gl_y = context.viewport.frameBufferSize.y - static_cast<int>(context.viewport_region->y) - static_cast<int>(context.viewport_region->height);
                        viewport_pos = glm::ivec2(static_cast<int>(context.viewport_region->x), gl_y);
                    }

                    glViewport(viewport_pos.x, viewport_pos.y, render_size.x, render_size.y);
                    glClearColor(settings_.background_color.r, settings_.background_color.g,
                                 settings_.background_color.b, 1.0f);
                    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

                    const auto present_result = engine_->presentToScreen(cached_result_, viewport_pos, render_size);
                    if (!present_result) {
                        LOG_ERROR("Failed to present point cloud render result: {}", present_result.error());
                    }
                } else {
                    LOG_ERROR("Failed to render point cloud: {}", render_result.error());
                }
            }
        }

        // Always render overlays
        renderOverlays(context);
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

        // Create viewport data
        lfs::rendering::ViewportData viewport_data{
            .rotation = context.viewport.getRotationMatrix(),
            .translation = context.viewport.getTranslation(),
            .size = render_size,
            .fov = settings_.fov,
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

        // Handle GT comparison mode
        if (settings_.split_view_mode == SplitViewMode::GTComparison) {
            if (current_camera_id_ < 0) {
                // Log this only once per second to avoid spam
                static auto last_log_time = std::chrono::steady_clock::now();
                auto now = std::chrono::steady_clock::now();
                if (now - last_log_time > std::chrono::seconds(1)) {
                    LOG_INFO("GT comparison enabled but no camera selected. Use arrow keys or click a camera to select one.");
                    last_log_time = now;
                }
                return std::nullopt;
            }

            // Get camera from trainer manager
            auto* trainer_manager = scene_manager->getTrainerManager();
            if (!trainer_manager || !trainer_manager->hasTrainer()) {
                LOG_WARN("GT comparison mode but no trainer available");
                return std::nullopt;
            }

            auto cam = trainer_manager->getCamById(current_camera_id_);
            if (!cam) {
                LOG_WARN("Camera {} not found", current_camera_id_);
                current_camera_id_ = -1; // Reset invalid camera ID
                return std::nullopt;
            }

            // Get GT texture
            unsigned int gt_texture = gt_texture_cache_.getGTTexture(current_camera_id_, cam->image_path());
            if (gt_texture == 0) {
                LOG_ERROR("Failed to get GT texture for camera {}", current_camera_id_);
                return std::nullopt;
            }

            // Make sure we have a valid render texture
            if (!render_texture_valid_) {
                // Force a render to texture
                const lfs::core::SplatData* model = scene_manager->getModelForRendering();
                if (model) {
                    renderToTexture(context, scene_manager, model);
                }
            }

            if (!render_texture_valid_) {
                LOG_ERROR("Failed to get cached render for GT comparison");
                return std::nullopt;
            }

            LOG_TRACE("Creating GT comparison split view for camera {}", current_camera_id_);

            return lfs::rendering::SplitViewRequest{
                .panels = {
                    {.content_type = lfs::rendering::PanelContentType::Image2D,
                     .model = nullptr,
                     .texture_id = gt_texture,
                     .label = "Ground Truth",
                     .start_position = 0.0f,
                     .end_position = settings_.split_position},
                    {.content_type = lfs::rendering::PanelContentType::CachedRender,
                     .model = nullptr,
                     .texture_id = cached_render_texture_,
                     .label = "Rendered",
                     .start_position = settings_.split_position,
                     .end_position = 1.0f}},
                .viewport = viewport_data,
                .scaling_modifier = settings_.scaling_modifier,
                .antialiasing = settings_.antialiasing,
                .sh_degree = settings_.sh_degree,
                .background_color = settings_.background_color,
                .crop_box = crop_box,
                .point_cloud_mode = settings_.point_cloud_mode,
                .voxel_size = settings_.voxel_size,
                .gut = settings_.gut,
                .show_rings = settings_.show_rings,
                .ring_width = settings_.ring_width,
                .show_dividers = true,
                .divider_color = glm::vec4(1.0f, 0.85f, 0.0f, 1.0f),
                .show_labels = true};
        }

        // Handle PLY comparison mode
        if (settings_.split_view_mode == SplitViewMode::PLYComparison) {
            auto visible_nodes = scene_manager->getScene().getVisibleNodes();
            if (visible_nodes.size() < 2) {
                LOG_TRACE("PLY comparison needs at least 2 visible nodes, have {}", visible_nodes.size());
                return std::nullopt;
            }

            // Calculate which pair to show
            size_t left_idx = settings_.split_view_offset % visible_nodes.size();
            size_t right_idx = (settings_.split_view_offset + 1) % visible_nodes.size();

            LOG_TRACE("Creating PLY comparison split view: {} vs {}",
                      visible_nodes[left_idx]->name, visible_nodes[right_idx]->name);

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
                .sh_degree = settings_.sh_degree,
                .background_color = settings_.background_color,
                .crop_box = crop_box,
                .point_cloud_mode = settings_.point_cloud_mode,
                .voxel_size = settings_.voxel_size,
                .gut = settings_.gut,
                .show_rings = settings_.show_rings,
                .ring_width = settings_.ring_width,
                .show_dividers = true,
                .divider_color = glm::vec4(1.0f, 0.85f, 0.0f, 1.0f),
                .show_labels = true};
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
            .fov = settings_.fov,
            .orthographic = settings_.orthographic,
            .ortho_scale = settings_.ortho_scale};

        // Render wireframe overlays before grid
        if (settings_.show_crop_box && engine_ && context.scene_manager) {
            const auto visible_cropboxes = context.scene_manager->getScene().getVisibleCropBoxes();
            const NodeId selected_cropbox_id = context.scene_manager->getSelectedNodeCropBoxId();

            for (const auto& cb : visible_cropboxes) {
                if (!cb.data)
                    continue;

                const lfs::rendering::BoundingBox box{
                    .min = cb.data->min,
                    .max = cb.data->max,
                    .transform = glm::inverse(cb.world_transform)};

                const glm::vec3 base_color = cb.data->inverse
                                                 ? glm::vec3(1.0f, 0.2f, 0.2f)
                                                 : cb.data->color;
                const bool is_selected = (cb.node_id == selected_cropbox_id);
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

        // Coordinate axes
        if (settings_.show_coord_axes && engine_) {
            auto axes_result = engine_->renderCoordinateAxes(viewport, settings_.axes_size, settings_.axes_visibility);
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

        // Camera frustums - requires both master toggle AND scene graph visibility
        if (settings_.show_camera_frustums && engine_ && context.scene_manager) {
            // Check which cameras are visible in the scene graph
            auto visible_indices = context.scene_manager->getScene().getVisibleCameraIndices();

            // Only render frustums if there are visible camera nodes
            if (!visible_indices.empty()) {
                // Get cameras from scene manager's trainer
                auto* trainer_manager = context.scene_manager->getTrainerManager();
                if (!trainer_manager || !trainer_manager->hasTrainer()) {
                    // No trainer, can't get camera data
                    return;
                }

                auto all_cameras = trainer_manager->getCamList();
                LOG_TRACE("Retrieved {} cameras from trainer manager", all_cameras.size());

                // Filter to only visible cameras
                std::vector<std::shared_ptr<const lfs::core::Camera>> cameras;
                cameras.reserve(visible_indices.size());
                for (size_t i = 0; i < all_cameras.size(); ++i) {
                    if (visible_indices.contains(static_cast<int>(i))) {
                        cameras.push_back(all_cameras[i]);
                    }
                }
                LOG_TRACE("Filtered to {} visible cameras", cameras.size());

                if (!cameras.empty()) {
                    // Find the actual index for the hovered camera ID
                    int highlight_index = -1;
                    if (hovered_camera_id_ >= 0) {
                        for (size_t i = 0; i < cameras.size(); ++i) {
                            if (cameras[i]->uid() == hovered_camera_id_) {
                                highlight_index = static_cast<int>(i);
                                break;
                            }
                        }
                    }

                    // Get scene transform from visible nodes (applies alignment transform)
                    glm::mat4 scene_transform(1.0f);
                    auto visible_transforms = context.scene_manager->getScene().getVisibleNodeTransforms();
                    if (!visible_transforms.empty()) {
                        scene_transform = visible_transforms[0];
                    }

                    // Render frustums with scene transform
                    LOG_TRACE("Rendering {} camera frustums with scale {}, highlighted index: {} (ID: {})",
                              cameras.size(), settings_.camera_frustum_scale, highlight_index, hovered_camera_id_);

                    auto frustum_result = engine_->renderCameraFrustumsWithHighlight(
                        cameras, viewport,
                        settings_.camera_frustum_scale,
                        settings_.train_camera_color,
                        settings_.eval_camera_color,
                        highlight_index,
                        scene_transform);

                    if (!frustum_result) {
                        LOG_ERROR("Failed to render camera frustums: {}", frustum_result.error());
                    }

                    // Perform picking if requested
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

                            // Only process if camera ID actually changed
                            if (cam_id != hovered_camera_id_) {
                                int old_hover = hovered_camera_id_;
                                hovered_camera_id_ = cam_id;

                                // Only mark dirty on actual change
                                markDirty();
                                LOG_DEBUG("Camera hover changed: {} -> {}", old_hover, cam_id);
                            }
                        } else if (hovered_camera_id_ != -1) {
                            // Lost hover - only update if we had a hover before
                            int old_hover = hovered_camera_id_;
                            hovered_camera_id_ = -1;
                            markDirty();
                            LOG_DEBUG("Camera hover lost (was ID: {})", old_hover);
                        }
                    }
                }
            }
        }

        // Grid (render last for proper wireframe compositing)
        if (settings_.show_grid && engine_) {
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
        if (!cached_result_.valid || !cached_result_.depth || !cached_result_.depth->is_valid()) {
            return -1.0f;
        }

        const auto& depth = *cached_result_.depth;
        // depth is [1, H, W]
        if (depth.ndim() != 3) {
            return -1.0f;
        }

        int height = static_cast<int>(depth.size(1));
        int width = static_cast<int>(depth.size(2));

        if (x < 0 || x >= width || y < 0 || y >= height) {
            return -1.0f;
        }

        // Copy single pixel from GPU to CPU
        auto depth_cpu = depth.cpu();
        const float* data = depth_cpu.ptr<float>();
        float d = data[y * width + x];

        // Check for invalid depth (large value means no hit)
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