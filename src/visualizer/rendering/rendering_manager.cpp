/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rendering_manager.hpp"
#include "core/image_io.hpp" // Use existing image_io utilities
#include "core/logger.hpp"
#include "core/splat_data.hpp"
#include "geometry/euclidean_transform.hpp"
#include "rendering/rendering.hpp"
#include "scene/scene_manager.hpp"
#include "training/training_manager.hpp"
#include <glad/glad.h>
#include <stdexcept>

namespace gs::visualizer {

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

    unsigned int GTTextureCache::getGTTexture(int cam_id, const std::filesystem::path& image_path) {
        // Check if already cached
        if (auto it = texture_cache_.find(cam_id); it != texture_cache_.end()) {
            it->second.last_access = std::chrono::steady_clock::now();
            LOG_TRACE("GT texture cache hit for camera {}", cam_id);
            return it->second.texture_id;
        }

        // Load new texture
        LOG_DEBUG("Loading GT image for camera {}: {}", cam_id, image_path.string());
        unsigned int texture_id = loadTexture(image_path);

        if (texture_id == 0) {
            LOG_ERROR("Failed to load GT texture for camera {}", cam_id);
            return 0;
        }

        // Evict oldest if cache is full
        if (texture_cache_.size() >= MAX_CACHE_SIZE) {
            evictOldest();
        }

        // Add to cache
        texture_cache_[cam_id] = {texture_id, std::chrono::steady_clock::now()};
        LOG_DEBUG("Cached GT texture {} for camera {}", texture_id, cam_id);

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
            auto [data, width, height, channels] = load_image(path);

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
            free_image(data);

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
    }

    void RenderingManager::initialize() {
        if (initialized_)
            return;

        LOG_TIMER("RenderingEngine initialization");

        engine_ = gs::rendering::RenderingEngine::create();
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
        events::cmd::ToggleSplitView::when([this](const auto&) {
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
        events::cmd::ToggleGTComparison::when([this](const auto&) {
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
            events::ui::GTComparisonModeChanged{
                .enabled = (settings_.split_view_mode == SplitViewMode::GTComparison)}
                .emit();
        });

        // Listen for camera view changes
        events::cmd::GoToCamView::when([this](const auto& event) {
            setCurrentCameraId(event.cam_id);
            LOG_DEBUG("Current camera ID set to: {}", event.cam_id);

            // If GT comparison was waiting for a camera, re-enable rendering
            if (settings_.split_view_mode == SplitViewMode::GTComparison && event.cam_id >= 0) {
                LOG_INFO("Camera {} selected, GT comparison now active", event.cam_id);
                markDirty();
            }
        });

        // Listen for split position changes
        events::ui::SplitPositionChanged::when([this](const auto& event) {
            std::lock_guard<std::mutex> lock(settings_mutex_);
            settings_.split_position = event.position;
            LOG_TRACE("Split position changed to: {}", event.position);
            markDirty();
        });

        // Listen for settings changes
        events::ui::RenderSettingsChanged::when([this](const auto& event) {
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
        events::ui::WindowResized::when([this](const auto&) {
            LOG_DEBUG("Window resized, clearing render cache");
            markDirty();
            cached_result_ = {};                  // Clear cache on resize
            last_render_size_ = glm::ivec2(0, 0); // Force size update
            render_texture_valid_ = false;
            gt_texture_cache_.clear(); // Clear GT cache on resize to avoid scaling issues
        });

        // Grid settings
        events::ui::GridSettingsChanged::when([this](const auto& event) {
            std::lock_guard<std::mutex> lock(settings_mutex_);
            settings_.show_grid = event.enabled;
            settings_.grid_plane = event.plane;
            settings_.grid_opacity = event.opacity;
            LOG_TRACE("Grid settings updated - enabled: {}, plane: {}, opacity: {}",
                      event.enabled, event.plane, event.opacity);
            markDirty();
        });

        // Scene changes
        events::state::SceneLoaded::when([this](const auto&) {
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

        events::state::SceneChanged::when([this](const auto&) {
            markDirty();
        });

        // PLY visibility changes
        events::cmd::SetPLYVisibility::when([this](const auto&) {
            markDirty();
        });

        // PLY added/removed
        events::state::PLYAdded::when([this](const auto&) {
            LOG_DEBUG("PLY added, marking render dirty");
            markDirty();
        });

        events::state::PLYRemoved::when([this](const auto&) {
            LOG_DEBUG("PLY removed, marking render dirty");
            markDirty();
        });

        // Crop box changes
        events::ui::CropBoxChanged::when([this](const auto& event) {
            std::lock_guard<std::mutex> lock(settings_mutex_);
            settings_.crop_min = event.min_bounds;
            settings_.crop_max = event.max_bounds;
            settings_.use_crop_box = event.enabled;
            LOG_TRACE("Crop box updated - enabled: {}", event.enabled);
            markDirty();
        });

        // Point cloud mode changes
        events::ui::PointCloudModeChanged::when([this](const auto& event) {
            std::lock_guard<std::mutex> lock(settings_mutex_);
            settings_.point_cloud_mode = event.enabled;
            settings_.voxel_size = event.voxel_size;
            LOG_DEBUG("Point cloud mode: {}, voxel size: {}",
                      event.enabled ? "enabled" : "disabled", event.voxel_size);
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
        settings_ = new_settings;
        markDirty();
    }

    RenderSettings RenderingManager::getSettings() const {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        return settings_;
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

    void RenderingManager::advanceSplitOffset() {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        settings_.split_view_offset++;
        markDirty();
    }

    SplitViewInfo RenderingManager::getSplitViewInfo() const {
        std::lock_guard<std::mutex> lock(split_info_mutex_);
        return current_split_info_;
    }

    gs::rendering::RenderingEngine* RenderingManager::getRenderingEngine() {
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

    void RenderingManager::renderToTexture(const RenderContext& context, SceneManager* scene_manager, const SplatData* model) {
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
        gs::rendering::ViewportData viewport_data{
            .rotation = context.viewport.getRotationMatrix(),
            .translation = context.viewport.getTranslation(),
            .size = render_size,
            .fov = settings_.fov};

        // Apply world transform
        if (!settings_.world_transform.isIdentity()) {
            glm::mat3 world_rot = settings_.world_transform.getRotationMat();
            glm::vec3 world_trans = settings_.world_transform.getTranslation();
            viewport_data.rotation = glm::transpose(world_rot) * viewport_data.rotation;
            viewport_data.translation = glm::transpose(world_rot) * (viewport_data.translation - world_trans);
        }

        gs::rendering::RenderRequest request{
            .viewport = viewport_data,
            .scaling_modifier = settings_.scaling_modifier,
            .antialiasing = settings_.antialiasing,
            .background_color = settings_.background_color,
            .crop_box = std::nullopt,
            .point_cloud_mode = settings_.point_cloud_mode,
            .voxel_size = settings_.voxel_size,
            .gut = settings_.gut,
            .equirectangular = settings_.equirectangular,
            .sh_degree = settings_.sh_degree};

        // Add crop box if enabled
        if (settings_.use_crop_box) {
            auto transform = settings_.crop_transform;
            request.crop_box = gs::rendering::BoundingBox{
                .min = settings_.crop_min,
                .max = settings_.crop_max,
                .transform = transform.inv().toMat4()};
        }

        // Render the gaussians
        auto render_result = engine_->renderGaussians(*model, request);
        if (render_result) {
            cached_result_ = *render_result;

            // Present to texture
            auto present_result = engine_->presentToScreen(
                cached_result_,
                glm::ivec2(0, 0),
                render_size);

            if (present_result) {
                render_texture_valid_ = true;
            } else {
                LOG_ERROR("Failed to present to texture: {}", present_result.error());
                render_texture_valid_ = false;
            }
        } else {
            LOG_ERROR("Failed to render gaussians to texture: {}", render_result.error());
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
            framerate_controller_.endFrame();
            return;
        }

        // Detect viewport size change and invalidate cache
        if (current_size != last_render_size_) {
            LOG_TRACE("Viewport size changed from {}x{} to {}x{}",
                      last_render_size_.x, last_render_size_.y,
                      current_size.x, current_size.y);
            needs_render_ = true;
            cached_result_ = {};
            render_texture_valid_ = false;
            last_render_size_ = current_size;
        }

        // Get current model
        const SplatData* model = scene_manager ? scene_manager->getModelForRendering() : nullptr;
        size_t model_ptr = reinterpret_cast<size_t>(model);

        // Detect model switch
        if (model_ptr != last_model_ptr_) {
            LOG_TRACE("Model pointer changed, clearing cache");
            needs_render_ = true;
            render_texture_valid_ = false;
            last_model_ptr_ = model_ptr;
            cached_result_ = {};
        }

        // Check if split view is enabled
        bool split_view_active = settings_.split_view_mode != SplitViewMode::Disabled;

        // For GT comparison, ensure we have a valid render texture
        if (settings_.split_view_mode == SplitViewMode::GTComparison) {
            if (current_camera_id_ < 0) {
                split_view_active = false;
                LOG_TRACE("GT comparison mode but no camera selected");
            } else if (!render_texture_valid_ && model) {
                // Force render to texture for GT comparison
                renderToTexture(context, scene_manager, model);
            }
        }

        // Determine if we should do a full render
        bool should_render = false;
        bool needs_render_now = needs_render_.load();

        // Always render if we need to update the display
        if (!cached_result_.image || needs_render_now || split_view_active) {
            should_render = true;
            needs_render_ = false;
        } else if (context.has_focus) {
            should_render = true;
        } else if (scene_manager && scene_manager->hasDataset()) {
            const auto* trainer_manager = scene_manager->getTrainerManager();
            if (trainer_manager && trainer_manager->isRunning()) {
                auto now = std::chrono::steady_clock::now();
                if (now - last_training_render_ > std::chrono::seconds(1)) {
                    should_render = true;
                    last_training_render_ = now;
                }
            }
        }

        // Clear and set viewport
        glViewport(0, 0, context.viewport.frameBufferSize.x, context.viewport.frameBufferSize.y);
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Set viewport region for rendering
        if (context.viewport_region) {
            glViewport(
                static_cast<GLint>(context.viewport_region->x),
                static_cast<GLint>(context.viewport_region->y),
                static_cast<GLsizei>(context.viewport_region->width),
                static_cast<GLsizei>(context.viewport_region->height));
        }

        if (should_render || !model) {
            doFullRender(context, scene_manager, model);
        } else if (cached_result_.image) {
            glm::ivec2 viewport_pos(0, 0);
            glm::ivec2 render_size = current_size;

            if (context.viewport_region) {
                viewport_pos = glm::ivec2(
                    static_cast<int>(context.viewport_region->x),
                    static_cast<int>(context.viewport_region->y));
            }

            engine_->presentToScreen(cached_result_, viewport_pos, render_size);
            renderOverlays(context);
        }

        framerate_controller_.endFrame();
    }

    void RenderingManager::doFullRender(const RenderContext& context, SceneManager* scene_manager, const SplatData* model) {
        LOG_TIMER_TRACE("Full render pass");

        render_count_++;
        LOG_TRACE("Render #{}, pick_requested: {}", render_count_, pick_requested_);

        glm::ivec2 render_size = context.viewport.windowSize;
        if (context.viewport_region) {
            render_size = glm::ivec2(
                static_cast<int>(context.viewport_region->width),
                static_cast<int>(context.viewport_region->height));
        }

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
                // Blit the texture to screen
                glm::ivec2 viewport_pos(0, 0);
                if (context.viewport_region) {
                    viewport_pos = glm::ivec2(
                        static_cast<int>(context.viewport_region->x),
                        static_cast<int>(context.viewport_region->y));
                }

                auto present_result = engine_->presentToScreen(
                    cached_result_,
                    viewport_pos,
                    render_size);

                if (!present_result) {
                    LOG_ERROR("Failed to present render result: {}", present_result.error());
                }
            }
        }

        // Always render overlays
        renderOverlays(context);
    }

    std::optional<gs::rendering::SplitViewRequest>
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
        gs::rendering::ViewportData viewport_data{
            .rotation = context.viewport.getRotationMatrix(),
            .translation = context.viewport.getTranslation(),
            .size = render_size,
            .fov = settings_.fov};

        // Apply world transform if needed
        if (!settings_.world_transform.isIdentity()) {
            glm::mat3 world_rot = settings_.world_transform.getRotationMat();
            glm::vec3 world_trans = settings_.world_transform.getTranslation();
            viewport_data.rotation = glm::transpose(world_rot) * viewport_data.rotation;
            viewport_data.translation = glm::transpose(world_rot) * (viewport_data.translation - world_trans);
        }

        // Create crop box if enabled
        std::optional<gs::rendering::BoundingBox> crop_box;
        if (settings_.use_crop_box) {
            auto transform = settings_.crop_transform;
            crop_box = gs::rendering::BoundingBox{
                .min = settings_.crop_min,
                .max = settings_.crop_max,
                .transform = transform.inv().toMat4()};
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
                const SplatData* model = scene_manager->getModelForRendering();
                if (model) {
                    renderToTexture(context, scene_manager, model);
                }
            }

            if (!render_texture_valid_) {
                LOG_ERROR("Failed to get cached render for GT comparison");
                return std::nullopt;
            }

            LOG_TRACE("Creating GT comparison split view for camera {}", current_camera_id_);

            return gs::rendering::SplitViewRequest{
                .panels = {
                    {.content_type = gs::rendering::PanelContentType::Image2D,
                     .model = nullptr,
                     .texture_id = gt_texture,
                     .label = "Ground Truth",
                     .start_position = 0.0f,
                     .end_position = settings_.split_position},
                    {.content_type = gs::rendering::PanelContentType::CachedRender,
                     .model = nullptr,
                     .texture_id = cached_render_texture_,
                     .label = "Rendered",
                     .start_position = settings_.split_position,
                     .end_position = 1.0f}},
                .viewport = viewport_data,
                .scaling_modifier = settings_.scaling_modifier,
                .antialiasing = settings_.antialiasing,
                .background_color = settings_.background_color,
                .crop_box = crop_box,
                .point_cloud_mode = settings_.point_cloud_mode,
                .voxel_size = settings_.voxel_size,
                .gut = settings_.gut,
                .show_dividers = true,
                .divider_color = glm::vec4(1.0f, 0.85f, 0.0f, 1.0f),
                .show_labels = true,
                .sh_degree = settings_.sh_degree};
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

            return gs::rendering::SplitViewRequest{
                .panels = {
                    {.content_type = gs::rendering::PanelContentType::Model3D,
                     .model = visible_nodes[left_idx]->model.get(),
                     .texture_id = 0,
                     .label = visible_nodes[left_idx]->name,
                     .start_position = 0.0f,
                     .end_position = settings_.split_position},
                    {.content_type = gs::rendering::PanelContentType::Model3D,
                     .model = visible_nodes[right_idx]->model.get(),
                     .texture_id = 0,
                     .label = visible_nodes[right_idx]->name,
                     .start_position = settings_.split_position,
                     .end_position = 1.0f}},
                .viewport = viewport_data,
                .scaling_modifier = settings_.scaling_modifier,
                .antialiasing = settings_.antialiasing,
                .background_color = settings_.background_color,
                .crop_box = crop_box,
                .point_cloud_mode = settings_.point_cloud_mode,
                .voxel_size = settings_.voxel_size,
                .gut = settings_.gut,
                .show_dividers = true,
                .divider_color = glm::vec4(1.0f, 0.85f, 0.0f, 1.0f),
                .show_labels = true,
                .sh_degree = settings_.sh_degree,
            };
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

        gs::rendering::ViewportData viewport{
            .rotation = context.viewport.getRotationMatrix(),
            .translation = context.viewport.getTranslation(),
            .size = render_size,
            .fov = settings_.fov};

        // Grid
        if (settings_.show_grid && engine_) {
            auto grid_result = engine_->renderGrid(
                viewport,
                static_cast<gs::rendering::GridPlane>(settings_.grid_plane),
                settings_.grid_opacity);

            if (!grid_result) {
                LOG_WARN("Failed to render grid: {}", grid_result.error());
            }
        }

        // Crop box wireframe
        if (settings_.show_crop_box && engine_) {
            auto transform = settings_.crop_transform;

            gs::rendering::BoundingBox box{
                .min = settings_.crop_min,
                .max = settings_.crop_max,
                .transform = transform.inv().toMat4()};

            auto bbox_result = engine_->renderBoundingBox(box, viewport, settings_.crop_color, settings_.crop_line_width);
            if (!bbox_result) {
                LOG_WARN("Failed to render bounding box: {}", bbox_result.error());
            }
        }

        // Coordinate axes
        if (settings_.show_coord_axes && engine_) {
            auto axes_result = engine_->renderCoordinateAxes(viewport, settings_.axes_size, settings_.axes_visibility);
            if (!axes_result) {
                LOG_WARN("Failed to render coordinate axes: {}", axes_result.error());
            }
        }

        // Camera frustums section
        if (settings_.show_camera_frustums && engine_) {
            LOG_TRACE("Camera frustums enabled, checking for scene_manager...");

            if (!context.scene_manager) {
                LOG_ERROR("Camera frustums enabled but scene_manager is null in render context!");
                return;
            }

            // Get cameras from scene manager's trainer
            std::vector<std::shared_ptr<const Camera>> cameras;
            auto* trainer_manager = context.scene_manager->getTrainerManager();

            if (!trainer_manager) {
                LOG_WARN("Camera frustums enabled but trainer_manager is null");
                return;
            }

            if (!trainer_manager->hasTrainer()) {
                LOG_TRACE("Camera frustums enabled but no trainer is loaded");
                return;
            }

            cameras = trainer_manager->getCamList();
            LOG_TRACE("Retrieved {} cameras from trainer manager", cameras.size());

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

                // Render frustums with world transform
                LOG_TRACE("Rendering {} camera frustums with scale {}, highlighted index: {} (ID: {})",
                          cameras.size(), settings_.camera_frustum_scale, highlight_index, hovered_camera_id_);

                // Apply world transform to camera frustums so they move with the gaussian splat
                glm::mat4 world_transform = settings_.world_transform.toMat4();

                auto frustum_result = engine_->renderCameraFrustumsWithHighlight(
                    cameras, viewport,
                    settings_.camera_frustum_scale,
                    settings_.train_camera_color,
                    settings_.eval_camera_color,
                    highlight_index,
                    world_transform);

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
                        world_transform);

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
            } else {
                LOG_WARN("Camera frustums enabled but no cameras available");
            }
        }

        // Translation gizmo (render last so it's on top)
        if (settings_.show_translation_gizmo && engine_) {
            glm::vec3 gizmo_pos = settings_.world_transform.getTranslation();
            auto gizmo_result = engine_->renderTranslationGizmo(gizmo_pos, viewport, settings_.gizmo_scale);
            if (!gizmo_result) {
                LOG_WARN("Failed to render translation gizmo: {}", gizmo_result.error());
            }
        }
    }
} // namespace gs::visualizer