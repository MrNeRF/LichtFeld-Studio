/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rendering_manager.hpp"
#include "core/camera.hpp"
#include "core/cuda_debug.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "core/mesh_data.hpp"
#include "core/path_utils.hpp"
#include "core/splat_data.hpp"
#include "geometry/euclidean_transform.hpp"
#include "passes/mesh_pass.hpp"
#include "passes/overlay_pass.hpp"
#include "passes/point_cloud_pass.hpp"
#include "passes/present_pass.hpp"
#include "passes/splat_raster_pass.hpp"
#include "passes/split_view_pass.hpp"
#include "render_pass.hpp"
#include "rendering/cuda_kernels.hpp"
#include "rendering/rasterizer/rasterization/include/rasterization_api_tensor.h"
#include "rendering/rasterizer/rasterization/include/rasterization_config.h"
#include "rendering/rendering.hpp"
#include "rendering/rendering_pipeline.hpp"
#include "scene/scene_manager.hpp"
#include "scene/scene_render_state.hpp"
#include "theme/theme.hpp"
#include "training/components/ppisp.hpp"
#include "training/components/ppisp_controller.hpp"
#include "training/trainer.hpp"
#include "training/training_manager.hpp"
#include <algorithm>
#include <cuda_runtime.h>
#include <glad/glad.h>
#include <glm/gtc/type_ptr.hpp>
#include <shared_mutex>
#include <stdexcept>
#include <string_view>

namespace lfs::vis {

    namespace {

        template <typename TRenderable>
        [[nodiscard]] const TRenderable* findRenderableByNodeId(const std::vector<TRenderable>& renderables,
                                                                const core::NodeId node_id) {
            const auto it = std::ranges::find_if(renderables, [node_id](const auto& item) {
                return item.node_id == node_id;
            });
            return it != renderables.end() ? &(*it) : nullptr;
        }

        [[nodiscard]] std::string makePipelineLoadSignature(
            const std::filesystem::path& image_path,
            const lfs::io::LoadParams& load_params) {
            auto signature = lfs::core::path_to_utf8(image_path) +
                             ":rf" + std::to_string(std::max(1, load_params.resize_factor)) +
                             "_mw" + std::to_string(load_params.max_width);
            if (load_params.undistort) {
                signature += "_ud";
            }
            return signature;
        }

        [[nodiscard]] std::string makeFallbackLoadSignature(
            const std::filesystem::path& image_path,
            const std::string_view loader_kind) {
            return lfs::core::path_to_utf8(image_path) + ":" + std::string(loader_kind);
        }

        [[nodiscard]] lfs::io::LoadParams normalizeGTLoadParams(const lfs::io::LoadParams& load_params) {
            auto effective_params = load_params;
            effective_params.resize_factor = std::max(1, effective_params.resize_factor);
            if (effective_params.max_width <= 0 || effective_params.max_width > GTTextureCache::MAX_TEXTURE_DIM) {
                effective_params.max_width = GTTextureCache::MAX_TEXTURE_DIM;
            }
            return effective_params;
        }

    } // namespace

    using namespace lfs::core::events;

    GTTextureCache::GTTextureCache() = default;

    GTTextureCache::~GTTextureCache() {
        clear();
    }

    void GTTextureCache::clear() {
        for (const auto& [id, entry] : texture_cache_) {
            if (!entry.interop_texture && entry.texture_id > 0) {
                glDeleteTextures(1, &entry.texture_id);
            }
        }
        texture_cache_.clear();
    }

    GTTextureCache::TextureInfo GTTextureCache::getGTTexture(
        const int cam_id,
        const std::filesystem::path& image_path,
        lfs::io::PipelinedImageLoader* const pipeline_loader,
        const lfs::io::LoadParams* const load_params) {
        const std::optional<lfs::io::LoadParams> effective_load_params =
            (pipeline_loader && load_params) ? std::optional<lfs::io::LoadParams>(normalizeGTLoadParams(*load_params))
                                             : std::nullopt;
        const std::string requested_signature =
            effective_load_params
                ? makePipelineLoadSignature(image_path, *effective_load_params)
                : makeFallbackLoadSignature(image_path, "fallback");

        if (const auto it = texture_cache_.find(cam_id);
            it != texture_cache_.end() && it->second.load_signature == requested_signature) {
            it->second.last_access = std::chrono::steady_clock::now();
            const auto& entry = it->second;
            const unsigned int tex_id = entry.interop_texture ? entry.interop_texture->getTextureID() : entry.texture_id;
            const glm::vec2 tex_scale = entry.interop_texture
                                            ? glm::vec2(entry.interop_texture->getTexcoordScaleX(),
                                                        entry.interop_texture->getTexcoordScaleY())
                                            : glm::vec2(1.0f);
            return {tex_id, entry.width, entry.height, entry.needs_flip, tex_scale};
        }

        if (const auto it = texture_cache_.find(cam_id); it != texture_cache_.end()) {
            if (!it->second.interop_texture && it->second.texture_id > 0) {
                glDeleteTextures(1, &it->second.texture_id);
            }
            texture_cache_.erase(it);
        }

        if (texture_cache_.size() >= MAX_CACHE_SIZE)
            evictOldest();

        const auto ext = image_path.extension().string();
        const bool is_jpeg = (ext == ".jpg" || ext == ".jpeg" || ext == ".JPG" || ext == ".JPEG");

        CacheEntry entry;
        entry.last_access = std::chrono::steady_clock::now();

        if (!nvcodec_loader_ && is_jpeg) {
            try {
                constexpr lfs::io::NvCodecImageLoader::Options OPTS{.device_id = 0, .decoder_pool_size = 2};
                nvcodec_loader_ = std::make_unique<lfs::io::NvCodecImageLoader>(OPTS);
            } catch (...) {
                nvcodec_loader_ = nullptr;
            }
        }

        TextureInfo info{};
        if (pipeline_loader && effective_load_params) {
            info = loadTextureFromLoader(*pipeline_loader, image_path, *effective_load_params, entry);
        }
        if (nvcodec_loader_ && is_jpeg) {
            if (info.texture_id == 0) {
                info = loadTextureGPU(image_path, entry);
            }
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
            return {};
        }

        entry.load_signature = requested_signature;
        texture_cache_[cam_id] = std::move(entry);
        return info;
    }

    void GTTextureCache::evictOldest() {
        if (texture_cache_.empty()) {
            return;
        }

        const auto oldest = std::min_element(texture_cache_.begin(), texture_cache_.end(),
                                             [](const auto& a, const auto& b) { return a.second.last_access < b.second.last_access; });

        if (!oldest->second.interop_texture && oldest->second.texture_id != 0) {
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

    GTTextureCache::TextureInfo GTTextureCache::loadTextureFromLoader(
        lfs::io::PipelinedImageLoader& loader,
        const std::filesystem::path& path,
        const lfs::io::LoadParams& params,
        CacheEntry& entry) {
        if (!std::filesystem::exists(path))
            return {};

        try {
            const auto tensor = loader.load_image_immediate(path, params);
            if (tensor.numel() == 0)
                return {};

            const auto& shape = tensor.shape();
            const int height = static_cast<int>(shape[1]);
            const int width = static_cast<int>(shape[2]);
            const auto hwc = tensor.permute({1, 2, 0}).contiguous();

            entry.interop_texture = std::make_unique<lfs::rendering::CudaGLInteropTexture>();
            if (auto result = entry.interop_texture->init(width, height); !result) {
                LOG_WARN("Failed to init GT interop texture from loader: {}", result.error());
                entry.interop_texture.reset();
                return loadTextureFromTensor(tensor, entry);
            }

            if (auto result = entry.interop_texture->updateFromTensor(hwc); !result) {
                LOG_WARN("Failed to upload GT texture from loader: {}", result.error());
                entry.interop_texture.reset();
                return loadTextureFromTensor(tensor, entry);
            }

            entry.width = width;
            entry.height = height;
            entry.needs_flip = true;

            const glm::vec2 tex_scale(
                entry.interop_texture->getTexcoordScaleX(),
                entry.interop_texture->getTexcoordScaleY());

            return {entry.interop_texture->getTextureID(), width, height, true, tex_scale};
        } catch (const std::exception& e) {
            LOG_WARN("GT loader path failed for {}: {}", lfs::core::path_to_utf8(path), e.what());
            entry.interop_texture.reset();
            return {};
        }
    }

    GTTextureCache::TextureInfo GTTextureCache::loadTextureFromTensor(const lfs::core::Tensor& tensor, CacheEntry& entry) {
        if (!tensor.is_valid() || tensor.ndim() != 3 || tensor.numel() == 0) {
            return {};
        }

        try {
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
                LOG_WARN("Unsupported GT tensor shape for CPU upload: [{}, {}, {}]",
                         static_cast<int>(shape[0]), static_cast<int>(shape[1]), static_cast<int>(shape[2]));
                return {};
            }

            if (formatted.device() == lfs::core::Device::CUDA) {
                formatted = formatted.cpu();
            }
            formatted = formatted.contiguous();

            if (formatted.dtype() != lfs::core::DataType::UInt8) {
                formatted = (formatted.clamp(0.0f, 1.0f) * 255.0f).to(lfs::core::DataType::UInt8);
            }

            unsigned int texture = 0;
            glGenTextures(1, &texture);
            glBindTexture(GL_TEXTURE_2D, texture);
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

            const GLenum format = (channels == 1) ? GL_RED : (channels == 4) ? GL_RGBA
                                                                             : GL_RGB;
            const GLenum internal = (channels == 1) ? GL_R8 : (channels == 4) ? GL_RGBA8
                                                                              : GL_RGB8;

            glTexImage2D(GL_TEXTURE_2D, 0, internal, width, height, 0,
                         format, GL_UNSIGNED_BYTE, formatted.ptr<unsigned char>());

            if (const GLenum gl_err = glGetError(); gl_err != GL_NO_ERROR) {
                glBindTexture(GL_TEXTURE_2D, 0);
                glDeleteTextures(1, &texture);
                LOG_WARN("Failed to upload GT texture through CPU path: {}", static_cast<int>(gl_err));
                return {};
            }
            glBindTexture(GL_TEXTURE_2D, 0);

            entry.interop_texture.reset();
            entry.texture_id = texture;
            entry.width = width;
            entry.height = height;
            entry.needs_flip = true;

            return {texture, width, height, true};
        } catch (const std::exception& e) {
            LOG_WARN("GT tensor CPU upload failed: {}", e.what());
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
        passes_.push_back(std::make_unique<SplitViewPass>());
        passes_.push_back(std::make_unique<SplatRasterPass>());
        splat_raster_pass_ = static_cast<SplatRasterPass*>(passes_.back().get());
        passes_.push_back(std::make_unique<PointCloudPass>());
        point_cloud_pass_ = static_cast<PointCloudPass*>(passes_.back().get());
        passes_.push_back(std::make_unique<PresentPass>());
        passes_.push_back(std::make_unique<MeshPass>());
        passes_.push_back(std::make_unique<OverlayPass>());
        overlay_pass_ = static_cast<OverlayPass*>(passes_.back().get());
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
            markDirty(DirtyFlag::SPLIT_VIEW);
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
                    pre_gt_equirectangular_ = settings_.equirectangular;
                    settings_.split_view_mode = SplitViewMode::GTComparison;
                    is_now_enabled = true;
                }
                markDirty(DirtyFlag::SPLIT_VIEW | DirtyFlag::SPLATS);
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
                markDirty(DirtyFlag::SPLIT_VIEW);
            }
        });

        // Listen for split position changes
        ui::SplitPositionChanged::when([this](const auto& event) {
            std::lock_guard<std::mutex> lock(settings_mutex_);
            settings_.split_position = event.position;
            LOG_TRACE("Split position changed to: {}", event.position);
            markDirty(DirtyFlag::SPLIT_VIEW);
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
            markDirty(DirtyFlag::SPLATS | DirtyFlag::CAMERA | DirtyFlag::BACKGROUND);
        });

        // Window resize
        ui::WindowResized::when([this](const auto&) {
            LOG_DEBUG("Window resized, clearing render cache");
            markDirty(DirtyFlag::VIEWPORT | DirtyFlag::CAMERA);
            cached_result_ = {};
            last_viewport_size_ = glm::ivec2(0, 0);
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
            markDirty(DirtyFlag::OVERLAY);
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
            if (point_cloud_pass_)
                point_cloud_pass_->resetCache();
            markDirty();
        });

        state::SceneCleared::when([this](const auto&) {
            cached_result_ = {};
            if (point_cloud_pass_)
                point_cloud_pass_->resetCache();
            render_texture_valid_.store(false, std::memory_order_relaxed);
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
            markDirty(DirtyFlag::SPLATS | DirtyFlag::MESH | DirtyFlag::OVERLAY);
        });

        // PLY added/removed
        state::PLYAdded::when([this](const auto&) {
            LOG_DEBUG("PLY added, marking render dirty");
            markDirty(DirtyFlag::SPLATS | DirtyFlag::MESH | DirtyFlag::OVERLAY);
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

            markDirty(DirtyFlag::SPLATS | DirtyFlag::MESH | DirtyFlag::OVERLAY | DirtyFlag::SPLIT_VIEW);
        });

        // Crop box changes (scene graph is source of truth, this just handles enable flag)
        ui::CropBoxChanged::when([this](const auto& event) {
            std::lock_guard<std::mutex> lock(settings_mutex_);
            settings_.use_crop_box = event.enabled;
            markDirty(DirtyFlag::SPLATS | DirtyFlag::OVERLAY);
        });

        // Ellipsoid changes (scene graph is source of truth, this just handles enable flag)
        ui::EllipsoidChanged::when([this](const auto& event) {
            std::lock_guard<std::mutex> lock(settings_mutex_);
            settings_.use_ellipsoid = event.enabled;
            markDirty(DirtyFlag::SPLATS | DirtyFlag::OVERLAY);
        });

        // Point cloud mode changes
        ui::PointCloudModeChanged::when([this](const auto& event) {
            std::lock_guard<std::mutex> lock(settings_mutex_);
            settings_.point_cloud_mode = event.enabled;
            settings_.voxel_size = event.voxel_size;
            LOG_DEBUG("Point cloud mode: {}, voxel size: {}",
                      event.enabled ? "enabled" : "disabled", event.voxel_size);
            cached_result_ = {};
            markDirty(DirtyFlag::SPLATS);
        });
    }

    void RenderingManager::markDirty() {
        markDirty(DirtyFlag::ALL);
    }

    void RenderingManager::markDirty(const DirtyMask flags) {
        dirty_mask_.fetch_or(flags, std::memory_order_relaxed);

        constexpr DirtyMask SPLAT_INVALIDATING =
            DirtyFlag::SPLATS | DirtyFlag::CAMERA | DirtyFlag::VIEWPORT |
            DirtyFlag::SELECTION | DirtyFlag::BACKGROUND | DirtyFlag::PPISP | DirtyFlag::SPLIT_VIEW;

        if (flags & SPLAT_INVALIDATING)
            render_texture_valid_.store(false, std::memory_order_relaxed);

        LOG_TRACE("Render marked dirty (flags: 0x{:x})", flags);
    }

    void RenderingManager::setViewportResizeActive(bool active) {
        const bool was_active = viewport_resize_active_.exchange(active);
        if (!was_active || active)
            return;

        if (viewport_resize_debounce_ == 0)
            viewport_resize_debounce_ = 1;

        markDirty(DirtyFlag::VIEWPORT | DirtyFlag::CAMERA | DirtyFlag::OVERLAY);
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
        markDirty(DirtyFlag::CAMERA);
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
        markDirty(DirtyFlag::CAMERA);
    }

    float RenderingManager::getScalingModifier() const {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        return settings_.scaling_modifier;
    }

    void RenderingManager::setScalingModifier(const float s) {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        settings_.scaling_modifier = s;
        markDirty(DirtyFlag::SPLATS);
    }

    void RenderingManager::syncSelectionGroupColor(const int group_id, const glm::vec3& color) {
        lfs::rendering::config::setSelectionGroupColor(group_id, make_float3(color.x, color.y, color.z));
        markDirty(DirtyFlag::SELECTION);
    }

    void RenderingManager::advanceSplitOffset() {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        settings_.split_view_offset++;
        markDirty(DirtyFlag::SPLIT_VIEW | DirtyFlag::SPLATS);
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
        if (!settings_.show_camera_frustums)
            return -1;

        auto now = std::chrono::steady_clock::now();
        if (now - last_pick_time_ < pick_throttle_interval_)
            return hovered_camera_id_;
        last_pick_time_ = now;

        if (!engine_ || !last_scene_manager_ || !has_pick_context_)
            return hovered_camera_id_;

        auto cameras = last_scene_manager_->getScene().getVisibleCameras();
        if (cameras.empty())
            return -1;

        glm::mat4 scene_transform(1.0f);
        auto transforms = last_scene_manager_->getScene().getVisibleNodeTransforms();
        if (!transforms.empty())
            scene_transform = transforms[0];

        auto pick_result = engine_->pickCameraFrustum(
            cameras, mouse_pos,
            glm::vec2(last_viewport_region_.x, last_viewport_region_.y),
            glm::vec2(last_viewport_region_.width, last_viewport_region_.height),
            last_viewport_data_,
            settings_.camera_frustum_scale,
            scene_transform);

        int cam_id = -1;
        if (pick_result)
            cam_id = *pick_result;

        if (cam_id != hovered_camera_id_) {
            LOG_DEBUG("Camera hover changed: {} -> {}", hovered_camera_id_, cam_id);
            hovered_camera_id_ = cam_id;
            markDirty(DirtyFlag::OVERLAY);
        }

        return hovered_camera_id_;
    }

    bool RenderingManager::renderPreviewFrame(SceneManager* const scene_manager,
                                              const glm::mat3& rotation,
                                              const glm::vec3& position,
                                              const float focal_length_mm,
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
            .viewport = {rotation, position, {width, height}, focal_length_mm, false, 1.0f},
            .scaling_modifier = settings_.scaling_modifier,
            .antialiasing = false,
            .sh_degree = 0,
            .background_color = bg,
            .crop_box = std::nullopt,
            .point_cloud_mode = settings_.point_cloud_mode,
            .voxel_size = settings_.voxel_size,
            .gut = settings_.gut,
            .equirectangular = settings_.equirectangular,
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

        if (scene_manager && (dirty_mask_.load(std::memory_order_relaxed) & DirtyFlag::SELECTION)) {
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
            const auto& shell_bg = theme().menu_background();
            glClearColor(shell_bg.x, shell_bg.y, shell_bg.z, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            return;
        }

        constexpr int kResizeDebounceFrames = 3;
        bool resize_completed = false;
        const bool resize_active = viewport_resize_active_.load(std::memory_order_relaxed);

        if (current_size != last_viewport_size_) {
            last_viewport_size_ = current_size;
            if (resize_active) {
                markDirty(DirtyFlag::OVERLAY);
                viewport_resize_debounce_ = kResizeDebounceFrames;
            } else {
                markDirty(DirtyFlag::VIEWPORT | DirtyFlag::CAMERA | DirtyFlag::OVERLAY);
            }
        } else if (viewport_resize_debounce_ > 0 && !resize_active) {
            if (--viewport_resize_debounce_ == 0) {
                markDirty(DirtyFlag::VIEWPORT | DirtyFlag::CAMERA);
                resize_completed = true;
            } else {
                markDirty(DirtyFlag::OVERLAY);
            }
        }

        const lfs::core::SplatData* const model = scene_manager ? scene_manager->getModelForRendering() : nullptr;
        const auto* const visible_point_cloud =
            (scene_manager && !model) ? scene_manager->getScene().getVisiblePointCloud() : nullptr;
        const bool has_visible_point_cloud = visible_point_cloud && visible_point_cloud->size() > 0;
        const size_t model_ptr = reinterpret_cast<size_t>(model);

        if (model_ptr != last_model_ptr_) {
            LOG_DEBUG("Model ptr changed: {} -> {}, size={}", last_model_ptr_, model_ptr, model ? model->size() : 0);
            markDirty(DirtyFlag::ALL);
            last_model_ptr_ = model_ptr;
            cached_result_ = {};
        }

        const bool is_training = scene_manager && scene_manager->hasDataset() &&
                                 scene_manager->getTrainerManager() &&
                                 scene_manager->getTrainerManager()->isRunning();

        if (is_training) {
            const auto now = std::chrono::steady_clock::now();
            const auto interval = std::chrono::duration<float>(
                framerate_controller_.getSettings().training_frame_refresh_time_sec);
            if (now - last_training_render_ > interval) {
                markDirty(DirtyFlag::SPLATS);
                last_training_render_ = now;
            }
        }

        if (settings_.split_view_mode == SplitViewMode::GTComparison) {
            if (current_camera_id_ < 0 || (!model && !has_visible_point_cloud)) {
                gt_context_.reset();
                gt_context_camera_id_ = -1;
            } else {
                gt_context_.reset();
                gt_context_camera_id_ = -1;

                if (auto* trainer_manager = scene_manager->getTrainerManager();
                    trainer_manager && trainer_manager->hasTrainer()) {
                    if (const auto* trainer = trainer_manager->getTrainer()) {
                        const auto loader_owner = trainer->getActiveImageLoader();
                        if (const auto cam = trainer_manager->getCamById(current_camera_id_)) {
                            lfs::io::LoadParams gt_load_params;
                            const lfs::io::LoadParams* gt_load_params_ptr = nullptr;
                            if (loader_owner) {
                                const auto gt_load_config = trainer->getGTLoadConfigSnapshot();
                                gt_load_params.resize_factor = gt_load_config.resize_factor;
                                gt_load_params.max_width = gt_load_config.max_width;
                                if (gt_load_config.undistort && cam->is_undistort_prepared()) {
                                    gt_load_params.undistort = &cam->undistort_params();
                                }
                                gt_load_params_ptr = &gt_load_params;
                            }

                            const auto gt_info = gt_texture_cache_.getGTTexture(
                                current_camera_id_,
                                cam->image_path(),
                                loader_owner.get(),
                                gt_load_params_ptr);
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

                if (gt_context_ && !render_texture_valid_.load(std::memory_order_relaxed)) {
                    dirty_mask_.fetch_or(DirtyFlag::SPLATS, std::memory_order_relaxed);
                }
            }
        } else {
            if (gt_context_) {
                gt_context_.reset();
                gt_context_camera_id_ = -1;
            }
        }

        if (!cached_result_.image &&
            (model || has_visible_point_cloud || settings_.split_view_mode != SplitViewMode::Disabled))
            dirty_mask_.fetch_or(DirtyFlag::ALL, std::memory_order_relaxed);
        if (settings_.split_view_mode != SplitViewMode::Disabled)
            dirty_mask_.fetch_or(DirtyFlag::SPLIT_VIEW, std::memory_order_relaxed);

        glViewport(0, 0, context.viewport.frameBufferSize.x, context.viewport.frameBufferSize.y);

        const auto& shell_bg = theme().menu_background();
        glClearColor(shell_bg.x, shell_bg.y, shell_bg.z, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        if (context.viewport_region) {
            const GLint x = static_cast<GLint>(context.viewport_region->x);
            const GLint y = context.viewport.frameBufferSize.y - static_cast<GLint>(context.viewport_region->y + context.viewport_region->height);
            const GLsizei w = static_cast<GLsizei>(context.viewport_region->width);
            const GLsizei h = static_cast<GLsizei>(context.viewport_region->height);
            glViewport(x, y, w, h);
            glScissor(x, y, w, h);
            glEnable(GL_SCISSOR_TEST);
        }

        doFullRender(context, scene_manager, model);

        if (resize_completed) {
            resize_completed_ = true;
            lfs::core::Tensor::trim_memory_pool();
        }

        if (context.viewport_region) {
            glDisable(GL_SCISSOR_TEST);
        }

        last_scene_manager_ = scene_manager;
        if (context.viewport_region) {
            last_viewport_region_ = *context.viewport_region;
            has_pick_context_ = true;
        }
    }

    void RenderingManager::doFullRender(const RenderContext& context, SceneManager* scene_manager,
                                        const lfs::core::SplatData* model) {
        LOG_TIMER_TRACE("RenderingManager::doFullRender");

        render_count_++;
        LOG_TRACE("Render #{}", render_count_);

        glm::ivec2 render_size = context.viewport.windowSize;
        glm::ivec2 viewport_pos(0, 0);
        if (context.viewport_region) {
            render_size = glm::ivec2(
                static_cast<int>(context.viewport_region->width),
                static_cast<int>(context.viewport_region->height));
            const int gl_y = context.viewport.frameBufferSize.y -
                             static_cast<int>(context.viewport_region->y) -
                             static_cast<int>(context.viewport_region->height);
            viewport_pos = glm::ivec2(static_cast<int>(context.viewport_region->x), gl_y);
        }

        glClearColor(settings_.background_color.r, settings_.background_color.g,
                     settings_.background_color.b, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        const DirtyMask frame_dirty = dirty_mask_.exchange(0);
        const bool count_frame = frame_dirty != 0;
        if (count_frame) {
            framerate_controller_.beginFrame();
        }

        SceneRenderState scene_state;
        if (scene_manager) {
            scene_state = scene_manager->buildRenderState();
        }

        const bool has_splats = model && model->size() > 0;
        const bool has_point_cloud = scene_state.point_cloud && scene_state.point_cloud->size() > 0;
        if (!has_point_cloud && point_cloud_pass_) {
            point_cloud_pass_->resetCache();
        }

        last_viewport_data_ = lfs::rendering::ViewportData{
            .rotation = context.viewport.getRotationMatrix(),
            .translation = context.viewport.getTranslation(),
            .size = render_size,
            .focal_length_mm = settings_.focal_length_mm,
            .orthographic = settings_.orthographic,
            .ortho_scale = settings_.ortho_scale};

        const FrameContext frame_ctx{
            .viewport = context.viewport,
            .viewport_region = context.viewport_region,
            .scene_manager = scene_manager,
            .model = model,
            .scene_state = std::move(scene_state),
            .settings = settings_,
            .render_size = render_size,
            .viewport_pos = viewport_pos,
            .frame_dirty = frame_dirty,
            .brush = {.active = brush_active_,
                      .x = brush_x_,
                      .y = brush_y_,
                      .radius = brush_radius_,
                      .add_mode = brush_add_mode_,
                      .selection_tensor = brush_selection_tensor_,
                      .preview_selection = preview_selection_,
                      .saturation_mode = brush_saturation_mode_,
                      .saturation_amount = brush_saturation_amount_,
                      .selection_mode = selection_mode_,
                      .output_screen_positions = output_screen_positions_},
            .gizmo = {.cropbox_active = cropbox_gizmo_active_,
                      .cropbox_min = pending_cropbox_min_,
                      .cropbox_max = pending_cropbox_max_,
                      .cropbox_transform = pending_cropbox_transform_,
                      .ellipsoid_active = ellipsoid_gizmo_active_,
                      .ellipsoid_radii = pending_ellipsoid_radii_,
                      .ellipsoid_transform = pending_ellipsoid_transform_},
            .hovered_camera_id = hovered_camera_id_,
            .current_camera_id = current_camera_id_,
            .hovered_gaussian_id = hovered_gaussian_id_,
            .selection_flash_intensity = getSelectionFlashIntensity(),
            .cached_render_texture = cached_render_texture_};

        FrameResources resources{
            .cached_result = cached_result_,
            .cached_result_size = cached_result_size_,
            .render_texture_valid = render_texture_valid_.load(std::memory_order_relaxed),
            .gt_context = gt_context_,
            .hovered_gaussian_id = hovered_gaussian_id_};

        if (!has_splats && !has_point_cloud) {
            const bool had_cached_output =
                resources.cached_result.image ||
                resources.cached_result.depth ||
                resources.cached_result.depth_right ||
                resources.cached_result.screen_positions ||
                resources.cached_result_size.x > 0 ||
                resources.cached_result_size.y > 0 ||
                resources.render_texture_valid;
            if (had_cached_output) {
                resources.cached_result = {};
                resources.cached_result_size = {0, 0};
                resources.render_texture_valid = false;
                resources.hovered_gaussian_id = -1;
                lfs::core::Tensor::trim_memory_pool();
            }
        }

        if (frame_ctx.settings.split_view_mode == SplitViewMode::GTComparison &&
            resources.gt_context && resources.gt_context->valid()) {
            const bool needs_gt_pre_render =
                !resources.render_texture_valid ||
                (has_splats && (frame_dirty & splat_raster_pass_->sensitivity())) ||
                (has_point_cloud && point_cloud_pass_ && (frame_dirty & point_cloud_pass_->sensitivity()));

            if (needs_gt_pre_render) {
                if (has_splats) {
                    splat_raster_pass_->execute(*engine_, frame_ctx, resources);
                    resources.splat_pre_rendered = true;
                } else if (has_point_cloud && point_cloud_pass_) {
                    point_cloud_pass_->execute(*engine_, frame_ctx, resources);
                    resources.splat_pre_rendered = true;
                }
            }
        }

        for (auto& pass : passes_) {
            if (pass->shouldExecute(frame_dirty, frame_ctx)) {
                pass->execute(*engine_, frame_ctx, resources);
            }
        }

        // Apply pass-produced side effects
        if (resources.additional_dirty)
            markDirty(resources.additional_dirty);
        if (resources.pivot_animation_end)
            setPivotAnimationEndTime(*resources.pivot_animation_end);

        // Write-back from FrameResources to manager state
        cached_result_ = resources.cached_result;
        cached_result_size_ = resources.cached_result_size;
        render_texture_valid_.store(resources.render_texture_valid, std::memory_order_relaxed);
        hovered_gaussian_id_ = resources.hovered_gaussian_id;

        {
            std::lock_guard<std::mutex> lock(split_info_mutex_);
            current_split_info_ = resources.split_view_executed ? resources.split_info : SplitViewInfo{};
        }

        if (count_frame) {
            framerate_controller_.endFrame();
        }
    }

    float RenderingManager::getDepthAtPixel(int x, int y) const {
        int viewport_width = cached_result_size_.x;
        int viewport_height = cached_result_size_.y;
        if (viewport_width <= 0 || viewport_height <= 0) {
            viewport_width = last_viewport_size_.x;
            viewport_height = last_viewport_size_.y;
            if (viewport_width <= 0 || viewport_height <= 0)
                return -1.0f;
        }

        float splat_depth = -1.0f;

        if (cached_result_.valid) {
            const lfs::core::Tensor* depth_ptr = nullptr;

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

            if (depth_ptr && depth_ptr->ndim() == 3) {
                const int depth_height = static_cast<int>(depth_ptr->size(1));
                const int depth_width = static_cast<int>(depth_ptr->size(2));

                int scaled_x = x;
                int scaled_y = y;
                if (depth_width != viewport_width || depth_height != viewport_height) {
                    scaled_x = static_cast<int>(static_cast<float>(x) * depth_width / viewport_width);
                    scaled_y = static_cast<int>(static_cast<float>(y) * depth_height / viewport_height);
                }

                if (scaled_x >= 0 && scaled_x < depth_width && scaled_y >= 0 && scaled_y < depth_height) {
                    float d;
                    const float* gpu_ptr = depth_ptr->ptr<float>() + scaled_y * depth_width + scaled_x;
                    CHECK_CUDA(cudaMemcpy(&d, gpu_ptr, sizeof(float), cudaMemcpyDeviceToHost));
                    if (d < 1e9f) {
                        splat_depth = d;
                    }
                }
            }
        }

        float mesh_depth = -1.0f;
        if (engine_ && engine_->hasMeshRender()) {
            const GLuint mesh_fbo = engine_->getMeshFramebuffer();
            if (mesh_fbo != 0 && x >= 0 && x < viewport_width && y >= 0 && y < viewport_height) {
                float ndc_depth = 1.0f;
                glBindFramebuffer(GL_READ_FRAMEBUFFER, mesh_fbo);
                glReadPixels(x, viewport_height - 1 - y, 1, 1,
                             GL_DEPTH_COMPONENT, GL_FLOAT, &ndc_depth);
                glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);

                constexpr float DEPTH_BG_THRESHOLD = 0.9999f;
                if (ndc_depth < DEPTH_BG_THRESHOLD) {
                    const float z_near = cached_result_.valid ? cached_result_.near_plane : lfs::rendering::DEFAULT_NEAR_PLANE;
                    const float z_far = cached_result_.valid ? cached_result_.far_plane : lfs::rendering::DEFAULT_FAR_PLANE;
                    const float z_ndc = ndc_depth * 2.0f - 1.0f;
                    const float A = (z_far + z_near) / (z_far - z_near);
                    const float B = (2.0f * z_far * z_near) / (z_far - z_near);
                    mesh_depth = B / (A - z_ndc);
                }
            }
        }

        if (splat_depth > 0.0f && mesh_depth > 0.0f) {
            return std::min(splat_depth, mesh_depth);
        }
        if (splat_depth > 0.0f) {
            return splat_depth;
        }
        if (mesh_depth > 0.0f) {
            return mesh_depth;
        }
        return -1.0f;
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

        const auto& cropboxes = sm->getScene().getVisibleCropBoxes();
        if (const auto* const cb = findRenderableByNodeId(cropboxes, sm->getActiveSelectionCropBoxId());
            cb && cb->data) {
            const glm::mat4 inv_transform = glm::inverse(cb->world_transform);
            const float* const t_ptr = glm::value_ptr(inv_transform);
            crop_t = lfs::core::Tensor::from_vector(std::vector<float>(t_ptr, t_ptr + 16), {4, 4});
            crop_min = lfs::core::Tensor::from_vector({cb->data->min.x, cb->data->min.y, cb->data->min.z}, {3});
            crop_max = lfs::core::Tensor::from_vector({cb->data->max.x, cb->data->max.y, cb->data->max.z}, {3});
            crop_inverse = cb->data->inverse;
        }

        lfs::core::Tensor ellip_t, ellip_radii;
        bool ellipsoid_inverse = false;

        const auto& ellipsoids = sm->getScene().getVisibleEllipsoids();
        if (const auto* const el = findRenderableByNodeId(ellipsoids, sm->getActiveSelectionEllipsoidId());
            el && el->data) {
            const glm::mat4 inv_transform = glm::inverse(el->world_transform);
            const float* const t_ptr = glm::value_ptr(inv_transform);
            ellip_t = lfs::core::Tensor::from_vector(std::vector<float>(t_ptr, t_ptr + 16), {4, 4});
            ellip_radii = lfs::core::Tensor::from_vector({el->data->radii.x, el->data->radii.y, el->data->radii.z}, {3});
            ellipsoid_inverse = el->data->inverse;
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

    void RenderingManager::applyDepthFilter(lfs::core::Tensor& selection) {
        if (!selection.is_valid())
            return;

        auto* const sm = services().sceneOrNull();
        if (!sm)
            return;

        const auto* const model = sm->getModelForRendering();
        if (!model || model->size() == 0)
            return;

        const auto settings = getSettings();
        if (!settings.depth_filter_enabled)
            return;

        const auto& means = model->means();
        if (!means.is_valid() || means.size(0) != selection.size(0))
            return;

        const glm::mat4 world_to_filter = settings.depth_filter_transform.inv().toMat4();
        const float* const t_ptr = glm::value_ptr(world_to_filter);
        const auto depth_t = lfs::core::Tensor::from_vector(std::vector<float>(t_ptr, t_ptr + 16), {4, 4});
        const auto depth_min = lfs::core::Tensor::from_vector(
            {settings.depth_filter_min.x, settings.depth_filter_min.y, settings.depth_filter_min.z}, {3});
        const auto depth_max = lfs::core::Tensor::from_vector(
            {settings.depth_filter_max.x, settings.depth_filter_max.y, settings.depth_filter_max.z}, {3});

        lfs::rendering::filter_selection_by_crop(
            selection, means,
            &depth_t, &depth_min, &depth_max, false,
            nullptr, nullptr, false);
    }

    void RenderingManager::applySelectionFilters(lfs::core::Tensor& selection,
                                                 const bool use_crop_filter,
                                                 const bool use_depth_filter) {
        if (use_crop_filter) {
            applyCropFilter(selection);
        }
        if (use_depth_filter) {
            applyDepthFilter(selection);
        }
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
        markDirty(DirtyFlag::SELECTION);
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
        markDirty(DirtyFlag::SELECTION);
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
        polygon_world_points_.clear();
        polygon_closed_ = closed;
        polygon_add_mode_ = add_mode;
        polygon_preview_world_space_ = false;
    }

    void RenderingManager::setPolygonPreviewWorldSpace(const std::vector<glm::vec3>& world_points,
                                                       const bool closed, const bool add_mode) {
        polygon_preview_active_ = true;
        polygon_points_.clear();
        polygon_world_points_ = world_points;
        polygon_closed_ = closed;
        polygon_add_mode_ = add_mode;
        polygon_preview_world_space_ = true;
    }

    void RenderingManager::clearPolygonPreview() {
        polygon_preview_active_ = false;
        polygon_points_.clear();
        polygon_world_points_.clear();
        polygon_closed_ = false;
        polygon_preview_world_space_ = false;
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

    void RenderingManager::clearSelectionPreviews() {
        clearPreviewSelection();
        clearBrushState();
        clearRectPreview();
        clearPolygonPreview();
        clearLassoPreview();
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

        markDirty(DirtyFlag::SPLATS);
    }

} // namespace lfs::vis
