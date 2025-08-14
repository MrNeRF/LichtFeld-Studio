#pragma once

#include "core/events.hpp"
#include "rendering/framerate_controller.hpp"
#include "rendering/render_infinite_grid.hpp"
#include "rendering/renderer.hpp"
#include "rendering/rendering_pipeline.hpp"
#include "rendering/shader.hpp"
#include "scene/scene_manager.hpp"
#include <memory>

namespace gs {
    class RenderCoordinateAxes;
}

namespace gs::visualizer {

    // Forward declaration
    class BackgroundTool;

    struct RenderSettings {
        float fov = 60.0f;
        float scaling_modifier = 1.0f;
        bool antialiasing = false;
        bool show_crop_box = false;
        bool use_crop_box = false;
        bool show_coord_axes = false;
        bool show_grid = true;
        int grid_plane = 1; // Default to XZ plane
        float grid_opacity = 0.5f;
        bool adaptive_frame_rate = true;
        bool point_cloud_mode = false;
        float voxel_size = 0.01f;
    };

    struct ViewportRegion {
        float x, y, width, height;
    };

    class RenderingManager {
    public:
        struct RenderContext {
            const Viewport& viewport;
            const RenderSettings& settings;
            const RenderBoundingBox* crop_box;
            const RenderCoordinateAxes* coord_axes;
            const geometry::EuclideanTransform* world_to_user;
            const ViewportRegion* viewport_region = nullptr;
            bool has_focus = false;                          // Indicates if the viewport has focus for input handling
            const BackgroundTool* background_tool = nullptr; // NEW
        };

        RenderingManager();
        ~RenderingManager();

        // Initialize rendering resources
        void initialize();

        // Main render function
        void renderFrame(const RenderContext& context, SceneManager* scene_manager);

        // Settings
        void updateSettings(const RenderSettings& settings) { settings_ = settings; }
        const RenderSettings& getSettings() const { return settings_; }

        // Framerate control
        void updateFramerateSettings(const FramerateSettings& settings) { framerate_controller_.updateSettings(settings); }
        const FramerateSettings& getFramerateSettings() const { return framerate_controller_.getSettings(); }
        float getCurrentFPS() const { return framerate_controller_.getCurrentFPS(); }
        float getAverageFPS() const { return framerate_controller_.getAverageFPS(); }
        bool isPerformanceCritical() const { return framerate_controller_.isPerformanceCritical(); }
        void resetFramerateController() { framerate_controller_.reset(); }

        // Get shader for external use (crop box rendering)
        std::shared_ptr<Shader> getQuadShader() const { return quad_shader_; }

    private:
        void initializeShaders();
        void drawSceneFrame(const RenderContext& context, SceneManager* scene_manager, bool skip_render);
        void drawFocusIndicator(const RenderContext& context);
        void drawCropBox(const RenderContext& context);
        void drawCoordAxes(const RenderContext& context);
        void drawGrid(const RenderContext& context);
        bool hasCamChanged(const Viewport& current_viewport);
        bool hasSceneChanged(const RenderContext& context);
        void setupEventHandlers();

        RenderSettings settings_;
        std::shared_ptr<ScreenQuadRenderer> screen_renderer_;
        std::shared_ptr<Shader> quad_shader_;
        std::unique_ptr<RenderInfiniteGrid> infinite_grid_;
        bool initialized_ = false;

        // Framerate control
        FramerateController framerate_controller_;
        Viewport prev_viewport_state_;
        float prev_fov_ = 0;
        geometry::EuclideanTransform prev_world_to_usr_inv_;
        glm::vec3 prev_background_color_;
        glm::ivec2 prev_render_size_;
        RenderingPipeline::RenderResult prev_result_;

        bool prev_point_cloud_mode_ = false;
        float prev_voxel_size_ = 0.01f;

        // Scene loading tracking - for frame control
        bool scene_just_loaded_ = false;
        event::HandlerId scene_loaded_handler_id_ = 0;
        event::HandlerId grid_settings_handler_id_ = 0;
    };

} // namespace gs::visualizer