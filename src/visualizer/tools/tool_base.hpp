/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "input/input_controller.hpp"
#include "rendering/rendering_manager.hpp"
#include <string>
#include <string_view>

// Forward declaration for GLFW
struct GLFWwindow;

namespace lfs::vis {
    // Forward declarations
    class SceneManager;
    namespace visualizer {
        class RenderingManager;
    }
} // namespace lfs::vis

class Viewport;

namespace lfs::vis::gui {
    struct UIContext;
}

namespace lfs::vis::command {
    class CommandHistory;
}

namespace lfs::vis {

    // Forward declarations
    class ToolContext;

    // Viewport bounds for coordinate transformations
    struct ViewportBounds {
        float x = 0, y = 0, width = 1920, height = 1080;
    };

    // C++23 concept defining what a tool must provide
    template <typename T>
    concept Tool = requires(T t, const ToolContext& ctx, const lfs::vis::gui::UIContext& ui_ctx, bool* p_open) {
                       { t.getName() } -> std::convertible_to<std::string_view>;
                       { t.getDescription() } -> std::convertible_to<std::string_view>;
                       { t.isEnabled() } -> std::convertible_to<bool>;
                       { t.setEnabled(bool{}) } -> std::same_as<void>;
                       { t.initialize(ctx) } -> std::same_as<bool>;
                       { t.shutdown() } -> std::same_as<void>;
                       { t.update(ctx) } -> std::same_as<void>;
                       { t.renderUI(ui_ctx, p_open) } -> std::same_as<void>;
                   };

    // Concrete context passed to tools for accessing visualizer resources
    class ToolContext {
    public:
        ToolContext(RenderingManager* rm, SceneManager* sm, const Viewport* vp, GLFWwindow* win,
                    command::CommandHistory* ch = nullptr)
            : rendering_manager(rm),
              scene_manager(sm),
              viewport(vp),
              window(win),
              command_history(ch) {}

        // Direct access to components
        RenderingManager* getRenderingManager() const { return rendering_manager; }
        SceneManager* getSceneManager() const { return scene_manager; }
        const Viewport& getViewport() const { return *viewport; }
        GLFWwindow* getWindow() const { return window; }
        const ViewportBounds& getViewportBounds() const { return viewport_bounds_; }
        command::CommandHistory* getCommandHistory() const { return command_history; }

        // Update viewport bounds (called by GUI manager)
        void updateViewportBounds(float x, float y, float w, float h) {
            viewport_bounds_ = {x, y, w, h};
        }

        // Helper methods
        void requestRender() const {
            if (rendering_manager) {
                rendering_manager->markDirty();
            }
        }

        void logMessage(const std::string& msg); // Implementation will be in cpp file to avoid circular deps

    private:
        RenderingManager* rendering_manager;
        SceneManager* scene_manager;
        const Viewport* viewport;
        GLFWwindow* window;
        ViewportBounds viewport_bounds_;
        command::CommandHistory* command_history;
    };

    // Base class providing default implementations
    class ToolBase {
    public:
        virtual ~ToolBase() = default;

        virtual std::string_view getName() const = 0;
        virtual std::string_view getDescription() const = 0;

        bool isEnabled() const { return enabled_; }
        void setEnabled(bool enabled) {
            if (enabled_ != enabled) {
                enabled_ = enabled;
                onEnabledChanged(enabled);
            }
        }

        virtual bool initialize([[maybe_unused]] const ToolContext& ctx) { return true; }
        virtual void shutdown() {}
        virtual void update([[maybe_unused]] const ToolContext& ctx) {}
        virtual void renderUI(const lfs::vis::gui::UIContext& ui_ctx, bool* p_open) = 0;

    protected:
        virtual void onEnabledChanged([[maybe_unused]] bool enabled) {}

    private:
        bool enabled_ = false;
    };

} // namespace lfs::vis