#pragma once

#include "core/events.hpp"
#include "input/input_types.hpp"
#include "internal/viewport.hpp"
#include "training/training_manager.hpp"
#include <GLFW/glfw3.h>
#include <chrono>
#include <glm/glm.hpp>
#include <memory>

namespace gs::visualizer {

    class InputController {
    public:
        InputController(GLFWwindow* window, Viewport& viewport);
        ~InputController();

        // Setup - MUST be called AFTER ImGui is initialized!
        void initialize();

        // Set training manager for camera view commands
        void setTrainingManager(std::shared_ptr<const TrainerManager> tm) {
            training_manager_ = tm;
        }

        // Called every frame by GUI manager to update viewport bounds
        void updateViewportBounds(float x, float y, float w, float h) {
            viewport_bounds_ = {x, y, w, h};
        }

        // Set special input modes
        void setPointCloudMode(bool enabled) {
            point_cloud_mode_ = enabled;
        }

    private:
        // Store original ImGui callbacks so we can chain
        struct {
            GLFWmousebuttonfun mouse_button = nullptr;
            GLFWcursorposfun cursor_pos = nullptr;
            GLFWscrollfun scroll = nullptr;
            GLFWkeyfun key = nullptr;
            GLFWdropfun drop = nullptr;
            GLFWwindowfocusfun focus = nullptr;
        } imgui_callbacks_;

        // Our callbacks that chain to ImGui
        static void mouseButtonCallback(GLFWwindow* w, int button, int action, int mods);
        static void cursorPosCallback(GLFWwindow* w, double x, double y);
        static void scrollCallback(GLFWwindow* w, double xoff, double yoff);
        static void keyCallback(GLFWwindow* w, int key, int scancode, int action, int mods);
        static void dropCallback(GLFWwindow* w, int count, const char** paths);
        static void windowFocusCallback(GLFWwindow* w, int focused);

        // Internal handlers
        void handleMouseButton(int button, int action, double x, double y);
        void handleMouseMove(double x, double y);
        void handleScroll(double xoff, double yoff);
        void handleKey(int key, int action, int mods);
        void handleFileDrop(const std::vector<std::string>& paths);
        void handleGoToCamView(const events::cmd::GoToCamView& event);

        // Helpers
        bool isInViewport(double x, double y) const;
        bool shouldCameraHandleInput() const;
        void updateCameraSpeed(bool increase);
        void publishCameraMove();

        // Core state
        GLFWwindow* window_;
        Viewport& viewport_;
        std::shared_ptr<const TrainerManager> training_manager_;

        // Viewport bounds for focus detection
        struct {
            float x, y, width, height;
        } viewport_bounds_{0, 0, 1920, 1080};

        // Camera state
        enum class DragMode { None,
                              Pan,
                              Rotate,
                              Orbit };
        DragMode drag_mode_ = DragMode::None;
        glm::dvec2 last_mouse_pos_{0, 0};

        // Key states (only what we actually need)
        bool key_r_pressed_ = false;
        bool key_ctrl_pressed_ = false;
        bool keys_wasd_[4] = {false, false, false, false}; // W,A,S,D

        // Special modes
        bool point_cloud_mode_ = false;

        // Throttling for camera events
        std::chrono::steady_clock::time_point last_camera_publish_;
        static constexpr auto camera_publish_interval_ = std::chrono::milliseconds(100);

        // Static instance for callbacks
        static InputController* instance_;
    };

} // namespace gs::visualizer
