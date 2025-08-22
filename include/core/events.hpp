#pragma once
#include "core/event_bus.hpp"
#include <filesystem>
#include <glm/glm.hpp>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace gs {

    // Forward declarations
    class Scene;

// Clean event macro - uses gs::event::bus()
#define EVENT(Name, ...)                                   \
    struct Name {                                          \
        using event_id = Name;                             \
        __VA_ARGS__                                        \
                                                           \
        void emit() const {                                \
            ::gs::event::bus().emit(*this);                \
        }                                                  \
                                                           \
        static auto when(auto&& handler) {                 \
            return ::gs::event::bus().when<Name>(          \
                std::forward<decltype(handler)>(handler)); \
        }                                                  \
    }

    namespace events {

        // ============================================================================
        // Commands - User actions that request something to happen
        // ============================================================================
        namespace cmd {
            EVENT(StartTraining, );
            EVENT(PauseTraining, );
            EVENT(ResumeTraining, );
            EVENT(StopTraining, );
            EVENT(SaveCheckpoint, std::optional<int> iteration;);
            EVENT(LoadFile, std::filesystem::path path; bool is_dataset;);
            EVENT(LoadProject, std::filesystem::path path;);
            EVENT(ClearScene, );
            EVENT(ResetCamera, );
            EVENT(ShowWindow, std::string window_name; bool show;);
            EVENT(ExecuteConsole, std::string command;);
            EVENT(GoToCamView, int cam_id;);
            EVENT(AddPLY, std::filesystem::path path; std::string name;);
            EVENT(RemovePLY, std::string name;);
            EVENT(SetPLYVisibility, std::string name; bool visible;);
        } // namespace cmd

        // ============================================================================
        // Tools - Tool system events
        // ============================================================================
        namespace tools {
            EVENT(ToolEnabled, std::string tool_name;);
            EVENT(ToolDisabled, std::string tool_name;);
            EVENT(CropBoxSettingsChanged, bool show_box; bool use_box;);
            EVENT(AxesSettingsChanged, bool show_axes;);
        } // namespace tools

        // ============================================================================
        // State - Notifications about what has happened (broadcasts)
        // ============================================================================
        namespace state {
            // Training state
            EVENT(TrainingStarted, int total_iterations;);
            EVENT(TrainingProgress, int iteration; float loss; int num_gaussians; bool is_refining = false;);
            EVENT(TrainingPaused, int iteration;);
            EVENT(TrainingResumed, int iteration;);
            EVENT(TrainingCompleted, int iteration; float final_loss; bool success; std::optional<std::string> error;);
            EVENT(TrainingStopped, int iteration; bool user_requested;);

            // Scene state
            EVENT(SceneLoaded,
                  Scene* scene;
                  std::filesystem::path path;
                  enum class Type{PLY, Dataset} type;
                  size_t num_gaussians;);
            EVENT(SceneCleared, );
            EVENT(ModelUpdated, int iteration; size_t num_gaussians;);
            EVENT(SceneChanged, );
            EVENT(PLYAdded, std::string name; size_t total_gaussians; bool is_visible;);
            EVENT(PLYRemoved, std::string name;);

            // Data loading
            EVENT(DatasetLoadStarted, std::filesystem::path path;);
            EVENT(DatasetLoadProgress, std::filesystem::path path; float progress; std::string step;);
            EVENT(DatasetLoadCompleted,
                  std::filesystem::path path;
                  bool success;
                  std::optional<std::string> error;
                  size_t num_images;
                  size_t num_points;);

            // Evaluation
            EVENT(EvaluationStarted, int iteration; size_t num_images;);
            EVENT(EvaluationProgress, int iteration; size_t current; size_t total;);
            EVENT(EvaluationCompleted,
                  int iteration;
                  float psnr;
                  float ssim;
                  float lpips;
                  float elapsed_time;
                  int num_gaussians;);

            // System state
            EVENT(CheckpointSaved, int iteration; std::filesystem::path path;);
            EVENT(MemoryUsage,
                  size_t gpu_used;
                  size_t gpu_total;
                  float gpu_percent;
                  size_t ram_used;
                  size_t ram_total;
                  float ram_percent;);
            EVENT(FrameRendered, float render_ms; float fps; int num_gaussians;);
        } // namespace state

        // ============================================================================
        // Notifications - Important messages for the user
        // ============================================================================
        namespace notify {
            EVENT(Info, std::string message; int duration_ms = 3000;);
            EVENT(Success, std::string message; int duration_ms = 3000;);
            EVENT(Warning, std::string message; int duration_ms = 5000;);
            EVENT(Error, std::string message; std::string details = ""; int duration_ms = 0;);

            EVENT(MemoryWarning,
                  enum class Type{GPU, RAM} type;
                  float usage_percent;
                  std::string message;);

            EVENT(Log,
                  enum class Level{Debug, Info, Warning, Error} level;
                  std::string message;
                  std::string source = "";);
        } // namespace notify

        // ============================================================================
        // UI - User interface updates
        // ============================================================================
        namespace ui {
            EVENT(WindowResized, int width; int height;);
            EVENT(CameraMove, glm::mat3 rotation; glm::vec3 translation;);
            EVENT(SpeedChanged, float current_speed; float max_speed;);
            EVENT(RenderSettingsChanged,
                  std::optional<float> fov;
                  std::optional<float> scaling_modifier;
                  std::optional<bool> antialiasing;
                  std::optional<glm::vec3> background_color;);
            EVENT(RenderModeChanged, std::string old_mode; std::string new_mode;);
            EVENT(PointCloudModeChanged, bool enabled; float voxel_size;);
            EVENT(GridSettingsChanged,
                  bool enabled;
                  int plane;
                  float opacity;);
            EVENT(NodeSelected,
                  std::string path;
                  std::string type;
                  std::unordered_map<std::string, std::string> metadata;);
            EVENT(CropBoxChanged,
                  glm::vec3 min_bounds;
                  glm::vec3 max_bounds;
                  bool enabled;);
            EVENT(CropBoxVisibilityChanged, bool visible;);
            EVENT(ConsoleResult, std::string command; std::string result;);
        } // namespace ui

        // ============================================================================
        // Internal - System coordination events (minimal)
        // ============================================================================
        namespace internal {
            EVENT(TrainerReady, );
            EVENT(TrainingReadyToStart, );
            EVENT(WindowFocusLost, );
        } // namespace internal
    }     // namespace events

    // ============================================================================
    // Convenience functions
    // ============================================================================
    template <event::Event E>
    inline void emit(const E& event) {
        event.emit();
    }

    template <event::Event E>
    inline auto when(auto&& handler) {
        return E::when(std::forward<decltype(handler)>(handler));
    }

} // namespace gs