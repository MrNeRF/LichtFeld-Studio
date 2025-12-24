/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include "core/event_bus.hpp"
#include "geometry/bounding_box.hpp"
#include <filesystem>
#include <glm/glm.hpp>
#include <optional>
#include <string>
#include <unordered_map>

namespace lfs::core {

    // Forward declarations
    class Scene;

    // Export format enum
    enum class ExportFormat { PLY,
                              SOG,
                              SPZ,
                              HTML_VIEWER };

// Clean event macro - uses lfs::core::event::bus()
#define EVENT(Name, ...)                                   \
    struct Name {                                          \
        using event_id = Name;                             \
        __VA_ARGS__                                        \
                                                           \
        void emit() const {                                \
            ::lfs::core::event::bus().emit(*this);         \
        }                                                  \
                                                           \
        static auto when(auto&& handler) {                 \
            return ::lfs::core::event::bus().when<Name>(   \
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
            EVENT(ResetTraining, );
            EVENT(SwitchToLatestCheckpoint, );
            EVENT(SaveCheckpoint, std::optional<int> iteration;);
            EVENT(LoadFile, std::filesystem::path path; bool is_dataset;);
            EVENT(LoadCheckpointForTraining, std::filesystem::path path;);
            EVENT(LoadConfigFile, std::filesystem::path path;);
            EVENT(ShowDatasetLoadPopup, std::filesystem::path dataset_path;);
            EVENT(ClearScene, );
            EVENT(SwitchToEditMode, ); // Keep trained model, discard dataset
            EVENT(ResetCamera, );
            EVENT(ShowWindow, std::string window_name; bool show;);
            EVENT(ExecuteConsole, std::string command;);
            EVENT(GoToCamView, int cam_id;);
            EVENT(AddPLY, std::filesystem::path path; std::string name;);
            EVENT(RemovePLY, std::string name; bool keep_children = false;);
            EVENT(RenamePLY, std::string old_name; std::string new_name;);
            EVENT(SetPLYVisibility, std::string name; bool visible;);
            EVENT(ExportNodeAs, std::string name; ExportFormat format;);
            EVENT(ExportAllMergedAs, ExportFormat format;);
            EVENT(ReparentNode, std::string node_name; std::string new_parent_name;); // Empty parent = root
            EVENT(AddGroup, std::string name; std::string parent_name;);              // Create empty group node
            EVENT(DuplicateNode, std::string name;);                                  // Duplicate node (and children if group)
            EVENT(MergeGroup, std::string name;);                                     // Merge group children into single PLY
            EVENT(SetNodeLocked, std::string name; bool locked;);                     // Lock/unlock node for editing
            EVENT(CropPLY, lfs::geometry::BoundingBox crop_box; bool inverse;);
            EVENT(ApplyCropBox, );
            EVENT(FitCropBoxToScene, bool use_percentile;);
            EVENT(ToggleCropInverse, );
            EVENT(CyclePLY, );
            EVENT(CycleSelectionVisualization, );
            EVENT(ToggleSplitView, );
            EVENT(ToggleGTComparison, );
            EVENT(Undo, );
            EVENT(Redo, );
            EVENT(DeleteSelected, ); // Delete selected Gaussians (soft delete)
            EVENT(InvertSelection, );
            EVENT(DeselectAll, );
            EVENT(SelectAll, );
            EVENT(CopySelection, );
            EVENT(PasteSelection, );
        } // namespace cmd

        // ============================================================================
        // Tools - Tool system events
        // ============================================================================
        namespace tools {
            EVENT(ToolEnabled, std::string tool_name;);
            EVENT(ToolDisabled, std::string tool_name;);
            EVENT(CropBoxSettingsChanged, bool show_box; bool use_box;);
            EVENT(AxesSettingsChanged, bool show_axes;);
            EVENT(TranslationGizmoSettingsChanged, bool enabled; float scale;);
            EVENT(SetToolbarTool, int tool_mode;);
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
            EVENT(TrainingCompleted, int iteration; float final_loss; float elapsed_seconds; bool success; std::optional<std::string> error;);
            EVENT(TrainingStopped, int iteration; bool user_requested;);

            // Scene state
            EVENT(SceneLoaded,
                  Scene* scene;
                  std::filesystem::path path;
                  enum class Type{PLY, Dataset, SOG, SPZ, Checkpoint} type;
                  size_t num_gaussians;
                  int checkpoint_iteration = 0;);
            EVENT(SceneCleared, );
            EVENT(ModelUpdated, int iteration; size_t num_gaussians;);
            EVENT(SceneChanged, );
            // node_type: 0=SPLAT, 1=GROUP, 2=CROPBOX
            EVENT(PLYAdded, std::string name; size_t node_gaussians; size_t total_gaussians; bool is_visible; std::string parent_name; bool is_group; int node_type;);
            EVENT(PLYRemoved, std::string name; bool children_kept = false; std::string parent_of_removed;);
            EVENT(NodeReparented, std::string name; std::string old_parent; std::string new_parent;);

            // Data loading
            EVENT(DatasetLoadStarted, std::filesystem::path path;);
            EVENT(DatasetLoadProgress, std::filesystem::path path; float progress; std::string step;);
            EVENT(DatasetLoadCompleted,
                  std::filesystem::path path;
                  bool success;
                  std::optional<std::string> error;
                  size_t num_images;
                  size_t num_points;);
            EVENT(ConfigLoadFailed, std::filesystem::path path; std::string error;);

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
        // UI - User interface updates
        // ============================================================================
        namespace ui {
            EVENT(WindowResized, int width; int height;);
            EVENT(CameraMove, glm::mat3 rotation; glm::vec3 translation;);
            EVENT(SpeedChanged, float current_speed; float max_speed;);
            EVENT(ZoomSpeedChanged, float zoom_speed; float max_zoom_speed;);
            EVENT(RenderSettingsChanged,
                  std::optional<int> sh_degree;
                  std::optional<float> fov;
                  std::optional<float> scaling_modifier;
                  std::optional<bool> antialiasing;
                  std::optional<glm::vec3> background_color;
                  std::optional<bool> equirectangular;);
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
            EVENT(NodeDeselected, );
            EVENT(CropBoxChanged,
                  glm::vec3 min_bounds;
                  glm::vec3 max_bounds;
                  bool enabled;);
            EVENT(CropBoxVisibilityChanged, bool visible;);
            EVENT(ConsoleResult, std::string command; std::string result;);
            EVENT(SplitPositionChanged, float position;);
            EVENT(GTComparisonModeChanged, bool enabled;);
            EVENT(FocusTrainingPanel, );
            EVENT(ToggleUI, );
            EVENT(ToggleFullscreen, );
        } // namespace ui

        // ============================================================================
        // Internal - System coordination events (minimal)
        // ============================================================================
        namespace internal {
            EVENT(TrainerReady, );
            EVENT(TrainingReadyToStart, );
            EVENT(WindowFocusLost, );
        } // namespace internal
    } // namespace events

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

} // namespace lfs::core