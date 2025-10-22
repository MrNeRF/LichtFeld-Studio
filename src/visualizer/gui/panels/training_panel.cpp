/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/panels/training_panel.hpp"
#include "core/events.hpp"
#include "core/logger.hpp"
#include "gui/ui_widgets.hpp"
#include "gui/utils/windows_utils.hpp"
#include "visualizer_impl.hpp"

#include <chrono>
#include <deque>
#include <imgui.h>

namespace gs::gui::panels {

    // Iteration rate tracking
    struct IterationRateTracker {
        struct Sample {
            int iteration;
            std::chrono::steady_clock::time_point timestamp;
        };

        std::deque<Sample> samples;
        float window_seconds = 5.0f; // Configurable averaging window

        void addSample(int iteration) {
            auto now = std::chrono::steady_clock::now();
            samples.push_back({iteration, now});

            // Remove old samples outside the window
            while (!samples.empty()) {
                auto age = std::chrono::duration_cast<std::chrono::milliseconds>(now - samples.front().timestamp).count() / 1000.0f;
                if (age <= window_seconds) {
                    break;
                }
                samples.pop_front();
            }
        }

        float getIterationsPerSecond() const {
            if (samples.size() < 2) {
                return 0.0f;
            }

            const auto& oldest = samples.front();
            const auto& newest = samples.back();

            int iter_diff = newest.iteration - oldest.iteration;
            auto time_diff = std::chrono::duration_cast<std::chrono::milliseconds>(newest.timestamp - oldest.timestamp).count() / 1000.0f;

            if (time_diff <= 0.0f) {
                return 0.0f;
            }

            return iter_diff / time_diff;
        }

        void clear() {
            samples.clear();
        }

        void setWindowSeconds(float seconds) {
            window_seconds = seconds;
        }
    };

#ifdef WIN32
    void SaveProjectFileDialog(bool* p_open) {
        // show native windows file dialog for project directory selection
        PWSTR filePath = nullptr;
        if (SUCCEEDED(gs::gui::utils::selectFileNative(filePath, nullptr, 0, true))) {
            std::filesystem::path project_path(filePath);
            events::cmd::SaveProject{project_path}.emit();
            LOG_INFO("Saving project file into : {}", std::filesystem::path(project_path).string());
            *p_open = false;
        }
    }
#endif // WIN32

    void DrawTrainingParameters(const UIContext& ctx) {
        auto* trainer_manager = ctx.viewer->getTrainerManager();
        if (!trainer_manager || !trainer_manager->hasTrainer()) {
            return;
        }

        // Get current state to determine if we can edit
        auto trainer_state = trainer_manager->getState();
        bool can_edit = (trainer_state == TrainerManager::State::Ready);

        // Get project to modify parameters if we're in edit mode
        auto project = ctx.viewer->getProject();
        if (!project) {
            return;
        }

        // Get parameters - either from project (if Ready) or from trainer (if training/completed)
        param::OptimizationParameters opt_params;
        param::DatasetConfig dataset_params;

        if (trainer_state == TrainerManager::State::Ready) {
            // Before training - get from project (editable)
            opt_params = project->getOptimizationParams();
            dataset_params = project->getProjectData().data_set_info;
        } else {
            // During/after training - get from trainer (read-only)
            const auto* trainer = trainer_manager->getTrainer();
            if (!trainer) {
                return;
            }
            const auto& params = trainer->getParams();
            opt_params = params.optimization;
            dataset_params = params.dataset;
        }

        // Track changes separately for optimization and dataset parameters
        bool opt_params_changed = false;
        bool dataset_params_changed = false;

        ImGui::PushStyleVar(ImGuiStyleVar_IndentSpacing, 12.0f);
        if (ImGui::BeginTable("DatasetTable", 2, ImGuiTableFlags_SizingStretchProp)) {
            ImGui::TableSetupColumn("Property", ImGuiTableColumnFlags_WidthFixed, 120.0f);
            ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);

            // Iterations - EDITABLE
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("Iterations:");
            ImGui::TableNextColumn();
            if (can_edit) {
                ImGui::PushItemWidth(-1);
                int iterations = static_cast<int>(opt_params.iterations);
                if (ImGui::InputInt("##iterations", &iterations, 1000, 5000)) {
                    if (iterations > 0 && iterations <= 1000000) {
                        opt_params.iterations = static_cast<size_t>(iterations);
                        opt_params_changed = true;
                    }
                }
                ImGui::PopItemWidth();
            } else {
                ImGui::Text("%zu", opt_params.iterations);
            }

            if (opt_params.strategy == "mcmc") {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Max Gaussians:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    if (ImGui::InputInt("##max_cap", &opt_params.max_cap, 10000, 100000)) {
                        if (opt_params.max_cap > 0) {
                            opt_params_changed = true;
                        }
                    }
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%d", opt_params.max_cap);
                }
            }
        }
        ImGui::EndTable();

        // Dataset Parameters
        if (ImGui::TreeNode("Dataset")) {
            if (ImGui::BeginTable("DatasetTable", 2, ImGuiTableFlags_SizingStretchProp)) {
                ImGui::TableSetupColumn("Property", ImGuiTableColumnFlags_WidthFixed, 120.0f);
                ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Path:");
                ImGui::TableNextColumn();
                ImGui::Text("%s", dataset_params.data_path.filename().string().c_str());

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Images:");
                ImGui::TableNextColumn();
                ImGui::Text("%s", dataset_params.images.c_str());

                // Resize Factor - EDITABLE
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Resize Factor:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    // Available options
                    static const int resize_options[] = {1, 2, 4, 8};
                    static const char* resize_labels[] = {"1", "2", "4", "8"};
                    static int current_index = 0; // default is 1
                    int array_size = IM_ARRAYSIZE(resize_labels);
                    // Set current_index to current value, if needed
                    for (int i = 0; i < array_size; ++i) {
                        if (dataset_params.resize_factor == resize_options[i]) {
                            current_index = i;
                        }
                    }

                    // Draw combo
                    if (ImGui::Combo("##resize_factor", &current_index, resize_labels, array_size)) {
                        dataset_params.resize_factor = resize_options[current_index];
                        dataset_params_changed = true;
                    }
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%d", dataset_params.resize_factor);
                }

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Max Width (px):");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    if (ImGui::InputInt("##max_width", &dataset_params.max_width, 80, 400)) {
                        if (dataset_params.max_width > 0 && dataset_params.max_width <= 4096) {
                            dataset_params_changed = true;
                        }
                    }
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%d", dataset_params.max_width);
                }

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("CPU Cache:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    if (ImGui::Checkbox("##use_cpu_cache", &dataset_params.loading_params.use_cpu_memory)) {
                        dataset_params_changed = true;
                    }
                } else {
                    ImGui::Text("%s", dataset_params.loading_params.use_cpu_memory ? "Enabled" : "Disabled");
                }

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("FS Cache:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    if (ImGui::Checkbox("##use_fs_cache", &dataset_params.loading_params.use_fs_cache)) {
                        dataset_params_changed = true;
                    }
                } else {
                    ImGui::Text("%s", dataset_params.loading_params.use_fs_cache ? "Enabled" : "Disabled");
                }

                // Test Every - EDITABLE (only shown if evaluation is enabled)
                if (opt_params.enable_eval) {
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("Test Every:");
                    ImGui::TableNextColumn();
                    if (can_edit) {
                        ImGui::PushItemWidth(-1);
                        if (ImGui::InputInt("##test_every", &dataset_params.test_every, 100, 500)) {
                            if (dataset_params.test_every > 0 && dataset_params.test_every <= 10000) {
                                dataset_params_changed = true;
                            }
                        }
                        ImGui::PopItemWidth();
                    } else {
                        ImGui::Text("%d", dataset_params.test_every);
                    }
                }

                if (!dataset_params.output_path.empty()) {
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("Output:");
                    ImGui::TableNextColumn();
                    ImGui::Text("%s", dataset_params.output_path.filename().string().c_str());
                }

                ImGui::EndTable();
            }
            ImGui::TreePop();
        }

        // Optimization Parameters
        if (ImGui::TreeNode("Optimization")) {
            if (ImGui::BeginTable("OptimizationTable", 2, ImGuiTableFlags_SizingStretchProp)) {
                ImGui::TableSetupColumn("Property", ImGuiTableColumnFlags_WidthFixed, 120.0f);
                ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Strategy:");
                ImGui::TableNextColumn();
                ImGui::Text("%s", opt_params.strategy.c_str());

                // Learning Rates section - ALL EDITABLE
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Learning Rates:");
                ImGui::TableNextColumn();

                // Position LR
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("  Position:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    if (ImGui::InputFloat("##means_lr", &opt_params.means_lr, 0.000001f, 0.00001f, "%.6f")) {
                        opt_params_changed = true;
                    }
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%.6f", opt_params.means_lr);
                }

                // SH Coeff LR
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("  SH Coeff:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    if (ImGui::InputFloat("##shs_lr", &opt_params.shs_lr, 0.0001f, 0.001f, "%.4f")) {
                        opt_params_changed = true;
                    }
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%.4f", opt_params.shs_lr);
                }

                // Opacity LR
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("  Opacity:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    if (ImGui::InputFloat("##opacity_lr", &opt_params.opacity_lr, 0.001f, 0.01f, "%.4f")) {
                        opt_params_changed = true;
                    }
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%.4f", opt_params.opacity_lr);
                }

                // Scaling LR
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("  Scaling:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    if (ImGui::InputFloat("##scaling_lr", &opt_params.scaling_lr, 0.0001f, 0.001f, "%.4f")) {
                        opt_params_changed = true;
                    }
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%.4f", opt_params.scaling_lr);
                }

                // Rotation LR
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("  Rotation:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    if (ImGui::InputFloat("##rotation_lr", &opt_params.rotation_lr, 0.0001f, 0.001f, "%.4f")) {
                        opt_params_changed = true;
                    }
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%.4f", opt_params.rotation_lr);
                }

                // Refinement section - ALL EDITABLE
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Refinement:");
                ImGui::TableNextColumn();

                // Refine Every
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("  Refine Every:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    int refine_every = static_cast<int>(opt_params.refine_every);
                    if (ImGui::InputInt("##refine_every", &refine_every, 10, 100)) {
                        if (refine_every > 0) {
                            opt_params.refine_every = static_cast<size_t>(refine_every);
                            opt_params_changed = true;
                        }
                    }
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%zu", opt_params.refine_every);
                }

                // Start Refine
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("  Start Refine:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    int start_refine = static_cast<int>(opt_params.start_refine);
                    if (ImGui::InputInt("##start_refine", &start_refine, 100, 500)) {
                        if (start_refine >= 0) {
                            opt_params.start_refine = static_cast<size_t>(start_refine);
                            opt_params_changed = true;
                        }
                    }
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%zu", opt_params.start_refine);
                }

                // Stop Refine
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("  Stop Refine:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    int stop_refine = static_cast<int>(opt_params.stop_refine);
                    if (ImGui::InputInt("##stop_refine", &stop_refine, 1000, 5000)) {
                        if (stop_refine >= 0) {
                            opt_params.stop_refine = static_cast<size_t>(stop_refine);
                            opt_params_changed = true;
                        }
                    }
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%zu", opt_params.stop_refine);
                }

                // Gradient Threshold
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("  Gradient Thr:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    if (ImGui::InputFloat("##grad_threshold", &opt_params.grad_threshold, 0.000001f, 0.00001f, "%.6f")) {
                        opt_params_changed = true;
                    }
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%.6f", opt_params.grad_threshold);
                }

                // Reset Every
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("  Reset Every:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    int reset_every = static_cast<int>(opt_params.reset_every);
                    if (ImGui::InputInt("##reset_every", &reset_every, 100, 1000)) {
                        if (reset_every >= 0) {
                            opt_params.reset_every = static_cast<size_t>(reset_every);
                            opt_params_changed = true;
                        }
                    }
                    ImGui::PopItemWidth();
                } else if (opt_params.reset_every > 0) {
                    ImGui::Text("%zu", opt_params.reset_every);
                } else {
                    ImGui::Text("Disabled");
                }

                // Strategy-specific parameters

                ImGui::EndTable();
            }
            ImGui::TreePop();
        }

        // Save Steps - FULLY EDITABLE
        if (ImGui::TreeNode("Save Steps")) {
            if (can_edit) {
                // Add new save step
                static int new_step = 1000;
                ImGui::InputInt("New Step", &new_step, 100, 1000);
                ImGui::SameLine();
                if (ImGui::Button("Add")) {
                    if (new_step > 0 && std::find(opt_params.save_steps.begin(),
                                                  opt_params.save_steps.end(),
                                                  new_step) == opt_params.save_steps.end()) {
                        opt_params.save_steps.push_back(new_step);
                        std::sort(opt_params.save_steps.begin(), opt_params.save_steps.end());
                        opt_params_changed = true;
                    }
                }

                ImGui::Separator();

                // List existing save steps with remove buttons
                for (size_t i = 0; i < opt_params.save_steps.size(); ++i) {
                    ImGui::PushID(static_cast<int>(i));

                    int step = static_cast<int>(opt_params.save_steps[i]);
                    ImGui::SetNextItemWidth(100);
                    if (ImGui::InputInt("##step", &step, 0, 0)) {
                        if (step > 0) {
                            opt_params.save_steps[i] = static_cast<size_t>(step);
                            std::sort(opt_params.save_steps.begin(), opt_params.save_steps.end());
                            opt_params_changed = true;
                        }
                    }

                    ImGui::SameLine();
                    if (ImGui::Button("Remove")) {
                        opt_params.save_steps.erase(opt_params.save_steps.begin() + i);
                        opt_params_changed = true;
                    }

                    ImGui::PopID();
                }

                if (opt_params.save_steps.empty()) {
                    ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "No save steps configured");
                }
            } else {
                // Read-only display
                if (!opt_params.save_steps.empty()) {
                    std::string steps_str;
                    for (size_t i = 0; i < opt_params.save_steps.size(); ++i) {
                        if (i > 0)
                            steps_str += ", ";
                        steps_str += std::to_string(opt_params.save_steps[i]);
                    }
                    ImGui::Text("%s", steps_str.c_str());
                } else {
                    ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "No save steps");
                }
            }
            ImGui::TreePop();
        }

        // Active Features - only show if any are enabled
        bool has_active_features = opt_params.use_bilateral_grid ||
                                   opt_params.pose_optimization != "none" ||
                                   opt_params.enable_eval ||
                                   opt_params.antialiasing ||
                                   opt_params.gut;

        has_active_features = true; // force show for now

        if (has_active_features && ImGui::TreeNode("Active Features")) {
            if (ImGui::BeginTable("FeaturesTable", 2, ImGuiTableFlags_SizingStretchProp)) {
                ImGui::TableSetupColumn("Feature", ImGuiTableColumnFlags_WidthFixed, 120.0f);
                ImGui::TableSetupColumn("Configuration", ImGuiTableColumnFlags_WidthStretch);

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                bool use_bilateral_grid_enabled = opt_params.use_bilateral_grid;
                if (!can_edit) {
                    ImGui::BeginDisabled();
                }
                if (ImGui::Checkbox("Bilateral Grid", &use_bilateral_grid_enabled)) {
                    opt_params.use_bilateral_grid = use_bilateral_grid_enabled;
                    opt_params_changed = true;
                }
                if (!can_edit) {
                    ImGui::EndDisabled();
                }

                ImGui::TableNextColumn();
                ImGui::Text("%dx%dx%d (LR: %.4f)",
                            opt_params.bilateral_grid_X,
                            opt_params.bilateral_grid_Y,
                            opt_params.bilateral_grid_W,
                            opt_params.bilateral_grid_lr);

                ImGui::TableNextRow();
                ImGui::TableNextColumn();

                if (!can_edit) {
                    ImGui::BeginDisabled();
                }
                bool gut_enabled = opt_params.gut;
                if (ImGui::Checkbox("GUT", &gut_enabled)) {
                    opt_params.gut = gut_enabled;
                    opt_params_changed = true;
                }
                if (!can_edit) {
                    ImGui::EndDisabled();
                }

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                bool bg_modulation_enabled = opt_params.bg_modulation;
                if (!can_edit) {
                    ImGui::BeginDisabled();
                }
                if (ImGui::Checkbox("BG Modulation", &bg_modulation_enabled)) {
                    opt_params.bg_modulation = bg_modulation_enabled;
                    opt_params_changed = true;
                }
                if (!can_edit) {
                    ImGui::EndDisabled();
                }

                if (opt_params.pose_optimization != "none") {
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("Pose Optimization:");
                    ImGui::TableNextColumn();
                    ImGui::Text("%s", opt_params.pose_optimization.c_str());
                }

                if (opt_params.enable_eval) {
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("Evaluation:");
                    ImGui::TableNextColumn();
                    if (!opt_params.eval_steps.empty()) {
                        std::string steps_str = "Steps: ";
                        for (size_t i = 0; i < opt_params.eval_steps.size(); ++i) {
                            if (i > 0)
                                steps_str += ", ";
                            steps_str += std::to_string(opt_params.eval_steps[i]);
                        }
                        ImGui::Text("%s", steps_str.c_str());
                    } else {
                        ImGui::Text("Enabled");
                    }
                }

                ImGui::EndTable();
            }
            ImGui::TreePop();
        }

        // Render Settings
        if (ImGui::TreeNode("Render Settings")) {
            if (ImGui::BeginTable("RenderTable", 2, ImGuiTableFlags_SizingStretchProp)) {
                ImGui::TableSetupColumn("Property", ImGuiTableColumnFlags_WidthFixed, 120.0f);
                ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Render Mode:");
                ImGui::TableNextColumn();
                ImGui::Text("%s", opt_params.render_mode.c_str());

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("SH Degree:");
                ImGui::TableNextColumn();
                ImGui::Text("%d", opt_params.sh_degree);

                ImGui::EndTable();
            }
            ImGui::TreePop();
        }

        // Apply changes if any were made and we can edit
        if ((opt_params_changed || dataset_params_changed) && can_edit) {
            // Update optimization parameters if they changed
            if (opt_params_changed) {
                project->setOptimizationParams(opt_params);
            }

            // Update dataset parameters if they changed
            if (dataset_params_changed) {
                auto project_data = project->getProjectData();

                // Only update the fields from DatasetConfig that we allow editing
                project_data.data_set_info.resize_factor = dataset_params.resize_factor;
                project_data.data_set_info.max_width = dataset_params.max_width;
                project_data.data_set_info.test_every = dataset_params.test_every;

                project_data.data_set_info.loading_params.use_cpu_memory = dataset_params.loading_params.use_cpu_memory;
                project_data.data_set_info.loading_params.use_fs_cache = dataset_params.loading_params.use_fs_cache;

                // Set the updated project data back
                project->setProjectData(project_data);
            }

            ImGui::Separator();
            ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.0f, 1.0f),
                               "Parameters updated - will be applied when training starts");
        }

        ImGui::PopStyleVar();
    }

    void DrawTrainingControls(const UIContext& ctx) {
        ImGui::Text("Training Control");
        ImGui::Separator();

        auto& state = TrainingPanelState::getInstance();

        // Direct call to TrainerManager - no state duplication
        auto* trainer_manager = ctx.viewer->getTrainerManager();
        if (!trainer_manager || !trainer_manager->hasTrainer()) {
            ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "No trainer loaded");
            return;
        }

        // Get state directly from the single source of truth
        auto trainer_state = trainer_manager->getState();
        int current_iteration = trainer_manager->getCurrentIteration();

        // Render controls based on trainer state
        switch (trainer_state) {
        case TrainerManager::State::Idle:
            ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "No trainer loaded");
            break;

        case TrainerManager::State::Ready:
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.6f, 0.2f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.7f, 0.3f, 1.0f));
            if (ImGui::Button("Start Training", ImVec2(-1, 0))) {
                events::cmd::StartTraining{}.emit();
            }
            ImGui::PopStyleColor(2);

            if (current_iteration > 0) {
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.5f, 0.5f, 0.7f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.6f, 0.6f, 0.8f, 1.0f));
                if (ImGui::Button("Reset Training", ImVec2(-1, 0))) {
                    trainer_manager->resetTraining();
                }
                ImGui::PopStyleColor(2);
            }

            break;

        case TrainerManager::State::Running:
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.7f, 0.5f, 0.1f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.8f, 0.6f, 0.2f, 1.0f));
            if (ImGui::Button("Pause", ImVec2(-1, 0))) {
                events::cmd::PauseTraining{}.emit();
            }
            ImGui::PopStyleColor(2);
            break;

        case TrainerManager::State::Paused:
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.6f, 0.2f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.7f, 0.3f, 1.0f));
            if (ImGui::Button("Resume", ImVec2(-1, 0))) {
                events::cmd::ResumeTraining{}.emit();
            }
            ImGui::PopStyleColor(2);

            // Reset button
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.5f, 0.5f, 0.7f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.6f, 0.6f, 0.8f, 1.0f));
            if (ImGui::Button("Reset Training", ImVec2(-1, 0))) {
                trainer_manager->resetTraining();
            }
            ImGui::PopStyleColor(2);

            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.7f, 0.2f, 0.2f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.8f, 0.3f, 0.3f, 1.0f));
            if (ImGui::Button("Stop Permanently", ImVec2(-1, 0))) {
                events::cmd::StopTraining{}.emit();
            }
            ImGui::PopStyleColor(2);
            break;

        case TrainerManager::State::Completed:
            ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), "Training Complete!");

            // Reset button for completed state
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.5f, 0.5f, 0.7f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.6f, 0.6f, 0.8f, 1.0f));
            if (ImGui::Button("Reset for New Training", ImVec2(-1, 0))) {
                trainer_manager->resetTraining();
            }
            ImGui::PopStyleColor(2);

            break;

        case TrainerManager::State::Error:
            ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "Training Error!");
            {
                auto error_msg = trainer_manager->getLastError();
                if (!error_msg.empty()) {
                    ImGui::TextWrapped("%s", error_msg.c_str());
                }
                // Reset button
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.5f, 0.5f, 0.7f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.6f, 0.6f, 0.8f, 1.0f));
                if (ImGui::Button("Reset Training", ImVec2(-1, 0))) {
                    trainer_manager->resetTraining();
                }
                ImGui::PopStyleColor(2);
            }
            break;

        case TrainerManager::State::Stopping:
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Stopping...");
            break;
        }

        // Save checkpoint button (available during training)
        if (trainer_state == TrainerManager::State::Running ||
            trainer_state == TrainerManager::State::Paused) {

            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.1f, 0.4f, 0.7f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.2f, 0.5f, 0.8f, 1.0f));
            if (ImGui::Button("Save Checkpoint", ImVec2(-1, 0))) {
                events::cmd::SaveCheckpoint{}.emit();
                state.save_in_progress = true;
                state.save_start_time = std::chrono::steady_clock::now();
            }
            ImGui::PopStyleColor(2);
        }

        ImGui::Separator();
        if (ImGui::CollapsingHeader("Training Parameters", ImGuiTreeNodeFlags_DefaultOpen)) {
            DrawTrainingParameters(ctx);
        }

        // Save feedback
        if (state.save_in_progress) {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                               now - state.save_start_time)
                               .count();
            if (elapsed < 2000) {
                ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), "Checkpoint saved!");
            } else {
                state.save_in_progress = false;
            }
        }

        // Status display
        ImGui::Separator();

        // Helper to convert state to string
        const char* state_str = "Unknown";
        switch (trainer_state) {
        case TrainerManager::State::Idle: state_str = "Idle"; break;
        case TrainerManager::State::Ready: state_str = "Ready"; break;
        case TrainerManager::State::Running: state_str = "Running"; break;
        case TrainerManager::State::Paused: state_str = "Paused"; break;
        case TrainerManager::State::Stopping: state_str = "Stopping"; break;
        case TrainerManager::State::Completed: state_str = "Completed"; break;
        case TrainerManager::State::Error: state_str = "Error"; break;
        }

        // Static tracker instance
        static IterationRateTracker g_iter_rate_tracker;

        ImGui::Text("Status: %s", state_str);
        // Update iteration rate tracker
        g_iter_rate_tracker.addSample(current_iteration);
        // Get iteration rate
        float iters_per_sec = g_iter_rate_tracker.getIterationsPerSecond();
        // Display iteration with rate
        ImGui::Text("Iteration: %d (%.1f iters/sec)", current_iteration, iters_per_sec);

        int num_splats = trainer_manager->getNumSplats();
        ImGui::Text("num Splats: %d", num_splats);

        // display memory usage
        size_t free_t, total_t;
        cudaMemGetInfo(&free_t, &total_t);
        size_t used_t = total_t - free_t;

        ImVec4 memColor = ImVec4(0.0f, 0.0f, 0.0f, 1.0f);
        float pctUsed = (used_t / 1e9f) / (total_t / 1e9f) * 100;

        if (pctUsed > 75) {
            memColor = ImVec4(0.9f, 0.2f, 0.2f, 1.0f); // red
        }

        ImGui::TextColored(memColor, "Used GPU Memory: %.1f%% (%.1f/%.1f GB)", pctUsed, used_t / 1e9f, total_t / 1e9f);
    }

} // namespace gs::gui::panels
