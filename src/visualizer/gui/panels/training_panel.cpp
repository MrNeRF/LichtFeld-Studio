/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/panels/training_panel.hpp"
#include "core/events.hpp"
#include "core/logger.hpp"
#include "core/parameter_manager.hpp"
#include "core/parameters.hpp"
#include "core/services.hpp"
#include "gui/ui_widgets.hpp"
#include "gui/utils/windows_utils.hpp"
#include "theme/theme.hpp"
#include "visualizer_impl.hpp"

#include <chrono>
#include <cstring>
#include <deque>
#include <filesystem>
#include <set>
#include <imgui.h>

#ifdef _WIN32
#include <Windows.h>
#endif

namespace lfs::vis::gui::panels {

    namespace {
        constexpr float RATE_WINDOW_SECONDS = 5.0f;

        void scale_steps_vector(std::vector<size_t>& steps, const float scaler) {
            std::set<size_t> unique;
            for (const auto s : steps) {
                if (const auto scaled = static_cast<size_t>(static_cast<float>(s) * scaler); scaled > 0) {
                    unique.insert(scaled);
                }
            }
            steps.assign(unique.begin(), unique.end());
        }

        void apply_step_scaling(lfs::core::param::OptimizationParameters& opt, const float scaler) {
            if (scaler <= 0.0f)
                return;
            const auto scale = [scaler](const size_t v) {
                return static_cast<size_t>(static_cast<float>(v) * scaler);
            };
            opt.iterations = scale(opt.iterations);
            opt.start_refine = scale(opt.start_refine);
            opt.reset_every = scale(opt.reset_every);
            opt.stop_refine = scale(opt.stop_refine);
            opt.refine_every = scale(opt.refine_every);
            opt.sh_degree_interval = scale(opt.sh_degree_interval);
            scale_steps_vector(opt.eval_steps, scaler);
            scale_steps_vector(opt.save_steps, scaler);
        }
    } // namespace

    struct IterationRateTracker {
        struct Sample {
            int iteration;
            std::chrono::steady_clock::time_point timestamp;
        };

        std::deque<Sample> samples;

        void addSample(const int iteration) {
            const auto now = std::chrono::steady_clock::now();
            samples.push_back({iteration, now});

            while (!samples.empty()) {
                const auto age = std::chrono::duration<float>(now - samples.front().timestamp).count();
                if (age <= RATE_WINDOW_SECONDS)
                    break;
                samples.pop_front();
            }
        }

        [[nodiscard]] float getIterationsPerSecond() const {
            if (samples.size() < 2)
                return 0.0f;

            const auto& oldest = samples.front();
            const auto& newest = samples.back();
            const int iter_diff = newest.iteration - oldest.iteration;
            const auto time_diff = std::chrono::duration<float>(newest.timestamp - oldest.timestamp).count();

            return (time_diff > 0.0f) ? static_cast<float>(iter_diff) / time_diff : 0.0f;
        }

        void clear() { samples.clear(); }
    };

    void DrawTrainingAdvancedParameters(const UIContext& ctx) {

        // Advanced Training Parameters

        auto* const trainer_manager = ctx.viewer->getTrainerManager();
        if (!trainer_manager || !trainer_manager->hasTrainer()) {
            return;
        }

        auto* const param_manager = services().paramsOrNull();
        if (!param_manager) {
            ImGui::TextColored(ImVec4(1, 0, 0, 1), "ParameterManager not available");
            return;
        }

        if (const auto result = param_manager->ensureLoaded(); !result) {
            ImGui::TextColored(ImVec4(1, 0.5f, 0, 1), "Failed to load params: %s", result.error().c_str());
            return;
        }

        const auto trainer_state = trainer_manager->getState();
        const int current_iteration = trainer_manager->getCurrentIteration();
        const bool can_edit = (trainer_state == TrainerManager::State::Ready) && (current_iteration == 0);

        auto& opt_params = param_manager->getActiveParams();

        lfs::core::param::DatasetConfig dataset_params;
        if (can_edit) {
            dataset_params = trainer_manager->getEditableDatasetParams();
        } else {
            const auto* const trainer = trainer_manager->getTrainer();
            if (!trainer)
                return;
            dataset_params = trainer->getParams().dataset;
        }

        bool dataset_params_changed = false;

        bool has_masks = false;
        if (!dataset_params.data_path.empty()) {
            static constexpr std::array<const char*, 3> MASK_FOLDERS = {"masks", "mask", "segmentation"};
            for (const auto* const folder : MASK_FOLDERS) {
                if (std::filesystem::exists(dataset_params.data_path / folder)) {
                    has_masks = true;
                    break;
                }
            }
        }

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

                // Resize Factor
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Resize Factor:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    static constexpr int RESIZE_OPTIONS[] = {1, 2, 4, 8};
                    static constexpr const char* const RESIZE_LABELS[] = {"1", "2", "4", "8"};
                    static int current_index = 0;
                    for (int i = 0; i < IM_ARRAYSIZE(RESIZE_LABELS); ++i) {
                        if (dataset_params.resize_factor == RESIZE_OPTIONS[i])
                            current_index = i;
                    }
                    if (ImGui::Combo("##resize_factor", &current_index, RESIZE_LABELS, IM_ARRAYSIZE(RESIZE_LABELS))) {
                        dataset_params.resize_factor = RESIZE_OPTIONS[current_index];
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

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Output:");
                ImGui::TableNextColumn();
                {
                    const std::string output_display = dataset_params.output_path.empty()
                                                           ? "(not set)"
                                                           : dataset_params.output_path.filename().string();
                    ImGui::Text("%s", output_display.c_str());
                    if (!dataset_params.output_path.empty() && ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("%s", dataset_params.output_path.string().c_str());
                    }
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

                // Learning Rates section
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextColored(theme().palette.text_dim, "Learning Rates:");
                ImGui::TableNextColumn();

                // Position LR
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("  Position:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    ImGui::InputFloat("##means_lr", &opt_params.means_lr, 0.000001f, 0.00001f, "%.6f");
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
                    ImGui::InputFloat("##shs_lr", &opt_params.shs_lr, 0.0001f, 0.001f, "%.4f");
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
                    ImGui::InputFloat("##opacity_lr", &opt_params.opacity_lr, 0.001f, 0.01f, "%.4f");
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
                    ImGui::InputFloat("##scaling_lr", &opt_params.scaling_lr, 0.0001f, 0.001f, "%.4f");
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
                    ImGui::InputFloat("##rotation_lr", &opt_params.rotation_lr, 0.0001f, 0.001f, "%.4f");
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%.4f", opt_params.rotation_lr);
                }

                // Refinement section
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextColored(theme().palette.text_dim, "Refinement:");
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
                    ImGui::InputFloat("##grad_threshold", &opt_params.grad_threshold, 0.000001f, 0.00001f, "%.6f");
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
                        }
                    }
                    ImGui::PopItemWidth();
                } else if (opt_params.reset_every > 0) {
                    ImGui::Text("%zu", opt_params.reset_every);
                } else {
                    ImGui::Text("Disabled");
                }

                // SH Degree Interval
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("  SH Upgrade Every:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    int sh_deg_interval = static_cast<int>(opt_params.sh_degree_interval);
                    if (ImGui::InputInt("##sh_degree_interval", &sh_deg_interval, 100, 500)) {
                        if (sh_deg_interval > 0) {
                            opt_params.sh_degree_interval = static_cast<size_t>(sh_deg_interval);
                        }
                    }
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%zu", opt_params.sh_degree_interval);
                }

                ImGui::EndTable();
            }
            ImGui::TreePop();
        }

        // Save Steps
        if (ImGui::TreeNode("Save Steps")) {
            if (can_edit) {
                static int new_step = 1000;
                ImGui::InputInt("New Step", &new_step, 100, 1000);
                ImGui::SameLine();
                if (ImGui::Button("Add")) {
                    if (new_step > 0 && std::find(opt_params.save_steps.begin(),
                                                  opt_params.save_steps.end(),
                                                  new_step) == opt_params.save_steps.end()) {
                        opt_params.save_steps.push_back(new_step);
                        std::sort(opt_params.save_steps.begin(), opt_params.save_steps.end());
                    }
                }

                ImGui::Separator();

                for (size_t i = 0; i < opt_params.save_steps.size(); ++i) {
                    ImGui::PushID(static_cast<int>(i));

                    int step = static_cast<int>(opt_params.save_steps[i]);
                    ImGui::SetNextItemWidth(100);
                    if (ImGui::InputInt("##step", &step, 0, 0)) {
                        if (step > 0) {
                            opt_params.save_steps[i] = static_cast<size_t>(step);
                            std::sort(opt_params.save_steps.begin(), opt_params.save_steps.end());
                        }
                    }

                    ImGui::SameLine();
                    if (ImGui::Button("Remove")) {
                        opt_params.save_steps.erase(opt_params.save_steps.begin() + i);
                    }

                    ImGui::PopID();
                }

                if (opt_params.save_steps.empty()) {
                    ImGui::TextColored(darken(theme().palette.text_dim, 0.15f), "No save steps configured");
                }
            } else {
                if (!opt_params.save_steps.empty()) {
                    std::string steps_str;
                    for (size_t i = 0; i < opt_params.save_steps.size(); ++i) {
                        if (i > 0)
                            steps_str += ", ";
                        steps_str += std::to_string(opt_params.save_steps[i]);
                    }
                    ImGui::Text("%s", steps_str.c_str());
                } else {
                    ImGui::TextColored(darken(theme().palette.text_dim, 0.15f), "No save steps");
                }
            }
            ImGui::TreePop();
        }

        // Bilateral Grid Settings
        if (opt_params.use_bilateral_grid && ImGui::TreeNode("Bilateral Grid Settings")) {
            if (ImGui::BeginTable("BilateralTable", 2, ImGuiTableFlags_SizingStretchProp)) {
                ImGui::TableSetupColumn("Property", ImGuiTableColumnFlags_WidthFixed, 140.0f);
                ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Grid X:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    if (ImGui::InputInt("##bilateral_grid_X", &opt_params.bilateral_grid_X, 1, 4)) {
                        opt_params.bilateral_grid_X = std::max(1, opt_params.bilateral_grid_X);
                    }
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%d", opt_params.bilateral_grid_X);
                }

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Grid Y:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    if (ImGui::InputInt("##bilateral_grid_Y", &opt_params.bilateral_grid_Y, 1, 4)) {
                        opt_params.bilateral_grid_Y = std::max(1, opt_params.bilateral_grid_Y);
                    }
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%d", opt_params.bilateral_grid_Y);
                }

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Grid W:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    if (ImGui::InputInt("##bilateral_grid_W", &opt_params.bilateral_grid_W, 1, 2)) {
                        opt_params.bilateral_grid_W = std::max(1, opt_params.bilateral_grid_W);
                    }
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%d", opt_params.bilateral_grid_W);
                }

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Learning Rate:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    ImGui::InputFloat("##bilateral_grid_lr", &opt_params.bilateral_grid_lr, 0.0001f, 0.001f, "%.5f");
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%.5f", opt_params.bilateral_grid_lr);
                }

                ImGui::EndTable();
            }
            ImGui::TreePop();
        }

        // Mask Settings
        if (opt_params.mask_mode != lfs::core::param::MaskMode::None && ImGui::TreeNode("Mask Settings")) {
            if (ImGui::BeginTable("MaskTable", 2, ImGuiTableFlags_SizingStretchProp)) {
                ImGui::TableSetupColumn("Property", ImGuiTableColumnFlags_WidthFixed, 140.0f);
                ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Invert Masks:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::Checkbox("##invert_masks", &opt_params.invert_masks);
                } else {
                    ImGui::Text("%s", opt_params.invert_masks ? "Yes" : "No");
                }

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Threshold:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    ImGui::SliderFloat("##mask_threshold", &opt_params.mask_threshold, 0.0f, 1.0f, "%.2f");
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%.2f", opt_params.mask_threshold);
                }

                if (opt_params.mask_mode == lfs::core::param::MaskMode::Segment) {
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("Penalty Weight:");
                    ImGui::TableNextColumn();
                    if (can_edit) {
                        ImGui::PushItemWidth(-1);
                        ImGui::InputFloat("##mask_penalty_weight", &opt_params.mask_opacity_penalty_weight, 0.1f, 0.5f, "%.2f");
                        ImGui::PopItemWidth();
                    } else {
                        ImGui::Text("%.2f", opt_params.mask_opacity_penalty_weight);
                    }

                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("Penalty Power:");
                    ImGui::TableNextColumn();
                    if (can_edit) {
                        ImGui::PushItemWidth(-1);
                        ImGui::InputFloat("##mask_penalty_power", &opt_params.mask_opacity_penalty_power, 0.5f, 1.0f, "%.1f");
                        ImGui::PopItemWidth();
                    } else {
                        ImGui::Text("%.1f", opt_params.mask_opacity_penalty_power);
                    }
                }

                ImGui::EndTable();
            }
            ImGui::TreePop();
        }

        // Evaluation Settings
        if (opt_params.enable_eval && ImGui::TreeNode("Evaluation Settings")) {
            if (ImGui::BeginTable("EvalTable", 2, ImGuiTableFlags_SizingStretchProp)) {
                ImGui::TableSetupColumn("Property", ImGuiTableColumnFlags_WidthFixed, 140.0f);
                ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Save Images:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::Checkbox("##enable_save_eval_images", &opt_params.enable_save_eval_images);
                } else {
                    ImGui::Text("%s", opt_params.enable_save_eval_images ? "Yes" : "No");
                }

                ImGui::EndTable();
            }

            // Eval Steps
            {
                ImGui::Separator();
                ImGui::Text("Evaluation Steps:");
                if (can_edit) {
                    static int new_eval_step = 7000;
                    ImGui::InputInt("New Eval Step", &new_eval_step, 1000, 5000);
                    ImGui::SameLine();
                    if (ImGui::Button("Add##eval")) {
                        if (new_eval_step > 0 && std::find(opt_params.eval_steps.begin(),
                                                           opt_params.eval_steps.end(),
                                                           new_eval_step) == opt_params.eval_steps.end()) {
                            opt_params.eval_steps.push_back(new_eval_step);
                            std::sort(opt_params.eval_steps.begin(), opt_params.eval_steps.end());
                        }
                    }

                    for (size_t i = 0; i < opt_params.eval_steps.size(); ++i) {
                        ImGui::PushID(static_cast<int>(i + 1000));

                        int step = static_cast<int>(opt_params.eval_steps[i]);
                        ImGui::SetNextItemWidth(100);
                        if (ImGui::InputInt("##eval_step", &step, 0, 0)) {
                            if (step > 0) {
                                opt_params.eval_steps[i] = static_cast<size_t>(step);
                                std::sort(opt_params.eval_steps.begin(), opt_params.eval_steps.end());
                            }
                        }

                        ImGui::SameLine();
                        if (ImGui::Button("Remove##eval")) {
                            opt_params.eval_steps.erase(opt_params.eval_steps.begin() + i);
                        }

                        ImGui::PopID();
                    }

                    if (opt_params.eval_steps.empty()) {
                        ImGui::TextColored(darken(theme().palette.text_dim, 0.15f), "No eval steps configured");
                    }
                } else {
                    if (!opt_params.eval_steps.empty()) {
                        std::string steps_str;
                        for (size_t i = 0; i < opt_params.eval_steps.size(); ++i) {
                            if (i > 0)
                                steps_str += ", ";
                            steps_str += std::to_string(opt_params.eval_steps[i]);
                        }
                        ImGui::Text("%s", steps_str.c_str());
                    } else {
                        ImGui::TextColored(darken(theme().palette.text_dim, 0.15f), "No eval steps");
                    }
                }
            }
            ImGui::TreePop();
        }

        // Loss Parameters
        if (ImGui::TreeNode("Loss Parameters")) {
            if (ImGui::BeginTable("LossTable", 2, ImGuiTableFlags_SizingStretchProp)) {
                ImGui::TableSetupColumn("Property", ImGuiTableColumnFlags_WidthFixed, 140.0f);
                ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Lambda DSSIM:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    ImGui::SliderFloat("##lambda_dssim", &opt_params.lambda_dssim, 0.0f, 1.0f, "%.2f");
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%.2f", opt_params.lambda_dssim);
                }

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Opacity Reg:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    ImGui::InputFloat("##opacity_reg", &opt_params.opacity_reg, 0.001f, 0.01f, "%.4f");
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%.4f", opt_params.opacity_reg);
                }

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Scale Reg:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    ImGui::InputFloat("##scale_reg", &opt_params.scale_reg, 0.001f, 0.01f, "%.4f");
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%.4f", opt_params.scale_reg);
                }

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("TV Loss Weight:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    ImGui::InputFloat("##tv_loss_weight", &opt_params.tv_loss_weight, 1.0f, 5.0f, "%.1f");
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%.1f", opt_params.tv_loss_weight);
                }

                ImGui::EndTable();
            }
            ImGui::TreePop();
        }

        // Initialization Parameters
        if (ImGui::TreeNode("Initialization")) {
            if (ImGui::BeginTable("InitTable", 2, ImGuiTableFlags_SizingStretchProp)) {
                ImGui::TableSetupColumn("Property", ImGuiTableColumnFlags_WidthFixed, 140.0f);
                ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Init Opacity:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    ImGui::SliderFloat("##init_opacity", &opt_params.init_opacity, 0.01f, 1.0f, "%.2f");
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%.2f", opt_params.init_opacity);
                }

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Init Scaling:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    ImGui::InputFloat("##init_scaling", &opt_params.init_scaling, 0.01f, 0.1f, "%.3f");
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%.3f", opt_params.init_scaling);
                }

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Random Init:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::Checkbox("##random", &opt_params.random);
                } else {
                    ImGui::Text("%s", opt_params.random ? "Yes" : "No");
                }

                if (opt_params.random) {
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("  Num Points:");
                    ImGui::TableNextColumn();
                    if (can_edit) {
                        ImGui::PushItemWidth(-1);
                        if (ImGui::InputInt("##init_num_pts", &opt_params.init_num_pts, 10000, 50000)) {
                            opt_params.init_num_pts = std::max(1, opt_params.init_num_pts);
                        }
                        ImGui::PopItemWidth();
                    } else {
                        ImGui::Text("%d", opt_params.init_num_pts);
                    }

                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("  Extent:");
                    ImGui::TableNextColumn();
                    if (can_edit) {
                        ImGui::PushItemWidth(-1);
                        if (ImGui::InputFloat("##init_extent", &opt_params.init_extent, 0.5f, 1.0f, "%.1f")) {
                            opt_params.init_extent = std::max(0.1f, opt_params.init_extent);
                        }
                        ImGui::PopItemWidth();
                    } else {
                        ImGui::Text("%.1f", opt_params.init_extent);
                    }
                }

                ImGui::EndTable();
            }
            ImGui::TreePop();
        }

        // Pruning/Growing Thresholds (for default strategy)
        if (opt_params.strategy == "default" && ImGui::TreeNode("Pruning/Growing")) {
            if (ImGui::BeginTable("PruneTable", 2, ImGuiTableFlags_SizingStretchProp)) {
                ImGui::TableSetupColumn("Property", ImGuiTableColumnFlags_WidthFixed, 140.0f);
                ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Min Opacity:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    ImGui::InputFloat("##min_opacity", &opt_params.min_opacity, 0.001f, 0.01f, "%.4f");
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%.4f", opt_params.min_opacity);
                }

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Prune Opacity:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    ImGui::InputFloat("##prune_opacity", &opt_params.prune_opacity, 0.001f, 0.01f, "%.4f");
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%.4f", opt_params.prune_opacity);
                }

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Grow Scale 3D:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    ImGui::InputFloat("##grow_scale3d", &opt_params.grow_scale3d, 0.001f, 0.01f, "%.4f");
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%.4f", opt_params.grow_scale3d);
                }

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Grow Scale 2D:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    ImGui::InputFloat("##grow_scale2d", &opt_params.grow_scale2d, 0.001f, 0.01f, "%.4f");
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%.4f", opt_params.grow_scale2d);
                }

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Prune Scale 3D:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    ImGui::InputFloat("##prune_scale3d", &opt_params.prune_scale3d, 0.01f, 0.05f, "%.3f");
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%.3f", opt_params.prune_scale3d);
                }

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Prune Scale 2D:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    ImGui::InputFloat("##prune_scale2d", &opt_params.prune_scale2d, 0.01f, 0.05f, "%.3f");
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%.3f", opt_params.prune_scale2d);
                }

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Pause After Reset:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    int pause_refine = static_cast<int>(opt_params.pause_refine_after_reset);
                    if (ImGui::InputInt("##pause_refine_after_reset", &pause_refine, 10, 100) && pause_refine >= 0) {
                        opt_params.pause_refine_after_reset = static_cast<size_t>(pause_refine);
                    }
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%zu", opt_params.pause_refine_after_reset);
                }

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Revised Opacity:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::Checkbox("##revised_opacity", &opt_params.revised_opacity);
                } else {
                    ImGui::Text("%s", opt_params.revised_opacity ? "Yes" : "No");
                }

                ImGui::EndTable();
            }
            ImGui::TreePop();
        }

        // Sparsity Settings
        if (opt_params.enable_sparsity && ImGui::TreeNode("Sparsity Settings")) {
            if (ImGui::BeginTable("SparsityTable", 2, ImGuiTableFlags_SizingStretchProp)) {
                ImGui::TableSetupColumn("Property", ImGuiTableColumnFlags_WidthFixed, 140.0f);
                ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Sparsify Steps:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    const int prev = opt_params.sparsify_steps;
                    if (ImGui::InputInt("##sparsify_steps", &opt_params.sparsify_steps, 1000, 5000)) {
                        opt_params.sparsify_steps = std::max(1, opt_params.sparsify_steps);
                        const int delta = opt_params.sparsify_steps - prev;
                        if (delta > 0) {
                            opt_params.iterations += static_cast<size_t>(delta);
                        } else if (opt_params.iterations > static_cast<size_t>(-delta)) {
                            opt_params.iterations -= static_cast<size_t>(-delta);
                        }
                    }
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%d", opt_params.sparsify_steps);
                }

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Init Rho:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    ImGui::InputFloat("##init_rho", &opt_params.init_rho, 0.0001f, 0.001f, "%.5f");
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%.5f", opt_params.init_rho);
                }

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Prune Ratio:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    ImGui::SliderFloat("##prune_ratio", &opt_params.prune_ratio, 0.0f, 1.0f, "%.2f");
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%.2f", opt_params.prune_ratio);
                }

                ImGui::EndTable();
            }
            ImGui::TreePop();
        }
    }

    void DrawTrainingParameters(const UIContext& ctx) {
        auto* const trainer_manager = ctx.viewer->getTrainerManager();
        if (!trainer_manager || !trainer_manager->hasTrainer()) {
            return;
        }

        auto* const param_manager = services().paramsOrNull();
        if (!param_manager) {
            ImGui::TextColored(ImVec4(1, 0, 0, 1), "ParameterManager not available");
            return;
        }

        if (const auto result = param_manager->ensureLoaded(); !result) {
            ImGui::TextColored(ImVec4(1, 0.5f, 0, 1), "Failed to load params: %s", result.error().c_str());
            return;
        }

        const auto trainer_state = trainer_manager->getState();
        const int current_iteration = trainer_manager->getCurrentIteration();
        const bool can_edit = (trainer_state == TrainerManager::State::Ready) && (current_iteration == 0);

        auto& opt_params = param_manager->getActiveParams();

        lfs::core::param::DatasetConfig dataset_params;
        if (can_edit) {
            dataset_params = trainer_manager->getEditableDatasetParams();
        } else {
            const auto* const trainer = trainer_manager->getTrainer();
            if (!trainer)
                return;
            dataset_params = trainer->getParams().dataset;
        }

        bool dataset_params_changed = false;

        bool has_masks = false;
        if (!dataset_params.data_path.empty()) {
            static constexpr std::array<const char*, 3> MASK_FOLDERS = {"masks", "mask", "segmentation"};
            for (const auto* const folder : MASK_FOLDERS) {
                if (std::filesystem::exists(dataset_params.data_path / folder)) {
                    has_masks = true;
                    break;
                }
            }
        }

        ImGui::PushStyleVar(ImGuiStyleVar_IndentSpacing, 12.0f);
        if (ImGui::BeginTable("DatasetTable", 2, ImGuiTableFlags_SizingStretchProp)) {
            ImGui::TableSetupColumn("Property", ImGuiTableColumnFlags_WidthFixed, 120.0f);
            ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);

            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("Strategy:");
            ImGui::TableNextColumn();
            if (can_edit) {
                ImGui::PushItemWidth(-1);
                static constexpr const char* const STRATEGY_LABELS[] = {"MCMC", "Default"};
                int current_strategy = (opt_params.strategy == "mcmc") ? 0 : 1;
                if (ImGui::Combo("##strategy", &current_strategy, STRATEGY_LABELS, 2)) {
                    const auto new_strategy = (current_strategy == 0) ? "mcmc" : "default";
                    if (new_strategy != opt_params.strategy) {
                        param_manager->setActiveStrategy(new_strategy);
                    }
                }
                ImGui::PopItemWidth();
            } else {
                ImGui::Text("%s", opt_params.strategy == "mcmc" ? "MCMC" : "Default");
            }

            // Iterations
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
                    }
                }
                ImGui::PopItemWidth();
            } else {
                ImGui::Text("%zu", opt_params.iterations);
            }

            // Max Gaussians
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("Max Gaussians:");
            ImGui::TableNextColumn();
            if (can_edit) {
                ImGui::PushItemWidth(-1);
                if (ImGui::InputInt("##max_cap", &opt_params.max_cap, 10000, 100000)) {
                    opt_params.max_cap = std::max(1, opt_params.max_cap);
                }
                ImGui::PopItemWidth();
            } else {
                ImGui::Text("%d", opt_params.max_cap);
            }

            // SH Degree
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("SH Degree:");
            ImGui::TableNextColumn();
            if (can_edit) {
                ImGui::PushItemWidth(-1);
                static constexpr const char* const SH_DEGREE_LABELS[] = {"0", "1", "2", "3"};
                ImGui::Combo("##sh_degree", &opt_params.sh_degree, SH_DEGREE_LABELS, 4);
                ImGui::PopItemWidth();
            } else {
                ImGui::Text("%d", opt_params.sh_degree);
            }

            // Tile Mode
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("Tile Mode:");
            ImGui::TableNextColumn();
            if (can_edit) {
                ImGui::PushItemWidth(-1);
                static constexpr int TILE_OPTIONS[] = {1, 2, 4};
                static constexpr const char* const TILE_LABELS[] = {"1 (Full)", "2 (Half)", "4 (Quarter)"};
                int current_tile_index = 0;
                for (int i = 0; i < 3; ++i) {
                    if (opt_params.tile_mode == TILE_OPTIONS[i])
                        current_tile_index = i;
                }
                if (ImGui::Combo("##tile_mode", &current_tile_index, TILE_LABELS, 3)) {
                    opt_params.tile_mode = TILE_OPTIONS[current_tile_index];
                }
                ImGui::PopItemWidth();
            } else {
                ImGui::Text("%d", opt_params.tile_mode);
            }

            // Num Workers
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("Num Workers:");
            ImGui::TableNextColumn();
            if (can_edit) {
                ImGui::PushItemWidth(-1);
                if (ImGui::InputInt("##num_workers", &opt_params.num_workers, 1, 4)) {
                    opt_params.num_workers = std::clamp(opt_params.num_workers, 1, 64);
                }
                ImGui::PopItemWidth();
            } else {
                ImGui::Text("%d", opt_params.num_workers);
            }

            // Steps Scaler
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("Steps Scaler:");
            ImGui::TableNextColumn();
            if (can_edit) {
                const float prev = opt_params.steps_scaler;
                ImGui::PushItemWidth(-1);
                if (ImGui::InputFloat("##steps_scaler", &opt_params.steps_scaler, 0.1f, 0.5f, "%.2f")) {
                    opt_params.steps_scaler = std::max(0.0f, opt_params.steps_scaler);
                    if (opt_params.steps_scaler > 0.0f) {
                        const float ratio = (prev > 0.0f) ? (opt_params.steps_scaler / prev) : opt_params.steps_scaler;
                        apply_step_scaling(opt_params, ratio);
                    }
                }
                ImGui::PopItemWidth();
            } else {
                ImGui::Text("%.2f", opt_params.steps_scaler);
            }

            // Bilateral Grid Enable
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("Bilateral Grid:");
            ImGui::TableNextColumn();
            if (can_edit) {
                ImGui::Checkbox("##use_bilateral_grid", &opt_params.use_bilateral_grid);
            } else {
                ImGui::Text("%s", opt_params.use_bilateral_grid ? "Enabled" : "Disabled");
            }

            // Mask Mode
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("Mask Mode:");
            ImGui::TableNextColumn();
            static constexpr const char* const MASK_MODE_LABELS[] = {"None", "Segment", "Ignore", "Alpha Consistent"};
            if (can_edit && has_masks) {
                ImGui::PushItemWidth(-1);
                int current_mask_mode = static_cast<int>(opt_params.mask_mode);
                if (ImGui::Combo("##mask_mode", &current_mask_mode, MASK_MODE_LABELS, IM_ARRAYSIZE(MASK_MODE_LABELS))) {
                    opt_params.mask_mode = static_cast<lfs::core::param::MaskMode>(current_mask_mode);
                }
                ImGui::PopItemWidth();
            } else {
                ImGui::Text("%s", MASK_MODE_LABELS[static_cast<int>(opt_params.mask_mode)]);
                if (!has_masks && can_edit) {
                    ImGui::SameLine();
                    ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "(no masks)");
                }
            }

            if (opt_params.mask_mode != lfs::core::param::MaskMode::None && has_masks) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("  Invert Masks:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::Checkbox("##invert_masks", &opt_params.invert_masks);
                } else {
                    ImGui::Text("%s", opt_params.invert_masks ? "Yes" : "No");
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Swap object/background in masks");
                }
            }

            if (opt_params.mask_mode == lfs::core::param::MaskMode::Segment) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("  Opacity Penalty Weight:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    ImGui::InputFloat("##mask_penalty_weight", &opt_params.mask_opacity_penalty_weight, 0.1f, 1.0f, "%.1f");
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%.1f", opt_params.mask_opacity_penalty_weight);
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Weight for opacity penalty in background regions");
                }

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("  Opacity Penalty Power:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    ImGui::SliderFloat("##mask_penalty_power", &opt_params.mask_opacity_penalty_power, 0.5f, 4.0f, "%.1f");
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%.1f", opt_params.mask_opacity_penalty_power);
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Penalty falloff: 1=linear, 2=quadratic, >2=gentler");
                }

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("  Mask Threshold:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    ImGui::SliderFloat("##mask_threshold", &opt_params.mask_threshold, 0.0f, 1.0f, "%.2f");
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%.2f", opt_params.mask_threshold);
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Values >= threshold become 1.0 (object)");
                }
            }

            // Enable Sparsity
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("Sparsity:");
            ImGui::TableNextColumn();
            if (can_edit) {
                if (ImGui::Checkbox("##enable_sparsity", &opt_params.enable_sparsity)) {
                    const auto steps = static_cast<size_t>(opt_params.sparsify_steps);
                    if (opt_params.enable_sparsity) {
                        opt_params.iterations += steps;
                    } else if (opt_params.iterations > steps) {
                        opt_params.iterations -= steps;
                    }
                }
            } else {
                ImGui::Text("%s", opt_params.enable_sparsity ? "Enabled" : "Disabled");
            }

            // GUT
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("GUT:");
            ImGui::TableNextColumn();
            if (can_edit) {
                ImGui::Checkbox("##gut", &opt_params.gut);
            } else {
                ImGui::Text("%s", opt_params.gut ? "Enabled" : "Disabled");
            }

            // Mip Filter (anti-aliasing)
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("Mip Filter:");
            ImGui::TableNextColumn();
            if (can_edit) {
                ImGui::Checkbox("##mip_filter", &opt_params.mip_filter);
            } else {
                ImGui::Text("%s", opt_params.mip_filter ? "Enabled" : "Disabled");
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Anti-aliasing filter for training (mip-splatting)");
            }

            // BG Modulation
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("BG Modulation:");
            ImGui::TableNextColumn();
            if (can_edit) {
                ImGui::Checkbox("##bg_modulation", &opt_params.bg_modulation);
            } else {
                ImGui::Text("%s", opt_params.bg_modulation ? "Enabled" : "Disabled");
            }

            // Evaluation
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("Evaluation:");
            ImGui::TableNextColumn();
            if (can_edit) {
                ImGui::Checkbox("##enable_eval", &opt_params.enable_eval);
            } else {
                ImGui::Text("%s", opt_params.enable_eval ? "Enabled" : "Disabled");
            }
        }
        ImGui::EndTable();

        // Advanced Training Parameters section
        ImGui::Spacing();
        if (can_edit && dataset_params_changed) {
            trainer_manager->getEditableDatasetParams() = dataset_params;
        }

        ImGui::PopStyleVar();
    }

    void DrawTrainingParams(const UIContext& ctx) {

        auto& state = TrainingPanelState::getInstance();
        const auto& t = theme();

        if (ImGui::CollapsingHeader("Basic Training Params", ImGuiTreeNodeFlags_DefaultOpen)) {
            DrawTrainingParameters(ctx);
        }
        if (ImGui::CollapsingHeader("Advanced Training Params")) {
            DrawTrainingAdvancedParameters(ctx);
        }

        // Save feedback
        if (state.save_in_progress) {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                               now - state.save_start_time)
                               .count();
            if (elapsed < 2000) {
                ImGui::TextColored(t.palette.success, "Checkpoint saved!");
            } else {
                state.save_in_progress = false;
            }
        }
    }

    void DrawTrainingStatus(const UIContext& ctx) {
        auto* trainer_manager = ctx.viewer->getTrainerManager();
        if (!trainer_manager || !trainer_manager->hasTrainer()) {
            ImGui::TextColored(darken(theme().palette.text_dim, 0.15f), "No trainer loaded");
            return;
        }

        auto trainer_state = trainer_manager->getState();
        int current_iteration = trainer_manager->getCurrentIteration();

        // Status display
        ImGui::Separator();

        const char* state_str = "Unknown";
        switch (trainer_state) {
        case TrainerManager::State::Idle: state_str = "Idle"; break;
        case TrainerManager::State::Ready: state_str = (current_iteration > 0) ? "Resume" : "Ready"; break;
        case TrainerManager::State::Running: state_str = "Running"; break;
        case TrainerManager::State::Paused: state_str = "Paused"; break;
        case TrainerManager::State::Stopping: state_str = "Stopping"; break;
        case TrainerManager::State::Finished: {
            const auto reason = trainer_manager->getStateMachine().getFinishReason();
            switch (reason) {
            case FinishReason::Completed: state_str = "Completed"; break;
            case FinishReason::UserStopped: state_str = "Stopped"; break;
            case FinishReason::Error: state_str = "Error"; break;
            default: state_str = "Finished";
            }
            break;
        }
        }

        static IterationRateTracker g_iter_rate_tracker;

        ImGui::Text("Status: %s", state_str);
        g_iter_rate_tracker.addSample(current_iteration);
        float iters_per_sec = g_iter_rate_tracker.getIterationsPerSecond();
        iters_per_sec = iters_per_sec > 0.0f ? iters_per_sec : 0.0f;

        ImGui::Text("Iteration: %d (%.1f iters/sec)", current_iteration, iters_per_sec);

        int num_splats = trainer_manager->getNumSplats();
        ImGui::Text("num Splats: %d", num_splats);
    }

    void DrawTrainingControls(const UIContext& ctx) {
        ImGui::Separator();

        auto& state = TrainingPanelState::getInstance();

        auto* trainer_manager = ctx.viewer->getTrainerManager();
        if (!trainer_manager || !trainer_manager->hasTrainer()) {
            ImGui::TextColored(darken(theme().palette.text_dim, 0.15f), "No trainer loaded");
            return;
        }

        auto trainer_state = trainer_manager->getState();
        int current_iteration = trainer_manager->getCurrentIteration();

        using widgets::ButtonStyle;
        using widgets::ColoredButton;

        const auto& t = theme();
        constexpr ImVec2 FULL_WIDTH = {-1, 0};

        switch (trainer_state) {
        case TrainerManager::State::Idle:
            ImGui::TextColored(darken(t.palette.text_dim, 0.15f), "No trainer loaded");
            break;

        case TrainerManager::State::Ready: {
            const char* const label = current_iteration > 0 ? "Resume Training" : "Start Training";
            if (ColoredButton(label, ButtonStyle::Success, FULL_WIDTH)) {
                lfs::core::events::cmd::StartTraining{}.emit();
            }
            if (current_iteration > 0) {
                if (ColoredButton("Reset Training", ButtonStyle::Secondary, FULL_WIDTH)) {
                    lfs::core::events::cmd::ResetTraining{}.emit();
                }
            }
            if (ColoredButton("Clear", ButtonStyle::Error, FULL_WIDTH)) {
                lfs::core::events::cmd::ClearScene{}.emit();
            }
            break;
        }

        case TrainerManager::State::Running:
            if (ColoredButton("Pause", ButtonStyle::Warning, FULL_WIDTH)) {
                lfs::core::events::cmd::PauseTraining{}.emit();
            }
            break;

        case TrainerManager::State::Paused:
            if (ColoredButton("Resume", ButtonStyle::Success, FULL_WIDTH)) {
                lfs::core::events::cmd::ResumeTraining{}.emit();
            }
            if (ColoredButton("Reset Training", ButtonStyle::Secondary, FULL_WIDTH)) {
                lfs::core::events::cmd::ResetTraining{}.emit();
            }
            if (ColoredButton("Stop Training", ButtonStyle::Error, FULL_WIDTH)) {
                lfs::core::events::cmd::StopTraining{}.emit();
            }
            break;

        case TrainerManager::State::Finished: {
            const auto reason = trainer_manager->getStateMachine().getFinishReason();
            switch (reason) {
            case FinishReason::Completed:
                ImGui::TextColored(t.palette.success, "Training Complete!");
                break;
            case FinishReason::UserStopped:
                ImGui::TextColored(t.palette.text_dim, "Training Stopped");
                break;
            case FinishReason::Error:
                ImGui::TextColored(t.palette.error, "Training Error!");
                if (const auto error_msg = trainer_manager->getLastError(); !error_msg.empty()) {
                    ImGui::TextWrapped("%s", error_msg.c_str());
                }
                break;
            default:
                ImGui::TextColored(t.palette.text_dim, "Training Finished");
            }

            if (reason == FinishReason::Completed || reason == FinishReason::UserStopped) {
                if (ColoredButton("Switch to Edit Mode", ButtonStyle::Success, FULL_WIDTH)) {
                    lfs::core::events::cmd::SwitchToEditMode{}.emit();
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Keep trained model, discard dataset");
                }
            }
            if (ColoredButton("Reset Training", ButtonStyle::Secondary, FULL_WIDTH)) {
                lfs::core::events::cmd::ResetTraining{}.emit();
            }

            if (ColoredButton("Clear", ButtonStyle::Error, FULL_WIDTH)) {
                lfs::core::events::cmd::ClearScene{}.emit();
            }

            break;
        }

        case TrainerManager::State::Stopping:
            ImGui::TextColored(t.palette.text_dim, "Stopping...");
            break;
        }

        // Save checkpoint button
        if (trainer_state == TrainerManager::State::Running ||
            trainer_state == TrainerManager::State::Paused) {
            if (ColoredButton("Save Checkpoint", ButtonStyle::Primary, FULL_WIDTH)) {
                lfs::core::events::cmd::SaveCheckpoint{}.emit();
                state.save_in_progress = true;
                state.save_start_time = std::chrono::steady_clock::now();
            }
        }

        ImGui::Separator();

        // Save feedback
        if (state.save_in_progress) {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                               now - state.save_start_time)
                               .count();
            if (elapsed < 2000) {
                ImGui::TextColored(t.palette.success, "Checkpoint saved!");
            } else {
                state.save_in_progress = false;
            }
        }
    }

} // namespace lfs::vis::gui::panels
