/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/panels/training_panel.hpp"
#include "core/events.hpp"
#include "core/logger.hpp"
#include "core/parameter_manager.hpp"
#include "core/parameters.hpp"
#include "core/path_utils.hpp"
#include "core/services.hpp"
#include "gui/dpi_scale.hpp"
#include "gui/localization_manager.hpp"
#include "gui/string_keys.hpp"
#include "gui/ui_widgets.hpp"
#include "gui/utils/windows_utils.hpp"
#include "theme/theme.hpp"
#include "visualizer_impl.hpp"

#include <chrono>
#include <cmath>
#include <cstring>
#include <deque>
#include <filesystem>
#include <set>
#include <imgui.h>

#ifdef _WIN32
#include <Windows.h>
#endif

namespace lfs::vis::gui::panels {

    using namespace lichtfeld::Strings;

    namespace {
        constexpr float RATE_WINDOW_SECONDS = 5.0f;

        void scale_steps_vector(std::vector<size_t>& steps, const float scaler) {
            std::set<size_t> unique;
            for (const auto s : steps) {
                if (const auto scaled = static_cast<size_t>(std::lround(static_cast<float>(s) * scaler)); scaled > 0) {
                    unique.insert(scaled);
                }
            }
            steps.assign(unique.begin(), unique.end());
        }

        void apply_step_scaling(lfs::core::param::OptimizationParameters& opt, const float scaler) {
            if (scaler <= 0.0f)
                return;
            const auto scale = [scaler](const size_t v) {
                return static_cast<size_t>(std::lround(static_cast<float>(v) * scaler));
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
            ImGui::TextColored(ImVec4(1, 0, 0, 1), "%s", LOC(Messages::PARAM_MANAGER_UNAVAILABLE));
            return;
        }

        if (const auto result = param_manager->ensureLoaded(); !result) {
            ImGui::TextColored(ImVec4(1, 0.5f, 0, 1), "%s %s", LOC(Messages::FAILED_TO_LOAD_PARAMS), result.error().c_str());
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
        if (ImGui::TreeNode(LOC(Training::Section::DATASET))) {
            if (ImGui::BeginTable("DatasetTable", 2, ImGuiTableFlags_SizingStretchProp)) {
                ImGui::TableSetupColumn("Property", ImGuiTableColumnFlags_WidthFixed, 120.0f);
                ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("%s", LOC(Training::Dataset::PATH));
                ImGui::TableNextColumn();
                ImGui::Text("%s", lfs::core::path_to_utf8(dataset_params.data_path.filename()).c_str());

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("%s", LOC(Training::Dataset::IMAGES));
                ImGui::TableNextColumn();
                ImGui::Text("%s", dataset_params.images.c_str());

                // Resize Factor
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("%s", LOC(Training::Dataset::RESIZE_FACTOR));
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
                if (ImGui::IsItemHovered()) {
                    widgets::SetThemedTooltip("%s", LOC(Training::Tooltip::RESIZE_FACTOR));
                }

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("%s", LOC(Training::Dataset::MAX_WIDTH));
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
                if (ImGui::IsItemHovered()) {
                    widgets::SetThemedTooltip("%s", LOC(Training::Tooltip::MAX_WIDTH));
                }

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("%s", LOC(Training::Dataset::CPU_CACHE));
                ImGui::TableNextColumn();
                if (can_edit) {
                    if (ImGui::Checkbox("##use_cpu_cache", &dataset_params.loading_params.use_cpu_memory)) {
                        dataset_params_changed = true;
                    }
                } else {
                    ImGui::Text("%s", dataset_params.loading_params.use_cpu_memory ? LOC(Training::Status::ENABLED) : LOC(Training::Status::DISABLED));
                }
                if (ImGui::IsItemHovered()) {
                    widgets::SetThemedTooltip("%s", LOC(Training::Tooltip::CPU_CACHE));
                }

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("%s", LOC(Training::Dataset::FS_CACHE));
                ImGui::TableNextColumn();
                if (can_edit) {
                    if (ImGui::Checkbox("##use_fs_cache", &dataset_params.loading_params.use_fs_cache)) {
                        dataset_params_changed = true;
                    }
                } else {
                    ImGui::Text("%s", dataset_params.loading_params.use_fs_cache ? LOC(Training::Status::ENABLED) : LOC(Training::Status::DISABLED));
                }
                if (ImGui::IsItemHovered()) {
                    widgets::SetThemedTooltip("%s", LOC(Training::Tooltip::FS_CACHE));
                }

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("%s", LOC(Training::Dataset::OUTPUT));
                ImGui::TableNextColumn();
                {
                    const std::string output_display = dataset_params.output_path.empty()
                                                           ? "(not set)"
                                                           : lfs::core::path_to_utf8(dataset_params.output_path.filename());
                    ImGui::Text("%s", output_display.c_str());
                    if (!dataset_params.output_path.empty() && ImGui::IsItemHovered()) {
                        widgets::SetThemedTooltip("%s", lfs::core::path_to_utf8(dataset_params.output_path).c_str());
                    }
                }

                ImGui::EndTable();
            }
            ImGui::TreePop();
        }

        // Optimization Parameters
        if (ImGui::TreeNode(LOC(Training::Section::OPTIMIZATION))) {
            if (ImGui::BeginTable("OptimizationTable", 2, ImGuiTableFlags_SizingStretchProp)) {
                ImGui::TableSetupColumn("Property", ImGuiTableColumnFlags_WidthFixed, 120.0f);
                ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("%s", LOC(Training::Opt::STRATEGY));
                ImGui::TableNextColumn();
                ImGui::Text("%s", opt_params.strategy.c_str());

                // Learning Rates section
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextColored(theme().palette.text_dim, "%s", LOC(Training::Opt::LEARNING_RATES));
                ImGui::TableNextColumn();

                // Position LR
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("%s", LOC(Training::Opt::LR_POSITION));
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    ImGui::InputFloat("##means_lr", &opt_params.means_lr, 0.000001f, 0.00001f, "%.6f");
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%.6f", opt_params.means_lr);
                }
                if (ImGui::IsItemHovered()) {
                    widgets::SetThemedTooltip("%s", LOC(Training::Tooltip::LR_POSITION));
                }

                // SH Coeff LR
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("%s", LOC(Training::Opt::LR_SH_COEFF));
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    ImGui::InputFloat("##shs_lr", &opt_params.shs_lr, 0.0001f, 0.001f, "%.4f");
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%.4f", opt_params.shs_lr);
                }
                if (ImGui::IsItemHovered()) {
                    widgets::SetThemedTooltip("%s", LOC(Training::Tooltip::LR_SH_COEFF));
                }

                // Opacity LR
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("%s", LOC(Training::Opt::LR_OPACITY));
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    ImGui::InputFloat("##opacity_lr", &opt_params.opacity_lr, 0.001f, 0.01f, "%.4f");
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%.4f", opt_params.opacity_lr);
                }
                if (ImGui::IsItemHovered()) {
                    widgets::SetThemedTooltip("%s", LOC(Training::Tooltip::LR_OPACITY));
                }

                // Scaling LR
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("%s", LOC(Training::Opt::LR_SCALING));
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    ImGui::InputFloat("##scaling_lr", &opt_params.scaling_lr, 0.0001f, 0.001f, "%.4f");
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%.4f", opt_params.scaling_lr);
                }
                if (ImGui::IsItemHovered()) {
                    widgets::SetThemedTooltip("%s", LOC(Training::Tooltip::LR_SCALING));
                }

                // Rotation LR
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("%s", LOC(Training::Opt::LR_ROTATION));
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    ImGui::InputFloat("##rotation_lr", &opt_params.rotation_lr, 0.0001f, 0.001f, "%.4f");
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%.4f", opt_params.rotation_lr);
                }
                if (ImGui::IsItemHovered()) {
                    widgets::SetThemedTooltip("%s", LOC(Training::Tooltip::LR_ROTATION));
                }

                // Refinement section
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextColored(theme().palette.text_dim, "%s", LOC(Training::Section::REFINEMENT));
                ImGui::TableNextColumn();

                // Refine Every
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("%s", LOC(Training::Refinement::REFINE_EVERY));
                ImGui::TableNextColumn();
                ImGui::PushItemWidth(-1);
                int refine_every = static_cast<int>(opt_params.refine_every);
                if (can_edit) {
                    if (widgets::InputIntFormatted("##refine_every", &refine_every, 10, 100) && refine_every > 0) {
                        opt_params.refine_every = static_cast<size_t>(refine_every);
                    }
                } else {
                    ImGui::Text("%s", widgets::formatNumber(refine_every).c_str());
                }
                ImGui::PopItemWidth();
                if (ImGui::IsItemHovered()) {
                    widgets::SetThemedTooltip("%s", LOC(Training::Tooltip::REFINE_EVERY));
                }

                // Start Refine
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("%s", LOC(Training::Refinement::START_REFINE));
                ImGui::TableNextColumn();
                ImGui::PushItemWidth(-1);
                int start_refine = static_cast<int>(opt_params.start_refine);
                if (can_edit) {
                    if (widgets::InputIntFormatted("##start_refine", &start_refine, 100, 500) && start_refine >= 0) {
                        opt_params.start_refine = static_cast<size_t>(start_refine);
                    }
                } else {
                    ImGui::Text("%s", widgets::formatNumber(start_refine).c_str());
                }
                ImGui::PopItemWidth();
                if (ImGui::IsItemHovered()) {
                    widgets::SetThemedTooltip("%s", LOC(Training::Tooltip::START_REFINE));
                }

                // Stop Refine
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("%s", LOC(Training::Refinement::STOP_REFINE));
                ImGui::TableNextColumn();
                ImGui::PushItemWidth(-1);
                int stop_refine = static_cast<int>(opt_params.stop_refine);
                if (can_edit) {
                    if (widgets::InputIntFormatted("##stop_refine", &stop_refine, 1000, 5000) && stop_refine >= 0) {
                        opt_params.stop_refine = static_cast<size_t>(stop_refine);
                    }
                } else {
                    ImGui::Text("%s", widgets::formatNumber(stop_refine).c_str());
                }
                ImGui::PopItemWidth();
                if (ImGui::IsItemHovered()) {
                    widgets::SetThemedTooltip("%s", LOC(Training::Tooltip::STOP_REFINE));
                }

                // Gradient Threshold
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("%s", LOC(Training::Refinement::GRADIENT_THR));
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    ImGui::InputFloat("##grad_threshold", &opt_params.grad_threshold, 0.000001f, 0.00001f, "%.6f");
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%.6f", opt_params.grad_threshold);
                }
                if (ImGui::IsItemHovered()) {
                    widgets::SetThemedTooltip("%s", LOC(Training::Tooltip::GRADIENT_THR));
                }

                // Reset Every
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("%s", LOC(Training::Refinement::RESET_EVERY));
                ImGui::TableNextColumn();
                ImGui::PushItemWidth(-1);
                int reset_every = static_cast<int>(opt_params.reset_every);
                if (can_edit) {
                    if (widgets::InputIntFormatted("##reset_every", &reset_every, 100, 1000) && reset_every >= 0) {
                        opt_params.reset_every = static_cast<size_t>(reset_every);
                    }
                } else if (reset_every > 0) {
                    ImGui::Text("%s", widgets::formatNumber(reset_every).c_str());
                } else {
                    ImGui::Text("%s", LOC(TrainingParams::DISABLED));
                }
                ImGui::PopItemWidth();
                if (ImGui::IsItemHovered()) {
                    widgets::SetThemedTooltip("%s", LOC(Training::Tooltip::RESET_EVERY));
                }

                // SH Degree Interval
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("%s", LOC(Training::Refinement::SH_UPGRADE_EVERY));
                ImGui::TableNextColumn();
                ImGui::PushItemWidth(-1);
                int sh_deg_interval = static_cast<int>(opt_params.sh_degree_interval);
                if (can_edit) {
                    if (widgets::InputIntFormatted("##sh_degree_interval", &sh_deg_interval, 100, 500) && sh_deg_interval > 0) {
                        opt_params.sh_degree_interval = static_cast<size_t>(sh_deg_interval);
                    }
                } else {
                    ImGui::Text("%s", widgets::formatNumber(sh_deg_interval).c_str());
                }
                ImGui::PopItemWidth();
                if (ImGui::IsItemHovered()) {
                    widgets::SetThemedTooltip("%s", LOC(Training::Tooltip::SH_UPGRADE_EVERY));
                }

                ImGui::EndTable();
            }
            ImGui::TreePop();
        }

        // Save Steps
        bool save_steps_open = ImGui::TreeNode(LOC(Training::Section::SAVE_STEPS));
        if (ImGui::IsItemHovered()) {
            widgets::SetThemedTooltip("%s", LOC(Training::Tooltip::SAVE_STEPS));
        }
        if (save_steps_open) {
            if (can_edit) {
                static int new_step = 1000;
                ImGui::InputInt("##new_step", &new_step, 100, 1000);
                if (ImGui::IsItemHovered()) {
                    widgets::SetThemedTooltip("%s", LOC(Training::Tooltip::SAVE_STEP_INPUT));
                }
                ImGui::SameLine();
                if (ImGui::Button(LOC(Training::Button::ADD))) {
                    if (new_step > 0 && std::find(opt_params.save_steps.begin(),
                                                  opt_params.save_steps.end(),
                                                  new_step) == opt_params.save_steps.end()) {
                        opt_params.save_steps.push_back(new_step);
                        std::sort(opt_params.save_steps.begin(), opt_params.save_steps.end());
                    }
                }
                if (ImGui::IsItemHovered()) {
                    widgets::SetThemedTooltip("%s", LOC(Training::Tooltip::SAVE_STEP_ADD));
                }

                ImGui::Separator();

                for (size_t i = 0; i < opt_params.save_steps.size(); ++i) {
                    ImGui::PushID(static_cast<int>(i));

                    int step = static_cast<int>(opt_params.save_steps[i]);
                    ImGui::SetNextItemWidth(100 * getDpiScale());
                    if (ImGui::InputInt("##step", &step, 0, 0)) {
                        if (step > 0) {
                            opt_params.save_steps[i] = static_cast<size_t>(step);
                            std::sort(opt_params.save_steps.begin(), opt_params.save_steps.end());
                        }
                    }

                    ImGui::SameLine();
                    if (ImGui::Button(LOC(Training::Button::REMOVE))) {
                        opt_params.save_steps.erase(opt_params.save_steps.begin() + i);
                    }
                    if (ImGui::IsItemHovered()) {
                        widgets::SetThemedTooltip("%s", LOC(Training::Tooltip::SAVE_STEP_REMOVE));
                    }

                    ImGui::PopID();
                }

                if (opt_params.save_steps.empty()) {
                    ImGui::TextColored(darken(theme().palette.text_dim, 0.15f), "%s", LOC(TrainingPanel::NO_SAVE_STEPS));
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
        if (opt_params.use_bilateral_grid && ImGui::TreeNode(LOC(Training::Section::BILATERAL_GRID))) {
            if (ImGui::BeginTable("BilateralTable", 2, ImGuiTableFlags_SizingStretchProp)) {
                ImGui::TableSetupColumn("Property", ImGuiTableColumnFlags_WidthFixed, 140.0f);
                ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("%s", LOC(Training::Bilateral::GRID_X));
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
                ImGui::Text("%s", LOC(Training::Bilateral::GRID_Y));
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
                ImGui::Text("%s", LOC(Training::Bilateral::GRID_W));
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
                ImGui::Text("%s", LOC(Training::Bilateral::LEARNING_RATE));
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

        // PPISP Settings
        if (opt_params.use_ppisp && ImGui::TreeNode("PPISP Settings")) {
            if (ImGui::BeginTable("PPISPTable", 2, ImGuiTableFlags_SizingStretchProp)) {
                ImGui::TableSetupColumn("Property", ImGuiTableColumnFlags_WidthFixed, 140.0f);
                ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Learning Rate");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    ImGui::InputFloat("##ppisp_lr", &opt_params.ppisp_lr, 0.0001f, 0.001f, "%.5f");
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%.5f", opt_params.ppisp_lr);
                }

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Regularization");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    ImGui::InputFloat("##ppisp_reg_weight", &opt_params.ppisp_reg_weight, 0.0001f, 0.001f, "%.5f");
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%.5f", opt_params.ppisp_reg_weight);
                }

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Warmup Steps");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    ImGui::InputInt("##ppisp_warmup", &opt_params.ppisp_warmup_steps, 100, 500);
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%d", opt_params.ppisp_warmup_steps);
                }

                // Controller settings
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextColored(theme().palette.text_dim, "Controller");
                ImGui::TableNextColumn();

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Enable Controller");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::Checkbox("##ppisp_use_controller", &opt_params.ppisp_use_controller);
                } else {
                    ImGui::Text("%s", opt_params.ppisp_use_controller ? "Enabled" : "Disabled");
                }
                if (ImGui::IsItemHovered()) {
                    widgets::SetThemedTooltip("Enable PPISP controller for novel view synthesis (Phase 2 distillation)");
                }

                if (opt_params.ppisp_use_controller) {
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("Activation Step");
                    ImGui::TableNextColumn();
                    if (can_edit) {
                        bool is_auto = opt_params.ppisp_controller_activation_step < 0;
                        if (ImGui::Checkbox("Auto##ppisp_auto_step", &is_auto)) {
                            opt_params.ppisp_controller_activation_step = is_auto ? -1 : static_cast<int>(opt_params.iterations) - 5000;
                        }
                        if (!is_auto) {
                            ImGui::SameLine();
                            ImGui::PushItemWidth(-1);
                            if (widgets::InputIntFormatted("##ppisp_controller_step", &opt_params.ppisp_controller_activation_step, 1000, 5000)) {
                                opt_params.ppisp_controller_activation_step = std::max(1, opt_params.ppisp_controller_activation_step);
                            }
                            ImGui::PopItemWidth();
                        }
                    } else {
                        if (opt_params.ppisp_controller_activation_step < 0) {
                            int auto_val = std::max(0, static_cast<int>(opt_params.iterations) - 5000);
                            ImGui::Text("Auto (%s)", widgets::formatNumber(auto_val).c_str());
                        } else {
                            ImGui::Text("%s", widgets::formatNumber(opt_params.ppisp_controller_activation_step).c_str());
                        }
                    }
                    if (ImGui::IsItemHovered()) {
                        widgets::SetThemedTooltip("Iteration to start Phase 2 controller distillation (-1 = auto: iterations - 5000)");
                    }

                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("Controller LR");
                    ImGui::TableNextColumn();
                    if (can_edit) {
                        ImGui::PushItemWidth(-1);
                        ImGui::InputFloat("##ppisp_controller_lr", &opt_params.ppisp_controller_lr, 0.0001f, 0.001f, "%.5f");
                        ImGui::PopItemWidth();
                    } else {
                        ImGui::Text("%.5f", opt_params.ppisp_controller_lr);
                    }
                    if (ImGui::IsItemHovered()) {
                        widgets::SetThemedTooltip("Learning rate for PPISP controller");
                    }

                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("Freeze Gaussians");
                    ImGui::TableNextColumn();
                    if (can_edit) {
                        ImGui::Checkbox("##ppisp_freeze_gaussians", &opt_params.ppisp_freeze_gaussians_on_distill);
                    } else {
                        ImGui::Text("%s", opt_params.ppisp_freeze_gaussians_on_distill ? "Yes" : "No");
                    }
                    if (ImGui::IsItemHovered()) {
                        widgets::SetThemedTooltip("Freeze Gaussian optimization during controller distillation phase");
                    }
                }

                ImGui::EndTable();
            }
            ImGui::TreePop();
        }

        // Mask Settings
        if (opt_params.mask_mode != lfs::core::param::MaskMode::None && ImGui::TreeNode(LOC(Training::Section::MASKING))) {
            if (ImGui::BeginTable("MaskTable", 2, ImGuiTableFlags_SizingStretchProp)) {
                ImGui::TableSetupColumn("Property", ImGuiTableColumnFlags_WidthFixed, 140.0f);
                ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("%s", LOC(Training::Masking::INVERT_MASKS));
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::Checkbox("##invert_masks", &opt_params.invert_masks);
                } else {
                    ImGui::Text("%s", opt_params.invert_masks ? LOC(Training::Status::YES) : LOC(Training::Status::NO));
                }

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("%s", LOC(Training::Masking::THRESHOLD));
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
                    ImGui::Text("%s", LOC(Training::Masking::PENALTY_WEIGHT));
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
                    ImGui::Text("%s", LOC(Training::Masking::PENALTY_POWER));
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

        // Loss Parameters
        if (ImGui::TreeNode(LOC(Training::Section::LOSSES))) {
            if (ImGui::BeginTable("LossTable", 2, ImGuiTableFlags_SizingStretchProp)) {
                ImGui::TableSetupColumn("Property", ImGuiTableColumnFlags_WidthFixed, 140.0f);
                ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("%s", LOC(Training::Losses::LAMBDA_DSSIM));
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    ImGui::SliderFloat("##lambda_dssim", &opt_params.lambda_dssim, 0.0f, 1.0f, "%.2f");
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%.2f", opt_params.lambda_dssim);
                }
                if (ImGui::IsItemHovered()) {
                    widgets::SetThemedTooltip("%s", LOC(Training::Tooltip::LAMBDA_DSSIM));
                }

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("%s", LOC(Training::Losses::OPACITY_REG));
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    ImGui::InputFloat("##opacity_reg", &opt_params.opacity_reg, 0.001f, 0.01f, "%.4f");
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%.4f", opt_params.opacity_reg);
                }
                if (ImGui::IsItemHovered()) {
                    widgets::SetThemedTooltip("%s", LOC(Training::Tooltip::OPACITY_REG));
                }

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("%s", LOC(Training::Losses::SCALE_REG));
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    ImGui::InputFloat("##scale_reg", &opt_params.scale_reg, 0.001f, 0.01f, "%.4f");
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%.4f", opt_params.scale_reg);
                }
                if (ImGui::IsItemHovered()) {
                    widgets::SetThemedTooltip("%s", LOC(Training::Tooltip::SCALE_REG));
                }

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("%s", LOC(Training::Losses::TV_LOSS_WEIGHT));
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    ImGui::InputFloat("##tv_loss_weight", &opt_params.tv_loss_weight, 1.0f, 5.0f, "%.1f");
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%.1f", opt_params.tv_loss_weight);
                }
                if (ImGui::IsItemHovered()) {
                    widgets::SetThemedTooltip("%s", LOC(Training::Tooltip::TV_LOSS_WEIGHT));
                }

                ImGui::EndTable();
            }
            ImGui::TreePop();
        }

        // Initialization Parameters
        if (ImGui::TreeNode(LOC(Training::Section::INITIALIZATION))) {
            if (ImGui::BeginTable("InitTable", 2, ImGuiTableFlags_SizingStretchProp)) {
                ImGui::TableSetupColumn("Property", ImGuiTableColumnFlags_WidthFixed, 140.0f);
                ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("%s", LOC(TrainingParams::INIT_OPACITY));
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    ImGui::SliderFloat("##init_opacity", &opt_params.init_opacity, 0.01f, 1.0f, "%.2f");
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%.2f", opt_params.init_opacity);
                }
                if (ImGui::IsItemHovered()) {
                    widgets::SetThemedTooltip("%s", LOC(Training::Tooltip::INIT_OPACITY));
                }

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("%s", LOC(TrainingParams::INIT_SCALING));
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    ImGui::InputFloat("##init_scaling", &opt_params.init_scaling, 0.01f, 0.1f, "%.3f");
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%.3f", opt_params.init_scaling);
                }
                if (ImGui::IsItemHovered()) {
                    widgets::SetThemedTooltip("%s", LOC(Training::Tooltip::INIT_SCALING));
                }

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("%s", LOC(TrainingParams::RANDOM_INIT));
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::Checkbox("##random", &opt_params.random);
                } else {
                    ImGui::Text("%s", opt_params.random ? "Yes" : "No");
                }
                if (ImGui::IsItemHovered()) {
                    widgets::SetThemedTooltip("%s", LOC(Training::Tooltip::RANDOM_INIT));
                }

                if (opt_params.random) {
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("%s", LOC(TrainingParams::NUM_POINTS));
                    ImGui::TableNextColumn();
                    ImGui::PushItemWidth(-1);
                    if (can_edit) {
                        if (widgets::InputIntFormatted("##init_num_pts", &opt_params.init_num_pts, 10000, 50000)) {
                            opt_params.init_num_pts = std::max(1, opt_params.init_num_pts);
                        }
                    } else {
                        ImGui::Text("%s", widgets::formatNumber(opt_params.init_num_pts).c_str());
                    }
                    ImGui::PopItemWidth();
                    if (ImGui::IsItemHovered()) {
                        widgets::SetThemedTooltip("%s", LOC(Training::Tooltip::NUM_POINTS));
                    }

                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("%s", LOC(TrainingParams::EXTENT));
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
                    if (ImGui::IsItemHovered()) {
                        widgets::SetThemedTooltip("%s", LOC(Training::Tooltip::EXTENT));
                    }
                }

                ImGui::EndTable();
            }
            ImGui::TreePop();
        }

        // Pruning/Growing Thresholds (for adc strategy)
        if (opt_params.strategy == "adc" && ImGui::TreeNode(LOC(Training::Section::PRUNING_GROWING))) {
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
                ImGui::Text("%s", LOC(TrainingParams::PRUNE_SCALE_3D));
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
                ImGui::Text("%s", LOC(TrainingParams::PRUNE_SCALE_2D));
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
                ImGui::Text("%s", LOC(TrainingParams::PAUSE_AFTER_RESET));
                ImGui::TableNextColumn();
                ImGui::PushItemWidth(-1);
                int pause_refine = static_cast<int>(opt_params.pause_refine_after_reset);
                if (can_edit) {
                    if (widgets::InputIntFormatted("##pause_refine_after_reset", &pause_refine, 10, 100) && pause_refine >= 0) {
                        opt_params.pause_refine_after_reset = static_cast<size_t>(pause_refine);
                    }
                } else {
                    ImGui::Text("%s", widgets::formatNumber(pause_refine).c_str());
                }
                ImGui::PopItemWidth();

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("%s", LOC(TrainingParams::REVISED_OPACITY));
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
        if (opt_params.enable_sparsity && ImGui::TreeNode(LOC(Training::Section::SPARSITY))) {
            if (ImGui::BeginTable("SparsityTable", 2, ImGuiTableFlags_SizingStretchProp)) {
                ImGui::TableSetupColumn("Property", ImGuiTableColumnFlags_WidthFixed, 140.0f);
                ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("%s", LOC(TrainingParams::SPARSIFY_STEPS));
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    if (widgets::InputIntFormatted("##sparsify_steps", &opt_params.sparsify_steps, 1000, 5000)) {
                        opt_params.sparsify_steps = std::max(1, opt_params.sparsify_steps);
                    }
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%s", widgets::formatNumber(opt_params.sparsify_steps).c_str());
                }

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("%s", LOC(TrainingParams::INIT_RHO));
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
                ImGui::Text("%s", LOC(TrainingParams::PRUNE_RATIO));
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

        // Save dataset params if changed
        if (can_edit && dataset_params_changed) {
            trainer_manager->getEditableDatasetParams() = dataset_params;
        }
    }

    void DrawTrainingParameters(const UIContext& ctx) {
        auto* const trainer_manager = ctx.viewer->getTrainerManager();
        if (!trainer_manager || !trainer_manager->hasTrainer()) {
            return;
        }

        auto* const param_manager = services().paramsOrNull();
        if (!param_manager) {
            ImGui::TextColored(ImVec4(1, 0, 0, 1), "%s", LOC(Messages::PARAM_MANAGER_UNAVAILABLE));
            return;
        }

        if (const auto result = param_manager->ensureLoaded(); !result) {
            ImGui::TextColored(ImVec4(1, 0.5f, 0, 1), "%s %s", LOC(Messages::FAILED_TO_LOAD_PARAMS), result.error().c_str());
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

        const auto* scene = trainer_manager->getScene();
        const bool has_alpha = scene && scene->imagesHaveAlpha();

        ImGui::PushStyleVar(ImGuiStyleVar_IndentSpacing, 12.0f);
        if (ImGui::BeginTable("DatasetTable", 2, ImGuiTableFlags_SizingStretchProp)) {
            ImGui::TableSetupColumn("Property", ImGuiTableColumnFlags_WidthFixed, 120.0f);
            ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);

            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("%s", LOC(TrainingParams::STRATEGY));
            ImGui::TableNextColumn();
            if (can_edit) {
                ImGui::PushItemWidth(-1);
                static constexpr const char* const STRATEGY_LABELS[] = {"MCMC", "ADC"};
                int current_strategy = (opt_params.strategy == "mcmc") ? 0 : 1;
                if (ImGui::Combo("##strategy", &current_strategy, STRATEGY_LABELS, 2)) {
                    const auto new_strategy = (current_strategy == 0) ? "mcmc" : "adc";
                    if (new_strategy != opt_params.strategy) {
                        param_manager->setActiveStrategy(new_strategy);
                    }
                }
                ImGui::PopItemWidth();
            } else {
                ImGui::Text("%s", opt_params.strategy == "mcmc" ? "MCMC" : "ADC");
            }
            if (ImGui::IsItemHovered()) {
                widgets::SetThemedTooltip("%s", LOC(Training::Tooltip::STRATEGY));
            }

            // Iterations
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("%s", LOC(TrainingParams::ITERATIONS));
            ImGui::TableNextColumn();
            ImGui::PushItemWidth(-1);
            int iterations = static_cast<int>(opt_params.iterations);
            if (can_edit) {
                if (widgets::InputIntFormatted("##iterations", &iterations, 1000, 5000) && iterations > 0 && iterations <= 1000000) {
                    opt_params.iterations = static_cast<size_t>(iterations);
                }
            } else {
                ImGui::Text("%s", widgets::formatNumber(iterations).c_str());
            }
            ImGui::PopItemWidth();
            if (ImGui::IsItemHovered()) {
                widgets::SetThemedTooltip("%s", LOC(Training::Tooltip::ITERATIONS));
            }

            // Max Gaussians
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("%s", LOC(TrainingParams::MAX_GAUSSIANS));
            ImGui::TableNextColumn();
            ImGui::PushItemWidth(-1);
            if (can_edit) {
                if (widgets::InputIntFormatted("##max_cap", &opt_params.max_cap, 10000, 100000)) {
                    opt_params.max_cap = std::max(1, opt_params.max_cap);
                }
            } else {
                ImGui::Text("%s", widgets::formatNumber(opt_params.max_cap).c_str());
            }
            ImGui::PopItemWidth();
            if (ImGui::IsItemHovered()) {
                widgets::SetThemedTooltip("%s", LOC(Training::Tooltip::MAX_GAUSSIANS));
            }

            // SH Degree
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("%s", LOC(TrainingParams::SH_DEGREE));
            ImGui::TableNextColumn();
            if (can_edit) {
                ImGui::PushItemWidth(-1);
                static constexpr const char* const SH_DEGREE_LABELS[] = {"0", "1", "2", "3"};
                ImGui::Combo("##sh_degree", &opt_params.sh_degree, SH_DEGREE_LABELS, 4);
                ImGui::PopItemWidth();
            } else {
                ImGui::Text("%d", opt_params.sh_degree);
            }
            if (ImGui::IsItemHovered()) {
                widgets::SetThemedTooltip("%s", LOC(Training::Tooltip::SH_DEGREE));
            }

            // Tile Mode
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("%s", LOC(TrainingParams::TILE_MODE));
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
            if (ImGui::IsItemHovered()) {
                widgets::SetThemedTooltip("%s", LOC(Training::Tooltip::TILE_MODE));
            }

            // Steps Scaler
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("%s", LOC(TrainingParams::STEPS_SCALER));
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
            if (ImGui::IsItemHovered()) {
                widgets::SetThemedTooltip("%s", LOC(Training::Tooltip::STEPS_SCALER));
            }

            // Bilateral Grid Enable
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("%s", LOC(TrainingParams::BILATERAL_GRID));
            ImGui::TableNextColumn();
            if (can_edit) {
                ImGui::Checkbox("##use_bilateral_grid", &opt_params.use_bilateral_grid);
            } else {
                ImGui::Text("%s", opt_params.use_bilateral_grid ? "Enabled" : "Disabled");
            }
            if (ImGui::IsItemHovered()) {
                widgets::SetThemedTooltip("%s", LOC(Training::Tooltip::BILATERAL_GRID));
            }

            // PPISP Enable
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("PPISP");
            ImGui::TableNextColumn();
            if (can_edit) {
                ImGui::Checkbox("##use_ppisp", &opt_params.use_ppisp);
            } else {
                ImGui::Text("%s", opt_params.use_ppisp ? "Enabled" : "Disabled");
            }
            if (ImGui::IsItemHovered()) {
                widgets::SetThemedTooltip("Enable PPISP (Physically-Plausible ISP) for per-camera appearance modeling");
            }

            // PPISP Controller (shown when PPISP enabled)
            if (opt_params.use_ppisp) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("  Controller");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::Checkbox("##ppisp_controller_main", &opt_params.ppisp_use_controller);
                } else {
                    ImGui::Text("%s", opt_params.ppisp_use_controller ? "Enabled" : "Disabled");
                }
                if (ImGui::IsItemHovered()) {
                    widgets::SetThemedTooltip("Enable controller for novel view synthesis (trains in last 5k iterations)");
                }
            }

            // Mask Mode
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("%s", LOC(TrainingParams::MASK_MODE));
            ImGui::TableNextColumn();
            static constexpr const char* const MASK_MODE_LABELS[] = {"None", "Segment", "Ignore", "Alpha Consistent"};
            if (can_edit && (has_masks || has_alpha)) {
                ImGui::PushItemWidth(-1);
                int current_mask_mode = static_cast<int>(opt_params.mask_mode);
                if (ImGui::Combo("##mask_mode", &current_mask_mode, MASK_MODE_LABELS, IM_ARRAYSIZE(MASK_MODE_LABELS))) {
                    opt_params.mask_mode = static_cast<lfs::core::param::MaskMode>(current_mask_mode);
                }
                ImGui::PopItemWidth();
            } else {
                ImGui::Text("%s", MASK_MODE_LABELS[static_cast<int>(opt_params.mask_mode)]);
                if (!has_masks && !has_alpha && can_edit) {
                    ImGui::SameLine();
                    ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "(no masks)");
                }
            }
            if (ImGui::IsItemHovered()) {
                widgets::SetThemedTooltip("%s", LOC(Training::Tooltip::MASK_MODE));
            }

            if (opt_params.mask_mode != lfs::core::param::MaskMode::None && has_alpha) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("%s", LOC(TrainingParams::USE_ALPHA_AS_MASK));
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::Checkbox("##use_alpha_as_mask", &opt_params.use_alpha_as_mask);
                } else {
                    ImGui::Text("%s", opt_params.use_alpha_as_mask ? "Yes" : "No");
                }
                if (ImGui::IsItemHovered()) {
                    widgets::SetThemedTooltip("%s", LOC(Training::Tooltip::USE_ALPHA_AS_MASK));
                }
            }

            if (opt_params.mask_mode != lfs::core::param::MaskMode::None && (has_masks || has_alpha)) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("%s", LOC(TrainingParams::INVERT_MASKS));
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::Checkbox("##invert_masks", &opt_params.invert_masks);
                } else {
                    ImGui::Text("%s", opt_params.invert_masks ? "Yes" : "No");
                }
                if (ImGui::IsItemHovered()) {
                    widgets::SetThemedTooltip("%s", LOC(Training::Tooltip::INVERT_MASKS));
                }
            }

            if (opt_params.mask_mode == lfs::core::param::MaskMode::Segment) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("%s", LOC(TrainingParams::OPACITY_PENALTY_WEIGHT));
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    ImGui::InputFloat("##mask_penalty_weight", &opt_params.mask_opacity_penalty_weight, 0.1f, 1.0f, "%.1f");
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%.1f", opt_params.mask_opacity_penalty_weight);
                }
                if (ImGui::IsItemHovered()) {
                    widgets::SetThemedTooltip("%s", LOC(Training::Tooltip::PENALTY_WEIGHT));
                }

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("%s", LOC(TrainingParams::OPACITY_PENALTY_POWER));
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    ImGui::SliderFloat("##mask_penalty_power", &opt_params.mask_opacity_penalty_power, 0.5f, 4.0f, "%.1f");
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%.1f", opt_params.mask_opacity_penalty_power);
                }
                if (ImGui::IsItemHovered()) {
                    widgets::SetThemedTooltip("%s", LOC(Training::Tooltip::PENALTY_POWER));
                }

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("%s", LOC(TrainingParams::MASK_THRESHOLD));
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    ImGui::SliderFloat("##mask_threshold", &opt_params.mask_threshold, 0.0f, 1.0f, "%.2f");
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%.2f", opt_params.mask_threshold);
                }
                if (ImGui::IsItemHovered()) {
                    widgets::SetThemedTooltip("%s", LOC(Training::Tooltip::MASK_THRESHOLD));
                }
            }

            // Enable Sparsity
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("%s", LOC(TrainingParams::SPARSITY));
            ImGui::TableNextColumn();
            if (can_edit) {
                if (ImGui::Checkbox("##enable_sparsity", &opt_params.enable_sparsity)) {
                }
            } else {
                ImGui::Text("%s", opt_params.enable_sparsity ? "Enabled" : "Disabled");
            }
            if (ImGui::IsItemHovered()) {
                widgets::SetThemedTooltip("%s", LOC(Training::Tooltip::SPARSITY));
            }

            // GUT
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("%s", LOC(TrainingParams::GUT));
            ImGui::TableNextColumn();
            if (can_edit) {
                ImGui::Checkbox("##gut", &opt_params.gut);
            } else {
                ImGui::Text("%s", opt_params.gut ? "Enabled" : "Disabled");
            }
            if (ImGui::IsItemHovered()) {
                widgets::SetThemedTooltip("%s", LOC(Training::Tooltip::GUT));
            }

            // Mip Filter (anti-aliasing)
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("%s", LOC(TrainingParams::MIP_FILTER));
            ImGui::TableNextColumn();
            if (can_edit) {
                ImGui::Checkbox("##mip_filter", &opt_params.mip_filter);
            } else {
                ImGui::Text("%s", opt_params.mip_filter ? "Enabled" : "Disabled");
            }
            if (ImGui::IsItemHovered()) {
                widgets::SetThemedTooltip("%s", LOC(Training::Tooltip::MIP_FILTER));
            }

            // Background Mode
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("%s", LOC(MainPanel::BACKGROUND));
            ImGui::TableNextColumn();
            {
                const char* bg_mode_items[] = {
                    LOC(TrainingParams::BG_MODE_COLOR),
                    LOC(TrainingParams::BG_MODE_MODULATION),
                    LOC(TrainingParams::BG_MODE_IMAGE),
                    LOC(TrainingParams::BG_MODE_RANDOM)};
                int bg_mode_idx = 0;
                switch (opt_params.bg_mode) {
                case lfs::core::param::BackgroundMode::Modulation:
                    bg_mode_idx = 1;
                    break;
                case lfs::core::param::BackgroundMode::Image:
                    bg_mode_idx = 2;
                    break;
                case lfs::core::param::BackgroundMode::Random:
                    bg_mode_idx = 3;
                    break;
                default:
                    bg_mode_idx = 0;
                    break;
                }
                ImGui::BeginDisabled(!can_edit);
                if (ImGui::Combo("##bg_mode", &bg_mode_idx, bg_mode_items, IM_ARRAYSIZE(bg_mode_items))) {
                    switch (bg_mode_idx) {
                    case 1:
                        opt_params.bg_mode = lfs::core::param::BackgroundMode::Modulation;
                        opt_params.bg_modulation = true;
                        break;
                    case 2:
                        opt_params.bg_mode = lfs::core::param::BackgroundMode::Image;
                        opt_params.bg_modulation = false;
                        break;
                    case 3:
                        opt_params.bg_mode = lfs::core::param::BackgroundMode::Random;
                        opt_params.bg_modulation = false;
                        break;
                    default:
                        opt_params.bg_mode = lfs::core::param::BackgroundMode::SolidColor;
                        opt_params.bg_modulation = false;
                        break;
                    }
                }
                ImGui::EndDisabled();
            }

            // Background Color - only for SolidColor/Modulation modes
            if (opt_params.bg_mode == lfs::core::param::BackgroundMode::SolidColor ||
                opt_params.bg_mode == lfs::core::param::BackgroundMode::Modulation) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("%s", LOC(TrainingParams::BG_COLOR));
                ImGui::TableNextColumn();
                ImGui::BeginDisabled(!can_edit);
                float bg_color[3] = {opt_params.bg_color[0], opt_params.bg_color[1], opt_params.bg_color[2]};
                if (ImGui::ColorEdit3("##bg_color", bg_color, ImGuiColorEditFlags_NoInputs)) {
                    opt_params.bg_color = {bg_color[0], bg_color[1], bg_color[2]};
                }
                ImGui::EndDisabled();
            }

            // Background Image path - only for Image mode
            if (opt_params.bg_mode == lfs::core::param::BackgroundMode::Image) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("%s", LOC(TrainingParams::BG_IMAGE));
                ImGui::TableNextColumn();
                const std::string display_name = opt_params.bg_image_path.empty()
                                                     ? "(none)"
                                                     : lfs::core::path_to_utf8(opt_params.bg_image_path.filename());
                ImGui::Text("%s", display_name.c_str());
                if (!opt_params.bg_image_path.empty() && ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("%s", lfs::core::path_to_utf8(opt_params.bg_image_path).c_str());
                }
                ImGui::SameLine();
                ImGui::BeginDisabled(!can_edit);
                if (ImGui::Button(LOC(TrainingParams::BG_IMAGE_BROWSE))) {
                    const auto selected = OpenImageFileDialog();
                    if (!selected.empty()) {
                        opt_params.bg_image_path = selected;
                    }
                }
                if (!opt_params.bg_image_path.empty()) {
                    ImGui::SameLine();
                    if (ImGui::Button(LOC(TrainingParams::BG_IMAGE_CLEAR))) {
                        opt_params.bg_image_path.clear();
                    }
                }
                ImGui::EndDisabled();
            }
            if (ImGui::IsItemHovered()) {
                widgets::SetThemedTooltip("%s", LOC(Training::Tooltip::BG_MODULATION));
            }
        }
        ImGui::EndTable();

        ImGui::PopStyleVar();
    }

    void DrawTrainingParams(const UIContext& ctx) {

        auto& state = TrainingPanelState::getInstance();
        const auto& t = theme();

        if (ImGui::CollapsingHeader(LOC(Training::Section::BASIC_PARAMS), ImGuiTreeNodeFlags_DefaultOpen)) {
            DrawTrainingParameters(ctx);
        }
        if (ImGui::CollapsingHeader(LOC(Training::Section::ADVANCED_PARAMS))) {
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

        ImGui::Text(LOC(Progress::STATUS_LABEL), state_str);
        g_iter_rate_tracker.addSample(current_iteration);
        float iters_per_sec = g_iter_rate_tracker.getIterationsPerSecond();
        iters_per_sec = iters_per_sec > 0.0f ? iters_per_sec : 0.0f;

        ImGui::Text("Iteration: %s (%.1f iters/sec)", widgets::formatNumber(current_iteration).c_str(), iters_per_sec);

        int num_splats = trainer_manager->getNumSplats();
        ImGui::Text(LOC(Progress::NUM_SPLATS), widgets::formatNumber(num_splats).c_str());
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
            const char* const label = current_iteration > 0 ? LOC(TrainingPanel::RESUME_TRAINING) : LOC(TrainingPanel::START_TRAINING);
            if (ColoredButton(label, ButtonStyle::Success, FULL_WIDTH)) {
                lfs::core::events::cmd::StartTraining{}.emit();
            }
            if (ImGui::IsItemHovered()) {
                widgets::SetThemedTooltip("%s", current_iteration > 0 ? LOC(Training::Tooltip::BTN_RESUME) : LOC(Training::Tooltip::BTN_START));
            }
            if (current_iteration > 0) {
                if (ColoredButton(LOC(TrainingPanel::RESET), ButtonStyle::Secondary, FULL_WIDTH)) {
                    lfs::core::events::cmd::ResetTraining{}.emit();
                }
                if (ImGui::IsItemHovered()) {
                    widgets::SetThemedTooltip("%s", LOC(Training::Tooltip::BTN_RESET));
                }
            }
            if (ColoredButton(LOC(TrainingPanel::CLEAR), ButtonStyle::Error, FULL_WIDTH)) {
                lfs::core::events::cmd::ClearScene{}.emit();
            }
            if (ImGui::IsItemHovered()) {
                widgets::SetThemedTooltip("%s", LOC(Training::Tooltip::BTN_CLEAR));
            }
            break;
        }

        case TrainerManager::State::Running:
            if (ColoredButton(LOC(TrainingPanel::PAUSE), ButtonStyle::Warning, FULL_WIDTH)) {
                lfs::core::events::cmd::PauseTraining{}.emit();
            }
            if (ImGui::IsItemHovered()) {
                widgets::SetThemedTooltip("%s", LOC(Training::Tooltip::BTN_PAUSE));
            }
            break;

        case TrainerManager::State::Paused:
            if (ColoredButton(LOC(TrainingPanel::RESUME), ButtonStyle::Success, FULL_WIDTH)) {
                lfs::core::events::cmd::ResumeTraining{}.emit();
            }
            if (ImGui::IsItemHovered()) {
                widgets::SetThemedTooltip("%s", LOC(Training::Tooltip::BTN_RESUME));
            }
            if (ColoredButton(LOC(TrainingPanel::RESET), ButtonStyle::Secondary, FULL_WIDTH)) {
                lfs::core::events::cmd::ResetTraining{}.emit();
            }
            if (ImGui::IsItemHovered()) {
                widgets::SetThemedTooltip("%s", LOC(Training::Tooltip::BTN_RESET));
            }
            if (ColoredButton(LOC(TrainingPanel::STOP), ButtonStyle::Error, FULL_WIDTH)) {
                lfs::core::events::cmd::StopTraining{}.emit();
            }
            if (ImGui::IsItemHovered()) {
                widgets::SetThemedTooltip("%s", LOC(Training::Tooltip::BTN_STOP));
            }
            break;

        case TrainerManager::State::Finished: {
            const auto reason = trainer_manager->getStateMachine().getFinishReason();
            switch (reason) {
            case FinishReason::Completed:
                ImGui::TextColored(t.palette.success, "%s", LOC(Messages::TRAINING_COMPLETE));
                break;
            case FinishReason::UserStopped:
                ImGui::TextColored(t.palette.text_dim, "%s", LOC(Messages::TRAINING_STOPPED));
                break;
            case FinishReason::Error:
                ImGui::TextColored(t.palette.error, "%s", LOC(Messages::TRAINING_ERROR));
                if (const auto error_msg = trainer_manager->getLastError(); !error_msg.empty()) {
                    ImGui::TextWrapped("%s", error_msg.c_str());
                }
                break;
            default:
                ImGui::TextColored(t.palette.text_dim, "%s", LOC(TrainingPanel::FINISHED));
            }

            if (reason == FinishReason::Completed || reason == FinishReason::UserStopped) {
                if (ColoredButton(LOC(TrainingPanel::SWITCH_EDIT_MODE), ButtonStyle::Success, FULL_WIDTH)) {
                    lfs::core::events::cmd::SwitchToEditMode{}.emit();
                }
                if (ImGui::IsItemHovered()) {
                    widgets::SetThemedTooltip("%s", LOC(Training::Tooltip::KEEP_MODEL));
                }
            }
            if (ColoredButton(LOC(TrainingPanel::RESET), ButtonStyle::Secondary, FULL_WIDTH)) {
                lfs::core::events::cmd::ResetTraining{}.emit();
            }
            if (ImGui::IsItemHovered()) {
                widgets::SetThemedTooltip("%s", LOC(Training::Tooltip::BTN_RESET));
            }

            if (ColoredButton(LOC(TrainingPanel::CLEAR), ButtonStyle::Error, FULL_WIDTH)) {
                lfs::core::events::cmd::ClearScene{}.emit();
            }
            if (ImGui::IsItemHovered()) {
                widgets::SetThemedTooltip("%s", LOC(Training::Tooltip::BTN_CLEAR));
            }

            break;
        }

        case TrainerManager::State::Stopping:
            ImGui::TextColored(t.palette.text_dim, "%s", LOC(Status::STOPPING));
            break;
        }

        // Save checkpoint button
        if (trainer_state == TrainerManager::State::Running ||
            trainer_state == TrainerManager::State::Paused) {
            if (ColoredButton(LOC(TrainingPanel::SAVE_CHECKPOINT), ButtonStyle::Primary, FULL_WIDTH)) {
                lfs::core::events::cmd::SaveCheckpoint{}.emit();
                state.save_in_progress = true;
                state.save_start_time = std::chrono::steady_clock::now();
            }
            if (ImGui::IsItemHovered()) {
                widgets::SetThemedTooltip("%s", LOC(Training::Tooltip::BTN_SAVE_CHECKPOINT));
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
                ImGui::TextColored(t.palette.success, "%s", LOC(TrainingPanel::CHECKPOINT_SAVED));
            } else {
                state.save_in_progress = false;
            }
        }
    }

} // namespace lfs::vis::gui::panels
