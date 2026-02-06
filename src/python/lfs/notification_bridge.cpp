/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "notification_bridge.hpp"
#include "core/event_bridge/localization_manager.hpp"
#include "core/events.hpp"
#include "core/path_utils.hpp"
#include "gui/string_keys.hpp"
#include "py_ui.hpp"

#include <cmath>
#include <format>

namespace lfs::python {

    using namespace lfs::core::events;

    static std::string formatDuration(const float seconds) {
        const float clamped = std::max(0.0f, seconds);
        const int total = static_cast<int>(std::round(clamped));
        const int hours = total / 3600;
        const int minutes = (total % 3600) / 60;
        const int secs = total % 60;

        if (hours > 0) {
            return std::format("{}h {}m {}s", hours, minutes, secs);
        }
        if (minutes > 0) {
            return std::format("{}m {}s", minutes, secs);
        }
        if (clamped >= 1.0f) {
            return std::format("{}s", secs);
        }
        return std::format("{:.1f}s", clamped);
    }

    void setup_notification_handlers() {
        state::DatasetLoadCompleted::when([](const auto& e) {
            if (!e.success && e.error.has_value()) {
                PyModalRegistry::instance().show_message(
                    "Failed to Load Dataset", *e.error, MessageStyle::Error);
            }
        });

        state::CudaVersionUnsupported::when([](const auto& e) {
            PyModalRegistry::instance().show_message(
                "Unsupported CUDA Driver",
                std::format("Your CUDA driver version ({}.{}) is not supported.\n\n"
                            "LichtFeld Studio requires CUDA {}.{} or later\n"
                            "(NVIDIA driver 570+).\n\n"
                            "Please update your NVIDIA driver for full functionality.",
                            e.major, e.minor, e.min_major, e.min_minor),
                MessageStyle::Warning);
        });

        state::ConfigLoadFailed::when([](const auto& e) {
            PyModalRegistry::instance().show_message(
                "Invalid Config File",
                std::format("Could not load '{}':\n\n{}", lfs::core::path_to_utf8(e.path.filename()), e.error),
                MessageStyle::Error);
        });

        state::FileDropFailed::when([](const auto& e) {
            constexpr size_t MAX_DISPLAY = 5;
            namespace Notif = lichtfeld::Strings::Notification;

            const size_t count = e.files.size();
            const size_t display_count = std::min(count, MAX_DISPLAY);

            std::string file_list;
            file_list.reserve(display_count * 64);

            for (size_t i = 0; i < display_count; ++i) {
                const std::filesystem::path p(e.files[i]);
                const bool is_dir = std::filesystem::is_directory(p);
                file_list += std::format("  - {} ({})\n", lfs::core::path_to_utf8(p.filename()),
                                         is_dir ? LOC(Notif::DIRECTORY) : LOC(Notif::FILE));
            }
            if (count > MAX_DISPLAY) {
                file_list += std::format("  {} {}\n", LOC(Notif::AND_MORE), count - MAX_DISPLAY);
            }

            const bool single_dir = count == 1 && std::filesystem::is_directory(e.files[0]);
            const char* item_type = count == 1 ? (single_dir ? LOC(Notif::DIRECTORY) : LOC(Notif::FILE))
                                               : LOC(Notif::ITEMS);

            PyModalRegistry::instance().show_message(
                LOC(Notif::CANNOT_OPEN),
                std::format("{} {}:\n\n{}\n{}", LOC(Notif::DROPPED_NOT_RECOGNIZED), item_type, file_list, e.error),
                MessageStyle::Error);
        });

        state::DiskSpaceSaveFailed::when([](const auto& e) {
            if (!e.is_disk_space_error) {
                const std::string title = e.is_checkpoint ? "Checkpoint Save Failed" : "Export Failed";
                const std::string msg = e.is_checkpoint
                                            ? std::format("Failed to save checkpoint at iteration {}:\n\n{}", e.iteration, e.error)
                                            : std::format("Failed to export:\n\n{}", e.error);
                PyModalRegistry::instance().show_message(title, msg, MessageStyle::Error);
            }
            // Note: Disk space error dialog is handled separately in gui_manager
        });

        state::TrainingCompleted::when([](const auto& e) {
            if (e.user_stopped)
                return;

            if (e.success) {
                namespace Str = lichtfeld::Strings::Training::Button;

                const auto message = std::format(
                    "Training completed successfully!\n\n"
                    "Iterations: {}\n"
                    "Final loss: {:.6f}\n"
                    "Duration: {}",
                    e.iteration, e.final_loss, formatDuration(e.elapsed_seconds));

                const std::string edit_label = LOC(Str::SWITCH_EDIT_MODE);
                PyModalRegistry::instance().show_confirm(
                    "Training Complete", message,
                    {edit_label, "OK"},
                    [edit_label](const std::string& clicked) {
                        if (clicked == edit_label)
                            cmd::SwitchToEditMode{}.emit();
                    });

                cmd::SwitchToLatestCheckpoint{}.emit();
            } else {
                std::string error_msg = e.error.value_or("Unknown error occurred during training.");

                // Check if this is an OOM error and format it clearly
                if (error_msg.find("CUDA out of memory") != std::string::npos ||
                    error_msg.find("OUT_OF_MEMORY") != std::string::npos) {
                    std::string size_info;
                    auto gb_pos = error_msg.find(" GB)");
                    if (gb_pos != std::string::npos) {
                        auto paren_pos = error_msg.rfind('(', gb_pos);
                        if (paren_pos != std::string::npos) {
                            size_info = error_msg.substr(paren_pos + 1, gb_pos - paren_pos + 3);
                        }
                    }

                    error_msg = "Out of GPU memory!\n\n";
                    if (!size_info.empty()) {
                        error_msg += "Tried to allocate " + size_info + "\n\n";
                    }
                    error_msg += "Suggestions:\n"
                                 "  - Reduce max Gaussians (max_cap)\n"
                                 "  - Lower spherical harmonics degree (sh_degree)\n"
                                 "  - Reduce image resolution (resize_factor)\n"
                                 "  - Enable tile mode for large images";
                }

                PyModalRegistry::instance().show_message(
                    "Training Failed", error_msg, MessageStyle::Error);
            }
        });
    }

} // namespace lfs::python
