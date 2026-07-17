/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "notification_bridge.hpp"
#include "core/event_bridge/localization_manager.hpp"
#include "core/events.hpp"
#include "gui/string_keys.hpp"
#include "py_ui.hpp"

#include <cmath>
#include <format>
#include <mutex>
#include <optional>

namespace lfs::python {

    using namespace lfs::core::events;

    namespace {
        struct LatestEvaluationMetrics {
            int iteration = 0;
            float psnr = 0.0f;
            float ssim = 0.0f;
        };

        std::mutex g_latest_eval_metrics_mutex;
        std::optional<LatestEvaluationMetrics> g_latest_eval_metrics;
    } // namespace

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
        // Failure events (dataset/config/export/video/mesh/training/CUDA/disk) now
        // surface through the native C++ ErrorBus (Phase 8, error_event_bridge),
        // so they appear even when Python is absent. Re-adding failure modals here
        // would double them. Only the success TrainingCompleted modal and the
        // eval-metrics tracking it depends on remain Python-owned.
        state::TrainingStarted::when([](const auto&) {
            std::lock_guard lock(g_latest_eval_metrics_mutex);
            g_latest_eval_metrics.reset();
        });

        state::EvaluationCompleted::when([](const auto& e) {
            std::lock_guard lock(g_latest_eval_metrics_mutex);
            g_latest_eval_metrics = LatestEvaluationMetrics{
                .iteration = e.iteration,
                .psnr = e.psnr,
                .ssim = e.ssim};
        });

        state::TrainingCompleted::when([](const auto& e) {
            if (e.user_stopped)
                return;
            if (!e.success)
                return; // failures surface via the native ErrorBus bridge

            namespace Str = lichtfeld::Strings::Training::Button;

            auto message = std::format(
                "Training completed successfully.\n\n"
                "{} iterations | loss {:.6f} | {}",
                e.iteration, e.final_loss, formatDuration(e.elapsed_seconds));
            std::optional<LatestEvaluationMetrics> eval_snapshot;
            {
                std::lock_guard lock(g_latest_eval_metrics_mutex);
                eval_snapshot = g_latest_eval_metrics;
            }
            if (eval_snapshot.has_value()) {
                if (eval_snapshot->iteration == e.iteration) {
                    message += std::format(
                        "\nFinal metrics: PSNR {:.2f} | SSIM {:.4f}",
                        eval_snapshot->psnr,
                        eval_snapshot->ssim);
                } else {
                    message += std::format(
                        "\nLast eval @ {}: PSNR {:.2f} | SSIM {:.4f}",
                        eval_snapshot->iteration,
                        eval_snapshot->psnr,
                        eval_snapshot->ssim);
                }
            }

            const std::string edit_label = LOC(Str::SWITCH_EDIT_MODE);
            PyModalRegistry::instance().show_confirm(
                "Training Complete", message,
                {edit_label, "OK"},
                [edit_label](const std::string& clicked) {
                    if (clicked == edit_label)
                        cmd::SwitchToEditMode{}.emit();
                });

            cmd::SwitchToLatestCheckpoint{}.emit();
        });
    }

} // namespace lfs::python
