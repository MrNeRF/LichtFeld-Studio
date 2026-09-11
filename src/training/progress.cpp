/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "progress.hpp"

#include "indicators.hpp"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace lfs::training {
    struct TrainingProgress::Impl {
        std::unique_ptr<indicators::ProgressBar> progress_bar;
        std::chrono::steady_clock::time_point start_time;
        int total_iterations;
        int update_frequency;
    };

    namespace {
        const char* phase_label(const TrainingProgress::Phase phase) {
            switch (phase) {
            case TrainingProgress::Phase::Train:
                return "";
            case TrainingProgress::Phase::Refine:
                return "(+)";
            case TrainingProgress::Phase::Controller:
                return "Ctrl";
            case TrainingProgress::Phase::Sparse:
                return "(-)";
            }
            return "";
        }
    } // namespace

    TrainingProgress::TrainingProgress(const int total_iterations, const int update_frequency)
        : impl_(std::make_unique<Impl>()) {
        impl_->total_iterations = total_iterations;
        impl_->update_frequency = update_frequency;
        impl_->progress_bar = std::make_unique<indicators::ProgressBar>();

        impl_->progress_bar->set_option(indicators::option::Start("["));

#ifdef _WIN32
        impl_->progress_bar->set_option(indicators::option::BarWidth(38));
        impl_->progress_bar->set_option(indicators::option::Fill("="));
        impl_->progress_bar->set_option(indicators::option::Lead(">"));
        impl_->progress_bar->set_option(indicators::option::Remainder(" "));
#else
        impl_->progress_bar->set_option(indicators::option::BarWidth(40));
        impl_->progress_bar->set_option(indicators::option::Fill("█"));
        impl_->progress_bar->set_option(indicators::option::Lead("▌"));
        impl_->progress_bar->set_option(indicators::option::Remainder("░"));
#endif
        impl_->progress_bar->set_option(indicators::option::End("]"));
        impl_->progress_bar->set_option(indicators::option::PrefixText("Training "));
        impl_->progress_bar->set_option(indicators::option::PostfixText("Initializing..."));
        impl_->progress_bar->set_option(indicators::option::ShowPercentage(true));
        impl_->progress_bar->set_option(indicators::option::ShowElapsedTime(true));
        impl_->progress_bar->set_option(indicators::option::ShowRemainingTime(true));
        impl_->progress_bar->set_option(indicators::option::ForegroundColor(indicators::Color::cyan));

        std::vector<indicators::FontStyle> styles;
        styles.push_back(indicators::FontStyle::bold);
        impl_->progress_bar->set_option(indicators::option::FontStyles(styles));
        impl_->start_time = std::chrono::steady_clock::now();
    }

    TrainingProgress::~TrainingProgress() {
        complete();
    }

    void TrainingProgress::update(
        const int current_iteration,
        const float loss,
        const int splat_count,
        const Phase phase) {
        if (current_iteration % impl_->update_frequency != 0) {
            return;
        }

        const float progress = static_cast<float>(current_iteration) / impl_->total_iterations * 100;
        impl_->progress_bar->set_progress(static_cast<size_t>(progress));

        std::ostringstream postfix;
        postfix << current_iteration << "/" << impl_->total_iterations
                << " | Loss: " << std::fixed << std::setprecision(4) << loss
                << " | Splats: " << splat_count
                << " " << phase_label(phase);
        impl_->progress_bar->set_option(indicators::option::PostfixText(postfix.str()));
    }

    void TrainingProgress::pause() {
        if (!impl_->progress_bar->is_completed()) {
            impl_->progress_bar->mark_as_completed();
            std::cout << std::endl;
        }
    }

    void TrainingProgress::resume(
        const int current_iteration,
        const float loss,
        const int splat_count,
        const Phase phase) {
        impl_->progress_bar->set_progress(static_cast<size_t>(
            static_cast<float>(current_iteration) / impl_->total_iterations * 100));
        update(current_iteration, loss, splat_count, phase);
    }

    void TrainingProgress::complete(const bool user_stopped, const int actual_iterations) {
        if (!impl_->progress_bar->is_completed()) {
            const int iterations = actual_iterations > 0 ? actual_iterations : impl_->total_iterations;
            const float fraction = impl_->total_iterations > 0
                                       ? static_cast<float>(iterations) / impl_->total_iterations
                                       : 0.0f;
            impl_->progress_bar->set_progress(static_cast<size_t>(std::clamp(fraction, 0.0f, 1.0f) * 100.0f));
            if (user_stopped) {
                impl_->progress_bar->set_option(indicators::option::PostfixText(
                    "Stopped by user at " + std::to_string(iterations) + "/" +
                    std::to_string(impl_->total_iterations)));
            }
            impl_->progress_bar->mark_as_completed();
            std::cout << std::endl;
        }
    }

    void TrainingProgress::print_final_summary(const int final_splats, const int actual_iterations,
                                               const bool user_stopped) {
        complete(user_stopped, actual_iterations);

        const auto end_time = std::chrono::steady_clock::now();
        const auto elapsed = std::chrono::duration<double>(end_time - impl_->start_time).count();
        const int iterations_used = actual_iterations > 0 ? actual_iterations : impl_->total_iterations;

        std::ostringstream summary;
        summary << std::fixed << std::setprecision(3);
#ifdef _WIN32
        summary << (user_stopped ? "* Training stopped by user at iteration "
                                 : "* Training completed in ");
#else
        summary << (user_stopped ? "Training stopped by user at iteration "
                                 : "✓ Training completed in ");
#endif
        if (user_stopped) {
            summary << iterations_used << " (checkpoint saved)";
        } else {
            summary << elapsed << "s (avg " << std::setprecision(1)
                    << (elapsed > 0.0 ? iterations_used / elapsed : 0.0) << " iter/s)";
        }
        std::cout << std::endl
                  << summary.str() << std::endl
#ifdef _WIN32
                  << "* Final splats: " << final_splats
#else
                  << (user_stopped ? "* Final splats: " : "✓ Final splats: ") << final_splats
#endif
                  << std::endl;
    }
} // namespace lfs::training
