/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "progress.hpp"

#include "indicators.hpp"

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

    void TrainingProgress::complete() {
        if (!impl_->progress_bar->is_completed()) {
            impl_->progress_bar->set_progress(100);
            impl_->progress_bar->mark_as_completed();
            std::cout << std::endl;
        }
    }

    void TrainingProgress::print_final_summary(const int final_splats, const int actual_iterations) {
        complete();

        const auto end_time = std::chrono::steady_clock::now();
        const auto elapsed = std::chrono::duration<double>(end_time - impl_->start_time).count();
        const int iterations_used = actual_iterations > 0 ? actual_iterations : impl_->total_iterations;

        std::cout << std::endl
#ifdef _WIN32
                  << "* Training completed in "
#else
                  << "✓ Training completed in "
#endif
                  << std::fixed << std::setprecision(3) << elapsed << "s"
                  << " (avg " << std::fixed << std::setprecision(1)
                  << iterations_used / elapsed << " iter/s)"
                  << std::endl
#ifdef _WIN32
                  << "* Final splats: " << final_splats
#else
                  << "✓ Final splats: " << final_splats
#endif
                  << std::endl;
    }
} // namespace lfs::training
