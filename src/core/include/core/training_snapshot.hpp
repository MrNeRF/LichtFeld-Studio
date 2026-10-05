/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "core/logger.hpp"
#include "core/tensor.hpp"
#include <filesystem>
#include <format>
#include <map>
#include <optional>
#include <string>

namespace lfs::core::debug {

    // Statistics stored in a training snapshot.
    struct TensorStats {
        float min = 0.0f;
        float max = 0.0f;
        float mean = 0.0f;
        float std = 0.0f;
        size_t numel = 0;
        TensorShape shape;
        DataType dtype = DataType::Float32;
        bool is_cuda = false;
    };

    // Memory info snapshot
    struct MemorySnapshot {
        size_t gpu_used_bytes = 0;
        size_t gpu_total_bytes = 0;
        size_t gpu_free_bytes = 0;
        size_t arena_used_bytes = 0;
        size_t arena_total_bytes = 0;

        [[nodiscard]] float gpu_usage_percent() const {
            return gpu_total_bytes > 0
                       ? 100.0f * static_cast<float>(gpu_used_bytes) / static_cast<float>(gpu_total_bytes)
                       : 0.0f;
        }

        [[nodiscard]] std::string to_string() const {
            return std::format("GPU: {:.1f}/{:.1f} GB ({:.1f}%), Arena: {:.1f}/{:.1f} GB",
                               gpu_used_bytes / 1e9, gpu_total_bytes / 1e9, gpu_usage_percent(),
                               arena_used_bytes / 1e9, arena_total_bytes / 1e9);
        }
    };

    // Get current memory state
    LFS_CORE_API MemorySnapshot get_memory_snapshot();

    // Training state snapshot for debugging
    struct LFS_CORE_API TrainingSnapshot {
        int iteration = 0;
        float loss = 0.0f;
        float learning_rate = 0.0f;
        size_t num_gaussians = 0;
        MemorySnapshot memory;
        std::map<std::string, TensorStats> param_stats;
        std::map<std::string, TensorStats> grad_stats;

        // Serialize to JSON string
        [[nodiscard]] std::string to_json() const;

        // Save to file
        bool save(const std::filesystem::path& path) const;

        // Load from file
        static std::optional<TrainingSnapshot> load(const std::filesystem::path& path);

        // Print summary to log
        void log_summary() const {
            LOG_INFO("=== Training Snapshot (iter {}) ===", iteration);
            LOG_INFO("  Loss: {:.6f}, LR: {:.6e}", loss, learning_rate);
            LOG_INFO("  Gaussians: {}", num_gaussians);
            LOG_INFO("  Memory: {}", memory.to_string());

            if (!param_stats.empty()) {
                LOG_INFO("  Parameters:");
                for (const auto& [name, stats] : param_stats) {
                    LOG_INFO("    {}: min={:.4f}, max={:.4f}, mean={:.4f}, std={:.4f}",
                             name, stats.min, stats.max, stats.mean, stats.std);
                }
            }

            if (!grad_stats.empty()) {
                LOG_INFO("  Gradients:");
                for (const auto& [name, stats] : grad_stats) {
                    LOG_INFO("    {}: min={:.4f}, max={:.4f}, mean={:.4f}, std={:.4f}",
                             name, stats.min, stats.max, stats.mean, stats.std);
                }
            }
        }
    };

    // Compare two snapshots and report differences
    struct SnapshotDiff {
        float loss_diff = 0.0f;
        int gaussian_diff = 0;
        std::map<std::string, float> param_mean_diffs;
        std::map<std::string, float> grad_mean_diffs;

        void log_summary() const {
            LOG_INFO("=== Snapshot Diff ===");
            LOG_INFO("  Loss diff: {:+.6f}", loss_diff);
            LOG_INFO("  Gaussian diff: {:+d}", gaussian_diff);

            for (const auto& [name, diff] : param_mean_diffs) {
                LOG_INFO("  Param {} mean diff: {:+.6f}", name, diff);
            }
        }
    };

    LFS_CORE_API SnapshotDiff diff_snapshots(const TrainingSnapshot& a, const TrainingSnapshot& b);

} // namespace lfs::core::debug
