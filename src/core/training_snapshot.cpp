/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/training_snapshot.hpp"
#include "core/path_utils.hpp"
#include <cuda_runtime.h>
#include <fstream>
#include <sstream>

namespace lfs::core::debug {

    MemorySnapshot get_memory_snapshot() {
        MemorySnapshot snapshot;

        size_t free_bytes = 0, total_bytes = 0;
        if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
            snapshot.gpu_free_bytes = free_bytes;
            snapshot.gpu_total_bytes = total_bytes;
            snapshot.gpu_used_bytes = total_bytes - free_bytes;
        }

        return snapshot;
    }

    std::string TrainingSnapshot::to_json() const {
        std::ostringstream oss;
        oss << "{\n";
        oss << "  \"iteration\": " << iteration << ",\n";
        oss << "  \"loss\": " << std::fixed << std::setprecision(8) << loss << ",\n";
        oss << "  \"learning_rate\": " << std::scientific << learning_rate << ",\n";
        oss << "  \"num_gaussians\": " << num_gaussians << ",\n";
        oss << "  \"memory\": {\n";
        oss << "    \"gpu_used_gb\": " << std::fixed << std::setprecision(3)
            << memory.gpu_used_bytes / 1e9 << ",\n";
        oss << "    \"gpu_total_gb\": " << memory.gpu_total_bytes / 1e9 << ",\n";
        oss << "    \"gpu_usage_percent\": " << std::setprecision(1)
            << memory.gpu_usage_percent() << "\n";
        oss << "  },\n";

        oss << "  \"parameters\": {\n";
        size_t i = 0;
        for (const auto& [name, stats] : param_stats) {
            oss << "    \"" << name << "\": {"
                << "\"min\": " << std::setprecision(6) << stats.min << ", "
                << "\"max\": " << stats.max << ", "
                << "\"mean\": " << stats.mean << ", "
                << "\"std\": " << stats.std << ", "
                << "\"numel\": " << stats.numel << "}";
            if (++i < param_stats.size())
                oss << ",";
            oss << "\n";
        }
        oss << "  },\n";

        oss << "  \"gradients\": {\n";
        i = 0;
        for (const auto& [name, stats] : grad_stats) {
            oss << "    \"" << name << "\": {"
                << "\"min\": " << std::setprecision(6) << stats.min << ", "
                << "\"max\": " << stats.max << ", "
                << "\"mean\": " << stats.mean << ", "
                << "\"std\": " << stats.std << "}";
            if (++i < grad_stats.size())
                oss << ",";
            oss << "\n";
        }
        oss << "  }\n";

        oss << "}\n";
        return oss.str();
    }

    bool TrainingSnapshot::save(const std::filesystem::path& path) const {
        std::ofstream file;
        if (!lfs::core::open_file_for_write(path, file)) {
            LOG_ERROR("Failed to open snapshot file for writing: {}", lfs::core::path_to_utf8(path));
            return false;
        }
        file << to_json();
        file.close();
        LOG_INFO("Saved training snapshot to {}", lfs::core::path_to_utf8(path));
        return true;
    }

    std::optional<TrainingSnapshot> TrainingSnapshot::load(const std::filesystem::path& path) {
        // Simple JSON parsing - for full implementation use nlohmann/json
        std::ifstream file;
        if (!lfs::core::open_file_for_read(path, file)) {
            LOG_ERROR("Failed to open snapshot file: {}", lfs::core::path_to_utf8(path));
            return std::nullopt;
        }

        TrainingSnapshot snapshot;
        std::string line;
        while (std::getline(file, line)) {
            if (line.find("\"iteration\":") != std::string::npos) {
                sscanf(line.c_str(), " \"iteration\": %d", &snapshot.iteration);
            } else if (line.find("\"loss\":") != std::string::npos) {
                sscanf(line.c_str(), " \"loss\": %f", &snapshot.loss);
            } else if (line.find("\"num_gaussians\":") != std::string::npos) {
                sscanf(line.c_str(), " \"num_gaussians\": %zu", &snapshot.num_gaussians);
            }
        }

        return snapshot;
    }

    SnapshotDiff diff_snapshots(const TrainingSnapshot& a, const TrainingSnapshot& b) {
        SnapshotDiff diff;
        diff.loss_diff = b.loss - a.loss;
        diff.gaussian_diff = static_cast<int>(b.num_gaussians) - static_cast<int>(a.num_gaussians);

        for (const auto& [name, stats_b] : b.param_stats) {
            if (auto it = a.param_stats.find(name); it != a.param_stats.end()) {
                diff.param_mean_diffs[name] = stats_b.mean - it->second.mean;
            }
        }

        for (const auto& [name, stats_b] : b.grad_stats) {
            if (auto it = a.grad_stats.find(name); it != a.grad_stats.end()) {
                diff.grad_mean_diffs[name] = stats_b.mean - it->second.mean;
            }
        }

        return diff;
    }

} // namespace lfs::core::debug
