/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/export.hpp"
#include "core/job_registry.hpp"
#include "core/tensor.hpp"
#include <array>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace lfs::vis::gui {
    struct AssistedMaskRequest {
        std::filesystem::path image;
        std::filesystem::path weights;
        std::string revision;
        int64_t mtime_ns = 0;
        uint64_t size_bytes = 0;
        std::vector<std::array<float, 3>> points;
        std::optional<std::array<float, 4>> box;
        uint64_t refine_request = 0;
        int refine_candidate = -1;
        int export_candidate = -1;
        bool exclude = false;
    };
    struct AssistedMaskSnapshot {
        uint64_t request = 0;
        std::string revision;
        std::string phase = "idle";
        std::string error;
        std::vector<core::Tensor> overlays;
        std::vector<float> scores;
        std::shared_ptr<const std::vector<std::byte>> mask;
        int width = 0, height = 0;
        std::filesystem::path image;
        std::filesystem::file_time_type source_time{};
        uint64_t source_bytes = 0;
    };

    // App-owned worker. Closing an editor requests retirement; only app shutdown joins.
    // Predictions are disposable proposals and never write dataset sidecars.
    class LFS_VIS_API AssistedMasking {
    public:
        AssistedMasking(JobRegistry& jobs, std::function<bool()> admission);
        ~AssistedMasking();
        uint64_t submit(AssistedMaskRequest request);
        AssistedMaskSnapshot poll();
        void close();
        void shutdown();

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace lfs::vis::gui
