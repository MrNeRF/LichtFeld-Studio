/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/error.hpp"
#include "core/export.hpp"
#include "core/tensor.hpp"
#include "core/tensor_ppisp.hpp"
#include "rendering_types.hpp"

#include <array>
#include <filesystem>
#include <istream>
#include <memory>
#include <string>
#include <vector>

namespace lfs::vis {
    class LFS_VIS_API AppearanceTensorModel {
    public:
        static lfs::Result<std::unique_ptr<AppearanceTensorModel>> load(
            const std::filesystem::path& path, lfs::core::GpuBackend backend);

        static lfs::Result<std::unique_ptr<AppearanceTensorModel>> load(
            std::istream& input, lfs::core::GpuBackend backend, int camera_index = 0);

        [[nodiscard]] bool hasController() const { return !controller_.empty(); }
        [[nodiscard]] lfs::core::Tensor predict(const lfs::core::Tensor& rgb_chw) const;
        [[nodiscard]] lfs::core::PpispParams parameters(int camera_uid, const PPISPOverrides& overrides,
                                                        const lfs::core::Tensor& controller_params = {}) const;
        [[nodiscard]] lfs::core::Tensor apply(const lfs::core::Tensor& rgb_chw,
                                              int camera_uid, const PPISPOverrides& overrides,
                                              bool use_controller) const;

    private:
        lfs::core::GpuBackend backend_ = lfs::core::GpuBackend::Vulkan;
        int num_cameras_ = 0;
        int camera_index_ = 0;
        int num_frames_ = 0;
        std::vector<float> exposure_, vignetting_, color_, crf_;
        std::array<lfs::core::Tensor, 6> convolution_;
        std::vector<std::array<lfs::core::Tensor, 8>> controller_;
    };
} // namespace lfs::vis
