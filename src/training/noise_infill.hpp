/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/error.hpp"
#include "core/splat_data.hpp"
#include "core/splat_data_transform.hpp"
#include "core/tensor.hpp"

#include <array>
#include <cstddef>
#include <filesystem>
#include <functional>
#include <optional>

namespace lfs::core {
    struct MeshData;
    namespace param {
        struct OptimizationParameters;
    }
} // namespace lfs::core

namespace lfs::training {

    class CameraDataset;

    // Noise guided splatting (arXiv 2510.15736): opaque Gaussians fill the inside of an object and take new colours on every
    // training step, so any transparency of its surface shows them and the photometric loss makes it opaque.

    // Cells kept free under the surface, so the noise stays behind it.
    inline constexpr int kNoiseInfillErosion = 2;
    using core::kHiddenOpacity;
    using core::kShownNoiseOpacity;
    using core::noise_infill_rows;
    using core::without_noise_infill;

    struct NoiseInfill {
        core::SplatData splats;
        float cell = 0.0f;
        std::size_t capped_loops = 0;
    };

    // One opaque Gaussian per cell inside `mesh` and clear of its surface. Inside are the cells of the collision grid of the
    // mesh's surface splats that no path of free cells joins to the grid's border; open boundary loops, such as the
    // underside of an object scanned on a table, are capped first.
    [[nodiscard]] lfs::Result<NoiseInfill> make_noise_infill(const core::MeshData& mesh);

    // The noise infill of the mesh file, moved from dataset coordinates to the training frame at `origin`.
    [[nodiscard]] lfs::Result<core::SplatData> load_noise_infill(const std::filesystem::path& mesh,
                                                                 const std::array<float, 3>& origin);

    // The training step at which the noise appears; until then its rows render invisible, so the surface forms undisturbed.
    [[nodiscard]] int noise_infill_start_iteration(const core::param::OptimizationParameters& params);

    // New colours for the first `count` rows of `model`, from the paper's six primary and secondary colours, which contrast
    // most with natural surface colours.
    void randomize_noise_colors(core::SplatData& model, std::size_t count);

    // Runs before each forward pass: new colours for noise that shows, and hidden noise appears once its start iteration
    // is reached. Returns whether it appears now.
    [[nodiscard]] bool noise_infill_step(core::SplatData& model, const core::param::OptimizationParameters& params, int iter);

    // `params` with the noise rows of `model` on top of its row cap: the noise never reaches an export, so it takes no row
    // from the surface.
    [[nodiscard]] core::param::OptimizationParameters with_noise_outside_cap(const core::param::OptimizationParameters& params,
                                                                             const core::SplatData* model);

    // Rows [begin, end) of a tensor set to `value`, restored when this goes out of scope.
    class RowOverride {
    public:
        RowOverride(core::Tensor& tensor, std::size_t begin, std::size_t end, float value);
        ~RowOverride();
        RowOverride(const RowOverride&) = delete;
        RowOverride& operator=(const RowOverride&) = delete;

        [[nodiscard]] core::Tensor& rows() { return rows_; }

    private:
        core::Tensor rows_;
        core::Tensor saved_;
    };

    // Writes green into the [count, 1, 3] colours of the noise rows.
    void paint_noise_green(core::Tensor& noise_sh0);

    // The share of the infill rows [0, `noise_rows`) of `model`, a copy it repaints, that shows through its surface on
    // `cameras`: renders with the surface black and the infill green, against renders of the infill alone. None if no view
    // sees the infill or `keep_going`, asked before each view, stops it.
    [[nodiscard]] std::optional<double> surface_see_through(core::SplatData& model, std::size_t noise_rows, const CameraDataset& cameras,
                                                            bool mip_filter, const std::function<bool()>& keep_going);

} // namespace lfs::training
