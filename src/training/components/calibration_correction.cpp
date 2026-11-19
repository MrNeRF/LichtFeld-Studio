// SPDX-License-Identifier: GPL-3.0-or-later
#include "calibration_correction.hpp"
#include "core/logger.hpp"
#include "core/nn/ops.hpp"
#include "core/tensor/internal/cuda_stream_context.hpp"
#include <algorithm>
#include <cmath>

namespace lfs::training {
    using namespace core;

    CalibrationCorrection::CalibrationCorrection(int slots)
        : CalibrationCorrection(slots, grid) {}

    CalibrationCorrection::CalibrationCorrection(int slots, int side)
        : field_(Tensor::zeros({static_cast<size_t>(slots), size_t(side), size_t(side), 2}, Device::CUDA)) {}

    Tensor CalibrationCorrection::align_for_scoring(const Tensor& render, const Tensor& photo, const Tensor& mask,
                                                    Tensor& displacement, Tensor* raw) {
        struct Level {
            Tensor render, photo, mask;
        };
        // A brightness difference must not be explained by a shift along a weak gradient,
        // so the fit registers against the photograph lifted to the render's mean level.
        Tensor offset = render - photo;
        if (mask.is_valid()) {
            const Tensor included = mask.gt(0.f).to(DataType::Float32);
            const float share = included.mean().item();
            offset = share > 0.f ? (offset * included.unsqueeze(0)).mean({1, 2}, true) / share : Tensor{};
        } else {
            offset = offset.mean({1, 2}, true);
        }
        std::vector<Level> levels{{render.contiguous(), (offset.is_valid() ? photo + offset : photo).contiguous(),
                                   mask.is_valid() ? mask.contiguous() : Tensor{}}};
        const auto half = [](const Tensor& image) {
            return nn::avg_pool2d(image.unsqueeze(0), 2, 2, 2, 2, 0, 0).squeeze(0);
        };
        while (levels.size() < 4) {
            const auto& level = levels.back();
            if (std::min(level.render.shape()[1], level.render.shape()[2]) / 2 < 48)
                break;
            // Only fully included boxes can contribute a coarse registration residual.
            Tensor included;
            if (level.mask.is_valid())
                included = half(level.mask.gt(0.f).to(DataType::Float32).unsqueeze(0)).squeeze(0).ge(1.f);
            levels.push_back({half(level.render), half(level.photo), std::move(included)});
        }

        CalibrationCorrection fit(1, scoring_grid);
        Tensor unused_mask, structure, edge, no_gradient;
        auto sample = [&](const Tensor& source, float scale) {
            return fit.sample_image(0, scale, scale, 0, 0, source, source, unused_mask, structure, edge, true);
        };
        for (int level = static_cast<int>(levels.size()) - 1; level >= 0; --level) {
            const auto& images = levels[level];
            images.photo.sync_to_stream(getCurrentCUDAStream());
            if (images.mask.is_valid())
                images.mask.sync_to_stream(getCurrentCUDAStream());
            // The field stays in full-resolution pixels across all pyramid levels.
            for (int iteration = 0; iteration < 10; ++iteration) {
                sample(images.render, std::ldexp(1.f, -level));
                fit.accumulate(images.photo, no_gradient, images.mask, 0.f);
                fit.update();
            }
        }
        displacement = fit.field_;
        if (raw && raw->is_valid()) {
            sample(raw->contiguous(), 1.f);
            *raw = std::move(fit.image_);
        }
        return sample(levels.front().render, 1.f);
    }

    void CalibrationCorrection::report(int iteration) const {
        const auto field = field_.to_vector();
        std::vector<float> lengths;
        size_t moved = 0;
        constexpr int stride = 2 * grid * grid;
        for (size_t row = 0; row < field.size(); row += stride) {
            if (std::all_of(field.begin() + row, field.begin() + row + stride, [](float v) { return v == 0; }))
                continue;
            ++moved;
            for (size_t i = row; i < row + stride; i += 2)
                lengths.push_back(std::hypot(field[i], field[i + 1]));
        }
        if (lengths.empty())
            lengths.push_back(0);
        std::sort(lengths.begin(), lengths.end());
        LOG_INFO("Calibration correction at {}: {} rows moved, |u| median {:.4f}, p95 {:.4f}, max {:.4f} mrad",
                 iteration, moved, lengths[(lengths.size() - 1) / 2], lengths[(lengths.size() - 1) * 95 / 100], lengths.back());
    }
} // namespace lfs::training
