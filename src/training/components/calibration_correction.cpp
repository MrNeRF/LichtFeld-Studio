// SPDX-License-Identifier: GPL-3.0-or-later
#include "calibration_correction.hpp"
#include "core/assert.hpp"
#include "core/logger.hpp"
#include "core/nn/ops.hpp"
#include "core/tensor_backend.hpp"
#include "lfs/training/ops/registry.hpp"
#include <algorithm>
#include <cmath>
#include <numbers>

namespace lfs::training {
    using namespace core;

    CalibrationCorrection::CalibrationCorrection(int slots)
        : CalibrationCorrection(slots, training_step) {}

    CalibrationCorrection::CalibrationCorrection(int slots, float step)
        : ops_(training_ops(default_gpu_backend()).calibration), step_(step), field_(Tensor::zeros({static_cast<size_t>(slots), size_t(grid), size_t(grid), 2}, Device::GPU)) {
        if (!ops_)
            throw std::runtime_error("Calibration correction is unavailable on the selected device");
    }

    CalibrationCorrection::Lens CalibrationCorrection::lens(CameraModelType model, float fx, float fy, int width, int height) {
        // An equirectangular camera carries only a placeholder focal length.
        if (model == CameraModelType::EQUIRECTANGULAR)
            return {static_cast<float>(width / (2 * std::numbers::pi)), static_cast<float>(height / std::numbers::pi), true};
        return {fx, fy, false};
    }

    bool CalibrationCorrection::report_due(int iteration, int total_iterations, int start_iteration) {
        return iteration >= start_iteration &&
               (iteration == total_iterations || iteration % std::max(1, total_iterations / 15) == 0);
    }

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

        CalibrationCorrection fit(1, scoring_step);
        Tensor unused_mask, structure, edge, no_gradient;
        auto sample = [&](const Tensor& source, float scale) {
            return fit.sample_image(0, scale, scale, 0, 0, source, source, unused_mask, structure, edge, true);
        };
        for (int level = static_cast<int>(levels.size()) - 1; level >= 0; --level) {
            const auto& images = levels[level];
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
    namespace {
        void reserve(Tensor& buffer, TensorShape shape, DataType dtype = DataType::Float32) {
            if (!buffer.is_valid() || buffer.shape() != shape || buffer.dtype() != dtype)
                buffer = Tensor::empty(shape, Device::GPU, dtype);
        }
        int chunks_for(CalibrationCorrection::Tile t) {
            const int cells = t.side - 1;
            return (((t.width + cells - 1) / cells) * ((t.height + cells - 1) / cells) + 511) / 512;
        }
    } // namespace

    Tensor CalibrationCorrection::warp(int slot, float fx, float fy, int x, int y, const Tensor& photo, const Tensor& render,
                                       Tensor& mask, Tensor& structure, Tensor& edge, bool first_tile, const Composite* composite,
                                       bool wrap) {
        return sample_image(slot, fx * .001f, fy * .001f, x, y, photo, render, mask, structure, edge, first_tile, composite, wrap);
    }

    Tensor CalibrationCorrection::sample_image(int slot, float sx, float sy, int x, int y, const Tensor& photo, const Tensor& render,
                                               Tensor& mask, Tensor& structure, Tensor& edge, bool first_tile,
                                               const Composite* composite, bool wrap) {
        const auto w = render.shape()[2], h = render.shape()[1];
        tile_ = {int(photo.shape()[2]), int(photo.shape()[1]), x, y, int(w), int(h), int(field_.shape()[1]), sx, sy, wrap};
        slot_ = slot;
        first_tile_ = first_tile;
        LFS_ASSERT(slot >= 0 && size_t(slot) < field_.shape()[0] && x >= 0 && y >= 0 &&
                   x + w <= photo.shape()[2] && y + h <= photo.shape()[1] && photo.is_contiguous() && render.is_contiguous());
        reserve(image_, {3, h, w});
        if (mask.is_valid())
            reserve(mask_, {h, w}, mask.dtype());
        if (structure.is_valid())
            reserve(structure_, {h, w});
        if (edge.is_valid())
            reserve(edge_, {h, w});
        if (composite) {
            LFS_ASSERT(composite->alpha->is_contiguous() && composite->alpha->numel() * 3 == photo.numel() &&
                       (composite->background->numel() == 3 || composite->background->numel() == photo.numel()));
        }
        ops_->sample(tile_, field_, slot, photo, render, image_, {mask, structure, edge, mask_, structure_, edge_, composite ? composite->alpha : nullptr, composite ? composite->background : nullptr, composite ? composite->stable_background : nullptr, composite ? composite->stable : nullptr});
        if (mask.is_valid())
            mask = mask_;
        if (structure.is_valid())
            structure = structure_;
        if (edge.is_valid())
            edge = edge_;
        return image_;
    }

    void CalibrationCorrection::accumulate(const Tensor& render, const Tensor& gradient, const Tensor& weight,
                                           float values, int margin) {
        const int cells = tile_.side - 1, chunks = chunks_for(tile_);
        reserve(partials_, {size_t(cells * cells), size_t(chunks), 21});
        ops_->accumulate(tile_, field_, slot_, image_, render, gradient, weight, values, margin, chunks, first_tile_, partials_);
    }

    void CalibrationCorrection::clear_outside(Tensor& gradient) const {
        LFS_ASSERT(gradient.is_contiguous() && gradient.dtype() == DataType::Float32 && gradient.numel() == size_t(3) * tile_.w * tile_.h);
        ops_->clear_outside(tile_, field_, slot_, gradient);
    }

    Tensor CalibrationCorrection::follow(const Tensor& prior, bool normal) const {
        const int channels = normal ? 3 : 1;
        const auto& shape = prior.shape();
        LFS_ASSERT(prior.device() == Device::GPU && prior.dtype() == DataType::Float32 && prior.is_contiguous() &&
                   shape.rank() >= 2 && prior.numel() == size_t(channels) * shape[shape.rank() - 1] * shape[shape.rank() - 2]);
        const int w = int(shape[shape.rank() - 1]), h = int(shape[shape.rank() - 2]);
        // The field is read at the prior's own pixel centres, its displacement scaled to the prior's pixels.
        const Tile t{w, h, 0, 0, w, h, tile_.side, tile_.sx * w / tile_.width, tile_.sy * h / tile_.height, tile_.wrap};
        auto output = Tensor::empty(shape, Device::GPU);
        ops_->follow(t, field_, slot_, prior, output, channels);
        return output;
    }

    void CalibrationCorrection::update() {
        ops_->update(tile_, field_, slot_, partials_, chunks_for(tile_), step_);
    }
} // namespace lfs::training
