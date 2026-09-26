/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/error.hpp"
#include "core/export.hpp"
#include "core/nn/ops.hpp"
#include "core/nn/weight_file.hpp"
#include "core/tensor.hpp"

#include <array>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace lfs::core::nn::models {

    // Dense warp from image A to image B plus the per-pixel overlap
    // probability. warp is [H, W, 2] float32 in normalized [-1, 1] coordinates
    // of image B; overlap is [H, W] float32 in [0, 1].
    struct RomaMatch {
        Tensor warp;
        Tensor overlap;
    };

    // Per-image state: the projected feature pyramid, which is everything that
    // depends on one image alone.
    struct LFS_CORE_API RomaV1Image {
        Tensor image;               // [1, 3, R, R] float32 in [0, 1]
        std::vector<Tensor> levels; // channel-last [pixels, dim] for scales 16, 8, 4, 2, 1
        // Grid size of each level. The pyramid calls the coarse level "16", but
        // it comes from a patch-14 backbone, so it is R/14 rather than R/16.
        std::vector<std::pair<int, int>> sizes;
    };

    // RoMa v1 dense matcher: a DINOv2 ViT-L/14 backbone and a VGG19-BN stem, a
    // Gaussian-process matcher over the coarse features, a transformer that
    // classifies each coarse pixel onto a grid of candidate correspondences, and
    // five convolutional refiners. RoMa v1 is MIT and DINOv2 is Apache-2.0, so
    // the whole model is distributable.
    class LFS_CORE_API RomaV1 {
    public:
        // `resolution` must be one of the resolutions the weight file baked a
        // position embedding for, and a multiple of 112 (the patch size times
        // the coarsest stride).
        static lfs::Result<RomaV1> load(const std::filesystem::path& weights, Device device,
                                        std::optional<DataType> compute = std::nullopt,
                                        int resolution = 448);

        // `image` is [1, 3, H, W] or [3, H, W] float, or [H, W, 3] uint8/float,
        // on the GPU at any resolution. A dataset camera's loaded image is the
        // [3, H, W] form.
        [[nodiscard]] lfs::Result<std::shared_ptr<RomaV1Image>> prepare(const Tensor& image);
        [[nodiscard]] lfs::Result<RomaMatch> match(const RomaV1Image& a, const RomaV1Image& b);

        // The layout densification consumes: the warp comes back [H, W, 4], the
        // reference pixel grid in channels 0 and 1 and the match in image B in
        // channels 2 and 3, so a row reads (x_a, y_a, x_b, y_b).
        [[nodiscard]] lfs::Result<RomaMatch> match_with_grid(const RomaV1Image& a,
                                                             const RomaV1Image& b);

        [[nodiscard]] DataType compute_dtype() const { return compute_; }
        [[nodiscard]] Device device() const { return device_; }
        [[nodiscard]] int resolution() const { return resolution_; }
        [[nodiscard]] std::size_t weights_bytes() const;
        [[nodiscard]] static constexpr int default_resolution() { return 448; }

    private:
        static constexpr int kPatch = 14;        // DINOv2 ViT-L/14
        static constexpr int kCoarseStride = 16; // the pyramid calls that level "16"
        static constexpr int kDinoBlocks = 24;
        static constexpr int kDecoderBlocks = 5;
        static constexpr int kRefinerBlocks = 9;
        static constexpr int kTrainResolution = 560;
        static constexpr std::array<int, 5> kScales{16, 8, 4, 2, 1};

        const Tensor& w(std::string_view name) const;
        Tensor ensure_workspace(std::size_t bytes, const Tensor& like);

        Tensor vit_block(const Tensor& x, const std::string& prefix, int heads, bool layer_scale,
                         bool qkv_bias);
        Tensor dino(const Tensor& normalized);
        std::vector<Tensor> vgg(const Tensor& normalized);
        Tensor gaussian_process(const Tensor& f_a, const Tensor& f_b, int height, int width);
        Tensor coarse_decoder(const Tensor& gp_feats, const Tensor& f_a);
        Tensor refiner(int scale, const Tensor& f_a, const Tensor& f_b, const Tensor& flow,
                       int height, int width);
        [[nodiscard]] int level_extent(int scale) const {
            return scale == kCoarseStride ? resolution_ / kPatch : resolution_ / scale;
        }

        std::unordered_map<std::string, Tensor> weights_;
        std::unordered_map<std::string, Tensor> weight_taps_;
        Tensor workspace_;
        cudaStream_t stream_ = nullptr;
        bool weights_on_stream_ = false;
        Device device_ = Device::GPU;
        DataType compute_ = DataType::Float16;
        int resolution_ = 448;
        int classes_side_ = 64;
        float temperature_ = 0.2f;
        float sigma_noise_ = 0.1f;
        // Twenty-four iterations put the solve an order of magnitude inside the
        // float16 noise of everything downstream; more is wasted work.
        int cg_iterations_ = 24;
    };

} // namespace lfs::core::nn::models
