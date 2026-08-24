/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/error.hpp"
#include "core/export.hpp"
#include "core/nn/activation_arena.hpp"
#include "core/nn/ops.hpp"
#include "core/nn/weight_file.hpp"
#include "core/tensor.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace lfs::core::nn::models {

    struct Moge2Outputs {
        Tensor points;
        Tensor normal;
        Tensor mask;
        Tensor metric_scale;
    };

    // Named activations matching the ONNX graph. Used by parity tests.
    struct Moge2Taps {
        Tensor patch_embed;
        std::array<Tensor, 12> blocks{};
        Tensor encoder_feat;
        std::array<Tensor, 5> neck{};
        Tensor points_head;
        Tensor normal_head;
        Tensor mask_head;
    };

    // MoGe-2 ViT-B/14 as exported in moge-2-vitb-normal.onnx (opset 14).
    // Weights are the `.lfw` dump of that graph's initializers.
    class LFS_CORE_API Moge2 {
    public:
        static lfs::Result<Moge2> load(const std::filesystem::path& weights, Device device,
                                       std::optional<DataType> compute = std::nullopt);

        // `image` is NCHW RGB in [0, 1], matching the ORT `image` input.
        // Outputs match the ONNX heads after postprocess (points remapped with
        // exp, normals L2-normalized, mask sigmoid, metric_scale exp).
        lfs::Result<Moge2Outputs> forward(const Tensor& image, std::int64_t num_tokens);

        lfs::Result<std::pair<Moge2Outputs, Moge2Taps>>
        forward_with_taps(const Tensor& image, std::int64_t num_tokens);

        [[nodiscard]] DataType compute_dtype() const { return compute_; }
        [[nodiscard]] Device device() const { return device_; }
        [[nodiscard]] std::size_t weights_bytes() const;
        [[nodiscard]] std::size_t arena_bytes() const { return arena_.capacity(); }
        [[nodiscard]] std::size_t workspace_bytes() const {
            return workspace_.is_valid() ? workspace_.bytes() : 0;
        }

        static void token_grid(int image_h, int image_w, std::int64_t num_tokens, int& token_h,
                               int& token_w);

    private:
        lfs::Result<Moge2Outputs> run(const Tensor& image, std::int64_t num_tokens,
                                      Moge2Taps* taps);

        const Tensor& w(std::string_view name) const;
        Tensor conv1x1(const Tensor& x, std::string_view weight, std::string_view bias);
        Tensor conv3x3(const Tensor& x, std::string_view weight, std::string_view bias);
        Tensor conv_transpose2x(const Tensor& x, std::string_view weight, std::string_view bias);
        Tensor gemm_nn(const Tensor& x, std::string_view weight, std::string_view bias);
        Tensor res_block(const Tensor& x, std::string_view w1, std::string_view b1,
                         std::string_view w2, std::string_view b2);
        Tensor vit_block(const Tensor& x, int index);
        Tensor interpolate_pos_embed(int batch, int token_h, int token_w, DataType dtype,
                                     cudaStream_t stream);
        Tensor uv_map(int height, int width, float aspect, DataType dtype, Device device,
                      cudaStream_t stream) const;
        Tensor ensure_workspace(std::size_t bytes, const Tensor& like);

        std::unordered_map<std::string, Tensor> weights_;
        Tensor workspace_;
        Tensor feat_hold_;
        Tensor cls_hold_;
        Tensor head_hold_[3];
        ActivationArena arena_;
        bool weights_on_stream_ = false;
        bool mempool_trimmed_ = false;
        Device device_ = Device::CUDA;
        DataType compute_ = DataType::Float16;
    };

} // namespace lfs::core::nn::models
