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

    struct Sam2Prediction {
        Tensor masks;          // [num_masks, H, W] fp32 logits, original image size
        Tensor iou;            // [num_masks] fp32 predicted mask quality
        Tensor low_res_logits; // [num_masks, 256, 256] fp32
    };

    struct Sam2PointPrompt {
        float x;
        float y;
        int label;
    }; // px in original image, label 1=fg 0=bg

    // Named activations matching tests/data/nn/sam2_ref_fixture.json.
    struct Sam2Taps {
        Tensor patch_embed;
        std::array<Tensor, 24> blocks{};
        std::array<Tensor, 4> trunk_stage{};
        std::array<Tensor, 3> fpn{};
        Tensor high_res_feats0;
        Tensor high_res_feats1;
        Tensor image_embed;
        Tensor dense_pe;
        Tensor sparse_embeddings;
        Tensor dense_embeddings;
        Tensor transformer_tokens;
        Tensor upscaled_embedding;
        Tensor low_res_logits;
        Tensor iou_predictions;
        Tensor object_score_logits;
    };

    class LFS_CORE_API Sam2 {
    public:
        static lfs::Result<Sam2> load(const std::filesystem::path& weights, Device device,
                                      std::optional<DataType> compute = std::nullopt);

        // NCHW RGB in [0, 1], any H/W. Resizes to 1024x1024 internally (squash, like the
        // official predictor), normalizes, runs the image encoder, caches embeddings.
        lfs::Result<void> set_image(const Tensor& image);

        lfs::Result<Sam2Taps> set_image_with_taps(const Tensor& image);

        // Points and/or box in original-image pixel coords. multimask=true returns 3
        // candidate masks, false returns 1. Requires set_image first.
        lfs::Result<Sam2Prediction> predict(const std::vector<Sam2PointPrompt>& points,
                                            const std::optional<std::array<float, 4>>& box,
                                            bool multimask = true);

        lfs::Result<std::pair<Sam2Prediction, Sam2Taps>>
        predict_with_taps(const std::vector<Sam2PointPrompt>& points,
                          const std::optional<std::array<float, 4>>& box, bool multimask = true);

        [[nodiscard]] DataType compute_dtype() const { return compute_; }
        [[nodiscard]] Device device() const { return device_; }
        [[nodiscard]] std::size_t weights_bytes() const;
        [[nodiscard]] std::size_t arena_bytes() const { return arena_.capacity(); }
        [[nodiscard]] std::size_t workspace_bytes() const {
            return workspace_.is_valid() ? workspace_.bytes() : 0;
        }
        [[nodiscard]] bool has_image() const { return image_set_; }
        [[nodiscard]] int orig_height() const { return orig_h_; }
        [[nodiscard]] int orig_width() const { return orig_w_; }

    private:
        lfs::Result<void> run_encoder(const Tensor& image, Sam2Taps* taps);
        lfs::Result<Sam2Prediction> run_decoder(const std::vector<Sam2PointPrompt>& points,
                                                const std::optional<std::array<float, 4>>& box,
                                                bool multimask, Sam2Taps* taps);

        const Tensor& w(std::string_view name) const;
        Tensor conv1x1(const Tensor& x, std::string_view weight, std::string_view bias);
        Tensor conv_k(const Tensor& x, std::string_view weight, std::string_view bias,
                      const Conv2dParams& params);
        Tensor conv_transpose2x(const Tensor& x, std::string_view weight, std::string_view bias);
        Tensor lin(const Tensor& x, std::string_view weight, std::string_view bias,
                   Activation activation = Activation::None, const Tensor* residual = nullptr);
        Tensor pool_bhwc(const Tensor& x);
        Tensor hiera_block(const Tensor& x, int index);
        Tensor hiera_pos_embed(int token_h, int token_w, cudaStream_t stream);
        Tensor two_way_attn(const Tensor& q, const Tensor& k, const Tensor& v,
                            std::string_view prefix);
        std::pair<Tensor, Tensor> two_way_block(const Tensor& queries, const Tensor& keys,
                                                const Tensor& query_pe, const Tensor& key_pe,
                                                int layer);
        std::pair<Tensor, Tensor> two_way_transformer(const Tensor& image, const Tensor& image_pe,
                                                      const Tensor& tokens);
        Tensor mlp(const Tensor& x, std::string_view prefix, int n_layers, Activation hidden_act,
                   bool sigmoid_out);
        Tensor encode_points(const std::vector<Sam2PointPrompt>& points,
                             const std::optional<std::array<float, 4>>& box);
        Tensor ensure_workspace(std::size_t bytes, const Tensor& like);
        void capture_tap(Tensor& slot, const Tensor& src);

        std::unordered_map<std::string, Tensor> weights_;
        Tensor workspace_;
        Tensor image_embed_hold_;
        Tensor high_res0_hold_;
        Tensor high_res1_hold_;
        Tensor token_hold_;
        std::array<Tensor, 4> stage_hold_{};
        Tensor dense_pe_;
        Tensor pos_embed_;
        int pos_embed_h_ = 0;
        int pos_embed_w_ = 0;
        Tensor mean_;
        Tensor std_;
        Sam2Taps encoder_taps_;
        bool encoder_taps_valid_ = false;
        ActivationArena arena_;
        bool weights_on_stream_ = false;
        bool mempool_trimmed_ = false;
        bool image_set_ = false;
        int orig_h_ = 0;
        int orig_w_ = 0;
        Device device_ = Device::CUDA;
        DataType compute_ = DataType::Float16;
    };

} // namespace lfs::core::nn::models
