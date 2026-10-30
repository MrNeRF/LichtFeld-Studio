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
#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace lfs::core::nn::models {

    struct Sam2Prediction {
        Tensor masks;          // [num_masks, H, W] fp32 logits, original image size
        Tensor iou;            // [num_masks] fp32 predicted mask quality
        Tensor low_res_logits; // [num_masks, 256, 256] fp32
        Tensor object_pointer;
        Tensor object_score_logits;
    };

    struct Sam2PointPrompt {
        float x;
        float y;
        int label;
    }; // px in original image, label 1=fg 0=bg

    inline constexpr float kSam2MinIou = 0.5f;
    inline constexpr float kSam2MaxArea = 0.4f;

    class LFS_CORE_API Sam2 {
    public:
        static lfs::Result<Sam2> load(const std::filesystem::path& weights, Device device,
                                      std::optional<DataType> compute = std::nullopt);

        // NCHW RGB in [0, 1], any H/W. Resizes to 1024x1024 internally (squash, like the
        // official predictor), normalizes, runs the image encoder, caches embeddings.
        lfs::Result<void> set_image(const Tensor& image);

        // Points and/or box in original-image pixel coords. multimask=true returns 3
        // candidate masks, false returns 1. mask_logits, as the official mask_input, is a previous low-resolution
        // prediction (1 x 1 x 256 x 256) to refine. Requires set_image first.
        lfs::Result<Sam2Prediction> predict(const std::vector<Sam2PointPrompt>& points,
                                            const std::optional<std::array<float, 4>>& box,
                                            bool multimask = true, const Tensor* mask_logits = nullptr);

        // One object and one automatic seed per sequence. A reverse pass starts a fresh bank from that seed.
        lfs::Result<void> set_video_image(const Tensor& image);
        lfs::Result<Sam2Prediction> seed_video(const Tensor& mask, int frame_index, int frame_count,
                                               bool reverse);
        lfs::Result<Sam2Prediction> track_video(int frame_index);
        void reset_video();
        [[nodiscard]] bool supports_video() const;

        // Runs the image encoder in `dtype` (tensor cores for Float16) while prompts, the mask decoder and video
        // tracking keep the compute dtype; the encoder's features are converted on the way out.
        void set_encoder_dtype(DataType dtype);

    private:
        lfs::Result<void> run_encoder(const Tensor& image, bool video = false);
        lfs::Result<Sam2Prediction> run_decoder(const std::vector<Sam2PointPrompt>& points,
                                                const std::optional<std::array<float, 4>>& box,
                                                bool multimask, bool video = false, const Tensor* mask_input = nullptr);

        const Tensor& w(std::string_view name) const;
        Tensor conv1x1(const Tensor& x, std::string_view weight, std::string_view bias);
        Tensor conv_k(const Tensor& x, std::string_view weight, std::string_view bias,
                      const Conv2dParams& params);
        Tensor conv_transpose2x(const Tensor& x, std::string_view weight, std::string_view bias);
        Tensor lin(const Tensor& x, std::string_view weight, std::string_view bias,
                   Activation activation = Activation::None, const Tensor* residual = nullptr);
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

        struct VideoMemory {
            Tensor features;
            Tensor pointer;
            bool reliable = true;
        };
        std::vector<int> reliable_memories(int frame_index) const;
        int choose_video_candidate(const Tensor& low_res, const std::vector<float>& iou);
        Tensor video_memory_attention(int frame_index);
        Tensor video_memory_encoder(const Tensor& high_res_mask, bool binary, bool object_present);
        Tensor video_attention(const Tensor& query, const Tensor& key, const Tensor& value,
                               std::string_view prefix, int pointer_tokens);
        Sam2Prediction video_output(Tensor low_res, Tensor pointer, Tensor object_score,
                                    Tensor iou);
        std::map<int, VideoMemory> video_bank_;
        int video_seed_ = -1, video_last_ = -1, video_frame_count_ = 0;
        bool video_reverse_ = false;
        bool video_image_ = false;
        bool video_oversized_ = false;
        Tensor raw_image_embed_;
        Tensor video_image_position_;
        Tensor video_memory_position_;
        std::unordered_map<std::string, Tensor> weights_;
        std::unordered_map<std::string, Tensor> encoder_weights_; // image_encoder.* in encoder_compute_, when it differs
        bool encoding_ = false;
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
        ActivationArena arena_;
        bool weights_on_stream_ = false;
        bool mempool_trimmed_ = false;
        bool image_set_ = false;
        int orig_h_ = 0;
        int orig_w_ = 0;
        Device device_ = Device::CUDA;
        DataType compute_ = DataType::Float16;
        DataType encoder_compute_ = DataType::Float16;
    };

} // namespace lfs::core::nn::models
