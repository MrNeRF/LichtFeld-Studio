/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/nn/models/sam2.hpp"

#include "core/assert.hpp"
#include "core/source_site.hpp"
#include "internal/cuda_stream_context.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <numbers>

namespace lfs::core::nn::models {
    namespace {
        constexpr int kGrid = 64, kImage = 1024, kMemoryFrames = 7, kPointerFrames = 16;

        Error video_error(std::string detail) {
            return make_error({.code = ErrorCode::FailedPrecondition,
                               .domain = ErrorDomain::Core,
                               .user_message = "SAM2 video inference failed",
                               .detail = std::move(detail),
                               .detection = LFS_SOURCE_SITE_CURRENT()});
        }

        Tensor sequence(const Tensor& image) {
            return image.reshape(TensorShape{std::vector<size_t>{1, image.size(1), image.size(2) * image.size(3)}})
                .permute({0, 2, 1})
                .contiguous();
        }

        Tensor sine_grid(int channels, Device device) {
            std::vector<float> values(size_t(channels) * kGrid * kGrid);
            const int half = channels / 2;
            for (int c = 0; c < channels; ++c) {
                const int frequency = c % half;
                const float divisor = std::pow(10000.0f, float(2 * (frequency / 2)) / half);
                for (int y = 0; y < kGrid; ++y)
                    for (int x = 0; x < kGrid; ++x) {
                        const float position = float((c < half ? y : x) + 1) / (float(kGrid) + 1e-6f);
                        const float angle = position * (2 * std::numbers::pi_v<float>) / divisor;
                        values[size_t(c * kGrid + y) * kGrid + x] =
                            frequency % 2 == 0 ? std::sin(angle) : std::cos(angle);
                    }
            }
            return Tensor::from_vector(values, {1, size_t(channels), kGrid, kGrid}, Device::CPU)
                .to(device, getCurrentCUDAStream());
        }

        // Positive share of each [count, 256, 256] candidate mask.
        std::vector<float> candidate_areas(const Tensor& logits) {
            auto areas = logits.gt(0.0f).to(DataType::Float32).sum({1, 2}).to(Device::CPU).contiguous().to_vector();
            for (auto& area : areas)
                area /= float(logits.size(1) * logits.size(2));
            return areas;
        }
    } // namespace

    bool Sam2::supports_video() const {
        return weights_.contains("memory_encoder.out_proj.weight") &&
               weights_.contains("memory_attention.layers.3.cross_attn_image.q_proj.weight") &&
               weights_.contains("obj_ptr_tpos_proj.weight") && weights_.contains("maskmem_tpos_enc");
    }

    void Sam2::reset_video() {
        video_bank_.clear();
        video_seed_ = video_last_ = -1;
        video_frame_count_ = 0;
    }

    lfs::Result<void> Sam2::set_video_image(const Tensor& image) {
        if (!supports_video())
            return lfs::Result<void>::failure(video_error("SAM2.1 video weights are required"));
        if (compute_ != DataType::Float32)
            return lfs::Result<void>::failure(video_error("Video propagation requires Float32 compute"));
        auto encoded = run_encoder(image, true);
        if (!encoded)
            return encoded;
        const CUDAStreamGuard guard(image.stream());
        if (!video_image_position_.is_valid()) {
            video_image_position_ = sequence(sine_grid(256, device_));
            video_memory_position_ = sequence(sine_grid(64, device_));
        }
        return {};
    }

    Tensor Sam2::video_attention(const Tensor& query, const Tensor& key, const Tensor& value,
                                 std::string_view prefix, int pointer_tokens) {
        auto q = lin(query, std::format("{}.q_proj.weight", prefix), std::format("{}.q_proj.bias", prefix));
        auto k = lin(key, std::format("{}.k_proj.weight", prefix), std::format("{}.k_proj.bias", prefix));
        auto v = lin(value, std::format("{}.v_proj.weight", prefix), std::format("{}.v_proj.bias", prefix));
        q = rotary_embedding_2d(q.unsqueeze(1), kGrid, kGrid);
        k = rotary_embedding_2d(k.unsqueeze(1), kGrid, kGrid, pointer_tokens);
        auto attended = attention(q, k, v.unsqueeze(1)).squeeze(1);
        return lin(attended, std::format("{}.out_proj.weight", prefix), std::format("{}.out_proj.bias", prefix));
    }

    int Sam2::choose_video_candidate(const Tensor& low_res, const std::vector<float>& iou) {
        const auto areas = candidate_areas(low_res.squeeze(0).to(DataType::Float32));
        int best = -1;
        for (int candidate = 1; candidate < int(iou.size()); ++candidate)
            if (areas[size_t(candidate)] <= kSam2MaxArea && (best < 0 || iou[size_t(candidate)] > iou[size_t(best)]))
                best = candidate;
        video_oversized_ = best < 0;
        if (video_oversized_)
            best = int(std::min_element(areas.begin() + 1, areas.end()) - areas.begin());
        return best;
    }

    std::vector<int> Sam2::reliable_memories(int frame_index) const {
        std::vector<int> recent;
        const auto take = [&](const auto& entry) {
            if (entry.first != video_seed_ && entry.first != frame_index && entry.second.reliable &&
                recent.size() < size_t(kMemoryFrames - 1))
                recent.push_back(entry.first);
        };
        if (video_reverse_)
            for (auto entry = video_bank_.begin(); entry != video_bank_.end(); ++entry)
                take(*entry);
        else
            for (auto entry = video_bank_.rbegin(); entry != video_bank_.rend(); ++entry)
                take(*entry);
        return recent;
    }

    Tensor Sam2::video_memory_attention(int frame_index) {
        std::vector<Tensor> memories, positions;
        const auto append_memory = [&](int index, int temporal_position) {
            const auto found = video_bank_.find(index);
            if (found == video_bank_.end())
                return;
            memories.push_back(sequence(found->second.features));
            positions.push_back(video_memory_position_.add(
                w("maskmem_tpos_enc").slice(0, temporal_position, temporal_position + 1).reshape(TensorShape{std::vector<size_t>{1, 1, 64}})));
        };
        append_memory(video_seed_, kMemoryFrames - 1);
        const int direction = video_reverse_ ? -1 : 1;
        const auto recent = reliable_memories(frame_index);
        for (int rank = int(recent.size()); rank >= 1; --rank)
            append_memory(recent[size_t(rank - 1)], rank - 1);
        std::vector<int> pointer_indices{video_seed_};
        const int maximum_pointers = std::min(video_frame_count_, kPointerFrames);
        for (int distance = 1; distance < maximum_pointers; ++distance) {
            const int index = frame_index - direction * distance;
            if (index != video_seed_ && video_bank_.contains(index))
                pointer_indices.push_back(index);
        }
        std::vector<Tensor> pointers;
        std::vector<float> temporal_values(pointer_indices.size() * 256);
        for (size_t i = 0; i < pointer_indices.size(); ++i) {
            pointers.push_back(video_bank_.at(pointer_indices[i]).pointer.reshape(TensorShape{std::vector<size_t>{1, 4, 64}}));
            const float distance = float((frame_index - pointer_indices[i]) * direction) / (maximum_pointers - 1);
            for (int j = 0; j < 128; ++j) {
                const float angle = distance / std::pow(10000.0f, float(2 * (j / 2)) / 128);
                temporal_values[i * 256 + j] = std::sin(angle);
                temporal_values[i * 256 + 128 + j] = std::cos(angle);
            }
        }
        auto temporal = Tensor::from_vector(temporal_values, {pointer_indices.size(), 256}, Device::CPU)
                            .to(device_, raw_image_embed_.stream());
        temporal = lin(temporal, "obj_ptr_tpos_proj.weight", "obj_ptr_tpos_proj.bias");
        temporal = temporal.reshape(TensorShape{std::vector<size_t>{1, pointer_indices.size(), 1, 64}})
                       .expand(TensorShape{std::vector<size_t>{1, pointer_indices.size(), 4, 64}})
                       .contiguous()
                       .reshape(TensorShape{std::vector<size_t>{1, pointer_indices.size() * 4, 64}});
        memories.push_back(Tensor::cat(pointers, 1));
        positions.push_back(temporal);
        const auto memory = Tensor::cat(memories, 1);
        const auto memory_position = Tensor::cat(positions, 1);
        auto output = sequence(raw_image_embed_).add(video_image_position_.mul(0.1f));
        const int pointer_tokens = int(pointer_indices.size() * 4);
        for (int layer = 0; layer < 4; ++layer) {
            const auto prefix = std::format("memory_attention.layers.{}", layer);
            auto normalized = layer_norm(output, w(prefix + ".norm1.weight"), w(prefix + ".norm1.bias"));
            output = output.add(video_attention(normalized, normalized, normalized, prefix + ".self_attn", 0));
            normalized = layer_norm(output, w(prefix + ".norm2.weight"), w(prefix + ".norm2.bias"));
            output = output.add(video_attention(normalized, memory.add(memory_position), memory,
                                                prefix + ".cross_attn_image", pointer_tokens));
            normalized = layer_norm(output, w(prefix + ".norm3.weight"), w(prefix + ".norm3.bias"));
            auto hidden = lin(normalized, prefix + ".linear1.weight", prefix + ".linear1.bias", Activation::Relu);
            hidden = lin(hidden, prefix + ".linear2.weight", prefix + ".linear2.bias");
            output = output.add(hidden);
        }
        output = layer_norm(output, w("memory_attention.norm.weight"), w("memory_attention.norm.bias"));
        return output.permute({0, 2, 1}).contiguous().reshape(TensorShape{std::vector<size_t>{1, 256, kGrid, kGrid}});
    }

    Tensor Sam2::video_memory_encoder(const Tensor& high_res_mask, bool binary, bool object_present) {
        auto mask = binary ? high_res_mask.gt(0).to(DataType::Float32) : sigmoid(high_res_mask);
        mask = mask.mul(20.0f).sub(10.0f);
        Conv2dParams downsample;
        downsample.stride_h = downsample.stride_w = 2;
        downsample.pad_h = downsample.pad_w = 1;
        for (int layer = 0; layer < 4; ++layer) {
            const auto prefix = std::format("memory_encoder.mask_downsampler.encoder.{}", layer * 3);
            mask = conv_k(mask, prefix + ".weight", prefix + ".bias", downsample);
            const auto norm = std::format("memory_encoder.mask_downsampler.encoder.{}", layer * 3 + 1);
            mask = layer_norm_2d(mask, w(norm + ".weight"), w(norm + ".bias"));
            mask = gelu(mask);
        }
        mask = conv1x1(mask, "memory_encoder.mask_downsampler.encoder.12.weight",
                       "memory_encoder.mask_downsampler.encoder.12.bias");
        auto pixels = conv1x1(raw_image_embed_, "memory_encoder.pix_feat_proj.weight", "memory_encoder.pix_feat_proj.bias");
        auto output = pixels.add(mask);
        for (int layer = 0; layer < 2; ++layer) {
            const auto prefix = std::format("memory_encoder.fuser.layers.{}", layer);
            Conv2dParams depthwise;
            depthwise.pad_h = depthwise.pad_w = 3;
            depthwise.groups = 256;
            auto hidden = conv_k(output, prefix + ".dwconv.weight", prefix + ".dwconv.bias", depthwise);
            hidden = layer_norm_2d(hidden, w(prefix + ".norm.weight"), w(prefix + ".norm.bias"));
            hidden = hidden.permute({0, 2, 3, 1}).contiguous();
            hidden = lin(hidden, prefix + ".pwconv1.weight", prefix + ".pwconv1.bias");
            hidden = lin(gelu(hidden), prefix + ".pwconv2.weight", prefix + ".pwconv2.bias");
            hidden = hidden.mul(w(prefix + ".gamma"));
            output = output.add(hidden.permute({0, 3, 1, 2}).contiguous());
        }
        output = conv1x1(output, "memory_encoder.out_proj.weight", "memory_encoder.out_proj.bias");
        if (!object_present)
            output = output.add(w("no_obj_embed_spatial").reshape(TensorShape{std::vector<size_t>{1, 64, 1, 1}}));
        output = round_bfloat16(output);
        return output;
    }

    Sam2Prediction Sam2::video_output(Tensor low_res, Tensor pointer, Tensor object_score,
                                      Tensor iou) {
        low_res = fill_small_background(low_res.reshape(TensorShape{std::vector<size_t>{1, 1, 256, 256}}), 0.1f);
        low_res = remove_small_foreground(low_res, -0.1f);
        return {.masks = resize2d(low_res, orig_h_, orig_w_, ResizeMode::Bilinear, CoordTransform::HalfPixel).squeeze(0),
                .iou = std::move(iou),
                .low_res_logits = low_res.squeeze(0),
                .object_pointer = std::move(pointer),
                .object_score_logits = std::move(object_score)};
    }

    lfs::Result<Sam2Prediction> Sam2::seed_video(const Tensor& mask, int frame_index, int frame_count,
                                                 bool reverse) {
        if (!video_image_ || !image_set_ || !supports_video() || frame_count < 1 || frame_index < 0 || frame_index >= frame_count)
            return video_error("A video image and valid seed frame are required");
        if (mask.ndim() != 4 || mask.size(0) != 1 || mask.size(1) != 1)
            return video_error("The seed mask must have shape 1 by 1 by height by width");
        reset_video();
        const CUDAStreamGuard guard(raw_image_embed_.stream());
        video_seed_ = video_last_ = frame_index;
        video_frame_count_ = frame_count;
        video_reverse_ = reverse;
        auto seed = resize2d_bilinear_antialiased(mask.to(DataType::Float32), kImage, kImage).ge(0.5f).to(DataType::Float32);
        image_embed_hold_ = raw_image_embed_.clone();
        Conv2dParams downsample;
        downsample.stride_h = downsample.stride_w = 4;
        const auto prompt = conv_k(seed, "mask_downsample.weight", "mask_downsample.bias", downsample);
        auto decoded = run_decoder({}, std::nullopt, false, true, &prompt);
        if (!decoded)
            return decoded;
        const bool present = seed.sum().item<float>() > 0;
        auto pointer = present ? decoded->object_pointer : w("no_obj_ptr");
        auto score = Tensor::full({1, 1}, present ? 10.0f : -10.0f, device_);
        auto low = resize2d_bilinear_antialiased(seed.mul(20).sub(10), 256, 256);
        auto result = video_output(low, pointer, score, Tensor::ones({1}, device_));
        const auto high = resize2d(result.low_res_logits.unsqueeze(0), kImage, kImage,
                                   ResizeMode::Bilinear, CoordTransform::HalfPixel);
        auto memory = video_memory_encoder(high, true, present);
        video_bank_.emplace(frame_index, VideoMemory{std::move(memory), std::move(pointer)});
        return result;
    }

    lfs::Result<Sam2Prediction> Sam2::track_video(int frame_index) {
        const int direction = video_reverse_ ? -1 : 1;
        if (!video_image_ || !image_set_ || video_seed_ < 0 || frame_index != video_last_ + direction ||
            frame_index < 0 || frame_index >= video_frame_count_)
            return video_error("Video frames must follow the seed consecutively in the selected direction");
        const CUDAStreamGuard guard(raw_image_embed_.stream());
        image_embed_hold_ = video_memory_attention(frame_index);
        video_oversized_ = false;
        auto decoded = run_decoder({}, std::nullopt, true, true);
        if (!decoded)
            return decoded;
        const bool present = decoded->object_score_logits.item<float>() > 0;
        const bool reliable =
            present && !video_oversized_ && decoded->iou.to(DataType::Float32).item<float>() >= kSam2MinIou;
        const auto high = resize2d(decoded->low_res_logits.unsqueeze(0), kImage, kImage,
                                   ResizeMode::Bilinear, CoordTransform::HalfPixel);
        auto memory = video_memory_encoder(high, false, present);
        auto result = video_output(decoded->low_res_logits, decoded->object_pointer, decoded->object_score_logits,
                                   decoded->iou);
        video_bank_.emplace(frame_index, VideoMemory{std::move(memory), decoded->object_pointer, reliable});
        video_last_ = frame_index;
        const auto kept = reliable_memories(frame_index + direction);
        for (auto entry = video_bank_.begin(); entry != video_bank_.end();) {
            if (entry->first != video_seed_ && std::abs(frame_index - entry->first) >= kPointerFrames &&
                std::find(kept.begin(), kept.end(), entry->first) == kept.end())
                entry = video_bank_.erase(entry);
            else
                ++entry;
        }
        return result;
    }
} // namespace lfs::core::nn::models
