/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "ppisp.hpp"
#include "config_serialization.hpp"
#include "core/assert.hpp"
#include "core/logger.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_execution.hpp"
#include "core/tensor_serialization.hpp"
#include "lfs/training/ops/registry.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace lfs::training {

    namespace {
        constexpr uint32_t CHECKPOINT_MAGIC = 0x4C465050; // "LFPP"
        constexpr uint32_t CHECKPOINT_MIN_VERSION = 2;
        constexpr uint32_t CHECKPOINT_VERSION = 4;
        constexpr uint32_t CONFIG_SCHEMA_VERSION = 2;
        constexpr uint32_t CONFIG_SCHEMA_V1_BYTES = 76;
        constexpr uint32_t CONFIG_SCHEMA_V2_BYTES = 80;

        struct LegacyConfigV2 {
            double lr;
            double beta1;
            double beta2;
            double eps;
            int warmup_steps;
            double warmup_start_factor;
            double final_lr_factor;
            float exposure_mean;
            float vig_center;
            float vig_channel;
            float vig_non_pos;
            float color_mean;
            float crf_channel;
        };

        void serialize_config(std::ostream& os, const PPISP::Config& config) {
            using config_serialization_detail::write_little_endian;

            // Schema v1 is append-only. Bump the schema version and append fields;
            // payload size lets older readers skip the suffix and retain defaults.
            write_little_endian(os, CONFIG_SCHEMA_VERSION, "PPISP config schema");
            write_little_endian(os, CONFIG_SCHEMA_V2_BYTES, "PPISP config size");
            write_little_endian(os, config.lr, "PPISP config lr");
            write_little_endian(os, config.beta1, "PPISP config beta1");
            write_little_endian(os, config.beta2, "PPISP config beta2");
            write_little_endian(os, config.eps, "PPISP config eps");
            write_little_endian(os, static_cast<int32_t>(config.warmup_steps), "PPISP config warmup_steps");
            write_little_endian(os, config.warmup_start_factor, "PPISP config warmup_start_factor");
            write_little_endian(os, config.final_lr_factor, "PPISP config final_lr_factor");
            write_little_endian(os, config.exposure_mean, "PPISP config exposure_mean");
            write_little_endian(os, config.vig_center, "PPISP config vig_center");
            write_little_endian(os, config.vig_channel, "PPISP config vig_channel");
            write_little_endian(os, config.vig_non_pos, "PPISP config vig_non_pos");
            write_little_endian(os, config.color_mean, "PPISP config color_mean");
            write_little_endian(os, config.crf_channel, "PPISP config crf_channel");
            write_little_endian(os, static_cast<int32_t>(config.train_crf ? 1 : 0), "PPISP config train_crf");
        }

        [[nodiscard]] PPISP::Config deserialize_config(std::istream& is) {
            using config_serialization_detail::read_little_endian;

            const uint32_t schema_version = read_little_endian<uint32_t>(is, "PPISP config schema");
            const uint32_t payload_bytes = read_little_endian<uint32_t>(is, "PPISP config size");
            if (schema_version == 0) {
                config_serialization_detail::throw_config_data_loss(
                    "PPISP config schema", "version must be positive");
            }
            if (payload_bytes < CONFIG_SCHEMA_V1_BYTES ||
                payload_bytes > config_serialization_detail::MAX_CONFIG_PAYLOAD_BYTES) {
                config_serialization_detail::throw_config_data_loss(
                    "PPISP config size", "payload size is out of bounds");
            }

            PPISP::Config config{};
            config.lr = read_little_endian<double>(is, "PPISP config lr");
            config.beta1 = read_little_endian<double>(is, "PPISP config beta1");
            config.beta2 = read_little_endian<double>(is, "PPISP config beta2");
            config.eps = read_little_endian<double>(is, "PPISP config eps");
            config.warmup_steps = read_little_endian<int32_t>(is, "PPISP config warmup_steps");
            config.warmup_start_factor =
                read_little_endian<double>(is, "PPISP config warmup_start_factor");
            config.final_lr_factor = read_little_endian<double>(is, "PPISP config final_lr_factor");
            config.exposure_mean = read_little_endian<float>(is, "PPISP config exposure_mean");
            config.vig_center = read_little_endian<float>(is, "PPISP config vig_center");
            config.vig_channel = read_little_endian<float>(is, "PPISP config vig_channel");
            config.vig_non_pos = read_little_endian<float>(is, "PPISP config vig_non_pos");
            config.color_mean = read_little_endian<float>(is, "PPISP config color_mean");
            config.crf_channel = read_little_endian<float>(is, "PPISP config crf_channel");
            uint32_t consumed = CONFIG_SCHEMA_V1_BYTES;
            if (payload_bytes >= CONFIG_SCHEMA_V2_BYTES) {
                const int32_t train_crf = read_little_endian<int32_t>(is, "PPISP config train_crf");
                config.train_crf = train_crf != 0;
                consumed = CONFIG_SCHEMA_V2_BYTES;
            }
            config_serialization_detail::skip_bytes(
                is, payload_bytes - consumed, "PPISP config");
            return config;
        }

        [[nodiscard]] PPISP::Config deserialize_legacy_config(std::istream& is) {
            static_assert(std::is_standard_layout_v<LegacyConfigV2>);
            static_assert(sizeof(LegacyConfigV2) == 80);
            static_assert(offsetof(LegacyConfigV2, lr) == 0);
            static_assert(offsetof(LegacyConfigV2, beta1) == 8);
            static_assert(offsetof(LegacyConfigV2, beta2) == 16);
            static_assert(offsetof(LegacyConfigV2, eps) == 24);
            static_assert(offsetof(LegacyConfigV2, warmup_steps) == 32);
            static_assert(offsetof(LegacyConfigV2, warmup_start_factor) == 40);
            static_assert(offsetof(LegacyConfigV2, final_lr_factor) == 48);
            static_assert(offsetof(LegacyConfigV2, exposure_mean) == 56);
            static_assert(offsetof(LegacyConfigV2, vig_center) == 60);
            static_assert(offsetof(LegacyConfigV2, vig_channel) == 64);
            static_assert(offsetof(LegacyConfigV2, vig_non_pos) == 68);
            static_assert(offsetof(LegacyConfigV2, color_mean) == 72);
            static_assert(offsetof(LegacyConfigV2, crf_channel) == 76);

            LegacyConfigV2 legacy{};
            lfs::core::serialization_detail::read_exact(
                is, &legacy, sizeof(legacy), "legacy PPISP configuration");
            return {
                .lr = legacy.lr,
                .beta1 = legacy.beta1,
                .beta2 = legacy.beta2,
                .eps = legacy.eps,
                .warmup_steps = legacy.warmup_steps,
                .warmup_start_factor = legacy.warmup_start_factor,
                .final_lr_factor = legacy.final_lr_factor,
                .exposure_mean = legacy.exposure_mean,
                .vig_center = legacy.vig_center,
                .vig_channel = legacy.vig_channel,
                .vig_non_pos = legacy.vig_non_pos,
                .color_mean = legacy.color_mean,
                .crf_channel = legacy.crf_channel,
            };
        }

        void serialize_int_map(std::ostream& os, const std::unordered_map<int, int>& m) {
            const auto size = static_cast<uint32_t>(m.size());
            os.write(reinterpret_cast<const char*>(&size), sizeof(size));
            for (const auto& [k, v] : m) {
                os.write(reinterpret_cast<const char*>(&k), sizeof(k));
                os.write(reinterpret_cast<const char*>(&v), sizeof(v));
            }
        }

        std::unordered_map<int, int> deserialize_int_map(
            std::istream& is,
            const uint32_t max_size,
            const std::string_view name) {
            uint32_t size = 0;
            lfs::core::serialization_detail::read_exact(is, &size, sizeof(size), name);
            if (size > max_size)
                throw std::runtime_error("Invalid PPISP checkpoint: map exceeds entry budget");
            std::unordered_map<int, int> result;
            result.reserve(size);
            for (uint32_t i = 0; i < size; ++i) {
                int key = 0, value = 0;
                lfs::core::serialization_detail::read_exact(is, &key, sizeof(key), name);
                lfs::core::serialization_detail::read_exact(is, &value, sizeof(value), name);
                if (!result.emplace(key, value).second)
                    throw std::runtime_error("Invalid PPISP checkpoint: duplicate map key");
            }
            return result;
        }
    } // namespace

    PPISP::PPISP(int total_iterations, Config config)
        : config_(config),
          current_lr_(config.warmup_steps > 0 ? config.lr * config.warmup_start_factor : config.lr),
          initial_lr_(config.lr),
          total_iterations_(total_iterations) {
    }

    void PPISP::register_frame(int uid, int camera_id) {
        assert(!finalized_ && "Cannot register frames after finalize()");
        assert(!is_known_frame(uid) && "Duplicate frame UID");

        if (!is_known_camera(camera_id)) {
            camera_id_to_idx_[camera_id] = static_cast<int>(camera_id_to_idx_.size());
        }
        uid_to_frame_idx_[uid] = static_cast<int>(uid_to_frame_idx_.size());
        uid_to_camera_id_[uid] = camera_id;
    }

    void PPISP::finalize() {
        assert(!finalized_ && "Already finalized");
        assert(!uid_to_frame_idx_.empty() && "No frames registered");
        assert(!camera_id_to_idx_.empty() && "No cameras registered");

        num_cameras_ = static_cast<int>(camera_id_to_idx_.size());
        num_frames_ = static_cast<int>(uid_to_frame_idx_.size());

        allocate_tensors();
        finalized_ = true;

        LOG_DEBUG("PPISP: {} cameras, {} frames, lr={:.2e}", num_cameras_, num_frames_, config_.lr);
    }

    void PPISP::seed_exposure(const std::vector<std::pair<int, float>>& uid_ev) {
        assert(finalized_ && "Must call finalize() before seed_exposure()");
        if (uid_ev.empty() || num_frames_ <= 0) {
            return;
        }

        double sum = 0.0;
        int n = 0;
        for (const auto& [uid, ev] : uid_ev) {
            if (!std::isfinite(ev) || !is_known_frame(uid)) {
                continue;
            }
            sum += static_cast<double>(ev);
            ++n;
        }
        if (n == 0) {
            return;
        }
        const float mean = static_cast<float>(sum / static_cast<double>(n));

        auto host = exposure_params_.cpu();
        float* const ptr = host.ptr<float>();
        for (const auto& [uid, ev] : uid_ev) {
            if (!std::isfinite(ev)) {
                continue;
            }
            const auto it = uid_to_frame_idx_.find(uid);
            if (it == uid_to_frame_idx_.end()) {
                continue;
            }
            ptr[it->second] = std::clamp(0.5f * (ev - mean), -16.0f, 16.0f); // PPISP_MIN/MAX_EXPOSURE_EV
        }
        exposure_params_.copy_from(host);
    }

    bool PPISP::is_known_frame(int uid) const { return uid_to_frame_idx_.find(uid) != uid_to_frame_idx_.end(); }

    bool PPISP::is_known_camera(int camera_id) const {
        return camera_id_to_idx_.find(camera_id) != camera_id_to_idx_.end();
    }

    int PPISP::camera_for_frame(int uid) const {
        auto it = uid_to_camera_id_.find(uid);
        assert(it != uid_to_camera_id_.end() && "Unknown frame UID");
        return it->second;
    }

    int PPISP::camera_index(int camera_id) const { return translate_camera(camera_id); }

    int PPISP::translate_camera(int camera_id) const {
        auto it = camera_id_to_idx_.find(camera_id);
        assert(it != camera_id_to_idx_.end() && "Unknown camera_id");
        return it->second;
    }

    int PPISP::translate_frame(int uid) const {
        auto it = uid_to_frame_idx_.find(uid);
        assert(it != uid_to_frame_idx_.end() && "Unknown frame UID");
        return it->second;
    }

    void PPISP::allocate_tensors() {
        assert(num_cameras_ > 0 && "num_cameras must be positive");
        assert(num_frames_ > 0 && "num_frames must be positive");

        // Allocate exposure params [num_frames]
        exposure_params_ = lfs::core::Tensor::zeros({static_cast<size_t>(num_frames_)}, lfs::core::Device::GPU);
        if (const auto reason = unavailable_training_family(lfs::core::gpu_backend_of(exposure_params_).value(), Family::PPISP)) {
            throw std::runtime_error(*reason);
        }
        exposure_exp_avg_ = lfs::core::Tensor::zeros({static_cast<size_t>(num_frames_)}, lfs::core::Device::GPU);
        exposure_exp_avg_sq_ = lfs::core::Tensor::zeros({static_cast<size_t>(num_frames_)}, lfs::core::Device::GPU);
        exposure_grad_ = lfs::core::Tensor::zeros({static_cast<size_t>(num_frames_)}, lfs::core::Device::GPU);

        // Allocate vignetting params [num_cameras * 3 * 5]
        size_t vig_size = static_cast<size_t>(num_cameras_) * 3 * 5;
        vignetting_params_ = lfs::core::Tensor::zeros({vig_size}, lfs::core::Device::GPU);
        vignetting_exp_avg_ = lfs::core::Tensor::zeros({vig_size}, lfs::core::Device::GPU);
        vignetting_exp_avg_sq_ = lfs::core::Tensor::zeros({vig_size}, lfs::core::Device::GPU);
        vignetting_grad_ = lfs::core::Tensor::zeros({vig_size}, lfs::core::Device::GPU);

        // Allocate color params [num_frames * 8]
        size_t color_size = static_cast<size_t>(num_frames_) * 8;
        color_params_ = lfs::core::Tensor::zeros({color_size}, lfs::core::Device::GPU);
        color_exp_avg_ = lfs::core::Tensor::zeros({color_size}, lfs::core::Device::GPU);
        color_exp_avg_sq_ = lfs::core::Tensor::zeros({color_size}, lfs::core::Device::GPU);
        color_grad_ = lfs::core::Tensor::zeros({color_size}, lfs::core::Device::GPU);

        // Allocate CRF params [num_cameras * 3 * 4]
        size_t crf_size = static_cast<size_t>(num_cameras_) * 3 * 4;
        crf_params_ = lfs::core::Tensor::zeros({crf_size}, lfs::core::Device::GPU);
        crf_exp_avg_ = lfs::core::Tensor::zeros({crf_size}, lfs::core::Device::GPU);
        crf_exp_avg_sq_ = lfs::core::Tensor::zeros({crf_size}, lfs::core::Device::GPU);
        crf_grad_ = lfs::core::Tensor::zeros({crf_size}, lfs::core::Device::GPU);

        override_exposure_ = lfs::core::Tensor::zeros({1}, lfs::core::Device::GPU);
        override_color_ = lfs::core::Tensor::zeros({8}, lfs::core::Device::GPU);
        vig_reg_loss_ = lfs::core::Tensor::zeros({1}, lfs::core::Device::GPU);

        // Scratch buffers for backward_with_controller_params
        ctrl_bwd_exposure_ = lfs::core::Tensor::zeros({1}, lfs::core::Device::GPU);
        ctrl_bwd_color_ = lfs::core::Tensor::zeros({8}, lfs::core::Device::GPU);
        ctrl_bwd_vignetting_ = lfs::core::Tensor::zeros({vig_size}, lfs::core::Device::GPU);
        ctrl_bwd_crf_ = lfs::core::Tensor::zeros({crf_size}, lfs::core::Device::GPU);
        ctrl_bwd_output_ = lfs::core::Tensor::empty({9}, lfs::core::Device::GPU);

        training_ops(lfs::core::gpu_backend_of(exposure_params_).value()).ppisp->initialize({exposure_params_, vignetting_params_, color_params_, crf_params_});

        init_color_pinv_block_diag();
    }

    void PPISP::init_color_pinv_block_diag() {
        // ZCA pinv block-diagonal matrix for color mean regularization
        // 8x8 block-diagonal: [Blue 2x2, Red 2x2, Green 2x2, Neutral 2x2]
        // From Python: _COLOR_PINV_BLOCK_DIAG
        // clang-format off
        color_pinv_block_diag_ = lfs::core::Tensor::from_vector({
            // Blue block
            0.0480542f, -0.0043631f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            -0.0043631f, 0.0481283f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            // Red block
            0.0f, 0.0f, 0.0580570f, -0.0179872f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, -0.0179872f, 0.0431061f, 0.0f, 0.0f, 0.0f, 0.0f,
            // Green block
            0.0f, 0.0f, 0.0f, 0.0f, 0.0433336f, -0.0180537f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, -0.0180537f, 0.0580500f, 0.0f, 0.0f,
            // Neutral block
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0128369f, -0.0034654f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -0.0034654f, 0.0128158f,
        }, {8, 8}, lfs::core::Device::GPU);
        // clang-format on
    }

    lfs::core::Tensor PPISP::apply_forward(const lfs::core::Tensor& rgb, int camera_idx, int frame_idx,
                                           const lfs::core::Tensor& exposure, const lfs::core::Tensor& color, int num_frames,
                                           const PPISPRegion& region) {
        const auto& shape = rgb.shape();
        assert(shape.rank() == 3 && shape[0] == 3 && "Expected CHW layout with 3 channels");

        const int h = static_cast<int>(shape[1]);
        const int full_h = region.full_height > 0 ? region.full_height : h;
        assert(region.y_offset >= 0 && region.y_offset + h <= full_h && "PPISP region out of bounds");

        auto output = lfs::core::Tensor::empty({3, shape[1], shape[2]}, lfs::core::Device::GPU);

        training_ops(lfs::core::gpu_backend_of(exposure_params_).value()).ppisp->forward({exposure, vignetting_params_, color, crf_params_}, rgb, output, {region.y_offset, full_h, num_cameras_, num_frames, camera_idx, frame_idx});

        return output;
    }

    lfs::core::Tensor PPISP::apply(const lfs::core::Tensor& rgb, int camera_id, int uid, const PPISPRegion& region) {
        assert(finalized_ && "Must call finalize() before apply()");
        const int camera_idx = translate_camera(camera_id);
        const int frame_idx = translate_frame(uid);
        return apply_forward(rgb, camera_idx, frame_idx, exposure_params_, color_params_,
                             num_frames_, region);
    }

    lfs::core::Tensor PPISP::apply_interpolated_frames(const lfs::core::Tensor& rgb, int camera_id,
                                                       int left_uid, int right_uid, float fraction) {
        assert(finalized_ && fraction >= 0.0f && fraction <= 1.0f);
        assert(exposure_params_.shape().rank() == 1 && exposure_params_.shape()[0] == static_cast<size_t>(num_frames_));
        assert(color_params_.shape().rank() == 1 && color_params_.shape()[0] == static_cast<size_t>(num_frames_ * 8));
        assert(is_known_frame(left_uid) && is_known_frame(right_uid) && is_known_camera(camera_id));
        assert(rgb.ndim() == 3 && rgb.shape()[0] == 3 && rgb.is_contiguous());
        assert(rgb.device() == lfs::core::Device::GPU && rgb.dtype() == lfs::core::DataType::Float32);
        if (left_uid == right_uid)
            return apply(rgb, camera_id, left_uid);
        const int left = translate_frame(left_uid);
        const int right = translate_frame(right_uid);
        auto exposure = exposure_params_.slice(0, left, left + 1) * (1.0f - fraction) +
                        exposure_params_.slice(0, right, right + 1) * fraction;
        auto color = color_params_.slice(0, left * 8, left * 8 + 8) * (1.0f - fraction) +
                     color_params_.slice(0, right * 8, right * 8 + 8) * fraction;
        return apply_forward(rgb, translate_camera(camera_id), 0, exposure, color, 1, {});
    }

    lfs::core::Tensor PPISP::apply_with_exposure(const lfs::core::Tensor& rgb, int camera_id, float exposure_ev,
                                                 const PPISPRegion& region) {
        assert(finalized_ && "Must call finalize() before apply_with_exposure()");
        const int camera_idx = translate_camera(camera_id);
        const float clamped = std::clamp(exposure_ev, -16.0f, 16.0f); // PPISP_MIN/MAX_EXPOSURE_EV
        override_exposure_.fill_(clamped);
        // The forward kernel resolves a null stream to the current stream.
        lfs::core::TensorExecutionTarget::current().wait_for(override_exposure_.execution_target());
        return apply_forward(rgb, camera_idx, 0, override_exposure_, override_color_, 1,
                             region);
    }

    lfs::core::Tensor PPISP::apply_with_exposure_and_overrides(const lfs::core::Tensor& rgb, int camera_id,
                                                               float exposure_ev,
                                                               const PPISPRenderOverrides& ov,
                                                               const PPISPRegion& region) {
        assert(finalized_ && "Must call finalize() before apply_with_exposure_and_overrides()");
        const int camera_idx = translate_camera(camera_id);

        const auto& shape = rgb.shape();
        assert(shape.rank() == 3 && shape[0] == 3 && "Expected CHW layout with 3 channels");

        const int h = static_cast<int>(shape[1]);
        const int full_h = region.full_height > 0 ? region.full_height : h;
        assert(region.y_offset >= 0 && region.y_offset + h <= full_h && "PPISP region out of bounds");

        auto output = lfs::core::Tensor::empty({3, shape[1], shape[2]}, lfs::core::Device::GPU);

        const float clamped = std::clamp(exposure_ev + ov.exposure_offset, -16.0f, 16.0f);
        override_exposure_.fill_(clamped);

        constexpr float COLOR_SCALE = 12.0f;
        constexpr float WB_SCALE = 24.0f;
        std::vector<float> color(8, 0.0f);
        color[0] = ov.color_blue_x * COLOR_SCALE;
        color[1] = ov.color_blue_y * COLOR_SCALE;
        color[2] = ov.color_red_x * COLOR_SCALE;
        color[3] = ov.color_red_y * COLOR_SCALE;
        color[4] = ov.color_green_x * COLOR_SCALE;
        color[5] = ov.color_green_y * COLOR_SCALE;
        color[6] = ov.wb_temperature * WB_SCALE;
        color[7] = ov.wb_tint * WB_SCALE;
        override_color_.copy_from(lfs::core::Tensor::from_vector(
            color, lfs::core::TensorShape({color.size()}), lfs::core::Device::CPU));

        auto vignetting_modified = vignetting_params_.clone();
        {
            auto vig_cpu = vignetting_modified.cpu();
            float* vig_ptr = vig_cpu.ptr<float>();
            const float mult = ov.vignette_enabled ? ov.vignette_strength : 0.0f;
            for (int ch = 0; ch < 3; ++ch) {
                const size_t base = static_cast<size_t>(camera_idx) * 15 + static_cast<size_t>(ch) * 5;
                vig_ptr[base + 2] *= mult;
                vig_ptr[base + 3] *= mult;
                vig_ptr[base + 4] *= mult;
            }
            const size_t copy_offset = static_cast<size_t>(camera_idx) * 15;
            vignetting_modified.flatten().slice(0, copy_offset, copy_offset + 15).copy_from(vig_cpu.flatten().slice(0, copy_offset, copy_offset + 15));
        }

        auto crf_modified = crf_params_.clone();
        {
            auto crf_cpu = crf_modified.cpu();
            float* crf_ptr = crf_cpu.ptr<float>();
            const float gamma_offsets[3] = {ov.gamma_red, ov.gamma_green, ov.gamma_blue};
            const float log_gamma_mult = std::log(ov.gamma_multiplier);
            for (int ch = 0; ch < 3; ++ch) {
                const size_t base = static_cast<size_t>(camera_idx) * 12 + static_cast<size_t>(ch) * 4;
                crf_ptr[base + 0] += ov.crf_toe;
                crf_ptr[base + 1] += ov.crf_shoulder;
                crf_ptr[base + 2] += log_gamma_mult + gamma_offsets[ch];
            }
            const size_t copy_offset = static_cast<size_t>(camera_idx) * 12;
            crf_modified.flatten().slice(0, copy_offset, copy_offset + 12).copy_from(crf_cpu.flatten().slice(0, copy_offset, copy_offset + 12));
        }

        lfs::core::TensorExecutionTarget::current().wait_for(override_exposure_.execution_target());
        training_ops(lfs::core::gpu_backend_of(exposure_params_).value()).ppisp->forward({override_exposure_, vignetting_modified, override_color_, crf_modified}, rgb, output, {region.y_offset, full_h, num_cameras_, 1, camera_idx, 0});
        return output;
    }

    lfs::core::Tensor PPISP::apply_with_controller_params(const lfs::core::Tensor& rgb,
                                                          const lfs::core::Tensor& controller_params,
                                                          int camera_idx,
                                                          const PPISPRegion& region) {
        assert(controller_params.shape().rank() == 2 && "Expected [1,9]");
        assert(controller_params.shape()[0] == 1 && controller_params.shape()[1] == 9);
        assert(camera_idx >= 0 && camera_idx < num_cameras_ && "camera_idx out of range");

        const auto& shape = rgb.shape();
        assert(shape.rank() == 3 && shape[0] == 3 && "Expected CHW layout with 3 channels");

        const int h = static_cast<int>(shape[1]);
        const int full_h = region.full_height > 0 ? region.full_height : h;
        assert(region.y_offset >= 0 && region.y_offset + h <= full_h && "PPISP region out of bounds");

        // Extract exposure (index 0) and color params (indices 1-8) from controller output
        auto exposure_temp = controller_params.slice(1, 0, 1).reshape({1});
        auto color_temp = controller_params.slice(1, 1, 9).reshape({8});

        auto output = lfs::core::Tensor::empty({3, shape[1], shape[2]}, lfs::core::Device::GPU);

        // Use controller-predicted exposure and color, but existing vignetting and CRF from camera
        training_ops(lfs::core::gpu_backend_of(exposure_params_).value()).ppisp->forward({exposure_temp, vignetting_params_, color_temp, crf_params_}, rgb, output, {region.y_offset, full_h, num_cameras_, 1, camera_idx, 0});

        return output;
    }

    lfs::core::Tensor PPISP::apply_with_controller_params_and_overrides(const lfs::core::Tensor& rgb,
                                                                        const lfs::core::Tensor& controller_params,
                                                                        int camera_idx,
                                                                        const PPISPRenderOverrides& ov,
                                                                        const PPISPRegion& region) {
        assert(controller_params.shape().rank() == 2 && "Expected [1,9]");
        assert(controller_params.shape()[0] == 1 && controller_params.shape()[1] == 9);
        assert(camera_idx >= 0 && camera_idx < num_cameras_ && "camera_idx out of range");

        const auto& shape = rgb.shape();
        assert(shape.rank() == 3 && shape[0] == 3 && "Expected CHW layout with 3 channels");

        const int h = static_cast<int>(shape[1]);
        const int full_h = region.full_height > 0 ? region.full_height : h;
        assert(region.y_offset >= 0 && region.y_offset + h <= full_h && "PPISP region out of bounds");

        auto output = lfs::core::Tensor::empty({3, shape[1], shape[2]}, lfs::core::Device::GPU);

        // Extract and modify exposure from controller output
        auto exposure_temp = controller_params.slice(1, 0, 1).reshape({1}).clone();
        if (ov.exposure_offset != 0.0f) {
            auto exp_cpu = exposure_temp.cpu();
            exp_cpu.ptr<float>()[0] += ov.exposure_offset;
            exposure_temp.copy_from(exp_cpu);
        }

        // Color params [b.x, b.y, r.x, r.y, g.x, g.y, n.x, n.y] - latent space, scaled for ZCA transform
        constexpr float COLOR_SCALE = 12.0f;
        constexpr float WB_SCALE = 24.0f;
        auto color_temp = controller_params.slice(1, 1, 9).reshape({8}).clone();
        {
            auto color_cpu = color_temp.cpu();
            float* p = color_cpu.ptr<float>();
            p[0] += ov.color_blue_x * COLOR_SCALE;
            p[1] += ov.color_blue_y * COLOR_SCALE;
            p[2] += ov.color_red_x * COLOR_SCALE;
            p[3] += ov.color_red_y * COLOR_SCALE;
            p[4] += ov.color_green_x * COLOR_SCALE;
            p[5] += ov.color_green_y * COLOR_SCALE;
            p[6] += ov.wb_temperature * WB_SCALE;
            p[7] += ov.wb_tint * WB_SCALE;
            color_temp.copy_from(color_cpu);
        }

        // Vignetting: multiply alpha coefficients by strength (or zero if disabled)
        auto vignetting_modified = vignetting_params_.clone();
        {
            auto vig_cpu = vignetting_modified.cpu();
            float* vig_ptr = vig_cpu.ptr<float>();
            const float mult = ov.vignette_enabled ? ov.vignette_strength : 0.0f;
            for (int ch = 0; ch < 3; ++ch) {
                const size_t base = static_cast<size_t>(camera_idx) * 15 + static_cast<size_t>(ch) * 5;
                vig_ptr[base + 2] *= mult;
                vig_ptr[base + 3] *= mult;
                vig_ptr[base + 4] *= mult;
            }
            const size_t copy_offset = static_cast<size_t>(camera_idx) * 15;
            vignetting_modified.flatten().slice(0, copy_offset, copy_offset + 15).copy_from(vig_cpu.flatten().slice(0, copy_offset, copy_offset + 15));
        }

        // CRF params [toe, shoulder, gamma, center] per channel
        auto crf_modified = crf_params_.clone();
        {
            auto crf_cpu = crf_modified.cpu();
            float* crf_ptr = crf_cpu.ptr<float>();
            const float gamma_offsets[3] = {ov.gamma_red, ov.gamma_green, ov.gamma_blue};
            const float log_gamma_mult = std::log(ov.gamma_multiplier);
            for (int ch = 0; ch < 3; ++ch) {
                const size_t base = static_cast<size_t>(camera_idx) * 12 + static_cast<size_t>(ch) * 4;
                crf_ptr[base + 0] += ov.crf_toe;
                crf_ptr[base + 1] += ov.crf_shoulder;
                crf_ptr[base + 2] += log_gamma_mult + gamma_offsets[ch];
            }
            const size_t copy_offset = static_cast<size_t>(camera_idx) * 12;
            crf_modified.flatten().slice(0, copy_offset, copy_offset + 12).copy_from(crf_cpu.flatten().slice(0, copy_offset, copy_offset + 12));
        }

        training_ops(lfs::core::gpu_backend_of(exposure_params_).value()).ppisp->forward({exposure_temp, vignetting_modified, color_temp, crf_modified}, rgb, output, {region.y_offset, full_h, num_cameras_, 1, camera_idx, 0});

        return output;
    }

    lfs::core::Tensor PPISP::apply_with_overrides(const lfs::core::Tensor& rgb, int camera_id, int uid,
                                                  const PPISPRenderOverrides& ov, const PPISPRegion& region) {
        assert(finalized_ && "Must call finalize() before apply_with_overrides()");
        const int camera_idx = translate_camera(camera_id);
        const int frame_idx = translate_frame(uid);

        const auto& shape = rgb.shape();
        assert(shape.rank() == 3 && shape[0] == 3 && "Expected CHW layout with 3 channels");

        const int h = static_cast<int>(shape[1]);
        const int full_h = region.full_height > 0 ? region.full_height : h;
        assert(region.y_offset >= 0 && region.y_offset + h <= full_h && "PPISP region out of bounds");

        auto output = lfs::core::Tensor::empty({3, shape[1], shape[2]}, lfs::core::Device::GPU);

        // Exposure: add offset to learned value
        auto exposure_modified = exposure_params_.clone();
        if (ov.exposure_offset != 0.0f) {
            auto exp_cpu = exposure_modified.slice(0, frame_idx, frame_idx + 1).cpu();
            exp_cpu.ptr<float>()[0] += ov.exposure_offset;
            exposure_modified.slice(0, frame_idx, frame_idx + 1).copy_from(exp_cpu);
        }

        // Vignetting: multiply alpha coefficients by strength (or zero if disabled)
        auto vignetting_modified = vignetting_params_.clone();
        {
            auto vig_cpu = vignetting_modified.cpu();
            float* vig_ptr = vig_cpu.ptr<float>();
            const float mult = ov.vignette_enabled ? ov.vignette_strength : 0.0f;
            for (int ch = 0; ch < 3; ++ch) {
                const size_t base = static_cast<size_t>(camera_idx) * 15 + static_cast<size_t>(ch) * 5;
                vig_ptr[base + 2] *= mult;
                vig_ptr[base + 3] *= mult;
                vig_ptr[base + 4] *= mult;
            }
            const size_t copy_offset = static_cast<size_t>(camera_idx) * 15;
            vignetting_modified.flatten().slice(0, copy_offset, copy_offset + 15).copy_from(vig_cpu.flatten().slice(0, copy_offset, copy_offset + 15));
        }

        // Color params [b.x, b.y, r.x, r.y, g.x, g.y, n.x, n.y] - latent space, scaled for ZCA transform
        constexpr float COLOR_SCALE = 12.0f;
        constexpr float WB_SCALE = 24.0f;
        auto color_modified = color_params_.clone();
        {
            auto color_cpu = color_modified.cpu();
            float* p = color_cpu.ptr<float>();
            const size_t base = static_cast<size_t>(frame_idx) * 8;
            p[base + 0] += ov.color_blue_x * COLOR_SCALE;
            p[base + 1] += ov.color_blue_y * COLOR_SCALE;
            p[base + 2] += ov.color_red_x * COLOR_SCALE;
            p[base + 3] += ov.color_red_y * COLOR_SCALE;
            p[base + 4] += ov.color_green_x * COLOR_SCALE;
            p[base + 5] += ov.color_green_y * COLOR_SCALE;
            p[base + 6] += ov.wb_temperature * WB_SCALE;
            p[base + 7] += ov.wb_tint * WB_SCALE;
            color_modified.flatten().slice(0, base, base + 8).copy_from(color_cpu.flatten().slice(0, base, base + 8));
        }

        // CRF params [toe, shoulder, gamma, center] per channel
        auto crf_modified = crf_params_.clone();
        {
            auto crf_cpu = crf_modified.cpu();
            float* crf_ptr = crf_cpu.ptr<float>();
            const float gamma_offsets[3] = {ov.gamma_red, ov.gamma_green, ov.gamma_blue};
            const float log_gamma_mult = std::log(ov.gamma_multiplier);
            for (int ch = 0; ch < 3; ++ch) {
                const size_t base = static_cast<size_t>(camera_idx) * 12 + static_cast<size_t>(ch) * 4;
                crf_ptr[base + 0] += ov.crf_toe;
                crf_ptr[base + 1] += ov.crf_shoulder;
                crf_ptr[base + 2] += log_gamma_mult + gamma_offsets[ch];
            }
            const size_t copy_offset = static_cast<size_t>(camera_idx) * 12;
            crf_modified.flatten().slice(0, copy_offset, copy_offset + 12).copy_from(crf_cpu.flatten().slice(0, copy_offset, copy_offset + 12));
        }

        training_ops(lfs::core::gpu_backend_of(exposure_params_).value()).ppisp->forward({exposure_modified, vignetting_modified, color_modified, crf_modified}, rgb, output, {region.y_offset, full_h, num_cameras_, num_frames_, camera_idx, frame_idx});

        return output;
    }

    lfs::core::Tensor PPISP::backward(const lfs::core::Tensor& rgb, const lfs::core::Tensor& grad_output, int camera_id,
                                      int uid) {
        assert(finalized_ && "Must call finalize() before backward()");
        const int camera_idx = translate_camera(camera_id);
        const int frame_idx = translate_frame(uid);

        const auto& shape = rgb.shape();
        assert(shape.rank() == 3 && shape[0] == 3 && "Expected CHW layout with 3 channels");

        auto grad_rgb = lfs::core::Tensor::empty({3, shape[1], shape[2]}, lfs::core::Device::GPU);

        training_ops(lfs::core::gpu_backend_of(exposure_params_).value()).ppisp->backward({exposure_params_, vignetting_params_, color_params_, crf_params_}, rgb, grad_output, {exposure_grad_, vignetting_grad_, color_grad_, crf_grad_}, grad_rgb, num_cameras_, num_frames_, camera_idx, frame_idx);

        return grad_rgb;
    }

    void PPISP::backward_in_place(const lfs::core::Tensor& rgb, lfs::core::Tensor& grad, int camera_id, int uid) {
        assert(finalized_ && "Must call finalize() before backward_in_place()");
        const int camera_idx = translate_camera(camera_id);
        const int frame_idx = translate_frame(uid);

        const auto& shape = rgb.shape();
        assert(shape.rank() == 3 && shape[0] == 3 && "Expected CHW layout with 3 channels");

        const auto& ops = training_ops(lfs::core::gpu_backend_of(exposure_params_).value());
        if (!ops.ppisp->in_place) {
            grad = backward(rgb, grad, camera_id, uid);
            return;
        }
        LFS_ASSERT(grad.is_contiguous() && grad.shape() == rgb.shape());
        ops.ppisp->backward({exposure_params_, vignetting_params_, color_params_, crf_params_}, rgb, grad, {exposure_grad_, vignetting_grad_, color_grad_, crf_grad_}, grad, num_cameras_, num_frames_, camera_idx, frame_idx);
    }

    lfs::core::Tensor PPISP::backward_with_controller_params(const lfs::core::Tensor& rgb,
                                                             const lfs::core::Tensor& grad_output,
                                                             const lfs::core::Tensor& controller_params,
                                                             int camera_idx) {
        assert(finalized_ && "Must call finalize() before backward_with_controller_params()");
        assert(controller_params.shape().rank() == 2 && "Expected [1,9]");
        assert(controller_params.shape()[0] == 1 && controller_params.shape()[1] == 9);
        assert(camera_idx >= 0 && camera_idx < num_cameras_ && "camera_idx out of range");

        const auto& shape = rgb.shape();
        assert(shape.rank() == 3 && shape[0] == 3 && "Expected CHW layout with 3 channels");

        const size_t h = shape[1];
        const size_t w = shape[2];

        // Lazy-resize rgb scratch buffer if image dimensions changed
        if (h != ctrl_bwd_rgb_h_ || w != ctrl_bwd_rgb_w_) {
            ctrl_bwd_rgb_ = lfs::core::Tensor::empty({3, h, w}, lfs::core::Device::GPU);
            ctrl_bwd_rgb_h_ = h;
            ctrl_bwd_rgb_w_ = w;
        }

        auto exposure_temp = controller_params.slice(1, 0, 1).reshape({1});
        auto color_temp = controller_params.slice(1, 1, 9).reshape({8});

        // Zero preallocated gradient scratch buffers
        ctrl_bwd_exposure_.zero_();
        ctrl_bwd_color_.zero_();
        ctrl_bwd_vignetting_.zero_();
        ctrl_bwd_crf_.zero_();

        training_ops(lfs::core::gpu_backend_of(exposure_params_).value()).ppisp->backward({exposure_temp, vignetting_params_, color_temp, crf_params_}, rgb, grad_output, {ctrl_bwd_exposure_, ctrl_bwd_vignetting_, ctrl_bwd_color_, ctrl_bwd_crf_}, ctrl_bwd_rgb_, num_cameras_, 1, camera_idx, 0);

        // Assemble [exposure(1), color(8)] -> [9] via D2D copy into preallocated output
        ctrl_bwd_output_.slice(0, 0, 1).copy_(ctrl_bwd_exposure_);
        ctrl_bwd_output_.slice(0, 1, 9).copy_(ctrl_bwd_color_);

        return ctrl_bwd_output_.reshape({1, 9});
    }

    namespace {
        // Smooth L1 (Huber): 0.5*x^2/beta if |x| < beta, else |x| - 0.5*beta.
        // With c = min(|x|, beta) both branches are c * (|x| - 0.5*c) / beta.
        lfs::core::Tensor smooth_l1(const lfs::core::Tensor& x, const float beta) {
            const auto magnitude = x.abs();
            const auto c = magnitude.minimum(beta);
            return c.mul(magnitude.sub(c.mul(0.5f))).div(beta);
        }

        // Gradient of smooth L1: x/beta if |x| < beta, else sign(x).
        lfs::core::Tensor smooth_l1_grad(const lfs::core::Tensor& x, const float beta) {
            return x.div(beta).clamp(-1.0f, 1.0f);
        }

        // [cameras * 3 * k] per-channel parameters minus their mean over the 3 channels.
        lfs::core::Tensor channel_deviation(const lfs::core::Tensor& params, const int cameras, const int k) {
            const auto grouped = params.reshape({cameras, 3, k});
            return grouped.sub(grouped.mean(1, true));
        }
    } // namespace

    // Regularizer terms stay on the device: no host round trip per step.
    // Vignetting (center, non-positivity, channel variance) is the
    // vignetting_regularization op; the rest are tensor expressions.
    lfs::core::Tensor PPISP::reg_loss_gpu() {
        vig_reg_loss_.zero_();
        lfs::core::Tensor unused;
        training_ops(lfs::core::gpu_backend_of(exposure_params_).value()).ppisp->vignetting_regularization(vignetting_params_, unused, vig_reg_loss_, config_.vig_center, config_.vig_channel, config_.vig_non_pos);
        lfs::core::Tensor loss = vig_reg_loss_;

        // Exposure mean: smooth_l1(mean(exposure), beta=0.1)
        if (config_.exposure_mean > 0.0f) {
            loss = loss.add(smooth_l1(exposure_params_.mean().reshape({1}), 0.1f).mul(config_.exposure_mean));
        }

        // Color mean: smooth_l1(mean(color @ pinv, dim=0), beta=0.005), averaged over the 8 offsets
        if (config_.color_mean > 0.0f) {
            const auto offsets = color_params_.reshape({num_frames_, 8}).mm(color_pinv_block_diag_).mean(0);
            loss = loss.add(smooth_l1(offsets, 0.005f).sum().reshape({1}).mul(config_.color_mean / 8.0f));
        }

        // CRF channel variance: mean(var(crf, dim=channel)), layout [cam][channel][toe, shoulder, gamma, center]
        if (config_.train_crf && config_.crf_channel > 0.0f) {
            const auto deviation = channel_deviation(crf_params_, num_cameras_, 4);
            loss = loss.add(deviation.square().sum().reshape({1}).mul(
                config_.crf_channel / static_cast<float>(num_cameras_ * 4 * 3)));
        }
        return loss;
    }

    void PPISP::reg_backward() {
        lfs::core::Tensor unused;
        training_ops(lfs::core::gpu_backend_of(exposure_params_).value()).ppisp->vignetting_regularization(vignetting_params_, vignetting_grad_, unused, config_.vig_center, config_.vig_channel, config_.vig_non_pos);

        if (config_.exposure_mean > 0.0f) {
            const auto grad_mean = smooth_l1_grad(exposure_params_.mean().reshape({1}), 0.1f);
            exposure_grad_ = exposure_grad_.add(
                grad_mean.mul(config_.exposure_mean / static_cast<float>(num_frames_)));
        }

        // d/d(color[f, k]) = sum_j pinv[k, j] * grad_offsets[j] / num_frames
        if (config_.color_mean > 0.0f) {
            const auto offsets = color_params_.reshape({num_frames_, 8}).mm(color_pinv_block_diag_).mean(0);
            const auto grad_offsets = smooth_l1_grad(offsets, 0.005f).mul(config_.color_mean / (8.0f * static_cast<float>(num_frames_)));
            const auto grad_row = color_pinv_block_diag_.mm(grad_offsets.reshape({8, 1})).reshape({1, 8});
            color_grad_ = color_grad_.reshape({num_frames_, 8}).add(grad_row).reshape({num_frames_ * 8});
        }

        // d/d(x_ch) of mean(var) = scale * 2 * (x_ch - mean) / 3
        if (config_.train_crf && config_.crf_channel > 0.0f) {
            const auto deviation = channel_deviation(crf_params_, num_cameras_, 4);
            crf_grad_ = crf_grad_.add(deviation.mul(
                                                   2.0f * config_.crf_channel / static_cast<float>(num_cameras_ * 4 * 3))
                                          .reshape({num_cameras_ * 12}));
        }
    }

    void PPISP::optimizer_step() {
        float bc1_rcp, bc2_sqrt_rcp;
        compute_bias_corrections(bc1_rcp, bc2_sqrt_rcp);

        const float lr = static_cast<float>(current_lr_);
        const float beta1 = static_cast<float>(config_.beta1);
        const float beta2 = static_cast<float>(config_.beta2);
        const float eps = static_cast<float>(config_.eps);

        lfs::core::Tensor absent;
        const lfs::gpu_ops::PPISPAdamGroup crf_group = config_.train_crf
                                                           ? lfs::gpu_ops::PPISPAdamGroup{crf_params_, crf_exp_avg_, crf_exp_avg_sq_, crf_grad_}
                                                           : lfs::gpu_ops::PPISPAdamGroup{absent, absent, absent, absent};
        training_ops(lfs::core::gpu_backend_of(exposure_params_).value()).ppisp->adam_batch({{{exposure_params_, exposure_exp_avg_, exposure_exp_avg_sq_, exposure_grad_}, {vignetting_params_, vignetting_exp_avg_, vignetting_exp_avg_sq_, vignetting_grad_}, {color_params_, color_exp_avg_, color_exp_avg_sq_, color_grad_}, crf_group}}, {lr, beta1, beta2, bc1_rcp, bc2_sqrt_rcp, eps});
    }

    void PPISP::zero_grad() {
        exposure_grad_.zero_();
        vignetting_grad_.zero_();
        color_grad_.zero_();
        crf_grad_.zero_();
    }

    void PPISP::scheduler_step() {
        ++step_;

        if (step_ <= config_.warmup_steps) {
            const double progress = static_cast<double>(step_) / config_.warmup_steps;
            const double scale = config_.warmup_start_factor + (1.0 - config_.warmup_start_factor) * progress;
            current_lr_ = initial_lr_ * scale;
        } else {
            const double gamma =
                std::pow(config_.final_lr_factor, 1.0 / (total_iterations_ - config_.warmup_steps));
            current_lr_ = initial_lr_ * std::pow(gamma, step_ - config_.warmup_steps);
        }
    }

    void PPISP::project_mean() {
        assert(finalized_);
        assert(exposure_params_.is_valid());
        assert(color_params_.is_valid());
        assert(exposure_params_.ndim() == 1);
        assert(static_cast<int>(exposure_params_.shape()[0]) == num_frames_);
        assert(color_params_.numel() == static_cast<size_t>(num_frames_) * 8);
        if (num_frames_ <= 0)
            return;

        training_ops(lfs::core::gpu_backend_of(exposure_params_).value()).ppisp->project_mean(exposure_params_, color_params_);
    }

    float PPISP::mean_exposure_ev() const {
        assert(finalized_);
        if (num_frames_ <= 0)
            return 0.0f;
        auto exposure_cpu = exposure_params_.cpu();
        const float* exp_ptr = exposure_cpu.ptr<float>();
        float exp_sum = 0.0f;
        for (int i = 0; i < num_frames_; ++i) {
            exp_sum += exp_ptr[i];
        }
        return exp_sum / static_cast<float>(num_frames_);
    }

    float PPISP::max_abs_color_offset_mean() const {
        assert(finalized_);
        if (num_frames_ <= 0)
            return 0.0f;
        auto color_cpu = color_params_.cpu();
        auto pinv_cpu = color_pinv_block_diag_.cpu();
        const float* color_ptr = color_cpu.ptr<float>();
        const float* pinv_ptr = pinv_cpu.ptr<float>();
        float color_mean_offsets[8] = {0.0f};
        for (int f = 0; f < num_frames_; ++f) {
            for (int j = 0; j < 8; ++j) {
                float dot = 0.0f;
                for (int k = 0; k < 8; ++k) {
                    dot += color_ptr[f * 8 + k] * pinv_ptr[k * 8 + j];
                }
                color_mean_offsets[j] += dot;
            }
        }
        float max_abs = 0.0f;
        for (int j = 0; j < 8; ++j) {
            color_mean_offsets[j] /= static_cast<float>(num_frames_);
            max_abs = std::max(max_abs, std::abs(color_mean_offsets[j]));
        }
        return max_abs;
    }

    void PPISP::log_eval_diagnostics() const {
        if (!finalized_)
            return;
        LOG_INFO("PPISP drift: mean_exposure_ev={:.6f} max_abs_color_offset_mean={:.6f}",
                 mean_exposure_ev(), max_abs_color_offset_mean());

        auto vig_cpu = vignetting_params_.cpu();
        auto crf_cpu = crf_params_.cpu();
        const float* vig_ptr = vig_cpu.ptr<float>();
        const float* crf_ptr = crf_cpu.ptr<float>();
        auto bounded_positive = [](float raw, float min_value) {
            const float value = std::min(32.0f, std::isfinite(raw) ? raw : 0.0f);
            return min_value + std::max(value, 0.0f) + std::log(1.0f + std::exp(-std::fabs(value)));
        };
        auto decode_center = [](float raw) {
            const float x = std::isfinite(raw) ? raw : 0.0f;
            const float sig = (x >= 0.0f)
                                  ? 1.0f / (1.0f + std::exp(-x))
                                  : std::exp(x) / (1.0f + std::exp(x));
            constexpr float eps = 1.0e-4f;
            return std::min(1.0f - eps, std::max(eps, sig));
        };
        const auto camera_ids = ordered_camera_ids();
        for (int cam = 0; cam < num_cameras_; ++cam) {
            const int camera_id = camera_ids[static_cast<size_t>(cam)];
            std::string vig_str;
            std::string crf_str;
            for (int ch = 0; ch < 3; ++ch) {
                const size_t vbase = static_cast<size_t>(cam) * 15 + static_cast<size_t>(ch) * 5;
                const size_t cbase = static_cast<size_t>(cam) * 12 + static_cast<size_t>(ch) * 4;
                vig_str += std::format(" ch{}:(cx={:.4f},cy={:.4f},a0={:.4f},a1={:.4f},a2={:.4f})",
                                       ch, vig_ptr[vbase + 0], vig_ptr[vbase + 1],
                                       vig_ptr[vbase + 2], vig_ptr[vbase + 3], vig_ptr[vbase + 4]);
                crf_str += std::format(" ch{}:(toe={:.4f},shoulder={:.4f},gamma={:.4f},center={:.4f})",
                                       ch,
                                       bounded_positive(crf_ptr[cbase + 0], 0.3f),
                                       bounded_positive(crf_ptr[cbase + 1], 0.3f),
                                       bounded_positive(crf_ptr[cbase + 2], 0.1f),
                                       decode_center(crf_ptr[cbase + 3]));
            }
            LOG_INFO("PPISP camera {} vignetting{} CRF{}", camera_id, vig_str, crf_str);
        }
    }

    std::vector<int> PPISP::ordered_camera_ids() const {
        assert(finalized_ && "Must call finalize() before ordered_camera_ids()");
        std::vector<int> ordered(static_cast<size_t>(num_cameras_));
        for (const auto& [camera_id, idx] : camera_id_to_idx_) {
            assert(idx >= 0 && idx < num_cameras_ && "Invalid camera index");
            ordered[static_cast<size_t>(idx)] = camera_id;
        }
        return ordered;
    }

    int PPISP::majority_camera_id() const {
        assert(finalized_ && !camera_id_to_idx_.empty());
        std::unordered_map<int, int> counts;
        counts.reserve(camera_id_to_idx_.size());
        for (const auto& [uid, camera_id] : uid_to_camera_id_) {
            (void)uid;
            ++counts[camera_id];
        }
        int best_id = camera_id_to_idx_.begin()->first;
        int best_count = -1;
        for (const auto& [camera_id, count] : counts) {
            if (count > best_count || (count == best_count && camera_id < best_id)) {
                best_count = count;
                best_id = camera_id;
            }
        }
        return best_id;
    }

    std::expected<void, std::string> PPISP::copy_inference_weights_from(
        const PPISP& source,
        const std::vector<int>& source_frame_indices,
        const std::vector<int>& source_camera_indices) {

        if (!finalized_) {
            return std::unexpected("Target PPISP must be finalized before importing inference weights");
        }
        if (!source.isFinalized()) {
            return std::unexpected("Source PPISP must be finalized before importing inference weights");
        }
        if (static_cast<int>(source_frame_indices.size()) != num_frames_) {
            return std::unexpected("Frame mapping size does not match target PPISP frame count");
        }
        if (static_cast<int>(source_camera_indices.size()) != num_cameras_) {
            return std::unexpected("Camera mapping size does not match target PPISP camera count");
        }

        auto validate_index_range = [](const std::vector<int>& indices, const int upper_bound, const char* label)
            -> std::expected<void, std::string> {
            for (size_t i = 0; i < indices.size(); ++i) {
                if (indices[i] < 0 || indices[i] >= upper_bound) {
                    return std::unexpected(std::string(label) + " mapping contains out-of-range index at position " +
                                           std::to_string(i));
                }
            }
            return {};
        };

        if (auto result = validate_index_range(source_frame_indices, source.num_frames_, "Frame"); !result) {
            return result;
        }
        if (auto result = validate_index_range(source_camera_indices, source.num_cameras_, "Camera"); !result) {
            return result;
        }

        const auto frame_indices = lfs::core::Tensor::from_vector(
                                       source_frame_indices,
                                       {source_frame_indices.size()},
                                       lfs::core::Device::GPU)
                                       .to(lfs::core::DataType::Int32);
        const auto camera_indices = lfs::core::Tensor::from_vector(
                                        source_camera_indices,
                                        {source_camera_indices.size()},
                                        lfs::core::Device::GPU)
                                        .to(lfs::core::DataType::Int32);

        exposure_params_ = source.exposure_params_.index_select(0, frame_indices).contiguous();
        color_params_ = source.color_params_.reshape({source.num_frames_, 8})
                            .index_select(0, frame_indices)
                            .reshape({num_frames_ * 8})
                            .contiguous();

        constexpr int VIGNETTING_PARAM_COUNT = 15;
        constexpr int CRF_PARAM_COUNT = 12;

        vignetting_params_ = source.vignetting_params_.reshape({source.num_cameras_, VIGNETTING_PARAM_COUNT})
                                 .index_select(0, camera_indices)
                                 .reshape({num_cameras_ * VIGNETTING_PARAM_COUNT})
                                 .contiguous();
        crf_params_ = source.crf_params_.reshape({source.num_cameras_, CRF_PARAM_COUNT})
                          .index_select(0, camera_indices)
                          .reshape({num_cameras_ * CRF_PARAM_COUNT})
                          .contiguous();

        exposure_exp_avg_.zero_();
        exposure_exp_avg_sq_.zero_();
        exposure_grad_.zero_();
        vignetting_exp_avg_.zero_();
        vignetting_exp_avg_sq_.zero_();
        vignetting_grad_.zero_();
        color_exp_avg_.zero_();
        color_exp_avg_sq_.zero_();
        color_grad_.zero_();
        crf_exp_avg_.zero_();
        crf_exp_avg_sq_.zero_();
        crf_grad_.zero_();
        step_ = 0;
        current_lr_ = config_.warmup_steps > 0 ? initial_lr_ * config_.warmup_start_factor : initial_lr_;

        return {};
    }

    void PPISP::serialize(std::ostream& os) const {
        os.write(reinterpret_cast<const char*>(&CHECKPOINT_MAGIC), sizeof(CHECKPOINT_MAGIC));
        os.write(reinterpret_cast<const char*>(&CHECKPOINT_VERSION), sizeof(CHECKPOINT_VERSION));

        os.write(reinterpret_cast<const char*>(&num_cameras_), sizeof(num_cameras_));
        os.write(reinterpret_cast<const char*>(&num_frames_), sizeof(num_frames_));
        serialize_config(os, config_);
        os.write(reinterpret_cast<const char*>(&step_), sizeof(step_));
        os.write(reinterpret_cast<const char*>(&current_lr_), sizeof(current_lr_));
        os.write(reinterpret_cast<const char*>(&initial_lr_), sizeof(initial_lr_));
        os.write(reinterpret_cast<const char*>(&total_iterations_), sizeof(total_iterations_));

        os << exposure_params_ << exposure_exp_avg_ << exposure_exp_avg_sq_;
        os << vignetting_params_ << vignetting_exp_avg_ << vignetting_exp_avg_sq_;
        os << color_params_ << color_exp_avg_ << color_exp_avg_sq_;
        os << crf_params_ << crf_exp_avg_ << crf_exp_avg_sq_;

        serialize_int_map(os, camera_id_to_idx_);
        serialize_int_map(os, uid_to_frame_idx_);
        serialize_int_map(os, uid_to_camera_id_);

        const uint8_t has_exif = exif_exposure_mean_.has_value() ? 1 : 0;
        os.write(reinterpret_cast<const char*>(&has_exif), sizeof(has_exif));
        if (has_exif) {
            const float mean = *exif_exposure_mean_;
            os.write(reinterpret_cast<const char*>(&mean), sizeof(mean));
        }
    }

    void PPISP::deserialize(std::istream& is) {
        uint32_t magic = 0, version = 0;
        lfs::core::serialization_detail::read_exact(is, &magic, sizeof(magic), "PPISP magic");
        lfs::core::serialization_detail::read_exact(is, &version, sizeof(version), "PPISP version");

        if (magic != CHECKPOINT_MAGIC) {
            throw std::runtime_error("Invalid PPISP checkpoint");
        }
        if (version < CHECKPOINT_MIN_VERSION || version > CHECKPOINT_VERSION) {
            config_serialization_detail::throw_unsupported_component_version(
                "PPISP", version, CHECKPOINT_MIN_VERSION, CHECKPOINT_VERSION);
        }

        int num_cameras = 0;
        int num_frames = 0;
        Config config{};
        int64_t step = 0;
        double current_lr = 0.0;
        double initial_lr = 0.0;
        int total_iterations = 0;
        lfs::core::serialization_detail::read_exact(is, &num_cameras, sizeof(num_cameras), "PPISP camera count");
        lfs::core::serialization_detail::read_exact(is, &num_frames, sizeof(num_frames), "PPISP frame count");
        config = version == CHECKPOINT_MIN_VERSION
                     ? deserialize_legacy_config(is)
                     : deserialize_config(is);
        lfs::core::serialization_detail::read_exact(is, &step, sizeof(step), "PPISP step");
        lfs::core::serialization_detail::read_exact(is, &current_lr, sizeof(current_lr), "PPISP learning rate");
        lfs::core::serialization_detail::read_exact(is, &initial_lr, sizeof(initial_lr), "PPISP initial learning rate");
        lfs::core::serialization_detail::read_exact(is, &total_iterations, sizeof(total_iterations), "PPISP iteration count");
        const std::array regularization_weights{
            config.exposure_mean,
            config.vig_center,
            config.vig_channel,
            config.vig_non_pos,
            config.color_mean,
            config.crf_channel,
        };
        if (num_cameras <= 0 || num_frames <= 0 || num_cameras > 10'000'000 || num_frames > 10'000'000 ||
            step < 0 || total_iterations <= 0 ||
            !std::isfinite(current_lr) || current_lr < 0.0 ||
            !std::isfinite(initial_lr) || initial_lr < 0.0 ||
            !std::isfinite(config.lr) || config.lr < 0.0 ||
            !std::isfinite(config.beta1) || config.beta1 < 0.0 || config.beta1 >= 1.0 ||
            !std::isfinite(config.beta2) || config.beta2 < 0.0 || config.beta2 >= 1.0 ||
            !std::isfinite(config.eps) || config.eps <= 0.0 || config.warmup_steps < 0 ||
            !std::isfinite(config.warmup_start_factor) || config.warmup_start_factor < 0.0 ||
            !std::isfinite(config.final_lr_factor) || config.final_lr_factor <= 0.0 ||
            std::ranges::any_of(regularization_weights, [](const float weight) { return !std::isfinite(weight) || weight < 0.0f; })) {
            throw std::runtime_error("Invalid PPISP checkpoint state");
        }

        const size_t exposure_size = static_cast<size_t>(num_frames);
        const size_t vig_size = static_cast<size_t>(num_cameras) * 3 * 5;
        const size_t color_size = static_cast<size_t>(num_frames) * 8;
        const size_t crf_size = static_cast<size_t>(num_cameras) * 3 * 4;
        lfs::core::Tensor exposure_params, exposure_exp_avg, exposure_exp_avg_sq;
        lfs::core::Tensor vignetting_params, vignetting_exp_avg, vignetting_exp_avg_sq;
        lfs::core::Tensor color_params, color_exp_avg, color_exp_avg_sq;
        lfs::core::Tensor crf_params, crf_exp_avg, crf_exp_avg_sq;
        is >> exposure_params >> exposure_exp_avg >> exposure_exp_avg_sq;
        is >> vignetting_params >> vignetting_exp_avg >> vignetting_exp_avg_sq;
        is >> color_params >> color_exp_avg >> color_exp_avg_sq;
        is >> crf_params >> crf_exp_avg >> crf_exp_avg_sq;

        const auto require_vector = [](const lfs::core::Tensor& tensor,
                                       const size_t size,
                                       const std::string_view name) {
            if (!tensor.is_valid() || tensor.dtype() != lfs::core::DataType::Float32 ||
                tensor.ndim() != 1 || tensor.numel() != size) {
                throw std::runtime_error("Invalid PPISP checkpoint tensor: " + std::string(name));
            }
        };
        require_vector(exposure_params, exposure_size, "exposure params");
        require_vector(exposure_exp_avg, exposure_size, "exposure exp_avg");
        require_vector(exposure_exp_avg_sq, exposure_size, "exposure exp_avg_sq");
        require_vector(vignetting_params, vig_size, "vignetting params");
        require_vector(vignetting_exp_avg, vig_size, "vignetting exp_avg");
        require_vector(vignetting_exp_avg_sq, vig_size, "vignetting exp_avg_sq");
        require_vector(color_params, color_size, "color params");
        require_vector(color_exp_avg, color_size, "color exp_avg");
        require_vector(color_exp_avg_sq, color_size, "color exp_avg_sq");
        require_vector(crf_params, crf_size, "crf params");
        require_vector(crf_exp_avg, crf_size, "crf exp_avg");
        require_vector(crf_exp_avg_sq, crf_size, "crf exp_avg_sq");

        const auto require_finite = [](lfs::core::Tensor& tensor, const std::string_view name) {
            try {
                tensor.assert_finite();
            } catch (const lfs::core::TensorError&) {
                throw std::runtime_error("Invalid PPISP checkpoint tensor values: " + std::string(name));
            }
        };
        require_finite(exposure_params, "exposure params");
        require_finite(exposure_exp_avg, "exposure exp_avg");
        require_finite(exposure_exp_avg_sq, "exposure exp_avg_sq");
        require_finite(vignetting_params, "vignetting params");
        require_finite(vignetting_exp_avg, "vignetting exp_avg");
        require_finite(vignetting_exp_avg_sq, "vignetting exp_avg_sq");
        require_finite(color_params, "color params");
        require_finite(color_exp_avg, "color exp_avg");
        require_finite(color_exp_avg_sq, "color exp_avg_sq");
        require_finite(crf_params, "crf params");
        require_finite(crf_exp_avg, "crf exp_avg");
        require_finite(crf_exp_avg_sq, "crf exp_avg_sq");

        auto camera_id_to_idx = deserialize_int_map(is, static_cast<uint32_t>(num_cameras), "PPISP camera map");
        auto uid_to_frame_idx = deserialize_int_map(is, static_cast<uint32_t>(num_frames), "PPISP frame map");
        auto uid_to_camera_id = deserialize_int_map(is, static_cast<uint32_t>(num_frames), "PPISP frame-camera map");
        std::optional<float> exif_mean;
        if (version >= 4) {
            uint8_t has_exif = 0;
            lfs::core::serialization_detail::read_exact(is, &has_exif, sizeof(has_exif), "PPISP exif mean flag");
            if (has_exif) {
                float mean = 0.0f;
                lfs::core::serialization_detail::read_exact(is, &mean, sizeof(mean), "PPISP exif mean");
                if (std::isfinite(mean)) {
                    exif_mean = mean;
                }
            }
        }
        if (camera_id_to_idx.size() != static_cast<size_t>(num_cameras) ||
            uid_to_frame_idx.size() != static_cast<size_t>(num_frames) ||
            uid_to_camera_id.size() != static_cast<size_t>(num_frames)) {
            throw std::runtime_error("Invalid PPISP checkpoint map cardinality");
        }
        std::vector<bool> seen_cameras(static_cast<size_t>(num_cameras));
        for (const auto& [_, index] : camera_id_to_idx) {
            if (index < 0 || index >= num_cameras || seen_cameras[static_cast<size_t>(index)])
                throw std::runtime_error("Invalid PPISP checkpoint camera map");
            seen_cameras[static_cast<size_t>(index)] = true;
        }
        std::vector<bool> seen_frames(static_cast<size_t>(num_frames));
        for (const auto& [uid, index] : uid_to_frame_idx) {
            if (index < 0 || index >= num_frames || seen_frames[static_cast<size_t>(index)] ||
                !uid_to_camera_id.contains(uid) || !camera_id_to_idx.contains(uid_to_camera_id.at(uid))) {
                throw std::runtime_error("Invalid PPISP checkpoint frame map");
            }
            seen_frames[static_cast<size_t>(index)] = true;
        }

        exposure_params = exposure_params.gpu();
        exposure_exp_avg = exposure_exp_avg.gpu();
        exposure_exp_avg_sq = exposure_exp_avg_sq.gpu();
        vignetting_params = vignetting_params.gpu();
        vignetting_exp_avg = vignetting_exp_avg.gpu();
        vignetting_exp_avg_sq = vignetting_exp_avg_sq.gpu();
        color_params = color_params.gpu();
        color_exp_avg = color_exp_avg.gpu();
        color_exp_avg_sq = color_exp_avg_sq.gpu();
        crf_params = crf_params.gpu();
        crf_exp_avg = crf_exp_avg.gpu();
        crf_exp_avg_sq = crf_exp_avg_sq.gpu();

        auto exposure_grad = lfs::core::Tensor::zeros({exposure_size}, lfs::core::Device::GPU);
        auto vignetting_grad = lfs::core::Tensor::zeros({vig_size}, lfs::core::Device::GPU);
        auto color_grad = lfs::core::Tensor::zeros({color_size}, lfs::core::Device::GPU);
        auto crf_grad = lfs::core::Tensor::zeros({crf_size}, lfs::core::Device::GPU);
        auto ctrl_bwd_exposure = lfs::core::Tensor::zeros({1}, lfs::core::Device::GPU);
        auto ctrl_bwd_color = lfs::core::Tensor::zeros({8}, lfs::core::Device::GPU);
        auto ctrl_bwd_vignetting = lfs::core::Tensor::zeros({vig_size}, lfs::core::Device::GPU);
        auto ctrl_bwd_crf = lfs::core::Tensor::zeros({crf_size}, lfs::core::Device::GPU);
        auto ctrl_bwd_output = lfs::core::Tensor::empty({9}, lfs::core::Device::GPU);
        auto vig_reg_loss = lfs::core::Tensor::zeros({1}, lfs::core::Device::GPU);

        num_cameras_ = num_cameras;
        num_frames_ = num_frames;
        config_ = config;
        step_ = step;
        current_lr_ = current_lr;
        initial_lr_ = initial_lr;
        total_iterations_ = total_iterations;
        exposure_params_ = std::move(exposure_params);
        exposure_exp_avg_ = std::move(exposure_exp_avg);
        exposure_exp_avg_sq_ = std::move(exposure_exp_avg_sq);
        vignetting_params_ = std::move(vignetting_params);
        vignetting_exp_avg_ = std::move(vignetting_exp_avg);
        vignetting_exp_avg_sq_ = std::move(vignetting_exp_avg_sq);
        color_params_ = std::move(color_params);
        color_exp_avg_ = std::move(color_exp_avg);
        color_exp_avg_sq_ = std::move(color_exp_avg_sq);
        crf_params_ = std::move(crf_params);
        crf_exp_avg_ = std::move(crf_exp_avg);
        crf_exp_avg_sq_ = std::move(crf_exp_avg_sq);
        camera_id_to_idx_ = std::move(camera_id_to_idx);
        uid_to_frame_idx_ = std::move(uid_to_frame_idx);
        uid_to_camera_id_ = std::move(uid_to_camera_id);
        exif_exposure_mean_ = exif_mean;
        exposure_grad_ = std::move(exposure_grad);
        vignetting_grad_ = std::move(vignetting_grad);
        color_grad_ = std::move(color_grad);
        crf_grad_ = std::move(crf_grad);
        ctrl_bwd_exposure_ = std::move(ctrl_bwd_exposure);
        ctrl_bwd_color_ = std::move(ctrl_bwd_color);
        ctrl_bwd_vignetting_ = std::move(ctrl_bwd_vignetting);
        ctrl_bwd_crf_ = std::move(ctrl_bwd_crf);
        ctrl_bwd_output_ = std::move(ctrl_bwd_output);
        vig_reg_loss_ = std::move(vig_reg_loss);
        ctrl_bwd_rgb_h_ = 0;
        ctrl_bwd_rgb_w_ = 0;

        init_color_pinv_block_diag();
        finalized_ = true;
    }

    void PPISP::adopt_checkpoint_state(PPISP& loaded) noexcept {
        std::swap(exposure_params_, loaded.exposure_params_);
        std::swap(exposure_exp_avg_, loaded.exposure_exp_avg_);
        std::swap(exposure_exp_avg_sq_, loaded.exposure_exp_avg_sq_);
        std::swap(exposure_grad_, loaded.exposure_grad_);
        std::swap(vignetting_params_, loaded.vignetting_params_);
        std::swap(vignetting_exp_avg_, loaded.vignetting_exp_avg_);
        std::swap(vignetting_exp_avg_sq_, loaded.vignetting_exp_avg_sq_);
        std::swap(vignetting_grad_, loaded.vignetting_grad_);
        std::swap(color_params_, loaded.color_params_);
        std::swap(color_exp_avg_, loaded.color_exp_avg_);
        std::swap(color_exp_avg_sq_, loaded.color_exp_avg_sq_);
        std::swap(color_grad_, loaded.color_grad_);
        std::swap(crf_params_, loaded.crf_params_);
        std::swap(crf_exp_avg_, loaded.crf_exp_avg_);
        std::swap(crf_exp_avg_sq_, loaded.crf_exp_avg_sq_);
        std::swap(crf_grad_, loaded.crf_grad_);
        std::swap(ctrl_bwd_exposure_, loaded.ctrl_bwd_exposure_);
        std::swap(ctrl_bwd_color_, loaded.ctrl_bwd_color_);
        std::swap(ctrl_bwd_vignetting_, loaded.ctrl_bwd_vignetting_);
        std::swap(ctrl_bwd_crf_, loaded.ctrl_bwd_crf_);
        std::swap(ctrl_bwd_rgb_, loaded.ctrl_bwd_rgb_);
        std::swap(ctrl_bwd_output_, loaded.ctrl_bwd_output_);
        std::swap(ctrl_bwd_rgb_h_, loaded.ctrl_bwd_rgb_h_);
        std::swap(ctrl_bwd_rgb_w_, loaded.ctrl_bwd_rgb_w_);
        std::swap(vig_reg_loss_, loaded.vig_reg_loss_);
        std::swap(config_, loaded.config_);
        std::swap(step_, loaded.step_);
        std::swap(current_lr_, loaded.current_lr_);
        std::swap(initial_lr_, loaded.initial_lr_);
        std::swap(total_iterations_, loaded.total_iterations_);
        std::swap(num_cameras_, loaded.num_cameras_);
        std::swap(num_frames_, loaded.num_frames_);
        camera_id_to_idx_.swap(loaded.camera_id_to_idx_);
        uid_to_frame_idx_.swap(loaded.uid_to_frame_idx_);
        uid_to_camera_id_.swap(loaded.uid_to_camera_id_);
        std::swap(finalized_, loaded.finalized_);
        std::swap(exif_exposure_mean_, loaded.exif_exposure_mean_);
    }

    void PPISP::serialize_inference(std::ostream& os) const {
        constexpr uint32_t INFERENCE_MAGIC = 0x4C465049; // "LFPI" - PPISP Inference
        constexpr uint32_t INFERENCE_VERSION = 1;

        os.write(reinterpret_cast<const char*>(&INFERENCE_MAGIC), sizeof(INFERENCE_MAGIC));
        os.write(reinterpret_cast<const char*>(&INFERENCE_VERSION), sizeof(INFERENCE_VERSION));

        os.write(reinterpret_cast<const char*>(&num_cameras_), sizeof(num_cameras_));
        os.write(reinterpret_cast<const char*>(&num_frames_), sizeof(num_frames_));

        os << exposure_params_;
        os << vignetting_params_;
        os << color_params_;
        os << crf_params_;
    }

    void PPISP::deserialize_inference(std::istream& is) {
        constexpr uint32_t INFERENCE_MAGIC = 0x4C465049;
        constexpr uint32_t INFERENCE_VERSION = 1;

        uint32_t magic, version;
        is.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        is.read(reinterpret_cast<char*>(&version), sizeof(version));

        if (magic != INFERENCE_MAGIC) {
            throw std::runtime_error("Invalid PPISP inference file");
        }
        if (version != INFERENCE_VERSION) {
            throw std::runtime_error("Unsupported PPISP inference version");
        }

        is.read(reinterpret_cast<char*>(&num_cameras_), sizeof(num_cameras_));
        is.read(reinterpret_cast<char*>(&num_frames_), sizeof(num_frames_));

        is >> exposure_params_;
        is >> vignetting_params_;
        is >> color_params_;
        is >> crf_params_;

        exposure_params_ = exposure_params_.gpu();
        vignetting_params_ = vignetting_params_.gpu();
        color_params_ = color_params_.gpu();
        crf_params_ = crf_params_.gpu();

        camera_id_to_idx_.clear();
        uid_to_frame_idx_.clear();
        uid_to_camera_id_.clear();
        for (int i = 0; i < num_cameras_; ++i) {
            camera_id_to_idx_[i] = i;
        }
        for (int i = 0; i < num_frames_; ++i) {
            uid_to_frame_idx_[i] = i;
            uid_to_camera_id_[i] = 0;
        }
        init_color_pinv_block_diag();
        if (!vig_reg_loss_.is_valid()) {
            vig_reg_loss_ = lfs::core::Tensor::zeros({1}, lfs::core::Device::GPU);
        }
        finalized_ = true;
    }

} // namespace lfs::training
