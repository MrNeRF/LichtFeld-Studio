/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "ppisp.hpp"
#include "core/logger.hpp"
#include "core/tensor/internal/tensor_serialization.hpp"
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace lfs::training {

    namespace {
        constexpr uint32_t CHECKPOINT_MAGIC = 0x4C465050; // "LFPP"
        constexpr uint32_t CHECKPOINT_VERSION = 2;

        void serialize_int_map(std::ostream& os, const std::unordered_map<int, int>& m) {
            const auto size = static_cast<uint32_t>(m.size());
            os.write(reinterpret_cast<const char*>(&size), sizeof(size));
            for (const auto& [k, v] : m) {
                os.write(reinterpret_cast<const char*>(&k), sizeof(k));
                os.write(reinterpret_cast<const char*>(&v), sizeof(v));
            }
        }

        void deserialize_int_map(std::istream& is, std::unordered_map<int, int>& m) {
            uint32_t size;
            is.read(reinterpret_cast<char*>(&size), sizeof(size));
            m.clear();
            m.reserve(size);
            for (uint32_t i = 0; i < size; ++i) {
                int k, v;
                is.read(reinterpret_cast<char*>(&k), sizeof(k));
                is.read(reinterpret_cast<char*>(&v), sizeof(v));
                m[k] = v;
            }
        }
    } // namespace

    PPISP::PPISP(int total_iterations, Config config)
        : config_(config),
          current_lr_(config.lr),
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
        exposure_params_ = lfs::core::Tensor::zeros({static_cast<size_t>(num_frames_)}, lfs::core::Device::CUDA);
        exposure_exp_avg_ = lfs::core::Tensor::zeros({static_cast<size_t>(num_frames_)}, lfs::core::Device::CUDA);
        exposure_exp_avg_sq_ = lfs::core::Tensor::zeros({static_cast<size_t>(num_frames_)}, lfs::core::Device::CUDA);
        exposure_grad_ = lfs::core::Tensor::zeros({static_cast<size_t>(num_frames_)}, lfs::core::Device::CUDA);

        // Allocate vignetting params [num_cameras * 3 * 5]
        size_t vig_size = static_cast<size_t>(num_cameras_) * 3 * 5;
        vignetting_params_ = lfs::core::Tensor::zeros({vig_size}, lfs::core::Device::CUDA);
        vignetting_exp_avg_ = lfs::core::Tensor::zeros({vig_size}, lfs::core::Device::CUDA);
        vignetting_exp_avg_sq_ = lfs::core::Tensor::zeros({vig_size}, lfs::core::Device::CUDA);
        vignetting_grad_ = lfs::core::Tensor::zeros({vig_size}, lfs::core::Device::CUDA);

        // Allocate color params [num_frames * 8]
        size_t color_size = static_cast<size_t>(num_frames_) * 8;
        color_params_ = lfs::core::Tensor::zeros({color_size}, lfs::core::Device::CUDA);
        color_exp_avg_ = lfs::core::Tensor::zeros({color_size}, lfs::core::Device::CUDA);
        color_exp_avg_sq_ = lfs::core::Tensor::zeros({color_size}, lfs::core::Device::CUDA);
        color_grad_ = lfs::core::Tensor::zeros({color_size}, lfs::core::Device::CUDA);

        // Allocate CRF params [num_cameras * 3 * 4]
        size_t crf_size = static_cast<size_t>(num_cameras_) * 3 * 4;
        crf_params_ = lfs::core::Tensor::zeros({crf_size}, lfs::core::Device::CUDA);
        crf_exp_avg_ = lfs::core::Tensor::zeros({crf_size}, lfs::core::Device::CUDA);
        crf_exp_avg_sq_ = lfs::core::Tensor::zeros({crf_size}, lfs::core::Device::CUDA);
        crf_grad_ = lfs::core::Tensor::zeros({crf_size}, lfs::core::Device::CUDA);

        // Scratch buffers for backward_with_controller_params
        ctrl_bwd_exposure_ = lfs::core::Tensor::zeros({1}, lfs::core::Device::CUDA);
        ctrl_bwd_color_ = lfs::core::Tensor::zeros({8}, lfs::core::Device::CUDA);
        ctrl_bwd_vignetting_ = lfs::core::Tensor::zeros({vig_size}, lfs::core::Device::CUDA);
        ctrl_bwd_crf_ = lfs::core::Tensor::zeros({crf_size}, lfs::core::Device::CUDA);
        ctrl_bwd_output_ = lfs::core::Tensor::empty({9}, lfs::core::Device::CUDA);

        kernels::launch_ppisp_init_identity(exposure_params_.ptr<float>(), vignetting_params_.ptr<float>(),
                                            color_params_.ptr<float>(), crf_params_.ptr<float>(), num_cameras_,
                                            num_frames_, nullptr);

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
        }, {8, 8}, lfs::core::Device::CUDA);
        // clang-format on
    }

    lfs::core::Tensor PPISP::apply(const lfs::core::Tensor& rgb, int camera_id, int uid) {
        assert(finalized_ && "Must call finalize() before apply()");
        const int camera_idx = translate_camera(camera_id);
        const int frame_idx = translate_frame(uid);

        const auto& shape = rgb.shape();
        assert(shape.rank() == 3 && shape[0] == 3 && "Expected CHW layout with 3 channels");

        const int h = static_cast<int>(shape[1]);
        const int w = static_cast<int>(shape[2]);

        auto output = lfs::core::Tensor::empty({3, shape[1], shape[2]}, lfs::core::Device::CUDA);

        kernels::launch_ppisp_forward_chw(exposure_params_.ptr<float>(), vignetting_params_.ptr<float>(),
                                          color_params_.ptr<float>(), crf_params_.ptr<float>(), rgb.ptr<float>(),
                                          output.ptr<float>(), h, w, num_cameras_, num_frames_, camera_idx, frame_idx,
                                          nullptr);

        return output;
    }

    lfs::core::Tensor PPISP::apply_with_controller_params(const lfs::core::Tensor& rgb,
                                                          const lfs::core::Tensor& controller_params,
                                                          int camera_idx) {
        assert(controller_params.shape().rank() == 2 && "Expected [1,9]");
        assert(controller_params.shape()[0] == 1 && controller_params.shape()[1] == 9);
        assert(camera_idx >= 0 && camera_idx < num_cameras_ && "camera_idx out of range");

        const auto& shape = rgb.shape();
        assert(shape.rank() == 3 && shape[0] == 3 && "Expected CHW layout with 3 channels");

        const int h = static_cast<int>(shape[1]);
        const int w = static_cast<int>(shape[2]);

        // Extract exposure (index 0) and color params (indices 1-8) from controller output
        auto exposure_temp = controller_params.slice(1, 0, 1).reshape({1});
        auto color_temp = controller_params.slice(1, 1, 9).reshape({8});

        auto output = lfs::core::Tensor::empty({3, shape[1], shape[2]}, lfs::core::Device::CUDA);

        // Use controller-predicted exposure and color, but existing vignetting and CRF from camera
        kernels::launch_ppisp_forward_chw(exposure_temp.ptr<float>(), vignetting_params_.ptr<float>(),
                                          color_temp.ptr<float>(), crf_params_.ptr<float>(), rgb.ptr<float>(),
                                          output.ptr<float>(), h, w, num_cameras_, 1, camera_idx, 0, nullptr);

        return output;
    }

    lfs::core::Tensor PPISP::apply_with_controller_params_and_overrides(const lfs::core::Tensor& rgb,
                                                                        const lfs::core::Tensor& controller_params,
                                                                        int camera_idx,
                                                                        const PPISPRenderOverrides& ov) {
        assert(controller_params.shape().rank() == 2 && "Expected [1,9]");
        assert(controller_params.shape()[0] == 1 && controller_params.shape()[1] == 9);
        assert(camera_idx >= 0 && camera_idx < num_cameras_ && "camera_idx out of range");

        const auto& shape = rgb.shape();
        assert(shape.rank() == 3 && shape[0] == 3 && "Expected CHW layout with 3 channels");

        const int h = static_cast<int>(shape[1]);
        const int w = static_cast<int>(shape[2]);

        auto output = lfs::core::Tensor::empty({3, shape[1], shape[2]}, lfs::core::Device::CUDA);

        // Extract and modify exposure from controller output
        auto exposure_temp = controller_params.slice(1, 0, 1).reshape({1}).clone();
        if (ov.exposure_offset != 0.0f) {
            auto exp_cpu = exposure_temp.cpu();
            exp_cpu.ptr<float>()[0] += ov.exposure_offset;
            cudaMemcpy(exposure_temp.ptr<float>(), exp_cpu.ptr<float>(), sizeof(float), cudaMemcpyHostToDevice);
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
            cudaMemcpy(color_temp.ptr<float>(), p, 8 * sizeof(float), cudaMemcpyHostToDevice);
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
            cudaMemcpy(vignetting_modified.ptr<float>() + copy_offset, vig_ptr + copy_offset, 15 * sizeof(float),
                       cudaMemcpyHostToDevice);
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
            cudaMemcpy(crf_modified.ptr<float>() + copy_offset, crf_ptr + copy_offset, 12 * sizeof(float),
                       cudaMemcpyHostToDevice);
        }

        kernels::launch_ppisp_forward_chw(exposure_temp.ptr<float>(), vignetting_modified.ptr<float>(),
                                          color_temp.ptr<float>(), crf_modified.ptr<float>(), rgb.ptr<float>(),
                                          output.ptr<float>(), h, w, num_cameras_, 1, camera_idx, 0, nullptr);

        return output;
    }

    lfs::core::Tensor PPISP::apply_with_overrides(const lfs::core::Tensor& rgb, int camera_id, int uid,
                                                  const PPISPRenderOverrides& ov) {
        assert(finalized_ && "Must call finalize() before apply_with_overrides()");
        const int camera_idx = translate_camera(camera_id);
        const int frame_idx = translate_frame(uid);

        const auto& shape = rgb.shape();
        assert(shape.rank() == 3 && shape[0] == 3 && "Expected CHW layout with 3 channels");

        const int h = static_cast<int>(shape[1]);
        const int w = static_cast<int>(shape[2]);

        auto output = lfs::core::Tensor::empty({3, shape[1], shape[2]}, lfs::core::Device::CUDA);

        // Exposure: add offset to learned value
        auto exposure_modified = exposure_params_.clone();
        if (ov.exposure_offset != 0.0f) {
            auto exp_cpu = exposure_modified.slice(0, frame_idx, frame_idx + 1).cpu();
            exp_cpu.ptr<float>()[0] += ov.exposure_offset;
            cudaMemcpy(exposure_modified.ptr<float>() + frame_idx, exp_cpu.ptr<float>(), sizeof(float),
                       cudaMemcpyHostToDevice);
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
            cudaMemcpy(vignetting_modified.ptr<float>() + copy_offset, vig_ptr + copy_offset, 15 * sizeof(float),
                       cudaMemcpyHostToDevice);
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
            cudaMemcpy(color_modified.ptr<float>() + base, p + base, 8 * sizeof(float), cudaMemcpyHostToDevice);
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
            cudaMemcpy(crf_modified.ptr<float>() + copy_offset, crf_ptr + copy_offset, 12 * sizeof(float),
                       cudaMemcpyHostToDevice);
        }

        kernels::launch_ppisp_forward_chw(exposure_modified.ptr<float>(), vignetting_modified.ptr<float>(),
                                          color_modified.ptr<float>(), crf_modified.ptr<float>(), rgb.ptr<float>(),
                                          output.ptr<float>(), h, w, num_cameras_, num_frames_, camera_idx, frame_idx,
                                          nullptr);

        return output;
    }

    lfs::core::Tensor PPISP::backward(const lfs::core::Tensor& rgb, const lfs::core::Tensor& grad_output, int camera_id,
                                      int uid) {
        assert(finalized_ && "Must call finalize() before backward()");
        const int camera_idx = translate_camera(camera_id);
        const int frame_idx = translate_frame(uid);

        const auto& shape = rgb.shape();
        assert(shape.rank() == 3 && shape[0] == 3 && "Expected CHW layout with 3 channels");

        const int h = static_cast<int>(shape[1]);
        const int w = static_cast<int>(shape[2]);

        auto grad_rgb = lfs::core::Tensor::empty({3, shape[1], shape[2]}, lfs::core::Device::CUDA);

        kernels::launch_ppisp_backward_chw(
            exposure_params_.ptr<float>(), vignetting_params_.ptr<float>(), color_params_.ptr<float>(),
            crf_params_.ptr<float>(), rgb.ptr<float>(), grad_output.ptr<float>(), exposure_grad_.ptr<float>(),
            vignetting_grad_.ptr<float>(), color_grad_.ptr<float>(), crf_grad_.ptr<float>(), grad_rgb.ptr<float>(), h,
            w, num_cameras_, num_frames_, camera_idx, frame_idx, nullptr);

        return grad_rgb;
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
            ctrl_bwd_rgb_ = lfs::core::Tensor::empty({3, h, w}, lfs::core::Device::CUDA);
            ctrl_bwd_rgb_h_ = h;
            ctrl_bwd_rgb_w_ = w;
        }

        auto exposure_temp = controller_params.slice(1, 0, 1).reshape({1});
        auto color_temp = controller_params.slice(1, 1, 9).reshape({8});

        // Zero preallocated gradient scratch buffers
        cudaMemsetAsync(ctrl_bwd_exposure_.ptr<float>(), 0, sizeof(float), nullptr);
        cudaMemsetAsync(ctrl_bwd_color_.ptr<float>(), 0, 8 * sizeof(float), nullptr);
        cudaMemsetAsync(ctrl_bwd_vignetting_.ptr<float>(), 0, ctrl_bwd_vignetting_.numel() * sizeof(float), nullptr);
        cudaMemsetAsync(ctrl_bwd_crf_.ptr<float>(), 0, ctrl_bwd_crf_.numel() * sizeof(float), nullptr);

        kernels::launch_ppisp_backward_chw(exposure_temp.ptr<float>(), vignetting_params_.ptr<float>(),
                                           color_temp.ptr<float>(), crf_params_.ptr<float>(), rgb.ptr<float>(),
                                           grad_output.ptr<float>(), ctrl_bwd_exposure_.ptr<float>(),
                                           ctrl_bwd_vignetting_.ptr<float>(), ctrl_bwd_color_.ptr<float>(),
                                           ctrl_bwd_crf_.ptr<float>(), ctrl_bwd_rgb_.ptr<float>(),
                                           static_cast<int>(h), static_cast<int>(w), num_cameras_, 1, camera_idx, 0,
                                           nullptr);

        // Assemble [exposure(1), color(8)] -> [9] via D2D copy into preallocated output
        cudaMemcpyAsync(ctrl_bwd_output_.ptr<float>(), ctrl_bwd_exposure_.ptr<float>(), sizeof(float),
                        cudaMemcpyDeviceToDevice, nullptr);
        cudaMemcpyAsync(ctrl_bwd_output_.ptr<float>() + 1, ctrl_bwd_color_.ptr<float>(), 8 * sizeof(float),
                        cudaMemcpyDeviceToDevice, nullptr);

        return ctrl_bwd_output_.reshape({1, 9});
    }

    namespace {
        // Smooth L1 loss (Huber loss): 0.5*x^2/beta if |x| < beta, else |x| - 0.5*beta
        inline float smooth_l1(float x, float beta) {
            const float abs_x = std::abs(x);
            if (abs_x < beta) {
                return 0.5f * x * x / beta;
            }
            return abs_x - 0.5f * beta;
        }

        // Gradient of smooth L1: x/beta if |x| < beta, else sign(x)
        inline float smooth_l1_grad(float x, float beta) {
            const float abs_x = std::abs(x);
            if (abs_x < beta) {
                return x / beta;
            }
            return (x > 0.0f) ? 1.0f : -1.0f;
        }
    } // namespace

    lfs::core::Tensor PPISP::reg_loss_gpu() {
        // Compute regularization on CPU (small params, avoid kernel overhead)
        // Transfer to CPU, compute, return GPU scalar for gradient flow
        auto exposure_cpu = exposure_params_.cpu();
        auto vignetting_cpu = vignetting_params_.cpu();
        auto color_cpu = color_params_.cpu();
        auto crf_cpu = crf_params_.cpu();

        const float* exp_ptr = exposure_cpu.ptr<float>();
        const float* vig_ptr = vignetting_cpu.ptr<float>();
        const float* color_ptr = color_cpu.ptr<float>();
        const float* crf_ptr = crf_cpu.ptr<float>();

        float total_loss = 0.0f;

        // 1. Exposure mean regularization: smooth_l1(mean(exposure), beta=0.1)
        if (config_.exposure_mean > 0.0f) {
            float exp_sum = 0.0f;
            for (int i = 0; i < num_frames_; ++i) {
                exp_sum += exp_ptr[i];
            }
            const float exp_mean = exp_sum / static_cast<float>(num_frames_);
            total_loss += config_.exposure_mean * smooth_l1(exp_mean, 0.1f);
        }

        // Vignetting layout: [num_cameras * 3 * 5] = [cam][channel][cx, cy, alpha0, alpha1, alpha2]
        // 2. Vignetting center loss: mean(cx^2 + cy^2)
        if (config_.vig_center > 0.0f) {
            float vig_center_sum = 0.0f;
            for (int cam = 0; cam < num_cameras_; ++cam) {
                for (int ch = 0; ch < 3; ++ch) {
                    size_t base = static_cast<size_t>(cam) * 15 + static_cast<size_t>(ch) * 5;
                    float cx = vig_ptr[base + 0];
                    float cy = vig_ptr[base + 1];
                    vig_center_sum += cx * cx + cy * cy;
                }
            }
            total_loss += config_.vig_center * vig_center_sum / static_cast<float>(num_cameras_ * 3);
        }

        // 3. Vignetting non-positivity: mean(relu(alphas))
        if (config_.vig_non_pos > 0.0f) {
            float vig_non_pos_sum = 0.0f;
            for (int cam = 0; cam < num_cameras_; ++cam) {
                for (int ch = 0; ch < 3; ++ch) {
                    size_t base = static_cast<size_t>(cam) * 15 + static_cast<size_t>(ch) * 5;
                    for (int a = 0; a < 3; ++a) {
                        float alpha = vig_ptr[base + 2 + a];
                        if (alpha > 0.0f) {
                            vig_non_pos_sum += alpha;
                        }
                    }
                }
            }
            total_loss += config_.vig_non_pos * vig_non_pos_sum / static_cast<float>(num_cameras_ * 3 * 3);
        }

        // 4. Vignetting channel variance: mean(var(vig, dim=channel))
        if (config_.vig_channel > 0.0f) {
            float vig_var_sum = 0.0f;
            for (int cam = 0; cam < num_cameras_; ++cam) {
                // For each of the 5 param indices, compute variance across 3 channels
                for (int p = 0; p < 5; ++p) {
                    float vals[3];
                    for (int ch = 0; ch < 3; ++ch) {
                        size_t idx = static_cast<size_t>(cam) * 15 + static_cast<size_t>(ch) * 5 + p;
                        vals[ch] = vig_ptr[idx];
                    }
                    float mean = (vals[0] + vals[1] + vals[2]) / 3.0f;
                    float var = 0.0f;
                    for (int ch = 0; ch < 3; ++ch) {
                        float diff = vals[ch] - mean;
                        var += diff * diff;
                    }
                    var /= 3.0f; // unbiased=False
                    vig_var_sum += var;
                }
            }
            total_loss += config_.vig_channel * vig_var_sum / static_cast<float>(num_cameras_ * 5);
        }

        // 5. Color mean regularization: smooth_l1(mean(color @ pinv, dim=0), beta=0.005)
        if (config_.color_mean > 0.0f) {
            auto pinv_cpu = color_pinv_block_diag_.cpu();
            const float* pinv_ptr = pinv_cpu.ptr<float>();

            // Compute color_offsets = color_params @ pinv (matrix multiply [num_frames, 8] @ [8, 8])
            // Then compute mean across frames for each of 8 outputs
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
            for (int j = 0; j < 8; ++j) {
                color_mean_offsets[j] /= static_cast<float>(num_frames_);
            }

            // smooth_l1 for each mean offset
            float color_loss = 0.0f;
            for (int j = 0; j < 8; ++j) {
                color_loss += smooth_l1(color_mean_offsets[j], 0.005f);
            }
            total_loss += config_.color_mean * color_loss / 8.0f;
        }

        // 6. CRF channel variance: mean(var(crf, dim=channel))
        // CRF layout: [num_cameras * 3 * 4] = [cam][channel][toe, shoulder, gamma, center]
        if (config_.crf_channel > 0.0f) {
            float crf_var_sum = 0.0f;
            for (int cam = 0; cam < num_cameras_; ++cam) {
                // For each of the 4 param indices, compute variance across 3 channels
                for (int p = 0; p < 4; ++p) {
                    float vals[3];
                    for (int ch = 0; ch < 3; ++ch) {
                        size_t idx = static_cast<size_t>(cam) * 12 + static_cast<size_t>(ch) * 4 + p;
                        vals[ch] = crf_ptr[idx];
                    }
                    float mean = (vals[0] + vals[1] + vals[2]) / 3.0f;
                    float var = 0.0f;
                    for (int ch = 0; ch < 3; ++ch) {
                        float diff = vals[ch] - mean;
                        var += diff * diff;
                    }
                    var /= 3.0f; // unbiased=False
                    crf_var_sum += var;
                }
            }
            total_loss += config_.crf_channel * crf_var_sum / static_cast<float>(num_cameras_ * 4);
        }

        // Return as GPU scalar
        auto loss = lfs::core::Tensor::full({1}, total_loss, lfs::core::Device::CUDA);
        return loss;
    }

    void PPISP::reg_backward() {
        // Compute regularization gradients on CPU (matching reg_loss_gpu)
        auto exposure_cpu = exposure_params_.cpu();
        auto vignetting_cpu = vignetting_params_.cpu();
        auto color_cpu = color_params_.cpu();
        auto crf_cpu = crf_params_.cpu();

        const float* exp_ptr = exposure_cpu.ptr<float>();
        const float* vig_ptr = vignetting_cpu.ptr<float>();
        const float* color_ptr = color_cpu.ptr<float>();
        const float* crf_ptr = crf_cpu.ptr<float>();

        // Allocate gradient buffers
        std::vector<float> exp_grad(num_frames_, 0.0f);
        std::vector<float> vig_grad(num_cameras_ * 3 * 5, 0.0f);
        std::vector<float> color_grad(num_frames_ * 8, 0.0f);
        std::vector<float> crf_grad(num_cameras_ * 3 * 4, 0.0f);

        // 1. Exposure mean gradient
        if (config_.exposure_mean > 0.0f) {
            float exp_sum = 0.0f;
            for (int i = 0; i < num_frames_; ++i) {
                exp_sum += exp_ptr[i];
            }
            const float exp_mean = exp_sum / static_cast<float>(num_frames_);
            const float grad_mean = smooth_l1_grad(exp_mean, 0.1f);
            const float grad_per_elem = config_.exposure_mean * grad_mean / static_cast<float>(num_frames_);
            for (int i = 0; i < num_frames_; ++i) {
                exp_grad[i] += grad_per_elem;
            }
        }

        // 2. Vignetting center gradient: d/d(cx,cy) of (cx^2 + cy^2) = 2*cx, 2*cy
        if (config_.vig_center > 0.0f) {
            const float scale = config_.vig_center * 2.0f / static_cast<float>(num_cameras_ * 3);
            for (int cam = 0; cam < num_cameras_; ++cam) {
                for (int ch = 0; ch < 3; ++ch) {
                    size_t base = static_cast<size_t>(cam) * 15 + static_cast<size_t>(ch) * 5;
                    vig_grad[base + 0] += scale * vig_ptr[base + 0]; // d/d(cx)
                    vig_grad[base + 1] += scale * vig_ptr[base + 1]; // d/d(cy)
                }
            }
        }

        // 3. Vignetting non-positivity gradient: d/d(alpha) of relu(alpha) = 1 if alpha > 0
        if (config_.vig_non_pos > 0.0f) {
            const float scale = config_.vig_non_pos / static_cast<float>(num_cameras_ * 3 * 3);
            for (int cam = 0; cam < num_cameras_; ++cam) {
                for (int ch = 0; ch < 3; ++ch) {
                    size_t base = static_cast<size_t>(cam) * 15 + static_cast<size_t>(ch) * 5;
                    for (int a = 0; a < 3; ++a) {
                        if (vig_ptr[base + 2 + a] > 0.0f) {
                            vig_grad[base + 2 + a] += scale;
                        }
                    }
                }
            }
        }

        // 4. Vignetting channel variance gradient
        if (config_.vig_channel > 0.0f) {
            const float scale = config_.vig_channel / static_cast<float>(num_cameras_ * 5);
            for (int cam = 0; cam < num_cameras_; ++cam) {
                for (int p = 0; p < 5; ++p) {
                    float vals[3];
                    size_t idxs[3];
                    for (int ch = 0; ch < 3; ++ch) {
                        idxs[ch] = static_cast<size_t>(cam) * 15 + static_cast<size_t>(ch) * 5 + p;
                        vals[ch] = vig_ptr[idxs[ch]];
                    }
                    float mean = (vals[0] + vals[1] + vals[2]) / 3.0f;
                    // d/d(x_i) of var = 2*(x_i - mean) / n
                    for (int ch = 0; ch < 3; ++ch) {
                        vig_grad[idxs[ch]] += scale * 2.0f * (vals[ch] - mean) / 3.0f;
                    }
                }
            }
        }

        // 5. Color mean gradient
        if (config_.color_mean > 0.0f) {
            auto pinv_cpu = color_pinv_block_diag_.cpu();
            const float* pinv_ptr = pinv_cpu.ptr<float>();

            // First compute mean offsets (same as forward)
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
            for (int j = 0; j < 8; ++j) {
                color_mean_offsets[j] /= static_cast<float>(num_frames_);
            }

            // Gradient of smooth_l1
            float grad_offsets[8];
            for (int j = 0; j < 8; ++j) {
                grad_offsets[j] = config_.color_mean * smooth_l1_grad(color_mean_offsets[j], 0.005f) / 8.0f;
            }

            // Chain rule: d/d(color) = grad_offsets @ pinv^T / num_frames
            for (int f = 0; f < num_frames_; ++f) {
                for (int k = 0; k < 8; ++k) {
                    float grad = 0.0f;
                    for (int j = 0; j < 8; ++j) {
                        grad += grad_offsets[j] * pinv_ptr[k * 8 + j];
                    }
                    color_grad[f * 8 + k] += grad / static_cast<float>(num_frames_);
                }
            }
        }

        // 6. CRF channel variance gradient
        if (config_.crf_channel > 0.0f) {
            const float scale = config_.crf_channel / static_cast<float>(num_cameras_ * 4);
            for (int cam = 0; cam < num_cameras_; ++cam) {
                for (int p = 0; p < 4; ++p) {
                    float vals[3];
                    size_t idxs[3];
                    for (int ch = 0; ch < 3; ++ch) {
                        idxs[ch] = static_cast<size_t>(cam) * 12 + static_cast<size_t>(ch) * 4 + p;
                        vals[ch] = crf_ptr[idxs[ch]];
                    }
                    float mean = (vals[0] + vals[1] + vals[2]) / 3.0f;
                    for (int ch = 0; ch < 3; ++ch) {
                        crf_grad[idxs[ch]] += scale * 2.0f * (vals[ch] - mean) / 3.0f;
                    }
                }
            }
        }

        // Add gradients to GPU gradient buffers (accumulate)
        auto exp_grad_tensor = lfs::core::Tensor::from_vector(exp_grad, {static_cast<size_t>(num_frames_)},
                                                              lfs::core::Device::CUDA);
        auto vig_grad_tensor = lfs::core::Tensor::from_vector(vig_grad, {vig_grad.size()},
                                                              lfs::core::Device::CUDA);
        auto color_grad_tensor = lfs::core::Tensor::from_vector(color_grad, {color_grad.size()},
                                                                lfs::core::Device::CUDA);
        auto crf_grad_tensor = lfs::core::Tensor::from_vector(crf_grad, {crf_grad.size()},
                                                              lfs::core::Device::CUDA);

        // Accumulate into existing gradients
        exposure_grad_ = exposure_grad_.add(exp_grad_tensor);
        vignetting_grad_ = vignetting_grad_.add(vig_grad_tensor);
        color_grad_ = color_grad_.add(color_grad_tensor);
        crf_grad_ = crf_grad_.add(crf_grad_tensor);
    }

    void PPISP::optimizer_step() {
        float bc1_rcp, bc2_sqrt_rcp;
        compute_bias_corrections(bc1_rcp, bc2_sqrt_rcp);

        const float lr = static_cast<float>(current_lr_);
        const float beta1 = static_cast<float>(config_.beta1);
        const float beta2 = static_cast<float>(config_.beta2);
        const float eps = static_cast<float>(config_.eps);

        // Update exposure
        kernels::launch_ppisp_adam_update(exposure_params_.ptr<float>(), exposure_exp_avg_.ptr<float>(),
                                          exposure_exp_avg_sq_.ptr<float>(), exposure_grad_.ptr<float>(),
                                          static_cast<int>(exposure_params_.numel()), lr, beta1, beta2, bc1_rcp,
                                          bc2_sqrt_rcp, eps, nullptr);

        // Update vignetting
        kernels::launch_ppisp_adam_update(vignetting_params_.ptr<float>(), vignetting_exp_avg_.ptr<float>(),
                                          vignetting_exp_avg_sq_.ptr<float>(), vignetting_grad_.ptr<float>(),
                                          static_cast<int>(vignetting_params_.numel()), lr, beta1, beta2, bc1_rcp,
                                          bc2_sqrt_rcp, eps, nullptr);

        // Update color
        kernels::launch_ppisp_adam_update(color_params_.ptr<float>(), color_exp_avg_.ptr<float>(),
                                          color_exp_avg_sq_.ptr<float>(), color_grad_.ptr<float>(),
                                          static_cast<int>(color_params_.numel()), lr, beta1, beta2, bc1_rcp,
                                          bc2_sqrt_rcp, eps, nullptr);

        // Update CRF
        kernels::launch_ppisp_adam_update(crf_params_.ptr<float>(), crf_exp_avg_.ptr<float>(),
                                          crf_exp_avg_sq_.ptr<float>(), crf_grad_.ptr<float>(),
                                          static_cast<int>(crf_params_.numel()), lr, beta1, beta2, bc1_rcp,
                                          bc2_sqrt_rcp, eps, nullptr);
    }

    void PPISP::zero_grad() {
        cudaMemsetAsync(exposure_grad_.ptr<float>(), 0, exposure_grad_.numel() * sizeof(float), nullptr);
        cudaMemsetAsync(vignetting_grad_.ptr<float>(), 0, vignetting_grad_.numel() * sizeof(float), nullptr);
        cudaMemsetAsync(color_grad_.ptr<float>(), 0, color_grad_.numel() * sizeof(float), nullptr);
        cudaMemsetAsync(crf_grad_.ptr<float>(), 0, crf_grad_.numel() * sizeof(float), nullptr);
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

    lfs::core::Tensor PPISP::get_params_for_frame(int uid) const {
        assert(finalized_ && "Must call finalize() before get_params_for_frame()");
        const int frame_idx = translate_frame(uid);

        // Get exposure param for this frame: exposure_params_[frame_idx]
        auto exposure = exposure_params_.slice(0, frame_idx, frame_idx + 1);

        // Get color params for this frame: color_params_ is flat [num_frames * 8]
        // Extract [frame_idx * 8 : (frame_idx + 1) * 8]
        size_t color_start = static_cast<size_t>(frame_idx) * 8;
        auto color = color_params_.slice(0, color_start, color_start + 8);

        // Concatenate: [1] + [8] = [9], then reshape to [1, 9]
        auto params = lfs::core::Tensor::cat({exposure, color}, 0);
        return params.reshape({1, 9});
    }

    void PPISP::serialize(std::ostream& os) const {
        os.write(reinterpret_cast<const char*>(&CHECKPOINT_MAGIC), sizeof(CHECKPOINT_MAGIC));
        os.write(reinterpret_cast<const char*>(&CHECKPOINT_VERSION), sizeof(CHECKPOINT_VERSION));

        os.write(reinterpret_cast<const char*>(&num_cameras_), sizeof(num_cameras_));
        os.write(reinterpret_cast<const char*>(&num_frames_), sizeof(num_frames_));
        os.write(reinterpret_cast<const char*>(&config_), sizeof(config_));
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
    }

    void PPISP::deserialize(std::istream& is) {
        uint32_t magic, version;
        is.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        is.read(reinterpret_cast<char*>(&version), sizeof(version));

        if (magic != CHECKPOINT_MAGIC) {
            throw std::runtime_error("Invalid PPISP checkpoint");
        }
        if (version != CHECKPOINT_VERSION) {
            throw std::runtime_error("Unsupported PPISP checkpoint version");
        }

        is.read(reinterpret_cast<char*>(&num_cameras_), sizeof(num_cameras_));
        is.read(reinterpret_cast<char*>(&num_frames_), sizeof(num_frames_));
        is.read(reinterpret_cast<char*>(&config_), sizeof(config_));
        is.read(reinterpret_cast<char*>(&step_), sizeof(step_));
        is.read(reinterpret_cast<char*>(&current_lr_), sizeof(current_lr_));
        is.read(reinterpret_cast<char*>(&initial_lr_), sizeof(initial_lr_));
        is.read(reinterpret_cast<char*>(&total_iterations_), sizeof(total_iterations_));

        is >> exposure_params_ >> exposure_exp_avg_ >> exposure_exp_avg_sq_;
        is >> vignetting_params_ >> vignetting_exp_avg_ >> vignetting_exp_avg_sq_;
        is >> color_params_ >> color_exp_avg_ >> color_exp_avg_sq_;
        is >> crf_params_ >> crf_exp_avg_ >> crf_exp_avg_sq_;

        deserialize_int_map(is, camera_id_to_idx_);
        deserialize_int_map(is, uid_to_frame_idx_);
        deserialize_int_map(is, uid_to_camera_id_);

        // Move to CUDA
        exposure_params_ = exposure_params_.cuda();
        exposure_exp_avg_ = exposure_exp_avg_.cuda();
        exposure_exp_avg_sq_ = exposure_exp_avg_sq_.cuda();
        vignetting_params_ = vignetting_params_.cuda();
        vignetting_exp_avg_ = vignetting_exp_avg_.cuda();
        vignetting_exp_avg_sq_ = vignetting_exp_avg_sq_.cuda();
        color_params_ = color_params_.cuda();
        color_exp_avg_ = color_exp_avg_.cuda();
        color_exp_avg_sq_ = color_exp_avg_sq_.cuda();
        crf_params_ = crf_params_.cuda();
        crf_exp_avg_ = crf_exp_avg_.cuda();
        crf_exp_avg_sq_ = crf_exp_avg_sq_.cuda();

        // Recreate gradient buffers
        exposure_grad_ = lfs::core::Tensor::zeros({static_cast<size_t>(num_frames_)}, lfs::core::Device::CUDA);
        size_t vig_size = static_cast<size_t>(num_cameras_) * 3 * 5;
        vignetting_grad_ = lfs::core::Tensor::zeros({vig_size}, lfs::core::Device::CUDA);
        color_grad_ = lfs::core::Tensor::zeros({static_cast<size_t>(num_frames_) * 8}, lfs::core::Device::CUDA);
        size_t crf_size = static_cast<size_t>(num_cameras_) * 3 * 4;
        crf_grad_ = lfs::core::Tensor::zeros({crf_size}, lfs::core::Device::CUDA);

        // Recreate scratch buffers for backward_with_controller_params
        ctrl_bwd_exposure_ = lfs::core::Tensor::zeros({1}, lfs::core::Device::CUDA);
        ctrl_bwd_color_ = lfs::core::Tensor::zeros({8}, lfs::core::Device::CUDA);
        ctrl_bwd_vignetting_ = lfs::core::Tensor::zeros({vig_size}, lfs::core::Device::CUDA);
        ctrl_bwd_crf_ = lfs::core::Tensor::zeros({crf_size}, lfs::core::Device::CUDA);
        ctrl_bwd_output_ = lfs::core::Tensor::empty({9}, lfs::core::Device::CUDA);
        ctrl_bwd_rgb_h_ = 0;
        ctrl_bwd_rgb_w_ = 0;

        finalized_ = true;
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

        exposure_params_ = exposure_params_.cuda();
        vignetting_params_ = vignetting_params_.cuda();
        color_params_ = color_params_.cuda();
        crf_params_ = crf_params_.cuda();

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
        finalized_ = true;
    }

} // namespace lfs::training
