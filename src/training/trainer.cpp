/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "trainer.hpp"
#include "backward.h" // BWD-A T_eff hist arm/flush
#include "components/bilateral_grid.hpp"
#include "components/ppisp.hpp"
#include "components/ppisp_controller_pool.hpp"
#include "components/ppisp_file.hpp"
#include "components/sparsity_optimizer.hpp"
#include "control/command_api.hpp"
#include "control/control_boundary.hpp"
#include "core/assert.hpp"
#include "core/checked_arithmetic.hpp"
#include "core/checkpoint_format.hpp"
#include "core/cuda/lanczos_resize/lanczos_resize.hpp"
#include "core/cuda/memory_arena.hpp"
#include "core/cuda_error.hpp"
#include "core/cuda_error_typed.hpp"
#include "core/environment.hpp"
#include "core/events.hpp"
#include "core/exif.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/provenance.hpp"
#include "core/scene.hpp"
#include "core/splat_data_transform.hpp"
#include "core/tensor/internal/cuda_stream_context.hpp"
#include "core/tensor/internal/memory_pool.hpp"
#include "core/tensor/internal/size_bucketed_pool.hpp"
#include "depth_anchor_cache.hpp"
#include "diagnostics/vram_profiler.hpp"
#include "io/cache_image_loader.hpp"
#include "io/cuda/image_format_kernels.cuh"
#include "io/filesystem_utils.hpp"
#include "io/project_container.hpp"
#include "io/project_document.hpp"
#include "io/project_recovery.hpp"
#include "io/scene_chapter_adapter.hpp"
#include "kernels/image_kernels.hpp"
#include "lfs/kernels/ssim.cuh"
#include "lfs/training/joint_adam_codec.hpp"
#include "lfs/training/live_model_mutation_guard.hpp"
#include "lfs/training/morton_reorder.hpp"
#include "lfs/training/perf_bench.hpp"
#include "lfs/training/sh_value_codec.hpp"
#include "lfs/training/vram_ledger.hpp"
#include "losses/losses.hpp"
#include "normal_auto_generate.hpp"
#include "optimizer/adam_optimizer.hpp"
#include "python/runner.hpp"
#include "rasterization/fast_rasterizer.hpp"
#include "rasterization/fastgs/rasterization/include/forward.h"
#include "rasterization/gsplat/Ops.h"
#include "rasterization/gsplat_rasterizer.hpp"
#include "strategies/mcmc.hpp"
#include "strategies/strategy_factory.hpp"
#include "strategies/strategy_utils.hpp"
#include "training/kernels/camera_loss_heatmap.cuh"
#include "training/kernels/depth_loss.hpp"
#include "training/kernels/grad_alpha.hpp"
#include "training/kernels/mrnf_kernels.hpp"
#include "training/kernels/normal_consistency_loss.hpp"
#include "training/kernels/normal_loss.hpp"
#include "training/kernels/roi_weight_map.hpp"
#include "training/training_setup.hpp"
#include "training_cropbox_mask.hpp"

#include <bit>
#include <cstdint>

#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <utility>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <expected>
#include <format>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <nvtx3/nvToolsExt.h>
#include <nvtx3/nvToolsExtCudaRt.h>
#include <optional>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <sys/resource.h>
#endif

namespace lfs::training {

    namespace {
        constexpr float CAMERA_LOSS_EMA_ALPHA = 0.2f;
        constexpr int CAMERA_LOSS_PUBLISH_INTERVAL = 16;

        [[nodiscard]] lfs::Error project_snapshot_error(
            const lfs::ErrorCode code,
            std::string detail,
            const lfs::core::SourceSite source) {
            return lfs::make_error(lfs::ErrorInit{
                .code = code,
                .domain = lfs::ErrorDomain::Training,
                .user_message =
                    "The training project snapshot could not be saved.",
                .detail = std::move(detail),
                .detection = source,
            });
        }

        // Dataset-level normal-prior convention resolution. Prior maps come in
        // several flavors (camera-space OpenCV/OpenGL, world-space in the
        // renderer's frame, linear or sRGB-encoded); wrong assumptions feed the
        // loss targets that are tens of degrees off everywhere. Both axes are
        // resolved from data: gamma by unit-norm deviation, frame by visibility
        // consistency (a visible surface must satisfy n_cam . ray < 0 at its
        // own pixel — an n_z test alone is ambiguous at wide FOV).
        struct NormalPriorConvention {
            bool usable = false;
            bool srgb = false;
            bool flip_yz = false;
            bool world_space = false;
            std::array<float, 9> world_rotation{1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f};
            float score = -1.0f;
            std::string description = "unresolved";
        };

        struct NormalPriorDecodedView {
            std::vector<std::array<float, 3>> normals;
            std::vector<std::array<float, 3>> rays;
            std::array<float, 9> w2c{};
        };

        [[nodiscard]] std::array<float, 9> camera_w2c_rotation_array(const lfs::core::Camera& cam) {
            std::array<float, 9> result{};
            auto R_cpu = cam.R().cpu().contiguous();
            auto R_acc = R_cpu.accessor<float, 2>();
            for (size_t row = 0; row < 3; ++row) {
                for (size_t col = 0; col < 3; ++col) {
                    result[row * 3 + col] = R_acc(row, col);
                }
            }
            return result;
        }

        [[nodiscard]] std::array<float, 3> apply_rotation(
            const std::array<float, 9>& m, const std::array<float, 3>& v) {
            return {m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
                    m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
                    m[6] * v[0] + m[7] * v[1] + m[8] * v[2]};
        }

        // The 24 proper signed-permutation rotations: every axis-convention
        // difference between a renderer world frame and the reconstruction
        // world frame (e.g. Blender z-up vs COLMAP y-down) is one of these.
        [[nodiscard]] std::vector<std::array<float, 9>> proper_signed_permutations() {
            constexpr int kPermutations[6][3] = {
                {0, 1, 2},
                {0, 2, 1},
                {1, 0, 2},
                {1, 2, 0},
                {2, 0, 1},
                {2, 1, 0}};
            constexpr int kPermutationSign[6] = {1, -1, -1, 1, 1, -1};
            std::vector<std::array<float, 9>> rotations;
            rotations.reserve(24);
            for (int p = 0; p < 6; ++p) {
                for (int sign_bits = 0; sign_bits < 8; ++sign_bits) {
                    const int signs[3] = {
                        (sign_bits & 1) ? -1 : 1,
                        (sign_bits & 2) ? -1 : 1,
                        (sign_bits & 4) ? -1 : 1};
                    if (kPermutationSign[p] * signs[0] * signs[1] * signs[2] != 1) {
                        continue;
                    }
                    std::array<float, 9> m{};
                    for (int row = 0; row < 3; ++row) {
                        m[row * 3 + kPermutations[p][row]] = static_cast<float>(signs[row]);
                    }
                    rotations.push_back(m);
                }
            }
            return rotations;
        }

        [[nodiscard]] std::string describe_axis_rotation(const std::array<float, 9>& m) {
            constexpr char kAxes[3] = {'x', 'y', 'z'};
            std::string out = "[";
            for (int row = 0; row < 3; ++row) {
                for (int col = 0; col < 3; ++col) {
                    const float v = m[row * 3 + col];
                    if (v != 0.0f) {
                        if (v < 0.0f) {
                            out += '-';
                        }
                        out += kAxes[col];
                    }
                }
                out += (row < 2) ? ", " : "]";
            }
            return out;
        }

        [[nodiscard]] float score_normal_prior_frame(
            const std::vector<NormalPriorDecodedView>& views,
            const bool flip_yz,
            const std::array<float, 9>* world_rotation) {
            size_t total = 0;
            size_t facing = 0;
            for (const auto& view : views) {
                for (size_t i = 0; i < view.normals.size(); ++i) {
                    std::array<float, 3> n = view.normals[i];
                    if (flip_yz) {
                        n[1] = -n[1];
                        n[2] = -n[2];
                    }
                    if (world_rotation) {
                        n = apply_rotation(view.w2c, apply_rotation(*world_rotation, n));
                    }
                    const auto& ray = view.rays[i];
                    ++total;
                    if (n[0] * ray[0] + n[1] * ray[1] + n[2] * ray[2] < 0.0f) {
                        ++facing;
                    }
                }
            }
            return total > 0 ? static_cast<float>(facing) / static_cast<float>(total) : -1.0f;
        }

        [[nodiscard]] NormalPriorConvention resolve_normal_prior_convention(
            const std::vector<const lfs::core::Camera*>& cameras,
            const lfs::core::param::NormalLossSpace forced_space) {
            constexpr size_t kProbeViews = 16;
            constexpr size_t kProbeSamplesPerView = 4096;
            constexpr float kMinValidNormSq = 0.25f;
            constexpr float kMaxUnitNormDeviation = 0.3f;
            constexpr float kMinAcceptScore = 0.9f;

            NormalPriorConvention result;

            std::vector<float> deviation_linear;
            std::vector<float> deviation_srgb;
            struct RawView {
                std::vector<lfs::core::NormalPriorSample> samples;
                const lfs::core::Camera* cam = nullptr;
            };

            // The probe reads decode full-resolution PNGs (~tens of ms each) and
            // are independent, so fan them out; the serial version dominated
            // training startup on large scenes.
            std::vector<const lfs::core::Camera*> probe_cams;
            const size_t stride = std::max<size_t>(1, cameras.size() / kProbeViews);
            for (size_t i = 0; i < cameras.size() && probe_cams.size() < kProbeViews; i += stride) {
                const auto* cam = cameras[i];
                if (cam->camera_width() <= 0 || cam->camera_height() <= 0 ||
                    cam->focal_x() <= 0.0f || cam->focal_y() <= 0.0f) {
                    continue;
                }
                probe_cams.push_back(cam);
            }

            std::vector<std::future<std::vector<lfs::core::NormalPriorSample>>> probe_futures;
            probe_futures.reserve(probe_cams.size());
            for (const auto* cam : probe_cams) {
                probe_futures.push_back(std::async(std::launch::async, [cam] {
                    return lfs::core::sample_normal_prior_pixels(cam->normal_path(), kProbeSamplesPerView);
                }));
            }

            std::vector<RawView> raw_views;
            raw_views.reserve(probe_cams.size());
            for (size_t idx = 0; idx < probe_cams.size(); ++idx) {
                auto samples = probe_futures[idx].get();
                if (samples.empty()) {
                    continue;
                }
                for (const auto& s : samples) {
                    const float lx = s.r * 2.0f - 1.0f;
                    const float ly = s.g * 2.0f - 1.0f;
                    const float lz = s.b * 2.0f - 1.0f;
                    const float sx = lfs::core::srgb_encoding_to_linear(s.r) * 2.0f - 1.0f;
                    const float sy = lfs::core::srgb_encoding_to_linear(s.g) * 2.0f - 1.0f;
                    const float sz = lfs::core::srgb_encoding_to_linear(s.b) * 2.0f - 1.0f;
                    const float norm_lin_sq = lx * lx + ly * ly + lz * lz;
                    const float norm_srgb_sq = sx * sx + sy * sy + sz * sz;
                    if (norm_lin_sq >= kMinValidNormSq) {
                        deviation_linear.push_back(std::abs(std::sqrt(norm_lin_sq) - 1.0f));
                    }
                    if (norm_srgb_sq >= kMinValidNormSq) {
                        deviation_srgb.push_back(std::abs(std::sqrt(norm_srgb_sq) - 1.0f));
                    }
                }
                raw_views.push_back(RawView{std::move(samples), probe_cams[idx]});
            }
            if (raw_views.empty() || (deviation_linear.empty() && deviation_srgb.empty())) {
                result.description = "no probeable normal maps";
                return result;
            }

            const auto median_of = [](std::vector<float>& values) {
                if (values.empty()) {
                    return std::numeric_limits<float>::infinity();
                }
                const size_t mid = values.size() / 2;
                std::nth_element(values.begin(), values.begin() + mid, values.end());
                return values[mid];
            };
            const float median_linear = median_of(deviation_linear);
            const float median_srgb = median_of(deviation_srgb);
            result.srgb = median_srgb < median_linear;
            const float best_deviation = result.srgb ? median_srgb : median_linear;
            if (best_deviation > kMaxUnitNormDeviation) {
                result.description = "pixels are not unit normals under linear or sRGB decoding";
                return result;
            }

            std::vector<NormalPriorDecodedView> views;
            views.reserve(raw_views.size());
            for (const auto& raw : raw_views) {
                NormalPriorDecodedView view;
                view.w2c = camera_w2c_rotation_array(*raw.cam);
                const float width = static_cast<float>(raw.cam->camera_width());
                const float height = static_cast<float>(raw.cam->camera_height());
                const float fx = raw.cam->focal_x();
                const float fy = raw.cam->focal_y();
                const float cx = raw.cam->center_x();
                const float cy = raw.cam->center_y();
                view.normals.reserve(raw.samples.size());
                view.rays.reserve(raw.samples.size());
                for (const auto& s : raw.samples) {
                    std::array<float, 3> n;
                    if (result.srgb) {
                        n = {lfs::core::srgb_encoding_to_linear(s.r) * 2.0f - 1.0f,
                             lfs::core::srgb_encoding_to_linear(s.g) * 2.0f - 1.0f,
                             lfs::core::srgb_encoding_to_linear(s.b) * 2.0f - 1.0f};
                    } else {
                        n = {s.r * 2.0f - 1.0f, s.g * 2.0f - 1.0f, s.b * 2.0f - 1.0f};
                    }
                    if (n[0] * n[0] + n[1] * n[1] + n[2] * n[2] < kMinValidNormSq) {
                        continue;
                    }
                    view.normals.push_back(n);
                    view.rays.push_back({(s.u * width - cx) / fx,
                                         (s.v * height - cy) / fy,
                                         1.0f});
                }
                if (!view.normals.empty()) {
                    views.push_back(std::move(view));
                }
            }
            if (views.empty()) {
                result.description = "no valid normal samples";
                return result;
            }

            struct Hypothesis {
                bool flip_yz = false;
                bool world_space = false;
                std::array<float, 9> world_rotation{};
                std::string description;
            };
            std::vector<Hypothesis> hypotheses;
            if (forced_space == lfs::core::param::NormalLossSpace::Auto ||
                forced_space == lfs::core::param::NormalLossSpace::CameraOpenCV) {
                hypotheses.push_back({false, false, {}, "camera-space OpenCV"});
            }
            if (forced_space == lfs::core::param::NormalLossSpace::Auto ||
                forced_space == lfs::core::param::NormalLossSpace::CameraOpenGL) {
                hypotheses.push_back({true, false, {}, "camera-space OpenGL"});
            }
            if (forced_space == lfs::core::param::NormalLossSpace::Auto ||
                forced_space == lfs::core::param::NormalLossSpace::World) {
                for (const auto& rotation : proper_signed_permutations()) {
                    hypotheses.push_back({false, true, rotation,
                                          "world-space, n_world = " + describe_axis_rotation(rotation) + " of prior"});
                }
            }

            const Hypothesis* best = nullptr;
            for (const auto& hypothesis : hypotheses) {
                const float score = score_normal_prior_frame(
                    views, hypothesis.flip_yz,
                    hypothesis.world_space ? &hypothesis.world_rotation : nullptr);
                if (score > result.score) {
                    result.score = score;
                    best = &hypothesis;
                }
            }
            if (best == nullptr) {
                result.description = "no frame hypothesis evaluated";
                return result;
            }

            result.flip_yz = best->flip_yz;
            result.world_space = best->world_space;
            if (best->world_space) {
                result.world_rotation = best->world_rotation;
            }
            result.description = best->description;
            const bool forced = forced_space != lfs::core::param::NormalLossSpace::Auto;
            result.usable = forced || result.score >= kMinAcceptScore;
            return result;
        }

        constexpr float kDepthLossFinalScale = 0.02f;
        constexpr float kDepthLossGradientTermWeight = 1.0f;

        [[nodiscard]] kernels::DepthPriorType depth_prior_from_mode(const std::string_view mode) {
            if (mode == "ssi-disparity") {
                return kernels::DepthPriorType::Disparity;
            }
            if (mode == "ssi-depth") {
                return kernels::DepthPriorType::Depth;
            }
            return kernels::DepthPriorType::Auto;
        }

        [[nodiscard]] bool depth_prior_mode_supported(const std::string_view mode) {
            return mode == "ssi" || mode == "ssi-disparity" || mode == "ssi-depth";
        }

        [[nodiscard]] const char* depth_prior_name(const kernels::DepthPriorType prior) {
            switch (prior) {
            case kernels::DepthPriorType::Disparity:
                return "ssi-disparity";
            case kernels::DepthPriorType::Depth:
                return "ssi-depth";
            case kernels::DepthPriorType::Auto:
            default:
                return "ssi";
            }
        }

        [[nodiscard]] const kernels::DepthAnchorCandidate* depth_anchor_candidate_for_prior(
            const kernels::DepthAnchor& anchor,
            const kernels::DepthPriorType prior) {
            switch (prior) {
            case kernels::DepthPriorType::Disparity:
                return &anchor.disparity;
            case kernels::DepthPriorType::Depth:
                return &anchor.depth;
            case kernels::DepthPriorType::Auto:
            default:
                return nullptr;
            }
        }

        void select_depth_anchor_candidate(
            kernels::DepthAnchor& anchor,
            const kernels::DepthPriorType prior,
            const kernels::DepthAnchorCandidate& candidate) {
            anchor.valid = candidate.valid;
            anchor.model = prior == kernels::DepthPriorType::Depth ? 1 : 0;
            anchor.scale = candidate.scale;
            anchor.shift = candidate.shift;
            anchor.corr = candidate.corr;
            anchor.samples = candidate.samples;
        }

        struct DepthAnchorCandidateStats {
            size_t count = 0;
            double abs_corr_sum = 0.0;

            void add(const kernels::DepthAnchorCandidate& candidate) {
                if (!candidate.valid) {
                    return;
                }
                ++count;
                abs_corr_sum += std::fabs(candidate.corr);
            }

            [[nodiscard]] double mean_abs_corr() const {
                return count > 0 ? abs_corr_sum / static_cast<double>(count) : 0.0;
            }
        };

        [[nodiscard]] size_t tensor_reserved_bytes(const lfs::core::Tensor& tensor) {
            if (!tensor.is_valid()) {
                return 0;
            }

            if (tensor.capacity() == 0 || tensor.ndim() == 0) {
                return tensor.bytes();
            }

            size_t row_elems = 1;
            if (tensor.ndim() > 1) {
                for (size_t dim = 1; dim < tensor.ndim(); ++dim) {
                    row_elems *= tensor.shape()[dim];
                }
            }

            return tensor.capacity() * row_elems * lfs::core::dtype_size(tensor.dtype());
        }

        template <typename Entries>
        void add_tensor_entry(Entries& entries, std::string label, const lfs::core::Tensor& tensor) {
            const size_t bytes = tensor_reserved_bytes(tensor);
            if (bytes > 0) {
                entries.emplace_back(std::move(label), bytes);
            }
        }

        struct LoadedCameraMetricsInputs {
            lfs::core::Tensor gt_image;
            lfs::core::Tensor mask;
        };

        [[nodiscard]] lfs::io::LoadParams make_metrics_load_params(
            const Trainer::GTLoadConfigSnapshot& gt_config,
            const lfs::core::Camera& camera,
            const bool apply_undistort,
            const bool output_uint8 = false) {
            lfs::io::LoadParams params;
            params.resize_factor = gt_config.resize_factor;
            params.max_width = gt_config.max_width;
            params.output_uint8 = output_uint8;
            if (apply_undistort && camera.is_undistort_prepared()) {
                params.undistort = &camera.undistort_params();
            }
            return params;
        }

        [[nodiscard]] lfs::core::Tensor normalize_mask_tensor(lfs::core::Tensor mask) {
            if (!mask.is_valid()) {
                return {};
            }

            if (mask.device() != lfs::core::Device::CUDA) {
                mask = mask.to(lfs::core::Device::CUDA);
            }

            if (mask.ndim() == 3 && mask.shape()[0] >= 3) {
                const auto r = mask.slice(0, 0, 1).squeeze(0);
                const auto g = mask.slice(0, 1, 2).squeeze(0);
                const auto b = mask.slice(0, 2, 3).squeeze(0);
                mask = ((r + g + b) / 3.0f).contiguous();
            } else if (mask.ndim() == 3 && mask.shape()[0] == 1) {
                mask = mask.squeeze(0).contiguous();
            }

            return mask;
        }

        std::expected<lfs::core::Tensor, std::string> load_external_mask_for_metrics(
            const lfs::core::Camera& camera,
            const Trainer::GTLoadConfigSnapshot& gt_config,
            const lfs::core::param::OptimizationParameters& opt_params,
            lfs::io::PipelinedImageLoader& image_loader) {
            try {
                auto mask = image_loader.load_image_immediate(
                    camera.mask_path(),
                    make_metrics_load_params(gt_config, camera, false, false));
                mask = normalize_mask_tensor(std::move(mask));
                if (!mask.is_valid()) {
                    return std::unexpected("failed to decode mask");
                }

                const size_t H = mask.shape()[0];
                const size_t W = mask.shape()[1];
                float* const mask_ptr = mask.ptr<float>();
                if (opt_params.invert_masks) {
                    lfs::io::cuda::launch_mask_invert(mask_ptr, H, W, nullptr);
                }
                if (opt_params.mask_threshold > 0.0f) {
                    lfs::io::cuda::launch_mask_threshold(mask_ptr, H, W, opt_params.mask_threshold, nullptr);
                }

                if (camera.is_undistort_prepared()) {
                    const auto scaled = lfs::core::scale_undistort_params(
                        camera.undistort_params(),
                        static_cast<int>(W),
                        static_cast<int>(H));
                    mask = lfs::core::undistort_mask(mask, scaled, nullptr);
                }

                return mask.ge(0.5f).to(lfs::core::DataType::UInt8).contiguous();
            } catch (const std::exception& e) {
                return std::unexpected(e.what());
            }
        }

        std::expected<LoadedCameraMetricsInputs, std::string> load_alpha_masked_metrics_inputs(
            const lfs::core::Camera& camera,
            const Trainer::GTLoadConfigSnapshot& gt_config,
            const lfs::core::param::OptimizationParameters& opt_params) {
            try {
                auto [img_data, width, height, channels] = lfs::core::load_image_with_alpha(
                    camera.image_path(), gt_config.resize_factor, gt_config.max_width);

                if (!img_data || channels != 4) {
                    if (img_data) {
                        lfs::core::free_image(img_data);
                    }
                    return std::unexpected("failed to decode RGBA image");
                }

                const auto H = static_cast<size_t>(height);
                const auto W = static_cast<size_t>(width);

                auto cpu_tensor = lfs::core::Tensor::from_blob(
                    img_data, lfs::core::TensorShape({H, W, 4}),
                    lfs::core::Device::CPU, lfs::core::DataType::UInt8);
                auto gpu_uint8 = cpu_tensor.to(lfs::core::Device::CUDA);
                lfs::core::free_image(img_data);

                auto rgb = lfs::core::Tensor::zeros(
                    lfs::core::TensorShape({3, H, W}),
                    lfs::core::Device::CUDA, lfs::core::DataType::UInt8);
                auto mask = lfs::core::Tensor::zeros(
                    lfs::core::TensorShape({H, W}),
                    lfs::core::Device::CUDA, lfs::core::DataType::Float32);

                lfs::io::cuda::launch_uint8_rgba_split_to_uint8_rgb_and_float32_alpha(
                    gpu_uint8.ptr<uint8_t>(), rgb.ptr<uint8_t>(), mask.ptr<float>(), H, W, nullptr);

                if (opt_params.invert_masks) {
                    lfs::io::cuda::launch_mask_invert(mask.ptr<float>(), H, W, nullptr);
                }
                if (opt_params.mask_threshold > 0.0f) {
                    lfs::io::cuda::launch_mask_threshold(mask.ptr<float>(), H, W, opt_params.mask_threshold, nullptr);
                }

                if (camera.is_undistort_prepared()) {
                    const auto scaled = lfs::core::scale_undistort_params(
                        camera.undistort_params(),
                        static_cast<int>(W),
                        static_cast<int>(H));
                    auto rgb_float = rgb.to(lfs::core::DataType::Float32) / 255.0f;
                    rgb_float = lfs::core::undistort_image(rgb_float, scaled, nullptr);
                    auto rgb_uint8 = lfs::core::Tensor::empty(
                        rgb_float.shape(), lfs::core::Device::CUDA, lfs::core::DataType::UInt8);
                    lfs::io::cuda::launch_float32_chw_to_uint8_chw(
                        rgb_float.ptr<float>(),
                        rgb_uint8.ptr<uint8_t>(),
                        rgb_float.shape()[1],
                        rgb_float.shape()[2],
                        rgb_float.shape()[0],
                        nullptr);
                    rgb = std::move(rgb_uint8);
                    mask = lfs::core::undistort_mask(mask, scaled, nullptr);
                }

                mask = mask.ge(0.5f).to(lfs::core::DataType::UInt8).contiguous();
                return LoadedCameraMetricsInputs{.gt_image = std::move(rgb), .mask = std::move(mask)};
            } catch (const std::exception& e) {
                return std::unexpected(e.what());
            }
        }

        std::expected<LoadedCameraMetricsInputs, std::string> load_camera_metrics_inputs(
            const lfs::core::Camera& camera,
            const Trainer::GTLoadConfigSnapshot& gt_config,
            const lfs::core::param::OptimizationParameters& opt_params,
            const std::shared_ptr<lfs::io::PipelinedImageLoader>& image_loader) {
            std::optional<lfs::io::PipelinedImageLoader> fallback_loader;
            lfs::io::PipelinedImageLoader* loader = image_loader.get();
            if (!loader) {
                fallback_loader.emplace();
                loader = &*fallback_loader;
            }

            const auto mask_mode = opt_params.mask_mode;
            const bool use_masking =
                mask_mode == lfs::core::param::MaskMode::Segment ||
                mask_mode == lfs::core::param::MaskMode::Ignore ||
                mask_mode == lfs::core::param::MaskMode::SegmentAndIgnore;

            // Sidecar mask file wins when present; alpha-as-mask is only used as fallback
            // (some datasets ship RGBA images with a degenerate constant alpha alongside
            // real per-pixel masks in masks/, and we must not let the alpha channel mask
            // them out).
            if (use_masking && !camera.has_mask() && opt_params.use_alpha_as_mask && camera.has_alpha()) {
                return load_alpha_masked_metrics_inputs(camera, gt_config, opt_params);
            }

            LoadedCameraMetricsInputs inputs;

            try {
                inputs.gt_image = loader->load_image_immediate(
                    camera.image_path(),
                    make_metrics_load_params(gt_config, camera, true, true));
            } catch (const std::exception& e) {
                return std::unexpected(e.what());
            }

            if (!inputs.gt_image.is_valid()) {
                return std::unexpected("failed to load ground-truth image");
            }

            if (use_masking && camera.has_mask()) {
                auto mask = load_external_mask_for_metrics(camera, gt_config, opt_params, *loader);
                if (!mask) {
                    return std::unexpected(mask.error());
                }
                inputs.mask = std::move(*mask);
            }

            return inputs;
        }

        struct WorkspaceDisclosure {
            size_t required = 0;
            size_t allocated = 0;
        };

        [[nodiscard]] WorkspaceDisclosure
        photometric_workspace_bytes(const losses::PhotometricLoss& photometric_loss) {
            const auto& arena = photometric_loss.arena();
            return {
                .required = arena.required_bytes(),
                .allocated = arena.allocated_bytes(),
            };
        }

        [[nodiscard]] size_t ssim_map_workspace_bytes(const kernels::SSIMMapWorkspace& workspace) {
            return tensor_reserved_bytes(workspace.ssim_map);
        }

        [[nodiscard]] bool live_vram_profiler_enabled() {
            return lfs::diagnostics::VramProfiler::instance().enabled();
        }

        void record_vram_current(std::string_view scope,
                                 std::string_view label,
                                 const size_t bytes,
                                 const bool publish_zero = false,
                                 const lfs::diagnostics::VramAllocationMethod method =
                                     lfs::diagnostics::VramAllocationMethod::External) {
            if (bytes == 0 && !publish_zero) {
                return;
            }
            lfs::diagnostics::VramProfiler::instance().recordCurrentBytes(scope, label, bytes, method);
        }

        void record_vram_tensor(std::string_view scope,
                                std::string_view label,
                                const lfs::core::Tensor& tensor) {
            // C7: Sampled disclosures must not claim Direct. Direct is reserved for
            // hooked cudaMalloc via try_allocate_direct. External storage is External;
            // ordinary CUDA tensors are Unknown (method census only).
            const auto method = tensor.is_external_storage()
                                    ? lfs::diagnostics::VramAllocationMethod::External
                                    : lfs::diagnostics::VramAllocationMethod::Unknown;
            record_vram_current(scope, label, tensor_reserved_bytes(tensor), false, method);
        }

        void record_rasterizer_arena_disclosure(std::string_view scope) {
            auto* arena = lfs::core::GlobalArenaManager::instance().try_get_arena();
            if (!arena) {
                return;
            }
            const auto info = arena->get_memory_info();
            record_vram_current(scope, "arena.capacity", info.arena_capacity);
            record_vram_current(scope, "arena.current_usage", info.current_usage);
            record_vram_current(scope, "arena.peak_usage", info.peak_usage);
            auto& profiler = lfs::diagnostics::VramProfiler::instance();
            profiler.setGauge("vram.audit.rasterizer_arena.required_bytes",
                              static_cast<double>(info.required_bytes));
            profiler.setGauge("vram.audit.rasterizer_arena.allocated_bytes",
                              static_cast<double>(info.arena_capacity));
            profiler.setGauge("vram.audit.rasterizer_arena.current_usage_bytes",
                              static_cast<double>(info.current_usage));
        }

        void resize_rasterizer_arena_at_boundary(std::string_view boundary,
                                                 bool release_all) {
            auto* arena = lfs::core::GlobalArenaManager::instance().try_get_arena();
            if (!arena) {
                return;
            }
            const bool resized = release_all
                                     ? arena->release_at_boundary()
                                     : arena->shrink_to_current_at_boundary();
            if (!resized) {
                LOG_WARN("Rasterizer arena {} boundary could not drain every CUDA device",
                         boundary);
            }
            if (release_all) {
                // B3/B6: gsplat's exact high-water workspaces are retained
                // across iterations (EXACT-2) and released only at a named
                // boundary after the arena has drained.
                (void)release_gsplat_rasterizer_thread_local_caches();
                (void)gsplat_lfs::release_intersect_thread_local_cache();
            }
        }

        void record_vram_entries(std::string_view scope,
                                 const std::vector<std::pair<std::string, size_t>>& entries) {
            for (const auto& [name, bytes] : entries) {
                record_vram_current(scope, name, bytes);
            }
        }

        void record_pipeline_vram_breakdown(const std::shared_ptr<lfs::io::PipelinedImageLoader>& loader) {
            if (!loader) {
                record_vram_current("io.pipeline", "output_queue.images", 0, true);
                record_vram_current("io.pipeline", "output_queue.masks", 0, true);
                record_vram_current("io.pipeline", "pending.images", 0, true);
                record_vram_current("io.pipeline", "pending.masks", 0, true);
                return;
            }

            const auto stats = loader->get_gpu_memory_stats();
            record_vram_current("io.pipeline", "output_queue.images", stats.output_image_bytes, true);
            record_vram_current("io.pipeline", "output_queue.masks", stats.output_mask_bytes, true);
            record_vram_current("io.pipeline", "pending.images", stats.pending_image_bytes, true);
            record_vram_current("io.pipeline", "pending.masks", stats.pending_mask_bytes, true);
        }

        void record_splat_vram_breakdown(const lfs::core::SplatData& splat) {
            std::vector<std::pair<std::string, size_t>> entries;
            add_tensor_entry(entries, "means", splat.means());
            add_tensor_entry(entries, "sh0", splat.sh0());
            add_tensor_entry(entries, "shN", splat.shN());
            add_tensor_entry(entries, "scaling", splat.scaling_raw());
            add_tensor_entry(entries, "rotation", splat.rotation_raw());
            add_tensor_entry(entries, "opacity", splat.opacity_raw());
            add_tensor_entry(entries, "deleted", splat.deleted());
            add_tensor_entry(entries, "densification_info", splat._densification_info);
            record_vram_entries("model.gaussians", entries);
        }

        void record_optimizer_vram_breakdown(const AdamOptimizer& optimizer) {
            for (const auto type : AdamOptimizer::all_param_types()) {
                const auto* state = optimizer.get_state(type);
                if (!state) {
                    continue;
                }
                const std::string prefix =
                    type == ParamType::Means      ? "means"
                    : type == ParamType::Sh0      ? "sh0"
                    : type == ParamType::ShN      ? "shN"
                    : type == ParamType::Scaling  ? "scaling"
                    : type == ParamType::Rotation ? "rotation"
                                                  : "opacity";
                record_vram_tensor("optimizer.adam", prefix + ".grad", state->grad);
                record_vram_tensor("optimizer.adam", prefix + ".exp_avg", state->exp_avg);
            }
        }

        void record_fastgs_vram_breakdown(const FastRasterizeContext& ctx,
                                          const RenderOutput& output,
                                          const lfs::core::Tensor& gt_tile,
                                          const lfs::core::Tensor& bg_tile,
                                          const lfs::core::Tensor& tile_error_map,
                                          const bool run_gaussian_backward,
                                          const std::size_t num_primitives) {
            constexpr std::string_view scope = "rasterizer.fastgs";
            record_vram_current(scope, "forward.per_primitive_buffers", ctx.forward_ctx.per_primitive_buffers_size);
            record_vram_current(scope, "forward.per_tile_buffers", ctx.forward_ctx.per_tile_buffers_size);
            record_vram_current(scope, "forward.sorted_indices_live", ctx.forward_ctx.sorted_primitive_indices_size);
            record_rasterizer_arena_disclosure(scope);
            const std::size_t raster_arena_live =
                ctx.forward_ctx.per_primitive_buffers_size +
                ctx.forward_ctx.per_tile_buffers_size;
            const std::size_t raster_sort_live =
                ctx.forward_ctx.sorted_primitive_indices_size;
            const std::size_t raster_live = raster_arena_live + raster_sort_live;
            auto& profiler = lfs::diagnostics::VramProfiler::instance();
            profiler.setGauge("vram.audit.fastgs_raster_live.required_bytes",
                              static_cast<double>(raster_live));
            profiler.setGauge("vram.audit.fastgs_raster_live.allocated_bytes",
                              static_cast<double>(raster_live));
            profiler.setGauge(
                "vram.audit.fastgs_sort.required_bytes",
                static_cast<double>(
                    fast_lfs::rasterization::sort_workspace_required_bytes()));
            profiler.setGauge(
                "vram.audit.fastgs_sort.allocated_bytes",
                static_cast<double>(
                    fast_lfs::rasterization::sort_workspace_allocated_bytes()));
            if (PerfBenchCollector::enabled()) {
                PerfBenchCollector::instance().set_fastgs_raster_live_bytes(
                    raster_arena_live, raster_sort_live);
            }
            // The exact sort block is disclosed by the gauges above and includes
            // sorted_indices_live. Keep legacy transient rows clear so the HUD
            // does not count the retained block twice.
            record_vram_current(scope, "forward.sort_scratch_transient", 0, true);
            record_vram_current(scope, "forward.sort_total_transient", 0, true);
            record_vram_current(scope, "backward.grad_mean2d_helper", num_primitives * 2 * sizeof(float));
            record_vram_current(scope, "backward.grad_conic_helper", num_primitives * 3 * sizeof(float));
            record_vram_current(scope,
                                "backward.fused_grad_opacity_helper",
                                run_gaussian_backward && ctx.forward_ctx.grad_opacity_helper
                                    ? num_primitives * sizeof(float)
                                    : 0,
                                true);
            record_vram_current(scope,
                                "backward.fused_grad_color_helper",
                                run_gaussian_backward && ctx.forward_ctx.grad_color_helper
                                    ? num_primitives * 3 * sizeof(float)
                                    : 0,
                                true);
            record_vram_tensor(scope, "output.image", output.image);
            record_vram_tensor(scope, "output.alpha", output.alpha);
            record_vram_tensor(scope, "saved.bg_color", ctx.bg_color);
            record_vram_tensor("train.inputs", "gt_tile", gt_tile);
            record_vram_tensor("train.inputs", "background_tile", bg_tile);
            record_vram_tensor("train.losses", "densification_error_map.live", tile_error_map);
        }

        void record_gsplat_vram_breakdown(const GsplatRasterizeContext& ctx,
                                          const RenderOutput& output,
                                          const lfs::core::Tensor& gt_tile,
                                          const lfs::core::Tensor& bg_tile,
                                          const lfs::core::Tensor& tile_error_map) {
            constexpr std::string_view scope = "rasterizer.gsplat";
            if (auto* arena = lfs::core::GlobalArenaManager::instance().try_get_arena()) {
                std::size_t frame_bytes = 0;
                for (const auto& buffer : arena->get_frame_buffers(ctx.frame_id)) {
                    frame_bytes += buffer.size;
                }
                record_vram_current(scope, "arena.frame_buffers", frame_bytes);
            }
            record_rasterizer_arena_disclosure(scope);
            record_vram_current(scope, "forward.isect_ids", static_cast<std::size_t>(ctx.n_isects) * sizeof(std::int64_t));
            record_vram_current(scope, "forward.flatten_ids", static_cast<std::size_t>(ctx.n_isects) * sizeof(std::int32_t));
            record_vram_tensor(scope, "output.image", output.image);
            record_vram_tensor(scope, "output.alpha", output.alpha);
            record_vram_tensor(scope, "camera.K_tensor", ctx.K_tensor);
            record_vram_tensor(scope, "camera.radial_cuda", ctx.radial_cuda);
            record_vram_tensor(scope, "camera.tangential_cuda", ctx.tangential_cuda);
            record_vram_tensor(scope, "camera.thin_prism_cuda", ctx.thin_prism_cuda);
            record_vram_tensor("train.inputs", "gt_tile", gt_tile);
            record_vram_tensor("train.inputs", "background_tile", bg_tile);
            record_vram_tensor("train.losses", "densification_error_map.live", tile_error_map);
        }

        void syncTrainingSceneTopology(lfs::core::Scene* const scene,
                                       const lfs::core::SplatData& model) {
            if (!scene) {
                return;
            }
            scene->syncTrainingModelTopology(static_cast<size_t>(model.size()));
        }

        [[nodiscard]] std::array<float, 3> lerp_color(const std::array<float, 3>& a,
                                                      const std::array<float, 3>& b,
                                                      const float t) {
            return {
                a[0] + (b[0] - a[0]) * t,
                a[1] + (b[1] - a[1]) * t,
                a[2] + (b[2] - a[2]) * t};
        }

        [[nodiscard]] std::array<float, 3> camera_loss_color(const float t) {
            constexpr std::array<float, 3> BEST{0.10f, 0.84f, 0.24f};
            constexpr std::array<float, 3> WORST{0.90f, 0.16f, 0.16f};
            return lerp_color(BEST, WORST, std::clamp(t, 0.0f, 1.0f));
        }

        template <typename Fn>
        class ScopeGuard {
        public:
            explicit ScopeGuard(Fn fn)
                : fn_(std::move(fn)) {}

            ScopeGuard(const ScopeGuard&) = delete;
            ScopeGuard& operator=(const ScopeGuard&) = delete;

            ScopeGuard(ScopeGuard&& other) noexcept
                : fn_(std::move(other.fn_)),
                  active_(other.active_) {
                other.active_ = false;
            }

            ScopeGuard& operator=(ScopeGuard&&) = delete;

            ~ScopeGuard() {
                if (active_) {
                    fn_();
                }
            }

            void release() noexcept { active_ = false; }

        private:
            Fn fn_;
            bool active_ = true;
        };

        template <typename Fn>
        ScopeGuard<Fn> makeScopeGuard(Fn fn) {
            return ScopeGuard<Fn>(std::move(fn));
        }

        struct PipelineMemoryEstimate {
            size_t max_slot_bytes = 0;
        };

        [[nodiscard]] PipelineMemoryEstimate estimatePipelineMemory(
            const std::shared_ptr<CameraDataset>& dataset,
            const lfs::io::PipelinedLoaderConfig& config,
            const PipelinedAuxiliaryImageConfig& aux_config) {
            PipelineMemoryEstimate estimate;
            if (!dataset) {
                return estimate;
            }

            for (const auto& cam : dataset->get_cameras()) {
                if (!cam) {
                    continue;
                }

                const size_t width = static_cast<size_t>(std::max(cam->image_width(), 0));
                const size_t height = static_cast<size_t>(std::max(cam->image_height(), 0));
                if (width == 0 || height == 0) {
                    continue;
                }

                // One in-flight cold request can hold the decoded output beside its
                // upload staging. Sixteen-bit sources become float32 RGB (12 B/px)
                // while retaining a uint16 RGB staging image (6 B/px). Eight-bit
                // requests retain uint8 RGB output and staging (3 B/px each).
                size_t bytes_per_pixel = config.use_16bit_color ? size_t{18} : size_t{6};
                const bool loads_mask =
                    (aux_config.load_masks && cam->has_mask()) ||
                    (aux_config.use_alpha_as_mask && cam->has_alpha());
                if (loads_mask) {
                    bytes_per_pixel = lfs::core::saturating_add(bytes_per_pixel, sizeof(float));
                }
                if (aux_config.load_depths && cam->has_depth()) {
                    bytes_per_pixel = lfs::core::saturating_add(bytes_per_pixel, sizeof(float));
                }
                if (aux_config.load_normals && cam->has_normal()) {
                    // Cached normal decode retains float HWC input beside float CHW output.
                    bytes_per_pixel = lfs::core::saturating_add(
                        bytes_per_pixel, size_t{6} * sizeof(float));
                }

                const size_t pixels = lfs::core::saturating_multiply(width, height);
                estimate.max_slot_bytes = std::max(
                    estimate.max_slot_bytes,
                    lfs::core::saturating_multiply(pixels, bytes_per_pixel));
            }

            return estimate;
        }

        [[nodiscard]] lfs::io::PipelinedLoaderConfig tunePipelinedLoaderConfig(
            lfs::io::PipelinedLoaderConfig config,
            const std::shared_ptr<CameraDataset>& dataset,
            const PipelinedAuxiliaryImageConfig& aux_config) {
            const auto estimate = estimatePipelineMemory(dataset, config, aux_config);
            if (estimate.max_slot_bytes == 0) {
                config.decoder_pool_size = std::min(config.decoder_pool_size, config.jpeg_batch_size);
                return config;
            }

            size_t free_bytes = 0;
            size_t total_bytes = 0;
            if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess || free_bytes == 0 || total_bytes == 0) {
                config.decoder_pool_size = std::min(config.decoder_pool_size, config.jpeg_batch_size);
                return config;
            }

            constexpr float NON_JPEG_THRESHOLD = 0.1f;
            constexpr size_t JPEG_HOT_OUTPUT_QUEUE_SIZE = 2;
            constexpr size_t JPEG_HOT_DECODER_POOL_SIZE = 2;
            const float non_jpeg_ratio = dataset ? dataset->get_non_jpeg_ratio() : 0.0f;
            if (non_jpeg_ratio <= NON_JPEG_THRESHOLD) {
                if (config.output_queue_size > JPEG_HOT_OUTPUT_QUEUE_SIZE) {
                    LOG_INFO(
                        "Reducing JPEG image ready queue {} -> {} (hot path keeps compressed prefetch)",
                        config.output_queue_size,
                        JPEG_HOT_OUTPUT_QUEUE_SIZE);
                    config.output_queue_size = JPEG_HOT_OUTPUT_QUEUE_SIZE;
                }
                if (config.decoder_pool_size > JPEG_HOT_DECODER_POOL_SIZE) {
                    LOG_INFO(
                        "Reducing nvImageCodec decoder pool {} -> {} for JPEG hot path",
                        config.decoder_pool_size,
                        JPEG_HOT_DECODER_POOL_SIZE);
                    config.decoder_pool_size = JPEG_HOT_DECODER_POOL_SIZE;
                }
            }

            constexpr size_t MIN_PIPELINE_BUDGET_BYTES = 256ULL * 1024 * 1024;
            constexpr size_t MAX_PIPELINE_BUDGET_BYTES = 512ULL * 1024 * 1024;
            const size_t target_pipeline_budget = std::max<size_t>(
                1,
                std::min(
                    free_bytes / 2,
                    std::clamp(free_bytes / 32, MIN_PIPELINE_BUDGET_BYTES, MAX_PIPELINE_BUDGET_BYTES)));

            config.prefetch_count = std::max<size_t>(1, config.prefetch_count);
            const size_t recommended_prefetch = std::min(
                config.prefetch_count,
                std::max<size_t>(1, target_pipeline_budget / estimate.max_slot_bytes));

            if (recommended_prefetch < config.prefetch_count) {
                LOG_INFO(
                    "Reducing image pipeline depth {} -> {} (largest slot {:.1f} MB, free VRAM {:.1f} GB)",
                    config.prefetch_count,
                    recommended_prefetch,
                    estimate.max_slot_bytes / (1024.0 * 1024.0),
                    free_bytes / (1024.0 * 1024.0 * 1024.0));
                config.prefetch_count = recommended_prefetch;
            }

            config.jpeg_batch_size = std::min(
                config.jpeg_batch_size,
                std::max<size_t>(1, config.prefetch_count));
            config.output_queue_size = std::min(
                config.output_queue_size,
                std::max<size_t>(1, config.prefetch_count / 2));
            config.decoder_pool_size = std::min(
                std::max<size_t>(1, config.decoder_pool_size),
                std::max<size_t>(1, config.jpeg_batch_size));

            return config;
        }

        PPISPRenderOverrides toRenderOverrides(const PPISPViewportOverrides& ov) {
            PPISPRenderOverrides r;
            r.exposure_offset = ov.exposure_offset;
            r.vignette_enabled = ov.vignette_enabled;
            r.vignette_strength = ov.vignette_strength;
            r.wb_temperature = ov.wb_temperature;
            r.wb_tint = ov.wb_tint;
            r.color_red_x = ov.color_red_x;
            r.color_red_y = ov.color_red_y;
            r.color_green_x = ov.color_green_x;
            r.color_green_y = ov.color_green_y;
            r.color_blue_x = ov.color_blue_x;
            r.color_blue_y = ov.color_blue_y;
            r.gamma_multiplier = ov.gamma_multiplier;
            r.gamma_red = ov.gamma_red;
            r.gamma_green = ov.gamma_green;
            r.gamma_blue = ov.gamma_blue;
            r.crf_toe = ov.crf_toe;
            r.crf_shoulder = ov.crf_shoulder;
            return r;
        }
    } // namespace

    Trainer::CameraLossHeatmapState::~CameraLossHeatmapState() {
        if (copy_stream) {
            LFS_CUDA_LOG_TEARDOWN(cudaStreamSynchronize(copy_stream), copy_stream,
                                  "heatmap state teardown: sync copy stream");
        }
        if (done_event) {
            LFS_CUDA_LOG_TEARDOWN(cudaEventDestroy(done_event), nullptr,
                                  "heatmap state teardown: destroy done event");
        }
        if (ready_event) {
            LFS_CUDA_LOG_TEARDOWN(cudaEventDestroy(ready_event), nullptr,
                                  "heatmap state teardown: destroy ready event");
        }
        if (copy_stream) {
            lfs::core::CudaMemoryPool::instance().release_stream(copy_stream);
            LFS_CUDA_LOG_TEARDOWN(cudaStreamDestroy(copy_stream), copy_stream,
                                  "heatmap state teardown: destroy copy stream");
        }
    }

    void Trainer::cleanup() {
        LOG_DEBUG("Cleaning up trainer for re-initialization");

        // Stop any ongoing operations
        stop_requested_ = true;
        finish_project_writer();
        {
            std::lock_guard lock(
                project_snapshot_mutex_);
            project_step_regression_.reset();
        }

        // Sync callback stream to avoid race conditions
        if (callback_stream_) {
            LFS_CUDA_LOG_TEARDOWN(cudaStreamSynchronize(callback_stream_), callback_stream_,
                                  "cleanup: sync callback stream");
        }
        callback_busy_ = false;

        // Reset all components
        progress_.reset();
        bilateral_grid_.reset();
        ppisp_.reset();
        ppisp_controller_pool_.reset();
        ppisp_exif_exposure_mean_.reset();
        eval_ppisp_applied_.store(0);
        eval_ppisp_exif_.store(0);
        sparsity_optimizer_.reset();
        evaluator_.reset();

        // Clear datasets (will be recreated)
        train_dataset_.reset();
        val_dataset_.reset();

        // Reset flags
        pause_requested_ = false;
        stop_requested_ = false;
        is_paused_ = false;
        {
            std::lock_guard<std::mutex> lock(params_mutex_);
            is_running_ = false;
        }
        training_complete_ = false;
        ready_to_start_ = false;
        current_iteration_ = 0;
        current_loss_ = 0.0f;
        train_dataset_size_ = 0;
        total_cameras_count_ = 0;
        setCameraLossHeatmap(nullptr);

        LOG_DEBUG("Trainer cleanup complete");
    }

    int Trainer::get_regular_iterations() const {
        return static_cast<int>(getParams().optimization.iterations);
    }

    int Trainer::get_active_sparsify_steps() const {
        const auto params = getParams();
        return params.optimization.enable_sparsity
                   ? std::max(0, params.optimization.sparsify_steps)
                   : 0;
    }

    int Trainer::get_sparsity_boundary_iteration() const {
        return get_regular_iterations();
    }

    int Trainer::get_total_iterations() const {
        return get_regular_iterations() + get_active_sparsify_steps();
    }

    lfs::core::param::OptimizationParameters Trainer::get_runtime_optimization_params() const {
        const auto params = getParams();
        auto runtime_params = params.optimization;
        const int sparsify_steps = runtime_params.enable_sparsity
                                       ? std::max(0, runtime_params.sparsify_steps)
                                       : 0;
        runtime_params.iterations = static_cast<size_t>(
            std::max(0, static_cast<int>(runtime_params.iterations) + sparsify_steps));
        return runtime_params;
    }

    void Trainer::sync_strategy_optimization_params() {
        if (strategy_) {
            strategy_->set_optimization_params(get_runtime_optimization_params());
        }
    }

    std::expected<void, std::string> Trainer::initialize_bilateral_grid() {
        if (!params_.optimization.use_bilateral_grid) {
            return {};
        }

        try {
            BilateralGrid::Config config;
            config.lr = params_.optimization.bilateral_grid_lr;

            // BilateralGrid is indexed with cam->uid() in the training loop. Those UIDs stay
            // in the original camera space even when train/val splits are enabled, so the grid
            // must be sized for the full camera set rather than only the training subset.
            bilateral_grid_ = std::make_unique<BilateralGrid>(
                static_cast<int>(total_cameras_count_),
                params_.optimization.bilateral_grid_X,
                params_.optimization.bilateral_grid_Y,
                params_.optimization.bilateral_grid_W,
                get_total_iterations(),
                config);

            LOG_INFO("Bilateral grid initialized: {}x{}x{} for {} camera slots ({} train images)",
                     params_.optimization.bilateral_grid_X,
                     params_.optimization.bilateral_grid_Y,
                     params_.optimization.bilateral_grid_W,
                     total_cameras_count_,
                     train_dataset_size_);

            return {};
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Failed to init bilateral grid: {}", e.what()));
        }
    }

    std::expected<void, std::string> Trainer::initialize_ppisp() {
        if (!params_.optimization.use_ppisp) {
            return {};
        }

        try {
            PPISPConfig config;
            config.lr = params_.optimization.ppisp_lr;
            config.warmup_steps = params_.optimization.ppisp_warmup_steps;
            const float reg_weight = params_.optimization.ppisp_reg_weight;
            config.exposure_mean *= reg_weight;
            config.vig_center *= reg_weight;
            config.vig_channel *= reg_weight;
            config.vig_non_pos *= reg_weight;
            config.color_mean *= reg_weight;
            config.crf_channel *= reg_weight;

            ppisp_ = std::make_unique<PPISP>(get_total_iterations(), config);
            for (const auto& cam : train_dataset_->get_cameras()) {
                if (cam) {
                    ppisp_->register_frame(cam->uid(), cam->camera_id());
                }
            }
            ppisp_->finalize();

            LOG_INFO("PPISP initialized: {} cameras (physical), {} frames, lr={:.2e}, warmup={}, reg_weight={:.2e}",
                     ppisp_->num_cameras(), ppisp_->num_frames(), params_.optimization.ppisp_lr,
                     config.warmup_steps, reg_weight);

            const bool resume = params_.resume_checkpoint.has_value() || params_.resume_project.has_value();
            if (params_.optimization.ppisp_exposure_from_exif && !resume) {
                std::vector<std::pair<int, float>> uid_ev;
                for (const auto& cam : train_dataset_->get_cameras()) {
                    if (!cam || !ppisp_->is_known_frame(cam->uid())) {
                        continue;
                    }
                    const auto ev = lfs::core::exif_exposure_ev_for_training_image(
                        cam->image_path(), params_.dataset.data_path);
                    if (ev) {
                        uid_ev.emplace_back(cam->uid(), static_cast<float>(*ev));
                    }
                }
                const int n = static_cast<int>(uid_ev.size());
                const int m = ppisp_->num_frames();
                if (n == 0) {
                    LOG_DEBUG("PPISP exposure EXIF seed skipped: no tags in {} frames", m);
                } else {
                    double sum = 0.0;
                    for (const auto& item : uid_ev) {
                        sum += static_cast<double>(item.second);
                    }
                    const double mean = sum / static_cast<double>(n);
                    ppisp_exif_exposure_mean_ = static_cast<float>(mean);
                    ppisp_->seed_exposure(uid_ev);
                    LOG_INFO("PPISP exposure seeded from EXIF for {} of {} frames (mean {:.2f} EV)",
                             n, m, mean);
                }
            }

            if (auto result = apply_ppisp_sidecar_if_configured(); !result) {
                return result;
            }

            return {};
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Failed to init PPISP: {}", e.what()));
        }
    }

    std::expected<PPISPFileMetadata, std::string> Trainer::build_ppisp_sidecar_metadata() const {
        if (!ppisp_ || !ppisp_->isFinalized()) {
            return std::unexpected("Cannot build PPISP sidecar metadata before PPISP is initialized");
        }
        if (!train_dataset_) {
            return std::unexpected("Cannot build PPISP sidecar metadata without an active training dataset");
        }

        PPISPFileMetadata metadata;
        metadata.dataset_path_utf8 = lfs::core::path_to_utf8(params_.dataset.data_path);
        metadata.images_folder = params_.dataset.images;
        metadata.camera_ids = ppisp_->ordered_camera_ids();

        for (const auto& cam : train_dataset_->get_cameras()) {
            if (!cam) {
                continue;
            }
            metadata.frame_image_names.push_back(cam->image_name());
            metadata.frame_camera_ids.push_back(cam->camera_id());
        }

        if (static_cast<int>(metadata.frame_image_names.size()) != ppisp_->num_frames() ||
            static_cast<int>(metadata.frame_camera_ids.size()) != ppisp_->num_frames()) {
            return std::unexpected(std::format(
                "PPISP metadata frame mismatch: metadata has {} names / {} camera ids but PPISP has {} frames",
                metadata.frame_image_names.size(),
                metadata.frame_camera_ids.size(),
                ppisp_->num_frames()));
        }
        if (static_cast<int>(metadata.camera_ids.size()) != ppisp_->num_cameras()) {
            return std::unexpected(std::format(
                "PPISP metadata camera mismatch: metadata has {} camera ids but PPISP has {} cameras",
                metadata.camera_ids.size(),
                ppisp_->num_cameras()));
        }

        return metadata;
    }

    std::expected<Trainer::PPISPSidecarMappings, std::string> Trainer::build_ppisp_sidecar_mappings(
        const PPISP& loaded_ppisp,
        const PPISPFileMetadata& metadata,
        const std::filesystem::path& sidecar_path) const {

        if (!ppisp_ || !ppisp_->isFinalized()) {
            return std::unexpected("Cannot apply PPISP sidecar before PPISP initialization is complete");
        }
        if (!train_dataset_) {
            return std::unexpected("Cannot apply PPISP sidecar without an active training dataset");
        }
        if (metadata.empty()) {
            return std::unexpected(std::format(
                "Frozen PPISP sidecar '{}' has no dataset metadata. Older sidecars cannot be verified against the current dataset; resave the source model with sidecar metadata first.",
                lfs::core::path_to_utf8(sidecar_path)));
        }
        if (static_cast<int>(metadata.frame_image_names.size()) != loaded_ppisp.num_frames() ||
            static_cast<int>(metadata.frame_camera_ids.size()) != loaded_ppisp.num_frames()) {
            return std::unexpected(std::format(
                "PPISP sidecar metadata frame count mismatch: metadata has {} names / {} camera ids but sidecar has {} frames",
                metadata.frame_image_names.size(),
                metadata.frame_camera_ids.size(),
                loaded_ppisp.num_frames()));
        }
        if (static_cast<int>(metadata.camera_ids.size()) != loaded_ppisp.num_cameras()) {
            return std::unexpected(std::format(
                "PPISP sidecar metadata camera count mismatch: metadata has {} camera ids but sidecar has {} cameras",
                metadata.camera_ids.size(),
                loaded_ppisp.num_cameras()));
        }

        const auto current_dataset_path = lfs::core::path_to_utf8(params_.dataset.data_path);
        if (!metadata.dataset_path_utf8.empty() && metadata.dataset_path_utf8 != current_dataset_path) {
            LOG_INFO("Frozen PPISP sidecar dataset path differs from current dataset path: '{}' vs '{}'",
                     metadata.dataset_path_utf8, current_dataset_path);
        }
        if (!metadata.images_folder.empty() && metadata.images_folder != params_.dataset.images) {
            LOG_INFO("Frozen PPISP sidecar images folder differs from current training config: '{}' vs '{}'",
                     metadata.images_folder, params_.dataset.images);
        }

        auto make_frame_key = [](std::string_view image_name, int camera_id) {
            return std::format("{}\n{}", image_name, camera_id);
        };

        std::unordered_map<std::string, int> source_frame_index_by_key;
        source_frame_index_by_key.reserve(metadata.frame_image_names.size());
        for (size_t i = 0; i < metadata.frame_image_names.size(); ++i) {
            auto [_, inserted] = source_frame_index_by_key.emplace(
                make_frame_key(metadata.frame_image_names[i], metadata.frame_camera_ids[i]),
                static_cast<int>(i));
            if (!inserted) {
                return std::unexpected(std::format(
                    "PPISP sidecar metadata contains duplicate frame key for image '{}' and camera {}",
                    metadata.frame_image_names[i],
                    metadata.frame_camera_ids[i]));
            }
        }

        PPISPSidecarMappings mappings;
        mappings.frame_mapping.reserve(static_cast<size_t>(ppisp_->num_frames()));
        std::unordered_set<std::string> seen_target_frames;
        seen_target_frames.reserve(static_cast<size_t>(ppisp_->num_frames()));
        for (const auto& cam : train_dataset_->get_cameras()) {
            if (!cam) {
                continue;
            }
            const auto key = make_frame_key(cam->image_name(), cam->camera_id());
            if (!seen_target_frames.insert(key).second) {
                return std::unexpected(std::format(
                    "Current training dataset contains duplicate frame key for image '{}' and camera {}",
                    cam->image_name(),
                    cam->camera_id()));
            }
            const auto it = source_frame_index_by_key.find(key);
            if (it == source_frame_index_by_key.end()) {
                return std::unexpected(std::format(
                    "Frozen PPISP sidecar is missing frame '{}' for camera {}",
                    cam->image_name(),
                    cam->camera_id()));
            }
            mappings.frame_mapping.push_back(it->second);
        }
        if (seen_target_frames.size() != source_frame_index_by_key.size()) {
            return std::unexpected(std::format(
                "Frozen PPISP sidecar dataset mismatch: sidecar has {} frame keys but current training dataset has {}",
                source_frame_index_by_key.size(),
                seen_target_frames.size()));
        }

        std::unordered_map<int, int> source_camera_index_by_id;
        source_camera_index_by_id.reserve(metadata.camera_ids.size());
        for (size_t i = 0; i < metadata.camera_ids.size(); ++i) {
            auto [_, inserted] = source_camera_index_by_id.emplace(metadata.camera_ids[i], static_cast<int>(i));
            if (!inserted) {
                return std::unexpected(std::format(
                    "PPISP sidecar metadata contains duplicate camera id {}",
                    metadata.camera_ids[i]));
            }
        }

        const auto target_camera_ids = ppisp_->ordered_camera_ids();
        mappings.camera_mapping.reserve(target_camera_ids.size());
        for (const int camera_id : target_camera_ids) {
            const auto it = source_camera_index_by_id.find(camera_id);
            if (it == source_camera_index_by_id.end()) {
                return std::unexpected(std::format(
                    "Frozen PPISP sidecar is missing camera id {} required by the current dataset",
                    camera_id));
            }
            mappings.camera_mapping.push_back(it->second);
        }
        if (target_camera_ids.size() != source_camera_index_by_id.size()) {
            return std::unexpected(std::format(
                "Frozen PPISP sidecar dataset mismatch: sidecar has {} camera ids but current training dataset uses {}",
                source_camera_index_by_id.size(),
                target_camera_ids.size()));
        }

        return mappings;
    }

    std::expected<void, std::string> Trainer::apply_ppisp_sidecar_if_configured() {
        if (!should_apply_ppisp_sidecar_on_init()) {
            return {};
        }
        if (!ppisp_ || !ppisp_->isFinalized()) {
            return std::unexpected("Cannot apply PPISP sidecar before PPISP initialization is complete");
        }

        PPISP loaded_ppisp(1);
        PPISPFileMetadata metadata;
        const auto sidecar_path = params_.optimization.ppisp_sidecar_path;

        if (auto result = load_ppisp_file(sidecar_path, loaded_ppisp, nullptr, &metadata); !result) {
            return std::unexpected(std::format(
                "Failed to load frozen PPISP sidecar '{}': {}",
                lfs::core::path_to_utf8(sidecar_path),
                result.error()));
        }

        auto mappings_result = build_ppisp_sidecar_mappings(loaded_ppisp, metadata, sidecar_path);
        if (!mappings_result) {
            return std::unexpected(mappings_result.error());
        }
        auto& mappings = *mappings_result;

        if (static_cast<int>(mappings.frame_mapping.size()) != ppisp_->num_frames()) {
            return std::unexpected(std::format(
                "Frozen PPISP sidecar frame mapping size mismatch: {} mappings for {} target frames",
                mappings.frame_mapping.size(),
                ppisp_->num_frames()));
        }
        if (static_cast<int>(mappings.camera_mapping.size()) != ppisp_->num_cameras()) {
            return std::unexpected(std::format(
                "Frozen PPISP sidecar camera mapping size mismatch: {} mappings for {} target cameras",
                mappings.camera_mapping.size(),
                ppisp_->num_cameras()));
        }

        if (auto result = ppisp_->copy_inference_weights_from(
                loaded_ppisp, mappings.frame_mapping, mappings.camera_mapping);
            !result) {
            return std::unexpected(std::format(
                "Failed to import frozen PPISP weights from '{}': {}",
                lfs::core::path_to_utf8(sidecar_path),
                result.error()));
        }

        LOG_INFO("Loaded frozen PPISP sidecar '{}' ({} cameras, {} frames{})",
                 lfs::core::path_to_utf8(sidecar_path),
                 loaded_ppisp.num_cameras(),
                 loaded_ppisp.num_frames(),
                 ", metadata-mapped");
        return {};
    }

    std::expected<void, std::string> Trainer::initialize_ppisp_controller() {
        if (!params_.optimization.ppisp_use_controller || !params_.optimization.use_ppisp) {
            return {};
        }

        if (!ppisp_) {
            return std::unexpected("PPISP must be initialized before controller");
        }

        try {
            const bool import_frozen_sidecar_controller = should_apply_ppisp_sidecar_on_init();
            const auto sidecar_path = params_.optimization.ppisp_sidecar_path;
            PPISPFileHeader sidecar_header{};
            if (import_frozen_sidecar_controller) {
                std::ifstream file;
                if (!lfs::core::open_file_for_read(sidecar_path, std::ios::binary, file)) {
                    return std::unexpected("Failed to open frozen PPISP sidecar: " +
                                           lfs::core::path_to_utf8(sidecar_path));
                }
                file.read(reinterpret_cast<char*>(&sidecar_header), sizeof(sidecar_header));
                if (!file) {
                    return std::unexpected("Failed to read frozen PPISP sidecar header: " +
                                           lfs::core::path_to_utf8(sidecar_path));
                }
                if (sidecar_header.magic != PPISP_FILE_MAGIC) {
                    return std::unexpected("Invalid frozen PPISP sidecar: wrong magic number");
                }
                if (sidecar_header.version > PPISP_FILE_VERSION) {
                    return std::unexpected("Unsupported frozen PPISP sidecar version: " +
                                           std::to_string(sidecar_header.version));
                }
                if (!has_flag(sidecar_header.flags, PPISPFileFlags::HAS_CONTROLLER)) {
                    LOG_INFO("Frozen PPISP sidecar '{}' has no controller pool; controller inference will remain disabled",
                             lfs::core::path_to_utf8(sidecar_path));
                    return {};
                }
            }

            PPISPControllerPool::Config config;
            config.lr = params_.optimization.ppisp_controller_lr;

            const int activation_step = params_.optimization.resolved_ppisp_controller_activation_step(get_total_iterations());
            if (params_.optimization.ppisp_controller_activation_step < 0) {
                params_.optimization.ppisp_controller_activation_step = activation_step;
            }
            int distillation_iters = get_total_iterations() - activation_step;
            int num_cameras = ppisp_->num_cameras();

            ppisp_controller_pool_ = std::make_unique<PPISPControllerPool>(num_cameras, distillation_iters, config);

            size_t max_h = 0, max_w = 0;
            for (const auto& cam : train_dataset_->get_cameras()) {
                if (cam) {
                    max_h = std::max(max_h, static_cast<size_t>(cam->image_height()));
                    max_w = std::max(max_w, static_cast<size_t>(cam->image_width()));
                }
            }
            ppisp_controller_pool_->allocate_buffers(max_h, max_w);

            LOG_INFO("PPISP controller pool initialized: num_cameras={}, activation_step={}, lr={:.2e}, max_image={}x{}",
                     num_cameras, activation_step,
                     params_.optimization.ppisp_controller_lr, static_cast<int>(max_h), static_cast<int>(max_w));

            if (import_frozen_sidecar_controller) {
                PPISP loaded_ppisp(1);
                auto loaded_controller = std::make_unique<PPISPControllerPool>(
                    static_cast<int>(sidecar_header.num_cameras),
                    1);
                PPISPFileMetadata metadata;
                if (auto result = load_ppisp_file(sidecar_path, loaded_ppisp, loaded_controller.get(), &metadata); !result) {
                    return std::unexpected(std::format(
                        "Failed to load frozen PPISP controller sidecar '{}': {}",
                        lfs::core::path_to_utf8(sidecar_path),
                        result.error()));
                }
                auto mappings_result = build_ppisp_sidecar_mappings(loaded_ppisp, metadata, sidecar_path);
                if (!mappings_result) {
                    return std::unexpected(mappings_result.error());
                }
                if (const auto error = ppisp_controller_pool_->copy_inference_weights_from(
                        *loaded_controller, mappings_result->camera_mapping);
                    !error.empty()) {
                    return std::unexpected(std::format(
                        "Failed to import frozen PPISP controller weights from '{}': {}",
                        lfs::core::path_to_utf8(sidecar_path),
                        error));
                }
                LOG_INFO("Loaded frozen PPISP controller from '{}' ({} cameras)",
                         lfs::core::path_to_utf8(sidecar_path),
                         loaded_controller->num_cameras());
            }

            return {};
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Failed to init PPISP controller pool: {}", e.what()));
        }
    }

    // Compute photometric loss AND gradient manually
    std::expected<Trainer::PhotometricLossResult, std::string> Trainer::compute_photometric_loss_with_gradient(
        const lfs::core::Tensor& corrected,
        const lfs::core::Tensor& gt_image,
        const lfs::core::param::OptimizationParameters& opt_params,
        const lfs::core::Tensor& raw_rendered) {
        const bool use_decoupled_appearance_loss =
            raw_rendered.is_valid() &&
            raw_rendered.numel() > 0 &&
            opt_params.lambda_dssim > 0.0f;

        if (use_decoupled_appearance_loss) {
            auto& decoupled_ws = photometric_loss_.arena().decoupled();
            auto [loss_tensor, ctx] = lfs::training::kernels::decoupled_fused_l1_ssim_forward(
                corrected, raw_rendered, gt_image, opt_params.lambda_dssim, decoupled_ws,
                /*apply_valid_padding=*/true);
            auto grads = lfs::training::kernels::decoupled_fused_l1_ssim_backward(ctx, decoupled_ws);

            if (corrected.ndim() == 3) {
                grads.grad_corrected = grads.grad_corrected.squeeze(0);
                grads.grad_raw = grads.grad_raw.squeeze(0);
            }

            return PhotometricLossResult{
                .loss = loss_tensor,
                .grad_corrected = grads.grad_corrected,
                .grad_raw = grads.grad_raw};
        }

        lfs::training::losses::PhotometricLoss::Params params{.lambda_dssim = opt_params.lambda_dssim};
        auto result = photometric_loss_.forward(corrected, gt_image, params);
        if (!result) {
            return std::unexpected(result.error());
        }
        auto [loss_tensor, ctx] = *result;
        return PhotometricLossResult{
            .loss = loss_tensor,
            .grad_corrected = ctx.grad_image,
            .grad_raw = {}};
    }

    std::expected<void, std::string> Trainer::validate_masks() {
        const auto& opt = params_.optimization;
        if (opt.mask_mode == lfs::core::param::MaskMode::None) {
            return {};
        }

        // Segment and ignore does not support mask invert
        if (opt.mask_mode == lfs::core::param::MaskMode::SegmentAndIgnore && opt.invert_masks) {
            LOG_WARN("invert_masks is ignored in SegmentAndIgnore mode (would scramble the mask bands)");
            params_.optimization.invert_masks = false;
        }

        size_t alpha_count = 0;
        size_t masks_found = 0;
        for (const auto& cam : train_dataset_->get_cameras()) {
            if (cam && cam->has_alpha())
                ++alpha_count;
            if (cam && cam->has_mask())
                ++masks_found;
        }

        // Sidecar masks take precedence over the alpha channel (see dataset prefetch),
        // so report them first; alpha-as-mask only covers cameras without a sidecar.
        if (masks_found > 0) {
            LOG_INFO("Found {} masks{}", masks_found, opt.invert_masks ? " (inverted)" : "");
            return {};
        }

        if (opt.use_alpha_as_mask && alpha_count > 0) {
            LOG_INFO("Using alpha channel as mask source ({}/{} cameras){}",
                     alpha_count, train_dataset_->get_cameras().size(),
                     opt.invert_masks ? " (inverted)" : "");
            return {};
        }

        const auto path_str = lfs::core::path_to_utf8(params_.dataset.data_path);
        if (opt.use_alpha_as_mask) {
            return std::unexpected(std::format(
                "Mask mode enabled with use_alpha_as_mask but no images have alpha and no mask files found in {}/masks/",
                path_str));
        }
        return std::unexpected(std::format(
            "Mask mode enabled but no masks found in {}/masks/",
            path_str));
    }

    std::expected<Trainer::MaskLossResult, std::string> Trainer::compute_photometric_loss_with_mask(
        const lfs::core::Tensor& corrected,
        const lfs::core::Tensor& gt_image,
        const lfs::core::Tensor& mask,
        const lfs::core::Tensor& roi_weight,
        const lfs::core::Tensor& alpha,
        const lfs::core::param::OptimizationParameters& opt_params,
        const lfs::core::Tensor& raw_rendered) {

        using namespace lfs::core;
        constexpr float ALPHA_CONSISTENCY_WEIGHT = 10.0f;

        const auto mode = opt_params.mask_mode;
        const bool has_user_mask = mask.is_valid() && mask.numel() > 0;
        const bool has_roi_weight = roi_weight.is_valid() && roi_weight.numel() > 0;
        const Tensor mask_2d =
            has_user_mask && mask.ndim() == 3 ? mask.squeeze(0) : mask;

        if (has_roi_weight) {
            LFS_ASSERT_MSG(
                roi_weight.dtype() == DataType::Float32 && roi_weight.ndim() == 2,
                "crop box ROI weight must be a 2D Float32 tensor");
        }

        const bool user_masks_photometric =
            has_user_mask &&
            (mode == param::MaskMode::Segment ||
             mode == param::MaskMode::Ignore ||
             mode == param::MaskMode::SegmentAndIgnore);

        // Fused mask preprocess: SegmentAndIgnore band remap + optional ROI → one kernel.
        // Steady state is allocation-free via mask_preprocess_workspace_ (grow-only).
        const Tensor photometric_weight = losses::fuse_photometric_mask_weight(
            mask_preprocess_workspace_,
            user_masks_photometric ? mask_2d : Tensor{},
            roi_weight,
            mode == param::MaskMode::SegmentAndIgnore);

        Tensor loss, grad_corrected, grad_raw, grad_alpha;
        const bool use_decoupled_appearance_loss =
            raw_rendered.is_valid() &&
            raw_rendered.numel() > 0 &&
            opt_params.lambda_dssim > 0.0f;

        if (photometric_weight.is_valid()) {
            if (use_decoupled_appearance_loss) {
                auto& masked_decoupled_ws = photometric_loss_.arena().masked_decoupled();
                auto [loss_tensor, ctx] = lfs::training::kernels::masked_decoupled_fused_l1_ssim_forward(
                    corrected, raw_rendered, gt_image, photometric_weight, opt_params.lambda_dssim,
                    masked_decoupled_ws);
                auto grads = lfs::training::kernels::masked_decoupled_fused_l1_ssim_backward(
                    ctx, masked_decoupled_ws);

                grad_corrected = grads.grad_corrected;
                grad_raw = grads.grad_raw;
                loss = loss_tensor;

                if (grad_corrected.ndim() == 4 && corrected.ndim() == 3) {
                    grad_corrected = grad_corrected.squeeze(0);
                }
                if (grad_raw.ndim() == 4 && corrected.ndim() == 3) {
                    grad_raw = grad_raw.squeeze(0);
                }
            } else {
                auto& masked_ws = photometric_loss_.arena().masked_fused();
                auto [loss_tensor, ctx] = lfs::training::kernels::masked_fused_l1_ssim_forward(
                    corrected, gt_image, photometric_weight, opt_params.lambda_dssim, masked_ws);

                grad_corrected = lfs::training::kernels::masked_fused_l1_ssim_backward(ctx, masked_ws);
                loss = loss_tensor;

                if (grad_corrected.ndim() == 4 && corrected.ndim() == 3) {
                    grad_corrected = grad_corrected.squeeze(0);
                }
            }

            if (has_user_mask &&
                (mode == param::MaskMode::Segment || mode == param::MaskMode::SegmentAndIgnore) &&
                alpha.is_valid()) {
                const Tensor alpha_2d = alpha.ndim() == 3 ? alpha.squeeze(0) : alpha;
                // Fused: band remap + (1-mask)^power + mean/grad (single kernel).
                const auto penalty = losses::fuse_mask_opacity_penalty(
                    mask_preprocess_workspace_,
                    alpha_2d,
                    mask_2d,
                    roi_weight,
                    opt_params.mask_opacity_penalty_power,
                    opt_params.mask_opacity_penalty_weight,
                    mode == param::MaskMode::SegmentAndIgnore);
                grad_alpha = penalty.grad_alpha;
                loss = loss + penalty.loss;
            }
        } else {
            auto fallback = compute_photometric_loss_with_gradient(corrected, gt_image, opt_params, raw_rendered);
            if (!fallback) {
                return std::unexpected(fallback.error());
            }
            if (has_user_mask && mode == param::MaskMode::AlphaConsistent) {
                loss = fallback->loss;
                grad_corrected = fallback->grad_corrected;
                grad_raw = fallback->grad_raw;
            } else {
                return MaskLossResult{
                    .loss = fallback->loss,
                    .grad_corrected = fallback->grad_corrected,
                    .grad_raw = fallback->grad_raw,
                    .grad_alpha = {}};
            }
        }

        if (has_user_mask && mode == param::MaskMode::AlphaConsistent && alpha.is_valid()) {
            const Tensor alpha_2d = alpha.ndim() == 3 ? alpha.squeeze(0) : alpha;
            // Fused: abs/sign + optional ROI + mean/grad (single kernel).
            const auto alpha_term = losses::fuse_alpha_consistent(
                mask_preprocess_workspace_,
                alpha_2d,
                mask_2d,
                roi_weight,
                ALPHA_CONSISTENCY_WEIGHT);
            loss = loss + alpha_term.loss;
            grad_alpha = alpha_term.grad_alpha;
        }

        return MaskLossResult{
            .loss = loss,
            .grad_corrected = grad_corrected,
            .grad_raw = grad_raw,
            .grad_alpha = grad_alpha};
    }

    // Returns GPU tensor for loss - NO SYNC!
    std::expected<lfs::core::Tensor, std::string> Trainer::compute_scale_reg_loss(
        lfs::core::SplatData& splatData,
        AdamOptimizer& optimizer,
        const lfs::core::param::OptimizationParameters& opt_params) {
        lfs::training::losses::ScaleRegularization::Params params{.weight = opt_params.scale_reg};
        return lfs::training::losses::ScaleRegularization::forward(splatData.scaling_raw(), optimizer.get_grad(ParamType::Scaling), params);
    }

    // Returns GPU tensor for loss - NO SYNC!
    std::expected<lfs::core::Tensor, std::string> Trainer::compute_opacity_reg_loss(
        lfs::core::SplatData& splatData,
        AdamOptimizer& optimizer,
        const lfs::core::param::OptimizationParameters& opt_params) {
        lfs::training::losses::OpacityRegularization::Params params{.weight = opt_params.opacity_reg};
        return lfs::training::losses::OpacityRegularization::forward(splatData.opacity_raw(), optimizer.get_grad(ParamType::Opacity), params);
    }

    std::expected<std::pair<lfs::core::Tensor, SparsityLossContext>, std::string>
    Trainer::compute_sparsity_loss_forward(const int iter, const lfs::core::SplatData& splat_data) {
        if (!sparsity_optimizer_ || !sparsity_optimizer_->should_apply_loss(iter)) {
            auto zero = lfs::core::Tensor::zeros({1}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);
            return std::make_pair(std::move(zero), SparsityLossContext{});
        }

        if (!sparsity_optimizer_->is_initialized()) {
            if (auto result = sparsity_optimizer_->initialize(splat_data.opacity_raw()); !result) {
                return std::unexpected(result.error());
            }
            LOG_DEBUG("Sparsity optimizer initialized at iteration {}", iter);
        }

        return sparsity_optimizer_->compute_loss_forward(splat_data.opacity_raw());
    }

    std::expected<void, std::string> Trainer::handle_sparsity_update(const int iter, lfs::core::SplatData& splat_data) {
        if (!sparsity_optimizer_ || !sparsity_optimizer_->should_update(iter)) {
            return {};
        }
        return sparsity_optimizer_->update_state(splat_data.opacity_raw());
    }

    std::expected<void, std::string> Trainer::apply_sparsity_pruning(const int iter, lfs::core::SplatData& splat_data) {
        if (!sparsity_optimizer_ || !sparsity_optimizer_->should_prune(iter)) {
            return {};
        }

        auto mask_result = sparsity_optimizer_->get_prune_mask(splat_data.opacity_raw());
        if (!mask_result) {
            return std::unexpected(mask_result.error());
        }

        const int n_before = static_cast<int>(splat_data.size());
        strategy_->remove_gaussians(*mask_result);
        const int n_after = static_cast<int>(splat_data.size());

        LOG_INFO("Sparsity pruning: {} -> {} Gaussians ({}% reduction)",
                 n_before, n_after, static_cast<int>(100.0f * (n_before - n_after) / n_before));

        sparsity_optimizer_.reset();
        return {};
    }

    bool Trainer::morton_reorder_due(const int iter) const {
        return morton::should_reorder(
            iter,
            params_.optimization.morton_reorder_interval,
            params_.optimization.stop_refine);
    }

    void Trainer::maybe_morton_reorder(const int iter) {
        if (!morton_reorder_due(iter) || !strategy_) {
            return;
        }
        auto& model = strategy_->get_model();
        const auto result = morton::apply_morton_reorder(model, &strategy_->get_optimizer());
        if (!result.applied) {
            return;
        }
        cropbox_damping_cache_valid_ = false;
        strategy_->permute_gaussian_rows(result.permutation);
        install_cropbox_step_damping(model, strategy_->get_optimizer());
    }

    void Trainer::install_cropbox_step_damping(
        lfs::core::SplatData& model,
        AdamOptimizer& optimizer) {
        const float scale = params_.optimization.cropbox_lr_scale;
        optimizer.set_cropbox_lr_scale(scale);

        if (scale == 1.0f || !scene_) {
            optimizer.set_crop_damping_mask({});
            cropbox_damping_cache_valid_ = false;
            return;
        }

        // rebuild only on cropbox geometry / topology (N) / scale change.
        // Means drift between densify events is intentionally not a rebuild trigger
        // (damping is a soft LR scale, not a hard cull).
        const size_t n = static_cast<size_t>(model.size());
        size_t geom_fp = 0;
        if (const auto geometry = resolve_training_cropbox_geom(*scene_)) {
            // FNV-ish fingerprint of crop bounds + inverse flag + a few matrix entries.
            auto mix = [&](size_t v) {
                geom_fp ^= v + 0x9e3779b97f4a7c15ull + (geom_fp << 6) + (geom_fp >> 2);
            };
            mix(static_cast<size_t>(geometry->inverse));
            const float* floats = &geometry->min.x;
            for (int i = 0; i < 6; ++i) {
                mix(static_cast<size_t>(std::bit_cast<uint32_t>(floats[i])));
            }
            const float* m = &geometry->model_to_cropbox[0][0];
            for (int i = 0; i < 16; ++i) {
                mix(static_cast<size_t>(std::bit_cast<uint32_t>(m[i])));
            }
        } else {
            optimizer.set_crop_damping_mask({});
            cropbox_damping_cache_valid_ = false;
            return;
        }

        if (cropbox_damping_cache_valid_ &&
            cropbox_damping_cached_n_ == n &&
            cropbox_damping_geom_fp_ == geom_fp &&
            cropbox_damping_cached_scale_ == scale &&
            cropbox_damping_cached_mask_.is_valid() &&
            cropbox_damping_cached_mask_.numel() == n) {
            optimizer.set_crop_damping_mask(cropbox_damping_cached_mask_);
            return;
        }

        auto crop_mask = compute_training_cropbox_remove_mask(*scene_, model);
        if (!crop_mask || !crop_mask->is_valid() || crop_mask->numel() == 0) {
            optimizer.set_crop_damping_mask({});
            cropbox_damping_cache_valid_ = false;
            return;
        }

        cropbox_damping_cached_mask_ = *crop_mask;
        cropbox_damping_cached_n_ = n;
        cropbox_damping_geom_fp_ = geom_fp;
        cropbox_damping_cached_scale_ = scale;
        cropbox_damping_cache_valid_ = true;
        ++cropbox_damping_rebuild_count_;
        optimizer.set_crop_damping_mask(cropbox_damping_cached_mask_);
    }

    Trainer::Trainer(lfs::core::Scene& scene)
        : scene_(&scene) {
        LFS_ASSERT_MSG(scene.hasTrainingData(), "Scene has no cameras");
        project_uuid_ = lfs::core::generate_uuid_v4();
        project_snapshot_service_ =
            std::make_unique<TrainingSnapshotService>();

        int device_count = 0;
        LFS_CUDA_TRY(cudaGetDeviceCount(&device_count), nullptr, "CUDA device discovery");
        LFS_ASSERT_MSG(device_count > 0, "CUDA is not available - aborting");
        createCudaResources();

        // One-lock: bind Scene live-model readers to this trainer's step-boundary
        // mutex for the lifetime of the trainer. TrainerManager::setTrainer also
        // wires this; constructor covers every Trainer(Scene) path.
        scene.setLiveModelMutex(&render_mutex_);
        LOG_INFO("Scene live-model mutex wired to trainer render_mutex_ (one-lock q16)");

        LOG_DEBUG("Trainer constructed from Scene with {} cameras", scene.getAllCameras().size());
    }

    void Trainer::createCudaResources() {
        try {
            LFS_CUDA_TRY(
                cudaStreamCreateWithFlags(&callback_stream_, cudaStreamNonBlocking),
                nullptr, "Trainer callback stream creation");

            // Use the default stream flags so synchronous readbacks and cold-path
            // uploads remain ordered with training work. Overlap partners use
            // non-blocking streams with explicit event edges.
            LFS_CUDA_TRY(
                cudaStreamCreate(&training_stream_), nullptr,
                "Trainer training stream creation");
            LFS_CUDA_TRY(
                cudaStreamCreateWithFlags(&metrics_stream_, cudaStreamNonBlocking),
                nullptr, "Trainer metrics stream creation");

            nvtxNameCudaStreamA(training_stream_, "lfs.train");
            nvtxNameCudaStreamA(callback_stream_, "lfs.train.callback");
            nvtxNameCudaStreamA(metrics_stream_, "lfs.metrics");
            createSyncPrimitives();
            PerfBenchCollector::instance().set_timing_stream(training_stream_);
        } catch (...) {
            // A C++ destructor is not invoked when its constructor throws.
            // Roll back every member handle published by this transaction.
            destroySyncPrimitives();
            if (metrics_stream_) {
                LFS_CUDA_LOG_TEARDOWN(cudaStreamDestroy(metrics_stream_), metrics_stream_,
                                      "rollback: destroy metrics stream");
                metrics_stream_ = nullptr;
            }
            if (training_stream_) {
                LFS_CUDA_LOG_TEARDOWN(cudaStreamDestroy(training_stream_), training_stream_,
                                      "rollback: destroy training stream");
                training_stream_ = nullptr;
            }
            if (callback_stream_) {
                LFS_CUDA_LOG_TEARDOWN(cudaStreamDestroy(callback_stream_), callback_stream_,
                                      "rollback: destroy callback stream");
                callback_stream_ = nullptr;
            }
            throw;
        }
    }

    void Trainer::createSyncPrimitives() {
        LFS_CUDA_TRY(
            cudaEventCreateWithFlags(&params_ready_event_, cudaEventDisableTiming),
            nullptr, "Trainer parameter-ready event creation");
        for (size_t i = 0; i < reader_done_events_.size(); ++i) {
            LFS_CUDA_TRY(
                cudaEventCreateWithFlags(&reader_done_events_[i], cudaEventDisableTiming),
                nullptr,
                ::lfs::core::detail::format_cuda_safe("Trainer reader event {} creation", i));
        }
        for (size_t i = 0; i < loss_slots_.size(); ++i) {
            LFS_CUDA_TRY(
                cudaEventCreateWithFlags(&loss_slots_[i].done, cudaEventDisableTiming),
                nullptr,
                ::lfs::core::detail::format_cuda_safe("Trainer loss event {} creation", i));
        }
        for (size_t i = 0; i < loss_slots_.size(); ++i) {
            loss_slots_[i].pinned = static_cast<float*>(
                lfs::core::PinnedMemoryAllocator::instance().allocate(sizeof(float)));
            LFS_ASSERT_MSG(
                loss_slots_[i].pinned != nullptr,
                std::format("Trainer loss slot {} pinned allocation failed", i));
        }
    }

    void Trainer::destroySyncPrimitives() {
        std::lock_guard<std::mutex> lock(stream_sync_mutex_);
        params_ready_recorded_ = false;
        reader_done_pending_ = 0;
        viewer_release_semaphore_ = nullptr;
        if (params_ready_event_) {
            LFS_CUDA_LOG_TEARDOWN(cudaEventDestroy(params_ready_event_), nullptr,
                                  "destroy params-ready event");
            params_ready_event_ = nullptr;
        }
        for (auto& event : reader_done_events_) {
            if (event) {
                LFS_CUDA_LOG_TEARDOWN(cudaEventDestroy(event), nullptr,
                                      "destroy reader-done event");
                event = nullptr;
            }
        }
        for (auto& slot : loss_slots_) {
            if (slot.done) {
                LFS_CUDA_LOG_TEARDOWN(cudaEventDestroy(slot.done), nullptr,
                                      "destroy loss-slot event");
                slot.done = nullptr;
            }
            if (slot.pinned) {
                lfs::core::PinnedMemoryAllocator::instance().deallocate(slot.pinned, nullptr);
                slot.pinned = nullptr;
            }
            slot.in_flight = false;
        }
        for (const cudaEvent_t event : orphaned_sidecar_events_) {
            LFS_CUDA_LOG_TEARDOWN(cudaEventDestroy(event), nullptr,
                                  "destroy orphaned sidecar event");
        }
        orphaned_sidecar_events_.clear();
    }

    void Trainer::submitLossReadback(const lfs::core::Tensor& total_loss, int iter) {
        LossReadbackSlot& slot = loss_slots_[loss_slot_head_];
        if (!slot.pinned || !slot.done) {
            return;
        }
        if (slot.in_flight) {
            // Ring full: the GPU is LOSS_RING submit intervals behind —
            // explicit backpressure instead of silently dropping the sample.
            // The caller harvests right before submitting, so this slot's
            // value was already consumed once the event completes.
            LFS_CUDA_LOG_TEARDOWN(cudaEventSynchronize(slot.done), nullptr,
                                  "loss readback: drain ring slot");
            slot.in_flight = false;
        }
        if (cudaMemcpyAsync(slot.pinned, total_loss.ptr<float>(), sizeof(float),
                            cudaMemcpyDeviceToHost, training_stream_) != cudaSuccess) {
            return;
        }
        if (cudaEventRecord(slot.done, training_stream_) == cudaSuccess) {
            slot.iter = iter;
            slot.in_flight = true;
            loss_slot_head_ = (loss_slot_head_ + 1) % LOSS_RING;
        }
    }

    std::expected<void, std::string> Trainer::harvestLossReadbacks(bool drain, bool in_controller_phase) {
        for (size_t i = 0; i < LOSS_RING; ++i) {
            LossReadbackSlot& slot = loss_slots_[(loss_slot_head_ + i) % LOSS_RING];
            if (!slot.in_flight) {
                continue;
            }
            if (drain) {
                if (cudaEventSynchronize(slot.done) != cudaSuccess) {
                    slot.in_flight = false;
                    continue;
                }
            } else if (cudaEventQuery(slot.done) != cudaSuccess) {
                break;
            }
            slot.in_flight = false;

            const float loss_value = *slot.pinned;
            if (std::isnan(loss_value) || std::isinf(loss_value)) {
                return std::unexpected(std::format("NaN/Inf loss at iteration {}", slot.iter));
            }

            current_loss_ = loss_value;
            if (progress_) {
                progress_->update(
                    slot.iter,
                    loss_value,
                    static_cast<int>(strategy_->get_model().size()),
                    get_progress_phase(slot.iter, in_controller_phase));
            }
            lfs::core::events::state::TrainingProgress{
                .iteration = slot.iter,
                .loss = loss_value,
                .num_gaussians = static_cast<int>(strategy_->get_model().size()),
                .is_refining = strategy_->is_refining(slot.iter)}
                .emit();
        }
        return {};
    }

    void Trainer::beginModelRead(cudaStream_t reader_stream) {
        std::lock_guard<std::mutex> lock(stream_sync_mutex_);
        if (params_ready_event_ && params_ready_recorded_) {
            LFS_CUDA_TRY(cudaStreamWaitEvent(reader_stream, params_ready_event_, 0),
                         reader_stream,
                         "begin model read: wait params-ready event");
        }
    }

    void Trainer::endModelRead(cudaStream_t reader_stream) {
        std::lock_guard<std::mutex> lock(stream_sync_mutex_);
        cudaEvent_t& slot = reader_done_events_[reader_done_head_];
        if (!slot) {
            return;
        }
        const uint32_t bit = 1u << reader_done_head_;
        if (reader_done_pending_ & bit) {
            // Ring full: the slot's previous record hasn't been consumed by a
            // step yet. Drain it host-side before reuse — re-recording would
            // drop the older reader's edge.
            LFS_CUDA_TRY(cudaEventSynchronize(slot), nullptr,
                         "end model read: drain reader-done ring");
        }
        LFS_CUDA_TRY(cudaEventRecord(slot, reader_stream), reader_stream,
                     "end model read: record reader-done event");
        reader_done_pending_ |= bit;
        reader_done_head_ = (reader_done_head_ + 1) % READER_DONE_RING;
    }

    void Trainer::setViewerReleaseFence(cudaExternalSemaphore_t semaphore) {
        std::lock_guard<std::mutex> lock(stream_sync_mutex_);
        if (viewer_release_semaphore_ == semaphore) {
            return;
        }
        viewer_release_semaphore_ = semaphore;
        viewer_borrow_waited_ = 0;
        // A new fence is a fresh timeline starting at 0 — a borrow value from
        // the previous timeline would make the trainer wait a value the new
        // semaphore never reaches.
        viewer_borrow_value_.store(0, std::memory_order_release);
    }

    Trainer::ExportableDensifyBarrierBegin Trainer::beginExportableDensifyBarrier() {
        if (exportable_densify_barrier_depth_ > 0) {
            ++exportable_densify_barrier_depth_;
            return ExportableDensifyBarrierBegin::Acquired;
        }
        if (!exportable_densify_barrier_begin_) {
            return ExportableDensifyBarrierBegin::NotInstalled;
        }
        if (!exportable_densify_barrier_begin_()) {
            return ExportableDensifyBarrierBegin::Failed;
        }
        exportable_densify_barrier_depth_ = 1;
        return ExportableDensifyBarrierBegin::Acquired;
    }

    bool Trainer::endExportableDensifyBarrier() {
        if (exportable_densify_barrier_depth_ <= 0) {
            return true;
        }
        --exportable_densify_barrier_depth_;
        if (exportable_densify_barrier_depth_ > 0) {
            return true;
        }
        if (exportable_densify_barrier_end_ && !exportable_densify_barrier_end_()) {
            LOG_ERROR("Failed to end exportable densify barrier; stopping training");
            request_stop();
            return false;
        }
        return true;
    }

    void Trainer::publishViewerBorrow(uint64_t value) {
        // Monotonic: prompt per-submit publishes and the frame-scope publisher
        // may interleave; never regress to an older value.
        uint64_t current = viewer_borrow_value_.load(std::memory_order_relaxed);
        while (current < value &&
               !viewer_borrow_value_.compare_exchange_weak(
                   current, value, std::memory_order_release, std::memory_order_relaxed)) {
        }
    }

    void Trainer::recordParamsReady() {
        std::lock_guard<std::mutex> lock(stream_sync_mutex_);
        if (!params_ready_event_) {
            return;
        }
        // training_stream_ is a blocking stream, so the record is also ordered
        // after the legacy-stream rasterizer writes enqueued this step.
        if (cudaEventRecord(params_ready_event_, training_stream_) == cudaSuccess) {
            params_ready_recorded_ = true;
        }
    }

    void Trainer::waitForModelReaders() {
        std::lock_guard<std::mutex> lock(stream_sync_mutex_);
        if (reader_done_pending_ != 0) {
            for (size_t i = 0; i < READER_DONE_RING; ++i) {
                const uint32_t bit = 1u << i;
                if (!(reader_done_pending_ & bit)) {
                    continue;
                }
                LFS_CUDA_TRY(
                    cudaStreamWaitEvent(training_stream_, reader_done_events_[i], 0), training_stream_,
                    ::lfs::core::detail::format_cuda_safe("model reader-done wait, slot {}", i));
                reader_done_pending_ &= ~bit;
            }
        }

        const uint64_t borrow = viewer_borrow_value_.load(std::memory_order_acquire);
        if (viewer_release_semaphore_ && borrow > viewer_borrow_waited_) {
            cudaExternalSemaphoreWaitParams wait_params{};
            wait_params.params.fence.value = borrow;
            LFS_CUDA_TRY(
                cudaWaitExternalSemaphoresAsync(&viewer_release_semaphore_, &wait_params, 1, training_stream_),
                training_stream_,
                ::lfs::core::detail::format_cuda_safe(
                    "viewer-release semaphore wait, borrow value {}", borrow));
            viewer_borrow_waited_ = borrow;
        }
    }

    // Fits each camera's depth-prior alignment against the initial point
    // cloud once, so the supervision target is absolute and multi-view
    // consistent instead of chasing the current render.
    void Trainer::fitDepthAnchors(const size_t cameras_with_depth) {
        nvtxRangePush("fit_depth_anchors");
        const auto fit_start = std::chrono::steady_clock::now();
        const auto phase_ms = [](std::chrono::steady_clock::time_point start) {
            return std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - start)
                .count();
        };

        depth_anchors_.clear();
        depth_anchor_fit_attempted_ = false;
        const auto configured_depth_prior = depth_prior_from_mode(params_.optimization.depth_loss_mode);
        resolved_depth_prior_ = configured_depth_prior;
        auto& model = strategy_->get_model();
        const auto& means = model.means();
        const auto num_points = static_cast<size_t>(model.size());
        if (num_points == 0) {
            LOG_WARN("Depth anchors: no initial point cloud available; depth supervision is disabled without anchors");
            nvtxRangePop();
            return;
        }
        depth_anchor_fit_attempted_ = true;

        // Prefer a precomputed sidecar (written by `preprocess` or a prior run);
        // fall back to fitting here and write it through so the next run is fast.
        const auto& cameras = train_dataset_->get_cameras();
        const auto sidecar = depthAnchorSidecarPath(cameras);

        nvtxRangePush("depth_anchors/fingerprint");
        const auto fingerprint_start = std::chrono::steady_clock::now();
        const auto fingerprint = computeAnchorFingerprint(cameras);
        const auto fingerprint_ms = phase_ms(fingerprint_start);
        nvtxRangePop();

        RawDepthAnchorMap raw;
        nvtxRangePush("depth_anchors/read_sidecar");
        const auto read_start = std::chrono::steady_clock::now();
        auto cached = readDepthAnchorSidecar(sidecar, fingerprint);
        const auto read_ms = phase_ms(read_start);
        nvtxRangePop();

        double source_ms = read_ms;
        if (cached) {
            raw = std::move(*cached);
            LOG_INFO("Depth anchors: loaded {} cached anchors from {}", raw.size(), sidecar.string());
        } else {
            nvtxRangePush("depth_anchors/compute");
            const auto compute_start = std::chrono::steady_clock::now();
            raw = computeRawDepthAnchors(
                means, cameras, params_.dataset.resize_factor, params_.dataset.max_width);
            source_ms = phase_ms(compute_start);
            nvtxRangePop();
            if (!sidecar.empty() && !writeDepthAnchorSidecar(sidecar, raw, fingerprint)) {
                LOG_WARN("Depth anchors: failed to write sidecar {}", sidecar.string());
            }
        }
        LOG_INFO("Depth anchors: fingerprint {:.1f} ms, {} {:.1f} ms",
                 fingerprint_ms, cached ? "sidecar load" : "recompute", source_ms);

        nvtxRangePush("depth_anchors/resolve");
        size_t processed = 0;
        for (const auto& cam : cameras) {
            if (!cam || !cam->has_depth()) {
                continue;
            }
            ++processed;
            if (const auto it = raw.find(cam->image_name()); it != raw.end()) {
                depth_anchors_[cam->uid()] = it->second;
            }
        }

        DepthAnchorCandidateStats disparity_stats;
        DepthAnchorCandidateStats depth_stats;
        for (const auto& entry : depth_anchors_) {
            disparity_stats.add(entry.second.disparity);
            depth_stats.add(entry.second.depth);
        }
        if (configured_depth_prior == lfs::training::kernels::DepthPriorType::Auto) {
            if (disparity_stats.count > 0 || depth_stats.count > 0) {
                resolved_depth_prior_ =
                    disparity_stats.abs_corr_sum >= depth_stats.abs_corr_sum
                        ? lfs::training::kernels::DepthPriorType::Disparity
                        : lfs::training::kernels::DepthPriorType::Depth;
                LOG_INFO("Depth anchors: auto resolved dataset prior as {} "
                         "(disparity: {} candidates, mean |corr| {:.3f}; depth: {} candidates, mean |corr| {:.3f})",
                         depth_prior_name(resolved_depth_prior_),
                         disparity_stats.count, disparity_stats.mean_abs_corr(),
                         depth_stats.count, depth_stats.mean_abs_corr());
            } else {
                resolved_depth_prior_ = lfs::training::kernels::DepthPriorType::Auto;
            }
        }

        size_t dropped_unusable = 0;
        size_t dropped_sign = 0;
        if (resolved_depth_prior_ != lfs::training::kernels::DepthPriorType::Auto) {
            size_t positive = 0;
            size_t negative = 0;
            for (const auto& entry : depth_anchors_) {
                const auto* candidate = depth_anchor_candidate_for_prior(entry.second, resolved_depth_prior_);
                if (candidate == nullptr || !candidate->valid) {
                    continue;
                }
                (candidate->scale > 0.0f ? positive : negative) += 1;
            }
            const bool filter_sign = positive != negative && (positive + negative) > 0;
            const bool keep_positive = positive > negative;

            for (auto it = depth_anchors_.begin(); it != depth_anchors_.end();) {
                auto& anchor = it->second;
                const auto* candidate = depth_anchor_candidate_for_prior(anchor, resolved_depth_prior_);
                if (candidate == nullptr || !candidate->valid) {
                    it = depth_anchors_.erase(it);
                    ++dropped_unusable;
                    continue;
                }
                if (filter_sign && ((candidate->scale > 0.0f) != keep_positive)) {
                    it = depth_anchors_.erase(it);
                    ++dropped_sign;
                    continue;
                }
                select_depth_anchor_candidate(anchor, resolved_depth_prior_, *candidate);
                ++it;
            }
            if (dropped_unusable > 0 || dropped_sign > 0) {
                LOG_INFO("Depth anchors: dropped {} anchors without a {} fit and {} sign-inconsistent fits",
                         dropped_unusable, depth_prior_name(resolved_depth_prior_), dropped_sign);
            }
        } else {
            depth_anchors_.clear();
        }

        size_t reliable_anchors = 0;
        double reliable_corr_sum = 0.0;
        for (const auto& entry : depth_anchors_) {
            const auto& anchor = entry.second;
            ++reliable_anchors;
            reliable_corr_sum += std::fabs(anchor.corr);
        }
        nvtxRangePop(); // depth_anchors/resolve

        const auto fit_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now() - fit_start)
                                .count();
        if (reliable_anchors > 0) {
            LOG_INFO("Depth anchors: {}/{} cameras aligned to the initial point cloud "
                     "(mean |corr| {:.3f}, {} ms)",
                     reliable_anchors, processed,
                     reliable_corr_sum / static_cast<double>(reliable_anchors), fit_ms);
        } else {
            LOG_WARN("Depth anchors: no camera could be aligned to the initial point cloud "
                     "({} cameras with depth); depth loss will be skipped for this anchored dataset",
                     cameras_with_depth);
        }
        nvtxRangePop();
    }

    bool Trainer::fillCameraLossColors(
        const std::vector<std::shared_ptr<const lfs::core::Camera>>& cameras,
        std::vector<std::array<float, 3>>& colors) const {
        constexpr std::array<float, 3> UNSEEN{1.0f, 1.0f, 1.0f};
        colors.assign(cameras.size(), UNSEEN);

        const auto heatmap = getCameraLossHeatmap();
        if (!heatmap) {
            return true;
        }

        std::shared_lock lock(heatmap->snapshot_mutex);
        if (heatmap->published_colors.empty() || heatmap->published_valid.empty()) {
            return true;
        }

        for (size_t i = 0; i < cameras.size(); ++i) {
            const auto& camera = cameras[i];
            if (!camera) {
                continue;
            }

            const auto it = heatmap->uid_to_slot.find(camera->uid());
            if (it == heatmap->uid_to_slot.end()) {
                continue;
            }

            const size_t slot = it->second;
            if (slot >= heatmap->published_colors.size() ||
                slot >= heatmap->published_valid.size() ||
                !heatmap->published_valid[slot]) {
                continue;
            }

            colors[i] = heatmap->published_colors[slot];
        }

        return true;
    }

    std::expected<void, std::string> Trainer::initialize_camera_loss_heatmap(
        const std::vector<std::shared_ptr<lfs::core::Camera>>& cameras) {
        setCameraLossHeatmap(nullptr);

        auto heatmap = std::make_shared<CameraLossHeatmapState>();
        heatmap->camera_uids.reserve(cameras.size());
        heatmap->uid_to_slot.reserve(cameras.size());

        for (const auto& camera : cameras) {
            if (!camera) {
                continue;
            }

            const int uid = camera->uid();
            if (heatmap->uid_to_slot.contains(uid)) {
                continue;
            }

            const size_t slot = heatmap->camera_uids.size();
            heatmap->camera_uids.push_back(uid);
            heatmap->uid_to_slot.emplace(uid, slot);
        }

        if (heatmap->camera_uids.empty()) {
            return std::unexpected("No cameras available for loss heatmap");
        }

        const lfs::core::TensorShape shape{heatmap->camera_uids.size()};
        heatmap->latest_loss_gpu = lfs::core::Tensor::full(shape, -1.0f, lfs::core::Device::CUDA);
        heatmap->ema_loss_gpu = lfs::core::Tensor::full(shape, -1.0f, lfs::core::Device::CUDA);
        heatmap->ema_loss_stage_cpu = lfs::core::Tensor::full(shape, -1.0f, lfs::core::Device::CPU);
        heatmap->published_colors.resize(heatmap->camera_uids.size());
        heatmap->published_valid.assign(heatmap->camera_uids.size(), 0u);

        if (const cudaError_t err = cudaStreamCreateWithFlags(&heatmap->copy_stream, cudaStreamNonBlocking);
            err != cudaSuccess) {
            return std::unexpected(std::format("Failed to create camera-loss copy stream: {}",
                                               cudaGetErrorString(err)));
        }

        if (const cudaError_t err = cudaEventCreateWithFlags(&heatmap->ready_event, cudaEventDisableTiming);
            err != cudaSuccess) {
            return std::unexpected(std::format("Failed to create camera-loss ready event: {}",
                                               cudaGetErrorString(err)));
        }

        if (const cudaError_t err = cudaEventCreateWithFlags(&heatmap->done_event, cudaEventDisableTiming);
            err != cudaSuccess) {
            return std::unexpected(std::format("Failed to create camera-loss done event: {}",
                                               cudaGetErrorString(err)));
        }

        setCameraLossHeatmap(std::move(heatmap));
        return {};
    }

    void Trainer::update_camera_loss_heatmap(const lfs::core::Camera& camera,
                                             const lfs::core::Tensor& image_loss) {
        const auto heatmap = getCameraLossHeatmap();
        if (!heatmap || !image_loss.is_valid() || image_loss.numel() != 1 ||
            image_loss.device() != lfs::core::Device::CUDA) {
            return;
        }

        const auto it = heatmap->uid_to_slot.find(camera.uid());
        if (it == heatmap->uid_to_slot.end()) {
            return;
        }

        const cudaStream_t stream = image_loss.stream();
        kernels::launch_update_camera_loss_heatmap(
            image_loss.ptr<float>(),
            static_cast<int>(it->second),
            CAMERA_LOSS_EMA_ALPHA,
            heatmap->latest_loss_gpu.ptr<float>(),
            heatmap->ema_loss_gpu.ptr<float>(),
            heatmap->camera_uids.size(),
            stream);

        heatmap->producer_stream = stream;
        heatmap->dirty = true;
    }

    void Trainer::publish_camera_loss_heatmap_snapshot() {
        const auto heatmap = getCameraLossHeatmap();
        if (!heatmap || !heatmap->ema_loss_stage_cpu.is_valid()) {
            return;
        }

        const size_t count = heatmap->camera_uids.size();
        const float* ema_ptr = heatmap->ema_loss_stage_cpu.ptr<float>();

        float min_loss = std::numeric_limits<float>::infinity();
        float max_loss = -std::numeric_limits<float>::infinity();
        size_t seen_count = 0;

        for (size_t i = 0; i < count; ++i) {
            const float loss = ema_ptr[i];
            if (!std::isfinite(loss) || loss < 0.0f) {
                continue;
            }
            min_loss = std::min(min_loss, loss);
            max_loss = std::max(max_loss, loss);
            ++seen_count;
        }

        std::vector<std::array<float, 3>> colors(count);
        std::vector<uint8_t> valid(count, 0u);

        if (seen_count > 0) {
            const bool degenerate_span = (max_loss - min_loss) < 1e-6f;
            const float inv_span = degenerate_span ? 0.0f : 1.0f / (max_loss - min_loss);

            for (size_t i = 0; i < count; ++i) {
                const float loss = ema_ptr[i];
                if (!std::isfinite(loss) || loss < 0.0f) {
                    continue;
                }

                const float t = degenerate_span ? 0.0f : std::clamp((loss - min_loss) * inv_span, 0.0f, 1.0f);
                colors[i] = camera_loss_color(t);
                valid[i] = 1u;
            }
        }

        std::unique_lock lock(heatmap->snapshot_mutex);
        heatmap->published_colors = std::move(colors);
        heatmap->published_valid = std::move(valid);
    }

    void Trainer::maybe_publish_camera_loss_heatmap(const int iter, const bool force) {
        const auto heatmap = getCameraLossHeatmap();
        if (!heatmap) {
            return;
        }

        if (heatmap->copy_in_flight) {
            cudaError_t status = force
                                     ? cudaEventSynchronize(heatmap->done_event)
                                     : cudaEventQuery(heatmap->done_event);
            if (status == cudaSuccess) {
                publish_camera_loss_heatmap_snapshot();
                heatmap->copy_in_flight = false;
            } else if (status != cudaErrorNotReady) {
                LOG_WARN("Camera-loss snapshot query failed: {}", cudaGetErrorString(status));
                heatmap->copy_in_flight = false;
                heatmap->dirty = true;
            }
        }

        // `nullptr` is a valid CUDA default stream, so only gate on actual work state here.
        if (!heatmap->dirty || heatmap->copy_in_flight) {
            return;
        }

        if (!force && (iter % CAMERA_LOSS_PUBLISH_INTERVAL) != 0) {
            return;
        }

        const size_t bytes = heatmap->camera_uids.size() * sizeof(float);

        if (const cudaError_t err = cudaEventRecord(heatmap->ready_event, heatmap->producer_stream);
            err != cudaSuccess) {
            LOG_WARN("Camera-loss ready event failed: {}", cudaGetErrorString(err));
            return;
        }
        if (const cudaError_t err = cudaStreamWaitEvent(heatmap->copy_stream, heatmap->ready_event, 0);
            err != cudaSuccess) {
            LOG_WARN("Camera-loss stream wait failed: {}", cudaGetErrorString(err));
            return;
        }
        if (const cudaError_t err = cudaMemcpyAsync(
                heatmap->ema_loss_stage_cpu.ptr<float>(),
                heatmap->ema_loss_gpu.ptr<float>(),
                bytes,
                cudaMemcpyDeviceToHost,
                heatmap->copy_stream);
            err != cudaSuccess) {
            LOG_WARN("Camera-loss async copy failed: {}", cudaGetErrorString(err));
            return;
        }
        if (const cudaError_t err = cudaEventRecord(heatmap->done_event, heatmap->copy_stream);
            err != cudaSuccess) {
            LOG_WARN("Camera-loss done event failed: {}", cudaGetErrorString(err));
            return;
        }

        heatmap->copy_in_flight = true;
        heatmap->dirty = false;

        if (force) {
            const cudaError_t status = cudaEventSynchronize(heatmap->done_event);
            if (status == cudaSuccess) {
                publish_camera_loss_heatmap_snapshot();
                heatmap->copy_in_flight = false;
            } else {
                LOG_WARN("Camera-loss snapshot sync failed: {}", cudaGetErrorString(status));
                heatmap->dirty = true;
            }
        }
    }

    std::expected<void, std::string> Trainer::initialize(const lfs::core::param::TrainingParameters& params) {
        if (const auto validation_error = params.validate(); !validation_error.empty()) {
            return std::unexpected("Invalid training parameters: " + validation_error);
        }

        // Thread-safe initialization using mutex
        std::lock_guard<std::mutex> lock(init_mutex_);

        // Check again after acquiring lock (double-checked locking pattern)
        if (initialized_.load()) {
            LOG_INFO("Re-initializing trainer with new parameters");
            // Clean up existing state for re-initialization
            cleanup();
        }

        LOG_INFO("Initializing trainer with {} iterations", params.optimization.iterations);

        try {
            params_ = params;

            // Wire CLI instruments (replaces former env flags).
            PerfBenchCollector::configure(
                params_.optimization.perf_bench,
                params_.optimization.perf_bench_warmup);

            if (params_.optimization.enable_sparsity) {
                const size_t stop_refine_limit = static_cast<size_t>(std::max(0, get_regular_iterations()));
                if (params_.optimization.stop_refine > stop_refine_limit) {
                    LOG_WARN("Sparsity: clamping stop_refine from {} to {} to freeze topology before pruning",
                             params_.optimization.stop_refine, stop_refine_limit);
                    params_.optimization.stop_refine = stop_refine_limit;
                }
            }

            // Create DatasetConfig for lfs::training::CameraDataset
            lfs::training::DatasetConfig dataset_config;
            dataset_config.resize_factor = params.dataset.resize_factor;
            dataset_config.max_width = params.dataset.max_width;
            dataset_config.test_every = params.dataset.test_every;

            // Get source cameras from Scene nodes or base_dataset_
            std::vector<std::shared_ptr<lfs::core::Camera>> source_cameras;
            std::vector<std::shared_ptr<lfs::core::Camera>> train_cameras;
            std::vector<std::shared_ptr<lfs::core::Camera>> val_cameras;
            if (scene_) {
                source_cameras = scene_->getActiveCameras();
                if (source_cameras.empty()) {
                    return std::unexpected("Scene has no active cameras enabled for training");
                }
                std::erase_if(source_cameras, [](const auto& camera) {
                    return !camera || !camera->has_image();
                });
                if (source_cameras.empty()) {
                    return std::unexpected(
                        "Scene has no cameras with image files available for training");
                }

                if (params.optimization.enable_eval) {
                    for (const auto& camera : source_cameras) {
                        switch (camera->split()) {
                        case lfs::core::CameraSplit::Train:
                            train_cameras.push_back(camera);
                            break;
                        case lfs::core::CameraSplit::Eval:
                            val_cameras.push_back(camera);
                            break;
                        default:
                            assert(false && "Camera split must be Train or Eval");
                            break;
                        }
                    }
                    assert(train_cameras.size() + val_cameras.size() == source_cameras.size());
                }
            } else if (base_dataset_) {
                source_cameras = base_dataset_->get_cameras();
                std::erase_if(source_cameras, [](const auto& camera) {
                    return !camera || !camera->has_image();
                });
                if (source_cameras.empty()) {
                    return std::unexpected(
                        "Dataset has no cameras with image files available for training");
                }
            } else {
                return std::unexpected("No camera source available");
            }

            total_cameras_count_ = source_cameras.size();

            if (auto result = initialize_camera_loss_heatmap(source_cameras); !result) {
                return std::unexpected(result.error());
            }

            // Handle dataset split based on evaluation flag
            if (params.optimization.enable_eval) {
                if (scene_) {
                    train_dataset_ = std::make_shared<CameraDataset>(
                        train_cameras, dataset_config, CameraDataset::Split::ALL);
                    val_dataset_ = std::make_shared<CameraDataset>(
                        val_cameras, dataset_config, CameraDataset::Split::ALL);
                } else {
                    // Create train/val split
                    train_dataset_ = std::make_shared<CameraDataset>(
                        source_cameras, dataset_config, CameraDataset::Split::TRAIN,
                        provided_splits_ ? std::make_optional(std::get<0>(*provided_splits_)) : std::nullopt);
                    val_dataset_ = std::make_shared<CameraDataset>(
                        source_cameras, dataset_config, CameraDataset::Split::VAL,
                        provided_splits_ ? std::make_optional(std::get<1>(*provided_splits_)) : std::nullopt);

                    LOG_INFO("Created train/val split: {} train, {} val images",
                             train_dataset_->size(),
                             val_dataset_->size());
                }
                if (train_dataset_->size() == 0) {
                    return std::unexpected("Evaluation split leaves no training images. Increase Test Every or disable evaluation.");
                }
                if (val_dataset_->size() == 0) {
                    return std::unexpected("Evaluation is enabled but the validation split is empty. Decrease Test Every or disable evaluation.");
                }
            } else {
                // Use all images for training
                train_dataset_ = std::make_shared<CameraDataset>(
                    source_cameras, dataset_config, CameraDataset::Split::ALL);
                val_dataset_ = nullptr;

                LOG_INFO("Using all {} images for training (no evaluation)",
                         train_dataset_->size());
            }

            train_dataset_size_ = train_dataset_->size();

            // If using Scene mode and no strategy yet, create one
            if (scene_ && !strategy_) {
                auto* model = scene_->getTrainingModel();
                if (!model) {
                    return std::unexpected("Scene has no training model set");
                }

                auto result = StrategyFactory::instance().create(params.optimization.strategy, *model);
                if (!result) {
                    return std::unexpected(result.error());
                }
                strategy_ = std::move(*result);
                LOG_DEBUG("Created {} strategy from Scene model", params.optimization.strategy);
            }

            auto& splat = strategy_->get_model();

            const int max_cap = params.optimization.max_cap;
            if (max_cap > 0 && max_cap < splat.size()) {
                LOG_WARN("Max cap ({}) is less than initial splat count ({}), randomly selecting {} splats",
                         max_cap, splat.size(), max_cap);
                lfs::core::random_choose(splat, max_cap);
                syncTrainingSceneTopology(scene_, splat);
            }

            // Re-initialize strategy with new parameters
            strategy_->set_training_dataset(train_dataset_);
            strategy_->initialize(get_runtime_optimization_params());
            apply_frozen_ranges_to_optimizer(
                splat,
                strategy_->get_optimizer(),
                params_.freeze_lr_scale);
            {
                std::unique_lock<std::shared_mutex> render_lock(render_mutex_);
                if (auto result = ensureModelTensorAllocatorStorage(splat, "strategy initialization"); !result) {
                    return std::unexpected(result.error());
                }
            }
            LOG_DEBUG("Strategy initialized");

            // Initialize bilateral grid if enabled
            if (auto result = initialize_bilateral_grid(); !result) {
                return std::unexpected(result.error());
            }

            // Initialize PPISP if enabled
            if (auto result = initialize_ppisp(); !result) {
                return std::unexpected(result.error());
            }

            // Initialize PPISP controller if enabled
            if (auto result = initialize_ppisp_controller(); !result) {
                return std::unexpected(result.error());
            }

            // Validate masks if mask mode is enabled
            if (auto result = validate_masks(); !result) {
                return std::unexpected(result.error());
            }

            // Apply undistortion to camera intrinsics (params already precomputed at load time)
            if (params.optimization.undistort) {
                int prepared = 0;
                for (auto& cam : train_dataset_->get_cameras()) {
                    if (cam && cam->has_distortion()) {
                        cam->prepare_undistortion();
                        ++prepared;
                    }
                }
                if (val_dataset_) {
                    for (auto& cam : val_dataset_->get_cameras()) {
                        if (cam && cam->has_distortion()) {
                            cam->prepare_undistortion();
                        }
                    }
                }
                if (prepared > 0) {
                    LOG_INFO("Prepared undistortion for {} cameras", prepared);
                }
            }

            // Initialize sparsity optimizer
            if (params.optimization.enable_sparsity) {
                constexpr int UPDATE_INTERVAL = 50;
                const int regular_iters = get_regular_iterations();
                const int sparsify_steps = get_active_sparsify_steps();
                const int first_sparsify_iter = sparsify_steps > 0 ? regular_iters + 1 : regular_iters;

                const ADMMSparsityOptimizer::Config config{
                    .sparsify_steps = sparsify_steps,
                    .init_rho = params.optimization.init_rho,
                    .prune_ratio = params.optimization.prune_ratio,
                    .update_every = UPDATE_INTERVAL,
                    .start_iteration = get_sparsity_boundary_iteration()};

                sparsity_optimizer_ = SparsityOptimizerFactory::create("admm", config);
                if (sparsity_optimizer_) {
                    LOG_INFO("Sparsity: regular={}, start={}, steps={}, total={}, prune={:.0f}%",
                             regular_iters, first_sparsify_iter, sparsify_steps, get_total_iterations(),
                             params.optimization.prune_ratio * 100);
                }
            }

            // Initialize background color tensor from params
            {
                const auto& bg_color = params.optimization.bg_color;
                background_ = lfs::core::Tensor::empty({3}, lfs::core::Device::CPU, lfs::core::DataType::Float32);
                auto* bg_ptr = background_.ptr<float>();
                bg_ptr[0] = bg_color[0];
                bg_ptr[1] = bg_color[1];
                bg_ptr[2] = bg_color[2];
                background_ = background_.to(lfs::core::Device::CUDA);
                LOG_INFO("Background color set to RGB({:.2f}, {:.2f}, {:.2f})", bg_color[0], bg_color[1], bg_color[2]);
            }

            // Initialize image cache loader before any code path that calls getInstance()
            auto& cache_loader = lfs::io::CacheLoader::getInstance(
                params_.dataset.loading_params.use_cpu_memory);
            cache_loader.update_cache_params(
                params_.dataset.loading_params.use_cpu_memory,
                train_dataset_size_,
                params_.dataset.loading_params.min_cpu_free_GB,
                params_.dataset.loading_params.min_cpu_free_memory_ratio,
                params_.dataset.loading_params.print_cache_status,
                params_.dataset.loading_params.print_status_freq_num);

            // Load background image if specified
            if (params.optimization.bg_mode == lfs::core::param::BackgroundMode::Image &&
                !params.optimization.bg_image_path.empty() &&
                std::filesystem::exists(params.optimization.bg_image_path)) {
                try {
                    auto& loader = lfs::io::CacheLoader::getInstance();
                    lfs::io::LoadParams load_params{
                        .resize_factor = 1,
                        .max_width = 0, // No max width limit
                        .cuda_stream = nullptr};
                    bg_image_base_ = loader.load_cached_image(params.optimization.bg_image_path, load_params);
                    if (bg_image_base_.device() != lfs::core::Device::CUDA) {
                        bg_image_base_ = bg_image_base_.to(lfs::core::Device::CUDA);
                    }
                    if (bg_image_base_.shape()[0] != 3) {
                        LOG_WARN("Background image has {} channels, expected 3 (RGB)", bg_image_base_.shape()[0]);
                        bg_image_base_ = {};
                        params_.optimization.bg_mode = lfs::core::param::BackgroundMode::SolidColor;
                    } else {
                        LOG_INFO("Background image: {} [{}x{}]",
                                 lfs::core::path_to_utf8(params.optimization.bg_image_path),
                                 bg_image_base_.shape()[2], bg_image_base_.shape()[1]);
                    }
                } catch (const std::exception& e) {
                    LOG_WARN("Failed to load background image: {}", e.what());
                    params_.optimization.bg_mode = lfs::core::param::BackgroundMode::SolidColor;
                }
            }

            // Create progress bar based on headless flag
            if (params.optimization.headless) {
                progress_ = std::make_unique<TrainingProgress>(
                    get_total_iterations(),
                    /*update_frequency=*/100);
                LOG_DEBUG("Progress bar initialized for {} total iterations", get_total_iterations());
            }

            // Initialize the evaluator - it handles all metrics internally
            evaluator_ = std::make_unique<lfs::training::MetricsEvaluator>(params_);
            if (params_.optimization.use_ppisp && ppisp_ && ppisp_->isFinalized()) {
                evaluator_->set_appearance([this](const lfs::core::Tensor& rgb, const lfs::core::Camera& cam) {
                    return applyPPISPForEval(rgb, cam);
                });
            }
            LOG_DEBUG("Metrics evaluator initialized");

            // Resume from checkpoint if provided
            if (params_.resume_checkpoint.has_value()) {
                auto resume_result = load_checkpoint(*params_.resume_checkpoint);
                if (!resume_result) {
                    return std::unexpected(std::format("Failed to resume from checkpoint: {}", resume_result.error()));
                }
                LOG_INFO("Resumed training from checkpoint at iteration {}", *resume_result);

                // Reload bg_image if checkpoint restored different settings
                if (params_.optimization.bg_mode == lfs::core::param::BackgroundMode::Image &&
                    !params_.optimization.bg_image_path.empty() &&
                    std::filesystem::exists(params_.optimization.bg_image_path) &&
                    !bg_image_base_.is_valid()) {
                    try {
                        auto& loader = lfs::io::CacheLoader::getInstance();
                        lfs::io::LoadParams load_params{.resize_factor = 1, .max_width = 0, .cuda_stream = nullptr};
                        bg_image_base_ = loader.load_cached_image(params_.optimization.bg_image_path, load_params);
                        if (bg_image_base_.device() != lfs::core::Device::CUDA) {
                            bg_image_base_ = bg_image_base_.to(lfs::core::Device::CUDA);
                        }
                        if (bg_image_base_.shape()[0] != 3) {
                            LOG_WARN("Background image has {} channels, expected 3", bg_image_base_.shape()[0]);
                            bg_image_base_ = {};
                            params_.optimization.bg_mode = lfs::core::param::BackgroundMode::SolidColor;
                        } else {
                            LOG_INFO("Background image from checkpoint: {} [{}x{}]",
                                     lfs::core::path_to_utf8(params_.optimization.bg_image_path),
                                     bg_image_base_.shape()[2], bg_image_base_.shape()[1]);
                        }
                    } catch (const std::exception& e) {
                        LOG_WARN("Failed to load background image from checkpoint: {}", e.what());
                        params_.optimization.bg_mode = lfs::core::param::BackgroundMode::SolidColor;
                    }
                }
            }

            // Print configuration
            LOG_INFO("Visualization: {}", params.optimization.headless ? "disabled" : "enabled");
            LOG_INFO("Strategy: {}", params.optimization.strategy);
            if (params.optimization.mask_mode != lfs::core::param::MaskMode::None) {
                static constexpr const char* MASK_MODE_NAMES[] = {"none", "segment", "ignore", "segment_and_ignore", "alpha_consistent"};
                LOG_INFO("Mask mode: {}", MASK_MODE_NAMES[static_cast<int>(params.optimization.mask_mode)]);
            }
            if (current_iteration_ > 0) {
                LOG_INFO("Starting from iteration: {}", current_iteration_.load());
            }

            // Expose initial snapshot for Python control (iteration 0)
            {
                lfs::training::HookContext ctx{
                    .iteration = current_iteration_.load(),
                    .loss = current_loss_.load(),
                    .num_gaussians = strategy_ ? strategy_->get_model().size() : 0,
                    .is_refining = strategy_ ? strategy_->is_refining(current_iteration_.load()) : false,
                    .trainer = this};
                lfs::training::CommandCenter::instance().set_phase(lfs::training::TrainingPhase::SafeControl);
                lfs::training::CommandCenter::instance().update_snapshot(
                    ctx, get_total_iterations(), is_paused_.load(), is_running_.load(), stop_requested_.load(),
                    lfs::training::TrainingPhase::SafeControl);
            }

            // Execute configured Python scripts to register iteration callbacks
            if (!python_scripts_.empty()) {
                auto py_result = lfs::python::run_scripts(python_scripts_);
                if (!py_result) {
                    return std::unexpected(std::format("Failed to run Python scripts: {}",
                                                       lfs::format_for_developer(py_result.error())));
                }
            }

            if (auto snapshot_service =
                    initialize_project_snapshot_service();
                !snapshot_service) {
                return std::unexpected(
                    std::format(
                        "Failed to initialize .licht snapshot service: {}",
                        lfs::format_for_developer(
                            snapshot_service.error())));
            }

            if (params_.save_project_at_iteration) {
                auto chapters =
                    reserve_project_snapshot_chapters();
                if (!chapters) {
                    return std::unexpected(
                        std::format(
                            "Failed to stage .licht CPU chapters: {}",
                            lfs::format_for_developer(
                                chapters.error())));
                }
                prestaged_project_chapters_ =
                    std::move(*chapters);
                prestaged_project_request_id_ = 0;
                LOG_INFO(
                    "Reserved .licht snapshot UUID before optimizer start; "
                    "CPU chapters remain deferred to the safe point");
            }

            initialized_ = true;

            LOG_INFO("Trainer initialization complete");
            return {};
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Failed to initialize trainer: {}", e.what()));
        }
    }

    Trainer::~Trainer() {
        shutdown();
    }

    std::shared_ptr<lfs::io::PipelinedImageLoader> Trainer::getActiveImageLoader() const {
        std::lock_guard<std::mutex> lock(active_image_loader_mutex_);
        return active_image_loader_;
    }

    Trainer::GTLoadConfigSnapshot Trainer::getGTLoadConfigSnapshot() const {
        std::lock_guard<std::mutex> lock(gt_load_config_mutex_);
        return gt_load_config_snapshot_;
    }

    std::shared_ptr<Trainer::CameraLossHeatmapState> Trainer::getCameraLossHeatmap() const {
        std::lock_guard<std::mutex> lock(camera_loss_heatmap_mutex_);
        return camera_loss_heatmap_;
    }

    void Trainer::setCameraLossHeatmap(std::shared_ptr<CameraLossHeatmapState> heatmap) {
        std::lock_guard<std::mutex> lock(camera_loss_heatmap_mutex_);
        camera_loss_heatmap_ = std::move(heatmap);
    }

    std::expected<void, std::string> Trainer::ensureModelTensorAllocatorStorage(
        lfs::core::SplatData& model,
        const std::string_view reason) {
        if (!splat_tensor_allocator_) {
            return {};
        }

        if (auto result = lfs::training::migrateTrainingModelToAllocator(
                params_, model, splat_tensor_allocator_);
            !result) {
            return std::unexpected(std::format(
                "Failed to keep training model in Vulkan-external storage after {}: {}",
                reason,
                result.error()));
        }
        return {};
    }

    std::expected<Trainer::CameraMetricsSnapshot, std::string> Trainer::computeCameraMetrics(
        const lfs::core::Camera& camera,
        const bool include_ssim,
        CameraMetricsAppearanceConfig appearance) {
        if (!initialized_.load() || !strategy_) {
            return std::unexpected("trainer is not initialized");
        }
        const auto params = getParams();

        auto inputs = load_camera_metrics_inputs(
            camera,
            getGTLoadConfigSnapshot(),
            params.optimization,
            getActiveImageLoader());
        if (!inputs) {
            return std::unexpected(inputs.error());
        }

        auto gt_image = std::move(inputs->gt_image);
        auto mask = std::move(inputs->mask);

        if (gt_image.device() != lfs::core::Device::CUDA) {
            gt_image = gt_image.to(lfs::core::Device::CUDA);
        }
        if (mask.is_valid() && mask.device() != lfs::core::Device::CUDA) {
            mask = mask.to(lfs::core::Device::CUDA);
        }

        lfs::core::Tensor rendered;
        {
            const std::shared_lock lock(render_mutex_);
            // Exclude the non-refining optimizer writes for the metric read window
            // so the live model can't be mutated mid-render (see getModelAccessMutex).
            const std::shared_lock model_read_lock(model_access_mutex_);
            // Run the metric render on the dedicated metrics stream (its kernels
            // and tensor ops overlap training; item() readbacks drain it). Cap
            // arena acquisition so a refining iteration holding the arena can't
            // deadlock this reader (which holds render_mutex_ shared) — on
            // timeout the rasterizer throws and the metric is skipped this call.
            const cudaStream_t reader_stream = metrics_stream_ ? metrics_stream_
                                                               : lfs::core::getCurrentCUDAStream();
            std::optional<lfs::core::CUDAStreamGuard> metrics_guard;
            if (metrics_stream_) {
                metrics_guard.emplace(metrics_stream_);
            }
            const lfs::core::RasterizerMemoryArena::ScopedBeginFrameTimeout arena_timeout(100);
            try {
                beginModelRead(reader_stream);
            } catch (const std::exception& e) {
                return std::unexpected(std::format("metric read window unavailable: {}", e.what()));
            }

            auto& model = strategy_->get_model();
            auto& background = background_;

            try {
                RenderOutput output;
                if (params.optimization.gut) {
                    output = gsplat_rasterize(
                        camera, model, background,
                        1.0f, false, GsplatRenderMode::RGB, true);
                } else {
                    output = fast_rasterize(
                        camera, model, background, params.optimization.mip_filter);
                }

                rendered = output.image;
                if (appearance.enabled) {
                    rendered = applyPPISPForViewport(
                        rendered, camera.uid(), appearance.overrides, appearance.use_controller);
                }
                rendered = rendered.clamp(0.0f, 1.0f);
            } catch (const std::exception& e) {
                // Arena busy (refining trainer holds the frame) or render error:
                // skip this metric sample; the panel retries on its next update.
                try {
                    endModelRead(reader_stream);
                } catch (const std::exception& end_error) {
                    LOG_ERROR("computeCameraMetrics: reader-done record failed during degradation: {}",
                              end_error.what());
                }
                return std::unexpected(std::format("metric render unavailable: {}", e.what()));
            }
            try {
                endModelRead(reader_stream);
            } catch (const std::exception& e) {
                // Without the reader-done edge the trainer may not order writes
                // against this reader's pending kernels — discard the sample.
                return std::unexpected(std::format("metric read-window close failed: {}", e.what()));
            }
        }

        CameraMetricsSnapshot snapshot;
        snapshot.used_mask = mask.is_valid();

        try {
            const PSNR psnr_metric(1.0f);
            snapshot.psnr = psnr_metric.compute(rendered, gt_image, mask);

            if (include_ssim) {
                SSIM ssim_metric(true);
                snapshot.ssim = ssim_metric.compute(rendered, gt_image, mask);
            }
        } catch (const std::exception& e) {
            return std::unexpected(e.what());
        }

        if (!std::isfinite(snapshot.psnr) ||
            (snapshot.ssim.has_value() && !std::isfinite(*snapshot.ssim))) {
            return std::unexpected("metric computation produced non-finite values");
        }

        return snapshot;
    }

    void Trainer::updateGTLoadConfigSnapshot() {
        GTLoadConfigSnapshot snapshot;
        if (train_dataset_) {
            snapshot.resize_factor = std::max(1, train_dataset_->get_resize_factor());
            snapshot.max_width = train_dataset_->get_max_width();

            for (const auto& cam : train_dataset_->get_cameras()) {
                if (cam && cam->is_undistort_prepared()) {
                    snapshot.undistort = true;
                    break;
                }
            }
        }

        std::lock_guard<std::mutex> lock(gt_load_config_mutex_);
        gt_load_config_snapshot_ = snapshot;
    }

    void Trainer::setActiveImageLoader(std::shared_ptr<lfs::io::PipelinedImageLoader> loader) {
        std::lock_guard<std::mutex> lock(active_image_loader_mutex_);
        active_image_loader_ = std::move(loader);
    }

    void Trainer::clearActiveImageLoader() {
        if (strategy_) {
            strategy_->set_image_loader(nullptr);
        }
        setActiveImageLoader(nullptr);
    }

    void Trainer::shutdown() {
        if (shutdown_complete_.exchange(true)) {
            return;
        }

        LOG_DEBUG("Trainer shutdown");
        if (scene_ && scene_->liveModelMutex() == &render_mutex_) {
            scene_->setLiveModelMutex(nullptr);
        }
        stop_requested_ = true;

        lfs::core::image_io::wait_for_pending_saves();

        if (callback_stream_) {
            LFS_CUDA_LOG_TEARDOWN(cudaStreamSynchronize(callback_stream_), callback_stream_,
                                  "shutdown: sync callback stream");
        }

        if (training_stream_) {
            LFS_CUDA_LOG_TEARDOWN(cudaStreamSynchronize(training_stream_), training_stream_,
                                  "shutdown: sync training stream");
        }

        if (metrics_stream_) {
            LFS_CUDA_LOG_TEARDOWN(cudaStreamSynchronize(metrics_stream_), metrics_stream_,
                                  "shutdown: sync metrics stream");
        }

        LFS_CUDA_LOG_TEARDOWN(cudaDeviceSynchronize(), nullptr, "shutdown: device sync");

        finish_project_writer();
        project_snapshot_service_.reset();

        const bool exiting_headless = params_.optimization.headless;
        if (callback_stream_) {
            lfs::core::CudaMemoryPool::instance().release_stream(callback_stream_);
        }

        if (metrics_stream_) {
            lfs::core::CudaMemoryPool::instance().release_stream(metrics_stream_);
        }

        if (training_stream_) {
            lfs::core::CudaMemoryPool::instance().release_stream(training_stream_);
        }

        if (strategy_) {
            strategy_->set_image_loader(nullptr);
        }
        background_ = {};
        bg_mix_buffer_ = {};
        bg_image_base_ = {};
        bg_image_cache_.clear();
        random_bg_buffer_ = {};
        pipelined_mask_ = {};
        pipelined_depth_ = {};
        pipelined_normal_ = {};
        photometric_loss_ = {};
        loss_accumulator_ = {};
        depth_loss_scalar_ = {};
        depth_loss_grad_ = {};
        depth_loss_grad_alpha_ = {};
        depth_loss_partials_ = {};
        normal_loss_scalar_ = {};
        normal_loss_grad_ = {};
        normal_loss_partials_ = {};
        normal_consistency_scalar_ = {};
        normal_consistency_partials_ = {};
        normal_prior_depth_scalar_ = {};
        densification_ssim_workspace_ = {};
        densification_error_map_ = {};
        edge_map_buffer_ = {};
        strategy_.reset();
        bilateral_grid_.reset();
        ppisp_.reset();
        ppisp_controller_pool_.reset();
        ppisp_exif_exposure_mean_.reset();
        eval_ppisp_applied_.store(0);
        eval_ppisp_exif_.store(0);
        sparsity_optimizer_.reset();
        evaluator_.reset();
        progress_.reset();
        base_dataset_.reset();
        train_dataset_.reset();
        val_dataset_.reset();
        setCameraLossHeatmap(nullptr);
        setActiveImageLoader(nullptr);

        if (callback_stream_) {
            LFS_CUDA_LOG_TEARDOWN(cudaStreamDestroy(callback_stream_), callback_stream_,
                                  "shutdown: destroy callback stream");
            callback_stream_ = nullptr;
        }
        callback_busy_ = false;

        if (metrics_stream_) {
            LFS_CUDA_LOG_TEARDOWN(cudaStreamDestroy(metrics_stream_), metrics_stream_,
                                  "shutdown: destroy metrics stream");
            metrics_stream_ = nullptr;
        }

        if (training_stream_) {
            destroySyncPrimitives();
            LFS_CUDA_LOG_TEARDOWN(cudaStreamDestroy(training_stream_), training_stream_,
                                  "shutdown: destroy training stream");
            training_stream_ = nullptr;
        }

        if (!exiting_headless) {
            // Never let a
            // poisoned CUDA context turn arena full_reset into std::terminate.
            try {
                lfs::core::Tensor::trim_memory_pool();
            } catch (const std::exception& e) {
                LOG_ERROR("Trainer::shutdown trim_memory_pool failed (continuing): {}",
                          e.what());
            } catch (...) {
                LOG_ERROR("Trainer::shutdown trim_memory_pool failed (unknown; continuing)");
            }
            try {
                lfs::core::GlobalArenaManager::instance().get_arena().full_reset();
            } catch (const std::exception& e) {
                LOG_ERROR("Trainer::shutdown arena full_reset failed (continuing): {}",
                          e.what());
            } catch (...) {
                LOG_ERROR("Trainer::shutdown arena full_reset failed (unknown; continuing)");
            }
            LFS_CUDA_LOG_TEARDOWN(cudaDeviceSynchronize(), nullptr,
                                  "shutdown: post-trim device sync");
        }
        LOG_DEBUG("GPU memory released");

        initialized_ = false;
        pause_requested_ = false;
        is_paused_ = false;
        {
            std::lock_guard<std::mutex> lock(params_mutex_);
            is_running_ = false;
        }
        training_complete_ = false;
    }

    void Trainer::setParams(const lfs::core::param::TrainingParameters& params) {
        if (const auto validation_error = params.validate(); !validation_error.empty()) {
            LOG_ERROR("Rejected invalid training parameter update: {}", validation_error);
            return;
        }

        bool bg_image_path_changed = false;
        {
            std::lock_guard<std::mutex> lock(params_mutex_);
            if (is_running_.load(std::memory_order_acquire)) {
                pending_params_ = params;
                return;
            }
            const auto& current = pending_params_ ? *pending_params_ : params_;
            bg_image_path_changed =
                params.optimization.bg_image_path != current.optimization.bg_image_path;
            params_ = params;
            pending_params_.reset();
        }
        apply_param_side_effects(params, bg_image_path_changed);
    }

    void Trainer::apply_pending_params_at_safe_point() {
        std::optional<lfs::core::param::TrainingParameters> update;
        bool bg_image_path_changed = false;
        {
            std::lock_guard<std::mutex> lock(params_mutex_);
            if (!pending_params_) {
                return;
            }
            update = std::move(pending_params_);
            pending_params_.reset();
            bg_image_path_changed =
                update->optimization.bg_image_path != params_.optimization.bg_image_path;
            params_ = *update;
        }
        apply_param_side_effects(*update, bg_image_path_changed);
    }

    void Trainer::apply_param_side_effects(
        const lfs::core::param::TrainingParameters& params,
        const bool bg_image_path_changed) {
        // Metrics render under render_mutex_ shared and read background_ directly.
        // Publish the parameter-dependent tensors as one exclusive update.
        const std::unique_lock<std::shared_mutex> render_lock(render_mutex_);
        const bool bg_mode_is_image =
            params.optimization.bg_mode == lfs::core::param::BackgroundMode::Image;

        // Load/reload background image if needed
        if (bg_mode_is_image && bg_image_path_changed &&
            !params.optimization.bg_image_path.empty() &&
            std::filesystem::exists(params.optimization.bg_image_path)) {
            try {
                auto& loader = lfs::io::CacheLoader::getInstance();
                lfs::io::LoadParams load_params{
                    .resize_factor = 1,
                    .max_width = 0,
                    .cuda_stream = nullptr};
                bg_image_base_ = loader.load_cached_image(params.optimization.bg_image_path, load_params);
                if (bg_image_base_.device() != lfs::core::Device::CUDA) {
                    bg_image_base_ = bg_image_base_.to(lfs::core::Device::CUDA);
                }
                clearBackgroundImageCache();
                if (bg_image_base_.shape()[0] != 3) {
                    LOG_WARN("Background image has {} channels, expected 3 (RGB)", bg_image_base_.shape()[0]);
                    bg_image_base_ = {};
                    std::lock_guard<std::mutex> lock(params_mutex_);
                    params_.optimization.bg_mode = lfs::core::param::BackgroundMode::SolidColor;
                } else {
                    LOG_INFO("Background image: {} [{}x{}]",
                             lfs::core::path_to_utf8(params.optimization.bg_image_path),
                             bg_image_base_.shape()[2], bg_image_base_.shape()[1]);
                }
            } catch (const std::exception& e) {
                LOG_WARN("Failed to load background image: {}", e.what());
                std::lock_guard<std::mutex> lock(params_mutex_);
                params_.optimization.bg_mode = lfs::core::param::BackgroundMode::SolidColor;
            }
        }

        if (!bg_mode_is_image && (bg_image_base_.is_valid() || !bg_image_cache_.empty())) {
            clearBackgroundImageCache();
            bg_image_base_ = {};
        }

        // Update background color tensor if changed
        const auto& bg_color = params.optimization.bg_color;
        if (background_.is_valid()) {
            auto bg_cpu = lfs::core::Tensor::empty({3}, lfs::core::Device::CPU, lfs::core::DataType::Float32);
            auto* bg_ptr = bg_cpu.ptr<float>();
            bg_ptr[0] = bg_color[0];
            bg_ptr[1] = bg_color[1];
            bg_ptr[2] = bg_color[2];
            background_ = bg_cpu.to(lfs::core::Device::CUDA);
        }
    }

    TrainingProgress::Phase Trainer::get_progress_phase(
        const int iter,
        const bool in_controller_phase) const {

        if (in_controller_phase) {
            return TrainingProgress::Phase::Controller;
        }

        const bool in_sparsification = get_active_sparsify_steps() > 0 &&
                                       iter > get_sparsity_boundary_iteration();
        if (in_sparsification) {
            return TrainingProgress::Phase::Sparse;
        }

        if (strategy_ && strategy_->is_refining(iter)) {
            return TrainingProgress::Phase::Refine;
        }

        return TrainingProgress::Phase::Train;
    }

    std::uint64_t Trainer::request_project_save(
        std::filesystem::path path,
        std::vector<std::byte> preview_png,
        std::optional<ProjectSnapshotDocumentContext>
            document_context) {
        if (path.empty()) {
            if (auto bound = bound_project_path()) {
                path = std::move(*bound);
            }
        }
        if (path.empty()) {
            std::lock_guard lock(project_snapshot_mutex_);
            const auto request_id =
                next_project_snapshot_request_id_++;
            last_project_writer_error_ =
                "No project destination is bound for this trainer";
            LOG_ERROR(
                "Cannot save project snapshot: {}",
                last_project_writer_error_);
            last_failed_project_request_id_ =
                std::max(
                    last_failed_project_request_id_,
                    request_id);
            return request_id;
        }
        attach_live_document_context(document_context);
        auto chapters =
            reserve_project_snapshot_chapters();
        std::lock_guard lock(project_snapshot_mutex_);
        const auto request_id =
            next_project_snapshot_request_id_++;
        if (!chapters) {
            const auto message =
                lfs::format_for_developer(
                    chapters.error());
            last_project_writer_error_ = message;
            LOG_ERROR(
                "Cannot reserve .licht snapshot identity: {}",
                message);
            last_failed_project_request_id_ =
                std::max(
                    last_failed_project_request_id_,
                    request_id);
            return request_id;
        }
        if (requested_project_request_id_) {
            last_failed_project_request_id_ =
                std::max(
                    last_failed_project_request_id_,
                    *requested_project_request_id_);
            last_project_writer_error_ =
                "A newer project snapshot request replaced the pending request";
        }
        (*chapters)->document_context =
            std::move(document_context);
        if ((*chapters)->document_context &&
            pending_snapshot_finish_reason_ !=
                lfs::io::project::
                    TrainingFinishReason::None) {
            (*chapters)
                ->document_context->metrics
                .finish_reason =
                pending_snapshot_finish_reason_;
        }
        // Coalesce to the newest destination. The next post-step safe point
        // consumes this request.
        prestaged_project_chapters_ =
            std::move(*chapters);
        prestaged_project_request_id_ =
            request_id;
        requested_project_path_ = std::move(path);
        requested_project_preview_png_ =
            std::move(preview_png);
        requested_project_request_id_ =
            request_id;
        requested_project_write_kind_ =
            ProjectSnapshotWriteKind::
                Explicit;
        requested_project_base_commit_uuid_ =
            {};
        requested_project_autosave_sequence_ =
            0;
        return request_id;
    }

    std::uint64_t
    Trainer::request_project_autosave(
        std::filesystem::path master_path,
        const lfs::core::Uuid
            base_explicit_commit_uuid,
        const std::uint64_t
            autosave_sequence,
        std::optional<
            ProjectSnapshotDocumentContext>
            document_context) {
        auto chapters =
            reserve_project_snapshot_chapters();
        std::lock_guard lock(
            project_snapshot_mutex_);
        const auto request_id =
            next_project_snapshot_request_id_++;
        if (!chapters ||
            master_path.empty() ||
            base_explicit_commit_uuid.is_nil() ||
            autosave_sequence == 0) {
            last_project_writer_error_ =
                chapters
                    ? "Autosave request has an invalid master binding"
                    : lfs::format_for_developer(
                          chapters.error());
            last_failed_project_request_id_ =
                std::max(
                    last_failed_project_request_id_,
                    request_id);
            return request_id;
        }
        if (requested_project_request_id_) {
            last_failed_project_request_id_ =
                std::max(
                    last_failed_project_request_id_,
                    *requested_project_request_id_);
            last_project_writer_error_ =
                "A newer project snapshot request replaced the pending request";
        }
        (*chapters)->document_context =
            std::move(document_context);
        prestaged_project_chapters_ =
            std::move(*chapters);
        prestaged_project_request_id_ =
            request_id;
        requested_project_path_ =
            std::move(master_path);
        requested_project_preview_png_.clear();
        requested_project_request_id_ =
            request_id;
        requested_project_write_kind_ =
            ProjectSnapshotWriteKind::
                Autosave;
        requested_project_base_commit_uuid_ =
            base_explicit_commit_uuid;
        requested_project_autosave_sequence_ =
            autosave_sequence;
        return request_id;
    }

    void Trainer::cancel_project_snapshot_request(
        const std::uint64_t request_id,
        const lfs::Error& reason) {
        const auto message =
            lfs::format_for_developer(reason);
        std::lock_guard lock(
            project_snapshot_mutex_);
        fail_project_request_locked(
            request_id, message);
        if (requested_project_request_id_ &&
            *requested_project_request_id_ ==
                request_id) {
            requested_project_path_.reset();
            requested_project_preview_png_.clear();
            requested_project_request_id_.reset();
            requested_project_write_kind_ =
                ProjectSnapshotWriteKind::Explicit;
            requested_project_base_commit_uuid_ = {};
            requested_project_autosave_sequence_ = 0;
        }
        if (prestaged_project_request_id_ ==
            request_id) {
            prestaged_project_chapters_.reset();
            prestaged_project_request_id_ = 0;
        }
        if (prepared_project_request_id_ ==
            request_id) {
            clear_prepared_project_request();
        }
    }

    lfs::Result<void>
    Trainer::initialize_project_snapshot_service() {
        if (!project_snapshot_service_ || !strategy_) {
            return lfs::Status::failure(
                project_snapshot_error(
                    lfs::ErrorCode::FailedPrecondition,
                    "Trainer has no snapshot service or strategy",
                    LFS_SOURCE_SITE_CURRENT()));
        }
        const auto checkpoint_params =
            params_for_project_snapshot();
        const std::array<cudaStream_t, 3>
            mutating_streams{
                training_stream_,
                metrics_stream_,
                callback_stream_,
            };
        const TrainingSnapshotCaptureRequest request{
            .iteration = current_iteration_.load(),
            .strategy = *strategy_,
            .params = checkpoint_params,
            .bilateral_grid = bilateral_grid_.get(),
            .ppisp = ppisp_.get(),
            .ppisp_controller_pool =
                controller_pool_for_save(
                    current_iteration_.load()),
            .sparsity_optimizer =
                dynamic_cast<
                    const ADMMSparsityOptimizer*>(
                    sparsity_optimizer_.get()),
            .mutating_streams = mutating_streams,
        };
        return project_snapshot_service_->initialize(
            request);
    }

    lfs::Result<
        std::shared_ptr<ProjectSnapshotChapters>>
    Trainer::reserve_project_snapshot_chapters() const {
        if (!scene_) {
            return project_snapshot_error(
                lfs::ErrorCode::FailedPrecondition,
                "Snapshot scene is unavailable",
                LFS_SOURCE_SITE_CURRENT());
        }
        const auto training_uuid =
            scene_->getTrainingModelNodeUuid();
        if (training_uuid.is_nil()) {
            return project_snapshot_error(
                lfs::ErrorCode::FailedPrecondition,
                "Snapshot scene has no training-model UUID",
                LFS_SOURCE_SITE_CURRENT());
        }

        auto chapters =
            std::make_shared<ProjectSnapshotChapters>();
        chapters->snapshot_uuid =
            lfs::core::generate_uuid_v4();
        return chapters;
    }

    Trainer::ProjectSnapshotRuntimeMetrics
    Trainer::get_project_snapshot_metrics() const {
        ProjectSnapshotRuntimeMetrics result;
        if (project_snapshot_service_) {
            result.capture =
                project_snapshot_service_->metrics();
        }
        std::lock_guard lock(project_snapshot_mutex_);
        result.last_path = last_project_snapshot_path_;
        result.last_writer_error =
            last_project_writer_error_;
        const auto step_metrics =
            project_step_regression_.metrics();
        result.pre_snapshot_step_mean_ms =
            step_metrics.pre_snapshot.mean_ms;
        result.pre_snapshot_step_first_iteration =
            step_metrics.pre_snapshot
                .first_iteration;
        result.pre_snapshot_step_last_iteration =
            step_metrics.pre_snapshot
                .last_iteration;
        result.pre_snapshot_step_samples =
            step_metrics.pre_snapshot
                .sample_count;
        result.post_resume_step_mean_ms =
            step_metrics.post_resume.mean_ms;
        result.post_resume_step_first_iteration =
            step_metrics.post_resume
                .first_iteration;
        result.post_resume_step_last_iteration =
            step_metrics.post_resume
                .last_iteration;
        result.post_resume_step_samples =
            step_metrics.post_resume
                .sample_count;
        result.post_resume_step_regression_percent =
            step_metrics.regression_percent;
        result.step_regression_gate_evaluated =
            step_metrics.gate_evaluated;
        result.step_regression_within_gate =
            step_metrics.within_gate;
        result.writer_in_flight =
            project_writer_in_flight_.load(
                std::memory_order_acquire);
        result.request_pending =
            requested_project_path_
                .has_value() ||
            result.writer_in_flight;
        result.last_completed_request_id =
            last_completed_project_request_id_;
        result.last_failed_request_id =
            last_failed_project_request_id_;
        result.last_autosave_sequence =
            last_completed_autosave_sequence_;
        result.last_completed_was_autosave =
            last_completed_was_autosave_;
        return result;
    }

    void Trainer::join_finished_project_writer() {
        if (project_writer_thread_.joinable() &&
            project_writer_done_.load(
                std::memory_order_acquire)) {
            project_writer_thread_.join();
        }
    }

    void Trainer::finish_project_writer() {
        if (project_writer_thread_.joinable()) {
            project_writer_thread_.join();
        }
        project_writer_done_.store(
            true, std::memory_order_release);
        project_writer_in_flight_.store(
            false, std::memory_order_release);
        abandon_pending_project_requests(
            "The pending project snapshot was abandoned before capture");
    }

    void Trainer::fail_project_request_locked(
        const std::uint64_t request_id,
        const std::string_view message) {
        if (request_id == 0) {
            return;
        }
        if (last_completed_project_request_id_ >=
                request_id ||
            last_failed_project_request_id_ >=
                request_id) {
            return;
        }
        last_failed_project_request_id_ =
            std::max(last_failed_project_request_id_,
                     request_id);
        last_project_writer_error_ = message;
    }

    void Trainer::clear_prepared_project_request() {
        prepared_project_snapshot_.reset();
        prepared_project_path_.clear();
        prepared_project_preview_png_.clear();
        prepared_project_iteration_ = 0;
        prepared_project_request_id_ = 0;
        prepared_project_write_kind_ =
            ProjectSnapshotWriteKind::Explicit;
        prepared_project_base_commit_uuid_ = {};
        prepared_project_autosave_sequence_ = 0;
    }

    void Trainer::abandon_pending_project_requests(
        const std::string_view reason) {
        std::lock_guard lock(project_snapshot_mutex_);
        if (requested_project_request_id_) {
            fail_project_request_locked(
                *requested_project_request_id_, reason);
        }
        fail_project_request_locked(
            prepared_project_request_id_, reason);
        fail_project_request_locked(
            prestaged_project_request_id_, reason);
        requested_project_path_.reset();
        requested_project_preview_png_.clear();
        requested_project_request_id_.reset();
        requested_project_write_kind_ =
            ProjectSnapshotWriteKind::Explicit;
        requested_project_base_commit_uuid_ = {};
        requested_project_autosave_sequence_ = 0;
        prestaged_project_chapters_.reset();
        prestaged_project_request_id_ = 0;
        clear_prepared_project_request();
    }

    void Trainer::prepare_project_snapshot_at_safe_point(
        const int capture_iteration,
        const std::filesystem::path& requested_path,
        std::vector<std::byte> preview_png,
        const std::uint64_t request_id,
        const ProjectSnapshotWriteKind write_kind,
        const lfs::core::Uuid
            base_explicit_commit_uuid,
        const std::uint64_t autosave_sequence) {
        join_finished_project_writer();
        if (write_kind ==
            ProjectSnapshotWriteKind::Autosave) {
            std::lock_guard lock(
                project_snapshot_mutex_);
            fail_project_request_locked(
                request_id,
                "Training-time autosave is light-only and does not capture a checkpoint");
            if (prestaged_project_request_id_ ==
                request_id) {
                prestaged_project_chapters_.reset();
                prestaged_project_request_id_ = 0;
            }
            LOG_INFO(
                "Refusing training autosave checkpoint capture for request {}",
                request_id);
            return;
        }
        if (!project_snapshot_service_ || !strategy_ ||
            !scene_) {
            {
                std::lock_guard lock(
                    project_snapshot_mutex_);
                fail_project_request_locked(
                    request_id,
                    "Trainer has no snapshot service, strategy, or scene");
                if (prestaged_project_request_id_ ==
                    request_id) {
                    prestaged_project_chapters_.reset();
                    prestaged_project_request_id_ = 0;
                }
            }
            LOG_ERROR(
                "Cannot prepare .licht training snapshot: "
                "trainer has no snapshot service, strategy, or scene");
            return;
        }
        if (project_writer_in_flight_.load(
                std::memory_order_acquire)) {
            LOG_INFO(
                "Coalescing .licht snapshot request at iteration {} "
                "because the prior project writer is still active",
                capture_iteration);
            std::lock_guard lock(project_snapshot_mutex_);
            if (requested_project_request_id_ &&
                *requested_project_request_id_ !=
                    request_id) {
                fail_project_request_locked(
                    request_id,
                    "A newer project snapshot request replaced the dequeued request");
                return;
            }
            requested_project_path_ = requested_path;
            requested_project_preview_png_ =
                std::move(preview_png);
            requested_project_request_id_ =
                request_id;
            requested_project_write_kind_ =
                write_kind;
            requested_project_base_commit_uuid_ =
                base_explicit_commit_uuid;
            requested_project_autosave_sequence_ =
                autosave_sequence;
            return;
        }

        auto path = requested_path;
        if (path.empty()) {
            LOG_ERROR(
                "No project destination is bound for this trainer");
            std::lock_guard lock(project_snapshot_mutex_);
            fail_project_request_locked(
                request_id,
                "No project destination is bound for this trainer");
            return;
        }
        if (path.extension() != ".licht") {
            LOG_ERROR(
                "Project snapshot destination must end in .licht: {}",
                lfs::core::path_to_utf8(path));
            std::lock_guard lock(project_snapshot_mutex_);
            fail_project_request_locked(
                request_id,
                "Project snapshot destination must end in .licht");
            return;
        }

        lfs::core::Uuid snapshot_uuid;
        {
            std::lock_guard lock(
                project_snapshot_mutex_);
            if (!prestaged_project_chapters_ ||
                prestaged_project_request_id_ !=
                    request_id ||
                prestaged_project_chapters_
                    ->snapshot_uuid.is_nil()) {
                fail_project_request_locked(
                    request_id,
                    "Snapshot UUID was not reserved for this request");
                LOG_ERROR(
                    "Cannot prepare .licht snapshot: snapshot UUID "
                    "was not reserved before the optimizer safe point");
                return;
            }
            snapshot_uuid =
                prestaged_project_chapters_
                    ->snapshot_uuid;
        }

        const auto checkpoint_params =
            params_for_project_snapshot();
        const std::array<cudaStream_t, 3>
            mutating_streams{
                training_stream_,
                metrics_stream_,
                callback_stream_,
            };
        const TrainingSnapshotCaptureRequest request{
            .iteration = capture_iteration,
            .snapshot_uuid = snapshot_uuid,
            .strategy = *strategy_,
            .params = checkpoint_params,
            .bilateral_grid = bilateral_grid_.get(),
            .ppisp = ppisp_.get(),
            .ppisp_controller_pool =
                controller_pool_for_save(
                    capture_iteration),
            .sparsity_optimizer =
                dynamic_cast<
                    const ADMMSparsityOptimizer*>(
                    sparsity_optimizer_.get()),
            .mutating_streams = mutating_streams,
        };
        auto prepared =
            project_snapshot_service_->prepare(request);
        if (!prepared) {
            const auto message =
                lfs::format_for_developer(
                    prepared.error());
            LOG_ERROR(
                "Failed to prepare .licht training snapshot "
                "at iteration {}: {}",
                capture_iteration, message);
            std::lock_guard lock(project_snapshot_mutex_);
            fail_project_request_locked(
                request_id, message);
            if (prestaged_project_request_id_ ==
                request_id) {
                prestaged_project_chapters_.reset();
                prestaged_project_request_id_ = 0;
            }
            return;
        }

        {
            std::lock_guard lock(project_snapshot_mutex_);
            last_project_writer_error_.clear();
        }
        prepared_project_snapshot_.emplace(
            std::move(*prepared));
        prepared_project_path_ = std::move(path);
        prepared_project_preview_png_ =
            std::move(preview_png);
        prepared_project_iteration_ =
            capture_iteration;
        prepared_project_request_id_ =
            request_id;
        prepared_project_write_kind_ =
            write_kind;
        prepared_project_base_commit_uuid_ =
            base_explicit_commit_uuid;
        prepared_project_autosave_sequence_ =
            autosave_sequence;
        LOG_INFO(
            "Prepared .licht snapshot {} for iteration {} "
            "({} checkpoint bytes)",
            prepared_project_snapshot_->snapshot_uuid()
                .to_string(),
            capture_iteration,
            prepared_project_snapshot_->checkpoint_bytes());
    }

    void Trainer::consume_requested_project_snapshot(
        const int iteration) {
        join_finished_project_writer();
        std::optional<std::filesystem::path>
            requested_project_path;
        std::vector<std::byte>
            requested_project_preview_png;
        std::uint64_t
            requested_project_request_id = 0;
        ProjectSnapshotWriteKind
            requested_project_write_kind =
                ProjectSnapshotWriteKind::
                    Explicit;
        lfs::core::Uuid
            requested_project_base_commit_uuid;
        std::uint64_t
            requested_project_autosave_sequence =
                0;
        {
            std::lock_guard lock(
                project_snapshot_mutex_);
            if (requested_project_path_) {
                requested_project_path =
                    std::move(
                        *requested_project_path_);
                requested_project_path_.reset();
                requested_project_preview_png =
                    std::move(
                        requested_project_preview_png_);
                requested_project_request_id =
                    requested_project_request_id_
                        .value_or(0);
                requested_project_request_id_
                    .reset();
                requested_project_write_kind =
                    std::exchange(
                        requested_project_write_kind_,
                        ProjectSnapshotWriteKind::
                            Explicit);
                requested_project_base_commit_uuid =
                    std::exchange(
                        requested_project_base_commit_uuid_,
                        lfs::core::Uuid{});
                requested_project_autosave_sequence =
                    std::exchange(
                        requested_project_autosave_sequence_,
                        0);
            }
        }
        if (requested_project_path) {
            if (requested_project_write_kind ==
                ProjectSnapshotWriteKind::
                    Autosave) {
                std::lock_guard lock(
                    project_snapshot_mutex_);
                fail_project_request_locked(
                    requested_project_request_id,
                    "Training-time autosave is light-only and does not capture a checkpoint");
                if (prestaged_project_request_id_ ==
                    requested_project_request_id) {
                    prestaged_project_chapters_.reset();
                    prestaged_project_request_id_ =
                        0;
                }
                LOG_INFO(
                    "Refusing training autosave checkpoint capture for request {}",
                    requested_project_request_id);
                return;
            }
            prepare_project_snapshot_at_safe_point(
                iteration, *requested_project_path,
                std::move(
                    requested_project_preview_png),
                requested_project_request_id,
                requested_project_write_kind,
                requested_project_base_commit_uuid,
                requested_project_autosave_sequence);
            capture_project_snapshot_at_safe_point(
                iteration);
        }
    }

    lfs::Result<std::filesystem::path>
    relocate_unreadable_trainer_default_project(
        const std::filesystem::path& destination,
        const lfs::Error& cause) {
        const auto unix_seconds =
            std::chrono::duration_cast<
                std::chrono::seconds>(
                std::chrono::system_clock::now()
                    .time_since_epoch())
                .count();
        auto aside = destination;
        aside += std::format(
            ".corrupt-{}", unix_seconds);
        std::error_code ec;
        std::filesystem::rename(
            destination, aside, ec);
        if (ec) {
            return lfs::from_std_error_code(
                ec,
                lfs::ErrorCode::Unavailable,
                lfs::ErrorDomain::IO,
                "rename unreadable trainer-default project aside",
                LFS_SOURCE_SITE_CURRENT());
        }
        LOG_WARN(
            "Unreadable trainer-default project renamed aside from {} to {}: {}",
            lfs::core::path_to_utf8(
                destination),
            lfs::core::path_to_utf8(aside),
            lfs::format_for_developer(cause));
        return aside;
    }

    void Trainer::capture_project_snapshot_at_safe_point(
        const int iteration) {
        if (!prepared_project_snapshot_ ||
            !project_snapshot_service_) {
            return;
        }
        if (prepared_project_iteration_ != iteration) {
            LOG_WARN(
                "Prepared .licht snapshot iteration {} reached "
                "safe point {}; coalescing for the current state",
                prepared_project_iteration_, iteration);
        }

        std::shared_ptr<ProjectSnapshotChapters>
            chapters;
        {
            std::lock_guard lock(
                project_snapshot_mutex_);
            if (prestaged_project_request_id_ ==
                prepared_project_request_id_) {
                chapters = std::move(
                    prestaged_project_chapters_);
                prestaged_project_request_id_ = 0;
            }
        }
        if (!chapters) {
            LOG_ERROR(
                "Cannot capture .licht snapshot: snapshot UUID "
                "was not reserved before the optimizer safe point");
            std::lock_guard lock(
                project_snapshot_mutex_);
            fail_project_request_locked(
                prepared_project_request_id_,
                "Snapshot UUID was not reserved for the prepared request");
            clear_prepared_project_request();
            return;
        }
        const auto checkpoint_params =
            params_for_project_snapshot();
        const std::array<cudaStream_t, 3>
            mutating_streams{
                training_stream_,
                metrics_stream_,
                callback_stream_,
            };
        const auto snapshot_uuid =
            prepared_project_snapshot_->snapshot_uuid();
        if (chapters->snapshot_uuid !=
            snapshot_uuid) {
            LOG_ERROR(
                "Prepared GPU snapshot UUID does not match "
                "the pre-staged CPU chapters");
            std::lock_guard lock(
                project_snapshot_mutex_);
            fail_project_request_locked(
                prepared_project_request_id_,
                "Prepared GPU/CPU snapshot UUID mismatch");
            clear_prepared_project_request();
            return;
        }
        auto cpu_state =
            std::make_shared<ProjectSnapshotCpuState>();
        const auto cpu_write_kind =
            prepared_project_write_kind_;
        TrainingSnapshotCaptureRequest request{
            .iteration = iteration,
            .snapshot_uuid = snapshot_uuid,
            .strategy = *strategy_,
            .params = checkpoint_params,
            .bilateral_grid = bilateral_grid_.get(),
            .ppisp = ppisp_.get(),
            .ppisp_controller_pool =
                controller_pool_for_save(iteration),
            .sparsity_optimizer =
                dynamic_cast<
                    const ADMMSparsityOptimizer*>(
                    sparsity_optimizer_.get()),
            .mutating_streams = mutating_streams,
            .capture_additional_cpu_state =
                [this, cpu_state, chapters,
                 cpu_write_kind, iteration,
                 &checkpoint_params](
                    const lfs::core::Uuid&
                        captured_uuid)
                -> lfs::Result<
                    TrainingSnapshotCpuStateMetrics> {
                if (!scene_) {
                    return project_snapshot_error(
                        lfs::ErrorCode::
                            FailedPrecondition,
                        "Snapshot scene disappeared",
                        LFS_SOURCE_SITE_CURRENT());
                }
                if (chapters->document_context) {
                    const auto& context =
                        *chapters->document_context;
                    return capture_project_snapshot_cpu_state(
                        *scene_, context.parameters,
                        captured_uuid, iteration,
                        *cpu_state,
                        context.selected_node_uuids);
                }
                if (cpu_write_kind ==
                    ProjectSnapshotWriteKind::Autosave) {
                    return project_snapshot_error(
                        lfs::ErrorCode::FailedPrecondition,
                        "Training autosave is missing live GUI project state",
                        LFS_SOURCE_SITE_CURRENT());
                }
                return capture_project_snapshot_cpu_state(
                    *scene_, checkpoint_params,
                    captured_uuid, iteration,
                    *cpu_state);
            },
        };

        request.safe_point_entered_at =
            std::chrono::steady_clock::now();
        // Block CPU-side reader setup while the service establishes all GPU
        // stream edges and captures the immutable state.
        std::unique_lock render_lock(render_mutex_);
        std::unique_lock model_lock(
            model_access_mutex_);
        waitForModelReaders();
        auto prepared =
            std::move(*prepared_project_snapshot_);
        prepared_project_snapshot_.reset();
        auto pending =
            project_snapshot_service_->capture(
                std::move(prepared), request);
        render_lock.unlock();
        model_lock.unlock();

        const auto path =
            std::move(prepared_project_path_);
        auto preview_png =
            std::move(
                prepared_project_preview_png_);
        const auto request_id =
            std::exchange(
                prepared_project_request_id_,
                0);
        const auto write_kind =
            std::exchange(
                prepared_project_write_kind_,
                ProjectSnapshotWriteKind::
                    Explicit);
        const auto base_explicit_commit_uuid =
            std::exchange(
                prepared_project_base_commit_uuid_,
                lfs::core::Uuid{});
        const auto autosave_sequence =
            std::exchange(
                prepared_project_autosave_sequence_,
                0);
        prepared_project_iteration_ = 0;
        if (!pending) {
            const auto message =
                lfs::format_for_developer(
                    pending.error());
            const bool requires_replan =
                pending.error().code() ==
                    lfs::ErrorCode::
                        FailedPrecondition &&
                pending.error().detail().find(
                    "layout changed") !=
                    std::string_view::npos;
            if (requires_replan) {
                LOG_WARN(
                    "Replanning .licht snapshot at iteration {}: "
                    "reason={}",
                    iteration, message);
            } else {
                LOG_ERROR(
                    "Failed to capture .licht training snapshot "
                    "at iteration {}: {}",
                    iteration, message);
            }
            {
                std::lock_guard lock(
                    project_snapshot_mutex_);
                last_project_writer_error_ = message;
                // A topology-changing step may invalidate a preplanned
                // layout. Coalesce and replan at the next post-step point.
                if (!requested_project_path_) {
                    requested_project_path_ = path;
                    requested_project_preview_png_ =
                        std::move(preview_png);
                    requested_project_request_id_ =
                        request_id;
                    requested_project_write_kind_ =
                        write_kind;
                    requested_project_base_commit_uuid_ =
                        base_explicit_commit_uuid;
                    requested_project_autosave_sequence_ =
                        autosave_sequence;
                    prestaged_project_chapters_ =
                        chapters;
                    prestaged_project_request_id_ =
                        request_id;
                } else {
                    fail_project_request_locked(
                        request_id,
                        "A newer project snapshot request replaced the failed capture");
                }
            }
            return;
        }
        if (cpu_state->snapshot_uuid !=
                snapshot_uuid ||
            cpu_state->iteration != iteration) {
            LOG_ERROR(
                "Snapshot CPU-state UUID/iteration proof failed");
            std::lock_guard lock(
                project_snapshot_mutex_);
            last_project_writer_error_ =
                "Snapshot CPU-state UUID/iteration proof failed";
            last_failed_project_request_id_ =
                std::max(
                    last_failed_project_request_id_,
                    request_id);
            return;
        }

        photometric_loss_.arena().shrink_to_required();

        bool first_publish_to_destination =
            false;
        {
            std::lock_guard lock(
                project_snapshot_mutex_);
            project_step_regression_
                .arm_after_snapshot(iteration);
            first_publish_to_destination =
                last_project_snapshot_path_
                    .lexically_normal() !=
                path.lexically_normal();
        }
        project_writer_done_.store(
            false, std::memory_order_release);
        project_writer_in_flight_.store(
            true, std::memory_order_release);

        try {
            project_writer_thread_ = std::jthread(
                [this, path, chapters, cpu_state,
                 request_id, write_kind,
                 base_explicit_commit_uuid,
                 autosave_sequence,
                 first_publish_to_destination,
                 recovery_session =
                     recovery_session_,
                 preview_png =
                     std::move(preview_png),
                 pending = std::move(*pending)](
                    std::stop_token) mutable {
#if defined(__linux__)
                    const int current_priority =
                        getpriority(PRIO_PROCESS, 0);
                    const int background_priority =
                        std::max(current_priority, 10);
                    if (background_priority >
                        current_priority) {
                        static_cast<void>(setpriority(
                            PRIO_PROCESS, 0,
                            background_priority));
                    }
#endif
                    bool compact_after_publish = false;
                    auto writer_result =
                        [&]()
                        -> lfs::Result<
                            TrainingSnapshotPauseMetrics> {
                        if (path.empty()) {
                            return project_snapshot_error(
                                lfs::ErrorCode::
                                    FailedPrecondition,
                                "No project destination is bound for this trainer",
                                LFS_SOURCE_SITE_CURRENT());
                        }
                        if (auto materialized =
                                materialize_project_snapshot_cpu_chapters(
                                    std::move(*cpu_state),
                                    *chapters);
                            !materialized) {
                            return std::move(materialized)
                                .error()
                                .with_context(
                                    "materialize captured project CPU chapters",
                                    LFS_SOURCE_SITE_CURRENT());
                        }
                        auto captured = pending.wait();
                        if (!captured) {
                            return std::move(captured)
                                .error()
                                .with_context(
                                    "drain captured training snapshot",
                                    LFS_SOURCE_SITE_CURRENT());
                        }
                        if (!captured->metrics
                                 .consistency_proven ||
                            captured->snapshot_uuid !=
                                chapters
                                    ->snapshot_uuid ||
                            captured->iteration !=
                                chapters->iteration ||
                            !captured
                                 ->checkpoint_bytes) {
                            return project_snapshot_error(
                                lfs::ErrorCode::
                                    ContractViolation,
                                "Captured CKPT and CPU chapter stamps disagree",
                                LFS_SOURCE_SITE_CURRENT());
                        }

                        std::error_code ec;
                        if (!path.parent_path()
                                 .empty()) {
                            std::filesystem::
                                create_directories(
                                    path.parent_path(),
                                    ec);
                        }
                        if (ec) {
                            return lfs::from_std_error_code(
                                ec,
                                lfs::ErrorCode::
                                    Unavailable,
                                lfs::ErrorDomain::IO,
                                "create training project snapshot directory",
                                LFS_SOURCE_SITE_CURRENT());
                        }

                        std::optional<
                            lfs::io::project::
                                ProjectDocument>
                            document;
                        const auto* const
                            document_context =
                                chapters
                                        ->document_context
                                    ? &*chapters
                                            ->document_context
                                    : nullptr;
                        std::optional<
                            std::filesystem::path>
                            source_path;
                        bool adopt_container_identity =
                            false;
                        if (document_context &&
                            document_context
                                ->source_path) {
                            source_path =
                                *document_context
                                     ->source_path;
                        } else if (
                            !document_context &&
                            std::filesystem::exists(
                                path)) {
                            source_path = path;
                            compact_after_publish =
                                first_publish_to_destination;
                        }
                        bool save_as =
                            !source_path &&
                            std::filesystem::exists(
                                path);
                        const auto create_fresh_document =
                            [&]()
                            -> lfs::Result<void> {
                            const auto created_ns =
                                static_cast<std::uint64_t>(
                                    std::chrono::
                                        duration_cast<
                                            std::chrono::
                                                nanoseconds>(
                                            std::chrono::
                                                system_clock::
                                                    now()
                                                        .time_since_epoch())
                                            .count());
                            auto created =
                                lfs::io::project::
                                    ProjectDocument::create(
                                        document_context
                                            ? document_context
                                                  ->project_uuid
                                            : project_uuid_,
                                        created_ns);
                            if (!created) {
                                return lfs::Status::failure(
                                    std::move(created)
                                        .error()
                                        .with_context(
                                            "create training project snapshot document",
                                            LFS_SOURCE_SITE_CURRENT()));
                            }
                            document.emplace(
                                std::move(*created));
                            return {};
                        };
                        if (source_path) {
                            auto opened =
                                lfs::io::project::
                                    ProjectDocument::open(
                                        *source_path,
                                        {
                                            .reader = {},
                                            .geometry = {},
                                            .defer_geometry_payloads =
                                                true,
                                        });
                            if (!opened) {
                                if (stale_trainer_default_recovery(
                                        adopt_container_identity,
                                        true,
                                        std::nullopt) ==
                                    StaleTrainerDefaultRecovery::
                                        RelocateAndFirstSave) {
                                    if (auto relocated =
                                            relocate_unreadable_trainer_default_project(
                                                path,
                                                opened.error());
                                        !relocated) {
                                        return std::move(
                                                   opened)
                                            .error()
                                            .with_context(
                                                "open training project for snapshot append",
                                                LFS_SOURCE_SITE_CURRENT());
                                    }
                                    source_path.reset();
                                    adopt_container_identity =
                                        false;
                                    save_as = false;
                                } else {
                                    return std::move(
                                               opened)
                                        .error()
                                        .with_context(
                                            "open training project for snapshot append",
                                            LFS_SOURCE_SITE_CURRENT());
                                }
                            } else {
                                save_as =
                                    source_path
                                        ->lexically_normal() !=
                                    path
                                        .lexically_normal();
                                document.emplace(
                                    std::move(*opened));
                            }
                        }
                        if (!document) {
                            if (auto created =
                                    create_fresh_document();
                                !created) {
                                return std::move(created)
                                    .error();
                            }
                        }

                        const auto stage_snapshot_chapters =
                            [&](lfs::io::project::
                                    ProjectDocument*
                                        previous)
                            -> lfs::Result<void> {
                            if (document_context) {
                                std::optional<
                                    lfs::core::Uuid>
                                    container_uuid;
                                std::optional<
                                    std::uint64_t>
                                    container_created_at;
                                if (adopt_container_identity) {
                                    container_uuid =
                                        document
                                            ->project_uuid();
                                    auto created_at =
                                        document
                                            ->project()
                                            .created_at_unix_ns();
                                    if (!created_at) {
                                        return lfs::Status::failure(
                                            std::move(
                                                created_at)
                                                .error()
                                                .with_context(
                                                    "read destination container created_at",
                                                    LFS_SOURCE_SITE_CURRENT()));
                                    }
                                    container_created_at =
                                        *created_at;
                                }
                                document->edit_project() =
                                    document_context
                                        ->project;
                                if (adopt_container_identity) {
                                    if (auto restored =
                                            document
                                                ->edit_project()
                                                .set_project_uuid(
                                                    *container_uuid);
                                        !restored) {
                                        return lfs::Status::failure(
                                            std::move(
                                                restored)
                                                .error()
                                                .with_context(
                                                    "restore destination container project uuid",
                                                    LFS_SOURCE_SITE_CURRENT()));
                                    }
                                    if (auto restored =
                                            document
                                                ->edit_project()
                                                .set_created_at_unix_ns(
                                                    *container_created_at);
                                        !restored) {
                                        return lfs::Status::failure(
                                            std::move(
                                                restored)
                                                .error()
                                                .with_context(
                                                    "restore destination container created_at",
                                                    LFS_SOURCE_SITE_CURRENT()));
                                    }
                                }
                                document->edit_references() =
                                    document_context
                                        ->references;
                                document->edit_gui_layout() =
                                    document_context
                                        ->gui_layout;
                                document->edit_view() =
                                    document_context
                                        ->view;
                                document->edit_editor() =
                                    document_context
                                        ->editor;
                                document->edit_sequencer() =
                                    document_context
                                        ->sequencer;
                                document->edit_metrics() =
                                    document_context
                                        ->metrics;
                            }
                            if (previous) {
                                document->edit_scene_graph() =
                                    std::move(
                                        previous
                                            ->edit_scene_graph());
                                document->edit_selection() =
                                    std::move(
                                        previous
                                            ->edit_selection());
                            } else {
                                document->edit_scene_graph() =
                                    std::move(
                                        chapters
                                            ->scene_graph);
                                document->edit_selection() =
                                    std::move(
                                        chapters
                                            ->selection);
                            }
                            document
                                ->remove_geometry_payloads_not_bound_by_scene();
                            auto params_status =
                                document
                                    ->edit_parameters()
                                    .set_snapshot(
                                        chapters
                                            ->parameters);
                            if (!params_status) {
                                return lfs::Status::failure(
                                    std::move(
                                        params_status)
                                        .error()
                                        .with_context(
                                            "stage project parameter snapshot",
                                            LFS_SOURCE_SITE_CURRENT()));
                            }
                            for (const auto& uuid :
                                 document
                                     ->checkpoint_uuids()) {
                                static_cast<void>(
                                    document
                                        ->remove_checkpoint(
                                            uuid));
                            }
                            auto lazy =
                                lfs::io::project::
                                    LazyChunkValue::from_owned(
                                        captured
                                            ->checkpoint_bytes,
                                        captured
                                            ->snapshot_uuid);
                            if (!lazy) {
                                return lfs::Status::failure(
                                    std::move(lazy)
                                        .error()
                                        .with_context(
                                            "own captured CKPT bytes",
                                            LFS_SOURCE_SITE_CURRENT()));
                            }
                            auto checkpoint_status =
                                document->set_checkpoint(
                                    captured
                                        ->snapshot_uuid,
                                    std::move(*lazy));
                            if (!checkpoint_status) {
                                return lfs::Status::failure(
                                    std::move(
                                        checkpoint_status)
                                        .error()
                                        .with_context(
                                            "bind captured CKPT chapter",
                                            LFS_SOURCE_SITE_CURRENT()));
                            }
                            return {};
                        };
                        if (auto staged =
                                stage_snapshot_chapters(
                                    nullptr);
                            !staged) {
                            return std::move(staged)
                                .error();
                        }

                        const auto wallclock_ns =
                            static_cast<std::uint64_t>(
                                std::chrono::
                                    duration_cast<
                                        std::chrono::
                                            nanoseconds>(
                                        std::chrono::
                                            system_clock::
                                                now()
                                                    .time_since_epoch())
                                        .count());
                        const lfs::io::project::
                            ProjectDocumentSaveOptions
                                save_options{
                                    .commit =
                                        {
                                            .kind =
                                                document_context
                                                    ? document_context
                                                          ->durable_commit_kind
                                                    : (recovery_session
                                                           ? lfs::io::project::
                                                                 CommitKind::Recovered
                                                           : lfs::io::project::
                                                                 CommitKind::Explicit),
                                            .commit_uuid =
                                                lfs::core::
                                                    generate_uuid_v4(),
                                            .snapshot_uuid =
                                                captured
                                                    ->snapshot_uuid,
                                            .wallclock_unix_ns =
                                                wallclock_ns,
                                            .extra_reader_capabilities =
                                                {},
                                            .extra_writer_capabilities =
                                                {},
                                        },
                                    .file_uuid =
                                        lfs::core::
                                            generate_uuid_v4(),
                                    .allow_existing_destination_replacement =
                                        document_context &&
                                        document_context
                                            ->allow_existing_destination_replacement,
                                    .preview_png =
                                        std::span<
                                            const std::byte>(
                                            preview_png),
                                    .writer_lock_lease =
                                        document_context &&
                                                document_context
                                                    ->writer_lock_lease
                                            ? document_context
                                                  ->writer_lock_lease
                                            : (recovery_session
                                                   ? std::optional{
                                                         recovery_session
                                                             ->writer_lock()}
                                                   : std::nullopt),
                                    .writer_lock_wait = std::chrono::seconds(30),
                                };
                        const auto save_snapshot =
                            [&]()
                            -> lfs::Result<
                                lfs::io::project::
                                    ProjectDocumentSaveReport> {
                            if (write_kind ==
                                ProjectSnapshotWriteKind::
                                    Autosave) {
                                if (save_as) {
                                    return lfs::Result<
                                        lfs::io::project::
                                            ProjectDocumentSaveReport>{
                                        project_snapshot_error(
                                            lfs::ErrorCode::
                                                FailedPrecondition,
                                            "Training autosave source differs from its bound master",
                                            LFS_SOURCE_SITE_CURRENT())};
                                }
                                return document
                                    ->save_autosave(
                                        lfs::io::project::
                                            autosave_sidecar_path(
                                                path),
                                        lfs::io::project::
                                            ProjectDocumentAutosaveOptions{
                                                .file_uuid =
                                                    lfs::core::
                                                        generate_uuid_v4(),
                                                .base_explicit_commit_uuid =
                                                    base_explicit_commit_uuid,
                                                .autosave_sequence =
                                                    autosave_sequence,
                                                .snapshot_uuid =
                                                    captured
                                                        ->snapshot_uuid,
                                                .index_compression =
                                                    lfs::io::project::
                                                        IndexCompression::
                                                            Zstd,
                                                .disk_reserve_bytes =
                                                    64ull *
                                                    1024 *
                                                    1024,
                                                .boundary_observer =
                                                    {},
                                            });
                            }
                            return save_as
                                       ? document
                                             ->save_as(
                                                 path,
                                                 save_options)
                                       : document->save(
                                             path,
                                             save_options);
                        };
                        auto saved = save_snapshot();
                        if (!saved &&
                            stale_trainer_default_recovery(
                                adopt_container_identity,
                                false,
                                saved.error().code()) ==
                                StaleTrainerDefaultRecovery::
                                    RelocateAndFirstSave) {
                            if (auto relocated =
                                    relocate_unreadable_trainer_default_project(
                                        path,
                                        saved.error());
                                !relocated) {
                                return std::move(saved)
                                    .error()
                                    .with_context(
                                        "append captured training project generation",
                                        LFS_SOURCE_SITE_CURRENT());
                            }
                            source_path.reset();
                            adopt_container_identity =
                                false;
                            save_as = false;
                            auto previous =
                                std::move(document);
                            if (auto created =
                                    create_fresh_document();
                                !created) {
                                return std::move(created)
                                    .error();
                            }
                            if (auto staged =
                                    stage_snapshot_chapters(
                                        previous
                                            ? &*previous
                                            : nullptr);
                                !staged) {
                                return std::move(staged)
                                    .error();
                            }
                            saved = save_snapshot();
                        }
                        if (!saved) {
                            return std::move(saved)
                                .error()
                                .with_context(
                                    "append captured training project generation",
                                    LFS_SOURCE_SITE_CURRENT());
                        }
                        if (write_kind ==
                            ProjectSnapshotWriteKind::
                                Explicit) {
                            const bool gui_retains_lock =
                                document_context &&
                                document_context
                                    ->writer_lock_lease
                                    .has_value();
                            if (recovery_session) {
                                if (auto released =
                                        recovery_session
                                            ->release();
                                    !released) {
                                    if (recovery_session
                                            ->document_attached() &&
                                        released.error().code() ==
                                            lfs::ErrorCode::
                                                FailedPrecondition) {
                                        LOG_DEBUG(
                                            "Training recovery merge is durable; the headless document owner retains staging cleanup for {}",
                                            lfs::core::
                                                path_to_utf8(
                                                    path));
                                    } else {
                                        LOG_WARN(
                                            "Training recovery merge is durable, but recovery staging cleanup failed for {}: {}",
                                            lfs::core::
                                                path_to_utf8(
                                                    path),
                                            lfs::format_for_developer(
                                                released.error()));
                                    }
                                }
                                const std::lock_guard lock(
                                    project_snapshot_mutex_);
                                recovery_session_.reset();
                            }
                            if (!gui_retains_lock) {
                                if (auto removed =
                                        lfs::io::project::
                                            remove_autosave_artifacts(
                                                path);
                                    !removed) {
                                    LOG_WARN(
                                        "Explicit training project save is durable, but autosave cleanup failed for {}: {}",
                                        lfs::core::
                                            path_to_utf8(
                                                path),
                                        lfs::format_for_developer(
                                            removed.error()));
                                }
                            }
                        }
                        LOG_INFO(
                            "Background .licht {} complete: {} "
                            "generation={} snapshot={} rewritten={} "
                            "reused={} final_drain={:.3f}ms",
                            write_kind ==
                                    ProjectSnapshotWriteKind::
                                        Autosave
                                ? "autosave"
                                : "append",
                            lfs::core::path_to_utf8(
                                path),
                            saved->generation,
                            saved->snapshot_uuid
                                .to_string(),
                            saved->rewritten_chunks,
                            saved->reused_chunks,
                            captured->metrics
                                .final_drain_ms);
                        return captured->metrics;
                    }();

                    if (writer_result &&
                        compact_after_publish &&
                        write_kind !=
                            ProjectSnapshotWriteKind::
                                Autosave) {
                        const auto compact_creation_ns =
                            static_cast<std::uint64_t>(
                                std::chrono::
                                    duration_cast<
                                        std::chrono::
                                            nanoseconds>(
                                        std::chrono::
                                            system_clock::
                                                now()
                                                    .time_since_epoch())
                                        .count());
                        const auto compact_wallclock_ns =
                            static_cast<std::uint64_t>(
                                std::chrono::
                                    duration_cast<
                                        std::chrono::
                                            nanoseconds>(
                                        std::chrono::
                                            system_clock::
                                                now()
                                                    .time_since_epoch())
                                        .count());
                        auto compacted =
                            lfs::io::project::
                                ProjectWriter::compact(
                                    path,
                                    lfs::io::project::
                                        CompactionOptions{
                                            .compatibility =
                                                {},
                                            .new_file_uuid =
                                                lfs::core::
                                                    generate_uuid_v4(),
                                            .commit_uuid =
                                                lfs::core::
                                                    generate_uuid_v4(),
                                            .snapshot_uuid =
                                                {},
                                            .creation_time_unix_ns =
                                                compact_creation_ns,
                                            .wallclock_unix_ns =
                                                compact_wallclock_ns,
                                        });
                        if (compacted) {
                            LOG_INFO(
                                "Compacted trainer project {} after first publish of this run",
                                lfs::core::path_to_utf8(
                                    path));
                        } else {
                            LOG_WARN(
                                "Failed to compact trainer project {} after first publish of this run: {}",
                                lfs::core::path_to_utf8(
                                    path),
                                lfs::format_for_developer(
                                    compacted.error()));
                        }
                    }

                    {
                        std::lock_guard lock(
                            project_snapshot_mutex_);
                        last_project_snapshot_path_ =
                            path;
                        if (writer_result) {
                            last_project_writer_error_
                                .clear();
                            last_completed_project_request_id_ =
                                std::max(
                                    last_completed_project_request_id_,
                                    request_id);
                            last_completed_was_autosave_ =
                                write_kind ==
                                ProjectSnapshotWriteKind::
                                    Autosave;
                            if (last_completed_was_autosave_) {
                                last_completed_autosave_sequence_ =
                                    autosave_sequence;
                            }
                        } else {
                            last_project_writer_error_ =
                                lfs::format_for_developer(
                                    writer_result
                                        .error());
                            last_failed_project_request_id_ =
                                std::max(
                                    last_failed_project_request_id_,
                                    request_id);
                        }
                    }
                    if (!writer_result) {
                        LOG_ERROR(
                            "Background .licht append failed for {}: {}",
                            lfs::core::path_to_utf8(
                                path),
                            lfs::format_for_developer(
                                writer_result.error()));
                    }
                    project_writer_in_flight_.store(
                        false,
                        std::memory_order_release);
                    project_writer_done_.store(
                        true,
                        std::memory_order_release);
                });
        } catch (const std::exception& error) {
            project_writer_in_flight_.store(
                false, std::memory_order_release);
            project_writer_done_.store(
                true, std::memory_order_release);
            const std::lock_guard lock(
                project_snapshot_mutex_);
            fail_project_request_locked(
                request_id, error.what());
            LOG_ERROR(
                "Failed to start .licht project writer: {}",
                error.what());
        }
    }

    void Trainer::observe_training_step_duration(
        const int iteration,
        const double elapsed_ms,
        const bool topology_changed) {
        if (!(elapsed_ms >= 0.0) ||
            !std::isfinite(elapsed_ms)) {
            return;
        }
        TrainingStepRegressionMetrics before;
        TrainingStepRegressionMetrics after;
        {
            std::lock_guard lock(
                project_snapshot_mutex_);
            before =
                project_step_regression_.metrics();
            project_step_regression_.observe(
                iteration, elapsed_ms,
                topology_changed);
            after =
                project_step_regression_.metrics();
        }
        if (!before.gate_evaluated &&
            after.gate_evaluated) {
            LOG_INFO(
                "Training snapshot post-resume step gate: "
                "pre=[{},{}] pre_n={} pre_mean={:.3f}ms "
                "post=[{},{}] post_n={} post_mean={:.3f}ms "
                "regression={:.3f}% {}",
                after.pre_snapshot
                    .first_iteration,
                after.pre_snapshot
                    .last_iteration,
                after.pre_snapshot
                    .sample_count,
                after.pre_snapshot.mean_ms,
                after.post_resume
                    .first_iteration,
                after.post_resume
                    .last_iteration,
                after.post_resume
                    .sample_count,
                after.post_resume.mean_ms,
                after.regression_percent,
                after.within_gate ? "PASS"
                                  : "FAIL");
        }
    }

    void Trainer::handle_control_requests(int iter, std::stop_token stop_token) {
        apply_pending_params_at_safe_point();

        // Check stop token first
        if (stop_token.stop_requested()) {
            stop_requested_ = true;
            return;
        }

        // Handle pause/resume
        if (pause_requested_.load() && !is_paused_.load()) {
            is_paused_ = true;
            if (progress_) {
                progress_->pause();
            }
            // B3: the previous step is complete; release the production loss arena.
            photometric_loss_.arena().reset();
            resize_rasterizer_arena_at_boundary("B3 pause", true);
            LOG_INFO("Training paused at iteration {}", iter);
            LOG_DEBUG("Click 'Resume Training' to continue.");
        } else if (!pause_requested_.load() && is_paused_.load()) {
            is_paused_ = false;
            if (progress_) {
                progress_->resume(
                    iter,
                    current_loss_.load(),
                    static_cast<int>(strategy_->get_model().size()),
                    get_progress_phase(iter));
            }
            LOG_INFO("Training resumed at iteration {}", iter);
        }

        // Handle stop request - this permanently stops training
        if (stop_requested_.load()) {
            // B3: no new forward work will consume these views.
            photometric_loss_.arena().reset();
            LOG_INFO("Stopping training permanently at iteration {}...", iter);
        }
    }

    inline float inv_weight_piecewise(int step, int max_steps) {
        // Phases by fraction of training
        const float phase = std::max(0.f, std::min(1.f, step / float(std::max(1, max_steps))));

        const float limit_hi = 1.0f / 4.0f;  // start limit
        const float limit_mid = 2.0f / 4.0f; // middle limit
        const float limit_lo = 3.0f / 4.0f;  // final limit

        const float weight_hi = 1.0f;  // start weight
        const float weight_mid = 0.5f; // middle weight
        const float weight_lo = 0.0f;  // final weight

        if (phase < limit_hi) {
            return weight_hi; // hold until bypasses the start limit
        } else if (phase < limit_mid) {
            const float t = (phase - limit_hi) / (limit_mid - limit_hi);
            return weight_hi + (weight_mid - weight_hi) * t; // decay to mid value
        } else {
            const float t = (phase - limit_mid) / (limit_lo - limit_mid);
            return weight_mid + (weight_lo - weight_mid) * t; // decay to final value
        }
    }

    namespace {
        constexpr float TWO_PI = static_cast<float>(M_PI * 2.0);
        constexpr float PHASE_OFFSET_G = TWO_PI / 3.0f;
        constexpr float PHASE_OFFSET_B = TWO_PI * 2.0f / 3.0f;
        constexpr float CLAMP_EPS = 1e-4f;
        constexpr int BG_PERIOD_R = 37;
        constexpr int BG_PERIOD_G = 41;
        constexpr int BG_PERIOD_B = 43;
    } // anonymous namespace

    lfs::core::Tensor& Trainer::background_for_step(int iter) {
        if (!params_.optimization.bg_modulation) {
            return background_;
        }

        const float w = inv_weight_piecewise(iter, get_total_iterations());
        if (w <= 0.0f) {
            return background_;
        }

        // Sine-based RGB with prime periods for color diversity
        const float pr = TWO_PI * static_cast<float>(iter % BG_PERIOD_R) / BG_PERIOD_R;
        const float pg = TWO_PI * static_cast<float>(iter % BG_PERIOD_G) / BG_PERIOD_G;
        const float pb = TWO_PI * static_cast<float>(iter % BG_PERIOD_B) / BG_PERIOD_B;

        const float result[3] = {
            std::clamp(0.5f * (1.0f + std::sin(pr)) * w, CLAMP_EPS, 1.0f - CLAMP_EPS),
            std::clamp(0.5f * (1.0f + std::sin(pg + PHASE_OFFSET_G)) * w, CLAMP_EPS, 1.0f - CLAMP_EPS),
            std::clamp(0.5f * (1.0f + std::sin(pb + PHASE_OFFSET_B)) * w, CLAMP_EPS, 1.0f - CLAMP_EPS)};

        if (bg_mix_buffer_.is_empty()) {
            bg_mix_buffer_ = lfs::core::Tensor::empty({3}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);
        }

        LFS_CUDA_TRY(
            cudaMemcpyAsync(bg_mix_buffer_.ptr<float>(), result, sizeof(result),
                            cudaMemcpyHostToDevice, bg_mix_buffer_.stream()),
            bg_mix_buffer_.stream(), "background modulation: upload mix color");
        return bg_mix_buffer_;
    }

    void Trainer::clearBackgroundImageCache() {
        bg_image_cache_.clear();
        bg_image_cache_bytes_ = 0;
        bg_image_cache_clock_ = 0;
    }

    lfs::core::Tensor Trainer::get_background_image_for_camera(int width, int height) {
        // Return empty tensor if no background image is loaded
        if (!bg_image_base_.is_valid() || bg_image_base_.is_empty()) {
            return lfs::core::Tensor();
        }

        // Check cache first - key is (height << 32) | width
        const uint64_t cache_key = (static_cast<uint64_t>(height) << 32) | static_cast<uint64_t>(width);
        auto it = bg_image_cache_.find(cache_key);
        if (it != bg_image_cache_.end()) {
            it->second.last_used = ++bg_image_cache_clock_;
            return it->second.tensor;
        }

        // Resize background image to match camera dimensions
        const int src_h = static_cast<int>(bg_image_base_.shape()[1]);
        const int src_w = static_cast<int>(bg_image_base_.shape()[2]);
        const int channels = static_cast<int>(bg_image_base_.shape()[0]);

        // If dimensions match, use the original
        if (src_w == width && src_h == height) {
            return bg_image_base_;
        }

        // Create resized tensor
        auto resized = lfs::core::Tensor::empty(
            {static_cast<size_t>(channels), static_cast<size_t>(height), static_cast<size_t>(width)},
            lfs::core::Device::CUDA,
            lfs::core::DataType::Float32);

        // Use bilinear resize kernel
        kernels::launch_bilinear_resize_chw(
            bg_image_base_.ptr<float>(),
            resized.ptr<float>(),
            channels,
            src_h, src_w,
            height, width,
            resized.stream());

        // Cache only if this physical bucket can fit under the hard byte ceiling.
        // Returned Tensor copies retain storage safely if an older entry is evicted.
        const size_t allocation_bytes = lfs::core::SizeBucketedPool::get_bucket_size(resized.bytes());
        if (allocation_bytes <= BG_IMAGE_CACHE_BUDGET_BYTES) {
            while (!bg_image_cache_.empty() &&
                   bg_image_cache_bytes_ > BG_IMAGE_CACHE_BUDGET_BYTES - allocation_bytes) {
                const auto lru = std::min_element(
                    bg_image_cache_.begin(), bg_image_cache_.end(), [](const auto& left, const auto& right) {
                        return left.second.last_used < right.second.last_used;
                    });
                bg_image_cache_bytes_ -= std::min(bg_image_cache_bytes_, lru->second.allocation_bytes);
                bg_image_cache_.erase(lru);
            }

            bg_image_cache_.emplace(
                cache_key,
                BackgroundImageCacheEntry{
                    .tensor = resized,
                    .allocation_bytes = allocation_bytes,
                    .last_used = ++bg_image_cache_clock_});
            bg_image_cache_bytes_ += allocation_bytes;
            LFS_ASSERT_MSG(
                bg_image_cache_bytes_ <= BG_IMAGE_CACHE_BUDGET_BYTES,
                "Background image cache exceeded its physical byte budget");
        }
        LOG_DEBUG("Background image resized: {}x{} -> {}x{}", src_w, src_h, width, height);

        return resized;
    }

    lfs::core::Tensor Trainer::get_random_background_for_camera(int width, int height, int iteration) {
        const size_t required_size = 3 * static_cast<size_t>(height) * static_cast<size_t>(width);

        if (!random_bg_buffer_.is_valid() || random_bg_buffer_.numel() != required_size) {
            random_bg_buffer_ = lfs::core::Tensor::empty(
                {3, static_cast<size_t>(height), static_cast<size_t>(width)},
                lfs::core::Device::CUDA,
                lfs::core::DataType::Float32);
        }

        kernels::launch_random_background(
            random_bg_buffer_.ptr<float>(),
            height, width,
            static_cast<uint64_t>(iteration),
            random_bg_buffer_.stream());

        return random_bg_buffer_;
    }

    Trainer::RetryDecision Trainer::classify_forward_retry(
        const lfs::Error& forward_error,
        MutationStamp /*stamp*/,
        const unsigned attempts) noexcept {
        if (attempts >= 2) {
            return RetryDecision::DoNotRetry;
        }
        if (forward_error.code() != lfs::ErrorCode::ResourceExhausted) {
            return RetryDecision::DoNotRetry;
        }
        return RetryDecision::RetryForwardOnce;
    }

    lfs::Status Trainer::recover_forward_oom(const lfs::Error& cause) {
        const auto synchronize = [this] {
            return recovery_sync_for_testing_ ? recovery_sync_for_testing_()
                                              : cudaDeviceSynchronize();
        };
        const auto cuda_recovery_error = [&cause](const cudaError_t status,
                                                  const std::string_view operation) {
            return lfs::make_error(lfs::ErrorInit{
                                       .code = lfs::ErrorCode::Internal,
                                       .domain = lfs::ErrorDomain::CUDA,
                                       .user_message = "A CUDA error was detected while recovering from OOM.",
                                       .detail = std::format("{}: {}", operation, cudaGetErrorString(status)),
                                       .detection = LFS_SOURCE_SITE_CURRENT(),
                                       .native = lfs::NativeError{
                                           lfs::ErrorDomain::CUDA,
                                           static_cast<std::int64_t>(status),
                                           cudaGetErrorName(status),
                                       },
                                   })
                .with_suppressed(cause);
        };

        const cudaError_t sync_status = synchronize();
        if (sync_status != cudaSuccess) {
            return lfs::Status::failure(cuda_recovery_error(
                sync_status, "cudaDeviceSynchronize during OOM recovery"));
        }
        static_cast<void>(cudaGetLastError());

        if (auto harvested = harvestLossReadbacks(true, false); !harvested) {
            auto typed = lfs::from_legacy_expected<void>(
                std::move(harvested),
                lfs::LegacyErrorContext{
                    .code = lfs::ErrorCode::Internal,
                    .domain = lfs::ErrorDomain::Training,
                    .operation = "harvest loss readbacks during OOM recovery",
                    .source = LFS_SOURCE_SITE_CURRENT(),
                });
            return lfs::Status::failure(
                std::move(typed).error().with_suppressed(cause));
        }

        lfs::core::GlobalArenaManager::instance().get_arena().full_reset();
        if (auto loader = getActiveImageLoader())
            loader->reclaim_idle_decoded_frames();
        lfs::core::Tensor::trim_memory_pool();

        const cudaError_t final_status = synchronize();
        if (final_status != cudaSuccess) {
            return lfs::Status::failure(cuda_recovery_error(
                final_status, "cudaDeviceSynchronize after OOM recovery reset"));
        }
        static_cast<void>(cudaGetLastError());
        return {};
    }

    namespace {
        // Profiling hooks for perf_campaign/profile.sh via CLI `--profile-window=START:STOP`.
        // cudaProfilerStart/Stop at [START, STOP); NVTX per-iter ranges while the window
        // is active (nvtxRange* are no-ops when NVTX is compiled out / unused by nsys).
        struct NvtxIterationGuard {
            bool active = false;
            ~NvtxIterationGuard() {
                if (active)
                    nvtxRangePop();
            }
        };
    } // namespace

    lfs::Result<Trainer::StepDisposition> Trainer::train_step(
        int iter,
        lfs::core::Camera* cam,
        lfs::core::Tensor gt_image,
        std::stop_token stop_token) {
        StepPhase current_phase = StepPhase::Forward;
        bool persistent_commit = false;
        const int prof_start = params_.optimization.profile_start_iter;
        const int prof_stop = params_.optimization.profile_stop_iter;
        const bool profile_window_active =
            prof_start >= 0 && prof_stop > prof_start && iter >= prof_start && iter < prof_stop;
        if (iter == prof_start)
            cudaProfilerStart();
        if (iter == prof_stop)
            cudaProfilerStop();
        NvtxIterationGuard nvtx_iter_guard;
        if (profile_window_active) {
            char range_name[32];
            std::snprintf(range_name, sizeof(range_name), "train_step:%d", iter);
            nvtxRangePushA(range_name);
            nvtx_iter_guard.active = true;
        }
        auto result = [&]() -> lfs::Result<StepDisposition> {
            try {
                LFS_VRAM_SCOPE("train.step");
                LOG_VRAM_DIFF("train.step");
                if (PerfBenchCollector::enabled()) {
                    PerfBenchCollector::instance().on_step_begin(iter);
                }
                PerfBenchCollector::phase_mark(PerfBenchCollector::PhaseBoundary::StepBegin, iter);
                if (live_vram_profiler_enabled()) {
                    auto& profiler = lfs::diagnostics::VramProfiler::instance();
                    profiler.beginIteration(iter);
                    profiler.sampleCudaMemory();
                }

                if (params_.optimization.gut) {
                    if (cam->camera_model_type() == core::CameraModelType::ORTHO) {
                        return lfs::make_error(lfs::ErrorInit{
                            .code = lfs::ErrorCode::InvalidArgument,
                            .domain = lfs::ErrorDomain::Training,
                            .user_message = "Training on cameras with ortho model is not supported yet.",
                            .detection = LFS_SOURCE_SITE_CURRENT(),
                        });
                    }
                } else if (!params_.optimization.undistort || !cam->is_undistort_prepared()) {
                    if (cam->radial_distortion().numel() != 0 ||
                        cam->tangential_distortion().numel() != 0) {
                        return lfs::make_error(lfs::ErrorInit{
                            .code = lfs::ErrorCode::InvalidArgument,
                            .domain = lfs::ErrorDomain::Training,
                            .user_message = "Distorted images detected. Use --gut or --undistort to train on cameras with distortion.",
                            .detection = LFS_SOURCE_SITE_CURRENT(),
                        });
                    }
                    if (cam->camera_model_type() != core::CameraModelType::PINHOLE) {
                        return lfs::make_error(lfs::ErrorInit{
                            .code = lfs::ErrorCode::InvalidArgument,
                            .domain = lfs::ErrorDomain::Training,
                            .user_message = "Use --gut or --undistort to train on cameras with non-pinhole model.",
                            .detection = LFS_SOURCE_SITE_CURRENT(),
                        });
                    }
                }

                current_iteration_ = iter;

                // Check control requests at the beginning
                handle_control_requests(iter, stop_token);

                if (on_iteration_start_)
                    on_iteration_start_();
                // Manager/Python callbacks publish parameter updates via setParams().
                // Install them before forward or optimizer work observes params_.
                apply_pending_params_at_safe_point();

                // Gate this step's in-place parameter writes behind in-flight model
                // reads (viewer packs, metric renders) — GPU-side waits, ~free once
                // the reads have retired.
                waitForModelReaders();

                // Python hook: iteration start (safe, pre-forward)
                {
                    lfs::training::HookContext ctx{
                        .iteration = iter,
                        .loss = current_loss_.load(),
                        .num_gaussians = strategy_ ? strategy_->get_model().size() : 0,
                        .is_refining = strategy_ ? strategy_->is_refining(iter) : false,
                        .trainer = this};
                    lfs::training::CommandCenter::instance().set_phase(lfs::training::TrainingPhase::IterationStart);
                    lfs::training::CommandCenter::instance().update_snapshot(
                        ctx, get_total_iterations(), is_paused_.load(), is_running_.load(), stop_requested_.load(),
                        lfs::training::TrainingPhase::IterationStart);
                    lfs::training::ControlBoundary::instance().notify(lfs::training::ControlHook::IterationStart, ctx);
                    auto view = lfs::training::CommandCenter::instance().snapshot();
                    lfs::training::CommandCenter::instance().drain_enqueued(view);
                }

                // Training step entering forward/backward/optimizer region (commands blocked)
                lfs::training::CommandCenter::instance().set_phase(lfs::training::TrainingPhase::Forward);

                // If stop requested, return Stop
                if (stop_requested_.load() || stop_token.stop_requested()) {
                    return StepDisposition::Stop;
                }

                // If paused, wait. Drain requested project snapshots at the
                // last completed iteration — the current `iter` has not run.
                while (is_paused_.load() && !stop_requested_.load() && !stop_token.stop_requested()) {
                    consume_requested_project_snapshot(
                        project_snapshot_iteration());
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    handle_control_requests(iter, stop_token);
                }

                // Check stop again after potential pause
                if (stop_requested_.load() || stop_token.stop_requested()) {
                    return StepDisposition::Stop;
                }

                lfs::core::Tensor* bg_ptr = nullptr;
                {
                    nvtxRangePush("background_for_step");
                    LFS_VRAM_SCOPE("train.background");
                    LOG_VRAM_DIFF("train.background");
                    bg_ptr = &background_for_step(iter);
                    nvtxRangePop();
                }
                lfs::core::Tensor& bg = *bg_ptr;

                lfs::core::Tensor bg_image;
                if (params_.optimization.bg_mode == lfs::core::param::BackgroundMode::Image) {
                    LFS_VRAM_SCOPE("train.background_image");
                    LOG_VRAM_DIFF("train.background_image");
                    bg_image = get_background_image_for_camera(cam->image_width(), cam->image_height());
                } else if (params_.optimization.bg_mode == lfs::core::param::BackgroundMode::Random) {
                    LFS_VRAM_SCOPE("train.random_background");
                    LOG_VRAM_DIFF("train.random_background");
                    bg_image = get_random_background_for_camera(cam->image_width(), cam->image_height(), iter);
                }

                const bool fastgs_path = !params_.optimization.gut;

                if (!loss_accumulator_.is_valid()) {
                    loss_accumulator_ = core::Tensor::zeros({1}, core::Device::CUDA);
                } else {
                    loss_accumulator_.zero_();
                }
                if (live_vram_profiler_enabled() && strategy_) {
                    record_splat_vram_breakdown(strategy_->get_model());
                    record_optimizer_vram_breakdown(strategy_->get_optimizer());
                    publish_training_state_ledger(strategy_->get_model(),
                                                  &strategy_->get_optimizer());
                    record_vram_tensor("train.persistent", "loss_accumulator", loss_accumulator_);
                    record_vram_tensor("train.persistent", "pipelined_mask", pipelined_mask_);
                    record_vram_tensor("train.persistent", "pipelined_depth", pipelined_depth_);
                    record_vram_tensor("train.persistent", "pipelined_normal", pipelined_normal_);
                    record_vram_tensor("train.persistent", "background", background_);
                    record_vram_tensor("train.persistent", "background_mix_buffer", bg_mix_buffer_);
                    record_vram_tensor("train.persistent", "background_image_base", bg_image_base_);
                    record_vram_current("train.persistent", "background_image_cache", bg_image_cache_bytes_);
                    record_pipeline_vram_breakdown(getActiveImageLoader());
                }
                auto& loss_tensor_gpu = loss_accumulator_;
                RenderOutput r_output;
                r_output.camera = cam;
                r_output.target_image = gt_image;
                int tiles_processed = 0;
                const bool in_sparsification = get_active_sparsify_steps() > 0 &&
                                               iter > get_sparsity_boundary_iteration();

                // Determine controller phase before render (does not depend on render results)
                const bool known_ppisp_camera = ppisp_ && ppisp_->is_known_camera(cam->camera_id());
                const int ppisp_cam_idx = known_ppisp_camera ? ppisp_->camera_index(cam->camera_id()) : -1;
                const int ppisp_activation_step = params_.optimization.resolved_ppisp_controller_activation_step(get_total_iterations());
                const bool ppisp_frozen = is_ppisp_frozen();
                const bool in_controller_phase = ppisp_controller_pool_ && known_ppisp_camera &&
                                                 params_.optimization.ppisp_use_controller &&
                                                 !ppisp_frozen &&
                                                 params_.optimization.ppisp_freeze_gaussians_on_distill &&
                                                 iter >= ppisp_activation_step &&
                                                 ppisp_cam_idx >= 0 &&
                                                 ppisp_cam_idx < ppisp_controller_pool_->num_cameras();
                const bool freeze_gaussians_this_iter = ppisp_controller_pool_ &&
                                                        params_.optimization.ppisp_use_controller &&
                                                        params_.optimization.ppisp_freeze_gaussians_on_distill &&
                                                        iter >= ppisp_activation_step;
                const bool use_pixel_error_densification =
                    (params_.optimization.strategy == "mcmc") ||
                    (params_.optimization.strategy == "igs+") ||
                    (core::param::is_mrnf_strategy(params_.optimization.strategy) &&
                     params_.optimization.use_error_map);
                const bool use_ssim_error = use_pixel_error_densification;
                DensificationType densification_type = DensificationType::None;
                if (params_.optimization.strategy == "mcmc")
                    densification_type = DensificationType::MCMC;
                else if (core::param::is_mrnf_strategy(params_.optimization.strategy))
                    densification_type = DensificationType::MRNF;
                const bool update_gaussians_this_iter = !freeze_gaussians_this_iter;
                const bool run_fastgs_gaussian_backward =
                    fastgs_path &&
                    update_gaussians_this_iter;

                bool fastgs_strategy_hooks_at_start = false;
                const bool refining_this_step =
                    strategy_ && strategy_->is_refining(iter);
                const bool morton_due = morton_reorder_due(iter);
                if (fastgs_path && !in_sparsification) {
                    current_phase = StepPhase::RefinementCommit;
                    LFS_VRAM_SCOPE("train.strategy.fastgs_pre_step");
                    LOG_VRAM_DIFF("train.strategy.fastgs_pre_step");
                    strategy_->pre_step(iter, r_output);

                    // Only exclude the render thread when this step actually mutates model
                    // topology (grow/prune → tensor reallocation, moving pointers). In-place
                    // parameter updates are ordered against the viewer's reads by the
                    // CUDA↔Vulkan interop semaphore, so render_mutex_ is redundant there.
                    // Holding it every iteration deadlocks at startup: the render holds a
                    // read-lock across its synchronous GPU readback while waiting for the
                    // first step's output, which this write-lock — taken before that step —
                    // blocks. See trainer.cpp step() lock below; both must be gated.
                    std::unique_lock<std::shared_mutex> lock(render_mutex_, std::defer_lock);
                    const bool refining = refining_this_step;
                    const bool need_exclusive = refining || morton_due;
                    if (need_exclusive) {
                        lock.lock();
                    }
                    // One-lock complete: exclusive render_mutex_ already bars new
                    // shared live-model readers. Also hold combined_model_mutex so
                    // any rebuild that started before exclusive cannot re-enter
                    // (or finish late) across float-workspace swap + trim.
                    std::unique_lock<std::mutex> combined_model_lock;
                    if (need_exclusive && scene_) {
                        combined_model_lock = scene_->acquireCombinedModelExclusive();
                    }
                    // Drain in-flight reader events immediately before post_backward's
                    // in-place writes — not only at the loop top — so the trainer stream
                    // is ordered after any read that began mid-step, collapsing the
                    // reader↔writer overlap to a sub-microsecond CPU window. The
                    // exclusive lock (when refining) additionally bars new readers.
                    waitForModelReaders();
                    // Single-buffer q16 commit into the live exportable block: drop
                    // Keep the Vulkan import absent throughout the densification window.
                    struct DensifyBarrierGuard {
                        Trainer* self = nullptr;
                        bool held = false;
                        explicit DensifyBarrierGuard(Trainer* t, bool need) : self(t) {
                            if (need && self) {
                                const auto result = self->beginExportableDensifyBarrier();
                                if (result == ExportableDensifyBarrierBegin::Failed) {
                                    throw std::runtime_error(
                                        "Failed to begin exportable densify barrier");
                                }
                                held = result == ExportableDensifyBarrierBegin::Acquired;
                            }
                        }
                        ~DensifyBarrierGuard() {
                            if (held && self) {
                                (void)self->endExportableDensifyBarrier();
                            }
                        }
                        DensifyBarrierGuard(const DensifyBarrierGuard&) = delete;
                        DensifyBarrierGuard& operator=(const DensifyBarrierGuard&) = delete;
                    } densify_barrier(this, need_exclusive);
                    // Nested LiveModelMutationGuard inside ensure/commit must no-op.
                    struct RefiningMutationMark {
                        explicit RefiningMutationMark(bool on) {
                            if (on) {
                                mark_live_model_mutation_lock_held(true);
                            }
                            on_ = on;
                        }
                        ~RefiningMutationMark() {
                            if (on_) {
                                mark_live_model_mutation_lock_held(false);
                            }
                        }
                        bool on_ = false;
                    } refining_mutation_mark(need_exclusive);
                    auto& model = strategy_->get_model();
                    const size_t model_size_before = static_cast<size_t>(model.size());
                    strategy_->post_backward(iter, r_output);
                    maybe_morton_reorder(iter);
                    install_cropbox_step_damping(model, strategy_->get_optimizer());
                    fastgs_strategy_hooks_at_start = true;

                    if (sparsity_optimizer_ &&
                        sparsity_optimizer_->is_initialized() &&
                        static_cast<size_t>(model.size()) != model_size_before) {
                        LOG_WARN("Sparsity: resetting ADMM state after topology change at iter {} ({} -> {})",
                                 iter, model_size_before, model.size());
                        sparsity_optimizer_->reset();
                    }
                    if (static_cast<size_t>(model.size()) != model_size_before) {
                        // Defer topology fan-out until the densify barrier stabilizes
                        // exportable q16 so cache rebuilds cannot race the next forward.
                        if (!refining) {
                            syncTrainingSceneTopology(scene_, model);
                        }
                    }
                    if (auto result = ensureModelTensorAllocatorStorage(model, "fastgs strategy post_backward"); !result) {
                        return lfs::from_legacy_expected<StepDisposition>(
                                   std::unexpected(result.error()),
                                   lfs::LegacyErrorContext{
                                       .code = lfs::ErrorCode::Internal,
                                       .domain = lfs::ErrorDomain::Training,
                                       .operation = "ensure model tensor allocator storage (fastgs post_backward)",
                                       .source = LFS_SOURCE_SITE_CURRENT(),
                                   })
                            .error();
                    }
                    // Readers can re-acquire the shared lock the moment the
                    // exclusive lock drops — re-mark consistency before that.
                    // densify_barrier dtor re-imports Vulkan before lock release.
                    if (lock.owns_lock()) {
                        current_phase = StepPhase::Publish;
                        recordParamsReady();
                    }
                    if (refining) {
                        resize_rasterizer_arena_at_boundary("B1 refine", false);
                    }
                    // densify_barrier dtor runs when leaving this block — topology
                    // fan-out only after q16 commit is stable (see defer above).
                    ++mutation_epoch_;
                    persistent_commit = true;
                }
                if (fastgs_path && refining_this_step && !in_sparsification) {
                    // Post-barrier topology publish (deferred from inside densify).
                    auto& model_after = strategy_->get_model();
                    syncTrainingSceneTopology(scene_, model_after);
                }
                if (fastgs_path && in_sparsification) {
                    install_cropbox_step_damping(strategy_->get_model(), strategy_->get_optimizer());
                }

                const bool normal_supervision_started = normal_supervision_active(iter);

                FastGSFusedExtraGradients fused_extra_gradients;
                lfs::core::Tensor fused_scale_reg_loss_gpu;
                lfs::core::Tensor fused_opacity_reg_loss_gpu;
                lfs::core::Tensor sparsity_loss_gpu;
                if (fastgs_path) {
                    LFS_VRAM_SCOPE("train.regularizers.fastgs_forward_only");
                    LOG_VRAM_DIFF("train.regularizers.fastgs_forward_only");
                    auto& model = strategy_->get_model();
                    if (run_fastgs_gaussian_backward) {
                        fused_extra_gradients.scale_reg_weight = params_.optimization.scale_reg;
                        fused_extra_gradients.opacity_reg_weight = params_.optimization.opacity_reg;
                        if (normal_supervision_started) {
                            fused_extra_gradients.flatten_reg_weight = params_.optimization.normal_flatten_weight;
                        }
                        // Fused backward owns regularization accumulation, avoiding
                        // separate full-N kernels and their temporary allocations.
                        if (params_.optimization.scale_reg > 0.0f) {
                            if (!fused_scale_reg_loss_.is_valid()) {
                                fused_scale_reg_loss_ = lfs::core::Tensor::zeros(
                                    {1}, lfs::core::Device::CUDA);
                            }
                            fused_scale_reg_loss_.zero_();
                            fused_extra_gradients.scale_reg_loss_out =
                                fused_scale_reg_loss_.ptr<float>();
                            fused_scale_reg_loss_gpu = fused_scale_reg_loss_;
                        }
                        if (params_.optimization.opacity_reg > 0.0f) {
                            if (!fused_opacity_reg_loss_.is_valid()) {
                                fused_opacity_reg_loss_ = lfs::core::Tensor::zeros(
                                    {1}, lfs::core::Device::CUDA);
                            }
                            fused_opacity_reg_loss_.zero_();
                            fused_extra_gradients.opacity_reg_loss_out =
                                fused_opacity_reg_loss_.ptr<float>();
                            fused_opacity_reg_loss_gpu = fused_opacity_reg_loss_;
                        }
                    } else {
                        // Freeze / non-backward FastGS iterations: keep legacy loss-only
                        // path so reported loss stays valid without a fused backward.
                        if (params_.optimization.scale_reg > 0.0f) {
                            auto scale_loss_result =
                                lfs::training::losses::ScaleRegularization::forward_loss_only(
                                    model.scaling_raw(),
                                    {.weight = params_.optimization.scale_reg});
                            if (!scale_loss_result) {
                                return lfs::from_legacy_expected<StepDisposition>(
                                           std::unexpected(scale_loss_result.error()),
                                           lfs::LegacyErrorContext{
                                               .code = lfs::ErrorCode::Internal,
                                               .domain = lfs::ErrorDomain::Training,
                                               .operation = "scale regularization forward (fastgs)",
                                               .source = LFS_SOURCE_SITE_CURRENT(),
                                           })
                                    .error();
                            }
                            fused_scale_reg_loss_gpu = *scale_loss_result;
                        }
                        if (params_.optimization.opacity_reg > 0.0f) {
                            auto opacity_loss_result =
                                lfs::training::losses::OpacityRegularization::forward_loss_only(
                                    model.opacity_raw(),
                                    {.weight = params_.optimization.opacity_reg});
                            if (!opacity_loss_result) {
                                return lfs::from_legacy_expected<StepDisposition>(
                                           std::unexpected(opacity_loss_result.error()),
                                           lfs::LegacyErrorContext{
                                               .code = lfs::ErrorCode::Internal,
                                               .domain = lfs::ErrorDomain::Training,
                                               .operation = "opacity regularization forward (fastgs)",
                                               .source = LFS_SOURCE_SITE_CURRENT(),
                                           })
                                    .error();
                            }
                            fused_opacity_reg_loss_gpu = *opacity_loss_result;
                        }
                    }
                    if (run_fastgs_gaussian_backward &&
                        sparsity_optimizer_ && sparsity_optimizer_->should_apply_loss(iter)) {
                        auto sparsity_result = compute_sparsity_loss_forward(iter, model);
                        if (!sparsity_result) {
                            return lfs::from_legacy_expected<StepDisposition>(
                                       std::unexpected(sparsity_result.error()),
                                       lfs::LegacyErrorContext{
                                           .code = lfs::ErrorCode::Internal,
                                           .domain = lfs::ErrorDomain::Training,
                                           .operation = "sparsity loss forward (fastgs)",
                                           .source = LFS_SOURCE_SITE_CURRENT(),
                                       })
                                .error();
                        }
                        auto& [loss_tensor, ctx] = *sparsity_result;
                        sparsity_loss_gpu = std::move(loss_tensor);
                        fused_extra_gradients.sparsity_opa_sigmoid = ctx.opa_sigmoid_ptr;
                        fused_extra_gradients.sparsity_z = ctx.z_ptr;
                        fused_extra_gradients.sparsity_u = ctx.u_ptr;
                        fused_extra_gradients.sparsity_n = static_cast<int>(ctx.n);
                        fused_extra_gradients.sparsity_rho = ctx.rho;
                        fused_extra_gradients.sparsity_grad_loss = 1.0f;
                    }
                }

                {
                    nvtxRangePush("rasterize");

                    lfs::core::Tensor gt_tile = gt_image;
                    lfs::core::Tensor bg_tile;
                    if (bg_image.is_valid() && !bg_image.is_empty()) {
                        bg_tile = bg_image;
                    }

                    // Render the tile
                    nvtxRangePush("rasterize_forward");
                    PerfBenchCollector::phase_mark(PerfBenchCollector::PhaseBoundary::FwdBegin, iter);

                    // Storage for render output (used by both paths)
                    RenderOutput output;
                    std::optional<FastRasterizeContext> fast_ctx;
                    std::optional<GsplatRasterizeContext> gsplat_ctx;

                    {
                        LFS_VRAM_SCOPE("train.rasterize_forward");
                        LOG_VRAM_DIFF("train.rasterize_forward");
                        current_phase = StepPhase::Forward;
                        if (params_.optimization.gut) {
                            auto rasterize_result = gsplat_rasterize_forward(
                                *cam, strategy_->get_model(), bg,
                                0, 0, 0, 0,
                                1.0f, false, GsplatRenderMode::RGB, true, bg_tile);

                            if (!rasterize_result) {
                                nvtxRangePop(); // rasterize_forward
                                nvtxRangePop(); // tile
                                return lfs::from_legacy_expected<StepDisposition>(
                                           std::unexpected(rasterize_result.error()),
                                           lfs::LegacyErrorContext{
                                               .code = lfs::ErrorCode::Internal,
                                               .domain = lfs::ErrorDomain::Rendering,
                                               .operation = "gsplat_rasterize_forward",
                                               .source = LFS_SOURCE_SITE_CURRENT(),
                                           })
                                    .error();
                            }

                            output = std::move(rasterize_result->first);
                            gsplat_ctx.emplace(std::move(rasterize_result->second));
                        } else {
                            const bool render_normal =
                                normal_supervision_started &&
                                ((params_.optimization.normal_loss_weight > 0.0f &&
                                  normal_prior_usable_ &&
                                  cam->has_normal()) ||
                                 params_.optimization.normal_consistency_weight > 0.0f);
                            const MutationStamp forward_stamp{
                                static_cast<std::uint64_t>(iter), mutation_epoch_,
                                StepPhase::Forward, fastgs_strategy_hooks_at_start};
                            unsigned forward_attempts = 0;
                            for (;;) {
                                ++forward_attempts;
                                auto rasterize_result = fast_rasterize_forward(
                                    *cam, strategy_->get_model(), bg,
                                    0, 0, 0, 0,
                                    params_.optimization.mip_filter, bg_tile,
                                    render_normal);
                                if (rasterize_result) {
                                    output = std::move(rasterize_result->first);
                                    fast_ctx.emplace(std::move(rasterize_result->second));
                                    break;
                                }

                                lfs::Error forward_error = std::move(rasterize_result.error());
                                // pathological 32-bit instance overflow must not
                                // kill the run — skip the step (loud error) and continue.
                                if (forward_error.code() == lfs::ErrorCode::FailedPrecondition &&
                                    forward_error.detail().find(
                                        "instance count exceeds 32-bit") != std::string::npos) {
                                    LOG_ERROR(
                                        "Skipping iteration {} after FastGS instance overflow "
                                        "(bad frame, training continues): {}",
                                        iter,
                                        forward_error.detail());
                                    nvtxRangePop();
                                    nvtxRangePop();
                                    return iter < get_total_iterations() &&
                                                   !stop_requested_.load() &&
                                                   !stop_token.stop_requested()
                                               ? StepDisposition::Continue
                                               : StepDisposition::Stop;
                                }
                                const RetryDecision decision = classify_forward_retry(
                                    forward_error, forward_stamp, forward_attempts);
                                if (decision == RetryDecision::DoNotRetry) {
                                    nvtxRangePop();
                                    nvtxRangePop();
                                    return std::move(forward_error);
                                }

                                LOG_ERROR(
                                    "FastGS forward exhausted a resource (attempt {}): {}",
                                    forward_attempts, forward_error.detail());
                                if (lfs::Status recovery = recover_forward_oom(forward_error);
                                    !recovery) {
                                    nvtxRangePop();
                                    nvtxRangePop();
                                    // recover_forward_oom is the sole attachment point for
                                    // the initiating OOM; do not suppress it a second time.
                                    return std::move(recovery).error();
                                }
                            }

                            if (fast_ctx->forward_ctx.n_instances == 0) {
                                fast_ctx->release_forward_context();
                                nvtxRangePop();
                                nvtxRangePop();
                                LOG_DEBUG("Skipping iteration {} - no visible primitives", iter);
                                return iter < get_total_iterations() && !stop_requested_.load() && !stop_token.stop_requested()
                                           ? StepDisposition::Continue
                                           : StepDisposition::Stop;
                            }
                        }
                    }

                    r_output = output; // Save last tile for densification
                    r_output.camera = cam;
                    r_output.target_image = gt_image;
                    nvtxRangePop();

                    bool tile_context_cleaned = false;
                    auto cleanup_tile_context = [&]() {
                        if (tile_context_cleaned) {
                            return;
                        }
                        tile_context_cleaned = true;

                        if (fast_ctx) {
                            fast_ctx->release_forward_context();
                            fast_ctx.reset();
                        } else if (gsplat_ctx) {
                            // Isect/flatten ids are TLS high-water (not owned by
                            // the context) — do not cudaFree; only end the arena
                            // frame that held the other forward scratch.
                            auto& arena = lfs::core::GlobalArenaManager::instance().get_arena();
                            gsplat_ctx->isect_ids_ptr = nullptr;
                            gsplat_ctx->flatten_ids_ptr = nullptr;
                            arena.end_frame(gsplat_ctx->frame_id, lfs::core::getCurrentCUDAStream());
                            gsplat_ctx.reset();
                        }
                    };

                    lfs::core::Tensor roi_weight;
                    const auto cropbox_loss_geometry =
                        scene_ ? resolve_training_cropbox_loss_geom(
                                     *scene_, params_.optimization.cropbox_loss_weight)
                               : std::nullopt;
                    if (cropbox_loss_geometry) {
                        LFS_ASSERT_MSG(
                            output.width > 0 && output.height > 0,
                            "ROI loss weighting requires positive render dimensions");
                        const lfs::core::TensorShape roi_shape{
                            static_cast<size_t>(output.height),
                            static_cast<size_t>(output.width)};
                        const cudaStream_t roi_stream = output.image.stream();
                        if (!roi_weight_map_.is_valid() ||
                            roi_weight_map_.shape() != roi_shape) {
                            roi_weight_map_ = lfs::core::Tensor::empty(
                                roi_shape,
                                lfs::core::Device::CUDA,
                                lfs::core::DataType::Float32);
                        }
                        if (roi_weight_map_.stream() != roi_stream) {
                            roi_weight_map_.set_stream(roi_stream);
                        }

                        cam->world_view_transform().sync_to_stream(roi_stream);
                        cam->cam_position().sync_to_stream(roi_stream);
                        const auto [fx, fy, cx, cy] = cam->get_intrinsics();
                        lfs::training::kernels::launch_roi_weight_map(
                            cam->world_view_transform_ptr(),
                            cam->cam_position_ptr(),
                            fx,
                            fy,
                            cx,
                            cy,
                            output.width,
                            output.height,
                            cropbox_loss_geometry->world_to_cropbox,
                            cropbox_loss_geometry->min,
                            cropbox_loss_geometry->max,
                            cropbox_loss_geometry->inverse,
                            params_.optimization.cropbox_loss_weight,
                            roi_weight_map_.ptr<float>(),
                            roi_stream);
                        roi_weight = roi_weight_map_;
                        roi_weight.sync_to_stream(lfs::core::getCurrentCUDAStream());
                    }

                    current_phase = StepPhase::Loss;
                    if (in_controller_phase) {
                        // Controller phase: forward through ISP with controller params, photometric loss,
                        // backward only through controller (base params frozen)
                        nvtxRangePush("controller_phase");
                        LFS_VRAM_SCOPE("train.controller_phase");
                        LOG_VRAM_DIFF("train.controller_phase");
                        auto tile_context_guard = makeScopeGuard(cleanup_tile_context);

                        // The predict() below fills shared pool buffers (and pred aliases one)
                        // that backward() consumes; hold the transaction lock so a concurrent
                        // viewport/export prediction cannot corrupt the pair.
                        std::lock_guard<std::mutex> controller_lock(ppisp_controller_pool_->predict_mutex());

                        lfs::core::Tensor corrected_image = output.image;
                        if (bilateral_grid_ && params_.optimization.use_bilateral_grid) {
                            LFS_VRAM_SCOPE("train.bilateral_grid.forward");
                            LOG_VRAM_DIFF("train.bilateral_grid.forward");
                            corrected_image = bilateral_grid_->apply(output.image, cam->uid());
                        }
                        auto ppisp_input = corrected_image;

                        lfs::core::Tensor pred;
                        {
                            LFS_VRAM_SCOPE("train.ppisp_controller.forward");
                            LOG_VRAM_DIFF("train.ppisp_controller.forward");
                            pred = ppisp_controller_pool_->predict(ppisp_cam_idx, corrected_image.unsqueeze(0), 1.0f);
                            corrected_image = ppisp_->apply_with_controller_params(corrected_image, pred, ppisp_cam_idx);
                        }
                        const lfs::core::Tensor raw_loss_input = output.image;

                        // Photometric loss
                        nvtxRangePush("compute_photometric_loss");
                        PerfBenchCollector::phase_mark(PerfBenchCollector::PhaseBoundary::LossBegin, iter);
                        lfs::core::Tensor tile_loss;
                        lfs::core::Tensor tile_grad;

                        {
                            LFS_VRAM_SCOPE("train.photometric_loss");
                            LOG_VRAM_DIFF("train.photometric_loss");
                            const bool use_mask = params_.optimization.mask_mode != lfs::core::param::MaskMode::None &&
                                                  (cam->has_mask() || (params_.optimization.use_alpha_as_mask && cam->has_alpha()));
                            if (use_mask || roi_weight.is_valid()) {
                                lfs::core::Tensor mask;
                                if (use_mask) {
                                    if (pipelined_mask_.is_valid() && pipelined_mask_.numel() > 0) {
                                        mask = pipelined_mask_;
                                    } else {
                                        mask = cam->load_and_get_mask(
                                            params_.dataset.resize_factor,
                                            params_.dataset.max_width,
                                            params_.optimization.invert_masks,
                                            params_.optimization.mask_threshold,
                                            params_.optimization.mask_mode != lfs::core::param::MaskMode::SegmentAndIgnore);
                                    }
                                }

                                lfs::core::Tensor mask_tile = mask;

                                auto result = compute_photometric_loss_with_mask(
                                    corrected_image, gt_tile, mask_tile, roi_weight, output.alpha,
                                    params_.optimization, raw_loss_input);
                                if (!result) {
                                    nvtxRangePop();
                                    nvtxRangePop();
                                    nvtxRangePop();
                                    return lfs::from_legacy_expected<StepDisposition>(
                                               std::unexpected(result.error()),
                                               lfs::LegacyErrorContext{
                                                   .code = lfs::ErrorCode::Internal,
                                                   .domain = lfs::ErrorDomain::Training,
                                                   .operation = "photometric loss with mask (controller phase)",
                                                   .source = LFS_SOURCE_SITE_CURRENT(),
                                               })
                                        .error();
                                }
                                tile_loss = result->loss;
                                tile_grad = result->grad_corrected;
                            } else {
                                auto result = compute_photometric_loss_with_gradient(
                                    corrected_image, gt_tile, params_.optimization, raw_loss_input);
                                if (!result) {
                                    nvtxRangePop();
                                    nvtxRangePop();
                                    nvtxRangePop();
                                    return lfs::from_legacy_expected<StepDisposition>(
                                               std::unexpected(result.error()),
                                               lfs::LegacyErrorContext{
                                                   .code = lfs::ErrorCode::Internal,
                                                   .domain = lfs::ErrorDomain::Training,
                                                   .operation = "photometric loss with gradient (controller phase)",
                                                   .source = LFS_SOURCE_SITE_CURRENT(),
                                               })
                                        .error();
                                }
                                tile_loss = result->loss;
                                tile_grad = result->grad_corrected;
                            }
                        }

                        loss_tensor_gpu = loss_tensor_gpu + tile_loss;
                        tiles_processed++;
                        nvtxRangePop(); // compute_photometric_loss
                        if (live_vram_profiler_enabled()) {
                            record_vram_tensor("train.losses", "controller.tile_loss", tile_loss);
                            record_vram_tensor("train.losses", "controller.tile_grad", tile_grad);
                            record_vram_tensor("train.appearance", "ppisp_controller.prediction", pred);
                            {
                                const auto loss_ws =
                                    photometric_workspace_bytes(photometric_loss_);
                                record_vram_current("train.losses", "loss_workspace_arena",
                                                    loss_ws.allocated);
                                auto& profiler = lfs::diagnostics::VramProfiler::instance();
                                profiler.setGauge("vram.audit.loss_workspace.required_bytes",
                                                  static_cast<double>(loss_ws.required));
                                profiler.setGauge("vram.audit.loss_workspace.allocated_bytes",
                                                  static_cast<double>(loss_ws.allocated));
                                if (PerfBenchCollector::enabled()) {
                                    PerfBenchCollector::instance().set_loss_workspace_bytes(
                                        loss_ws.required, loss_ws.allocated);
                                }
                            }
                        }

                        // ISP backward for controller params
                        {
                            LFS_VRAM_SCOPE("train.ppisp_controller.backward");
                            LOG_VRAM_DIFF("train.ppisp_controller.backward");
                            auto ctrl_grad = ppisp_->backward_with_controller_params(ppisp_input, tile_grad, pred, ppisp_cam_idx);
                            ppisp_controller_pool_->backward(ppisp_cam_idx, ctrl_grad);
                        }

                        nvtxRangePop(); // controller_phase
                    } else {
                        // Normal phase: full forward + backward through all components
                        auto tile_context_guard = makeScopeGuard(cleanup_tile_context);

                        lfs::core::Tensor corrected_image = output.image;
                        if (bilateral_grid_ && params_.optimization.use_bilateral_grid) {
                            nvtxRangePush("bilateral_grid_forward");
                            LFS_VRAM_SCOPE("train.bilateral_grid.forward");
                            LOG_VRAM_DIFF("train.bilateral_grid.forward");
                            corrected_image = bilateral_grid_->apply(output.image, cam->uid());
                            nvtxRangePop();
                        }

                        lfs::core::Tensor ppisp_input;
                        if (ppisp_ && params_.optimization.use_ppisp) {
                            nvtxRangePush("ppisp_forward");
                            LFS_VRAM_SCOPE("train.ppisp.forward");
                            LOG_VRAM_DIFF("train.ppisp.forward");
                            ppisp_input = corrected_image;
                            corrected_image = ppisp_->apply(ppisp_input, cam->camera_id(), cam->uid());
                            nvtxRangePop();
                        }
                        // For decoupled D-SSIM, the active appearance model provides the corrected image
                        // for the L1/luminance terms, while contrast/structure still use the raw render.
                        const lfs::core::Tensor raw_loss_input =
                            ((bilateral_grid_ && params_.optimization.use_bilateral_grid) ||
                             (ppisp_ && params_.optimization.use_ppisp))
                                ? output.image
                                : lfs::core::Tensor{};

                        // Final tonemapping: clamp to [0, 1] for loss computation.
                        // skip when PPISP is active — CRF already clamps.
                        if (!(ppisp_ && params_.optimization.use_ppisp)) {
                            corrected_image.clamp_(0.0f, 1.0f);
                        }

                        nvtxRangePush("compute_photometric_loss");
                        PerfBenchCollector::phase_mark(PerfBenchCollector::PhaseBoundary::LossBegin, iter);
                        lfs::core::Tensor tile_loss;
                        lfs::core::Tensor tile_grad;
                        lfs::core::Tensor tile_grad_raw;
                        lfs::core::Tensor tile_grad_alpha;
                        lfs::core::Tensor tile_grad_depth;
                        lfs::core::Tensor tile_grad_normal;
                        lfs::core::Tensor tile_error_map;
                        lfs::core::Tensor mask_tile;
                        bool depth_grad_buffers_active = false;
                        const auto roi_weight_ptr_on_stream =
                            [&](const cudaStream_t stream) -> const float* {
                            if (!roi_weight.is_valid()) {
                                return nullptr;
                            }
                            roi_weight.sync_to_stream(stream);
                            return roi_weight.ptr<float>();
                        };

                        const auto ensure_depth_grad_buffers =
                            [&](const lfs::core::Tensor& rendered_depth,
                                const cudaStream_t stream,
                                const bool clear_for_accumulation) {
                                if (!depth_loss_grad_.is_valid() ||
                                    depth_loss_grad_.shape() != rendered_depth.shape()) {
                                    depth_loss_grad_ = lfs::core::Tensor::empty(
                                        rendered_depth.shape(), lfs::core::Device::CUDA);
                                }
                                depth_loss_grad_.set_stream(stream);
                                if (!depth_loss_grad_alpha_.is_valid() ||
                                    depth_loss_grad_alpha_.shape() != rendered_depth.shape()) {
                                    depth_loss_grad_alpha_ = lfs::core::Tensor::empty(
                                        rendered_depth.shape(), lfs::core::Device::CUDA);
                                }
                                depth_loss_grad_alpha_.set_stream(stream);
                                if (clear_for_accumulation) {
                                    depth_loss_grad_.zero_();
                                    depth_loss_grad_alpha_.zero_();
                                }
                                depth_grad_buffers_active = true;
                            };

                        const auto merge_depth_grad_buffers = [&]() {
                            if (!depth_grad_buffers_active) {
                                return;
                            }
                            if (tile_grad_alpha.is_valid() && tile_grad_alpha.numel() > 0) {
                                auto existing_alpha_grad = tile_grad_alpha;
                                if (existing_alpha_grad.ndim() == 3 && existing_alpha_grad.shape()[0] == 1) {
                                    existing_alpha_grad = existing_alpha_grad.squeeze(0);
                                }
                                depth_loss_grad_alpha_.add_(existing_alpha_grad);
                            }
                            tile_grad_depth = depth_loss_grad_;
                            tile_grad_alpha = depth_loss_grad_alpha_;
                        };

                        // 1) Compute photometric loss (populates ssim_map in workspace)
                        const bool use_mask = params_.optimization.mask_mode != lfs::core::param::MaskMode::None &&
                                              (cam->has_mask() || (params_.optimization.use_alpha_as_mask && cam->has_alpha()));
                        const bool used_masked_fused =
                            (roi_weight.is_valid() ||
                             (use_mask &&
                              (params_.optimization.mask_mode == lfs::core::param::MaskMode::Segment ||
                               params_.optimization.mask_mode == lfs::core::param::MaskMode::Ignore ||
                               params_.optimization.mask_mode == lfs::core::param::MaskMode::SegmentAndIgnore))) &&
                            params_.optimization.lambda_dssim > 0.0f;
                        {
                            LFS_VRAM_SCOPE("train.photometric_loss");
                            LOG_VRAM_DIFF("train.photometric_loss");
                            if (use_mask || roi_weight.is_valid()) {
                                lfs::core::Tensor mask;
                                if (use_mask) {
                                    if (pipelined_mask_.is_valid() && pipelined_mask_.numel() > 0) {
                                        mask = pipelined_mask_;
                                    } else {
                                        mask = cam->load_and_get_mask(
                                            params_.dataset.resize_factor,
                                            params_.dataset.max_width,
                                            params_.optimization.invert_masks,
                                            params_.optimization.mask_threshold,
                                            params_.optimization.mask_mode != lfs::core::param::MaskMode::SegmentAndIgnore);
                                    }
                                }

                                mask_tile = mask;

                                auto result = compute_photometric_loss_with_mask(
                                    corrected_image, gt_tile, mask_tile, roi_weight, output.alpha,
                                    params_.optimization, raw_loss_input);
                                if (!result) {
                                    nvtxRangePop();
                                    nvtxRangePop();
                                    return lfs::from_legacy_expected<StepDisposition>(
                                               std::unexpected(result.error()),
                                               lfs::LegacyErrorContext{
                                                   .code = lfs::ErrorCode::Internal,
                                                   .domain = lfs::ErrorDomain::Training,
                                                   .operation = "photometric loss with mask",
                                                   .source = LFS_SOURCE_SITE_CURRENT(),
                                               })
                                        .error();
                                }
                                tile_loss = result->loss;
                                tile_grad = result->grad_corrected;
                                tile_grad_raw = result->grad_raw;
                                tile_grad_alpha = result->grad_alpha;
                            } else {
                                auto result = compute_photometric_loss_with_gradient(
                                    corrected_image, gt_tile, params_.optimization, raw_loss_input);
                                if (!result) {
                                    nvtxRangePop();
                                    nvtxRangePop();
                                    return lfs::from_legacy_expected<StepDisposition>(
                                               std::unexpected(result.error()),
                                               lfs::LegacyErrorContext{
                                                   .code = lfs::ErrorCode::Internal,
                                                   .domain = lfs::ErrorDomain::Training,
                                                   .operation = "photometric loss with gradient",
                                                   .source = LFS_SOURCE_SITE_CURRENT(),
                                               })
                                        .error();
                                }
                                tile_loss = result->loss;
                                tile_grad = result->grad_corrected;
                                tile_grad_raw = result->grad_raw;
                            }
                        }

                        if (run_fastgs_gaussian_backward &&
                            params_.optimization.use_depth_loss &&
                            params_.optimization.depth_loss_weight > 0.0f &&
                            output.depth.is_valid() &&
                            output.depth.numel() > 0 &&
                            output.alpha.is_valid() &&
                            output.alpha.numel() > 0) {
                            LFS_VRAM_SCOPE("train.depth_loss");
                            LOG_VRAM_DIFF("train.depth_loss");

                            lfs::core::Tensor target_depth;
                            if (pipelined_depth_.is_valid() && pipelined_depth_.numel() > 0) {
                                target_depth = pipelined_depth_;
                            } else if (cam->has_depth()) {
                                target_depth = cam->load_and_get_depth(
                                    params_.dataset.resize_factor,
                                    params_.dataset.max_width);
                            }

                            if (target_depth.is_valid() && target_depth.numel() > 0) {
                                if (target_depth.ndim() == 3 && target_depth.shape()[0] == 1) {
                                    target_depth = target_depth.squeeze(0);
                                }
                                if (target_depth.device() != lfs::core::Device::CUDA) {
                                    target_depth = target_depth.cuda();
                                }
                                if (!target_depth.is_contiguous()) {
                                    target_depth = target_depth.contiguous();
                                }

                                lfs::core::Tensor rendered_depth = output.depth;
                                if (rendered_depth.ndim() == 3 && rendered_depth.shape()[0] == 1) {
                                    rendered_depth = rendered_depth.squeeze(0);
                                }
                                if (!rendered_depth.is_contiguous()) {
                                    rendered_depth = rendered_depth.contiguous();
                                }

                                lfs::core::Tensor rendered_alpha = output.alpha;
                                if (rendered_alpha.ndim() == 3 && rendered_alpha.shape()[0] == 1) {
                                    rendered_alpha = rendered_alpha.squeeze(0);
                                }
                                if (!rendered_alpha.is_contiguous()) {
                                    rendered_alpha = rendered_alpha.contiguous();
                                }

                                const cudaStream_t depth_stream = rendered_depth.stream();

                                if (target_depth.ndim() == 2 && rendered_depth.ndim() == 2 &&
                                    (target_depth.shape()[0] != rendered_depth.shape()[0] ||
                                     target_depth.shape()[1] != rendered_depth.shape()[1])) {
                                    const int render_h = static_cast<int>(rendered_depth.shape()[0]);
                                    const int render_w = static_cast<int>(rendered_depth.shape()[1]);
                                    target_depth = lfs::core::lanczos_resize_grayscale(
                                        target_depth, render_h, render_w, 2, depth_stream);
                                }

                                const bool depth_shape_matches =
                                    target_depth.is_valid() &&
                                    target_depth.ndim() == 2 &&
                                    rendered_depth.ndim() == 2 &&
                                    rendered_alpha.ndim() == 2 &&
                                    target_depth.shape()[0] == rendered_depth.shape()[0] &&
                                    target_depth.shape()[1] == rendered_depth.shape()[1] &&
                                    target_depth.shape()[0] == rendered_alpha.shape()[0] &&
                                    target_depth.shape()[1] == rendered_alpha.shape()[1];

                                if (depth_shape_matches) {
                                    const size_t num_depth_pixels = rendered_depth.numel();
                                    const size_t depth_partials =
                                        lfs::training::kernels::depth_loss_partial_count(num_depth_pixels);
                                    const int total_iterations =
                                        std::max(1, params_.optimization.resolved_total_iterations());
                                    const float depth_progress =
                                        std::min(static_cast<float>(iter) / static_cast<float>(total_iterations), 1.0f);
                                    const float depth_weight_now =
                                        params_.optimization.depth_loss_weight *
                                        std::pow(kDepthLossFinalScale, depth_progress);

                                    if (!depth_loss_scalar_.is_valid()) {
                                        depth_loss_scalar_ = lfs::core::Tensor::zeros({1}, lfs::core::Device::CUDA);
                                    }
                                    depth_loss_scalar_.set_stream(depth_stream);
                                    if (!depth_loss_partials_.is_valid() ||
                                        depth_loss_partials_.shape()[0] != depth_partials) {
                                        depth_loss_partials_ = lfs::core::Tensor::empty({depth_partials}, lfs::core::Device::CUDA);
                                    }
                                    depth_loss_partials_.set_stream(depth_stream);

                                    const lfs::training::kernels::DepthAnchor* depth_anchor = nullptr;
                                    if (const auto anchor_it = depth_anchors_.find(cam->uid());
                                        anchor_it != depth_anchors_.end()) {
                                        depth_anchor = &anchor_it->second;
                                    }
                                    if (depth_anchor_fit_attempted_ && depth_anchor != nullptr && depth_anchor->valid) {
                                        ensure_depth_grad_buffers(rendered_depth, depth_stream, false);
                                        const int depth_width = static_cast<int>(rendered_depth.shape()[1]);
                                        const int depth_height = static_cast<int>(rendered_depth.shape()[0]);
                                        const float depth_prior_qstep = cam->depth_prior_quantization_step();
                                        const float* const depth_pixel_weight =
                                            roi_weight_ptr_on_stream(depth_stream);
                                        lfs::core::pin_operands({&rendered_depth, &rendered_alpha, &target_depth});
                                        lfs::training::kernels::launch_depth_loss(
                                            rendered_depth.ptr<float>(),
                                            rendered_alpha.ptr<float>(),
                                            target_depth.ptr<float>(),
                                            depth_loss_grad_.ptr<float>(),
                                            depth_loss_grad_alpha_.ptr<float>(),
                                            depth_loss_scalar_.ptr<float>(),
                                            depth_loss_partials_.ptr<float>(),
                                            depth_width,
                                            depth_height,
                                            depth_weight_now,
                                            kDepthLossGradientTermWeight,
                                            depth_prior_qstep,
                                            depth_anchor,
                                            depth_stream,
                                            depth_pixel_weight);

                                        tile_loss = tile_loss + depth_loss_scalar_;
                                    }
                                } else {
                                    LOG_WARN("Skipping depth loss for '{}': rendered depth shape and target depth shape differ",
                                             cam->image_name());
                                }
                            }
                        }

                        if (run_fastgs_gaussian_backward &&
                            normal_supervision_started &&
                            params_.optimization.normal_loss_weight > 0.0f &&
                            output.normal.is_valid() &&
                            output.normal.numel() > 0 &&
                            output.alpha.is_valid() &&
                            output.alpha.numel() > 0) {
                            LFS_VRAM_SCOPE("train.normal_loss");
                            LOG_VRAM_DIFF("train.normal_loss");

                            lfs::core::Tensor target_normal;
                            if (pipelined_normal_.is_valid() && pipelined_normal_.numel() > 0) {
                                target_normal = pipelined_normal_;
                            } else if (cam->has_normal()) {
                                target_normal = cam->load_and_get_normal(
                                    params_.dataset.resize_factor,
                                    params_.dataset.max_width,
                                    lfs::core::Camera::NormalPriorDecode{
                                        normal_prior_srgb_,
                                        normal_prior_flip_yz_,
                                        normal_prior_world_space_,
                                        normal_prior_world_rotation_});
                                // Per-camera GPU caching of [3,H,W] priors accumulates
                                // unbounded VRAM on large datasets; the fallback path
                                // re-decodes instead (the pipelined loader is the fast path).
                                cam->release_normal_cache();
                            }

                            if (target_normal.is_valid() && target_normal.numel() > 0) {
                                lfs::core::Tensor rendered_normal = output.normal;
                                if (!rendered_normal.is_contiguous()) {
                                    rendered_normal = rendered_normal.contiguous();
                                }

                                lfs::core::Tensor rendered_alpha = output.alpha;
                                if (rendered_alpha.ndim() == 3 && rendered_alpha.shape()[0] == 1) {
                                    rendered_alpha = rendered_alpha.squeeze(0);
                                }
                                if (!rendered_alpha.is_contiguous()) {
                                    rendered_alpha = rendered_alpha.contiguous();
                                }

                                const cudaStream_t normal_stream = rendered_normal.stream();
                                const int render_h = static_cast<int>(rendered_normal.shape()[1]);
                                const int render_w = static_cast<int>(rendered_normal.shape()[2]);

                                if (target_normal.ndim() == 3 &&
                                    (target_normal.shape()[1] != rendered_normal.shape()[1] ||
                                     target_normal.shape()[2] != rendered_normal.shape()[2])) {
                                    target_normal = lfs::core::lanczos_resize_float_chw(
                                        target_normal, render_h, render_w, 2, normal_stream);
                                }

                                const bool normal_shape_matches =
                                    target_normal.is_valid() &&
                                    target_normal.ndim() == 3 &&
                                    target_normal.shape()[0] == 3 &&
                                    target_normal.shape()[1] == rendered_normal.shape()[1] &&
                                    target_normal.shape()[2] == rendered_normal.shape()[2];

                                if (normal_shape_matches) {
                                    if (!target_normal.is_contiguous()) {
                                        target_normal = target_normal.contiguous();
                                    }
                                    const size_t num_normal_pixels =
                                        static_cast<size_t>(render_h) * static_cast<size_t>(render_w);
                                    const size_t normal_partials =
                                        lfs::training::kernels::normal_loss_partial_count(num_normal_pixels);

                                    if (!normal_loss_scalar_.is_valid()) {
                                        normal_loss_scalar_ = lfs::core::Tensor::zeros({1}, lfs::core::Device::CUDA);
                                    }
                                    normal_loss_scalar_.set_stream(normal_stream);
                                    if (!normal_loss_grad_.is_valid() ||
                                        normal_loss_grad_.shape() != rendered_normal.shape()) {
                                        normal_loss_grad_ = lfs::core::Tensor::empty(rendered_normal.shape(), lfs::core::Device::CUDA);
                                    }
                                    normal_loss_grad_.set_stream(normal_stream);
                                    if (!normal_loss_partials_.is_valid() ||
                                        normal_loss_partials_.shape()[0] != normal_partials) {
                                        normal_loss_partials_ = lfs::core::Tensor::empty({normal_partials}, lfs::core::Device::CUDA);
                                    }
                                    normal_loss_partials_.set_stream(normal_stream);

                                    const float* const normal_pixel_weight =
                                        roi_weight_ptr_on_stream(normal_stream);
                                    lfs::core::pin_operands({&rendered_normal, &rendered_alpha, &target_normal});
                                    lfs::training::kernels::launch_normal_loss(
                                        rendered_normal.ptr<float>(),
                                        rendered_alpha.ptr<float>(),
                                        target_normal.ptr<float>(),
                                        normal_loss_grad_.ptr<float>(),
                                        normal_loss_scalar_.ptr<float>(),
                                        normal_loss_partials_.ptr<float>(),
                                        render_w,
                                        render_h,
                                        params_.optimization.normal_loss_weight,
                                        normal_stream,
                                        normal_pixel_weight);

                                    if (output.depth.is_valid() &&
                                        output.depth.numel() > 0 &&
                                        cam->camera_width() > 0 && cam->camera_height() > 0 &&
                                        cam->focal_x() > 0.0f && cam->focal_y() > 0.0f) {
                                        lfs::core::Tensor rendered_depth = output.depth;
                                        if (rendered_depth.ndim() == 3 && rendered_depth.shape()[0] == 1) {
                                            rendered_depth = rendered_depth.squeeze(0);
                                        }
                                        if (!rendered_depth.is_contiguous()) {
                                            rendered_depth = rendered_depth.contiguous();
                                        }
                                        const bool prior_depth_shapes_match =
                                            rendered_depth.ndim() == 2 &&
                                            rendered_alpha.ndim() == 2 &&
                                            rendered_depth.shape()[0] == rendered_normal.shape()[1] &&
                                            rendered_depth.shape()[1] == rendered_normal.shape()[2] &&
                                            rendered_alpha.shape()[0] == rendered_normal.shape()[1] &&
                                            rendered_alpha.shape()[1] == rendered_normal.shape()[2];
                                        if (prior_depth_shapes_match) {
                                            ensure_depth_grad_buffers(
                                                rendered_depth, normal_stream, !depth_grad_buffers_active);

                                            const size_t prior_depth_partials =
                                                lfs::training::kernels::normal_consistency_partial_count(num_normal_pixels);
                                            if (!normal_prior_depth_scalar_.is_valid()) {
                                                normal_prior_depth_scalar_ =
                                                    lfs::core::Tensor::zeros({1}, lfs::core::Device::CUDA);
                                            }
                                            normal_prior_depth_scalar_.set_stream(normal_stream);
                                            if (!normal_consistency_partials_.is_valid() ||
                                                normal_consistency_partials_.shape()[0] != prior_depth_partials) {
                                                normal_consistency_partials_ =
                                                    lfs::core::Tensor::empty({prior_depth_partials}, lfs::core::Device::CUDA);
                                            }
                                            normal_consistency_partials_.set_stream(normal_stream);

                                            const float fx = cam->focal_x() * static_cast<float>(render_w) /
                                                             static_cast<float>(cam->camera_width());
                                            const float fy = cam->focal_y() * static_cast<float>(render_h) /
                                                             static_cast<float>(cam->camera_height());
                                            const float cx = cam->center_x() * static_cast<float>(render_w) /
                                                             static_cast<float>(cam->camera_width());
                                            const float cy = cam->center_y() * static_cast<float>(render_h) /
                                                             static_cast<float>(cam->camera_height());

                                            lfs::core::pin_operands({&target_normal, &rendered_depth, &rendered_alpha});
                                            lfs::training::kernels::launch_normal_prior_depth_loss(
                                                target_normal.ptr<float>(),
                                                rendered_depth.ptr<float>(),
                                                rendered_alpha.ptr<float>(),
                                                depth_loss_grad_.ptr<float>(),
                                                depth_loss_grad_alpha_.ptr<float>(),
                                                normal_prior_depth_scalar_.ptr<float>(),
                                                normal_consistency_partials_.ptr<float>(),
                                                render_w,
                                                render_h,
                                                fx,
                                                fy,
                                                cx,
                                                cy,
                                                params_.optimization.normal_loss_weight,
                                                normal_stream,
                                                normal_pixel_weight);
                                            tile_loss = tile_loss + normal_prior_depth_scalar_;
                                        }
                                    }

                                    tile_grad_normal = normal_loss_grad_;
                                    tile_loss = tile_loss + normal_loss_scalar_;
                                } else {
                                    LOG_WARN("Skipping normal loss for '{}': rendered normal shape and target normal shape differ",
                                             cam->image_name());
                                }
                            }
                        }

                        if (run_fastgs_gaussian_backward &&
                            normal_supervision_started &&
                            params_.optimization.normal_consistency_weight > 0.0f &&
                            output.normal.is_valid() &&
                            output.normal.numel() > 0 &&
                            output.depth.is_valid() &&
                            output.depth.numel() > 0 &&
                            output.alpha.is_valid() &&
                            output.alpha.numel() > 0) {
                            LFS_VRAM_SCOPE("train.normal_consistency_loss");
                            LOG_VRAM_DIFF("train.normal_consistency_loss");

                            lfs::core::Tensor rendered_normal = output.normal;
                            if (!rendered_normal.is_contiguous()) {
                                rendered_normal = rendered_normal.contiguous();
                            }
                            lfs::core::Tensor rendered_depth = output.depth;
                            if (rendered_depth.ndim() == 3 && rendered_depth.shape()[0] == 1) {
                                rendered_depth = rendered_depth.squeeze(0);
                            }
                            if (!rendered_depth.is_contiguous()) {
                                rendered_depth = rendered_depth.contiguous();
                            }
                            lfs::core::Tensor rendered_alpha = output.alpha;
                            if (rendered_alpha.ndim() == 3 && rendered_alpha.shape()[0] == 1) {
                                rendered_alpha = rendered_alpha.squeeze(0);
                            }
                            if (!rendered_alpha.is_contiguous()) {
                                rendered_alpha = rendered_alpha.contiguous();
                            }

                            const int render_h = static_cast<int>(rendered_normal.shape()[1]);
                            const int render_w = static_cast<int>(rendered_normal.shape()[2]);
                            const bool consistency_shapes_match =
                                rendered_normal.ndim() == 3 &&
                                rendered_normal.shape()[0] == 3 &&
                                rendered_depth.ndim() == 2 &&
                                rendered_alpha.ndim() == 2 &&
                                rendered_depth.shape()[0] == rendered_normal.shape()[1] &&
                                rendered_depth.shape()[1] == rendered_normal.shape()[2] &&
                                rendered_alpha.shape()[0] == rendered_normal.shape()[1] &&
                                rendered_alpha.shape()[1] == rendered_normal.shape()[2] &&
                                cam->camera_width() > 0 && cam->camera_height() > 0 &&
                                cam->focal_x() > 0.0f && cam->focal_y() > 0.0f;

                            if (consistency_shapes_match) {
                                const cudaStream_t consistency_stream = rendered_normal.stream();

                                if (!tile_grad_normal.is_valid()) {
                                    if (!normal_loss_grad_.is_valid() ||
                                        normal_loss_grad_.shape() != rendered_normal.shape()) {
                                        normal_loss_grad_ = lfs::core::Tensor::empty(rendered_normal.shape(), lfs::core::Device::CUDA);
                                    }
                                    normal_loss_grad_.set_stream(consistency_stream);
                                    normal_loss_grad_.zero_();
                                    tile_grad_normal = normal_loss_grad_;
                                }
                                ensure_depth_grad_buffers(
                                    rendered_depth, consistency_stream, !depth_grad_buffers_active);

                                const size_t num_consistency_pixels =
                                    static_cast<size_t>(render_h) * static_cast<size_t>(render_w);
                                const size_t consistency_partials =
                                    lfs::training::kernels::normal_consistency_partial_count(num_consistency_pixels);
                                if (!normal_consistency_scalar_.is_valid()) {
                                    normal_consistency_scalar_ = lfs::core::Tensor::zeros({1}, lfs::core::Device::CUDA);
                                }
                                normal_consistency_scalar_.set_stream(consistency_stream);
                                if (!normal_consistency_partials_.is_valid() ||
                                    normal_consistency_partials_.shape()[0] != consistency_partials) {
                                    normal_consistency_partials_ = lfs::core::Tensor::empty({consistency_partials}, lfs::core::Device::CUDA);
                                }
                                normal_consistency_partials_.set_stream(consistency_stream);

                                const float fx = cam->focal_x() * static_cast<float>(render_w) /
                                                 static_cast<float>(cam->camera_width());
                                const float fy = cam->focal_y() * static_cast<float>(render_h) /
                                                 static_cast<float>(cam->camera_height());
                                const float cx = cam->center_x() * static_cast<float>(render_w) /
                                                 static_cast<float>(cam->camera_width());
                                const float cy = cam->center_y() * static_cast<float>(render_h) /
                                                 static_cast<float>(cam->camera_height());

                                lfs::core::pin_operands(
                                    {&rendered_normal, &rendered_depth, &rendered_alpha, &tile_grad_normal});
                                const float* const consistency_pixel_weight =
                                    roi_weight_ptr_on_stream(consistency_stream);
                                lfs::training::kernels::launch_normal_consistency_loss(
                                    rendered_normal.ptr<float>(),
                                    rendered_depth.ptr<float>(),
                                    rendered_alpha.ptr<float>(),
                                    tile_grad_normal.ptr<float>(),
                                    depth_loss_grad_.ptr<float>(),
                                    depth_loss_grad_alpha_.ptr<float>(),
                                    normal_consistency_scalar_.ptr<float>(),
                                    normal_consistency_partials_.ptr<float>(),
                                    render_w,
                                    render_h,
                                    fx,
                                    fy,
                                    cx,
                                    cy,
                                    params_.optimization.normal_consistency_weight,
                                    consistency_stream,
                                    consistency_pixel_weight);

                                tile_loss = tile_loss + normal_consistency_scalar_;
                            }
                        }

                        merge_depth_grad_buffers();

                        // 2) Extract error map from workspace's ssim_map
                        if (use_pixel_error_densification) {
                            LFS_VRAM_SCOPE("train.densification_error_map");
                            LOG_VRAM_DIFF("train.densification_error_map");
                            if (use_ssim_error && params_.optimization.lambda_dssim > 0.0f) {
                                lfs::core::Tensor ssim_map;
                                if (used_masked_fused && raw_loss_input.is_valid()) {
                                    ssim_map = photometric_loss_.arena().masked_decoupled().ssim_map;
                                } else if (used_masked_fused) {
                                    ssim_map = photometric_loss_.arena().masked_fused().ssim_map;
                                } else if (raw_loss_input.is_valid()) {
                                    ssim_map = photometric_loss_.arena().decoupled().ssim_map;
                                } else if (params_.optimization.lambda_dssim < 1.0f) {
                                    ssim_map = photometric_loss_.fused_workspace().ssim_map;
                                } else {
                                    ssim_map = photometric_loss_.ssim_workspace().ssim_map;
                                }
                                if (ssim_map.shape()[0] == 1 && ssim_map.shape()[1] == 1 &&
                                    ssim_map.is_contiguous()) {
                                    const size_t H = ssim_map.shape()[2];
                                    const size_t W = ssim_map.shape()[3];
                                    tile_error_map = ssim_map.reshape({static_cast<int>(H), static_cast<int>(W)});
                                    lfs::training::kernels::launch_ssim_to_error_map(ssim_map, tile_error_map);
                                } else {
                                    const size_t H = ssim_map.shape()[2];
                                    const size_t W = ssim_map.shape()[3];
                                    if (!densification_error_map_.is_valid() ||
                                        densification_error_map_.shape()[0] != H ||
                                        densification_error_map_.shape()[1] != W) {
                                        densification_error_map_ = core::Tensor::empty(
                                            {static_cast<size_t>(H), static_cast<size_t>(W)},
                                            core::Device::CUDA);
                                    }
                                    lfs::training::kernels::launch_ssim_to_error_map(ssim_map, densification_error_map_);
                                    tile_error_map = densification_error_map_;
                                }
                            } else if (use_ssim_error) {
                                // lambda_dssim == 0 but error-priority densification still needs SSIM error
                                lfs::core::Tensor pred_chw = corrected_image;
                                lfs::core::Tensor gt_chw = gt_tile;
                                if (pred_chw.ndim() == 3 && pred_chw.shape()[2] == 3 &&
                                    gt_chw.ndim() == 3 && gt_chw.shape()[2] == 3) {
                                    pred_chw = pred_chw.permute({2, 0, 1}).contiguous();
                                    gt_chw = gt_chw.permute({2, 0, 1}).contiguous();
                                }
                                lfs::training::kernels::ssim_error_map_forward(
                                    pred_chw, gt_chw, densification_ssim_workspace_, densification_error_map_);
                                tile_error_map = densification_error_map_;
                            } else {
                                const auto gt_for_error = gt_tile.dtype() == lfs::core::DataType::UInt8
                                                              ? gt_tile.to(lfs::core::DataType::Float32) / 255.0f
                                                              : gt_tile;
                                const lfs::core::Tensor abs_diff = (corrected_image - gt_for_error).abs();
                                if (abs_diff.ndim() == 3 && abs_diff.shape()[0] == 3) {
                                    tile_error_map = abs_diff.mean({0}, false);
                                } else if (abs_diff.ndim() == 3 && abs_diff.shape()[2] == 3) {
                                    tile_error_map = abs_diff.mean({2}, false);
                                } else {
                                    tile_error_map = abs_diff;
                                }
                                tile_error_map = tile_error_map.contiguous();
                            }

                            if (use_mask &&
                                params_.optimization.mask_mode == lfs::core::param::MaskMode::SegmentAndIgnore) {
                                const auto mask_for_error = mask_tile.gt(250).to(lfs::core::DataType::Float32);
                                tile_error_map.mul_(mask_for_error);
                            }

                            if (use_mask &&
                                (params_.optimization.mask_mode == lfs::core::param::MaskMode::Segment ||
                                 params_.optimization.mask_mode == lfs::core::param::MaskMode::Ignore)) {
                                const auto mask_for_error =
                                    (mask_tile.dtype() == lfs::core::DataType::UInt8 ||
                                     mask_tile.dtype() == lfs::core::DataType::Bool)
                                        ? mask_tile.to(lfs::core::DataType::Float32)
                                        : mask_tile;
                                tile_error_map.mul_(mask_for_error);
                            }
                        }

                        if (tile_error_map.is_valid() && core::param::is_mrnf_strategy(params_.optimization.strategy)) {
                            LFS_VRAM_SCOPE("train.densification_error_map");
                            LOG_VRAM_DIFF("train.densification_error_map.normalize");
                            const auto map_mean = tile_error_map.mean();
                            lfs::core::pin_operands({&tile_error_map, &map_mean});
                            lfs::training::kernels::launch_normalize_by_device_scalar(
                                tile_error_map.ptr<float>(), tile_error_map.numel(),
                                map_mean.ptr<float>(), 1e-6f);
                        }

                        if (live_vram_profiler_enabled()) {
                            if (fast_ctx) {
                                record_fastgs_vram_breakdown(*fast_ctx,
                                                             output,
                                                             gt_tile,
                                                             bg_tile,
                                                             tile_error_map,
                                                             run_fastgs_gaussian_backward,
                                                             static_cast<std::size_t>(strategy_->get_model().size()));
                            } else if (gsplat_ctx) {
                                record_gsplat_vram_breakdown(*gsplat_ctx, output, gt_tile, bg_tile, tile_error_map);
                            }
                            record_vram_tensor("train.losses", "tile_loss", tile_loss);
                            record_vram_tensor("train.losses", "tile_grad_corrected", tile_grad);
                            record_vram_tensor("train.losses", "tile_grad_raw", tile_grad_raw);
                            record_vram_tensor("train.losses", "tile_grad_alpha", tile_grad_alpha);
                            record_vram_tensor("train.losses", "densification_error_map.live", tile_error_map);
                            {
                                const auto loss_ws =
                                    photometric_workspace_bytes(photometric_loss_);
                                record_vram_current("train.losses", "loss_workspace_arena",
                                                    loss_ws.allocated);
                                auto& profiler = lfs::diagnostics::VramProfiler::instance();
                                profiler.setGauge("vram.audit.loss_workspace.required_bytes",
                                                  static_cast<double>(loss_ws.required));
                                profiler.setGauge("vram.audit.loss_workspace.allocated_bytes",
                                                  static_cast<double>(loss_ws.allocated));
                                if (PerfBenchCollector::enabled()) {
                                    PerfBenchCollector::instance().set_loss_workspace_bytes(
                                        loss_ws.required, loss_ws.allocated);
                                }
                            }
                            record_vram_current("train.losses", "densification_ssim.workspace",
                                                ssim_map_workspace_bytes(densification_ssim_workspace_));
                            record_vram_tensor("train.losses", "densification_error_map.buffer", densification_error_map_);
                            record_vram_tensor("train.losses", "edge_map_buffer", edge_map_buffer_);
                        }

                        loss_tensor_gpu = loss_tensor_gpu + tile_loss;
                        tiles_processed++;
                        nvtxRangePop();

                        lfs::core::Tensor raster_grad = tile_grad;
                        if (ppisp_ && params_.optimization.use_ppisp) {
                            nvtxRangePush("ppisp_backward");
                            LFS_VRAM_SCOPE("train.ppisp.backward");
                            LOG_VRAM_DIFF("train.ppisp.backward");
                            raster_grad = ppisp_->backward(ppisp_input, raster_grad, cam->camera_id(), cam->uid());
                            if (ppisp_frozen) {
                                ppisp_->zero_grad();
                            }
                            nvtxRangePop();
                        }

                        if (bilateral_grid_ && params_.optimization.use_bilateral_grid) {
                            nvtxRangePush("bilateral_grid_backward");
                            LFS_VRAM_SCOPE("train.bilateral_grid.backward");
                            LOG_VRAM_DIFF("train.bilateral_grid.backward");
                            raster_grad = bilateral_grid_->backward(output.image, raster_grad, cam->uid());
                            nvtxRangePop();
                        }

                        if (tile_grad_raw.is_valid() && tile_grad_raw.numel() > 0) {
                            raster_grad = raster_grad + tile_grad_raw;
                        }

                        current_phase = StepPhase::Backward;
                        nvtxRangePush("rasterize_backward");
                        PerfBenchCollector::phase_mark(PerfBenchCollector::PhaseBoundary::BwdBegin, iter);
                        {
                            LFS_VRAM_SCOPE("train.rasterize_backward");
                            LOG_VRAM_DIFF("train.rasterize_backward");
                            if (gsplat_ctx) {
                                auto grad_alpha = tile_grad_alpha.is_valid()
                                                      ? tile_grad_alpha
                                                      : lfs::core::Tensor::zeros_like(output.alpha);
                                tile_context_guard.release();
                                gsplat_rasterize_backward(*gsplat_ctx, raster_grad, grad_alpha,
                                                          strategy_->get_model(), strategy_->get_optimizer(),
                                                          use_pixel_error_densification ? tile_error_map : lfs::core::Tensor{});
                            } else {
                                tile_context_guard.release();
                                if (run_fastgs_gaussian_backward) {
                                    // Topology-only locking (see post_backward/step above): the
                                    // fused backward updates params in place; the interop
                                    // semaphore orders those against the viewer's reads, so the
                                    // render thread is only excluded on reallocation iterations.
                                    std::unique_lock<std::shared_mutex> model_write_lock(render_mutex_, std::defer_lock);
                                    if (strategy_->is_refining(iter))
                                        model_write_lock.lock();
                                    fast_rasterize_backward(*fast_ctx, raster_grad, strategy_->get_model(),
                                                            strategy_->get_optimizer(), tile_grad_alpha,
                                                            use_pixel_error_densification ? tile_error_map : lfs::core::Tensor{},
                                                            densification_type,
                                                            iter,
                                                            fused_extra_gradients,
                                                            tile_grad_depth,
                                                            tile_grad_normal);
                                    if (model_write_lock.owns_lock()) {
                                        recordParamsReady();
                                    }
                                } else {
                                    cleanup_tile_context();
                                }
                            }
                        }
                        nvtxRangePop();
                    }

                    nvtxRangePop(); // End rasterize
                }

                if (tiles_processed == 0) {
                    LOG_DEBUG("Skipping iteration {} - no visible primitives", iter);
                    return iter < get_total_iterations() && !stop_requested_.load() && !stop_token.stop_requested()
                               ? StepDisposition::Continue
                               : StepDisposition::Stop;
                }

                update_camera_loss_heatmap(*cam, loss_tensor_gpu);
                maybe_publish_camera_loss_heatmap(iter);

                if (in_controller_phase) {
                    current_phase = StepPhase::OptimizerCommit;
                    // Controller phase: only update controller weights
                    nvtxRangePush("controller_optimizer_step");
                    PerfBenchCollector::phase_mark(PerfBenchCollector::PhaseBoundary::OptBegin, iter);
                    LFS_VRAM_SCOPE("train.optimizer.ppisp_controller_step");
                    LOG_VRAM_DIFF("train.optimizer.ppisp_controller_step");
                    ppisp_controller_pool_->optimizer_step(ppisp_cam_idx);
                    ppisp_controller_pool_->zero_grad();
                    ppisp_controller_pool_->scheduler_step(ppisp_cam_idx);
                    ++mutation_epoch_;
                    persistent_commit = true;
                    nvtxRangePop();
                } else {
                    // Default path has no controller_optimizer_step NVTX; mark the
                    // same OptBegin boundary at the start of the optimizer region.
                    PerfBenchCollector::phase_mark(PerfBenchCollector::PhaseBoundary::OptBegin, iter);
                    // Normal phase: regularization losses + optimizer steps for all components

                    if (params_.optimization.scale_reg > 0.0f) {
                        nvtxRangePush("compute_scale_reg_loss");
                        LFS_VRAM_SCOPE("train.regularizers.scale_loss");
                        LOG_VRAM_DIFF("train.regularizers.scale_loss");
                        if (fastgs_path) {
                            loss_tensor_gpu = loss_tensor_gpu + fused_scale_reg_loss_gpu;
                        } else {
                            auto scale_loss_result = compute_scale_reg_loss(strategy_->get_model(), strategy_->get_optimizer(), params_.optimization);
                            if (!scale_loss_result) {
                                return lfs::from_legacy_expected<StepDisposition>(
                                           std::unexpected(scale_loss_result.error()),
                                           lfs::LegacyErrorContext{
                                               .code = lfs::ErrorCode::Internal,
                                               .domain = lfs::ErrorDomain::Training,
                                               .operation = "scale regularization loss",
                                               .source = LFS_SOURCE_SITE_CURRENT(),
                                           })
                                    .error();
                            }
                            loss_tensor_gpu = loss_tensor_gpu + *scale_loss_result;
                        }
                        nvtxRangePop();
                    }

                    if (params_.optimization.opacity_reg > 0.0f) {
                        nvtxRangePush("compute_opacity_reg_loss");
                        LFS_VRAM_SCOPE("train.regularizers.opacity_loss");
                        LOG_VRAM_DIFF("train.regularizers.opacity_loss");
                        if (fastgs_path) {
                            loss_tensor_gpu = loss_tensor_gpu + fused_opacity_reg_loss_gpu;
                        } else {
                            auto opacity_loss_result = compute_opacity_reg_loss(strategy_->get_model(), strategy_->get_optimizer(), params_.optimization);
                            if (!opacity_loss_result) {
                                return lfs::from_legacy_expected<StepDisposition>(
                                           std::unexpected(opacity_loss_result.error()),
                                           lfs::LegacyErrorContext{
                                               .code = lfs::ErrorCode::Internal,
                                               .domain = lfs::ErrorDomain::Training,
                                               .operation = "opacity regularization loss",
                                               .source = LFS_SOURCE_SITE_CURRENT(),
                                           })
                                    .error();
                            }
                            loss_tensor_gpu = loss_tensor_gpu + *opacity_loss_result;
                        }
                        nvtxRangePop();
                    }

                    if (bilateral_grid_ && params_.optimization.use_bilateral_grid) {
                        current_phase = StepPhase::OptimizerCommit;
                        nvtxRangePush("bilateral_grid_tv_and_step");
                        LFS_VRAM_SCOPE("train.bilateral_grid.tv_and_step");
                        LOG_VRAM_DIFF("train.bilateral_grid.tv_and_step");
                        const float tv_weight = params_.optimization.tv_loss_weight;

                        loss_tensor_gpu = loss_tensor_gpu + bilateral_grid_->tv_loss_gpu() * tv_weight;
                        bilateral_grid_->tv_backward(tv_weight);
                        bilateral_grid_->optimizer_step();
                        bilateral_grid_->zero_grad();
                        bilateral_grid_->scheduler_step();
                        ++mutation_epoch_;
                        persistent_commit = true;

                        nvtxRangePop();
                    }

                    if (ppisp_ && params_.optimization.use_ppisp && !ppisp_frozen) {
                        current_phase = StepPhase::OptimizerCommit;
                        nvtxRangePush("ppisp_reg_and_step");
                        LFS_VRAM_SCOPE("train.ppisp.reg_and_step");
                        LOG_VRAM_DIFF("train.ppisp.reg_and_step");

                        loss_tensor_gpu = loss_tensor_gpu + ppisp_->reg_loss_gpu();
                        ppisp_->reg_backward();
                        ppisp_->optimizer_step();
                        ppisp_->zero_grad();
                        ppisp_->scheduler_step();
                        ++mutation_epoch_;
                        persistent_commit = true;

                        nvtxRangePop();
                    }
                }

                // Sparsity loss - ALL ON GPU, no CPU sync here
                if (sparsity_optimizer_ &&
                    sparsity_optimizer_->should_apply_loss(iter) &&
                    (!fastgs_path || update_gaussians_this_iter)) {
                    nvtxRangePush("sparsity_loss");
                    LFS_VRAM_SCOPE("train.regularizers.sparsity_loss");
                    LOG_VRAM_DIFF("train.regularizers.sparsity_loss");
                    if (!run_fastgs_gaussian_backward) {
                        auto sparsity_result = compute_sparsity_loss_forward(iter, strategy_->get_model());
                        if (!sparsity_result) {
                            nvtxRangePop();
                            return lfs::from_legacy_expected<StepDisposition>(
                                       std::unexpected(sparsity_result.error()),
                                       lfs::LegacyErrorContext{
                                           .code = lfs::ErrorCode::Internal,
                                           .domain = lfs::ErrorDomain::Training,
                                           .operation = "sparsity loss forward",
                                           .source = LFS_SOURCE_SITE_CURRENT(),
                                       })
                                .error();
                        }
                        auto& [loss_tensor, ctx] = *sparsity_result;
                        sparsity_loss_gpu = std::move(loss_tensor);

                        if (ctx.n > 0) {
                            if (auto result = sparsity_optimizer_->compute_loss_backward(
                                    ctx, 1.0f, strategy_->get_optimizer().get_grad(ParamType::Opacity));
                                !result) {
                                nvtxRangePop();
                                return lfs::from_legacy_expected<StepDisposition>(
                                           std::unexpected(result.error()),
                                           lfs::LegacyErrorContext{
                                               .code = lfs::ErrorCode::Internal,
                                               .domain = lfs::ErrorDomain::Training,
                                               .operation = "sparsity loss backward",
                                               .source = LFS_SOURCE_SITE_CURRENT(),
                                           })
                                    .error();
                            }
                        }
                    }
                    nvtxRangePop();
                }

                // Sparsification phase logging (once per phase transition)
                if (params_.optimization.enable_sparsity) {
                    const int first_sparsify_iter = get_sparsity_boundary_iteration() + 1;
                    if (get_active_sparsify_steps() > 0 && iter == first_sparsify_iter) {
                        LOG_INFO("Entering sparsification: {} Gaussians, target prune={}%",
                                 strategy_->get_model().size(), params_.optimization.prune_ratio * 100);
                    }
                }

                // Loss readback at intervals, async: enqueue the D2H into the
                // pinned ring and report harvested samples from earlier iterations
                // — no pipeline stall.
                constexpr int LOSS_SYNC_INTERVAL = 10;
                if (iter % LOSS_SYNC_INTERVAL == 0 || iter == 1) {
                    lfs::core::Tensor total_loss = sparsity_loss_gpu.numel() > 0
                                                       ? (loss_tensor_gpu + sparsity_loss_gpu)
                                                       : loss_tensor_gpu;
                    if (auto harvested = harvestLossReadbacks(false, in_controller_phase); !harvested) {
                        return lfs::from_legacy_expected<StepDisposition>(
                                   std::unexpected(harvested.error()),
                                   lfs::LegacyErrorContext{
                                       .code = lfs::ErrorCode::Internal,
                                       .domain = lfs::ErrorDomain::Training,
                                       .operation = "harvest loss readbacks",
                                       .source = LFS_SOURCE_SITE_CURRENT(),
                                   })
                            .error();
                    }
                    submitLossReadback(total_loss, iter);
                }

                if (!in_sparsification && !fastgs_strategy_hooks_at_start) {
                    strategy_->pre_step(iter, r_output);
                }

                {
                    DeferredEvents deferred;
                    {
                        // Same rationale as the post_backward lock above: only block the
                        // render thread when topology actually changes this iteration. The
                        // optimizer step's in-place writes are ordered against the viewer by
                        // the interop semaphore (the render waits for the step's signal before
                        // reading), so the CPU write-lock is needed only for reallocation.
                        std::unique_lock<std::shared_mutex> lock(render_mutex_, std::defer_lock);
                        std::unique_lock<std::shared_mutex> model_write_lock(model_access_mutex_, std::defer_lock);
                        const bool refining = strategy_->is_refining(iter);
                        const bool morton_here =
                            morton_due && !fastgs_strategy_hooks_at_start && !in_sparsification;
                        const bool need_exclusive = refining || morton_here;
                        if (need_exclusive) {
                            lock.lock();
                        } else {
                            // Non-refining in-place writes: hold the model-access lock
                            // exclusive across the optimizer step so viewer/metric
                            // readers (which take it shared) cannot enter mid-write and
                            // tear the model. Refining excludes them via render_mutex_.
                            model_write_lock.lock();
                        }
                        // One-lock complete: see fastgs post_backward block above.
                        std::unique_lock<std::mutex> combined_model_lock;
                        if (need_exclusive && scene_) {
                            combined_model_lock = scene_->acquireCombinedModelExclusive();
                        }
                        // Drain in-flight reader events immediately before the optimizer
                        // step's in-place writes — not only at the loop top — so the
                        // trainer stream is ordered after any read that began mid-step.
                        // The exclusive lock (when refining) additionally bars new readers.
                        waitForModelReaders();
                        // Densify-window Vulkan drop/re-import for single-buffer q16
                        // commit (only when this path owns post_backward densify).
                        struct DensifyBarrierGuard {
                            Trainer* self = nullptr;
                            bool held = false;
                            explicit DensifyBarrierGuard(Trainer* t, bool need) : self(t) {
                                if (need && self) {
                                    const auto result = self->beginExportableDensifyBarrier();
                                    if (result == ExportableDensifyBarrierBegin::Failed) {
                                        throw std::runtime_error(
                                            "Failed to begin exportable densify barrier");
                                    }
                                    held = result == ExportableDensifyBarrierBegin::Acquired;
                                }
                            }
                            ~DensifyBarrierGuard() {
                                if (held && self) {
                                    (void)self->endExportableDensifyBarrier();
                                }
                            }
                            DensifyBarrierGuard(const DensifyBarrierGuard&) = delete;
                            DensifyBarrierGuard& operator=(const DensifyBarrierGuard&) = delete;
                        } densify_barrier(
                            this,
                            (refining || morton_here) && !in_sparsification &&
                                !fastgs_strategy_hooks_at_start);
                        LFS_VRAM_SCOPE("train.optimizer.strategy_step");
                        LOG_VRAM_DIFF("train.optimizer.strategy_step");
                        auto& model = strategy_->get_model();
                        const size_t model_size_before = static_cast<size_t>(model.size());

                        // Python hook: pre-optimizer-step (post-backward, pre-step)
                        {
                            lfs::training::HookContext ctx{
                                .iteration = iter,
                                .loss = current_loss_.load(),
                                .num_gaussians = strategy_ ? strategy_->get_model().size() : 0,
                                .is_refining = strategy_ ? strategy_->is_refining(iter) : false,
                                .trainer = this};
                            lfs::training::CommandCenter::instance().set_phase(lfs::training::TrainingPhase::OptimizerStep);
                            lfs::training::CommandCenter::instance().update_snapshot(
                                ctx, get_total_iterations(), is_paused_.load(), is_running_.load(), stop_requested_.load(),
                                lfs::training::TrainingPhase::OptimizerStep);
                            lfs::training::ControlBoundary::instance().notify(lfs::training::ControlHook::PreOptimizerStep, ctx);
                        }

                        if (!in_sparsification && !fastgs_strategy_hooks_at_start) {
                            current_phase = StepPhase::RefinementCommit;
                            ++mutation_epoch_;
                            persistent_commit = true;
                            strategy_->post_backward(iter, r_output);
                            maybe_morton_reorder(iter);
                        }
                        if (!fastgs_path) {
                            install_cropbox_step_damping(model, strategy_->get_optimizer());
                        }

                        // Skip strategy step if we're in controller distillation phase and freeze is enabled
                        const int ppisp_activation_step = params_.optimization.resolved_ppisp_controller_activation_step(get_total_iterations());
                        const bool freeze_gaussians = ppisp_controller_pool_ &&
                                                      params_.optimization.ppisp_use_controller &&
                                                      params_.optimization.ppisp_freeze_gaussians_on_distill &&
                                                      iter >= ppisp_activation_step;
                        if (!freeze_gaussians) {
                            current_phase = strategy_->is_refining(iter)
                                                ? StepPhase::RefinementCommit
                                                : StepPhase::OptimizerCommit;
                            ++mutation_epoch_;
                            persistent_commit = true;
                            strategy_->step(iter);
                        }

                        if (sparsity_optimizer_ &&
                            sparsity_optimizer_->is_initialized() &&
                            static_cast<size_t>(model.size()) != model_size_before) {
                            LOG_WARN("Sparsity: resetting ADMM state after topology change at iter {} ({} -> {})",
                                     iter, model_size_before, model.size());
                            sparsity_optimizer_->reset();
                        }

                        current_phase = StepPhase::RefinementCommit;
                        ++mutation_epoch_;
                        persistent_commit = true;
                        if (auto result = handle_sparsity_update(iter, model); !result) {
                            LOG_ERROR("Sparsity update: {}", result.error());
                        }
                        ++mutation_epoch_;
                        persistent_commit = true;
                        if (auto result = apply_sparsity_pruning(iter, model); !result) {
                            LOG_ERROR("Sparsity pruning: {}", result.error());
                        }

                        const bool topology_changed =
                            static_cast<size_t>(model.size()) != model_size_before;
                        if (topology_changed) {
                            syncTrainingSceneTopology(scene_, model);
                        }
                        if (auto result = ensureModelTensorAllocatorStorage(model, "strategy step"); !result) {
                            return lfs::from_legacy_expected<StepDisposition>(
                                       std::unexpected(result.error()),
                                       lfs::LegacyErrorContext{
                                           .code = lfs::ErrorCode::Internal,
                                           .domain = lfs::ErrorDomain::Training,
                                           .operation = "ensure model tensor allocator storage (strategy step)",
                                           .source = LFS_SOURCE_SITE_CURRENT(),
                                       })
                                .error();
                        }

                        if (topology_changed && !params_.optimization.headless) {
                            // Interactive sessions: densify/prune transients leave the
                            // pool at a high-water mark it never revisits. Return the
                            // slack to the driver at the commit boundary (device-sync
                            // cost lands on an already-synchronizing phase) so the
                            // viewer and other applications can use the VRAM.
                            lfs::core::Tensor::trim_memory_pool();
                        }

                        // End-of-step: parameters are consistent until the next
                        // step's writes; readers wait on this point.
                        current_phase = StepPhase::Publish;
                        recordParamsReady();
                        if (refining && !fastgs_strategy_hooks_at_start) {
                            resize_rasterizer_arena_at_boundary("B1 refine", false);
                        }
                    }

                    // Clean evaluation - let the evaluator handle everything
                    if (evaluator_->is_enabled() && evaluator_->should_evaluate(iter)) {
                        evaluator_->print_evaluation_header(iter);
                        eval_ppisp_applied_.store(0);
                        eval_ppisp_exif_.store(0);
                        auto metrics = evaluator_->evaluate(iter,
                                                            strategy_->get_model(),
                                                            val_dataset_,
                                                            background_);
                        if (evaluator_->has_appearance()) {
                            const int n = eval_ppisp_applied_.load();
                            const int k = eval_ppisp_exif_.load();
                            LOG_INFO("Eval: PPISP applied to {} held-out frames ({} with EXIF exposure, {} at mean exposure)",
                                     n, k, n - k);
                        }
                        LOG_INFO("{}", metrics.to_string());
                        // B2: retain only the current active shape after evaluation.
                        photometric_loss_.arena().shrink_to_required();
                    }

                    const bool save_regular_phase_output = get_active_sparsify_steps() > 0 &&
                                                           iter == get_sparsity_boundary_iteration();
                    current_phase = StepPhase::TerminalCleanup;
                    TrainerProjectSavePolicy project_save_policy;
                    std::optional<std::filesystem::path> step_project_path;
                    {
                        std::lock_guard lock(project_snapshot_mutex_);
                        project_save_policy = trainer_project_save_policy_;
                        if (live_project_path_ &&
                            !live_project_path_->empty()) {
                            step_project_path = *live_project_path_;
                        }
                    }
                    const bool may_save_at_step_boundary =
                        project_save_policy.at_step_boundaries &&
                        step_project_path.has_value();
                    if (save_regular_phase_output) {
                        if (may_save_at_step_boundary) {
                            LOG_INFO(
                                "Queuing regular-phase .licht generation at iteration {} before sparsification",
                                iter);
                            static_cast<void>(
                                request_project_save(
                                    *step_project_path));
                        } else {
                            LOG_DEBUG(
                                "Skipping regular-phase project save at iteration {}: trainer-initiated saves are not authorized",
                                iter);
                        }
                    }

                    // Publish a project generation at specified steps unless
                    // the sparsity-boundary save already handled it.
                    for (size_t save_step : params_.optimization.save_steps) {
                        if (iter == static_cast<int>(save_step) &&
                            iter != get_total_iterations() &&
                            !save_regular_phase_output) {
                            if (may_save_at_step_boundary) {
                                static_cast<void>(
                                    request_project_save(
                                        *step_project_path));
                            } else {
                                LOG_DEBUG(
                                    "Skipping step-boundary project save at iteration {}: trainer-initiated saves are not authorized",
                                    iter);
                            }
                        }
                    }

                    if (!params_.dataset.timelapse_images.empty() && iter % params_.dataset.timelapse_every == 0) {
                        for (const auto& img_name : params_.dataset.timelapse_images) {
                            auto train_cam = train_dataset_->get_camera_by_filename(img_name);
                            auto val_cam = val_dataset_ ? val_dataset_->get_camera_by_filename(img_name) : std::nullopt;
                            if (train_cam.has_value() || val_cam.has_value()) {
                                lfs::core::Camera* cam_to_use = train_cam.has_value() ? train_cam.value() : val_cam.value();

                                // Image size isn't correct until the image has been loaded once
                                // If we use the camera before it's loaded, it will render images at the non-scaled size
                                if ((cam_to_use->camera_height() == cam_to_use->image_height() && params_.dataset.resize_factor != 1) ||
                                    (params_.dataset.max_width > 0 &&
                                     (cam_to_use->image_height() > params_.dataset.max_width ||
                                      cam_to_use->image_width() > params_.dataset.max_width))) {
                                    cam_to_use->load_image_size(params_.dataset.resize_factor, params_.dataset.max_width);
                                }

                                RenderOutput rendered_timelapse_output;
                                if (params_.optimization.gut) {
                                    rendered_timelapse_output = gsplat_rasterize(*cam_to_use, strategy_->get_model(), background_,
                                                                                 1.0f, false, GsplatRenderMode::RGB, true);
                                } else {
                                    rendered_timelapse_output = fast_rasterize(*cam_to_use, strategy_->get_model(), background_);
                                }

                                // Get folder name to save in by stripping file extension
                                std::string folder_name = lfs::io::strip_extension(img_name);

                                auto output_path = params_.dataset.output_path / "timelapse" / folder_name;
                                std::filesystem::create_directories(output_path);

                                auto stamp = params_.include_provenance ? lfs::core::make_provenance_stamp()
                                                                        : lfs::core::make_minimal_provenance_stamp();
                                if (params_.include_provenance) {
                                    stamp.iteration = iter;
                                    const auto strategy = lfs::core::param::canonical_strategy_name(params_.optimization.strategy);
                                    if (!strategy.empty())
                                        stamp.strategy = std::string(strategy);
                                }
                                lfs::core::image_io::save_image_async(output_path / std::format("{:06d}.jpg", iter),
                                                                      rendered_timelapse_output.image,
                                                                      lfs::core::provenance_to_json(stamp));
                            } else {
                                LOG_WARN("Timelapse image '{}' not found in dataset.", img_name);
                            }
                        }
                    }
                }

                // Python hook: post-step (after optimizer and side-effects)
                {
                    lfs::training::HookContext ctx{
                        .iteration = iter,
                        .loss = current_loss_.load(),
                        .num_gaussians = strategy_ ? strategy_->get_model().size() : 0,
                        .is_refining = strategy_ ? strategy_->is_refining(iter) : false,
                        .trainer = this};
                    lfs::training::CommandCenter::instance().set_phase(lfs::training::TrainingPhase::SafeControl);
                    lfs::training::CommandCenter::instance().update_snapshot(
                        ctx, get_total_iterations(), is_paused_.load(), is_running_.load(), stop_requested_.load(),
                        lfs::training::TrainingPhase::SafeControl);
                    lfs::training::ControlBoundary::instance().notify(lfs::training::ControlHook::PostStep, ctx);
                }

                consume_requested_project_snapshot(iter);

                const auto project_hook =
                    getParams()
                        .save_project_at_iteration;
                if (project_hook &&
                    *project_hook <=
                        static_cast<std::size_t>(
                            std::numeric_limits<int>::
                                max())) {
                    TrainerProjectSavePolicy policy;
                    std::optional<std::filesystem::path> bound;
                    {
                        std::lock_guard lock(project_snapshot_mutex_);
                        policy = trainer_project_save_policy_;
                        if (live_project_path_ && !live_project_path_->empty())
                            bound = *live_project_path_;
                    }
                    const int target =
                        static_cast<int>(
                            *project_hook);
                    std::filesystem::path
                        project_hook_destination =
                            getParams()
                                .save_project_path;
                    if (project_hook_destination
                            .empty() &&
                        bound) {
                        project_hook_destination =
                            *bound;
                    }
                    const bool may_save_at_project_hook =
                        policy
                            .at_step_boundaries &&
                        !project_hook_destination
                             .empty();
                    if (!may_save_at_project_hook) {
                        if (iter + 1 == target ||
                            iter == target) {
                            LOG_DEBUG(
                                "Skipping save-project-at-iter project save at iteration {}: trainer-initiated saves are not authorized",
                                iter);
                        }
                    } else {
                        if (iter + 1 == target &&
                            !prepared_project_snapshot_) {
                            prepare_project_snapshot_at_safe_point(
                                target,
                                project_hook_destination);
                        }
                        if (iter == target) {
                            if (!prepared_project_snapshot_) {
                                prepare_project_snapshot_at_safe_point(
                                    iter,
                                    project_hook_destination);
                            }
                            capture_project_snapshot_at_safe_point(
                                iter);
                        }
                    }
                }

                if (live_vram_profiler_enabled() && strategy_) {
                    record_splat_vram_breakdown(strategy_->get_model());
                    record_optimizer_vram_breakdown(strategy_->get_optimizer());
                    publish_training_state_ledger(strategy_->get_model(),
                                                  &strategy_->get_optimizer());
                    lfs::diagnostics::VramProfiler::instance().sampleCudaMemory();
                }
                PerfBenchCollector::phase_mark(PerfBenchCollector::PhaseBoundary::StepEnd, iter);
                if (PerfBenchCollector::enabled() && strategy_) {
                    auto& bench = PerfBenchCollector::instance();
                    const auto ledger = compute_training_state_ledger(
                        strategy_->get_model(), &strategy_->get_optimizer());
                    bench.set_ledger(ledger);
                    bench.set_training_state_reserved_bytes(
                        compute_training_state_reserved_bytes(
                            strategy_->get_model(), &strategy_->get_optimizer()));
                    bench.on_step_end(iter,
                                      current_loss_.load(),
                                      strategy_->get_model().size());
                }

                // Return Continue if we should continue training
                if (iter < get_total_iterations() && !stop_requested_.load() && !stop_token.stop_requested()) {
                    return StepDisposition::Continue;
                } else {
                    return StepDisposition::Stop;
                }
            } catch (const lfs::Exception& e) {
                return e.error();
            } catch (const std::exception& e) {
                return lfs::make_error(lfs::ErrorInit{
                    .code = lfs::ErrorCode::Internal,
                    .domain = lfs::ErrorDomain::Training,
                    .user_message = "Training step failed.",
                    .detail = std::format("Training step failed: {}", e.what()),
                    .detection = LFS_SOURCE_SITE_CURRENT(),
                });
            }
        }();

        if (!result) {
            const auto phase_name = [](const StepPhase phase) constexpr -> std::string_view {
                switch (phase) {
                case StepPhase::AcquireData: return "AcquireData";
                case StepPhase::Forward: return "Forward";
                case StepPhase::Loss: return "Loss";
                case StepPhase::Backward: return "Backward";
                case StepPhase::OptimizerCommit: return "OptimizerCommit";
                case StepPhase::RefinementCommit: return "RefinementCommit";
                case StepPhase::Publish: return "Publish";
                case StepPhase::TerminalCleanup: return "TerminalCleanup";
                }
                return "Unknown";
            };
            lfs::SmallFields fields;
            fields.add("iteration", static_cast<std::int64_t>(iter))
                .add("mutation_epoch", static_cast<std::int64_t>(mutation_epoch_))
                .add("step_phase", phase_name(current_phase))
                .add("persistent_commit", persistent_commit);
            return std::move(result).error().with_context(
                "train_step", LFS_SOURCE_SITE_CURRENT(), std::move(fields));
        }
        return result;
    }

    lfs::Status Trainer::train(std::stop_token stop_token) {
        const std::uint64_t train_start_epoch = mutation_epoch_;
        StepPhase train_phase = StepPhase::AcquireData;
        const auto phase_name = [](const StepPhase phase) constexpr -> std::string_view {
            switch (phase) {
            case StepPhase::AcquireData: return "AcquireData";
            case StepPhase::Forward: return "Forward";
            case StepPhase::Loss: return "Loss";
            case StepPhase::Backward: return "Backward";
            case StepPhase::OptimizerCommit: return "OptimizerCommit";
            case StepPhase::RefinementCommit: return "RefinementCommit";
            case StepPhase::Publish: return "Publish";
            case StepPhase::TerminalCleanup: return "TerminalCleanup";
            }
            return "Unknown";
        };
        const auto attach_train_stamp = [this, &train_phase, &phase_name,
                                         train_start_epoch](lfs::Error error) -> lfs::Error {
            lfs::SmallFields fields;
            fields.add("iteration", static_cast<std::int64_t>(current_iteration_.load()))
                .add("mutation_epoch", static_cast<std::int64_t>(mutation_epoch_))
                .add("step_phase", phase_name(train_phase))
                .add("persistent_commit", mutation_epoch_ != train_start_epoch);
            return std::move(error).with_context(
                "train", LFS_SOURCE_SITE_CURRENT(), std::move(fields));
        };

        // Check if initialized
        if (!initialized_.load()) {
            return lfs::Status::failure(attach_train_stamp(lfs::make_error(lfs::ErrorInit{
                .code = lfs::ErrorCode::FailedPrecondition,
                .domain = lfs::ErrorDomain::Training,
                .user_message = "Trainer not initialized. Call initialize() before train().",
                .detection = LFS_SOURCE_SITE_CURRENT(),
            })));
        }

        training_complete_ = false;
        pending_snapshot_finish_reason_ =
            lfs::io::project::TrainingFinishReason::None;
        ready_to_start_ = false; // Reset the flag
        lfs::training::CommandCenter::instance().set_phase(lfs::training::TrainingPhase::SafeControl);

        ready_to_start_ = true; // Skip GUI wait for now

        {
            std::lock_guard<std::mutex> lock(params_mutex_);
            is_running_ = true; // Active setParams() calls queue from this point onward.
        }
        apply_pending_params_at_safe_point();
        LOG_INFO("Starting training loop");
        if (PerfBenchCollector::enabled()) {
            PerfBenchCollector::instance().on_training_start(get_total_iterations());
        }
        auto& cache_loader = lfs::io::CacheLoader::getInstance();
        std::optional<lfs::Error> terminal_error;
        const auto append_terminal_error = [&terminal_error](lfs::Error error) {
            if (terminal_error) {
                terminal_error = std::move(*terminal_error).with_suppressed(std::move(error));
            } else {
                terminal_error = std::move(error);
            }
        };

        try {
            cache_loader.reset_cache();
            cache_loader.update_cache_params(params_.dataset.loading_params.use_cpu_memory,
                                             train_dataset_size_,
                                             params_.dataset.loading_params.min_cpu_free_GB,
                                             params_.dataset.loading_params.min_cpu_free_memory_ratio,
                                             params_.dataset.loading_params.print_cache_status,
                                             params_.dataset.loading_params.print_status_freq_num);

            // Notify Python control layer that training is starting
            {
                lfs::training::HookContext ctx{
                    .iteration = 0,
                    .loss = current_loss_.load(),
                    .num_gaussians = strategy_ ? strategy_->get_model().size() : 0,
                    .is_refining = strategy_ ? strategy_->is_refining(0) : false,
                    .trainer = this};
                lfs::training::CommandCenter::instance().set_phase(lfs::training::TrainingPhase::SafeControl);
                lfs::training::CommandCenter::instance().update_snapshot(
                    ctx, get_total_iterations(), is_paused_.load(), is_running_.load(), stop_requested_.load(),
                    lfs::training::TrainingPhase::SafeControl);
                lfs::training::ControlBoundary::instance().notify(lfs::training::ControlHook::TrainingStart, ctx);
            }

            std::optional<lfs::core::CUDAStreamGuard> stream_guard;
            if (training_stream_) {
                stream_guard.emplace(training_stream_);
                // initialize() ran on another thread; order all of its CUDA work
                // before the first training-stream kernel.
                LFS_CUDA_TRY(
                    cudaDeviceSynchronize(), training_stream_,
                    "order initialize() work before first training-stream kernel");
            }

            // Start from current_iteration_ (allows resume from checkpoint)
            int iter = current_iteration_.load() > 0 ? current_iteration_.load() + 1 : 1;
            if (progress_) {
                progress_->update(
                    iter,
                    current_loss_.load(),
                    static_cast<int>(strategy_->get_model().size()),
                    get_progress_phase(iter));
            }

            // Conservative prefetch to avoid VRAM exhaustion
            lfs::io::PipelinedLoaderConfig pipelined_config;
            pipelined_config.jpeg_batch_size = 8;
            pipelined_config.prefetch_count = 8;
            pipelined_config.output_queue_size = 4;
            pipelined_config.io_threads = 2;
            pipelined_config.use_16bit_color = params_.dataset.loading_params.use_16bit_color;

            // Non-JPEG images (PNG, WebP) need CPU decoding - use more threads until cache warms
            constexpr float NON_JPEG_THRESHOLD = 0.1f;
            constexpr size_t MIN_COLD_THREADS = 4;
            constexpr size_t COLD_PREFETCH_COUNT = 16;
            const float non_jpeg_ratio = train_dataset_->get_non_jpeg_ratio();
            if (non_jpeg_ratio > NON_JPEG_THRESHOLD) {
                const size_t cold_threads = std::max(MIN_COLD_THREADS,
                                                     static_cast<size_t>(std::thread::hardware_concurrency() / 2));
                pipelined_config.cold_process_threads = cold_threads;
                pipelined_config.prefetch_count = COLD_PREFETCH_COUNT;
                LOG_INFO("{:.0f}% non-JPEG images, using {} cold threads", non_jpeg_ratio * 100.0f, cold_threads);
            }

            const bool alpha_available = scene_ && scene_->imagesHaveAlpha();
            PipelinedAuxiliaryImageConfig aux_pipeline_config;
            if (params_.optimization.use_depth_loss &&
                !depth_prior_mode_supported(params_.optimization.depth_loss_mode)) {
                LOG_WARN("Unknown depth loss mode '{}'; disabling depth loss",
                         params_.optimization.depth_loss_mode);
                params_.optimization.use_depth_loss = false;
            }
            aux_pipeline_config.load_depths =
                params_.optimization.use_depth_loss &&
                params_.optimization.depth_loss_weight > 0.0f;
            if (aux_pipeline_config.load_depths) {
                size_t cameras_with_depth = 0;
                for (const auto& cam : train_dataset_->get_cameras()) {
                    if (cam->has_depth()) {
                        ++cameras_with_depth;
                    }
                }
                LOG_INFO("Depth loss enabled (mode={}, weight={}, exponential decay to {:.0f}% by end of training)",
                         depth_prior_name(depth_prior_from_mode(params_.optimization.depth_loss_mode)),
                         params_.optimization.depth_loss_weight,
                         kDepthLossFinalScale * 100.0f);
                if (cameras_with_depth == 0) {
                    LOG_WARN("Depth loss is enabled but none of the {} training cameras has a depth map — "
                             "the depth loss will be inactive. Depth files go in <dataset>/depths (or depth/) "
                             "named like the image or sharing its trailing frame number.",
                             train_dataset_->get_cameras().size());
                } else {
                    LOG_INFO("Depth maps available for {}/{} training cameras",
                             cameras_with_depth, train_dataset_->get_cameras().size());
                    fitDepthAnchors(cameras_with_depth);
                }
            }
            aux_pipeline_config.load_normals =
                params_.optimization.use_normal_loss &&
                params_.optimization.normal_loss_weight > 0.0f;
            if (aux_pipeline_config.load_normals) {
                ensure_training_normal_maps(params_, train_dataset_->get_cameras());
                normal_prior_flip_yz_ = false;
                normal_prior_world_space_ = false;
                normal_prior_srgb_ = false;
                normal_prior_world_rotation_ = {1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f};
                normal_prior_usable_ = true;
                std::vector<const lfs::core::Camera*> normal_cameras;
                for (const auto& cam : train_dataset_->get_cameras()) {
                    if (cam->has_normal()) {
                        normal_cameras.push_back(cam.get());
                    }
                }
                LOG_INFO("Normal loss enabled (weight={}, space={})",
                         params_.optimization.normal_loss_weight,
                         lfs::core::param::normal_loss_space_name(params_.optimization.normal_loss_space));
                if (normal_cameras.empty()) {
                    LOG_WARN("Normal loss is enabled but none of the {} training cameras has a normal map — "
                             "the normal loss will be inactive. Normal files go in <dataset>/normals (or normal/) "
                             "named like the image or sharing its trailing frame number.",
                             train_dataset_->get_cameras().size());
                } else {
                    LOG_INFO("Normal maps available for {}/{} training cameras",
                             normal_cameras.size(), train_dataset_->get_cameras().size());

                    nvtxRangePush("resolve_normal_prior_convention");
                    const auto normal_resolve_start = std::chrono::steady_clock::now();
                    const auto convention = resolve_normal_prior_convention(
                        normal_cameras, params_.optimization.normal_loss_space);
                    const auto normal_resolve_ms = std::chrono::duration<double, std::milli>(
                                                       std::chrono::steady_clock::now() - normal_resolve_start)
                                                       .count();
                    nvtxRangePop();
                    LOG_INFO("Normal priors: convention resolve {:.1f} ms", normal_resolve_ms);
                    normal_prior_usable_ = convention.usable;
                    normal_prior_flip_yz_ = convention.flip_yz;
                    normal_prior_world_space_ = convention.world_space;
                    normal_prior_srgb_ = convention.srgb;
                    normal_prior_world_rotation_ = convention.world_rotation;
                    if (convention.usable) {
                        LOG_INFO("Normal priors resolved as {} ({} encoding, visibility consistency {:.3f}, space='{}')",
                                 convention.description,
                                 convention.srgb ? "sRGB" : "linear",
                                 convention.score,
                                 lfs::core::param::normal_loss_space_name(params_.optimization.normal_loss_space));
                        if (convention.score >= 0.0f && convention.score < 0.9f) {
                            LOG_WARN("Normal prior visibility consistency is only {:.3f} — the forced "
                                     "convention '{}' may not match the data",
                                     convention.score,
                                     lfs::core::param::normal_loss_space_name(params_.optimization.normal_loss_space));
                        }
                    } else {
                        LOG_WARN("Normal prior convention could not be resolved ({}; best visibility "
                                 "consistency {:.3f}, space='{}') — normal loss disabled",
                                 convention.description, convention.score,
                                 lfs::core::param::normal_loss_space_name(params_.optimization.normal_loss_space));
                    }
                }
                aux_pipeline_config.load_normals = normal_prior_usable_;
                aux_pipeline_config.normal_flip_yz = normal_prior_flip_yz_;
                aux_pipeline_config.normal_world_space = normal_prior_world_space_;
                aux_pipeline_config.normal_srgb = normal_prior_srgb_;
                aux_pipeline_config.normal_world_rotation = normal_prior_world_rotation_;
                if (normal_prior_world_space_) {
                    aux_pipeline_config.normal_world_to_camera_by_source.resize(
                        train_dataset_->get_cameras().size());
                    for (size_t i = 0; i < train_dataset_->get_cameras().size(); ++i) {
                        aux_pipeline_config.normal_world_to_camera_by_source[i] =
                            camera_world_to_camera_normal_matrix(
                                *train_dataset_->get_cameras()[i],
                                normal_prior_world_rotation_);
                    }
                }
            }
            if (evaluator_) {
                evaluator_->set_normal_prior_decode({
                    .srgb = normal_prior_srgb_,
                    .flip_yz = normal_prior_flip_yz_,
                    .world_space = normal_prior_world_space_,
                    .world_rotation = normal_prior_world_rotation_,
                });
            }
            if (params_.optimization.mask_mode != lfs::core::param::MaskMode::None) {
                aux_pipeline_config.invert_masks = params_.optimization.invert_masks;
                aux_pipeline_config.mask_threshold = params_.optimization.mask_threshold;
                if (params_.optimization.mask_mode == lfs::core::param::MaskMode::SegmentAndIgnore) {
                    aux_pipeline_config.mask_threshold = 0.0f;
                }
                if (params_.optimization.use_alpha_as_mask && alpha_available) {
                    aux_pipeline_config.use_alpha_as_mask = true;
                    aux_pipeline_config.load_masks = true;
                    LOG_INFO("Alpha-as-mask enabled (invert={}, threshold={})",
                             aux_pipeline_config.invert_masks, aux_pipeline_config.mask_threshold);
                } else {
                    aux_pipeline_config.load_masks = true;
                    LOG_INFO("Mask file loading enabled (invert={}, threshold={})",
                             aux_pipeline_config.invert_masks, aux_pipeline_config.mask_threshold);
                }
            }

            if (aux_pipeline_config.load_depths || aux_pipeline_config.load_normals) {
                constexpr size_t SIDECAR_COLD_THREAD_LIMIT = 8;
                constexpr size_t SIDECAR_PREFETCH_COUNT = 16;
                const size_t hw_threads = std::max<size_t>(1, std::thread::hardware_concurrency());
                const size_t sidecar_cold_threads =
                    std::max<size_t>(2, std::min(hw_threads / 2, SIDECAR_COLD_THREAD_LIMIT));
                if (pipelined_config.cold_process_threads < sidecar_cold_threads) {
                    LOG_INFO("Depth/normal sidecars active, using {} cold threads", sidecar_cold_threads);
                    pipelined_config.cold_process_threads = sidecar_cold_threads;
                }
                if (pipelined_config.prefetch_count < SIDECAR_PREFETCH_COUNT) {
                    LOG_INFO("Depth/normal sidecars active, increasing prefetch {} -> {}",
                             pipelined_config.prefetch_count,
                             SIDECAR_PREFETCH_COUNT);
                    pipelined_config.prefetch_count = SIDECAR_PREFETCH_COUNT;
                    pipelined_config.jpeg_batch_size = std::min(
                        pipelined_config.jpeg_batch_size,
                        std::max<size_t>(1, pipelined_config.prefetch_count));
                }
            }

            pipelined_config = tunePipelinedLoaderConfig(
                pipelined_config, train_dataset_, aux_pipeline_config);

            // Keep the camera stream stable across checkpoint resume.  The
            // loader is intentionally rebuilt after the checkpoint is loaded;
            // replaying the same seeded permutations from the saved iteration
            // prevents sidecar losses from seeing a different camera sequence.
            constexpr std::uint64_t TRAINING_SAMPLER_SEED = 0x4c46535f73616d70ULL;
            auto train_dataloader = create_infinite_pipelined_dataloader(
                train_dataset_,
                pipelined_config,
                aux_pipeline_config,
                TRAINING_SAMPLER_SEED,
                static_cast<size_t>(std::max(0, iter - 1)));
            auto active_image_loader_guard = makeScopeGuard([this]() {
                clearActiveImageLoader();
            });
            updateGTLoadConfigSnapshot();
            setActiveImageLoader(train_dataloader->get_loader_shared());
            strategy_->set_image_loader(train_dataloader->get_loader());

            // PipelinedDataLoader prefetched its initial batch in the constructor.
            // Do not reset it here: reset() clears those requests and immediately
            // refills the queue, producing avoidable startup cancellations.

            LOG_DEBUG("Starting training iterations");
            bool logged_epoch2_loader_cache = false;
            const size_t epoch2_loader_sample_count =
                train_dataset_ ? train_dataset_->size() * size_t{2} : size_t{0};
            while (iter <= get_total_iterations()) {
                lfs::core::Tensor::set_memory_pool_iteration(iter);

                if (stop_token.stop_requested() || stop_requested_.load())
                    break;
                if (callback_busy_.load(std::memory_order_acquire)) {
                    const cudaError_t callback_status = cudaStreamQuery(callback_stream_);
                    if (callback_status == cudaSuccess) {
                        callback_busy_.store(false, std::memory_order_release);
                    } else if (callback_status != cudaErrorNotReady) {
                        LOG_WARN("Callback stream query failed: {}", cudaGetErrorString(callback_status));
                        callback_busy_.store(false, std::memory_order_release);
                    }
                }

                lfs::core::Camera* cam = nullptr;
                lfs::core::Tensor gt_image;
                train_phase = StepPhase::AcquireData;
                // Dataloader wait is outside train_step / steady_ms.
                const auto dl_wait_t0 = std::chrono::steady_clock::now();
                auto example_opt = train_dataloader->next();
                const auto dl_wait_t1 = std::chrono::steady_clock::now();
                const double dl_wait_ms =
                    std::chrono::duration<double, std::milli>(dl_wait_t1 - dl_wait_t0).count();
                if (PerfBenchCollector::enabled()) {
                    PerfBenchCollector::instance().record_dataloader_wait(iter, dl_wait_ms);
                }
                if (!example_opt) {
                    const std::string detail = std::format(
                        "DataLoader ended unexpectedly at iteration {}",
                        current_iteration_.load());
                    LOG_ERROR("{}", detail);
                    append_terminal_error(lfs::make_error(lfs::ErrorInit{
                        .code = lfs::ErrorCode::Unavailable,
                        .domain = lfs::ErrorDomain::Training,
                        .user_message = "Training data loader ended unexpectedly.",
                        .detail = detail,
                        .detection = LFS_SOURCE_SITE_CURRENT(),
                    }));
                    break;
                }
                auto& example = *example_opt;
                cam = example.data.camera;
                gt_image = std::move(example.data.image);

                for (CUevent_st** event : {&example.depth_ready_event, &example.normal_ready_event}) {
                    if (!*event) {
                        continue;
                    }
                    const auto wait_site = LFS_SOURCE_SITE_CURRENT();
                    lfs::core::record_cuda_breadcrumb(
                        "cudaStreamWaitEvent(training_stream_, *event, 0)", __FILE__, __LINE__,
                        training_stream_);
                    const auto wait_state = lfs::core::prepare_cuda_check(
                        "cudaStreamWaitEvent(training_stream_, *event, 0)", wait_site,
                        training_stream_);
                    const cudaError_t wait_err = cudaStreamWaitEvent(training_stream_, *event, 0);
                    const auto wait_completion =
                        lfs::core::complete_cuda_check(wait_err, wait_state);
                    if (wait_completion.effective_error != cudaSuccess) {
                        orphaned_sidecar_events_.push_back(*event);
                        *event = nullptr;
                        lfs::core::throw_cuda_error(
                            wait_err, wait_state, wait_completion,
                            "cudaStreamWaitEvent(training_stream_, *event, 0)",
                            ::lfs::core::detail::format_cuda_safe(
                                "sidecar loader ready, iteration {}", iter),
                            wait_site);
                    }
                    LFS_CUDA_LOG_TEARDOWN(cudaEventDestroy(*event), nullptr,
                                          "destroy sidecar loader-ready event");
                    *event = nullptr;
                }
                // Store pipelined mask for use in train_step
                pipelined_mask_ = example.mask.has_value() ? std::move(*example.mask) : lfs::core::Tensor();
                pipelined_depth_ = example.depth.has_value() ? std::move(*example.depth) : lfs::core::Tensor();
                pipelined_normal_ = example.normal.has_value() ? std::move(*example.normal) : lfs::core::Tensor();
                if (pipelined_depth_.is_valid()) {
                    pipelined_depth_.set_stream(training_stream_);
                }
                if (pipelined_normal_.is_valid()) {
                    pipelined_normal_.set_stream(training_stream_);
                }
                if (gt_image.is_valid()) {
                    gt_image.set_stream(training_stream_);
                }
                if (pipelined_mask_.is_valid()) {
                    pipelined_mask_.set_stream(training_stream_);
                }

                if (!logged_epoch2_loader_cache && epoch2_loader_sample_count > 0 &&
                    static_cast<size_t>(iter) >= epoch2_loader_sample_count) {
                    const auto stats = train_dataloader->get_stats();
                    LOG_DEBUG("[PipelinedImageLoader] after epoch 2: {} compressed entries, {:.1f} MiB RAM, "
                              "{} hits, {} misses",
                              stats.jpeg_cache_entries,
                              stats.jpeg_cache_bytes / (1024.0 * 1024.0),
                              stats.hot_path_hits,
                              stats.cold_path_misses);
                    logged_epoch2_loader_cache = true;
                }

                train_phase = StepPhase::Forward;
                const auto training_step_begin =
                    std::chrono::steady_clock::now();
                auto step_result = train_step(iter, cam, gt_image, stop_token);
                const double training_step_ms =
                    std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() -
                        training_step_begin)
                        .count();
                if (!step_result) {
                    terminal_error = std::move(step_result).error();
                    break;
                }
                observe_training_step_duration(
                    iter, training_step_ms,
                    strategy_ &&
                        strategy_->is_refining(iter));
                train_dataloader->observe_training_iteration(
                    training_step_ms, dl_wait_ms, iter);

                // Transition to safe control phase and execute deferred Python callbacks
                lfs::training::CommandCenter::instance().set_phase(lfs::training::TrainingPhase::SafeControl);
                lfs::training::ControlBoundary::instance().drain_callbacks();

                if (*step_result == StepDisposition::Stop) {
                    break;
                }

                // Launch callback for async progress update (except first iteration)
                if (iter > 1 && callback_ && !callback_busy_.load(std::memory_order_acquire)) {
                    callback_busy_.store(true, std::memory_order_release);
                    auto err = cudaLaunchHostFunc(
                        callback_stream_,
                        [](void* self) {
                            auto* trainer = static_cast<Trainer*>(self);
                            if (trainer->callback_) {
                                trainer->callback_();
                            }
                            trainer->callback_busy_.store(false, std::memory_order_release);
                        },
                        this);
                    if (err != cudaSuccess) {
                        LOG_WARN("Failed to launch callback: {}", cudaGetErrorString(err));
                        callback_busy_.store(false, std::memory_order_release);
                    }
                }

                ++iter;
            }

            clearActiveImageLoader();
            active_image_loader_guard.release();

            maybe_publish_camera_loss_heatmap(current_iteration_.load(), true);

            if (auto harvested = harvestLossReadbacks(true, false); !harvested) {
                auto typed = lfs::from_legacy_expected<void>(
                    std::move(harvested),
                    lfs::LegacyErrorContext{
                        .code = lfs::ErrorCode::Internal,
                        .domain = lfs::ErrorDomain::Training,
                        .operation = "loss readback finalization",
                        .source = LFS_SOURCE_SITE_CURRENT(),
                    });
                append_terminal_error(std::move(typed).error());
            }
        } catch (const lfs::Exception& e) {
            append_terminal_error(e.error());
        } catch (const std::exception& e) {
            append_terminal_error(lfs::make_error(lfs::ErrorInit{
                .code = lfs::ErrorCode::Internal,
                .domain = lfs::ErrorDomain::Training,
                .user_message = "Training failed.",
                .detail = std::format("Training failed: {}", e.what()),
                .detection = LFS_SOURCE_SITE_CURRENT(),
            }));
        }

        train_phase = StepPhase::TerminalCleanup;

        if (callback_busy_.load()) {
            const auto callback_status = cudaStreamSynchronize(callback_stream_);
            if (callback_status != cudaSuccess) {
                append_terminal_error(lfs::make_error(lfs::ErrorInit{
                    .code = lfs::ErrorCode::Internal,
                    .domain = lfs::ErrorDomain::Training,
                    .user_message = "Failed to finish the training callback.",
                    .detail = std::format("Failed to finish training callback: {}",
                                          cudaGetErrorString(callback_status)),
                    .detection = LFS_SOURCE_SITE_CURRENT(),
                }));
            }
        }

        const int terminal_iteration = current_iteration_.load();
        const bool user_stopped = stop_requested_.load() || stop_token.stop_requested();
        apply_pending_params_at_safe_point();
        pending_snapshot_finish_reason_ =
            terminal_error
                ? lfs::io::project::TrainingFinishReason::Error
            : user_stopped
                ? lfs::io::project::TrainingFinishReason::UserStopped
                : lfs::io::project::TrainingFinishReason::Completed;
        TrainerProjectSavePolicy terminal_save_policy;
        std::optional<std::filesystem::path> terminal_project_path;
        {
            std::lock_guard lock(project_snapshot_mutex_);
            terminal_save_policy = trainer_project_save_policy_;
            if (live_project_path_ &&
                !live_project_path_->empty()) {
                terminal_project_path = *live_project_path_;
            }
        }
        const bool authorize_terminal_save =
            (user_stopped || terminal_error)
                ? terminal_save_policy.on_stop_or_error
                : terminal_save_policy.on_completion;
        if (authorize_terminal_save &&
            terminal_project_path) {
            const auto terminal_save_started = std::chrono::steady_clock::now();
            saving_model_.store(true, std::memory_order_release);
            try {
                LOG_INFO("Saving {} project at iteration {}...",
                         terminal_error ? "recovery" : (user_stopped ? "stopped" : "final"),
                         terminal_iteration);
                if (auto save_result = save_project_to(
                        *terminal_project_path,
                        terminal_iteration);
                    !save_result) {
                    append_terminal_error(lfs::make_error(lfs::ErrorInit{
                        .code = lfs::ErrorCode::Internal,
                        .domain = lfs::ErrorDomain::Training,
                        .user_message = "Terminal training save failed.",
                        .detail = std::format("Terminal save failed at iteration {}: {}",
                                              terminal_iteration, save_result.error()),
                        .detection = LFS_SOURCE_SITE_CURRENT(),
                    }));
                }
            } catch (const std::exception& e) {
                append_terminal_error(lfs::make_error(lfs::ErrorInit{
                    .code = lfs::ErrorCode::Internal,
                    .domain = lfs::ErrorDomain::Training,
                    .user_message = "Terminal training save failed.",
                    .detail = std::format("Terminal save threw at iteration {}: {}",
                                          terminal_iteration, e.what()),
                    .detection = LFS_SOURCE_SITE_CURRENT(),
                }));
            }
            LOG_INFO("Terminal {} save phase took {:.3f}s",
                     user_stopped ? "stop" : "completion",
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - terminal_save_started).count());
            pending_snapshot_finish_reason_ =
                lfs::io::project::TrainingFinishReason::None;
            saving_model_.store(false, std::memory_order_release);
        } else {
            const char* skip_reason =
                authorize_terminal_save
                    ? "no project path is bound"
                : user_stopped
                    ? "saves on user stop are not authorized"
                : terminal_error
                    ? "saves on training error are not authorized"
                    : "not authorized";
            LOG_INFO(
                "Training ended at iteration {} without a project save: {}",
                terminal_iteration,
                skip_reason);
        }

        // A dead train loop must not report an active pause: stale is_paused_
        // makes finished-trainer saves skip the synchronous flush and wait
        // forever on a safe point.
        pause_requested_ = false;
        is_paused_ = false;
        {
            std::lock_guard<std::mutex> lock(params_mutex_);
            is_running_ = false;
        }
        training_complete_ = true;
        clearActiveImageLoader();
        cache_loader.clear_cpu_cache();
        lfs::core::image_io::wait_for_pending_saves();

        try {
            if (progress_) {
                progress_->complete(user_stopped, terminal_iteration);
            }
            if (evaluator_) {
                evaluator_->save_report();
            }
            if (progress_ && strategy_) {
                progress_->print_final_summary(static_cast<int>(strategy_->get_model().size()),
                                               terminal_iteration, user_stopped);
            }
            if (PerfBenchCollector::enabled()) {
                const auto report_path =
                    params_.dataset.output_path / "perf_bench.json";
                PerfBenchCollector::instance().finalize(report_path);
            }
        } catch (const std::exception& e) {
            append_terminal_error(lfs::make_error(lfs::ErrorInit{
                .code = lfs::ErrorCode::Internal,
                .domain = lfs::ErrorDomain::Training,
                .user_message = "Terminal training reporting failed.",
                .detail = std::format("Terminal reporting failed: {}", e.what()),
                .detection = LFS_SOURCE_SITE_CURRENT(),
            }));
        }

        // B3: training has stopped or completed; the editor may remain alive.
        photometric_loss_.arena().reset();
        resize_rasterizer_arena_at_boundary("B3 training end", true);

        auto& command_center = lfs::training::CommandCenter::instance();
        auto snapshot_guard = makeScopeGuard([&command_center, this]() {
            command_center.clear_snapshot(this);
        });
        try {
            lfs::training::HookContext ctx{
                .iteration = terminal_iteration,
                .loss = current_loss_.load(),
                .num_gaussians = strategy_ ? strategy_->get_model().size() : 0,
                .is_refining = strategy_ ? strategy_->is_refining(terminal_iteration) : false,
                .trainer = this};
            command_center.set_phase(lfs::training::TrainingPhase::SafeControl);
            command_center.update_snapshot(
                ctx, get_total_iterations(), is_paused_.load(), false, user_stopped,
                lfs::training::TrainingPhase::SafeControl);
            auto& boundary = lfs::training::ControlBoundary::instance();
            boundary.notify(lfs::training::ControlHook::TrainingEnd, ctx);
            boundary.drain_callbacks();
        } catch (const std::exception& e) {
            append_terminal_error(lfs::make_error(lfs::ErrorInit{
                .code = lfs::ErrorCode::Internal,
                .domain = lfs::ErrorDomain::Training,
                .user_message = "Training-end callback dispatch failed.",
                .detail = std::format("TrainingEnd callback dispatch failed: {}", e.what()),
                .detection = LFS_SOURCE_SITE_CURRENT(),
            }));
        }

        if (terminal_error) {
            return lfs::Status::failure(
                attach_train_stamp(std::move(*terminal_error)));
        }
        LOG_INFO("Training {}", user_stopped ? "stopped by user" : "completed successfully");
        return {};
    }

    void Trainer::set_trainer_project_save_policy(
        TrainerProjectSavePolicy policy) {
        std::lock_guard lock(project_snapshot_mutex_);
        trainer_project_save_policy_ = policy;
    }

    Trainer::TrainerProjectSavePolicy
    Trainer::trainer_project_save_policy() const {
        std::lock_guard lock(project_snapshot_mutex_);
        return trainer_project_save_policy_;
    }

    std::optional<std::filesystem::path>
    Trainer::bound_project_path() const {
        std::lock_guard lock(project_snapshot_mutex_);
        if (live_project_path_ &&
            !live_project_path_->empty()) {
            return *live_project_path_;
        }
        return std::nullopt;
    }

    void Trainer::set_live_project_snapshot(
        std::optional<std::filesystem::path> path,
        std::function<std::optional<
            ProjectSnapshotDocumentContext>()>
            context_provider) {
        std::lock_guard lock(project_snapshot_mutex_);
        live_project_path_ = std::move(path);
        live_document_context_provider_ =
            std::move(context_provider);
    }

    void Trainer::attach_live_document_context(
        std::optional<ProjectSnapshotDocumentContext>&
            document_context) const {
        if (document_context) {
            return;
        }
        std::function<std::optional<
            ProjectSnapshotDocumentContext>()>
            provider;
        {
            std::lock_guard lock(
                project_snapshot_mutex_);
            provider = live_document_context_provider_;
        }
        if (!provider) {
            return;
        }
        document_context = provider();
    }

    int Trainer::project_snapshot_iteration() const {
        const int current = current_iteration_.load();
        if (is_paused_.load() && current > 0) {
            return current - 1;
        }
        return current;
    }

    std::expected<std::filesystem::path, std::string>
    Trainer::save_project_to(
        const std::filesystem::path& requested_path,
        const int iteration,
        std::optional<ProjectSnapshotDocumentContext>
            document_context) {
        if (!strategy_ || !scene_) {
            return std::unexpected(
                "Cannot save project: no initialized training scene");
        }
        if (iteration < 0) {
            return std::unexpected(
                "Cannot save project: iteration is negative");
        }

        attach_live_document_context(document_context);
        std::filesystem::path path = requested_path;
        if (path.empty()) {
            auto bound = bound_project_path();
            if (!bound) {
                return std::unexpected(
                    "No project destination is bound for this trainer");
            }
            path = std::move(*bound);
        }
        if (path.extension() != ".licht") {
            return std::unexpected(
                "Project destination must end in .licht");
        }

        finish_project_writer();
        auto chapters =
            reserve_project_snapshot_chapters();
        if (!chapters) {
            return std::unexpected(
                lfs::format_for_developer(
                    chapters.error()));
        }
        if (document_context &&
            pending_snapshot_finish_reason_ !=
                lfs::io::project::
                    TrainingFinishReason::None) {
            document_context->metrics.finish_reason =
                pending_snapshot_finish_reason_;
        }
        {
            std::lock_guard lock(
                project_snapshot_mutex_);
            (*chapters)->document_context =
                std::move(document_context);
            prestaged_project_chapters_ =
                std::move(*chapters);
            prestaged_project_request_id_ = 0;
            last_project_writer_error_.clear();
        }

        prepare_project_snapshot_at_safe_point(
            iteration, path);
        capture_project_snapshot_at_safe_point(
            iteration);
        finish_project_writer();

        std::lock_guard lock(project_snapshot_mutex_);
        if (!last_project_writer_error_.empty()) {
            return std::unexpected(
                last_project_writer_error_);
        }
        if (last_project_snapshot_path_ != path) {
            return std::unexpected(
                "Project snapshot did not publish the requested destination");
        }
        return path;
    }

    lfs::core::Tensor Trainer::applyPPISPForEval(const lfs::core::Tensor& rgb,
                                                 const lfs::core::Camera& cam) const {
        eval_ppisp_applied_.fetch_add(1, std::memory_order_relaxed);

        const auto params = getParams();
        if (!ppisp_ || !params.optimization.use_ppisp || !ppisp_->isFinalized() ||
            rgb.shape().rank() != 3) {
            return rgb;
        }

        auto rgb_chw = rgb.device() == lfs::core::Device::CUDA ? rgb : rgb.cuda();
        if (rgb_chw.shape()[0] != 3 && rgb_chw.shape()[2] == 3) {
            rgb_chw = rgb_chw.permute({2, 0, 1}).contiguous();
        } else if (!rgb_chw.is_contiguous()) {
            rgb_chw = rgb_chw.contiguous();
        }

        auto* const pool = controller_pool_for_save(get_current_iteration());
        if (pool && params.optimization.ppisp_use_controller) {
            const bool is_training_camera = ppisp_->is_known_frame(cam.uid());
            const int camera_idx =
                is_training_camera ? ppisp_->camera_index(ppisp_->camera_for_frame(cam.uid())) : 0;
            const int controller_idx =
                (camera_idx >= 0 && camera_idx < pool->num_cameras()) ? camera_idx : 0;
            std::lock_guard<std::mutex> controller_lock(pool->predict_mutex());
            const auto controller_params = pool->predict(controller_idx, rgb_chw.unsqueeze(0), 1.0f);
            return ppisp_->apply_with_controller_params(rgb_chw, controller_params, controller_idx);
        }

        int camera_id = cam.camera_id();
        if (!ppisp_->is_known_camera(camera_id)) {
            camera_id = ppisp_->any_camera_id();
        }

        if (ppisp_->is_known_frame(cam.uid())) {
            return ppisp_->apply(rgb_chw, camera_id, cam.uid());
        }

        float exposure = 0.0f;
        if (ppisp_exif_exposure_mean_) {
            const auto ev = lfs::core::exif_exposure_ev_for_training_image(
                cam.image_path(), params.dataset.data_path);
            if (ev) {
                exposure = std::clamp(0.5f * (static_cast<float>(*ev) - *ppisp_exif_exposure_mean_),
                                      -16.0f, 16.0f);
                eval_ppisp_exif_.fetch_add(1, std::memory_order_relaxed);
            }
        }

        return ppisp_->apply_with_exposure(rgb_chw, camera_id, exposure);
    }

    lfs::core::Tensor Trainer::applyPPISPForViewport(const lfs::core::Tensor& rgb, const int camera_uid,
                                                     const PPISPViewportOverrides& overrides,
                                                     const bool use_controller) const {
        const auto params = getParams();
        if (!ppisp_ || !params.optimization.use_ppisp || rgb.shape().rank() != 3) {
            return rgb;
        }

        const bool is_chw = (rgb.shape()[0] == 3);
        const auto rgb_chw = is_chw ? rgb : rgb.permute({2, 0, 1}).contiguous();
        const bool is_training_camera = ppisp_->is_known_frame(camera_uid);
        const bool has_controller = ppisp_controller_pool_ && params.optimization.ppisp_use_controller;
        const int camera_idx = is_training_camera ? ppisp_->camera_index(ppisp_->camera_for_frame(camera_uid)) : 0;

        lfs::core::Tensor result;

        if (use_controller && has_controller) {
            const int controller_idx =
                (camera_idx >= 0 && camera_idx < ppisp_controller_pool_->num_cameras()) ? camera_idx : 0;
            std::lock_guard<std::mutex> controller_lock(ppisp_controller_pool_->predict_mutex());
            const auto controller_params = ppisp_controller_pool_->predict(controller_idx, rgb_chw.unsqueeze(0), 1.0f);
            result = overrides.isIdentity()
                         ? ppisp_->apply_with_controller_params(rgb_chw, controller_params, controller_idx)
                         : ppisp_->apply_with_controller_params_and_overrides(rgb_chw, controller_params, controller_idx,
                                                                              toRenderOverrides(overrides));
        } else if (is_training_camera) {
            const int camera_id = ppisp_->camera_for_frame(camera_uid);
            result = overrides.isIdentity() ? ppisp_->apply(rgb_chw, camera_id, camera_uid)
                                            : ppisp_->apply_with_overrides(rgb_chw, camera_id, camera_uid,
                                                                           toRenderOverrides(overrides));
        } else {
            // Manual viewport overrides should remain usable on novel views even without a controller.
            // Fall back to any learned PPISP frame/camera pair so the user can still inspect exposure,
            // vignette, white-balance, and CRF adjustments on free-orbit renders.
            const int fallback_camera = ppisp_->any_camera_id();
            const int fallback_frame = ppisp_->any_frame_uid();
            result = overrides.isIdentity()
                         ? ppisp_->apply(rgb_chw, fallback_camera, fallback_frame)
                         : ppisp_->apply_with_overrides(rgb_chw, fallback_camera, fallback_frame,
                                                        toRenderOverrides(overrides));
        }

        return is_chw ? result : result.permute({1, 2, 0}).contiguous();
    }

    PPISPControllerPool* Trainer::controller_pool_for_save(const int iteration) const {
        if (!ppisp_controller_pool_) {
            return nullptr;
        }
        if (is_ppisp_frozen()) {
            return ppisp_controller_pool_.get();
        }
        const auto params = getParams();
        return iteration >= params.optimization.resolved_ppisp_controller_activation_step(get_total_iterations())
                   ? ppisp_controller_pool_.get()
                   : nullptr;
    }

    lfs::core::param::TrainingParameters Trainer::params_for_project_snapshot() const {
        auto params = getParams();
        if (scene_) {
            const auto disabled = scene_->getTrainingDisabledCameraUids();
            params.disabled_camera_uids.assign(disabled.begin(), disabled.end());
        }
        absolutize_dataset_path_for_snapshot(params.dataset.data_path);
        return params;
    }

    std::expected<int, std::string> Trainer::load_checkpoint(
        const std::filesystem::path& checkpoint_path) {
        std::error_code error;
        const auto source_bytes =
            std::filesystem::file_size(
                checkpoint_path, error);
        if (error) {
            return std::unexpected(
                std::format(
                    "Cannot stat checkpoint '{}': {}",
                    lfs::core::path_to_utf8(
                        checkpoint_path),
                    error.message()));
        }
        std::ifstream source(
            checkpoint_path,
            std::ios::binary);
        if (!source) {
            return std::unexpected(
                std::format(
                    "Cannot open checkpoint '{}'",
                    lfs::core::path_to_utf8(
                        checkpoint_path)));
        }
        return load_checkpoint(
            source, source_bytes,
            lfs::core::path_to_utf8(
                checkpoint_path));
    }

    CheckpointLoadResult Trainer::load_checkpoint(
        std::istream& source,
        const std::uint64_t source_bytes,
        const std::string_view source_name) {
        if (!strategy_) {
            return std::unexpected("Cannot load checkpoint: no strategy initialized");
        }
        if (source_bytes == 0 ||
            source_bytes >
                lfs::core::
                    MAX_CHECKPOINT_FILE_BYTES) {
            return std::unexpected(
                "Embedded checkpoint size is invalid");
        }

        // Create bilateral grid before loading if needed (checkpoint may contain grid state)
        if (params_.optimization.use_bilateral_grid && !bilateral_grid_) {
            if (auto init_result = initialize_bilateral_grid(); !init_result) {
                LOG_WARN("Failed to init bilateral grid for resume: {}", init_result.error());
            }
        }

        // Create PPISP before loading if needed
        if (params_.optimization.use_ppisp && !ppisp_) {
            if (auto init_result = initialize_ppisp(); !init_result) {
                LOG_WARN("Failed to init PPISP for resume: {}", init_result.error());
            }
        }

        // Create PPISP controller pool before loading if needed
        if (params_.optimization.ppisp_use_controller && !ppisp_controller_pool_) {
            bool should_initialize_controller = true;
            if (is_ppisp_frozen()) {
                source.clear();
                source.seekg(0, std::ios::beg);
                const auto checkpoint_header =
                    lfs::core::load_checkpoint_header(
                        source, source_bytes);
                if (!checkpoint_header) {
                    LOG_WARN("Failed to inspect checkpoint header for PPISP controller state: {}",
                             checkpoint_header.error());
                    should_initialize_controller = false;
                } else {
                    should_initialize_controller =
                        lfs::core::has_flag(checkpoint_header->flags, lfs::core::CheckpointFlags::HAS_PPISP_CONTROLLER);
                    if (!should_initialize_controller) {
                        LOG_INFO("Checkpoint has no PPISP controller pool; frozen controller state remains disabled");
                    }
                }
            }
            if (should_initialize_controller) {
                if (auto init_result = initialize_ppisp_controller(); !init_result) {
                    LOG_WARN("Failed to init PPISP controller pool for resume: {}", init_result.error());
                }
            }
        }

        source.clear();
        source.seekg(0, std::ios::beg);
        if (!source) {
            return std::unexpected(
                std::format(
                    "Cannot seek {} to byte zero",
                    source_name));
        }
        auto result = lfs::training::load_checkpoint(
            source, source_bytes, *strategy_, params_,
            bilateral_grid_.get(), ppisp_.get(),
            ppisp_controller_pool_.get(),
            dynamic_cast<ADMMSparsityOptimizer*>(sparsity_optimizer_.get()),
            splat_tensor_allocator_, source_name);
        if (!result) {
            return result;
        }
        if (params_.resume_project && scene_) {
            const auto disabled =
                scene_->getTrainingDisabledCameraUids();
            params_.disabled_camera_uids.assign(
                disabled.begin(), disabled.end());
        }
        {
            std::unique_lock<std::shared_mutex> render_lock(render_mutex_);
            if (auto storage_result = ensureModelTensorAllocatorStorage(
                    strategy_->get_model(), "checkpoint load");
                !storage_result) {
                return std::unexpected(storage_result.error());
            }
            apply_frozen_ranges_to_optimizer(strategy_->get_model(), strategy_->get_optimizer());
        }
        if (params_.optimization.enable_sparsity) {
            const size_t stop_refine_limit = static_cast<size_t>(std::max(0, get_regular_iterations()));
            if (params_.optimization.stop_refine > stop_refine_limit) {
                LOG_WARN("Sparsity: clamping stop_refine from {} to {} after checkpoint load",
                         params_.optimization.stop_refine, stop_refine_limit);
                params_.optimization.stop_refine = stop_refine_limit;
            }
        }
        sync_strategy_optimization_params();

        current_iteration_ = *result;

        LOG_INFO("Restored training state from checkpoint at iteration {}", *result);
        return result;
    }

} // namespace lfs::training
