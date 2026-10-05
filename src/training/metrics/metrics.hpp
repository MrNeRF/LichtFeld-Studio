/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "../dataset.hpp"
#include "core/error.hpp"
#include "core/nn/models/lpips.hpp"
#include "core/parameters.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "training/optimizer/render_output.hpp"
#include <array>
#include <cmath>
#include <expected>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace lfs::io {
    class MaskDirCache;
}

namespace lfs::training {

    // Peak Signal-to-Noise Ratio
    class PSNR {
    public:
        explicit PSNR(const float data_range = 1.0f) : data_range_(data_range) {
        }

        float compute(const lfs::core::Tensor& pred, const lfs::core::Tensor& target,
                      const lfs::core::Tensor& mask = {}) const;

    private:
        const float data_range_;
    };

    // Structural Similarity Index (using LibTorch-free kernels)
    class SSIM {
    public:
        SSIM(bool apply_valid_padding = true);

        float compute(const lfs::core::Tensor& pred, const lfs::core::Tensor& target,
                      const lfs::core::Tensor& mask = {});

    private:
        bool apply_valid_padding_;
    };

    struct ViewMetrics {
        int index = 0;
        std::string image_name;
        int width = 0;
        int height = 0;
        std::optional<float> psnr;
        std::optional<float> ssim;
        std::optional<float> lpips;
        std::optional<float> flip;
        bool masked = false;
        int bit_depth = 8;
        float evaluated_pixel_fraction = 0.0f;
        bool validity_mask_applied = false;
        std::string skipped_reason;
    };

    struct EvalMetrics {
        float psnr = 0.0f;
        float ssim = 0.0f;
        std::optional<float> lpips;
        std::optional<float> flip;
        float elapsed_time = 0.0f;
        int num_gaussians = 0;
        int iteration = 0;
        bool valid = false;
        std::optional<float> normal_angle_deg;
        std::optional<float> depth_absrel;
        float bias_r = 0.0f;
        float bias_g = 0.0f;
        float bias_b = 0.0f;
        float bias_corr_r = 0.0f;
        float bias_corr_g = 0.0f;
        float bias_corr_b = 0.0f;
        std::vector<ViewMetrics> views;

        [[nodiscard]] std::string to_string() const {
            if (!valid) {
                return "No valid evaluation images";
            }
            std::stringstream ss;
            ss << std::fixed << std::setprecision(4);
            ss << "PSNR: " << psnr
               << ", SSIM: " << ssim;
            if (lpips && std::isfinite(*lpips)) {
                ss << ", LPIPS: " << *lpips;
            }
            if (flip && std::isfinite(*flip)) {
                ss << ", FLIP: " << *flip;
            }
            ss << ", Time: " << elapsed_time << "s/image"
               << ", #GS: " << num_gaussians
               << ", bias=(" << bias_r << "," << bias_g << "," << bias_b << ")"
               << ", bias_corr=(" << bias_corr_r << "," << bias_corr_g << "," << bias_corr_b << ")";
            if (normal_angle_deg && std::isfinite(*normal_angle_deg)) {
                ss << ", normal_angle_deg: " << *normal_angle_deg;
            }
            if (depth_absrel && std::isfinite(*depth_absrel)) {
                ss << ", depth_absrel: " << *depth_absrel;
            }
            return ss.str();
        }

        static std::string to_csv_header() {
            return "iteration,psnr,ssim,lpips,time_per_image,num_gaussians,normal_angle_deg,depth_absrel,bias_r,bias_g,bias_b,bias_corr_r,bias_corr_g,bias_corr_b,flip";
        }

        [[nodiscard]] std::string to_csv_row() const {
            std::stringstream ss;
            ss << iteration << ","
               << std::fixed << std::setprecision(6)
               << psnr << ","
               << ssim << ",";
            if (lpips && std::isfinite(*lpips)) {
                ss << *lpips;
            }
            ss << ","
               << elapsed_time << ","
               << num_gaussians << ",";
            if (normal_angle_deg && std::isfinite(*normal_angle_deg)) {
                ss << *normal_angle_deg;
            }
            ss << ",";
            if (depth_absrel && std::isfinite(*depth_absrel)) {
                ss << *depth_absrel;
            }
            ss << "," << bias_r << "," << bias_g << "," << bias_b
               << "," << bias_corr_r << "," << bias_corr_g << "," << bias_corr_b << ",";
            if (flip && std::isfinite(*flip)) {
                ss << *flip;
            }
            return ss.str();
        }
    };

    [[nodiscard]] lfs::core::Tensor image_for_metrics_and_save(const lfs::core::Tensor& image);
    // 8 or 16 quantizes to that integer grid, 32 (float references) only clamps to [0, 1].
    [[nodiscard]] lfs::core::Tensor image_for_metrics(const lfs::core::Tensor& image, int bit_depth);
    [[nodiscard]] int evaluation_bit_depth(const std::filesystem::path& reference,
                                           lfs::core::param::EvalBitDepth setting);

    struct EvaluationViewInputs {
        lfs::core::Tensor gt_image;
        lfs::core::Tensor user_mask;
        int source_width = 0;
        int source_height = 0;
        int bit_depth = 8;
    };

    struct EvaluationRenderResult {
        RenderOutput output;
        lfs::core::Tensor raw_image;
    };

    struct EvaluationRenderGeometry {
        int width = 0;
        int height = 0;
        float fx = 0.0f;
        float fy = 0.0f;
        float cx = 0.0f;
        float cy = 0.0f;
        bool undistorted = false;
    };

    struct PreparedEvaluationView {
        EvaluationViewInputs inputs;
        RenderOutput output;
        lfs::core::Tensor raw_image;
        lfs::core::Tensor metric_mask;
        EvaluationRenderGeometry render_geometry;
        bool validity_mask_applied = false;
        bool erode_ssim_mask = false;
    };

    struct EvaluationMesh {
        lfs::core::Tensor vertices; // [V,3] CUDA Float32 in the training world frame
        lfs::core::Tensor indices;  // [F,3] CUDA Int32
        float z_near = 0.0f;
        bool invert = false;
    };

    struct EvaluationPoints {
        lfs::core::Tensor means; // [N,3] CUDA Float32 in the training world frame
        int radius = 2;
        int close = 3;
        bool invert = false;
    };

    // A splat whose rendered coverage selects the evaluated pixels.
    struct EvaluationSplat {
        lfs::core::SplatData model; // CUDA, SH degree 0, in the training world frame
        float opacity = 0.85f;      // rendered opacity a pixel needs to count as covered
        bool invert = false;
    };

    struct EvaluationMaskSources {
        const EvaluationMesh* mesh = nullptr;
        const EvaluationPoints* points = nullptr;
        const lfs::io::MaskDirCache* folder = nullptr;
        const EvaluationSplat* splat = nullptr;
    };

    [[nodiscard]] lfs::Result<EvaluationMesh> load_evaluation_mesh(
        const std::filesystem::path& path, const std::array<float, 3>& training_origin, bool invert);

    // A splat PLY moved into the training frame, kept with its geometry and opacity only.
    [[nodiscard]] lfs::Result<lfs::core::SplatData> load_evaluation_splat(
        const std::filesystem::path& path, const std::array<float, 3>& training_origin);

    // Positions [N,3] of a splat or point cloud PLY, moved into the training frame.
    [[nodiscard]] lfs::Result<lfs::core::Tensor> load_evaluation_points(
        const std::filesystem::path& path, const std::array<float, 3>& training_origin);

    // Closed box from 8 corners in the training frame, corner index bits = (x, y, z) side; selected exactly like a mesh.
    [[nodiscard]] EvaluationMesh make_evaluation_box(const std::array<std::array<float, 3>, 8>& corners, bool invert);
    [[nodiscard]] std::array<std::array<float, 3>, 8> axis_aligned_box_corners(
        const std::array<float, 6>& box, const std::array<float, 3>& training_origin);

    using EvaluationRenderFn =
        std::function<lfs::Result<EvaluationRenderResult>(lfs::core::Camera&, float)>;

    [[nodiscard]] lfs::Error evaluation_error(std::string detail, lfs::core::SourceSite site);

    // SSIM counts only complete windows inside the mask; a mask without any keeps partial windows.
    [[nodiscard]] lfs::core::Tensor ssim_evaluation_mask(
        const lfs::core::Tensor& mask, bool complete_windows_only, std::string_view camera_name,
        cudaStream_t stream);

    [[nodiscard]] lfs::Result<PreparedEvaluationView> prepare_evaluation_view(
        lfs::core::Camera& camera,
        const lfs::core::param::TrainingParameters& params,
        const EvaluationRenderFn& render,
        const EvaluationViewInputs* cached_inputs = nullptr,
        lfs::io::PipelinedImageLoader* image_loader = nullptr,
        const EvaluationMaskSources& mask_sources = {},
        const lfs::core::Tensor& background = {});

    [[nodiscard]] std::optional<float> mean_normal_angle_deg(
        const lfs::core::Tensor& rendered_normal,
        const lfs::core::Tensor& prior_normal,
        const lfs::core::Tensor& rendered_alpha);

    struct DepthAbsRelSample {
        float u = 0.0f;
        float v = 0.0f;
        float true_depth = 0.0f;
    };

    [[nodiscard]] std::optional<float> median_depth_absrel(
        const lfs::core::Tensor& rendered_depth,
        const std::vector<DepthAbsRelSample>& samples);

    // Configured evaluation steps a run with this last iteration never reaches.
    [[nodiscard]] std::vector<size_t> unreachable_eval_steps(const std::vector<size_t>& eval_steps,
                                                             size_t last_iteration);

    // Adds this step's result for one image to the per-image document, which maps
    // image names to their size and evaluations. An earlier entry for the same
    // step and split is replaced; entries stay sorted by step.
    [[nodiscard]] nlohmann::json add_view_evaluation(nlohmann::json document, const ViewMetrics& view, int step,
                                                     std::string_view split);

    // Metrics reporter class
    class MetricsReporter {
    public:
        explicit MetricsReporter(const std::filesystem::path& output_dir);

        void add_metrics(const EvalMetrics& metrics);

        // Adds every evaluated image to <output>/per_image_metrics.json.
        void write_view_evaluations(const EvalMetrics& metrics, std::string_view split) const;

        void write_training_config(const lfs::core::param::TrainingParameters& params) const;

        void save_report() const;

        [[nodiscard]] const std::filesystem::path& per_image_path() const { return per_image_path_; }

    private:
        const std::filesystem::path output_dir_;
        std::vector<EvalMetrics> all_metrics_;
        const std::filesystem::path csv_path_;
        const std::filesystem::path txt_path_;
        const std::filesystem::path per_image_path_;
    };

    // Main evaluator class that handles all metrics computation and visualization
    class MetricsEvaluator {
    public:
        explicit MetricsEvaluator(const lfs::core::param::TrainingParameters& params);

        using AppearanceFn =
            std::function<lfs::core::Tensor(const lfs::core::Tensor& rgb_chw, const lfs::core::Camera& cam)>;

        void set_appearance(AppearanceFn fn) { appearance_ = std::move(fn); }
        [[nodiscard]] bool has_appearance() const { return static_cast<bool>(appearance_); }

        void set_lpips_weights_path(std::optional<std::filesystem::path> path) {
            _lpips_weights_path = std::move(path);
            _lpips_metric.reset();
            _lpips_load_attempted = false;
        }

        void set_eval_mesh(EvaluationMesh mesh) { _eval_mesh = std::move(mesh); }
        [[nodiscard]] const EvaluationMesh* eval_mesh() const { return _eval_mesh ? &*_eval_mesh : nullptr; }

        void set_eval_mask_folder(std::shared_ptr<const lfs::io::MaskDirCache> folder) {
            _eval_mask_folder = std::move(folder);
        }
        void set_eval_points(EvaluationPoints points) { _eval_points = std::move(points); }
        void set_eval_splat(EvaluationSplat splat) { _eval_splat = std::move(splat); }
        [[nodiscard]] EvaluationMaskSources mask_sources() const {
            return {.mesh = eval_mesh(),
                    .points = _eval_points ? &*_eval_points : nullptr,
                    .folder = _eval_mask_folder.get(),
                    .splat = _eval_splat ? &*_eval_splat : nullptr};
        }

        void set_normal_prior_decode(const lfs::core::Camera::NormalPriorDecode& decode) {
            _normal_prior_decode = decode;
        }

        // Check if evaluation is enabled
        bool is_enabled() const { return _params.optimization.enable_eval; }

        // Configured eval steps plus the final iteration
        bool should_evaluate(int iteration, int final_iteration) const;

        // Records the configuration the run trains with next to the per-image results
        void write_training_config(const lfs::core::param::TrainingParameters& params) const;

        // Main evaluation method
        EvalMetrics evaluate(const int iteration,
                             const lfs::core::SplatData& splatData,
                             std::shared_ptr<CameraDataset> val_dataset,
                             lfs::core::Tensor& background,
                             lfs::io::PipelinedImageLoader* image_loader = nullptr);

        // Save final report
        void save_report() const {
            if (_reporter)
                _reporter->save_report();
        }

        // Print evaluation header
        void print_evaluation_header(const int iteration) const {
            std::cout << std::endl;
            std::cout << "[Evaluation at step " << iteration << "]" << std::endl;
        }

    private:
        [[nodiscard]] std::string_view evaluated_split() const {
            return _params.optimization.eval_all ? "train" : "test";
        }

        // Configuration
        const lfs::core::param::TrainingParameters _params;
        lfs::core::Camera::NormalPriorDecode _normal_prior_decode{};

        // Metrics
        std::unique_ptr<PSNR> _psnr_metric;
        std::unique_ptr<SSIM> _ssim_metric;
        std::optional<lfs::core::nn::models::Lpips> _lpips_metric;
        std::optional<std::filesystem::path> _lpips_weights_path;
        bool _lpips_load_attempted = false;
        std::unique_ptr<MetricsReporter> _reporter;
        AppearanceFn appearance_;
        std::optional<EvaluationMesh> _eval_mesh;
        std::shared_ptr<const lfs::io::MaskDirCache> _eval_mask_folder;
        std::optional<EvaluationPoints> _eval_points;
        std::optional<EvaluationSplat> _eval_splat;
    };
} // namespace lfs::training
