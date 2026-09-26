/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "../dataset.hpp"
#include "core/nn/models/lpips.hpp"
#include "core/parameters.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "lfs/training/ops/loss.hpp"
#include <cmath>
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

        void set_ops(const lfs::gpu_ops::PhotometricOps* ops) { ops_ = ops; }

        float compute(const lfs::core::Tensor& pred, const lfs::core::Tensor& target,
                      const lfs::core::Tensor& mask = {});

    private:
        bool apply_valid_padding_;
        const lfs::gpu_ops::PhotometricOps* ops_ = nullptr;
        lfs::gpu_ops::PhotoSaved saved_{};
    };

    struct ViewMetrics {
        int index = 0;
        std::string image_name;
        int width = 0;
        int height = 0;
        std::optional<float> psnr;
        std::optional<float> ssim;
        std::optional<float> lpips;
        bool masked = false;
        std::string skipped_reason;
    };

    struct EvalMetrics {
        float psnr = 0.0f;
        float ssim = 0.0f;
        std::optional<float> lpips;
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
            return "iteration,psnr,ssim,lpips,time_per_image,num_gaussians,normal_angle_deg,depth_absrel,bias_r,bias_g,bias_b,bias_corr_r,bias_corr_g,bias_corr_b";
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
               << "," << bias_corr_r << "," << bias_corr_g << "," << bias_corr_b;
            return ss.str();
        }
    };

    [[nodiscard]] lfs::core::Tensor image_for_metrics_and_save(const lfs::core::Tensor& image);

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

        void set_photometric(const lfs::gpu_ops::PhotometricOps* ops) {
            if (_ssim_metric) {
                _ssim_metric->set_ops(ops);
            }
        }

        void set_appearance(AppearanceFn fn) { appearance_ = std::move(fn); }
        [[nodiscard]] bool has_appearance() const { return static_cast<bool>(appearance_); }

        void set_lpips_weights_path(std::optional<std::filesystem::path> path) {
            _lpips_weights_path = std::move(path);
            _lpips_metric.reset();
            _lpips_load_attempted = false;
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
                             lfs::core::Tensor& background);

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

        // Helper functions
        lfs::core::Tensor load_eval_mask(lfs::core::Camera* cam, lfs::core::Tensor& gt_image,
                                         bool alpha_as_mask) const;
    };
} // namespace lfs::training
