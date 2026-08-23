/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "../dataset.hpp"
#include "core/parameters.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
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

        float compute(const lfs::core::Tensor& pred, const lfs::core::Tensor& target,
                      const lfs::core::Tensor& mask = {});

    private:
        bool apply_valid_padding_;
    };

    // Evaluation result structure (no LPIPS)
    struct EvalMetrics {
        float psnr = 0.0f;
        float ssim = 0.0f;
        float elapsed_time = 0.0f;
        int num_gaussians = 0;
        int iteration = 0;
        bool valid = false;
        std::optional<float> normal_angle_deg;
        std::optional<float> depth_absrel;

        [[nodiscard]] std::string to_string() const {
            if (!valid) {
                return "No valid evaluation images";
            }
            std::stringstream ss;
            ss << std::fixed << std::setprecision(4);
            ss << "PSNR: " << psnr
               << ", SSIM: " << ssim
               << ", Time: " << elapsed_time << "s/image"
               << ", #GS: " << num_gaussians;
            if (normal_angle_deg && std::isfinite(*normal_angle_deg)) {
                ss << ", normal_angle_deg: " << *normal_angle_deg;
            }
            if (depth_absrel && std::isfinite(*depth_absrel)) {
                ss << ", depth_absrel: " << *depth_absrel;
            }
            return ss.str();
        }

        static std::string to_csv_header() {
            return "iteration,psnr,ssim,time_per_image,num_gaussians,normal_angle_deg,depth_absrel";
        }

        [[nodiscard]] std::string to_csv_row() const {
            std::stringstream ss;
            ss << iteration << ","
               << std::fixed << std::setprecision(6)
               << psnr << ","
               << ssim << ","
               << elapsed_time << ","
               << num_gaussians << ",";
            if (normal_angle_deg && std::isfinite(*normal_angle_deg)) {
                ss << *normal_angle_deg;
            }
            ss << ",";
            if (depth_absrel && std::isfinite(*depth_absrel)) {
                ss << *depth_absrel;
            }
            return ss.str();
        }
    };

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

    // Metrics reporter class
    class MetricsReporter {
    public:
        explicit MetricsReporter(const std::filesystem::path& output_dir);

        void add_metrics(const EvalMetrics& metrics);

        void save_report() const;

    private:
        const std::filesystem::path output_dir_;
        std::vector<EvalMetrics> all_metrics_;
        const std::filesystem::path csv_path_;
        const std::filesystem::path txt_path_;
    };

    // Main evaluator class that handles all metrics computation and visualization
    class MetricsEvaluator {
    public:
        explicit MetricsEvaluator(const lfs::core::param::TrainingParameters& params);

        using AppearanceFn =
            std::function<lfs::core::Tensor(const lfs::core::Tensor& rgb_chw, const lfs::core::Camera& cam)>;

        void set_appearance(AppearanceFn fn) { appearance_ = std::move(fn); }
        [[nodiscard]] bool has_appearance() const { return static_cast<bool>(appearance_); }

        void set_normal_prior_decode(const lfs::core::Camera::NormalPriorDecode& decode) {
            _normal_prior_decode = decode;
        }


        // Check if evaluation is enabled
        bool is_enabled() const { return _params.optimization.enable_eval; }

        // Check if we should evaluate at this iteration
        bool should_evaluate(const int iteration) const;

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
        // Configuration
        const lfs::core::param::TrainingParameters _params;
        lfs::core::Camera::NormalPriorDecode _normal_prior_decode{};

        // Metrics
        std::unique_ptr<PSNR> _psnr_metric;
        std::unique_ptr<SSIM> _ssim_metric;
        std::unique_ptr<MetricsReporter> _reporter;
        AppearanceFn appearance_;

        // Helper functions
        lfs::core::Tensor load_eval_mask(lfs::core::Camera* cam, lfs::core::Tensor& gt_image,
                                         bool alpha_as_mask) const;
    };
} // namespace lfs::training
