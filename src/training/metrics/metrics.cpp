/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "metrics.hpp"
#include "core/events.hpp"
#include "core/gpu_device_runtime.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "core/mesh_data.hpp"
#include "core/path_utils.hpp"
#include "core/provenance.hpp"
#include "core/shared_image_ops.hpp"
#include "core/splat_data.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_execution.hpp"
#include "core/tensor_image.hpp"
#include "core/tensor_label.hpp"
#include "core/tensor_upload.hpp"
#include "eval_mask.hpp"
#include "io/filesystem_utils.hpp"
#include "io/loader.hpp"
#include "io/pipelined_image_loader.hpp"
#include "lfs/training/ops/fast_services.hpp"
#include "lfs/training/ops/geometry_types.hpp"
#include "lfs/training/ops/gsplat_services.hpp"
#include "lfs/training/ops/registry.hpp"
#include "mesh_mask.hpp"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <future>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace lfs::training {

    namespace {
        // LPIPS tiles exactly, so the budget only trades speed for memory: 384 MiB evaluates 10 MP views at
        // full speed instead of reserving 1.5 GiB on top of training.
        constexpr std::size_t kEvalLpipsActivationBudget = 384ULL << 20;

        struct TensorLayoutInfo {
            int n;
            int c;
            int h;
            int w;
        };

        TensorLayoutInfo get_layout_info(const lfs::core::Tensor& t, const char* name) {
            if (t.ndim() == 3) {
                return {
                    .n = 1,
                    .c = static_cast<int>(t.shape()[0]),
                    .h = static_cast<int>(t.shape()[1]),
                    .w = static_cast<int>(t.shape()[2])};
            }
            if (t.ndim() == 4) {
                return {
                    .n = static_cast<int>(t.shape()[0]),
                    .c = static_cast<int>(t.shape()[1]),
                    .h = static_cast<int>(t.shape()[2]),
                    .w = static_cast<int>(t.shape()[3])};
            }

            throw std::runtime_error(std::string(name) + ": expected tensor rank 3 or 4");
        }

        float get_non_empty_mask_sum_or_throw(const lfs::core::Tensor& mask, const char* name) {
            const float mask_sum = mask.sum().item<float>();
            if (!std::isfinite(mask_sum) || mask_sum <= 0.0f) {
                throw std::runtime_error(std::string(name) + ": mask is empty or invalid");
            }
            return mask_sum;
        }

        void validate_mask_shape_or_throw(const lfs::core::Tensor& mask,
                                          const TensorLayoutInfo& layout,
                                          const char* name) {
            if (mask.ndim() != 2) {
                throw std::runtime_error(std::string(name) + ": expected 2D mask [H, W]");
            }

            const int mask_h = static_cast<int>(mask.shape()[0]);
            const int mask_w = static_cast<int>(mask.shape()[1]);
            if (mask_h != layout.h || mask_w != layout.w) {
                throw std::runtime_error(
                    std::string(name) + ": mask shape does not match image shape");
            }
        }
        lfs::core::Tensor expand_mask(const lfs::core::Tensor& mask,
                                      const TensorLayoutInfo& layout,
                                      int target_ndim) {
            assert(mask.ndim() == 2);
            assert(target_ndim == 3 || target_ndim == 4);
            if (target_ndim == 3) {
                return mask.unsqueeze(0).expand({layout.c, layout.h, layout.w});
            }
            return mask.unsqueeze(0).unsqueeze(0).expand({layout.n, layout.c, layout.h, layout.w});
        }

        lfs::core::Tensor image_as_float01(const lfs::core::Tensor& image) {
            return image.dtype() == lfs::core::DataType::UInt8
                       ? image.to(lfs::core::DataType::Float32) / 255.0f
                       : image;
        }

        lfs::core::Tensor mask_as_float01(const lfs::core::Tensor& mask) {
            return (mask.dtype() == lfs::core::DataType::UInt8 || mask.dtype() == lfs::core::DataType::Bool)
                       ? mask.to(lfs::core::DataType::Float32)
                       : mask;
        }

        lfs::core::Tensor mask_image(const lfs::core::Tensor& image,
                                     const lfs::core::Tensor& mask) {
            if (!mask.is_valid())
                return image;
            const auto mask_f = mask_as_float01(mask);
            const int channels = static_cast<int>(image.shape()[image.ndim() - 3]);
            const int height = static_cast<int>(image.shape()[image.ndim() - 2]);
            const int width = static_cast<int>(image.shape()[image.ndim() - 1]);
            const auto expanded = image.ndim() == 3
                                      ? mask_f.unsqueeze(0).expand({channels, height, width})
                                      : mask_f.unsqueeze(0).unsqueeze(0).expand({static_cast<int>(image.shape()[0]), channels, height, width});
            return image * expanded;
        }

        struct RestoreCameraImageDimensions {
            lfs::core::Camera& camera;
            int width;
            int height;
            bool size_loaded;
            bool changed = false;

            void set(const int new_width, const int new_height) {
                if (camera.image_width() == new_width && camera.image_height() == new_height &&
                    camera.image_size_loaded())
                    return;
                camera.set_image_dimensions(new_width, new_height);
                changed = true;
            }

            ~RestoreCameraImageDimensions() {
                if (changed)
                    camera.restore_image_dimensions(width, height, size_loaded);
            }
        };

        std::unique_ptr<lfs::io::PipelinedImageLoader> make_eval_image_loader(
            const lfs::core::param::TrainingParameters& params) {
            lfs::io::PipelinedLoaderConfig config;
            config.backend = lfs::core::default_gpu_backend();
            config.jpeg_batch_size = 1;
            config.prefetch_count = 1;
            config.output_queue_size = 1;
            config.decode_frame_ring_capacity = 2;
            config.decoder_pool_size = 1;
            config.io_threads = 1;
            config.cold_process_threads = 1;
            config.use_16bit_color = params.dataset.loading_params.use_16bit_color;
            return std::make_unique<lfs::io::PipelinedImageLoader>(config);
        }

        lfs::io::LoadParams evaluation_load_params(const lfs::core::Camera& camera,
                                                   const lfs::core::param::TrainingParameters& params) {
            lfs::io::LoadParams load_params;
            load_params.resize_factor = params.dataset.resize_factor;
            load_params.max_width = params.dataset.max_width;
            load_params.output_uint8 = !params.dataset.loading_params.use_16bit_color;
            if (params.optimization.undistort && camera.is_undistort_prepared() &&
                params.optimization.eval_space == lfs::core::param::EvalSpace::Undistorted)
                load_params.undistort = &camera.undistort_params();
            return load_params;
        }
    } // namespace

    lfs::core::Tensor image_for_metrics_and_save(const lfs::core::Tensor& image) {
        if (image.device() == lfs::core::Device::GPU && image.dtype() == lfs::core::DataType::Float32) {
            const auto* ops = training_ops(*lfs::core::gpu_backend_of(image)).training_image;
            if (ops && ops->quantize_to_8bit_grid)
                return ops->quantize_to_8bit_grid(image);
        }
        return image.clamp(0.0f, 1.0f)
            .mul(255.0f)
            .add(0.5f)
            .to(lfs::core::DataType::UInt8)
            .to(lfs::core::DataType::Float32)
            .div(255.0f)
            .contiguous();
    }

    float PSNR::compute(const lfs::core::Tensor& pred, const lfs::core::Tensor& target,
                        const lfs::core::Tensor& mask) const {
        if (pred.shape() != target.shape()) {
            throw std::runtime_error("PSNR: prediction and target must have the same shape");
        }

        const auto layout = get_layout_info(pred, "PSNR");
        const auto target_float = image_as_float01(target);

        auto squared_diff = (pred - target_float).square();

        float mse;
        if (mask.is_valid()) {
            const auto mask_f = mask_as_float01(mask);
            validate_mask_shape_or_throw(mask_f, layout, "PSNR");
            const float mask_sum = get_non_empty_mask_sum_or_throw(mask_f, "PSNR");

            const auto expanded = expand_mask(mask_f, layout, pred.ndim());
            const auto weighted_sum = (squared_diff * expanded).sum();

            const float denom = mask_sum * static_cast<float>(layout.c * layout.n);
            mse = weighted_sum.item<float>() / denom;
        } else {
            mse = squared_diff.mean().item<float>();
        }

        if (!std::isfinite(mse)) {
            throw std::runtime_error("PSNR: produced non-finite MSE");
        }

        if (mse < 1e-10f)
            mse = 1e-10f;

        return 20.0f * std::log10(data_range_ / std::sqrt(mse));
    }

    // SSIM Implementation using LibTorch-free kernels
    SSIM::SSIM(bool apply_valid_padding)
        : apply_valid_padding_(apply_valid_padding) {
    }

    float SSIM::compute(const lfs::core::Tensor& pred, const lfs::core::Tensor& target,
                        const lfs::core::Tensor& mask) {
        if (pred.shape() != target.shape()) {
            throw std::runtime_error("SSIM: prediction and target must have the same shape");
        }

        if (ops_ == nullptr) {
            const auto backend = lfs::core::default_gpu_backend();
            if (const auto reason = unavailable_training_family(backend, Family::Photometric)) {
                throw std::runtime_error(*reason);
            }
            ops_ = training_ops(backend).photometric;
        }
        if (ops_ == nullptr || ops_->metric == nullptr || ops_->create == nullptr) {
            throw std::runtime_error("SSIM: photometric ops are unavailable");
        }
        if (!saved_.backend) {
            saved_.backend = ops_->create();
        }

        if (mask.is_valid()) {
            // Match masked training semantics: no valid-padding crop, masked mean over all pixels.
            const auto layout = get_layout_info(pred, "SSIM");
            const auto mask_f = mask_as_float01(mask);
            validate_mask_shape_or_throw(mask_f, layout, "SSIM");
            const float mask_sum = get_non_empty_mask_sum_or_throw(mask_f, "SSIM");

            (void)ops_->metric(saved_, pred, target, true, false);
            auto ssim_map = saved_.ssim_map;
            assert(ssim_map.ndim() == 4);
            assert(static_cast<int>(ssim_map.shape()[2]) == layout.h);
            assert(static_cast<int>(ssim_map.shape()[3]) == layout.w);

            const auto expanded = expand_mask(mask_f, layout, 4);
            const auto weighted_sum = (ssim_map * expanded).sum();

            const float denom = mask_sum * static_cast<float>(layout.c * layout.n);
            const float masked_ssim = weighted_sum.item<float>() / denom;
            if (!std::isfinite(masked_ssim)) {
                throw std::runtime_error("SSIM: produced non-finite masked SSIM");
            }
            return masked_ssim;
        }

        auto ssim_value = ops_->metric(saved_, pred, target, false, apply_valid_padding_);
        const float value = ssim_value.mean().item<float>();
        if (!std::isfinite(value)) {
            throw std::runtime_error("SSIM: produced non-finite SSIM");
        }
        return value;
    }

    namespace {
        lfs::core::Tensor squeeze_to_hw(const lfs::core::Tensor& t) {
            if (t.ndim() == 3 && t.shape()[0] == 1) {
                return t.squeeze(0);
            }
            return t;
        }

        float bilinear_sample_hw(const float* depth, const int height, const int width,
                                 const float x, const float y) {
            const float px = std::clamp(x - 0.5f, 0.0f, static_cast<float>(std::max(width - 1, 0)));
            const float py = std::clamp(y - 0.5f, 0.0f, static_cast<float>(std::max(height - 1, 0)));
            const int x0 = static_cast<int>(px);
            const int y0 = static_cast<int>(py);
            const int x1 = std::min(x0 + 1, std::max(width - 1, 0));
            const int y1 = std::min(y0 + 1, std::max(height - 1, 0));
            const float wx = px - static_cast<float>(x0);
            const float wy = py - static_cast<float>(y0);
            const float v00 = depth[static_cast<size_t>(y0) * static_cast<size_t>(width) + static_cast<size_t>(x0)];
            const float v10 = depth[static_cast<size_t>(y0) * static_cast<size_t>(width) + static_cast<size_t>(x1)];
            const float v01 = depth[static_cast<size_t>(y1) * static_cast<size_t>(width) + static_cast<size_t>(x0)];
            const float v11 = depth[static_cast<size_t>(y1) * static_cast<size_t>(width) + static_cast<size_t>(x1)];
            // Pixels without depth (empty or outside the valid region) would pull the blend toward zero.
            if (!(v00 > 0.0f && v10 > 0.0f && v01 > 0.0f && v11 > 0.0f))
                return std::numeric_limits<float>::quiet_NaN();
            return v00 * (1.0f - wx) * (1.0f - wy) +
                   v10 * wx * (1.0f - wy) +
                   v01 * (1.0f - wx) * wy +
                   v11 * wx * wy;
        }

        std::optional<float> mean_of_finite(const std::vector<float>& values) {
            double sum = 0.0;
            size_t count = 0;
            for (const float value : values) {
                if (std::isfinite(value)) {
                    sum += static_cast<double>(value);
                    ++count;
                }
            }
            if (count == 0) {
                return std::nullopt;
            }
            return static_cast<float>(sum / static_cast<double>(count));
        }

        [[nodiscard]] bool eval_uses_masks(const lfs::core::param::MaskMode mode) {
            return mode == lfs::core::param::MaskMode::Segment ||
                   mode == lfs::core::param::MaskMode::Ignore ||
                   mode == lfs::core::param::MaskMode::SegmentAndIgnore;
        }

        [[nodiscard]] nlohmann::json json_metric(const std::optional<float>& value) {
            return value && std::isfinite(*value) ? nlohmann::json(*value) : nlohmann::json(nullptr);
        }

        void write_json_file(const std::filesystem::path& path, const nlohmann::json& document) {
            std::error_code ec;
            std::filesystem::create_directories(path.parent_path(), ec);
            auto temp_path = path;
            temp_path += ".tmp";
            {
                std::ofstream out;
                if (!lfs::core::open_file_for_write(temp_path, std::ios::out | std::ios::binary, out)) {
                    LOG_WARN("Eval: failed to open '{}'", lfs::core::path_to_utf8(temp_path));
                    return;
                }
                out << document.dump(2, ' ', false, nlohmann::json::error_handler_t::replace) << '\n';
                if (!out) {
                    LOG_WARN("Eval: failed to write '{}'", lfs::core::path_to_utf8(temp_path));
                    return;
                }
            }
            std::filesystem::rename(temp_path, path, ec);
            if (ec)
                LOG_WARN("Eval: failed to replace '{}': {}", lfs::core::path_to_utf8(path), ec.message());
        }

        [[nodiscard]] nlohmann::json read_json_object(const std::filesystem::path& path) {
            std::ifstream in;
            if (!lfs::core::open_file_for_read(path, std::ios::in | std::ios::binary, in))
                return nlohmann::json::object();
            try {
                auto document = nlohmann::json::parse(in);
                if (document.is_object())
                    return document;
            } catch (const nlohmann::json::exception& e) {
                LOG_WARN("Eval: replacing unreadable '{}' ({})", lfs::core::path_to_utf8(path), e.what());
            }
            return nlohmann::json::object();
        }
    } // namespace

    lfs::core::Tensor ssim_evaluation_mask(const lfs::core::Tensor& mask, const bool complete_windows_only,
                                           const std::string_view camera_name) {
        if (!complete_windows_only || !mask.is_valid())
            return mask;
        auto complete_windows = lfs::training::erode_metrics_mask(mask, 5);
        if (complete_windows.to(lfs::core::DataType::Float32).sum().item<float>() > 0.0f)
            return complete_windows;
        LOG_WARN("Eval: camera '{}' has no complete SSIM window inside the evaluated pixels; SSIM includes partial windows",
                 camera_name);
        return mask;
    }

    lfs::Error evaluation_error(std::string detail, const lfs::core::SourceSite site) {
        return lfs::make_error(lfs::ErrorInit{
            .code = lfs::ErrorCode::Internal,
            .domain = lfs::ErrorDomain::Training,
            .user_message = detail,
            .detail = std::move(detail),
            .detection = site,
        });
    }

    lfs::Result<EvaluationMesh> load_evaluation_mesh(
        const std::filesystem::path& path, const std::array<float, 3>& training_origin, const bool invert) {
        lfs::io::LoadOptions options;
        options.mesh_geometry_only = true;
        auto loaded = lfs::io::Loader::create()->load(path, options);
        if (!loaded)
            return evaluation_error(loaded.error().format(), LFS_SOURCE_SITE_CURRENT());
        const auto* mesh = std::get_if<std::shared_ptr<lfs::core::MeshData>>(&loaded->data);
        if (!mesh || !*mesh)
            return evaluation_error("the file does not contain a triangle mesh", LFS_SOURCE_SITE_CURRENT());
        const auto& data = **mesh;
        if (!data.indices.is_valid() || data.indices.numel() == 0)
            return evaluation_error("the mesh has no triangles", LFS_SOURCE_SITE_CURRENT());
        assert(data.vertices.ndim() == 2 && data.vertices.shape()[1] == 3);
        assert(data.indices.ndim() == 2 && data.indices.shape()[1] == 3);

        const auto positions = data.vertices.cpu().contiguous().to_vector();
        std::array<float, 3> lower{std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
                                   std::numeric_limits<float>::max()};
        std::array<float, 3> upper{std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(),
                                   std::numeric_limits<float>::lowest()};
        for (size_t i = 0; i < positions.size(); ++i) {
            lower[i % 3] = std::min(lower[i % 3], positions[i]);
            upper[i % 3] = std::max(upper[i % 3], positions[i]);
        }
        const float diagonal = std::hypot(upper[0] - lower[0], upper[1] - lower[1], upper[2] - lower[2]);

        const auto origin = lfs::core::Tensor::from_vector(
            {training_origin[0], training_origin[1], training_origin[2]},
            lfs::core::TensorShape({1, 3}), lfs::core::Device::GPU);
        return EvaluationMesh{
            .vertices = (data.vertices.to(lfs::core::Device::GPU) - origin).contiguous(),
            .indices = data.indices.to(lfs::core::Device::GPU).contiguous(),
            .z_near = std::max(1.0e-6f, 1.0e-4f * diagonal),
            .invert = invert};
    }

    lfs::Result<lfs::core::SplatData> load_evaluation_splat(
        const std::filesystem::path& path, const std::array<float, 3>& training_origin) {
        if (!lfs::io::is_gaussian_splat_ply(path))
            return evaluation_error("the file is not a splat PLY", LFS_SOURCE_SITE_CURRENT());
        auto loaded = lfs::io::Loader::create()->load(path);
        if (!loaded)
            return evaluation_error(loaded.error().format(), LFS_SOURCE_SITE_CURRENT());
        auto* const splat = std::get_if<std::shared_ptr<lfs::core::SplatData>>(&loaded->data);
        if (!splat || !*splat || (*splat)->size() == 0)
            return evaluation_error("the file does not contain splats", LFS_SOURCE_SITE_CURRENT());
        auto model = std::move(**splat);
        model.set_sh_degree(0);
        assert(model.means().ndim() == 2 && model.means().shape()[1] == 3);
        const auto origin = lfs::core::Tensor::from_vector(
            {training_origin[0], training_origin[1], training_origin[2]},
            lfs::core::TensorShape({1, 3}), model.means().device());
        model.means() = (model.means() - origin).contiguous();
        return model;
    }

    lfs::Result<lfs::core::Tensor> load_evaluation_points(
        const std::filesystem::path& path, const std::array<float, 3>& training_origin) {
        lfs::core::Tensor means;
        if (lfs::io::is_gaussian_splat_ply(path)) {
            auto loaded = lfs::io::Loader::create()->load(path);
            if (!loaded)
                return evaluation_error(loaded.error().format(), LFS_SOURCE_SITE_CURRENT());
            const auto* splat = std::get_if<std::shared_ptr<lfs::core::SplatData>>(&loaded->data);
            if (!splat || !*splat)
                return evaluation_error("the file does not contain splats", LFS_SOURCE_SITE_CURRENT());
            means = (*splat)->means();
        } else {
            auto cloud = lfs::io::load_ply_point_cloud(path);
            if (!cloud)
                return evaluation_error(cloud.error(), LFS_SOURCE_SITE_CURRENT());
            means = cloud->means;
        }
        if (!means.is_valid() || means.numel() == 0)
            return evaluation_error("the file has no points", LFS_SOURCE_SITE_CURRENT());
        assert(means.ndim() == 2 && means.shape()[1] == 3);
        const auto origin = lfs::core::Tensor::from_vector(
            {training_origin[0], training_origin[1], training_origin[2]},
            lfs::core::TensorShape({1, 3}), lfs::core::Device::GPU);
        return (means.to(lfs::core::Device::GPU).to(lfs::core::DataType::Float32) - origin).contiguous();
    }

    std::array<std::array<float, 3>, 8> axis_aligned_box_corners(
        const std::array<float, 6>& box, const std::array<float, 3>& training_origin) {
        std::array<std::array<float, 3>, 8> corners{};
        for (size_t corner = 0; corner < corners.size(); ++corner) {
            for (size_t axis = 0; axis < 3; ++axis) {
                assert(box[axis] < box[axis + 3]);
                corners[corner][axis] = box[(corner >> axis) & 1 ? axis + 3 : axis] - training_origin[axis];
            }
        }
        return corners;
    }

    EvaluationMesh make_evaluation_box(const std::array<std::array<float, 3>, 8>& corners, const bool invert) {
        std::vector<float> vertices;
        vertices.reserve(8 * 3);
        for (const auto& corner : corners)
            vertices.insert(vertices.end(), corner.begin(), corner.end());
        static const std::vector<int> faces{0, 2, 6, 0, 6, 4, 1, 3, 7, 1, 7, 5,
                                            0, 1, 5, 0, 5, 4, 2, 3, 7, 2, 7, 6,
                                            0, 1, 3, 0, 3, 2, 4, 5, 7, 4, 7, 6};
        const auto& a = corners[0];
        const auto& b = corners[7];
        const float diagonal = std::hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
        return EvaluationMesh{
            .vertices = lfs::core::Tensor::from_vector(vertices, lfs::core::TensorShape({8, 3}), lfs::core::Device::GPU),
            .indices = lfs::core::Tensor::from_vector(faces, lfs::core::TensorShape({12, 3}), lfs::core::Device::GPU),
            .z_near = std::max(1.0e-6f, 1.0e-4f * diagonal),
            .invert = invert};
    }

    lfs::Result<PreparedEvaluationView> prepare_evaluation_view(
        lfs::core::Camera& camera,
        const lfs::core::param::TrainingParameters& params,
        const EvaluationRenderFn& render,
        const EvaluationViewInputs* cached_inputs,
        lfs::io::PipelinedImageLoader* image_loader,
        const EvaluationMaskSources& mask_sources,
        const lfs::core::Tensor& background) {
        const auto* mesh = mask_sources.mesh;
        const auto* points = mask_sources.points;
        const auto* mask_folder = mask_sources.folder;
        const auto* mask_splat = mask_sources.splat;
        RestoreCameraImageDimensions restore_dimensions{
            camera, camera.image_width(), camera.image_height(), camera.image_size_loaded()};

        EvaluationViewInputs inputs;
        std::optional<lfs::core::UndistortParams> scaled_undistort;
        std::unique_ptr<lfs::io::PipelinedImageLoader> fallback_image_loader;
        try {
            const bool undistorted_reference =
                params.optimization.undistort && camera.is_undistort_prepared() &&
                params.optimization.eval_space == lfs::core::param::EvalSpace::Undistorted;
            if (cached_inputs) {
                inputs = *cached_inputs;
                if (!inputs.gt_image.is_valid() || inputs.source_width <= 0 ||
                    inputs.source_height <= 0) {
                    return evaluation_error("cached evaluation inputs are invalid", LFS_SOURCE_SITE_CURRENT());
                }
            } else {
                if (!image_loader) {
                    fallback_image_loader = make_eval_image_loader(params);
                    image_loader = fallback_image_loader.get();
                }
                inputs.gt_image = image_loader->load_image_immediate(
                    camera.image_path(), evaluation_load_params(camera, params));
                if (!inputs.gt_image.is_valid() || inputs.gt_image.ndim() != 3 ||
                    inputs.gt_image.shape()[0] != 3)
                    return evaluation_error("failed to load evaluation image", LFS_SOURCE_SITE_CURRENT());

                if (undistorted_reference) {
                    auto [source_width, source_height, source_channels] =
                        lfs::core::get_image_info(camera.image_path());
                    if (source_width <= 0 || source_height <= 0 || source_channels <= 0)
                        return evaluation_error("failed to read evaluation image dimensions", LFS_SOURCE_SITE_CURRENT());
                    inputs.source_width = source_width;
                    inputs.source_height = source_height;
                } else {
                    inputs.source_height = static_cast<int>(inputs.gt_image.shape()[1]);
                    inputs.source_width = static_cast<int>(inputs.gt_image.shape()[2]);
                }
            }

            if (params.optimization.undistort && camera.is_undistort_prepared()) {
                scaled_undistort = lfs::core::prepare_undistort_params(
                    camera.undistort_params(), inputs.source_width, inputs.source_height,
                    params.dataset.resize_factor, params.dataset.max_width);
            }

            if (undistorted_reference && scaled_undistort) {
                if (inputs.gt_image.ndim() != 3 || inputs.gt_image.shape()[0] != 3 ||
                    static_cast<int>(inputs.gt_image.shape()[1]) != scaled_undistort->dst_height ||
                    static_cast<int>(inputs.gt_image.shape()[2]) != scaled_undistort->dst_width) {
                    return evaluation_error("undistorted evaluation image does not match the training grid",
                                            LFS_SOURCE_SITE_CURRENT());
                }
            }

            if (!cached_inputs) {
                if (mask_folder) {
                    const auto mask_path = mask_folder->find(camera.image_name());
                    if (mask_path.empty())
                        return evaluation_error(std::format("no evaluation mask for '{}' in the mask folder",
                                                            camera.image_name()),
                                                LFS_SOURCE_SITE_CURRENT());
                    inputs.user_mask = camera.load_mask_file(
                        mask_path, params.dataset.resize_factor, params.dataset.max_width,
                        params.optimization.eval_mask_invert, params.optimization.mask_threshold, true,
                        undistorted_reference);
                    if (inputs.user_mask.numel() != inputs.gt_image.shape()[1] * inputs.gt_image.shape()[2])
                        return evaluation_error(std::format("evaluation mask '{}' does not match the image size",
                                                            lfs::core::path_to_utf8(mask_path)),
                                                LFS_SOURCE_SITE_CURRENT());
                } else if (!mesh && eval_uses_masks(params.optimization.mask_mode)) {
                    auto mask_config = metrics_mask_config_from(params);
                    mask_config.apply_undistortion = undistorted_reference;
                    mask_config.replace_gt_image = false;
                    const bool alpha_as_mask = params.optimization.use_alpha_as_mask &&
                                               camera.has_alpha();
                    inputs.user_mask = lfs::training::load_eval_mask(
                        &camera, inputs.gt_image, alpha_as_mask, mask_config);
                }
                if (params.optimization.mask_mode == lfs::core::param::MaskMode::None &&
                    params.optimization.use_alpha_as_mask && camera.has_alpha()) {
                    auto alpha_config = metrics_mask_config_from(params);
                    alpha_config.apply_undistortion = undistorted_reference;
                    const auto alpha = lfs::training::load_eval_alpha(camera, alpha_config);
                    if (alpha.numel() != inputs.gt_image.shape()[1] * inputs.gt_image.shape()[2])
                        return evaluation_error("evaluation alpha does not match the evaluation image",
                                                LFS_SOURCE_SITE_CURRENT());
                    const auto& color = params.optimization.bg_color;
                    inputs.gt_image = composite_over_background(
                        inputs.gt_image, alpha,
                        background.is_valid() ? background
                                              : lfs::core::Tensor::from_vector(
                                                    std::vector<float>{color[0], color[1], color[2]}, {3},
                                                    lfs::core::Device::GPU));
                }
            }

            EvaluationRenderGeometry geometry;
            std::optional<lfs::core::UndistortParams> inverse_warp;
            std::optional<lfs::core::Camera> scoped_camera;
            lfs::core::Camera* render_camera = &camera;
            int render_width = inputs.source_width;
            int render_height = inputs.source_height;
            float dilation_scale = 1.0f;
            const bool distorted_evaluation =
                scaled_undistort &&
                params.optimization.eval_space == lfs::core::param::EvalSpace::Distorted;
            const bool native_gut_evaluation = distorted_evaluation && params.optimization.raster_backend() == lfs::core::param::RasterBackendId::ThreeDGUT;
            const bool warp_to_distorted = distorted_evaluation && !native_gut_evaluation;
            // Geometric masks follow the lens the scored image was taken with: the source lens when
            // evaluating distorted images, and the camera's own lens when GUT renders it without --undistort.
            std::optional<lfs::core::UndistortParams> mask_lens;
            if (distorted_evaluation) {
                mask_lens = scaled_undistort;
            } else if (params.optimization.raster_backend() == lfs::core::param::RasterBackendId::ThreeDGUT && !camera.is_undistort_prepared() && camera.has_distortion()) {
                camera.precompute_undistortion();
                if (camera.is_undistort_precomputed())
                    mask_lens = lfs::core::prepare_undistort_params(
                        camera.undistort_params(), inputs.source_width, inputs.source_height,
                        params.dataset.resize_factor, params.dataset.max_width);
            }

            if (native_gut_evaluation) {
                const auto& native = camera.undistort_params();
                const float source_scale_x = static_cast<float>(inputs.source_width) /
                                             static_cast<float>(native.src_width);
                const float source_scale_y = static_cast<float>(inputs.source_height) /
                                             static_cast<float>(native.src_height);
                scoped_camera.emplace(
                    camera.R(), camera.T(),
                    native.src_fx * source_scale_x, native.src_fy * source_scale_y,
                    native.src_cx * source_scale_x, native.src_cy * source_scale_y,
                    camera.radial_distortion(), camera.tangential_distortion(),
                    camera.camera_model_type(), camera.image_name(), camera.image_path(),
                    camera.mask_path(), inputs.source_width, inputs.source_height,
                    camera.uid(), camera.camera_id(), camera.depth_path(), camera.normal_path());
                render_camera = &*scoped_camera;
                const auto [fx, fy, cx, cy] = render_camera->get_intrinsics();
                geometry = {
                    .width = inputs.source_width,
                    .height = inputs.source_height,
                    .fx = fx,
                    .fy = fy,
                    .cx = cx,
                    .cy = cy,
                    .undistorted = false};
            } else if (scaled_undistort) {
                constexpr int evaluation_supersample = 2;
                const int factor = warp_to_distorted ? evaluation_supersample : 1;
                scoped_camera.emplace(camera, camera.world_view_transform());
                scoped_camera->set_image_dimensions(
                    scaled_undistort->dst_width * factor,
                    scaled_undistort->dst_height * factor);
                render_camera = &*scoped_camera;
                render_width = render_camera->image_width();
                render_height = render_camera->image_height();
                const auto [fx, fy, cx, cy] = render_camera->get_intrinsics();
                if (warp_to_distorted) {
                    inverse_warp = *scaled_undistort;
                    inverse_warp->dst_width = render_width;
                    inverse_warp->dst_height = render_height;
                    inverse_warp->dst_fx = fx;
                    inverse_warp->dst_fy = fy;
                    inverse_warp->dst_cx = cx;
                    inverse_warp->dst_cy = cy;
                    dilation_scale = static_cast<float>(factor * factor);
                    geometry = {
                        .width = inputs.source_width,
                        .height = inputs.source_height,
                        .fx = inverse_warp->src_fx,
                        .fy = inverse_warp->src_fy,
                        .cx = inverse_warp->src_cx,
                        .cy = inverse_warp->src_cy,
                        .undistorted = false};
                } else {
                    geometry = {
                        .width = render_width,
                        .height = render_height,
                        .fx = fx,
                        .fy = fy,
                        .cx = cx,
                        .cy = cy,
                        .undistorted = true};
                }
            } else {
                restore_dimensions.set(inputs.source_width, inputs.source_height);
                const auto [fx, fy, cx, cy] = camera.get_intrinsics();
                geometry = {
                    .width = inputs.source_width,
                    .height = inputs.source_height,
                    .fx = fx,
                    .fy = fy,
                    .cx = cx,
                    .cy = cy,
                    .undistorted = false};
            }

            auto rendered = render(*render_camera, dilation_scale);
            if (!rendered)
                return std::move(rendered.error());
            if (!rendered->output.image.is_valid())
                return evaluation_error("evaluation render is empty", LFS_SOURCE_SITE_CURRENT());

            assert(rendered->output.image.ndim() == 3);
            assert(rendered->output.image.shape()[0] == 3);
            assert(rendered->output.image.shape()[1] == static_cast<size_t>(render_height));
            assert(rendered->output.image.shape()[2] == static_cast<size_t>(render_width));
            if (rendered->raw_image.is_valid()) {
                assert(rendered->raw_image.ndim() == 3);
                assert(rendered->raw_image.shape() == rendered->output.image.shape());
            }

            // The render may finish on another stream; everything below runs on the caller's.
            const auto consumer = lfs::core::TensorExecutionTarget::current();
            for (auto* produced : {&rendered->output.image, &rendered->raw_image, &rendered->output.alpha,
                                   &rendered->output.depth, &rendered->output.normal}) {
                if (produced->is_valid())
                    produced->set_stream(consumer);
            }

            auto metric_mask = inputs.user_mask;
            lfs::core::Tensor validity_mask;
            if (warp_to_distorted) {
                rendered->output.image = lfs::core::distort_image_to_source(
                    rendered->output.image, *inverse_warp, validity_mask,
                    rendered->output.image.stream());
                assert(validity_mask.ndim() == 2);
                assert(validity_mask.shape()[0] == static_cast<size_t>(inputs.source_height));
                assert(validity_mask.shape()[1] == static_cast<size_t>(inputs.source_width));
                if (rendered->raw_image.is_valid()) {
                    lfs::core::Tensor raw_validity;
                    rendered->raw_image = lfs::core::distort_image_to_source(
                        rendered->raw_image, *inverse_warp, raw_validity,
                        rendered->raw_image.stream());
                }
                if (rendered->output.alpha.is_valid()) {
                    rendered->output.alpha = lfs::core::distort_mask_to_source_area(
                        rendered->output.alpha, *inverse_warp,
                        rendered->output.alpha.stream());
                }
                if (rendered->output.depth.is_valid()) {
                    rendered->output.depth = lfs::core::distort_depth_to_source_area(
                        rendered->output.depth, *inverse_warp,
                        rendered->output.depth.stream());
                }
                if (rendered->output.normal.is_valid()) {
                    rendered->output.normal = lfs::core::distort_normal_to_source_area(
                        rendered->output.normal, *inverse_warp,
                        rendered->output.normal.stream());
                }
                const auto validity_float = validity_mask.to(lfs::core::DataType::Float32);
                const auto apply_validity = [&validity_float](lfs::core::Tensor& value) {
                    if (!value.is_valid())
                        return;
                    if (value.ndim() == 2) {
                        value = value * validity_float;
                    } else if (value.ndim() == 3) {
                        value = value * validity_float.unsqueeze(0).expand(
                                            {static_cast<int>(value.shape()[0]),
                                             static_cast<int>(validity_float.shape()[0]),
                                             static_cast<int>(validity_float.shape()[1])});
                    }
                };
                apply_validity(rendered->output.alpha);
                apply_validity(rendered->output.depth);
                apply_validity(rendered->output.normal);
                rendered->output.width = inputs.source_width;
                rendered->output.height = inputs.source_height;
                if (metric_mask.is_valid()) {
                    assert(metric_mask.ndim() == 2);
                    assert(metric_mask.shape() == validity_mask.shape());
                    metric_mask = (metric_mask.to(lfs::core::DataType::Float32) *
                                   validity_mask.to(lfs::core::DataType::Float32))
                                      .gt(0.5f)
                                      .to(lfs::core::DataType::UInt8)
                                      .contiguous();
                } else {
                    metric_mask = validity_mask;
                }
            }

            if (const auto range = lfs::core::param::parse_eval_mask_depth(params.optimization.eval_mask)) {
                if (!rendered->output.depth.is_valid() || !rendered->output.alpha.is_valid())
                    return evaluation_error("a depth evaluation mask needs rendered depth and alpha",
                                            LFS_SOURCE_SITE_CURRENT());
                const auto alpha = squeeze_to_hw(rendered->output.alpha);
                const auto depth = squeeze_to_hw(rendered->output.depth);
                assert(alpha.ndim() == 2 && alpha.shape() == depth.shape());
                // FastGS accumulates alpha-weighted depth; the GUT path renders expected depth.
                const auto expected = params.optimization.raster_backend() == lfs::core::param::RasterBackendId::ThreeDGUT ? depth : depth / alpha.clamp_min(1.0e-6f);
                auto coverage = (alpha.gt(0.5f) && expected.ge((*range)[0]) && expected.le((*range)[1]))
                                    .to(lfs::core::DataType::UInt8);
                if (params.optimization.eval_mask_invert)
                    coverage = coverage.eq(0).to(lfs::core::DataType::UInt8);
                metric_mask = validity_mask.is_valid()
                                  ? (coverage.to(lfs::core::DataType::Float32) *
                                     validity_mask.to(lfs::core::DataType::Float32))
                                        .gt(0.5f)
                                        .to(lfs::core::DataType::UInt8)
                                        .contiguous()
                                  : coverage.contiguous();
                if (metric_mask.to(lfs::core::DataType::Float32).sum().item<float>() <= 0.0f)
                    return evaluation_error("depth mask covers no evaluated pixel", LFS_SOURCE_SITE_CURRENT());
            }

            if (mask_splat) {
                // Rendered with the same camera and rasterizer as the scored image, so it follows the same lens.
                auto& model = const_cast<lfs::core::SplatData&>(mask_splat->model);
                auto mask_background =
                    background.is_valid() ? background : lfs::core::Tensor::zeros({3}, lfs::core::Device::GPU);
                lfs::gpu_ops::GsplatSaved mask_gsplat;
                lfs::gpu_ops::FastSaved mask_fast;
                lfs::core::Tensor alpha;
                try {
                    const auto& ops = training_ops(lfs::core::default_gpu_backend());
                    if (params.optimization.raster_backend() == lfs::core::param::RasterBackendId::ThreeDGUT) {
                        if (!ops.gsplat)
                            throw std::runtime_error(*unavailable_training_family(lfs::core::default_gpu_backend(), Family::Gsplat));
                        mask_gsplat.backend = ops.gsplat->create();
                        alpha = gsplat_infer(*ops.gsplat, mask_gsplat, *render_camera, model, mask_background).alpha;
                    } else {
                        if (!ops.fast)
                            throw std::runtime_error(*unavailable_training_family(lfs::core::default_gpu_backend(), Family::Fast));
                        mask_fast.backend = ops.fast->create();
                        alpha = fast_infer(*ops.fast, mask_fast, *render_camera, model, mask_background,
                                           params.optimization.mip_filter, {}, false, dilation_scale)
                                    .alpha;
                    }
                } catch (const std::exception& e) {
                    // LFS-CENSUS-OK(empty-catch): converted into a typed evaluation error
                    return evaluation_error(std::string("splat mask render failed: ") + e.what(),
                                            LFS_SOURCE_SITE_CURRENT());
                }
                if (!alpha.is_valid())
                    return evaluation_error("splat mask render has no alpha", LFS_SOURCE_SITE_CURRENT());
                alpha.set_stream(consumer);
                if (warp_to_distorted)
                    alpha = lfs::core::distort_mask_to_source_area(alpha, *inverse_warp, alpha.stream());
                auto coverage = squeeze_to_hw(alpha).ge(mask_splat->opacity).to(lfs::core::DataType::UInt8);
                if (mask_splat->invert)
                    coverage = coverage.eq(0).to(lfs::core::DataType::UInt8);
                metric_mask = validity_mask.is_valid()
                                  ? (coverage.to(lfs::core::DataType::Float32) *
                                     validity_mask.to(lfs::core::DataType::Float32))
                                        .gt(0.5f)
                                        .to(lfs::core::DataType::UInt8)
                                        .contiguous()
                                  : coverage.contiguous();
                if (metric_mask.to(lfs::core::DataType::Float32).sum().item<float>() <= 0.0f)
                    return evaluation_error("splat mask covers no evaluated pixel", LFS_SOURCE_SITE_CURRENT());
            }

            if (points) {
                const auto stream = rendered->output.image.stream();
                const auto R = camera.R().cpu().contiguous().to_vector();
                const auto T = camera.T().cpu().contiguous().to_vector();
                MeshMaskCamera point_camera;
                point_camera.world_to_camera = {R[0], R[1], R[2], T[0],
                                                R[3], R[4], R[5], T[1],
                                                R[6], R[7], R[8], T[2]};
                if (mask_lens) {
                    point_camera.fx = mask_lens->src_fx;
                    point_camera.fy = mask_lens->src_fy;
                    point_camera.cx = mask_lens->src_cx;
                    point_camera.cy = mask_lens->src_cy;
                    point_camera.width = inputs.source_width;
                    point_camera.height = inputs.source_height;
                } else {
                    point_camera.fx = geometry.fx;
                    point_camera.fy = geometry.fy;
                    point_camera.cx = geometry.cx;
                    point_camera.cy = geometry.cy;
                    point_camera.width = geometry.width;
                    point_camera.height = geometry.height;
                }
                auto coverage = splat_point_coverage(points->means, point_camera, points->radius, points->close,
                                                     mask_lens ? &*mask_lens : nullptr, stream);
                if (points->invert)
                    coverage = coverage.eq(0).to(lfs::core::DataType::UInt8);
                metric_mask = validity_mask.is_valid()
                                  ? (coverage.to(lfs::core::DataType::Float32) *
                                     validity_mask.to(lfs::core::DataType::Float32))
                                        .gt(0.5f)
                                        .to(lfs::core::DataType::UInt8)
                                        .contiguous()
                                  : coverage;
                if (metric_mask.to(lfs::core::DataType::Float32).sum().item<float>() <= 0.0f)
                    return evaluation_error("point mask covers no evaluated pixel", LFS_SOURCE_SITE_CURRENT());
            }

            if (mesh) {
                const auto stream = rendered->output.image.stream();
                const auto R = camera.R().cpu().contiguous().to_vector();
                const auto T = camera.T().cpu().contiguous().to_vector();
                MeshMaskCamera mesh_camera;
                mesh_camera.world_to_camera = {R[0], R[1], R[2], T[0],
                                               R[3], R[4], R[5], T[1],
                                               R[6], R[7], R[8], T[2]};
                lfs::core::Tensor coverage;
                if (mask_lens) {
                    mesh_camera.fx = mask_lens->src_fx;
                    mesh_camera.fy = mask_lens->src_fy;
                    mesh_camera.cx = mask_lens->src_cx;
                    mesh_camera.cy = mask_lens->src_cy;
                    mesh_camera.width = inputs.source_width;
                    mesh_camera.height = inputs.source_height;
                    const auto samples = lfs::core::inverse_distortion_sample_map(*mask_lens, stream);
                    coverage = rasterize_mesh_coverage(
                        mesh->vertices, mesh->indices, mesh_camera, samples, *mask_lens,
                        mesh->z_near, stream);
                } else {
                    mesh_camera.fx = geometry.fx;
                    mesh_camera.fy = geometry.fy;
                    mesh_camera.cx = geometry.cx;
                    mesh_camera.cy = geometry.cy;
                    mesh_camera.width = geometry.width;
                    mesh_camera.height = geometry.height;
                    coverage = rasterize_mesh_coverage(
                        mesh->vertices, mesh->indices, mesh_camera, mesh->z_near, stream);
                }
                if (mesh->invert)
                    coverage = coverage.eq(0).to(lfs::core::DataType::UInt8);
                metric_mask = validity_mask.is_valid()
                                  ? (coverage.to(lfs::core::DataType::Float32) *
                                     validity_mask.to(lfs::core::DataType::Float32))
                                        .gt(0.5f)
                                        .to(lfs::core::DataType::UInt8)
                                        .contiguous()
                                  : coverage;
                if (metric_mask.to(lfs::core::DataType::Float32).sum().item<float>() <= 0.0f)
                    return evaluation_error("mesh mask covers no evaluated pixel", LFS_SOURCE_SITE_CURRENT());
            }

            if (rendered->raw_image.is_valid())
                rendered->raw_image = rendered->raw_image.clamp(0.0f, 1.0f);
            rendered->output.image = image_for_metrics_and_save(rendered->output.image);

            assert(inputs.gt_image.ndim() == 3);
            assert(inputs.gt_image.shape()[0] == 3);
            assert(rendered->output.image.shape() == inputs.gt_image.shape());
            if (rendered->raw_image.is_valid()) {
                assert(rendered->raw_image.ndim() == 3);
                assert(rendered->raw_image.shape() == inputs.gt_image.shape());
            }
            if (metric_mask.is_valid()) {
                assert(metric_mask.ndim() == 2);
                assert(metric_mask.shape()[0] == inputs.gt_image.shape()[1]);
                assert(metric_mask.shape()[1] == inputs.gt_image.shape()[2]);
            }

            return PreparedEvaluationView{
                .inputs = std::move(inputs),
                .output = std::move(rendered->output),
                .raw_image = std::move(rendered->raw_image),
                .metric_mask = std::move(metric_mask),
                .render_geometry = geometry,
                .validity_mask_applied = warp_to_distorted,
                .erode_ssim_mask = warp_to_distorted || mesh != nullptr || points != nullptr || mask_folder != nullptr ||
                                   mask_splat != nullptr ||
                                   lfs::core::param::is_eval_mask_depth(params.optimization.eval_mask)};
        } catch (const std::exception& e) {
            // LFS-CENSUS-OK(empty-catch): converted into a typed evaluation error
            return evaluation_error(e.what(), LFS_SOURCE_SITE_CURRENT());
        }
    }

    std::optional<float> mean_normal_angle_deg(
        const lfs::core::Tensor& rendered_normal,
        const lfs::core::Tensor& prior_normal,
        const lfs::core::Tensor& rendered_alpha) {
        if (!rendered_normal.is_valid() || !prior_normal.is_valid() || !rendered_alpha.is_valid()) {
            return std::nullopt;
        }
        if (rendered_normal.ndim() != 3 || rendered_normal.shape()[0] != 3 ||
            prior_normal.ndim() != 3 || prior_normal.shape()[0] != 3) {
            return std::nullopt;
        }
        const int height = static_cast<int>(rendered_normal.shape()[1]);
        const int width = static_cast<int>(rendered_normal.shape()[2]);
        if (height <= 0 || width <= 0) {
            return std::nullopt;
        }
        if (prior_normal.shape()[1] != static_cast<size_t>(height) ||
            prior_normal.shape()[2] != static_cast<size_t>(width)) {
            return std::nullopt;
        }

        auto alpha = squeeze_to_hw(rendered_alpha);
        if (alpha.ndim() != 2 ||
            alpha.shape()[0] != static_cast<size_t>(height) ||
            alpha.shape()[1] != static_cast<size_t>(width)) {
            return std::nullopt;
        }

        const auto rendered_cpu = rendered_normal.cpu().contiguous();
        const auto prior_cpu = prior_normal.cpu().contiguous();
        const auto alpha_cpu = alpha.cpu().contiguous();
        if (rendered_cpu.dtype() != lfs::core::DataType::Float32 ||
            prior_cpu.dtype() != lfs::core::DataType::Float32 ||
            alpha_cpu.dtype() != lfs::core::DataType::Float32) {
            return std::nullopt;
        }

        const float* const rendered = rendered_cpu.ptr<float>();
        const float* const prior = prior_cpu.ptr<float>();
        const float* const alpha_ptr = alpha_cpu.ptr<float>();
        const size_t hw = static_cast<size_t>(height) * static_cast<size_t>(width);

        double sum_deg = 0.0;
        size_t count = 0;
        constexpr float kRad2Deg = 57.29577951308232f;
        for (size_t i = 0; i < hw; ++i) {
            const float a = alpha_ptr[i];
            if (!std::isfinite(a) || a <= kernels::kNormalLossMinAlpha) {
                continue;
            }
            const float tx = prior[i];
            const float ty = prior[hw + i];
            const float tz = prior[2 * hw + i];
            const float nx = rendered[i];
            const float ny = rendered[hw + i];
            const float nz = rendered[2 * hw + i];
            if (!std::isfinite(tx) || !std::isfinite(ty) || !std::isfinite(tz) ||
                !std::isfinite(nx) || !std::isfinite(ny) || !std::isfinite(nz)) {
                continue;
            }
            const float t_norm = std::sqrt(tx * tx + ty * ty + tz * tz);
            const float n_norm = std::sqrt(nx * nx + ny * ny + nz * nz);
            if (t_norm < kernels::kNormalLossMinPriorNorm ||
                n_norm < kernels::kNormalLossMinRenderNorm) {
                continue;
            }
            float cos_ang = (tx * nx + ty * ny + tz * nz) / (t_norm * n_norm);
            cos_ang = std::clamp(cos_ang, -1.0f, 1.0f);
            sum_deg += static_cast<double>(std::acos(cos_ang) * kRad2Deg);
            ++count;
        }
        if (count == 0) {
            return std::nullopt;
        }
        return static_cast<float>(sum_deg / static_cast<double>(count));
    }

    std::optional<float> median_depth_absrel(
        const lfs::core::Tensor& rendered_depth,
        const std::vector<DepthAbsRelSample>& samples) {
        if (!rendered_depth.is_valid() || samples.empty()) {
            return std::nullopt;
        }
        auto depth = squeeze_to_hw(rendered_depth);
        if (depth.ndim() != 2) {
            return std::nullopt;
        }
        const int height = static_cast<int>(depth.shape()[0]);
        const int width = static_cast<int>(depth.shape()[1]);
        if (height <= 0 || width <= 0) {
            return std::nullopt;
        }

        const auto depth_cpu = depth.cpu().contiguous();
        if (depth_cpu.dtype() != lfs::core::DataType::Float32) {
            return std::nullopt;
        }
        const float* const depth_ptr = depth_cpu.ptr<float>();

        std::vector<float> errors;
        errors.reserve(samples.size());
        for (const auto& sample : samples) {
            if (!std::isfinite(sample.true_depth) || sample.true_depth <= 1.0e-6f ||
                !std::isfinite(sample.u) || !std::isfinite(sample.v)) {
                continue;
            }
            if (sample.u < 0.0f || sample.v < 0.0f ||
                sample.u > static_cast<float>(width) ||
                sample.v > static_cast<float>(height)) {
                continue;
            }
            const float rendered = bilinear_sample_hw(depth_ptr, height, width, sample.u, sample.v);
            if (!std::isfinite(rendered) || rendered <= 0.0f) {
                continue;
            }
            errors.push_back(std::abs(rendered - sample.true_depth) / sample.true_depth);
        }
        if (errors.empty()) {
            return std::nullopt;
        }
        const size_t mid = errors.size() / 2;
        std::nth_element(errors.begin(), errors.begin() + static_cast<std::ptrdiff_t>(mid), errors.end());
        if (errors.size() % 2 == 0 && mid > 0) {
            const float upper = errors[mid];
            const float lower = *std::max_element(errors.begin(), errors.begin() + static_cast<std::ptrdiff_t>(mid));
            return 0.5f * (lower + upper);
        }
        return errors[mid];
    }

    // MetricsReporter Implementation
    MetricsReporter::MetricsReporter(const std::filesystem::path& output_dir)
        : output_dir_(output_dir),
          csv_path_(output_dir_ / "metrics.csv"),
          txt_path_(output_dir_ / "metrics_report.txt"),
          per_image_path_(output_dir_ / "per_image_metrics.json") {
        // Create CSV header if file doesn't exist
        if (!std::filesystem::exists(csv_path_)) {
            std::ofstream csv_file;
            if (lfs::core::open_file_for_write(csv_path_, csv_file)) {
                csv_file << EvalMetrics{}.to_csv_header() << std::endl;
                csv_file.close();
            }
        }
    }

    std::vector<size_t> unreachable_eval_steps(const std::vector<size_t>& eval_steps, const size_t last_iteration) {
        std::vector<size_t> unreachable;
        std::ranges::copy_if(eval_steps, std::back_inserter(unreachable),
                             [last_iteration](const size_t step) { return step > last_iteration; });
        return unreachable;
    }

    nlohmann::json add_view_evaluation(nlohmann::json document, const ViewMetrics& view, const int step,
                                       const std::string_view split) {
        if (!document.is_object())
            document = nlohmann::json::object();
        auto& record = document[view.image_name];
        if (!record.is_object())
            record = nlohmann::json::object();
        if (view.width > 0 && view.height > 0) {
            record["width"] = view.width;
            record["height"] = view.height;
        }
        nlohmann::json entry = {
            {"step", step},
            {"split", split},
            {"psnr", json_metric(view.psnr)},
            {"ssim", json_metric(view.ssim)},
            {"lpips", json_metric(view.lpips)},
            {"masked", view.masked},
            {"evaluated_pixel_fraction", view.evaluated_pixel_fraction},
            {"validity_mask_applied", view.validity_mask_applied},
        };
        if (!view.skipped_reason.empty())
            entry["skipped_reason"] = view.skipped_reason;

        std::vector<nlohmann::json> evaluations;
        if (const auto existing = record.find("evaluations");
            existing != record.end() && existing->is_array()) {
            for (auto& previous : *existing) {
                if (!previous.is_object())
                    continue;
                const auto previous_step = previous.find("step");
                if (previous_step == previous.end() || !previous_step->is_number_integer())
                    continue;
                const auto previous_split = previous.find("split");
                const bool same_split = previous_split != previous.end() && previous_split->is_string() &&
                                        previous_split->get<std::string>() == split;
                if (previous_step->get<int>() != step || !same_split)
                    evaluations.push_back(std::move(previous));
            }
        }
        evaluations.push_back(std::move(entry));
        std::ranges::stable_sort(evaluations, {}, [](const nlohmann::json& e) { return e.at("step").get<int>(); });
        record["evaluations"] = std::move(evaluations);
        return document;
    }

    void MetricsReporter::add_metrics(const EvalMetrics& metrics) {
        all_metrics_.push_back(metrics);

        // Append to CSV immediately
        std::ofstream csv_file;
        if (lfs::core::open_file_for_write(csv_path_, std::ios::app, csv_file)) {
            csv_file << metrics.to_csv_row() << std::endl;
            csv_file.close();
        }
    }

    void MetricsReporter::write_view_evaluations(const EvalMetrics& metrics, const std::string_view split) const {
        if (metrics.views.empty())
            return;
        auto document = read_json_object(per_image_path_);
        for (const auto& view : metrics.views)
            document = add_view_evaluation(std::move(document), view, metrics.iteration, split);
        write_json_file(per_image_path_, document);
    }

    void MetricsReporter::write_training_config(const lfs::core::param::TrainingParameters& params) const {
        std::error_code ec;
        std::filesystem::create_directories(output_dir_, ec);
        if (const auto saved = lfs::core::param::save_training_parameters_to_json(
                params, output_dir_ / "training_config.json");
            !saved) {
            LOG_WARN("Eval: failed to write the training configuration: {}", saved.error());
        }
    }

    void MetricsReporter::save_report() const {
        std::ofstream report_file;
        if (!lfs::core::open_file_for_write(txt_path_, report_file)) {
            std::cerr << "Failed to open report file: " << lfs::core::path_to_utf8(txt_path_) << std::endl;
            return;
        }

        // Write header
        report_file << "==============================================\n";
        report_file << "3D Gaussian Splatting Evaluation Report\n";
        report_file << "==============================================\n";
        report_file << "Output Directory: " << lfs::core::path_to_utf8(output_dir_) << "\n";

        // Get current time
        const auto now = std::chrono::system_clock::now();
        const auto time_t_val = std::chrono::system_clock::to_time_t(now);
        std::tm tm{};
#ifdef _WIN32
        localtime_s(&tm, &time_t_val);
#else
        localtime_r(&time_t_val, &tm);
#endif
        report_file << "Generated: " << std::put_time(&tm, "%Y-%m-%d %H:%M:%S") << "\n\n";

        // Summary statistics
        if (!all_metrics_.empty()) {
            report_file << "Summary Statistics:\n";
            report_file << "------------------\n";

            // Find best metrics
            const auto best_psnr = std::max_element(all_metrics_.begin(), all_metrics_.end(),
                                                    [](const EvalMetrics& a, const EvalMetrics& b) {
                                                        return a.psnr < b.psnr;
                                                    });
            const auto best_ssim = std::max_element(all_metrics_.begin(), all_metrics_.end(),
                                                    [](const EvalMetrics& a, const EvalMetrics& b) {
                                                        return a.ssim < b.ssim;
                                                    });

            report_file << std::fixed << std::setprecision(4);
            report_file << "Best PSNR:  " << best_psnr->psnr << " (at iteration " << best_psnr->iteration << ")\n";
            report_file << "Best SSIM:  " << best_ssim->ssim << " (at iteration " << best_ssim->iteration << ")\n";

            // Final metrics
            const auto& final = all_metrics_.back();
            report_file << "\nFinal Metrics (iteration " << final.iteration << "):\n";
            report_file << "PSNR:  " << final.psnr << "\n";
            report_file << "SSIM:  " << final.ssim << "\n";
            if (final.lpips && std::isfinite(*final.lpips))
                report_file << "LPIPS: " << *final.lpips << "\n";
            report_file << "Time per image: " << final.elapsed_time << " seconds\n";
            report_file << "Number of Gaussians: " << final.num_gaussians << "\n";
            report_file << "Bias (raw):  (" << final.bias_r << ", " << final.bias_g << ", " << final.bias_b << ")\n";
            report_file << "Bias (corrected):  (" << final.bias_corr_r << ", " << final.bias_corr_g << ", "
                        << final.bias_corr_b << ")\n";
            if (final.normal_angle_deg && std::isfinite(*final.normal_angle_deg)) {
                report_file << "Normal angle (deg): " << *final.normal_angle_deg << "\n";
            }
            if (final.depth_absrel && std::isfinite(*final.depth_absrel)) {
                report_file << "Depth AbsRel: " << *final.depth_absrel << "\n";
            }
        }

        // Detailed results
        report_file << "\nDetailed Results:\n";
        report_file << "-----------------\n";
        report_file << std::setw(10) << "Iteration"
                    << std::setw(10) << "PSNR"
                    << std::setw(10) << "SSIM"
                    << std::setw(10) << "LPIPS"
                    << std::setw(15) << "Time(s/img)"
                    << std::setw(15) << "#Gaussians"
                    << "\n";
        report_file << std::string(60, '-') << "\n";

        for (const auto& m : all_metrics_) {
            report_file << std::setw(10) << m.iteration
                        << std::setw(10) << std::fixed << std::setprecision(4) << m.psnr
                        << std::setw(10) << m.ssim;
            if (m.lpips && std::isfinite(*m.lpips)) {
                report_file << std::setw(10) << *m.lpips;
            } else {
                report_file << std::setw(10) << "";
            }
            report_file << std::setw(15) << m.elapsed_time
                        << std::setw(15) << m.num_gaussians << "\n";
        }

        report_file.close();
        std::cout << "Evaluation report saved to: " << lfs::core::path_to_utf8(txt_path_) << std::endl;
        std::cout << "Metrics CSV saved to: " << lfs::core::path_to_utf8(csv_path_) << std::endl;
        if (std::filesystem::exists(per_image_path_))
            std::cout << "Per-image metrics JSON saved to: " << lfs::core::path_to_utf8(per_image_path_) << std::endl;
    }

    // MetricsEvaluator Implementation
    MetricsEvaluator::MetricsEvaluator(const lfs::core::param::TrainingParameters& params)
        : _params(params) {
        if (!params.optimization.enable_eval) {
            return;
        }

        // Initialize metrics
        _psnr_metric = std::make_unique<PSNR>(1.0f);
        _ssim_metric = std::make_unique<SSIM>(true); // apply_valid_padding = true

        // Initialize reporter
        _reporter = std::make_unique<MetricsReporter>(params.dataset.output_path);
    }

    void MetricsEvaluator::write_training_config(const lfs::core::param::TrainingParameters& params) const {
        if (_reporter)
            _reporter->write_training_config(params);
    }

    bool MetricsEvaluator::should_evaluate(const int iteration, const int final_iteration) const {
        if (!_params.optimization.enable_eval)
            return false;
        if (iteration == final_iteration)
            return true;

        return std::find(_params.optimization.eval_steps.cbegin(), _params.optimization.eval_steps.cend(), iteration) !=
               _params.optimization.eval_steps.cend();
    }

    EvalMetrics MetricsEvaluator::evaluate(const int iteration,
                                           const lfs::core::SplatData& splatData,
                                           std::shared_ptr<CameraDataset> val_dataset,
                                           lfs::core::Tensor& background,
                                           lfs::io::PipelinedImageLoader* image_loader) {
        if (!_params.optimization.enable_eval) {
            throw std::runtime_error("Evaluation is not enabled");
        }
        if (const auto reason = unavailable_training_family(
                lfs::core::default_gpu_backend(), Family::Photometric)) {
            throw std::runtime_error(*reason);
        }
        lfs::core::TensorLabelScope evaluation_label("eval.workspace");

        std::unique_ptr<lfs::io::PipelinedImageLoader> fallback_image_loader;
        if (!image_loader) {
            fallback_image_loader = make_eval_image_loader(_params);
            image_loader = fallback_image_loader.get();
        }

        EvalMetrics result;
        result.num_gaussians = static_cast<int>(splatData.size());
        result.iteration = iteration;

        std::vector<float> psnr_values, ssim_values, lpips_values, normal_values, depth_values;
        std::vector<float> bias_r_values, bias_g_values, bias_b_values;
        std::vector<float> bias_corr_r_values, bias_corr_g_values, bias_corr_b_values;
        const auto start_time = std::chrono::steady_clock::now();

        // Create directory for evaluation images
        const std::filesystem::path eval_dir = _params.dataset.output_path /
                                               ("eval_step_" + std::to_string(iteration));
        if (_params.optimization.enable_save_eval_images) {
            std::filesystem::create_directories(eval_dir);
        }

        const size_t val_dataset_size = val_dataset->size();
        size_t skipped_images = 0;
        size_t evaluated_images = 0;
        size_t saved_images = 0;
        std::optional<std::pair<int, int>> lpips_preflight_size;
        bool lpips_allocator_trimmed = false;
        double lpips_elapsed_ms = 0.0;
        std::size_t lpips_timed_images = 0;

        if (!_lpips_load_attempted && _lpips_weights_path) {
            _lpips_load_attempted = true;
            const auto& weights_path = *_lpips_weights_path;
            try {
                auto loaded = lfs::core::nn::models::Lpips::load(
                    weights_path, lfs::core::Device::GPU, lfs::core::DataType::Float16,
                    lfs::core::nn::models::InputScaling::Identity, kEvalLpipsActivationBudget);
                if (loaded) {
                    const auto backend = lfs::core::default_gpu_backend();
                    const auto* lpips = training_ops(backend).lpips;
                    if (!lpips)
                        throw std::runtime_error(*unavailable_training_family(backend, Family::Lpips));
                    loaded->set_dispatch(*lpips);
                    _lpips_metric.emplace(std::move(*loaded));
                } else {
                    LOG_WARN("Eval: LPIPS unavailable at '{}' ({})",
                             lfs::core::path_to_utf8(weights_path), loaded.error().detail());
                }
            } catch (const std::exception& e) {
                LOG_WARN("Eval: LPIPS unavailable at '{}' ({})",
                         lfs::core::path_to_utf8(weights_path), e.what());
            }
        }
        bool render_normal = false;
        if (_params.optimization.raster_backend() == lfs::core::param::RasterBackendId::ThreeDGS) {
            for (size_t image_idx = 0; image_idx < val_dataset_size; ++image_idx) {
                if (val_dataset->get_camera(image_idx)->has_normal()) {
                    render_normal = true;
                    break;
                }
            }
        }

        result.views.reserve(val_dataset_size);
        for (size_t image_idx = 0; image_idx < val_dataset_size; ++image_idx) {
            lfs::core::Camera* cam = val_dataset->get_camera(image_idx);
            auto& view = result.views.emplace_back();
            view.index = static_cast<int>(image_idx);
            view.image_name = cam->image_name();
            auto& splatData_mutable = const_cast<lfs::core::SplatData&>(splatData);
            auto prepared = prepare_evaluation_view(
                *cam, _params,
                [&](lfs::core::Camera& render_camera, const float dilation_scale)
                    -> lfs::Result<EvaluationRenderResult> {
                    try {
                        RenderOutput output;
                        if (_params.optimization.raster_backend() == lfs::core::param::RasterBackendId::ThreeDGUT) {
                            if (_gsplat_ops == nullptr) {
                                _gsplat_ops = lfs::training::training_ops(lfs::core::default_gpu_backend()).gsplat;
                            }
                            if (_gsplat_ops == nullptr) {
                                throw std::runtime_error(*lfs::training::unavailable_training_family(
                                    lfs::core::default_gpu_backend(), lfs::training::Family::Gsplat));
                            }
                            auto& saved = _gsplat_saved != nullptr ? *_gsplat_saved : _gsplat_owned;
                            if (!saved.backend)
                                saved.backend = _gsplat_ops->create();
                            output = gsplat_infer(*_gsplat_ops, saved, render_camera, splatData_mutable, background, 1.0f, false,
                                                  lfs::core::param::is_eval_mask_depth(_params.optimization.eval_mask) ? GsplatRenderMode::RGB_ED : GsplatRenderMode::RGB);
                        } else {
                            if (_fast_ops == nullptr) {
                                _fast_ops = lfs::training::training_ops(lfs::core::default_gpu_backend()).fast;
                            }
                            if (_fast_ops == nullptr) {
                                throw std::runtime_error(*lfs::training::unavailable_training_family(
                                    lfs::core::default_gpu_backend(), lfs::training::Family::Fast));
                            }
                            lfs::gpu_ops::FastSaved& saved = _fast_saved != nullptr ? *_fast_saved : _fast_owned;
                            if (!saved.backend) {
                                saved.backend = _fast_ops->create();
                            }
                            output = lfs::training::fast_infer(
                                *_fast_ops, saved, render_camera, splatData_mutable, background,
                                _params.optimization.mip_filter, {}, render_normal, dilation_scale);
                        }
                        auto raw_image = output.image;
                        if (appearance_ && output.image.is_valid())
                            output.image = appearance_(output.image, render_camera);
                        return EvaluationRenderResult{
                            .output = std::move(output),
                            .raw_image = std::move(raw_image)};
                    } catch (const std::exception& e) {
                        // LFS-CENSUS-OK(empty-catch): converted into a typed evaluation error
                        return evaluation_error(e.what(), LFS_SOURCE_SITE_CURRENT());
                    }
                },
                nullptr,
                image_loader,
                mask_sources(),
                background);
            // The next reference decodes on the host while this view is scored; its upload waits for its turn.
            if (image_idx + 1 < val_dataset_size) {
                const auto* const next = val_dataset->get_camera(image_idx + 1);
                image_loader->decode_ahead(next->image_path(), evaluation_load_params(*next, _params));
            }
            if (!prepared) {
                LOG_WARN("Eval: skipping camera '{}' (view preparation failed: {})",
                         cam->image_name(), prepared.error().detail());
                view.skipped_reason = std::format("view preparation failed: {}", prepared.error().detail());
                skipped_images++;
                continue;
            }

            auto gt_image = prepared->inputs.gt_image;
            auto mask = prepared->metric_mask;
            auto r_output = std::move(prepared->output);
            auto render_raw = std::move(prepared->raw_image);
            const auto render_geometry = prepared->render_geometry;
            const bool erode_ssim_mask = prepared->erode_ssim_mask;
            const bool distorted_coordinates =
                _params.optimization.undistort && cam->is_undistort_prepared() &&
                _params.optimization.eval_space == lfs::core::param::EvalSpace::Distorted;
            view.height = static_cast<int>(gt_image.shape()[1]);
            view.width = static_cast<int>(gt_image.shape()[2]);
            view.validity_mask_applied = prepared->validity_mask_applied;
            view.evaluated_pixel_fraction = mask.is_valid()
                                                ? mask.to(lfs::core::DataType::Float32).mean().item<float>()
                                                : 1.0f;

            float psnr = 0.0f;
            float ssim = 0.0f;
            try {
                psnr = _psnr_metric->compute(r_output.image, gt_image, mask);
                ssim = _ssim_metric->compute(
                    r_output.image, gt_image,
                    ssim_evaluation_mask(mask, erode_ssim_mask, cam->image_name()));
            } catch (const std::exception& e) {
                LOG_WARN("Eval: skipping camera '{}' (metric computation failed: {})", cam->image_name(), e.what());
                view.skipped_reason = std::string("metric computation failed: ") + e.what();
                skipped_images++;
                continue;
            }

            if (!std::isfinite(psnr) || !std::isfinite(ssim)) {
                LOG_WARN("Eval: skipping camera '{}' (non-finite metric values: PSNR={}, SSIM={})",
                         cam->image_name(), psnr, ssim);
                view.skipped_reason = "non-finite metric values";
                skipped_images++;
                continue;
            }

            psnr_values.push_back(psnr);
            ssim_values.push_back(ssim);
            view.psnr = psnr;
            view.ssim = ssim;
            view.masked = mask.is_valid();
            evaluated_images++;

            auto gt_float = image_as_float01(gt_image);
            if (gt_image.dtype() != lfs::core::DataType::UInt8)
                gt_float = gt_float.clamp(0.0f, 1.0f);
            gt_float.set_name("eval.gt_float");
            std::optional<float> lpips;
            std::function<void()> deferred_lpips;
            std::future<void> pending_lpips;
            const auto launch_lpips = [&] {
                if (!deferred_lpips)
                    return;
                auto work = std::exchange(deferred_lpips, {});
                try {
                    pending_lpips = std::async(std::launch::async, work);
                } catch (const std::system_error&) {
                    pending_lpips = std::async(std::launch::deferred, std::move(work));
                }
            };
            if (_lpips_metric) {
                try {
                    const auto pred_lpips = mask_image(r_output.image, mask);
                    const auto target_lpips = mask_image(
                        gt_float.to(lfs::core::Device::GPU), mask);
                    const int image_height = static_cast<int>(gt_image.shape()[1]);
                    const int image_width = static_cast<int>(gt_image.shape()[2]);
                    const std::pair<int, int> image_size{image_height, image_width};
                    const bool size_changed = !lpips_preflight_size || *lpips_preflight_size != image_size;
                    lpips_preflight_size = image_size;
                    const auto required = _lpips_metric->estimated_peak_bytes(image_height, image_width, mask.is_valid());
                    const std::size_t free_bytes =
                        lfs::core::gpu_backend_memory_info(lfs::core::default_gpu_backend()).free_bytes;
                    const bool lpips_preflight_ok = free_bytes >= required && free_bytes != 0;
                    if (!lpips_preflight_ok && size_changed) {
                        const auto shortfall = required > free_bytes ? required - free_bytes : 0;
                        LOG_WARN("Eval: LPIPS skipped for this image size; tile={} required={} free={} shortfall={} bytes",
                                 _lpips_metric->tile_size_for(image_height, image_width), required,
                                 free_bytes, shortfall);
                    } else if (lpips_preflight_ok && size_changed) {
                        LOG_DEBUG("Eval: LPIPS preflight passed; tile={} required={} free={} bytes",
                                  _lpips_metric->tile_size_for(image_height, image_width), required, free_bytes);
                    }
                    if (lpips_preflight_ok) {
                        if (!lpips_allocator_trimmed) {
                            // PSNR/SSIM scalar readback has completed their GPU work.
                            // Return their free blocks before the much larger LPIPS
                            // workspace establishes the terminal evaluation peak.
                            lfs::core::gpu_trim_cached_memory(
                                lfs::core::default_gpu_backend());
                            lpips_allocator_trimmed = true;
                        }
                        auto evaluate_lpips = [&, pred_lpips, target_lpips] {
                            lfs::core::TensorLabelScope lpips_label("eval.lpips");
                            const auto lpips_wall_start = std::chrono::steady_clock::now();
                            auto value = mask.is_valid()
                                             ? _lpips_metric->forward(pred_lpips, target_lpips, mask,
                                                                      lfs::core::nn::models::InputScaling::Identity)
                                             : _lpips_metric->forward(pred_lpips, target_lpips,
                                                                      lfs::core::nn::models::InputScaling::Identity);
                            if (value && std::isfinite(*value)) {
                                const auto elapsed_ms = std::chrono::duration<float, std::milli>(
                                                            std::chrono::steady_clock::now() - lpips_wall_start)
                                                            .count();
                                if (std::isfinite(elapsed_ms)) {
                                    lpips_elapsed_ms += elapsed_ms;
                                    lpips_timed_images++;
                                }
                                lpips = *value;
                                lpips_values.push_back(*value);
                            } else if (!value) {
                                LOG_WARN("Eval: LPIPS failed for camera '{}' ({})", cam->image_name(),
                                         value.error().detail());
                            } else {
                                LOG_WARN("Eval: LPIPS produced a non-finite value for camera '{}'",
                                         cam->image_name());
                            }
                        };
                        if (_lpips_metric->prefers_independent_queue() &&
                            render_normal && cam->has_normal()) {
                            // Materialize on the producer queue before sharing with the worker.
                            (void)pred_lpips.data_ptr();
                            (void)target_lpips.data_ptr();
                            deferred_lpips = [evaluate_lpips = std::move(evaluate_lpips),
                                              backend = *lfs::core::gpu_backend_of(pred_lpips)] {
                                const lfs::core::GpuBackendScope backend_scope(backend);
                                lfs::core::TensorWorkQueue queue(backend);
                                const lfs::core::TensorWorkQueue::Scope scope(queue);
                                evaluate_lpips();
                            };
                        } else {
                            evaluate_lpips();
                        }
                    }
                } catch (const std::exception& e) {
                    LOG_WARN("Eval: LPIPS failed for camera '{}' ({})", cam->image_name(), e.what());
                }
            }
            view.lpips = lpips;
            auto accumulate_bias = [&](const lfs::core::Tensor& image,
                                       std::vector<float>& br,
                                       std::vector<float>& bg,
                                       std::vector<float>& bb) {
                if (!image.is_valid() || image.shape() != gt_float.shape() ||
                    image.ndim() != 3 || image.shape()[0] != 3) {
                    return;
                }
                const auto difference = image - gt_float;
                const auto channel_mean =
                    mask.is_valid()
                        ? mask_image(difference, mask).sum({1, 2}) / mask_as_float01(mask).sum().item<float>()
                        : difference.mean({1, 2});
                const auto channel_cpu = channel_mean.cpu().contiguous();
                const float* const bias = channel_cpu.ptr<float>();
                br.push_back(bias[0]);
                bg.push_back(bias[1]);
                bb.push_back(bias[2]);
            };
            accumulate_bias(render_raw, bias_r_values, bias_g_values, bias_b_values);
            accumulate_bias(r_output.image, bias_corr_r_values, bias_corr_g_values, bias_corr_b_values);

            if (render_normal && cam->has_normal()) {
                try {
                    if (!r_output.normal.is_valid() || r_output.normal.numel() == 0) {
                        LOG_DEBUG("Eval: normal_angle_deg skipped for '{}': rendered normal is empty",
                                  cam->image_name());
                    } else {
                        RestoreCameraImageDimensions restore_normal_dimensions{
                            *cam, cam->image_width(), cam->image_height(),
                            cam->image_size_loaded()};
                        cam->release_normal_cache();
                        if (distorted_coordinates)
                            restore_normal_dimensions.set(view.width, view.height);
                        lfs::core::Tensor prior = cam->load_and_get_normal(
                            _params.dataset.resize_factor,
                            _params.dataset.max_width,
                            _normal_prior_decode,
                            !distorted_coordinates);
                        cam->release_normal_cache();
                        if (!prior.is_valid() || prior.ndim() != 3 || prior.shape()[0] != 3) {
                            LOG_DEBUG("Eval: normal_angle_deg skipped for '{}': prior normal is missing",
                                      cam->image_name());
                        } else {
                            const int render_h = static_cast<int>(r_output.normal.shape()[1]);
                            const int render_w = static_cast<int>(r_output.normal.shape()[2]);
                            if (static_cast<int>(prior.shape()[1]) != render_h ||
                                static_cast<int>(prior.shape()[2]) != render_w) {
                                prior = lfs::training::training_ops(lfs::core::default_gpu_backend()).shared_image->resize(prior, render_h, render_w, lfs::gpu_ops::Resample::LanczosFloatCHW, 2);
                            }
                            std::optional<float> angle;
                            if (deferred_lpips) {
                                // Finish the normal readbacks before LPIPS occupies the GPU.
                                // The unchanged CPU reduction can then run alongside LPIPS.
                                const auto rendered_cpu = r_output.normal.cpu().contiguous();
                                const auto prior_cpu = prior.cpu().contiguous();
                                const auto alpha_cpu = r_output.alpha.cpu().contiguous();
                                launch_lpips();
                                angle = mean_normal_angle_deg(rendered_cpu, prior_cpu, alpha_cpu);
                            } else {
                                angle = mean_normal_angle_deg(r_output.normal, prior, r_output.alpha);
                            }
                            if (angle)
                                normal_values.push_back(*angle);
                            if (_params.optimization.enable_save_eval_images &&
                                std::getenv("LFS_EVAL_SAVE_NORMALS")) {
                                const std::vector<lfs::core::Tensor> normal_maps = {
                                    r_output.normal.clamp(-1.0f, 1.0f).mul(0.5f) + 0.5f,
                                    prior.clamp(-1.0f, 1.0f).mul(0.5f) + 0.5f};
                                lfs::core::image_io::save_images_async(
                                    eval_dir / (std::to_string(image_idx) + "_normals.png"),
                                    normal_maps,
                                    true, // horizontal: rendered | prior
                                    4,
                                    lfs::core::provenance_to_json(
                                        lfs::core::make_minimal_provenance_stamp()));
                            }
                        }
                    }
                } catch (const std::exception& e) {
                    LOG_DEBUG("Eval: normal_angle_deg skipped for '{}': {}", cam->image_name(), e.what());
                }
            }

            try {
                const auto& observations = cam->sfm_observations();
                if (!observations.empty() &&
                    r_output.depth.is_valid() &&
                    r_output.depth.numel() > 0 &&
                    cam->camera_width() > 0 &&
                    cam->camera_height() > 0) {
                    auto depth = squeeze_to_hw(r_output.depth);
                    if (depth.ndim() == 2) {
                        const int depth_h = static_cast<int>(depth.shape()[0]);
                        const int depth_w = static_cast<int>(depth.shape()[1]);
                        assert(depth_h == render_geometry.height);
                        assert(depth_w == render_geometry.width);
                        const int observation_width = distorted_coordinates
                                                          ? cam->undistort_params().src_width
                                                          : cam->camera_width();
                        const int observation_height = distorted_coordinates
                                                           ? cam->undistort_params().src_height
                                                           : cam->camera_height();
                        const float u_scale = static_cast<float>(depth_w) /
                                              static_cast<float>(observation_width);
                        const float v_scale = static_cast<float>(depth_h) /
                                              static_cast<float>(observation_height);
                        auto R_cpu = cam->R().cpu().contiguous();
                        auto T_cpu = cam->T().cpu().contiguous();
                        const float* const R = R_cpu.ptr<float>();
                        const float* const T = T_cpu.ptr<float>();
                        // Observed pixels live in the distorted source image; the undistorted
                        // render is sampled at the point's projection through its own camera.
                        std::vector<DepthAbsRelSample> samples;
                        samples.reserve(observations.size());
                        for (const auto& observation : observations) {
                            const float z = R[6] * observation.x + R[7] * observation.y +
                                            R[8] * observation.z + T[2];
                            if (!std::isfinite(z) || z <= 1.0e-6f) {
                                continue;
                            }
                            if (render_geometry.undistorted) {
                                const float x = R[0] * observation.x + R[1] * observation.y +
                                                R[2] * observation.z + T[0];
                                const float y = R[3] * observation.x + R[4] * observation.y +
                                                R[5] * observation.z + T[1];
                                samples.push_back(DepthAbsRelSample{
                                    .u = render_geometry.fx * x / z + render_geometry.cx,
                                    .v = render_geometry.fy * y / z + render_geometry.cy,
                                    .true_depth = z});
                                continue;
                            }
                            samples.push_back(DepthAbsRelSample{
                                .u = observation.u * u_scale,
                                .v = observation.v * v_scale,
                                .true_depth = z});
                        }
                        if (const auto absrel = median_depth_absrel(depth, samples)) {
                            depth_values.push_back(*absrel);
                        }
                    }
                }
            } catch (const std::exception& e) {
                LOG_DEBUG("Eval: depth_absrel skipped for '{}': {}", cam->image_name(), e.what());
            }

            launch_lpips();
            if (pending_lpips.valid()) {
                try {
                    pending_lpips.get();
                } catch (const std::exception& e) {
                    LOG_WARN("Eval: LPIPS failed for camera '{}' ({})", cam->image_name(), e.what());
                }
                view.lpips = lpips;
            }

            if (_params.optimization.enable_save_eval_images) {
                auto gt_vis = image_as_float01(gt_image);
                auto render_vis = r_output.image;
                if (mask.is_valid()) {
                    auto mask_f = mask_as_float01(mask);
                    const int C = static_cast<int>(gt_image.shape()[0]);
                    const int H = static_cast<int>(mask_f.shape()[0]);
                    const int W = static_cast<int>(mask_f.shape()[1]);
                    auto mask_3d = mask_f.unsqueeze(0).expand({C, H, W});
                    gt_vis = gt_vis * mask_3d;
                    render_vis = r_output.image * mask_3d;
                }
                // The saver prepares one CPU grid synchronously before enqueueing, so
                // retaining two extra full-resolution GPU clones only raises eval peak.
                const std::vector<lfs::core::Tensor> rgb_images = {gt_vis, render_vis};
                auto stamp = _params.include_provenance ? lfs::core::make_provenance_stamp()
                                                        : lfs::core::make_minimal_provenance_stamp();
                if (_params.include_provenance) {
                    stamp.iteration = iteration;
                    const auto strategy = lfs::core::param::canonical_strategy_name(_params.optimization.strategy);
                    if (!strategy.empty())
                        stamp.strategy = std::string(strategy);
                }
                lfs::core::image_io::save_images_async(
                    eval_dir / (std::to_string(image_idx) + ".png"),
                    rgb_images,
                    true, // horizontal
                    4,    // separator width
                    lfs::core::provenance_to_json(stamp));
                if (view.validity_mask_applied) {
                    lfs::core::image_io::save_image_async(
                        eval_dir / (std::to_string(image_idx) + "_metric_mask.png"),
                        mask.to(lfs::core::DataType::Float32)
                            .unsqueeze(0)
                            .expand({3, view.height, view.width}));
                }
                saved_images++;
            }
        }

        // Wait for queued eval PNGs before returning so callers can read them
        // without racing the BatchImageSaver workers.
        if (_params.optimization.enable_save_eval_images) {
            lfs::core::image_io::wait_for_pending_saves();
        }

        if (lpips_timed_images > 0) {
            LOG_DEBUG("Eval: LPIPS-only {:.3f} ms/image over {} images",
                      lpips_elapsed_ms / static_cast<double>(lpips_timed_images), lpips_timed_images);
        }

        if (_lpips_metric) {
            _lpips_metric->release_activations();
            lfs::core::Tensor::trim_memory_pool();
        }
        const auto end_time = std::chrono::steady_clock::now();
        const auto elapsed = std::chrono::duration<float>(end_time - start_time).count();

        // Compute averages
        if (!psnr_values.empty()) {
            result.psnr = std::accumulate(psnr_values.begin(), psnr_values.end(), 0.0f) / psnr_values.size();
            result.ssim = std::accumulate(ssim_values.begin(), ssim_values.end(), 0.0f) / ssim_values.size();
        }
        result.lpips = mean_of_finite(lpips_values);
        if (!bias_r_values.empty()) {
            const auto n = static_cast<float>(bias_r_values.size());
            result.bias_r = std::accumulate(bias_r_values.begin(), bias_r_values.end(), 0.0f) / n;
            result.bias_g = std::accumulate(bias_g_values.begin(), bias_g_values.end(), 0.0f) / n;
            result.bias_b = std::accumulate(bias_b_values.begin(), bias_b_values.end(), 0.0f) / n;
        }
        if (!bias_corr_r_values.empty()) {
            const auto n = static_cast<float>(bias_corr_r_values.size());
            result.bias_corr_r = std::accumulate(bias_corr_r_values.begin(), bias_corr_r_values.end(), 0.0f) / n;
            result.bias_corr_g = std::accumulate(bias_corr_g_values.begin(), bias_corr_g_values.end(), 0.0f) / n;
            result.bias_corr_b = std::accumulate(bias_corr_b_values.begin(), bias_corr_b_values.end(), 0.0f) / n;
        }
        result.normal_angle_deg = mean_of_finite(normal_values);
        result.depth_absrel = mean_of_finite(depth_values);
        const size_t elapsed_denom = evaluated_images > 0 ? evaluated_images : std::max<size_t>(val_dataset_size, 1);
        result.elapsed_time = elapsed / static_cast<float>(elapsed_denom);

        if (skipped_images > 0) {
            LOG_WARN("Eval: skipped {} / {} images due to mask/metric failures", skipped_images, val_dataset_size);
        }
        if (evaluated_images == 0) {
            LOG_WARN("Eval: no images were successfully evaluated at iteration {}", iteration);
            _reporter->write_view_evaluations(result, evaluated_split());
            return result;
        }

        result.valid = true;

        // Emit evaluation completed event for GUI display and other subscribers.
        lfs::core::events::state::EvaluationCompleted{
            .iteration = result.iteration,
            .psnr = result.psnr,
            .ssim = result.ssim,
            .lpips = result.lpips,
            .elapsed_time = result.elapsed_time,
            .num_gaussians = result.num_gaussians}
            .emit();

        // Add metrics to reporter
        _reporter->add_metrics(result);
        _reporter->write_view_evaluations(result, evaluated_split());

        if (_params.optimization.enable_save_eval_images) {
            std::cout << "Saved " << saved_images << " evaluation images to: " << lfs::core::path_to_utf8(eval_dir) << std::endl;
        }

        return result;
    }
} // namespace lfs::training
