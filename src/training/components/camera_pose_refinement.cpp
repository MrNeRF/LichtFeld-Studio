/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "camera_pose_refinement.hpp"
#include "core/error.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/scene.hpp"
#include "core/splat_data.hpp"
#include "io/cache_image_loader.hpp"
#include "io/formats/colmap.hpp"
#include "lfs/training/camera_pose_math.hpp"
#include "lfs/training/ops/fast_services.hpp"
#include "lfs/training/ops/loss.hpp"
#include "lfs/training/ops/registry.hpp"
#include "rasterization/fast_rasterizer.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <fstream>
#include <stdexcept>

namespace lfs::training {

    namespace {
        using lfs::core::Device;
        using lfs::core::Tensor;

        constexpr double kRadToDeg = 57.295779513082321;
        // Adam step size per camera update at full rate.
        constexpr float kRotationLr = static_cast<float>(0.01 / kRadToDeg);
        constexpr float kTranslationLr = 5e-4f;
        constexpr float kHalvings = 7.0f;
        constexpr int kRefitSteps = 200;

        float pose_rate(const int iteration, const int start, const int hold, const int last) {
            if (iteration < start || iteration > last)
                return 0.0f;
            if (iteration <= hold)
                return 1.0f;
            const float half_life = std::max(1.0f, static_cast<float>(last - hold) / kHalvings);
            return std::exp2(-static_cast<float>(iteration - hold) / half_life);
        }

        // W0^T v: a vector in the start camera's axes, in world axes.
        std::array<double, 3> to_world(const float* w2c0, const float* v) {
            std::array<double, 3> out{};
            for (int j = 0; j < 3; ++j)
                out[j] = double(w2c0[j]) * v[0] + double(w2c0[4 + j]) * v[1] + double(w2c0[8 + j]) * v[2];
            return out;
        }

        double norm3(const float* v) {
            return std::sqrt(double(v[0]) * v[0] + double(v[1]) * v[1] + double(v[2]) * v[2]);
        }

        lfs::Error pose_error(const lfs::ErrorCode code, std::string message, const lfs::core::SourceSite site) {
            return lfs::make_error(lfs::ErrorInit{
                .code = code,
                .domain = lfs::ErrorDomain::Training,
                .user_message = message,
                .detail = std::move(message),
                .detection = site,
            });
        }

        // Moves the camera to the row-major 4x4 world-to-camera pose w2c.
        void set_camera_pose(lfs::core::Camera& camera, const float* w2c) {
            camera.set_pose(
                Tensor::from_vector({w2c[0], w2c[1], w2c[2], w2c[4], w2c[5], w2c[6], w2c[8], w2c[9], w2c[10]}, {3, 3},
                                    Device::CPU),
                Tensor::from_vector({w2c[3], w2c[7], w2c[11]}, {3}, Device::CPU));
        }

        struct Spread {
            double median = 0.0, p90 = 0.0, max = 0.0;
        };
        Spread spread(std::vector<double> v) {
            if (v.empty())
                return {};
            std::sort(v.begin(), v.end());
            return {v[v.size() / 2], v[std::min(v.size() - 1, v.size() * 9 / 10)], v.back()};
        }
    } // namespace

    CameraPoseOffsets::CameraPoseOffsets(std::vector<std::shared_ptr<lfs::core::Camera>> cameras)
        : cameras_(std::move(cameras)) {
        const size_t n = cameras_.size();
        start_host_.reserve(n * 16);
        for (size_t row = 0; row < n; ++row) {
            const auto& camera = *cameras_[row];
            row_of_uid_[camera.uid()] = static_cast<int>(row);
            const auto w2c = camera.world_view_transform().cpu().contiguous().to_vector();
            if (w2c.size() != 16)
                throw std::runtime_error(std::format("camera '{}' has no 4x4 world-to-camera transform", camera.image_name()));
            start_host_.insert(start_host_.end(), w2c.begin(), w2c.end());
        }
        offsets_.resize(n * 6);
        adam_m_.resize(n * 6);
        adam_v_.resize(n * 6);
        steps_.resize(n);
        grad_w2c_ = Tensor::zeros({16}, Device::GPU);
        grad_cam_position_ = Tensor::zeros({3}, Device::GPU);
    }

    int CameraPoseOffsets::row_of(const lfs::core::Camera& camera) const {
        const auto it = row_of_uid_.find(camera.uid());
        if (it == row_of_uid_.end())
            throw std::logic_error(std::format("camera '{}' has no pose offset", camera.image_name()));
        return it->second;
    }

    void CameraPoseOffsets::step(const int row, const float lr_rotation, const float lr_translation) {
        const auto grad_matrix = grad_w2c_.cpu();
        const auto grad_position = grad_cam_position_.cpu();
        float* offset = offsets_.data() + row * 6;
        float gradient[6];
        camera_pose::offset_grad(start_pose(row), offset, grad_matrix.ptr<float>(), grad_position.ptr<float>(), gradient);
        if (std::all_of(std::begin(gradient), std::end(gradient), [](float v) { return std::isfinite(v); })) {
            const float t = ++steps_[row];
            const float bias1 = 1.f - std::pow(0.9f, t), bias2 = 1.f - std::pow(0.999f, t);
            for (int k = 0; k < 6; ++k) {
                const size_t i = row * 6 + k;
                adam_m_[i] = 0.9f * adam_m_[i] + (1.f - 0.9f) * gradient[k];
                adam_v_[i] = 0.999f * adam_v_[i] + (1.f - 0.999f) * gradient[k] * gradient[k];
                const float rate = k < 3 ? lr_rotation : lr_translation;
                offset[k] -= rate * (adam_m_[i] / bias1) / (std::sqrt(adam_v_[i] / bias2) + 1e-8f);
            }
        }
        std::vector<float> matrix(16), position(3);
        camera_pose::compose(start_pose(row), offset, matrix.data(), position.data());
        cameras_[row]->set_device_pose(Tensor::from_vector(matrix, {4, 4}, Device::CPU),
                                       Tensor::from_vector(position, {3}, Device::CPU));
    }

    CameraPoseOffsets::HostState CameraPoseOffsets::download() const {
        return {offsets_, steps_};
    }

    void CameraPoseOffsets::commit() {
        const auto state = download();
        for (size_t row = 0; row < cameras_.size(); ++row) {
            const float* off = state.offsets.data() + 6 * row;
            if (std::none_of(off, off + 6, [](const float x) { return x != 0.0f; }))
                continue;
            float w2c[16], centre[3];
            camera_pose::compose(start_pose(static_cast<int>(row)), off, w2c, centre);
            set_camera_pose(*cameras_[row], w2c);
        }
    }

    void CameraPoseOffsets::restore_start_poses() {
        for (size_t row = 0; row < cameras_.size(); ++row)
            set_camera_pose(*cameras_[row], start_pose(static_cast<int>(row)));
    }

    lfs::Result<std::unique_ptr<CameraPoseRefinement>> CameraPoseRefinement::create(
        std::vector<std::shared_ptr<lfs::core::Camera>> cameras, const int iterations) {
        if (cameras.empty())
            return pose_error(lfs::ErrorCode::FailedPrecondition, "--pose-opt: there are no training cameras",
                              LFS_SOURCE_SITE_CURRENT());
        const auto& ops = training_ops(*lfs::core::gpu_backend_of(cameras.front()->world_view_transform()));
        if (!ops.fast || !ops.fast->camera_gradient)
            return pose_error(lfs::ErrorCode::Unsupported, "Pose refinement requires raster camera gradients on this device",
                              LFS_SOURCE_SITE_CURRENT());
        for (const auto& camera : cameras) {
            if (camera->camera_model_type() != lfs::core::CameraModelType::PINHOLE && !camera->is_undistort_prepared())
                return pose_error(lfs::ErrorCode::Unsupported,
                                  std::format("--pose-opt needs pinhole cameras or --undistort ('{}' is not a pinhole camera)",
                                              camera->image_name()),
                                  LFS_SOURCE_SITE_CURRENT());
        }
        return std::make_unique<CameraPoseRefinement>(std::move(cameras), iterations);
    }

    CameraPoseRefinement::CameraPoseRefinement(std::vector<std::shared_ptr<lfs::core::Camera>> cameras,
                                               const int iterations)
        : offsets_(std::move(cameras)),
          iterations_(iterations),
          start_(iterations / 10),
          hold_(iterations / 2) {
        LOG_INFO("Pose refinement: {} training cameras, from iteration {}, full rate until {}, 1/128 of it at {}",
                 offsets_.cameras().size(), start_, hold_, iterations_);
    }

    float CameraPoseRefinement::rate(const int iteration) const {
        return pose_rate(iteration, start_, hold_, iterations_);
    }

    void CameraPoseRefinement::bind_gradients(FastGSFusedExtraGradients& extra, const int iteration) {
        if (rate(iteration) <= 0.0f)
            return;
        extra.grad_w2c = offsets_.grad_w2c();
        extra.grad_cam_position = offsets_.grad_cam_position();
    }

    void CameraPoseRefinement::step(const lfs::core::Camera& camera, const int iteration) {
        const float r = rate(iteration);
        if (r <= 0.0f)
            return;
        offsets_.step(offsets_.row_of(camera), kRotationLr * r, kTranslationLr * r);
    }

    bool CameraPoseRefinement::reports_at(const int iteration) const {
        return iteration >= start_ && (iteration == iterations_ || iteration % std::max(1, iterations_ / 15) == 0);
    }

    std::string CameraPoseRefinement::summary(const int iteration) const {
        const auto state = offsets_.download();
        std::vector<double> moves, turns;
        for (size_t row = 0; row < offsets_.cameras().size(); ++row) {
            if (state.steps[row] <= 0.0f)
                continue;
            const float* off = state.offsets.data() + 6 * row;
            turns.push_back(norm3(off) * kRadToDeg);
            moves.push_back(norm3(off + 3));
        }
        const auto move = spread(moves);
        const auto turn = spread(turns);
        return std::format("Pose refinement at {}: rate x{:.3g}, {} of {} cameras updated; move median {:.5f} / p90 {:.5f} / "
                           "max {:.5f} units, turn median {:.4f} / p90 {:.4f} / max {:.4f} deg",
                           iteration, rate(iteration), moves.size(), offsets_.cameras().size(), move.median, move.p90,
                           move.max, turn.median, turn.p90, turn.max);
    }

    lfs::Status CameraPoseRefinement::write_table(const std::filesystem::path& path) const {
        const auto state = offsets_.download();
        std::ofstream out;
        if (!lfs::core::open_file_for_write(path, std::ios::out | std::ios::binary, out))
            return lfs::Status::failure(pose_error(lfs::ErrorCode::PermissionDenied,
                                                   std::format("cannot write '{}'", lfs::core::path_to_utf8(path)),
                                                   LFS_SOURCE_SITE_CURRENT()));
        out << "name\tupdates\tmove_units\tturn_deg\tmove_x\tmove_y\tmove_z\tturn_x_deg\tturn_y_deg\tturn_z_deg\n";
        for (size_t row = 0; row < offsets_.cameras().size(); ++row) {
            const float* off = state.offsets.data() + 6 * row;
            const float* start = offsets_.start_pose(static_cast<int>(row));
            const auto move = to_world(start, off + 3);
            const auto turn = to_world(start, off);
            out << std::format("{}\t{:.0f}\t{:.7f}\t{:.6f}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.6f}\t{:.6f}\t{:.6f}\n",
                               offsets_.cameras()[row]->image_name(), state.steps[row], norm3(off + 3),
                               norm3(off) * kRadToDeg, move[0], move[1], move[2], turn[0] * kRadToDeg,
                               turn[1] * kRadToDeg, turn[2] * kRadToDeg);
        }
        if (!out)
            return lfs::Status::failure(pose_error(lfs::ErrorCode::DataLoss,
                                                   std::format("writing '{}' failed", lfs::core::path_to_utf8(path)),
                                                   LFS_SOURCE_SITE_CURRENT()));
        return {};
    }

    RefittedCameraPoses::RefittedCameraPoses(std::vector<std::shared_ptr<lfs::core::Camera>> cameras,
                                             lfs::core::SplatData& model,
                                             lfs::core::Tensor& background,
                                             const Settings& settings)
        : offsets_(std::move(cameras)) {
        try {
            refit(model, background, settings);
        } catch (...) {
            offsets_.restore_start_poses();
            throw;
        }
    }

    void RefittedCameraPoses::refit(lfs::core::SplatData& model, lfs::core::Tensor& background,
                                    const Settings& settings) {
        const auto& ops = training_ops(*lfs::core::gpu_backend_of(model.means_raw()));
        if (!ops.fast || !ops.fast->camera_gradient || !ops.photometric)
            throw std::runtime_error("Camera refit requires raster camera gradients and photometric loss");
        lfs::gpu_ops::FastSaved frame{ops.fast->create()};
        lfs::gpu_ops::PhotoSaved photo{.backend = ops.photometric->create()};
        auto& loader = lfs::io::CacheLoader::getInstance();
        for (size_t row = 0; row < offsets_.cameras().size(); ++row) {
            auto& camera = *offsets_.cameras()[row];
            if (!camera.image_size_loaded())
                camera.load_image_size(settings.resize_factor, settings.max_width);
            const auto target = loader.load_cached_image(
                camera.image_path(),
                lfs::io::LoadParams{
                    .resize_factor = settings.resize_factor,
                    .max_width = settings.max_width,
                    .undistort = camera.is_undistort_prepared() ? &camera.undistort_params() : nullptr,
                });
            for (int k = 0; k < kRefitSteps; ++k) {
                RenderOutput output;
                const auto result = fast_render(*ops.fast, frame, camera, model, background, 0, 0, 0, 0,
                                                settings.mip_filter, {}, false, false, output, 1.f, false);
                if (result.code != lfs::gpu_ops::RasterResult::Code::Success)
                    throw std::runtime_error(std::format("Refit render of '{}' failed: {}", camera.image_name(), result.message));
                if (!result.has_work) {
                    ops.fast->release(frame);
                    continue;
                }
                Tensor none, value, gradient, raw_gradient;
                ops.photometric->evaluate(photo, output.image, none, target, none,
                                          {.path = settings.lambda_dssim > 0 ? lfs::gpu_ops::PhotoPath::Fused : lfs::gpu_ops::PhotoPath::L1,
                                           .ssim_weight = settings.lambda_dssim},
                                          value, gradient, raw_gradient);
                fast_camera_backward(*ops.fast, frame, gradient, {}, offsets_.grad_w2c(), offsets_.grad_cam_position());
                const float r = pose_rate(k, 0, kRefitSteps / 2, kRefitSteps - 1);
                offsets_.step(static_cast<int>(row), kRotationLr * r, kTranslationLr * r);
            }
        }
    }

    lfs::Status write_scene_cameras_colmap(const lfs::core::Scene& scene,
                                           const std::filesystem::path& dataset_path,
                                           const std::filesystem::path& output_sparse_path) {
        std::vector<lfs::io::ColmapCameraWriteData> cameras;
        for (const auto& camera : scene.getAllCameras()) {
            if (camera)
                cameras.push_back({.camera = camera,
                                   .data_world_transform =
                                       scene.getCameraSceneTransformByUid(camera->uid()).value_or(glm::mat4(1.0f))});
        }
        glm::mat4 points_transform(1.0f);
        for (const auto* node : scene.getNodes()) {
            if (node && node->type == lfs::core::NodeType::DATASET) {
                points_transform = scene.getWorldTransform(node->id);
                break;
            }
        }
        auto written = lfs::io::write_colmap_reconstruction(dataset_path, output_sparse_path, cameras, nullptr,
                                                            points_transform);
        if (!written)
            return lfs::Status::failure(lfs::io::to_lfs_error(written.error(), LFS_SOURCE_SITE_CURRENT()));
        return {};
    }

} // namespace lfs::training
