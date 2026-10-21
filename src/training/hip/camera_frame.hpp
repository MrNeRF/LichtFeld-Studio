/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "session.hpp"
#include <cmath>
#include <core/camera.hpp>
#include <cstring>
#include <stdexcept>

namespace lfs::training::hip {
    // Owns the prepared CPU image across loader reuse. Resolution comes from the
    // actual image, not Camera's mutable last-loaded image dimensions. Input has
    // the same float CHW values as the trainer; no additional color transform.
    class CameraFrame {
    public:
        CameraFrame(const core::Camera& camera, const core::Tensor& image,
                    float near_plane = 0.01f, float far_plane = 1000.f, bool undistorted = false, HostTargets* host_targets = nullptr) {
            using namespace core;
            if (camera.camera_model_type() != CameraModelType::PINHOLE)
                throw std::invalid_argument("HIP training currently requires a pinhole camera");
            if (undistorted && !camera.is_undistort_prepared())
                throw std::invalid_argument("Undistorted HIP target requires prepared camera calibration");
            for (const auto& coefficients : {camera.radial_distortion(), camera.tangential_distortion()}) {
                if (coefficients.is_valid() && coefficients.numel()) {
                    const auto values = coefficients.cpu().contiguous().to_vector();
                    for (float value : values)
                        if (value != 0 && !undistorted)
                            throw std::invalid_argument("HIP training requires an undistorted camera calibration");
                }
            }
            if (!image.is_valid() || image.device() != Device::CPU || image.dtype() != DataType::Float32 ||
                image.ndim() != 3 || image.shape()[0] != 3 || !image.shape()[1] || !image.shape()[2] ||
                image.shape()[1] > UINT32_MAX || image.shape()[2] > UINT32_MAX)
                throw std::invalid_argument("HIP training requires a prepared CPU float32 RGB CHW image");
            if (camera.camera_width() <= 0 || camera.camera_height() <= 0 ||
                !std::isfinite(near_plane) || !std::isfinite(far_plane) || near_plane <= 0 || far_plane <= near_plane)
                throw std::invalid_argument("Invalid HIP training camera extent or clipping planes");
            auto contiguous = image.contiguous();
            auto owner = host_targets ? host_targets->allocate(contiguous.bytes()) : std::shared_ptr<void>{};
            if (owner) {
                std::memcpy(owner.get(), contiguous.ptr<float>(), contiguous.bytes());
                image_ = Tensor::from_external_owner(owner.get(), contiguous.shape(), Device::CPU,
                                                    DataType::Float32, owner, contiguous.bytes());
            } else {
                image_ = contiguous.clone();
            }
            frame_.struct_bytes = sizeof(frame_);
            frame_.width = static_cast<uint32_t>(image.shape()[2]);
            frame_.height = static_cast<uint32_t>(image.shape()[1]);
            const float sx = float(frame_.width) / camera.camera_width();
            const float sy = float(frame_.height) / camera.camera_height();
            frame_.fx = camera.focal_x() * sx;
            frame_.fy = camera.focal_y() * sy;
            frame_.cx = camera.center_x() * sx;
            frame_.cy = camera.center_y() * sy;
            if (!std::isfinite(frame_.fx) || !std::isfinite(frame_.fy) || frame_.fx <= 0 || frame_.fy <= 0 ||
                !std::isfinite(frame_.cx) || !std::isfinite(frame_.cy))
                throw std::invalid_argument("Invalid HIP training intrinsics");
            auto rotation = camera.R().cpu().contiguous();
            auto translation = camera.T().cpu().contiguous();
            if (rotation.dtype() != DataType::Float32 || translation.dtype() != DataType::Float32 ||
                rotation.numel() != 9 || translation.numel() != 3)
                throw std::invalid_argument("Invalid HIP training camera extrinsics");
            const auto r = rotation.to_vector();
            const auto t = translation.to_vector();
            for (float value : r)
                if (!std::isfinite(value))
                    throw std::invalid_argument("Nonfinite HIP training rotation");
            for (float value : t)
                if (!std::isfinite(value))
                    throw std::invalid_argument("Nonfinite HIP training translation");
            for (size_t row = 0; row < 3; ++row) {
                for (size_t col = 0; col < 3; ++col)
                    frame_.world_to_camera[row * 4 + col] = r[row * 3 + col];
                frame_.world_to_camera[row * 4 + 3] = t[row];
            }
            frame_.world_to_camera[15] = 1;
            frame_.near_plane = near_plane;
            frame_.far_plane = far_plane;
        }
        // Form the borrowed pointer on demand, so copying/moving this owner
        // never leaves a descriptor pointing into a different owner's storage.
        LfsHipTrainFrame descriptor() const {
            auto result = frame_;
            result.target_chw = image_.ptr<float>();
            return result;
        }
        void bind(Session& session) const { session.set_frame(descriptor()); }

    private:
        core::Tensor image_;
        LfsHipTrainFrame frame_{};
    };
} // namespace lfs::training::hip
