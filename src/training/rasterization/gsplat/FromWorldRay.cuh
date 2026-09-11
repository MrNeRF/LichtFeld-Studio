#pragma once

#include "Cameras.cuh"
#include "Common.h"

#include <cassert>

namespace gsplat_lfs {

    inline __device__ WorldRay from_world_pixel_ray_pinhole_global(
        const uint32_t image_width,
        const uint32_t image_height,
        const float px,
        const float py,
        const float* viewmats0,
        const float* Ks,
        const uint32_t cid) {
        auto rs_params = RollingShutterParameters(viewmats0 + cid * 16, nullptr);
        const vec2 focal_length = {Ks[cid * 9 + 0], Ks[cid * 9 + 4]};
        const vec2 principal_point = {Ks[cid * 9 + 2], Ks[cid * 9 + 5]};
        PerfectPinholeCameraModel::Parameters cm_params = {};
        cm_params.resolution = {image_width, image_height};
        cm_params.shutter_type = ShutterType::GLOBAL;
        cm_params.principal_point = {principal_point.x, principal_point.y};
        cm_params.focal_length = {focal_length.x, focal_length.y};
        PerfectPinholeCameraModel camera_model(cm_params);
        return camera_model.image_point_to_world_ray_shutter_pose(vec2(px, py), rs_params);
    }

    inline __device__ WorldRay from_world_pixel_ray_generic(
        const CameraModelType camera_model_type,
        const ShutterType rs_type,
        const uint32_t image_width,
        const uint32_t image_height,
        const float px,
        const float py,
        const float* viewmats0,
        const float* viewmats1,
        const float* Ks,
        const uint32_t cid,
        const float* radial_coeffs,
        const float* tangential_coeffs,
        const float* thin_prism_coeffs) {
        auto rs_params = RollingShutterParameters(
            viewmats0 + cid * 16,
            viewmats1 == nullptr ? nullptr : viewmats1 + cid * 16);
        const vec2 focal_length = {Ks[cid * 9 + 0], Ks[cid * 9 + 4]};
        const vec2 principal_point = {Ks[cid * 9 + 2], Ks[cid * 9 + 5]};

        WorldRay ray;
        if (camera_model_type == CameraModelType::PINHOLE) {
            if (radial_coeffs == nullptr && tangential_coeffs == nullptr && thin_prism_coeffs == nullptr) {
                PerfectPinholeCameraModel::Parameters cm_params = {};
                cm_params.resolution = {image_width, image_height};
                cm_params.shutter_type = rs_type;
                cm_params.principal_point = {principal_point.x, principal_point.y};
                cm_params.focal_length = {focal_length.x, focal_length.y};
                PerfectPinholeCameraModel camera_model(cm_params);
                ray = camera_model.image_point_to_world_ray_shutter_pose(vec2(px, py), rs_params);
            } else {
                OpenCVPinholeCameraModel<>::Parameters cm_params = {};
                cm_params.resolution = {image_width, image_height};
                cm_params.shutter_type = rs_type;
                cm_params.principal_point = {principal_point.x, principal_point.y};
                cm_params.focal_length = {focal_length.x, focal_length.y};
                if (radial_coeffs != nullptr) {
                    cm_params.radial_coeffs = make_array<float, 6>(radial_coeffs + cid * 6);
                }
                if (tangential_coeffs != nullptr) {
                    cm_params.tangential_coeffs = make_array<float, 2>(tangential_coeffs + cid * 2);
                }
                if (thin_prism_coeffs != nullptr) {
                    cm_params.thin_prism_coeffs = make_array<float, 4>(thin_prism_coeffs + cid * 4);
                }
                OpenCVPinholeCameraModel camera_model(cm_params);
                ray = camera_model.image_point_to_world_ray_shutter_pose(vec2(px, py), rs_params);
            }
        } else if (camera_model_type == CameraModelType::FISHEYE) {
            OpenCVFisheyeCameraModel<>::Parameters cm_params = {};
            cm_params.resolution = {image_width, image_height};
            cm_params.shutter_type = rs_type;
            cm_params.principal_point = {principal_point.x, principal_point.y};
            cm_params.focal_length = {focal_length.x, focal_length.y};
            if (radial_coeffs != nullptr) {
                cm_params.radial_coeffs = make_array<float, 4>(radial_coeffs + cid * 4);
            }
            OpenCVFisheyeCameraModel camera_model(cm_params);
            ray = camera_model.image_point_to_world_ray_shutter_pose(vec2(px, py), rs_params);
        } else if (camera_model_type == CameraModelType::EQUIRECTANGULAR) {
            const uint32_t full_image_width = static_cast<uint32_t>(focal_length.x);
            const uint32_t full_image_height = static_cast<uint32_t>(focal_length.y);
            const float tile_x_offset = principal_point.x;
            const float tile_y_offset = principal_point.y;

            EquirectangularCameraModel::Parameters cm_params = {};
            cm_params.resolution = {full_image_width, full_image_height};
            cm_params.shutter_type = rs_type;
            EquirectangularCameraModel camera_model(cm_params);

            const float px_full = px + tile_x_offset;
            const float py_full = py + tile_y_offset;
            ray = camera_model.image_point_to_world_ray_shutter_pose(vec2(px_full, py_full), rs_params);
        } else if (camera_model_type == CameraModelType::THIN_PRISM_FISHEYE) {
            ThinPrismFisheyeCameraModel<>::Parameters cm_params = {};
            cm_params.resolution = {image_width, image_height};
            cm_params.shutter_type = rs_type;
            cm_params.principal_point = {principal_point.x, principal_point.y};
            cm_params.focal_length = {focal_length.x, focal_length.y};
            if (radial_coeffs != nullptr) {
                cm_params.radial_coeffs = make_array<float, 4>(radial_coeffs + cid * 4);
            }
            if (thin_prism_coeffs != nullptr) {
                cm_params.thin_prism_coeffs = make_array<float, 4>(thin_prism_coeffs + cid * 4);
            }
            ThinPrismFisheyeCameraModel camera_model(cm_params);
            ray = camera_model.image_point_to_world_ray_shutter_pose(vec2(px, py), rs_params);
        } else {
            assert(false);
            ray = {vec3{}, vec3{}, false};
        }
        return ray;
    }

    // kPerfectPinhole inlines the undistorted pinhole formula. Otherwise the
    // generic camera model runs once per pixel before the splat loop.
    template <bool kPerfectPinhole>
    inline __device__ WorldRay from_world_pixel_ray(
        const CameraModelType camera_model_type,
        const ShutterType rs_type,
        const uint32_t image_width,
        const uint32_t image_height,
        const float px,
        const float py,
        const float* viewmats0,
        const float* viewmats1,
        const float* Ks,
        const uint32_t cid,
        const float* radial_coeffs,
        const float* tangential_coeffs,
        const float* thin_prism_coeffs) {
        if constexpr (kPerfectPinhole) {
            return from_world_pixel_ray_pinhole_global(
                image_width, image_height, px, py, viewmats0, Ks, cid);
        } else {
            return from_world_pixel_ray_generic(
                camera_model_type, rs_type, image_width, image_height, px, py,
                viewmats0, viewmats1, Ks, cid,
                radial_coeffs, tangential_coeffs, thin_prism_coeffs);
        }
    }

    inline bool is_global_shutter_launch(
        const ShutterType rs_type,
        const float* viewmats1) {
        return rs_type == ShutterType::GLOBAL && viewmats1 == nullptr;
    }

    inline bool is_perfect_pinhole_launch(
        const CameraModelType camera_model,
        const float* radial_coeffs,
        const float* tangential_coeffs,
        const float* thin_prism_coeffs) {
        return camera_model == PINHOLE &&
               radial_coeffs == nullptr &&
               tangential_coeffs == nullptr &&
               thin_prism_coeffs == nullptr;
    }

} // namespace gsplat_lfs
