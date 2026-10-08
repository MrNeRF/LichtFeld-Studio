/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gut_camera_model_cuda.hpp"
#include "training/rasterization/gsplat/Cameras.cuh"

#include <cuda_runtime.h>
#include <stdexcept>

namespace gut_camera_model_test {
    namespace {
        using Model = ThinPrismFisheyeCameraModel<>;

        Model::Parameters parameters_of(const ThinPrismCamera& camera) {
            Model::Parameters parameters{};
            parameters.resolution = {static_cast<uint32_t>(camera.resolution[0]),
                                     static_cast<uint32_t>(camera.resolution[1])};
            parameters.focal_length = camera.focal;
            parameters.principal_point = camera.principal;
            parameters.radial_coeffs = camera.radial;
            parameters.thin_prism_coeffs = camera.thin_prism;
            return parameters;
        }

        __global__ void project_kernel(const Model model, const float* rays, const int count, float* points) {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= count)
                return;
            const auto result = model.camera_ray_to_image_point(glm::fvec3{rays[3 * i], rays[3 * i + 1], rays[3 * i + 2]}, 0.f);
            points[3 * i] = result.imagePoint.x;
            points[3 * i + 1] = result.imagePoint.y;
            points[3 * i + 2] = result.valid_flag ? 1.f : 0.f;
        }

        __global__ void unproject_kernel(const Model model, const float* points, const int count, float* rays) {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= count)
                return;
            const auto result = model.image_point_to_camera_ray(glm::fvec2{points[2 * i], points[2 * i + 1]});
            rays[4 * i] = result.ray_dir.x;
            rays[4 * i + 1] = result.ray_dir.y;
            rays[4 * i + 2] = result.ray_dir.z;
            rays[4 * i + 3] = result.valid_flag ? 1.f : 0.f;
        }

        template <typename Kernel>
        std::vector<float> run(Kernel kernel, const Model& model, const std::vector<float>& input, const int count,
                               const int output_width) {
            float* device_input = nullptr;
            float* device_output = nullptr;
            std::vector<float> output(static_cast<size_t>(count) * output_width);
            if (cudaMalloc(&device_input, input.size() * sizeof(float)) != cudaSuccess ||
                cudaMalloc(&device_output, output.size() * sizeof(float)) != cudaSuccess)
                throw std::runtime_error("cudaMalloc failed");
            cudaMemcpy(device_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice);
            kernel<<<(count + 63) / 64, 64>>>(model, device_input, count, device_output);
            const auto status = cudaMemcpy(output.data(), device_output, output.size() * sizeof(float),
                                           cudaMemcpyDeviceToHost);
            cudaFree(device_input);
            cudaFree(device_output);
            if (status != cudaSuccess)
                throw std::runtime_error(cudaGetErrorString(status));
            return output;
        }
    } // namespace

    std::vector<float> project(const ThinPrismCamera& camera, const std::vector<float>& rays) {
        return run(project_kernel, Model(parameters_of(camera)), rays, static_cast<int>(rays.size() / 3), 3);
    }

    std::vector<float> unproject(const ThinPrismCamera& camera, const std::vector<float>& points) {
        return run(unproject_kernel, Model(parameters_of(camera)), points, static_cast<int>(points.size() / 2), 4);
    }

} // namespace gut_camera_model_test
