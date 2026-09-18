/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "fastgs_fisheye_cuda.hpp"
#include "fisheye_kb.cuh"

#include <cuda_runtime.h>

namespace fastgs_fisheye_test {
    namespace {

        __global__ void project_kernel(DeviceProjectSample* samples, const int n) {
            const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            if (i >= n)
                return;
            DeviceProjectSample s = samples[i];
            const auto proj = fast_lfs::rasterization::fisheye_kb::kb_project(
                s.Px, s.Py, s.Pz, s.fx, s.fy, s.cx, s.cy, s.k1, s.k2, s.k3, s.k4);
            s.px = proj.px;
            s.py = proj.py;
            s.J00 = proj.J00;
            s.J01 = proj.J01;
            s.J02 = proj.J02;
            s.J10 = proj.J10;
            s.J11 = proj.J11;
            s.J12 = proj.J12;
            s.theta = proj.theta;
            s.D = proj.D;
            s.finite = proj.finite ? 1 : 0;
            samples[i] = s;
        }

    } // namespace

    bool project_on_device(DeviceProjectSample* samples, std::size_t n) {
        if (n == 0)
            return true;
        DeviceProjectSample* d_samples = nullptr;
        const std::size_t bytes = n * sizeof(DeviceProjectSample);
        if (cudaMalloc(&d_samples, bytes) != cudaSuccess)
            return false;
        if (cudaMemcpy(d_samples, samples, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaFree(d_samples);
            return false;
        }
        const int threads = 128;
        const int blocks = static_cast<int>((n + threads - 1) / threads);
        project_kernel<<<blocks, threads>>>(d_samples, static_cast<int>(n));
        const cudaError_t launch_err = cudaGetLastError();
        const cudaError_t sync_err = cudaDeviceSynchronize();
        if (launch_err != cudaSuccess || sync_err != cudaSuccess) {
            cudaFree(d_samples);
            return false;
        }
        if (cudaMemcpy(samples, d_samples, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
            cudaFree(d_samples);
            return false;
        }
        cudaFree(d_samples);
        return true;
    }

} // namespace fastgs_fisheye_test
