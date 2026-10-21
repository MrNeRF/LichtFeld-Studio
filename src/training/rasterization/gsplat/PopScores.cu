/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../../kernels/kernel_stream.hpp"
#include "FromWorldRay.cuh"
#include "PopScores.h"
#include "Utils.cuh"

namespace gsplat_lfs {
    __device__ float pop_alpha(const PopScoreInputs& p, int g, const WorldRay& ray) {
        const vec3 xyz = reinterpret_cast<const vec3*>(p.means)[g];
        const vec3 scale = activated_scale(reinterpret_cast<const vec3*>(p.scales)[g]);
        const mat3 R = quat_to_rotmat(reinterpret_cast<const vec4*>(p.quats)[g]);
        const mat3 S(1.f / scale[0], 0.f, 0.f, 0.f, 1.f / scale[1], 0.f, 0.f, 0.f, 1.f / scale[2]);
        const mat3 iscl_rot = S * glm::transpose(R);
        const vec3 gro = iscl_rot * (ray.ray_org - xyz);
        const vec3 grd = safe_normalize(iscl_rot * ray.ray_dir);
        const vec3 cross = glm::cross(grd, gro);
        return min(0.999f, activated_opacity(p.opacities[g]) * __expf(-0.5f * glm::dot(cross, cross)));
    }

    template <bool Perfect>
    __global__ void pop_scores_kernel(PopScoreInputs p, double* scores) {
        const uint32_t pix = blockIdx.x * blockDim.x + threadIdx.x;
        const uint32_t pixels = p.width * p.height;
        if (pix >= pixels)
            return;
        const uint32_t x = pix % p.width, y = pix / p.width;
        const WorldRay ray = from_world_pixel_ray<Perfect>(p.camera_model, ShutterType::GLOBAL,
                                                           p.width, p.height, float(x) + 0.5f, float(y) + 0.5f, p.viewmat, nullptr, p.K, 0,
                                                           p.radial, p.tangential, p.thin_prism);
        if (!ray.valid_flag)
            return;
        const uint32_t tile = (y / p.tile_size) * ((p.width + p.tile_size - 1) / p.tile_size) + x / p.tile_size;
        const int begin = p.offsets[tile], end = p.offsets[tile + 1];
        // Replay the forward's exclusive cutoff; no bounded contributor buffer.
        float T = 1.f;
        int stop = end;
        for (int i = begin; i < end; ++i) {
            const float a = pop_alpha(p, p.ids[i], ray);
            if (a < 1.f / 255.f)
                continue;
            const float next = T * (1.f - a);
            if (next <= 1.e-4f) {
                stop = i;
                break;
            }
            T = next;
        }
        double behind[3];
        for (int c = 0; c < 3; ++c)
            behind[c] = p.background_image ? p.background_image[c * pixels + pix] : (p.background ? p.background[c] : 0.f);
        double transmittance = T;
        for (int i = stop - 1; i >= begin; --i) {
            const int g = p.ids[i];
            const float a = pop_alpha(p, g, ray);
            if (a < 1.f / 255.f)
                continue;
            transmittance /= (1.f - a);
            double score = 0;
            for (int c = 0; c < 3; ++c) {
                const double color = p.colors[g * p.channels + c];
                const double delta = transmittance * a * (color - behind[c]);
                score += delta * delta;
                behind[c] = a * color + (1.f - a) * behind[c];
            }
            atomicAdd(scores + g, score);
        }
    }

    void launch_pop_scores(const PopScoreInputs& p, double* scores, cudaStream_t stream) {
        stream = lfs::resolve_stream(stream);
        const uint32_t blocks = (p.width * p.height + 255) / 256;
        if (p.channels == 3 && is_perfect_pinhole_launch(p.camera_model, p.radial, p.tangential, p.thin_prism))
            pop_scores_kernel<true><<<blocks, 256, 0, stream>>>(p, scores);
        else
            pop_scores_kernel<false><<<blocks, 256, 0, stream>>>(p, scores);
        LFS_CUDA_LAUNCH_CHECK(stream, "gsplat.pop_scores");
    }
} // namespace gsplat_lfs
