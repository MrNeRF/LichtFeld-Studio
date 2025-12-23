/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "backward.h"
#include "buffer_utils.h"
#include "helper_math.h"
#include "kernels_backward.cuh"
#include "rasterization_config.h"
#include "utils.h"
#include <cub/cub.cuh>
#include <functional>

void fast_lfs::rasterization::backward(
    const float* grad_image,
    const float* grad_alpha,
    const float* image,
    const float* alpha,
    const float3* means,
    const float3* scales_raw,
    const float4* rotations_raw,
    const float* raw_opacities,
    const float3* sh_coefficients_rest,
    const float4* w2c,
    const float3* cam_position,
    char* per_primitive_buffers_blob,
    char* per_tile_buffers_blob,
    char* per_instance_buffers_blob,
    char* per_bucket_buffers_blob,
    float3* grad_means,
    float3* grad_scales_raw,
    float4* grad_rotations_raw,
    float* grad_opacities_raw,
    float3* grad_sh_coefficients_0,
    float3* grad_sh_coefficients_rest,
    float2* grad_mean2d_helper,
    float* grad_conic_helper,
    float4* grad_w2c,
    float* densification_info,
    const int n_primitives,
    const int n_visible_primitives,
    const int n_instances,
    const int n_buckets,
    const int primitive_primitive_indices_selector,
    const int instance_primitive_indices_selector,
    const int active_sh_bases,
    const int total_bases_sh_rest,
    const int width,
    const int height,
    const float fx,
    const float fy,
    const float cx,
    const float cy,
    bool mip_filter) {
    const dim3 grid(div_round_up(width, config::tile_width), div_round_up(height, config::tile_height), 1);
    const int n_tiles = grid.x * grid.y;

    // These blobs are from the arena and are guaranteed to be valid
    PerPrimitiveBuffers per_primitive_buffers = PerPrimitiveBuffers::from_blob(per_primitive_buffers_blob, n_primitives);
    PerTileBuffers per_tile_buffers = PerTileBuffers::from_blob(per_tile_buffers_blob, n_tiles);
    PerInstanceBuffers per_instance_buffers = PerInstanceBuffers::from_blob(per_instance_buffers_blob, n_instances);
    PerBucketBuffers per_bucket_buffers = PerBucketBuffers::from_blob(per_bucket_buffers_blob, n_buckets);

    // Restore selectors from forward pass
    per_primitive_buffers.primitive_indices.selector = primitive_primitive_indices_selector;
    per_instance_buffers.primitive_indices.selector = instance_primitive_indices_selector;

    // Backward blend
    kernels::backward::blend_backward_cu<<<n_buckets, 32>>>(
        per_tile_buffers.instance_ranges,
        per_tile_buffers.bucket_offsets,
        per_instance_buffers.primitive_indices.Current(),
        per_primitive_buffers.mean2d,
        per_primitive_buffers.conic_opacity,
        per_primitive_buffers.color,
        raw_opacities,
        grad_image,
        grad_alpha,
        image,
        alpha,
        per_tile_buffers.max_n_contributions,
        per_tile_buffers.n_contributions,
        per_bucket_buffers.tile_index,
        per_bucket_buffers.color_transmittance,
        grad_mean2d_helper,
        grad_conic_helper,
        grad_opacities_raw,
        grad_sh_coefficients_0, // used to store intermediate gradients
        n_buckets,
        n_primitives,
        width,
        height,
        grid.x,
        mip_filter);
    CHECK_CUDA(config::debug, "blend_backward")

    // Backward preprocess
    kernels::backward::preprocess_backward_cu<<<div_round_up(n_primitives, config::block_size_preprocess_backward), config::block_size_preprocess_backward>>>(
        means,
        scales_raw,
        rotations_raw,
        sh_coefficients_rest,
        w2c,
        cam_position,
        per_primitive_buffers.n_touched_tiles,
        grad_mean2d_helper,
        grad_conic_helper,
        grad_means,
        grad_scales_raw,
        grad_rotations_raw,
        grad_sh_coefficients_0,
        grad_sh_coefficients_rest,
        grad_w2c,
        densification_info,
        n_primitives,
        active_sh_bases,
        total_bases_sh_rest,
        static_cast<float>(width),
        static_cast<float>(height),
        fx,
        fy,
        cx,
        cy,
        mip_filter);
    CHECK_CUDA(config::debug, "preprocess_backward")
}