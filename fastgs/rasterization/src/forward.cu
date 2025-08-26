#include "buffer_utils.h"
#include "forward.h"
#include "helper_math.h"
#include "kernels_forward.cuh"
#include "rasterization_config.h"
#include "utils.h"
#include <cub/cub.cuh>
#include <functional>

// sorting is done separately for depth and tile as proposed in https://github.com/m-schuetz/Splatshop
std::tuple<int, int, int, int, int> fast_gs::rasterization::forward(
    std::function<char*(size_t)> per_primitive_buffers_func,
    std::function<char*(size_t)> per_tile_buffers_func,
    std::function<char*(size_t)> per_instance_buffers_func,
    std::function<char*(size_t)> per_bucket_buffers_func,
    const float3* means,
    const float3* scales_raw,
    const float4* rotations_raw,
    const float* opacities_raw,
    const float3* sh_coefficients_0,
    const float3* sh_coefficients_rest,
    const float4* w2c,
    const float3* cam_position,
    float* image,
    float* alpha,
    const int n_primitives,
    const int active_sh_bases,
    const int total_bases_sh_rest,
    const int width,
    const int height,
    const float fx,
    const float fy,
    const float cx,
    const float cy,
    const float near_, // near and far are macros in windowns
    const float far_) {
    const dim3 grid(div_round_up(width, config::tile_width), div_round_up(height, config::tile_height), 1);
    const dim3 block(config::tile_width, config::tile_height, 1);
    const int n_tiles = grid.x * grid.y;

    char* per_tile_buffers_blob = per_tile_buffers_func(required<PerTileBuffers>(n_tiles));
    PerTileBuffers per_tile_buffers = PerTileBuffers::from_blob(per_tile_buffers_blob, n_tiles);

    static cudaStream_t memset_stream = 0;
    if constexpr (!config::debug) {
        static bool memset_stream_initialized = false;
        if (!memset_stream_initialized) {
            cudaStreamCreate(&memset_stream);
            memset_stream_initialized = true;
        }
        cudaMemsetAsync(per_tile_buffers.instance_ranges, 0, sizeof(uint2) * n_tiles, memset_stream);
    } else
        cudaMemset(per_tile_buffers.instance_ranges, 0, sizeof(uint2) * n_tiles);

    char* per_primitive_buffers_blob = per_primitive_buffers_func(required<PerPrimitiveBuffers>(n_primitives));
    PerPrimitiveBuffers per_primitive_buffers = PerPrimitiveBuffers::from_blob(per_primitive_buffers_blob, n_primitives);

    cudaMemset(per_primitive_buffers.n_visible_primitives, 0, sizeof(uint));
    cudaMemset(per_primitive_buffers.n_instances, 0, sizeof(uint));

    kernels::forward::preprocess_cu<<<div_round_up(n_primitives, config::block_size_preprocess), config::block_size_preprocess>>>(
        means,
        scales_raw,
        rotations_raw,
        opacities_raw,
        sh_coefficients_0,
        sh_coefficients_rest,
        w2c,
        cam_position,
        per_primitive_buffers.depth_keys.Current(),
        per_primitive_buffers.primitive_indices.Current(),
        per_primitive_buffers.n_touched_tiles,
        per_primitive_buffers.screen_bounds,
        per_primitive_buffers.mean2d,
        per_primitive_buffers.conic_opacity,
        per_primitive_buffers.color,
        per_primitive_buffers.n_visible_primitives,
        per_primitive_buffers.n_instances,
        n_primitives,
        grid.x,
        grid.y,
        active_sh_bases,
        total_bases_sh_rest,
        static_cast<float>(width),
        static_cast<float>(height),
        fx,
        fy,
        cx,
        cy,
        near_,
        far_);
    CHECK_CUDA(config::debug, "preprocess")

    int n_visible_primitives;
    cudaMemcpy(&n_visible_primitives, per_primitive_buffers.n_visible_primitives, sizeof(uint), cudaMemcpyDeviceToHost);
    int n_instances;
    cudaMemcpy(&n_instances, per_primitive_buffers.n_instances, sizeof(uint), cudaMemcpyDeviceToHost);

    cub::DeviceRadixSort::SortPairs(
        per_primitive_buffers.cub_workspace,
        per_primitive_buffers.cub_workspace_size,
        per_primitive_buffers.depth_keys,
        per_primitive_buffers.primitive_indices,
        n_visible_primitives);
    CHECK_CUDA(config::debug, "cub::DeviceRadixSort::SortPairs (Depth)")

    kernels::forward::apply_depth_ordering_cu<<<div_round_up(n_visible_primitives, config::block_size_apply_depth_ordering), config::block_size_apply_depth_ordering>>>(
        per_primitive_buffers.primitive_indices.Current(),
        per_primitive_buffers.n_touched_tiles,
        per_primitive_buffers.offset,
        n_visible_primitives);
    CHECK_CUDA(config::debug, "apply_depth_ordering")

    cub::DeviceScan::ExclusiveSum(
        per_primitive_buffers.cub_workspace,
        per_primitive_buffers.cub_workspace_size,
        per_primitive_buffers.offset,
        per_primitive_buffers.offset,
        n_visible_primitives);
    CHECK_CUDA(config::debug, "cub::DeviceScan::ExclusiveSum (Primitive Offsets)")

    char* per_instance_buffers_blob = per_instance_buffers_func(required<PerInstanceBuffers>(n_instances));
    PerInstanceBuffers per_instance_buffers = PerInstanceBuffers::from_blob(per_instance_buffers_blob, n_instances);

    kernels::forward::create_instances_cu<<<div_round_up(n_visible_primitives, config::block_size_create_instances), config::block_size_create_instances>>>(
        per_primitive_buffers.primitive_indices.Current(),
        per_primitive_buffers.offset,
        per_primitive_buffers.screen_bounds,
        per_primitive_buffers.mean2d,
        per_primitive_buffers.conic_opacity,
        per_instance_buffers.keys.Current(),
        per_instance_buffers.primitive_indices.Current(),
        grid.x,
        n_visible_primitives);
    CHECK_CUDA(config::debug, "create_instances")

    cub::DeviceRadixSort::SortPairs(
        per_instance_buffers.cub_workspace,
        per_instance_buffers.cub_workspace_size,
        per_instance_buffers.keys,
        per_instance_buffers.primitive_indices,
        n_instances);
    CHECK_CUDA(config::debug, "cub::DeviceRadixSort::SortPairs (Tile)")

    if constexpr (!config::debug)
        cudaStreamSynchronize(memset_stream);

    if (n_instances > 0) {
        kernels::forward::extract_instance_ranges_cu<<<div_round_up(n_instances, config::block_size_extract_instance_ranges), config::block_size_extract_instance_ranges>>>(
            per_instance_buffers.keys.Current(),
            per_tile_buffers.instance_ranges,
            n_instances);
        CHECK_CUDA(config::debug, "extract_instance_ranges")
    }

    kernels::forward::extract_bucket_counts<<<div_round_up(n_tiles, config::block_size_extract_bucket_counts), config::block_size_extract_bucket_counts>>>(
        per_tile_buffers.instance_ranges,
        per_tile_buffers.n_buckets,
        n_tiles);
    CHECK_CUDA(config::debug, "extract_bucket_counts")

    cub::DeviceScan::InclusiveSum(
        per_tile_buffers.cub_workspace,
        per_tile_buffers.cub_workspace_size,
        per_tile_buffers.n_buckets,
        per_tile_buffers.bucket_offsets,
        n_tiles);
    CHECK_CUDA(config::debug, "cub::DeviceScan::InclusiveSum (Bucket Counts)")

    int n_buckets;
    cudaMemcpy(&n_buckets, per_tile_buffers.bucket_offsets + n_tiles - 1, sizeof(uint), cudaMemcpyDeviceToHost);

    char* per_bucket_buffers_blob = per_bucket_buffers_func(required<PerBucketBuffers>(n_buckets));
    PerBucketBuffers per_bucket_buffers = PerBucketBuffers::from_blob(per_bucket_buffers_blob, n_buckets);

    kernels::forward::blend_cu<<<grid, block>>>(
        per_tile_buffers.instance_ranges,
        per_tile_buffers.bucket_offsets,
        per_instance_buffers.primitive_indices.Current(),
        per_primitive_buffers.mean2d,
        per_primitive_buffers.conic_opacity,
        per_primitive_buffers.color,
        image,
        alpha,
        per_tile_buffers.max_n_contributions,
        per_tile_buffers.n_contributions,
        per_bucket_buffers.tile_index,
        per_bucket_buffers.color_transmittance,
        width,
        height,
        grid.x);
    CHECK_CUDA(config::debug, "blend")

    return {n_visible_primitives, n_instances, n_buckets, per_primitive_buffers.primitive_indices.selector, per_instance_buffers.primitive_indices.selector};
}
