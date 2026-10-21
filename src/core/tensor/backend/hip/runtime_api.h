/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include <stddef.h>
#include <stdint.h>

/* Private, versioned C boundary: no CUDA/HIP headers or C++ ownership cross it.
 * Calls on one context must be serialized by the host. Buffers belong to their
 * context; destruction drains its queue before releasing live allocations.
 * A nonzero result leaves details in the caller-owned error buffer. */
#if defined(__HIP_DEVICE_COMPILE__)
#define LFS_HIP_API
#elif defined(_WIN32)
#ifdef LFS_HIP_RUNTIME_EXPORTS
#define LFS_HIP_API __declspec(dllexport)
#else
#define LFS_HIP_API __declspec(dllimport)
#endif
#else
#define LFS_HIP_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct LfsHipContext LfsHipContext;
typedef struct LfsHipBuffer LfsHipBuffer;
typedef struct LfsHipReadback LfsHipReadback;

typedef struct LfsHipError {
    int32_t native_code;
    char message[256];
} LfsHipError;

typedef struct LfsHipDevice {
    char name[256];
    char architecture[256];
    char pci_bus_id[32];
    uint64_t total_bytes;
    int32_t warp_size;
} LfsHipDevice;

LFS_HIP_API uint32_t lfs_hip_runtime_abi(void);
LFS_HIP_API int lfs_hip_device_count(int32_t* count, LfsHipError* error);
LFS_HIP_API int lfs_hip_device_info(int32_t ordinal, LfsHipDevice* info, LfsHipError* error);
LFS_HIP_API int lfs_hip_context_create(int32_t ordinal, LfsHipContext** context, LfsHipError* error);
LFS_HIP_API int lfs_hip_context_destroy(LfsHipContext* context, LfsHipError* error);
/* Opaque native HIP queue for separately compiled HIP training kernels. */
LFS_HIP_API int lfs_hip_context_queue(LfsHipContext*, uint64_t* queue, int32_t* device, LfsHipError*);
LFS_HIP_API int lfs_hip_context_ready(LfsHipContext*, int32_t* ready, LfsHipError*);
LFS_HIP_API int lfs_hip_host_callback(LfsHipContext*, void (*callback)(void*), void* user, LfsHipError*);
LFS_HIP_API int lfs_hip_context_synchronize(LfsHipContext* context, LfsHipError* error);
LFS_HIP_API int lfs_hip_buffer_allocate(LfsHipContext* context, size_t bytes, LfsHipBuffer** buffer, LfsHipError* error);
LFS_HIP_API int lfs_hip_buffer_release(LfsHipContext* context, LfsHipBuffer* buffer, LfsHipError* error);
/* Upload/download are synchronous: caller-owned host memory may be reused on return. */
LFS_HIP_API int lfs_hip_upload(LfsHipContext* context, LfsHipBuffer* buffer, size_t offset, const void* host, size_t bytes, LfsHipError* error);
LFS_HIP_API int lfs_hip_download(LfsHipContext* context, const LfsHipBuffer* buffer, size_t offset, void* host, size_t bytes, LfsHipError* error);
/* Reusable pinned staging. Context calls are serialized by the host adapter.
 * enqueue snapshots the source on the context queue. poll consumes a completed
 * snapshot; a null destination discards it. wait does not consume. destroy
 * drains pending copies and must precede context destruction. */
LFS_HIP_API int lfs_hip_readback_create(LfsHipContext*, LfsHipReadback**, LfsHipError*);
LFS_HIP_API int lfs_hip_readback_enqueue(LfsHipContext*, LfsHipReadback*, const LfsHipBuffer*, size_t offset, size_t bytes, LfsHipError*);
/* Snapshots CPU input into owned pinned staging before returning, then uploads
 * asynchronously. The same wait/poll/destroy contract applies; poll(nullptr)
 * consumes completion. No borrowed host pointer survives this call. */
LFS_HIP_API int lfs_hip_staged_upload(LfsHipContext*, LfsHipReadback*, LfsHipBuffer*, size_t offset, const void* source, size_t bytes, LfsHipError*);
/* Re-record a zero-byte marker without waiting on the CPU. Data staging is rejected. */
LFS_HIP_API int lfs_hip_marker_record(LfsHipContext*, LfsHipReadback*, LfsHipError*);
/* Packed readback slot: reserve once, append copies on the same queue and
 * consume completion before reusing. Returned host storage lives until destroy. */
LFS_HIP_API int lfs_hip_readback_reserve(LfsHipContext*, LfsHipReadback*, size_t bytes, uint64_t* host, LfsHipError*);
LFS_HIP_API int lfs_hip_readback_append(LfsHipContext*, LfsHipReadback*, const LfsHipBuffer*, size_t source_offset, size_t bytes, size_t destination_offset, LfsHipError*);
LFS_HIP_API int lfs_hip_readback_wait(LfsHipContext*, LfsHipReadback*, LfsHipError*);
LFS_HIP_API int lfs_hip_readback_poll(LfsHipContext*, LfsHipReadback*, void* destination, int32_t* ready, LfsHipError*);
LFS_HIP_API int lfs_hip_readback_destroy(LfsHipContext*, LfsHipReadback*, LfsHipError*);
/* Device operations enqueue on the context's nonblocking stream. */
LFS_HIP_API int lfs_hip_fill_f32(LfsHipContext* context, LfsHipBuffer* buffer, size_t offset, size_t count, float value, LfsHipError* error);
LFS_HIP_API int lfs_hip_affine_f32(LfsHipContext* context, const LfsHipBuffer* input, LfsHipBuffer* output, size_t count, float scale, float bias, LfsHipError* error);

/* Private native Tensor seam. Offsets are bytes; handles retain context ownership. */
typedef struct LfsHipView {
    LfsHipBuffer* buffer;
    size_t offset;
} LfsHipView;
/* Image conversion ordinals match gpu_ops::ImageConversion (0..15).
 * 16/17 sentinel fill/check, 18/19 RGBA float/byte RGB, 20/21/22 mask invert/threshold/byte-grid quantization.
 * Views and extents are validated against context-owned allocations. */
typedef struct LfsHipSharedImage {
    uint32_t struct_bytes, operation, seed;
    uint32_t srgb, flip_yz, world_to_camera;
    size_t height, width, channels;
    float threshold, w2c[9];
} LfsHipSharedImage;
LFS_HIP_API int lfs_hip_shared_image(LfsHipContext*, const LfsHipSharedImage*, LfsHipView source,
                                   LfsHipView destination, LfsHipView auxiliary, LfsHipError*);
/* Five Resample modes: RGB HWC, gray HW, float CHW, depth prior, normal prior.
 * Coefficient workspaces are supplied by the Tensor owner, not hidden allocations. */
typedef struct LfsHipSharedResize {
    uint32_t struct_bytes, mode, dtype, channels, kernel_size, stride_x, stride_y;
    int32_t source_h, source_w, destination_h, destination_w;
} LfsHipSharedResize;
LFS_HIP_API int lfs_hip_shared_resize(LfsHipContext*, const LfsHipSharedResize*, LfsHipView source,
                                    LfsHipView destination, LfsHipView coefficients_x, LfsHipView coefficients_y, LfsHipError*);
/* Native inverse lens sample map; currently pinhole only. Output is H,W,2
 * normalized undistorted rays, NaN where the first monotonic branch has no ray. */
typedef struct LfsHipLens {
    float src_fx, src_fy, src_cx, src_cy;
    float dst_fx, dst_fy, dst_cx, dst_cy;
    int32_t src_width, src_height, dst_width, dst_height;
    int32_t model_type;
    float distortion[12];
    int32_t num_distortion;
} LfsHipLens;
LFS_HIP_API int lfs_hip_inverse_lens(LfsHipContext*, LfsHipView output, const LfsHipLens*, LfsHipError*);
/* Forward pinhole RGB area/Lanczos warp. Input is contiguous CHW F32 or U8,
 * output contiguous CHW F32. Work is split into bounded output-row batches.
 * No input/output overlap, no host fallback, same context queue as Tensor ops. */
LFS_HIP_API int lfs_hip_warp_rgb(LfsHipContext*, LfsHipView input, uint32_t dtype,
                                 LfsHipView output, const LfsHipLens*, LfsHipError*);

/* General forward camera preprocessing: mode 0 Lanczos color, 1 mask area,
 * 2 positive depth area, 3 unit normal area. Channels 1..4; all lens models. */
LFS_HIP_API int lfs_hip_shared_warp(LfsHipContext*, LfsHipView input, uint32_t dtype, uint32_t channels,
                                  uint32_t mode, LfsHipView output, const LfsHipLens*, LfsHipError*);

enum LfsHipPointwiseOp {
#define LFS_POINTWISE_OP(Id, FunctorType, Name) LFS_HIP_OP_##Id,
#include "../../../include/core/detail/pointwise_ops.def"
#undef LFS_POINTWISE_OP
    LFS_HIP_OP_COUNT
};
LFS_HIP_API int lfs_hip_buffer_address(LfsHipContext*, const LfsHipBuffer*, uint64_t* address, LfsHipError*);
LFS_HIP_API int lfs_hip_copy(LfsHipContext*, LfsHipView src, LfsHipView dst, size_t bytes, LfsHipError*);
LFS_HIP_API int lfs_hip_memset(LfsHipContext*, LfsHipView dst, size_t bytes, uint8_t value, LfsHipError*);
LFS_HIP_API int lfs_hip_boolean_pointwise(LfsHipContext*, uint32_t op, LfsHipView lhs, LfsHipView rhs, LfsHipView output, size_t count, float scalar, LfsHipError*);
LFS_HIP_API int lfs_hip_clamp_i32(LfsHipContext*, LfsHipView data, size_t count, int32_t minimum, int32_t maximum, LfsHipError*);
LFS_HIP_API int lfs_hip_pointwise_f32(LfsHipContext*, uint32_t op, LfsHipView lhs, LfsHipView rhs, LfsHipView output, size_t count, float scalar, LfsHipError*);
LFS_HIP_API int lfs_hip_pointwise_f16(LfsHipContext*, uint32_t op, LfsHipView lhs, LfsHipView rhs, LfsHipView output, size_t count, float scalar, LfsHipError*);
LFS_HIP_API int lfs_hip_masked_scatter(LfsHipContext*, LfsHipView output, LfsHipView mask, LfsHipView source,
                                       uint32_t dtype, size_t count, size_t selected_count, LfsHipError*);
LFS_HIP_API int lfs_hip_arange_f32(LfsHipContext*, LfsHipView dst, size_t count, float start, float step, LfsHipError*);

enum { LFS_HIP_REDUCTION_WORKSPACE_FLOATS = 4097 };
enum LfsHipReduceOp { LFS_HIP_REDUCE_SUM,
                      LFS_HIP_REDUCE_MEAN,
                      LFS_HIP_REDUCE_MIN,
                      LFS_HIP_REDUCE_MAX,
                      LFS_HIP_REDUCE_HAS_NAN,
                      LFS_HIP_REDUCE_HAS_INF };
typedef struct LfsHipMemoryInfo {
    uint64_t total_bytes, free_bytes, owned_bytes, workspace_bytes;
} LfsHipMemoryInfo;
LFS_HIP_API int lfs_hip_reduce_f32(LfsHipContext*, LfsHipView input, size_t count, uint32_t operation, float* output, LfsHipError*);
LFS_HIP_API int lfs_hip_memory_info(LfsHipContext*, LfsHipMemoryInfo*, LfsHipError*);
LFS_HIP_API int lfs_hip_trim_workspace(LfsHipContext*, LfsHipError*);

typedef struct LfsHipChainOp {
    uint32_t op;
    float scalar;
    LfsHipView rhs;
} LfsHipChainOp;
typedef struct LfsHipChain {
    uint32_t count;
    LfsHipChainOp ops[16];
} LfsHipChain;
LFS_HIP_API int lfs_hip_chain_f32(LfsHipContext*, LfsHipView input, LfsHipView output, size_t count, const LfsHipChain*, LfsHipError*);
LFS_HIP_API int lfs_hip_chain_reduce_f32(LfsHipContext*, LfsHipView input, LfsHipView output, size_t segments, size_t segment_size, const LfsHipChain*, uint32_t operation, LfsHipError*);
typedef struct LfsHipPool {
    size_t count;
    int32_t height,width,out_height,out_width,kernel_h,kernel_w,stride_h,stride_w,pad_h,pad_w;
    uint32_t average,include_pad;
} LfsHipPool;
LFS_HIP_API int lfs_hip_pool_f32(LfsHipContext*,LfsHipView input,LfsHipView output,const LfsHipPool*,LfsHipError*);
LFS_HIP_API int lfs_hip_where_scalar(LfsHipContext*,LfsHipView mask,LfsHipView source,LfsHipView output,size_t count,uint32_t dtype,float value,LfsHipError*);

typedef struct LfsHipLayout {
    uint32_t rank;
    uint64_t dims[8], strides[8], elements;
} LfsHipLayout;
typedef struct LfsHipLayoutCopy {
    LfsHipView input, output;
    LfsHipLayout input_layout, output_layout;
    uint32_t element_bytes;
} LfsHipLayoutCopy;
/* Validate the complete batch before launching any copy. Layouts are logical,
 * positive-stride views; overlapping ranges within a request are rejected. */
LFS_HIP_API int lfs_hip_copy_layouts(LfsHipContext*, const LfsHipLayoutCopy*, size_t count, LfsHipError*);
LFS_HIP_API int lfs_hip_strided_scatter(LfsHipContext*, LfsHipView input, LfsHipView output, const LfsHipLayout*, uint32_t element_bytes, uint32_t int32_to_float32, LfsHipError*);
LFS_HIP_API int lfs_hip_strided_copy(LfsHipContext*, LfsHipView input, LfsHipView output, const LfsHipLayout*, uint32_t element_bytes, LfsHipError*);
LFS_HIP_API int lfs_hip_fill_layout_bits(LfsHipContext*, LfsHipView output, const LfsHipLayout*, uint32_t element_bytes, uint64_t value_bits, LfsHipError*);
LFS_HIP_API int lfs_hip_fill_strided_f32(LfsHipContext*, LfsHipView output, const LfsHipLayout*, float value, LfsHipError*);

enum { LFS_HIP_ERROR_INVALID_INDEX = -2 };
enum LfsHipDtype { LFS_HIP_F32 = 0,
                   LFS_HIP_F16 = 1,
                   LFS_HIP_I32 = 2,
                   LFS_HIP_I64 = 3,
                   LFS_HIP_U8 = 4,
                   LFS_HIP_BOOL = 5,
                   LFS_HIP_U32 = 6 };
LFS_HIP_API int lfs_hip_cumsum(LfsHipContext*, LfsHipView data, uint32_t dtype, const LfsHipLayout*, uint32_t axis, LfsHipError*);
LFS_HIP_API int lfs_hip_sort_pairs(LfsHipContext*, LfsHipView values, LfsHipView indices, size_t outer, size_t dimension, size_t inner, uint32_t descending, LfsHipError*);
/* kind 0: stable masked values; kind 1: flat nonzero Int64 positions.
 * capacity is in output elements. An undersized capacity preserves output. */
LFS_HIP_API int lfs_hip_select(LfsHipContext*, LfsHipView input, LfsHipView mask, LfsHipView output, uint32_t dtype, uint32_t kind, size_t count, size_t capacity, uint64_t* selected, LfsHipError*);
enum LfsHipRandomKind { LFS_HIP_RANDOM_UNIFORM,
                        LFS_HIP_RANDOM_BERNOULLI,
                        LFS_HIP_RANDOM_INTEGER,
                        LFS_HIP_RANDOM_NORMAL };
typedef struct LfsHipRandom {
    uint32_t kind;
    uint64_t seed;
    size_t count;
    float first, second;
    int32_t low, high;
} LfsHipRandom;
LFS_HIP_API int lfs_hip_random(LfsHipContext*, LfsHipView output, const LfsHipRandom*, LfsHipError*);
LFS_HIP_API int lfs_hip_count_nonzero(LfsHipContext*, LfsHipView input, uint32_t dtype, size_t count, uint64_t* result, LfsHipError*);
LFS_HIP_API int lfs_hip_convert(LfsHipContext*, LfsHipView input, uint32_t input_dtype, LfsHipView output, uint32_t output_dtype, size_t count, LfsHipError*);
LFS_HIP_API int lfs_hip_index_cast(LfsHipContext*, LfsHipView input, LfsHipView output, size_t count, size_t extent, LfsHipError*);
LFS_HIP_API int lfs_hip_index_update(LfsHipContext*, LfsHipView output, uint32_t dtype, LfsHipView indices, uint32_t index_dtype, LfsHipView source, const LfsHipLayout* output_layout, uint32_t dim, size_t index_count, uint32_t add, LfsHipError*);
LFS_HIP_API int lfs_hip_index_fill(LfsHipContext*, LfsHipView output, uint32_t dtype, LfsHipView indices, uint32_t index_dtype, const LfsHipLayout* output_layout, uint32_t dim, size_t index_count, uint64_t value_bits, LfsHipError*);
LFS_HIP_API int lfs_hip_gather(LfsHipContext*, LfsHipView input, uint32_t dtype, LfsHipView indices, uint32_t index_dtype, LfsHipView output, const LfsHipLayout* input_layout, const LfsHipLayout* index_layout, uint32_t dim, uint32_t boundary_mode, uint32_t index_select, LfsHipError*);
/* Flat indices count negative positions from the end, then clamp to the extent,
 * matching the existing fused gather and flat index_put backend contracts. */
LFS_HIP_API int lfs_hip_gather_unary_f32(LfsHipContext*, LfsHipView input, LfsHipView indices, uint32_t index_dtype,
                                       LfsHipView output, size_t input_size, size_t index_count, uint32_t operation, LfsHipError*);
LFS_HIP_API int lfs_hip_index_put_f32(LfsHipContext*, LfsHipView output, LfsHipView indices, uint32_t index_dtype,
                                    LfsHipView values, size_t output_size, size_t index_count, LfsHipError*);
typedef struct LfsHipEnvironment {
    float rotation[9];
    int32_t full_width, full_height, band_width, band_height, y_offset;
    float focal_x, focal_y, center_x, center_y;
    int32_t equirect_view;
    float exposure_factor, env_rotation_radians;
    int32_t env_width, env_height;
} LfsHipEnvironment;
LFS_HIP_API int lfs_hip_environment_composite(LfsHipContext*, LfsHipView rgb, LfsHipView alpha,
                                               LfsHipView environment, LfsHipView output,
                                               const LfsHipEnvironment*, LfsHipError*);
LFS_HIP_API int lfs_hip_reduce_layout_f32(LfsHipContext*, LfsHipView input, LfsHipView output, const LfsHipLayout*, uint32_t axes_mask, uint32_t operation, LfsHipError*);
LFS_HIP_API int lfs_hip_arg_extreme_f32(LfsHipContext*, LfsHipView input, LfsHipView values, LfsHipView indices,
                                      size_t outer, size_t reduced, size_t inner, uint32_t maximum,
                                      size_t* workspace_growth_bytes, LfsHipError*);
LFS_HIP_API int lfs_hip_broadcast_f32(LfsHipContext*, uint32_t op, LfsHipView lhs, LfsHipView rhs, LfsHipView output, const LfsHipLayout* lhs_layout, const LfsHipLayout* rhs_layout, const LfsHipLayout* output_layout, uint32_t output_element_bytes, LfsHipError*);

#ifdef __cplusplus
}
#endif
