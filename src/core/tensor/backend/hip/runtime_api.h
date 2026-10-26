/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include <stddef.h>
#include <stdint.h>

/* Private, versioned C boundary: no CUDA/HIP headers or C++ ownership cross it.
 * Calls on one context must be serialized by the host. Buffers belong to their
 * context; destruction drains its queue before releasing live allocations.
 * Contexts must remain alive throughout their calls. Successfully released
 * handles are invalid and must not be reused, even if their address recurs.
 * A nonzero result leaves details in the optional caller-owned error buffer;
 * success clears it. native_code is a HIP error code, -2 for an invalid index,
 * or -1 for other host exceptions. */
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

/* Exact private ABI revision; check before binding any revision-specific exports. */
#define LFS_HIP_RUNTIME_ABI_VERSION 5u
LFS_HIP_API uint32_t lfs_hip_runtime_abi(void);
LFS_HIP_API int lfs_hip_device_count(int32_t* count, LfsHipError* error);
LFS_HIP_API int lfs_hip_device_info(int32_t ordinal, LfsHipDevice* info, LfsHipError* error);
LFS_HIP_API int lfs_hip_context_create(int32_t ordinal, LfsHipContext** context, LfsHipError* error);
LFS_HIP_API int lfs_hip_context_synchronize(LfsHipContext* context, LfsHipError* error);
LFS_HIP_API int lfs_hip_buffer_allocate(LfsHipContext* context, size_t bytes, LfsHipBuffer** buffer, LfsHipError* error);
LFS_HIP_API int lfs_hip_buffer_release(LfsHipContext* context, LfsHipBuffer* buffer, LfsHipError* error);
LFS_HIP_API int lfs_hip_context_destroy(LfsHipContext* context, LfsHipError* error);
/* Blocking transfers: the caller may reuse host storage on successful return. */
LFS_HIP_API int lfs_hip_upload(LfsHipContext* context, LfsHipBuffer* buffer, size_t offset, const void* host, size_t bytes, LfsHipError* error);
LFS_HIP_API int lfs_hip_download(LfsHipContext* context, const LfsHipBuffer* buffer, size_t offset, void* host, size_t bytes, LfsHipError* error);
/* Enqueued operations: synchronize or use a blocking transfer before consuming
 * results on the host. Immediate launch checks do not imply GPU completion. */
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
 * consume completion before reusing. The returned host address is borrowed;
 * it remains valid until capacity growth or destruction, and may only be read
 * after wait succeeds. Appends and poll are bounded by the reservation size,
 * independently of a larger cached capacity. */
LFS_HIP_API int lfs_hip_readback_reserve(LfsHipContext*, LfsHipReadback*, size_t bytes, uint64_t* host, LfsHipError*);
LFS_HIP_API int lfs_hip_readback_append(LfsHipContext*, LfsHipReadback*, const LfsHipBuffer*, size_t source_offset, size_t bytes, size_t destination_offset, LfsHipError*);
LFS_HIP_API int lfs_hip_readback_wait(LfsHipContext*, LfsHipReadback*, LfsHipError*);
/* Poll does not block. destination_bytes must describe the writable destination
 * extent; an insufficient extent fails without consuming the snapshot.
 * A valid snapshot is consumed only on success with ready=1; idle slots return 0.
 * wait alone does not consume completion or permit staging reuse. A copy or
 * captured execution error stays latched: wait/poll report it before exposing
 * bytes or ready=1. Execution errors are captured at enqueue/append/marker time;
 * work submitted later cannot invalidate an earlier successful snapshot.
 * Packed appends retain every captured error until consumption or discard.
 * Once drained, poll(nullptr, 0) discards a failed result, returning that error
 * with ready=0; the slot may then be reused. Event-record errors alone do not
 * invalidate an accepted copy, whose completion is checked through the stream.
 * Context synchronization reports/consumes its own pending error independently;
 * clearing it does not validate a failed snapshot. Borrowed host subranges may
 * overlap upload sources or poll destinations. */
LFS_HIP_API int lfs_hip_readback_poll(LfsHipContext*, LfsHipReadback*, void* destination, size_t destination_bytes, int32_t* ready, LfsHipError*);
LFS_HIP_API int lfs_hip_readback_destroy(LfsHipContext*, LfsHipReadback*, LfsHipError*);
LFS_HIP_API int lfs_hip_fill_f32(LfsHipContext* context, LfsHipBuffer* buffer, size_t offset, size_t count, float value, LfsHipError* error);
LFS_HIP_API int lfs_hip_affine_f32(LfsHipContext* context, const LfsHipBuffer* input, LfsHipBuffer* output, size_t count, float scale, float bias, LfsHipError* error);

/* Contiguous, context-owned views. Offsets and extents are bytes. Predicate
 * results are byte booleans, independently of the floating input storage.
 * Exact same-type input/output aliasing is supported; shifted overlap or
 * overlapping predicate output is rejected before a kernel is enqueued. */
typedef struct LfsHipView {
    LfsHipBuffer* buffer;
    size_t offset;
} LfsHipView;
enum LfsHipPointwiseOp {
#define LFS_POINTWISE_OP(Id, FunctorType, Name) LFS_HIP_OP_##Id,
#include "../../../include/core/detail/pointwise_ops.def"
#undef LFS_POINTWISE_OP
    LFS_HIP_OP_COUNT
};
LFS_HIP_API int lfs_hip_pointwise_f32(LfsHipContext*, uint32_t operation, LfsHipView lhs, LfsHipView rhs, LfsHipView output, size_t count, float scalar, LfsHipError*);
LFS_HIP_API int lfs_hip_pointwise_f16(LfsHipContext*, uint32_t operation, LfsHipView lhs, LfsHipView rhs, LfsHipView output, size_t count, float scalar, LfsHipError*);
LFS_HIP_API int lfs_hip_arange_f32(LfsHipContext*, LfsHipView output, size_t count, float start, float step, LfsHipError*);
/* Device copies enqueue on the same context queue. Identical source/destination
 * is a no-op; shifted overlapping ranges are rejected. */
LFS_HIP_API int lfs_hip_copy(LfsHipContext*, LfsHipView source, LfsHipView destination, size_t bytes, LfsHipError*);
LFS_HIP_API int lfs_hip_memset(LfsHipContext*, LfsHipView destination, size_t bytes, uint8_t value, LfsHipError*);

typedef struct LfsHipMemoryInfo {
    uint64_t total_bytes, free_bytes, owned_bytes, workspace_bytes;
} LfsHipMemoryInfo;
typedef struct LfsHipChainOp {
    uint32_t op;
    float scalar;
    LfsHipView rhs;
} LfsHipChainOp;
typedef struct LfsHipChain {
    uint32_t count;
    LfsHipChainOp ops[16];
} LfsHipChain;
typedef struct LfsHipLayout {
    uint32_t rank;
    uint64_t dims[8], strides[8], elements;
} LfsHipLayout;
typedef struct LfsHipLayoutCopy {
    LfsHipView input, output;
    LfsHipLayout input_layout, output_layout;
    uint32_t element_bytes;
} LfsHipLayoutCopy;
typedef struct LfsHipRandom {
    uint32_t kind;
    uint64_t seed;
    size_t count;
    float first, second;
    int32_t low, high;
} LfsHipRandom;
/* One opcode domain for every reduction entry point; never Core ReduceOp ordinals. */
enum LfsHipReduceOp { LFS_HIP_REDUCE_SUM,
                      LFS_HIP_REDUCE_MEAN,
                      LFS_HIP_REDUCE_MIN,
                      LFS_HIP_REDUCE_MAX,
                      LFS_HIP_REDUCE_HAS_NAN,
                      LFS_HIP_REDUCE_HAS_INF,
                      LFS_HIP_REDUCE_ANY,
                      LFS_HIP_REDUCE_ALL };
enum LfsHipDtype { LFS_HIP_F32 = 0,
                   LFS_HIP_F16 = 1,
                   LFS_HIP_I32 = 2,
                   LFS_HIP_I64 = 3,
                   LFS_HIP_U8 = 4,
                   LFS_HIP_BOOL = 5,
                   LFS_HIP_U32 = 6 };
enum LfsHipRandomKind { LFS_HIP_RANDOM_UNIFORM,
                        LFS_HIP_RANDOM_BERNOULLI,
                        LFS_HIP_RANDOM_INTEGER,
                        LFS_HIP_RANDOM_NORMAL };
enum { LFS_HIP_ERROR_INVALID_INDEX = -2 };
/* Invalid device indices are skipped and reported by context synchronization or
 * blocking host reads. Valid lanes may complete before that asynchronous error. */
LFS_HIP_API int lfs_hip_clamp_i32(LfsHipContext*, LfsHipView data, size_t count, int32_t minimum, int32_t maximum, LfsHipError*);
LFS_HIP_API int lfs_hip_scalar_left_typed(LfsHipContext*, uint32_t op, LfsHipView lhs, LfsHipView output, size_t count, float scalar, uint32_t dtype, LfsHipError*);
LFS_HIP_API int lfs_hip_broadcast_typed(LfsHipContext*, uint32_t op, LfsHipView lhs, LfsHipView rhs, LfsHipView output, const LfsHipLayout* lhs_layout, const LfsHipLayout* rhs_layout, const LfsHipLayout* output_layout, uint32_t input_dtype, uint32_t output_element_bytes, LfsHipError*);
LFS_HIP_API int lfs_hip_reduce_f32(LfsHipContext*, LfsHipView input, size_t count, uint32_t operation, float* output, LfsHipError*);
LFS_HIP_API int lfs_hip_copy_layouts(LfsHipContext*, const LfsHipLayoutCopy*, size_t count, LfsHipError*);
typedef struct LfsHipSelection {
    uint32_t struct_bytes, dtype, kind, count_known;
    uint64_t count, capacity, selected_count;
} LfsHipSelection;
/* count_known requires selected_count to describe this unchanged mask exactly.
 * It skips host count reads, matching the Tensor frontend's pre-counted contract. */
LFS_HIP_API int lfs_hip_select(LfsHipContext*, LfsHipView input, LfsHipView mask, LfsHipView output,
                               const LfsHipSelection*, uint64_t* selected, LfsHipError*);
LFS_HIP_API int lfs_hip_masked_scatter(LfsHipContext*, LfsHipView output, LfsHipView mask, LfsHipView source,
                                       uint32_t dtype, size_t count, size_t selected_count, LfsHipError*);
LFS_HIP_API int lfs_hip_sort_pairs(LfsHipContext*, LfsHipView values, LfsHipView indices, size_t outer, size_t dimension, size_t inner, uint32_t descending, LfsHipError*);
LFS_HIP_API int lfs_hip_cumsum(LfsHipContext*, LfsHipView data, uint32_t dtype, const LfsHipLayout*, uint32_t axis, LfsHipError*);
LFS_HIP_API int lfs_hip_trim_workspace(LfsHipContext*, LfsHipError*);
LFS_HIP_API int lfs_hip_memory_info(LfsHipContext*, LfsHipMemoryInfo*, LfsHipError*);
LFS_HIP_API int lfs_hip_chain_f32(LfsHipContext*, LfsHipView input, LfsHipView output, size_t count, const LfsHipChain*, LfsHipError*);
LFS_HIP_API int lfs_hip_strided_scatter(LfsHipContext*, LfsHipView input, LfsHipView output, const LfsHipLayout*, uint32_t element_bytes, uint32_t int32_to_float32, LfsHipError*);
LFS_HIP_API int lfs_hip_strided_copy(LfsHipContext*, LfsHipView input, LfsHipView output, const LfsHipLayout*, uint32_t element_bytes, LfsHipError*);
LFS_HIP_API int lfs_hip_random(LfsHipContext*, LfsHipView output, const LfsHipRandom*, LfsHipError*);
LFS_HIP_API int lfs_hip_count_nonzero(LfsHipContext*, LfsHipView input, uint32_t dtype, size_t count, uint64_t* result, LfsHipError*);
LFS_HIP_API int lfs_hip_convert(LfsHipContext*, LfsHipView input, uint32_t input_dtype, LfsHipView output, uint32_t output_dtype, size_t count, LfsHipError*);
LFS_HIP_API int lfs_hip_index_cast(LfsHipContext*, LfsHipView input, LfsHipView output, size_t count, size_t extent, LfsHipError*);
typedef struct LfsHipIndexUpdate {
    uint32_t struct_bytes, dtype, index_dtype, dim, add;
    uint64_t index_count;
    LfsHipLayout output_layout;
} LfsHipIndexUpdate;
typedef struct LfsHipIndexFill {
    uint32_t struct_bytes, dtype, index_dtype, dim;
    uint64_t index_count, value_bits;
    LfsHipLayout output_layout;
} LfsHipIndexFill;
typedef struct LfsHipGather {
    uint32_t struct_bytes, dtype, index_dtype, dim, boundary_mode, index_select;
    LfsHipLayout input_layout, index_layout;
} LfsHipGather;
LFS_HIP_API int lfs_hip_index_update(LfsHipContext*, LfsHipView output, LfsHipView indices, LfsHipView source, const LfsHipIndexUpdate*, LfsHipError*);
LFS_HIP_API int lfs_hip_index_fill(LfsHipContext*, LfsHipView output, LfsHipView indices, const LfsHipIndexFill*, LfsHipError*);
LFS_HIP_API int lfs_hip_gather_unary_f32(LfsHipContext*, LfsHipView input, LfsHipView indices, uint32_t index_dtype,
                                         LfsHipView output, size_t input_size, size_t index_count, uint32_t operation, LfsHipError*);
LFS_HIP_API int lfs_hip_index_put_f32(LfsHipContext*, LfsHipView output, LfsHipView indices, uint32_t index_dtype,
                                      LfsHipView values, size_t output_size, size_t index_count, LfsHipError*);
LFS_HIP_API int lfs_hip_gather(LfsHipContext*, LfsHipView input, LfsHipView indices, LfsHipView output, const LfsHipGather*, LfsHipError*);
LFS_HIP_API int lfs_hip_reduce_layout_f32(LfsHipContext*, LfsHipView input, LfsHipView output, const LfsHipLayout*, uint32_t axes_mask, uint32_t operation, LfsHipError*);
LFS_HIP_API int lfs_hip_fill_layout_bits(LfsHipContext*, LfsHipView output, const LfsHipLayout*, uint32_t element_bytes, uint64_t value_bits, LfsHipError*);
LFS_HIP_API int lfs_hip_chain_reduce_f32(LfsHipContext*, LfsHipView input, LfsHipView output, size_t segments, size_t segment_size, const LfsHipChain*, uint32_t operation, LfsHipError*);
LFS_HIP_API int lfs_hip_arg_extreme_f32(LfsHipContext*, LfsHipView input, LfsHipView values, LfsHipView indices,
                                        size_t outer, size_t reduced, size_t inner, uint32_t maximum,
                                        LfsHipError*);
LFS_HIP_API int lfs_hip_cdist_f32(LfsHipContext*, LfsHipView lhs, LfsHipView rhs, LfsHipView output,
                                  size_t lhs_rows, size_t rhs_rows, size_t columns, float p, LfsHipError*);

#ifdef __cplusplus
}
#endif
