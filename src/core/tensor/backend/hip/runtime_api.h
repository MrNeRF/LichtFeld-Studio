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
 * success clears it. native_code is a HIP error code, or -1 for a host exception. */
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
 * Completion is consumed only on success with ready=1; idle slots return 0.
 * wait alone does not consume completion or permit staging reuse. A copy error
 * stays latched: wait/poll report it and never return a successful snapshot.
 * Once drained, poll(nullptr, 0) discards a failed result, returning that error
 * with ready=0; the slot may then be reused. Event-record errors alone do not
 * invalidate an accepted copy, whose completion is checked through the stream.
 * Borrowed host subranges may overlap upload sources or poll destinations. */
LFS_HIP_API int lfs_hip_readback_poll(LfsHipContext*, LfsHipReadback*, void* destination, size_t destination_bytes, int32_t* ready, LfsHipError*);
LFS_HIP_API int lfs_hip_readback_destroy(LfsHipContext*, LfsHipReadback*, LfsHipError*);
LFS_HIP_API int lfs_hip_fill_f32(LfsHipContext* context, LfsHipBuffer* buffer, size_t offset, size_t count, float value, LfsHipError* error);
LFS_HIP_API int lfs_hip_affine_f32(LfsHipContext* context, const LfsHipBuffer* input, LfsHipBuffer* output, size_t count, float scale, float bias, LfsHipError* error);

#ifdef __cplusplus
} // extern "C"
#endif
