/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "../../core/tensor/backend/hip/runtime_api.h"

#if defined(__HIP_DEVICE_COMPILE__)
#define LFS_HIP_TRAIN_API
#elif defined(_WIN32)
#ifdef LFS_HIP_TRAIN_EXPORTS
#define LFS_HIP_TRAIN_API __declspec(dllexport)
#else
#define LFS_HIP_TRAIN_API __declspec(dllimport)
#endif
#else
#define LFS_HIP_TRAIN_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct LfsHipTraining LfsHipTraining;
enum LfsHipTrainStatus {
    LFS_HIP_TRAIN_OK = 0,
    LFS_HIP_TRAIN_INVALID = 1,
    LFS_HIP_TRAIN_RESOURCE_EXHAUSTED = 2,
    LFS_HIP_TRAIN_GPU_ERROR = 3,
    LFS_HIP_TRAIN_CANCELLED = 4,
    LFS_HIP_TRAIN_POISONED = 5
};

typedef struct LfsHipTrainConfig {
    uint32_t struct_bytes;
    int32_t device_ordinal;
    uint64_t gaussian_count;
    /* Hard cap for allocations owned by this session, including replacement
     * overlap. Does not represent the driver's total UMA/VRAM consumption. */
    uint64_t allocation_budget_bytes;
} LfsHipTrainConfig;

typedef struct LfsHipTrainModel {
    const float* means;         /* N*3, xyz */
    const float* quaternions;   /* N*4, wxyz, nonzero norm */
    const float* log_scales;    /* N*3 */
    const float* sh0;           /* N*3, RGB = max(0, SH_C0 * coefficient + 0.5) */
    const float* raw_opacities; /* N, sigmoid activation in rasterizer */
} LfsHipTrainModel;

typedef struct LfsHipTrainModelLayout {
    uint32_t struct_bytes;
    uint64_t required_bytes;
    uint64_t offsets_bytes[5]; /* Same family order as read_parameter. */
    uint64_t float_counts[5];
} LfsHipTrainModelLayout;

/* Experimental Windows dedicated OPAQUE_WIN32 allocation, bound at offset zero.
 * Caller retains the NT handle and Vulkan allocation; both must outlive create,
 * and the Vulkan allocation must outlive the session. Caller must release Vulkan
 * queue ownership to EXTERNAL and wait for completion before create/step, then
 * acquire ownership only after the synchronous HIP call returns. No simultaneous
 * access is supported. The full imported allocation counts against the budget.
 * Vulkan deviceLUID/node mask are checked against the selected HIP adapter. */
typedef struct LfsHipTrainSharedModel {
    uint32_t struct_bytes;
    uint32_t device_node_mask;
    uint8_t device_luid[8];
    uint64_t win32_handle;
    uint64_t allocation_bytes;
} LfsHipTrainSharedModel;

/* Borrowed native HIP allocation, initialized by the application. Must be an
 * allocation base on config.device_ordinal, not an interior/host/managed pointer.
 * Caller orders all producers before attachment and retains the allocation
 * until session destruction. Calls finish before returning; no concurrent access.
 * The allocation is charged once to this session's budget but never freed by it. */
typedef struct LfsHipTrainDeviceModel {
    uint32_t struct_bytes;
    int32_t device_ordinal;
    uint64_t address;
    uint64_t allocation_bytes;
} LfsHipTrainDeviceModel;

typedef struct LfsHipTrainFrame {
    uint32_t struct_bytes;
    uint32_t width;
    uint32_t height;
    float world_to_camera[16]; /* Row-major rigid transform, global pinhole */
    float fx, fy, cx, cy;
    float near_plane, far_plane;
    const float* target_chw; /* width*height*3 linear float RGB, consumed on return */
} LfsHipTrainFrame;

typedef struct LfsHipTrainRates {
    float means, quaternions, log_scales, sh0, raw_opacities;
} LfsHipTrainRates;

enum LfsHipTrainLossKind { LFS_HIP_LOSS_MSE = 0,
                           LFS_HIP_LOSS_L1_SSIM = 1 };
typedef struct LfsHipTrainLoss {
    uint32_t struct_bytes;
    uint32_t kind;
    float ssim_weight;      /* [0,1]; MSE requires zero. */
    uint32_t valid_padding; /* 0/1; fused path crops 5 pixels per dimension >10. */
} LfsHipTrainLoss;

typedef struct LfsHipTrainSh {
    uint32_t struct_bytes;
    uint32_t max_degree;
    uint32_t active_degree;
    float rest_learning_rate;
    const float* canonical_rest; /* [N, (max_degree+1)^2-1, 3], RGB */
    uint64_t float_count;
} LfsHipTrainSh;

/* FP32 float4-swizzled SH-rest view within the session's existing model
 * allocation. No value copy or second allocation charge. The model allocation
 * and its producer/consumer synchronization cover this region as well. */
typedef struct LfsHipTrainResidentSh {
    uint32_t struct_bytes;
    uint32_t max_degree;
    uint32_t active_degree;
    float rest_learning_rate;
    uint64_t offset_bytes;
} LfsHipTrainResidentSh;

typedef struct LfsHipTrainInfo {
    uint64_t completed_steps;
    uint64_t allocated_bytes;
    uint64_t peak_allocated_bytes;
    uint64_t budget_bytes;
    uint64_t intersection_count;
    uint64_t intersection_capacity;
    uint32_t width, height;
    uint32_t poisoned;
} LfsHipTrainInfo;

/* Experimental fixed-count SH0..3 session, no refinement or viewer integration yet.
 * Calls on one session must be serialized, except set_cancel may run concurrently
 * while a step is running. Stop/join all calls before destruction or module unload.
 * Uploads, steps, renders and downloads complete before returning. GPU failures
 * poison a session; they do not silently switch backend. Invalid/budget/cancel
 * failures before Adam preserve parameters and optimizer step. */
LFS_HIP_TRAIN_API uint32_t lfs_hip_train_abi(void);
LFS_HIP_TRAIN_API const char* lfs_hip_train_targets(void);
LFS_HIP_TRAIN_API const char* lfs_hip_train_set_id(void);
LFS_HIP_TRAIN_API int lfs_hip_train_create(const LfsHipTrainConfig*, const LfsHipTrainModel*, LfsHipTraining**, LfsHipError*);
LFS_HIP_TRAIN_API int lfs_hip_train_model_layout(uint64_t gaussian_count, LfsHipTrainModelLayout*, LfsHipError*);
LFS_HIP_TRAIN_API int lfs_hip_train_create_shared(const LfsHipTrainConfig*, const LfsHipTrainModel*, const LfsHipTrainSharedModel*, LfsHipTraining**, LfsHipError*);
/* Explicit base-family offsets for an application-owned packed allocation.
 * Layout counts must equal the live model; offsets are 256-byte aligned and
 * disjoint, required_bytes includes any reserved gaps. Creation uploads only
 * live elements, preserving all padding/reserved rows. Ownership/synchronization
 * are the same as create_shared. SH-rest remains private. */
LFS_HIP_TRAIN_API int lfs_hip_train_create_shared_layout(const LfsHipTrainConfig*, const LfsHipTrainModel*, const LfsHipTrainSharedModel*, const LfsHipTrainModelLayout*, LfsHipTraining**, LfsHipError*);
/* Attach an already initialized allocation without a model upload/readback.
 * Validates live parameters on GPU and reads back only a scalar status.
 * Does not restore optimizer history: new Adam state starts at step zero. */
LFS_HIP_TRAIN_API int lfs_hip_train_attach_shared_layout(const LfsHipTrainConfig*, const LfsHipTrainSharedModel*, const LfsHipTrainModelLayout*, LfsHipTraining**, LfsHipError*);
LFS_HIP_TRAIN_API int lfs_hip_train_attach_device_layout(const LfsHipTrainConfig*, const LfsHipTrainDeviceModel*, const LfsHipTrainModelLayout*, LfsHipTraining**, LfsHipError*);
/* Optional owned host staging. Caller retains the module until successful free.
 * Does not attach a target or borrow memory beyond set_frame's synchronous copy. */
LFS_HIP_TRAIN_API int lfs_hip_train_host_alloc(int device, size_t bytes, void** output, LfsHipError*);
LFS_HIP_TRAIN_API int lfs_hip_train_host_free(int device, void* pointer, LfsHipError*);
LFS_HIP_TRAIN_API int lfs_hip_train_destroy(LfsHipTraining*, LfsHipError*);
LFS_HIP_TRAIN_API int lfs_hip_train_set_frame(LfsHipTraining*, const LfsHipTrainFrame*, LfsHipError*);
/* Explicit experiment selection before any frame or optimizer step. 0=3DGUT,
 * 1=FastGS (requires opt-in LFS_HIP_FAST_RUNTIME build). Never selected by default. */
LFS_HIP_TRAIN_API int lfs_hip_train_set_rasterizer(LfsHipTraining*, uint32_t kind, LfsHipError*);
/* Before frame/step: 0=FP32 reference, 1=FastGS fused joint moments (16-bit base,
 * 8-bit SH-rest). Parameters and arithmetic remain FP32. Recreate the session
 * to return from compressed moments to FP32; conversion is not implicit. */
LFS_HIP_TRAIN_API int lfs_hip_train_set_optimizer(LfsHipTraining*, uint32_t kind, LfsHipError*);
LFS_HIP_TRAIN_API int lfs_hip_train_step(LfsHipTraining*, const LfsHipTrainRates*, LfsHipError*);
LFS_HIP_TRAIN_API int lfs_hip_train_render(LfsHipTraining*, float* output_chw, size_t float_count, LfsHipError*);
/* Family indices: means=0, quaternion=1, log_scale=2, sh0=3, raw_opacity=4,
 * canonical SH-rest=5 (only after SH configuration). */
LFS_HIP_TRAIN_API int lfs_hip_train_read_parameter(LfsHipTraining*, uint32_t family, float* output, size_t float_count, LfsHipError*);
LFS_HIP_TRAIN_API int lfs_hip_train_info(LfsHipTraining*, LfsHipTrainInfo*, LfsHipError*);
LFS_HIP_TRAIN_API int lfs_hip_train_set_budget(LfsHipTraining*, uint64_t allocation_budget_bytes, LfsHipError*);
LFS_HIP_TRAIN_API int lfs_hip_train_set_loss(LfsHipTraining*, const LfsHipTrainLoss*, LfsHipError*);
/* Initial SH storage may be configured at any step when none exists, allowing
 * allocation to be deferred until the first SH activation. Replacing an existing
 * SH store is allowed only before the first optimizer step. SH-rest storage and
 * moments are currently private HIP allocations, even with a shared SH0 model.
 * Active degree may subsequently change without reallocating. Inactive bases
 * preserve values and moments. Bias correction uses the session's global step.
 * Viewing directions are detached, matching the existing gsplat training path. */
LFS_HIP_TRAIN_API int lfs_hip_train_set_sh(LfsHipTraining*, const LfsHipTrainSh*, LfsHipError*);
/* Attach initialized values before the first optimizer update. Requires a
 * borrowed/imported model allocation and a disjoint, aligned resident region.
 * Uses fresh optimizer moments; frozen/growth/checkpoint state is not inferred. */
LFS_HIP_TRAIN_API int lfs_hip_train_attach_resident_sh(LfsHipTraining*, const LfsHipTrainResidentSh*, LfsHipError*);
LFS_HIP_TRAIN_API int lfs_hip_train_set_sh_degree(LfsHipTraining*, uint32_t degree, float rest_learning_rate, LfsHipError*);
/* Diagnostic evaluation renders and reduces on GPU; no image is read back.
 * Does not change parameters/moments or optimizer step. */
LFS_HIP_TRAIN_API int lfs_hip_train_loss(LfsHipTraining*, float* output, LfsHipError*);
LFS_HIP_TRAIN_API void lfs_hip_train_set_cancel(LfsHipTraining*, int cancelled);

#ifdef __cplusplus
}
#endif
