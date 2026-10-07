add_library(lfs_training_vulkan STATIC
    ../ops/fast_backward_vulkan.cpp
    ../ops/fast_vulkan.cpp
    ../ops/gsplat_vulkan.cpp
    dispatch.cpp
    ../ops/blob_vulkan.cpp
    ../ops/structure_vulkan.cpp
    ../ops/flip_vulkan.cpp
    ../ops/photometric_vulkan.cpp
    ../ops/session_vulkan.cpp
    ../ops/extra_loss_vulkan.cpp
    ../ops/geometry_vulkan.cpp
    ../ops/masks_vulkan.cpp
    ../ops/bilateral_vulkan.cpp
    ../ops/ppisp_vulkan.cpp
    ../ops/controller_vulkan.cpp
    ../ops/adam_vulkan.cpp
    ../ops/sh_vulkan.cpp
    ../ops/pair_sort_vulkan.cpp
    ../ops/training_image_vulkan.cpp
    ../ops/refine_vulkan.cpp
    ../ops/mrnf_vulkan.cpp
    ../ops/morton_vulkan.cpp
    ../ops/mcmc_vulkan.cpp
    ../ops/lpips_vulkan.cpp
)
set(LFS_TRAINING_SPV_DIR "${CMAKE_BINARY_DIR}/lfs_training_spv")
file(MAKE_DIRECTORY "${LFS_TRAINING_SPV_DIR}")
set(_pair_sort_source "${CMAKE_CURRENT_SOURCE_DIR}/shaders/pair_sort.slang")
set(_pair_sort_raw "${LFS_TRAINING_SPV_DIR}/pair_sort.raw.spv")
set(_pair_sort_spv "${LFS_TRAINING_SPV_DIR}/pair_sort.spv")
set(_pair_sort_manifest "${LFS_TRAINING_SPV_DIR}/pair_sort.json")
set(_pair_sort_match_raw "${LFS_TRAINING_SPV_DIR}/pair_sort_match.raw.spv")
set(_pair_sort_match_spv "${LFS_TRAINING_SPV_DIR}/pair_sort_match.spv")
set(_pair_sort_match_manifest "${LFS_TRAINING_SPV_DIR}/pair_sort_match.json")
add_custom_command(
    OUTPUT "${_pair_sort_spv}" "${_pair_sort_manifest}"
    BYPRODUCTS "${_pair_sort_raw}"
    COMMAND ${LFS_TENSOR_SLANGC} "${_pair_sort_source}"
            -entry main -stage compute -O -fp-mode precise
            -line-directive-mode none -target spirv -profile glsl_460
            -emit-spirv-directly -D PAIR_SORT_SUBGROUP_MATCH=0
            -I "${CMAKE_CURRENT_SOURCE_DIR}/shaders"
            -I "${CMAKE_SOURCE_DIR}/src/core/tensor/backend/vulkan/shaders"
            -o "${_pair_sort_raw}"
    COMMAND lfs_tensor_spirv_finalize "${_pair_sort_raw}" "${_pair_sort_spv}" "${_pair_sort_manifest}"
    DEPENDS "${_pair_sort_source}" lfs_tensor_spirv_finalize
    COMMENT "slangc training/pair_sort.spv"
    VERBATIM)
set(_geometry_source "${CMAKE_CURRENT_SOURCE_DIR}/shaders/geometry.slang")
set(_geometry_spv "${LFS_TRAINING_SPV_DIR}/geometry.spv")
set(_geometry_fp32_spv "${LFS_TRAINING_SPV_DIR}/geometry_fp32.spv")
# geometry_fp32 serves devices without shaderFloat64 (Apple GPUs via MoltenVK).
foreach(_variant IN ITEMS geometry geometry_fp32)
    if(_variant STREQUAL "geometry_fp32")
        set(_geometry_fp64 0)
    else()
        set(_geometry_fp64 1)
    endif()
    add_custom_command(
        OUTPUT "${LFS_TRAINING_SPV_DIR}/${_variant}.spv" "${LFS_TRAINING_SPV_DIR}/${_variant}.json"
        BYPRODUCTS "${LFS_TRAINING_SPV_DIR}/${_variant}.raw.spv"
        COMMAND ${LFS_TENSOR_SLANGC} "${_geometry_source}"
                -entry main -stage compute -O -fp-mode precise
                -line-directive-mode none -target spirv -profile glsl_460
                -emit-spirv-directly -D GEOMETRY_FP64=${_geometry_fp64}
                -o "${LFS_TRAINING_SPV_DIR}/${_variant}.raw.spv"
        COMMAND lfs_tensor_spirv_finalize "${LFS_TRAINING_SPV_DIR}/${_variant}.raw.spv"
                "${LFS_TRAINING_SPV_DIR}/${_variant}.spv" "${LFS_TRAINING_SPV_DIR}/${_variant}.json"
        DEPENDS "${_geometry_source}" lfs_tensor_spirv_finalize
        COMMENT "slangc training/${_variant}.spv"
        VERBATIM)
endforeach()
set(_sh_ops_source "${CMAKE_CURRENT_SOURCE_DIR}/shaders/sh_ops.slang")
set(_sh_ops_raw "${LFS_TRAINING_SPV_DIR}/sh_ops.raw.spv")
set(_sh_ops_spv "${LFS_TRAINING_SPV_DIR}/sh_ops.spv")
set(_sh_ops_manifest "${LFS_TRAINING_SPV_DIR}/sh_ops.json")
add_custom_command(
    OUTPUT "${_sh_ops_spv}" "${_sh_ops_manifest}"
    BYPRODUCTS "${_sh_ops_raw}"
    COMMAND ${LFS_TENSOR_SLANGC} "${_sh_ops_source}"
            -entry main -stage compute -O -fp-mode precise
            -line-directive-mode none -target spirv -profile glsl_460
            -emit-spirv-directly
            -I "${CMAKE_CURRENT_SOURCE_DIR}/shaders"
            -I "${CMAKE_SOURCE_DIR}/src/core/tensor/backend/vulkan/shaders"
            -o "${_sh_ops_raw}"
    COMMAND lfs_tensor_spirv_finalize "${_sh_ops_raw}" "${_sh_ops_spv}" "${_sh_ops_manifest}"
    DEPENDS "${_sh_ops_source}" "${CMAKE_SOURCE_DIR}/src/core/tensor/backend/vulkan/shaders/sh_codec.slangh"
            "${CMAKE_SOURCE_DIR}/src/core/tensor/backend/vulkan/shaders/sh_decode.slangh" "${CMAKE_SOURCE_DIR}/src/core/include/core/rounded_math.slangh" lfs_tensor_spirv_finalize
    COMMENT "slangc training/sh_ops.spv"
    VERBATIM)
set(_extra_loss_source "${CMAKE_CURRENT_SOURCE_DIR}/shaders/extra_loss.slang")
set(_extra_loss_raw "${LFS_TRAINING_SPV_DIR}/extra_loss.raw.spv")
set(_extra_loss_spv "${LFS_TRAINING_SPV_DIR}/extra_loss.spv")
set(_extra_loss_manifest "${LFS_TRAINING_SPV_DIR}/extra_loss.json")
add_custom_command(
    OUTPUT "${_extra_loss_spv}" "${_extra_loss_manifest}"
    BYPRODUCTS "${_extra_loss_raw}"
    COMMAND ${LFS_TENSOR_SLANGC} "${_extra_loss_source}"
            -entry main -stage compute -O -fp-mode precise
            -line-directive-mode none -target spirv -profile glsl_460
            -emit-spirv-directly
            -I "${CMAKE_CURRENT_SOURCE_DIR}/shaders"
            -o "${_extra_loss_raw}"
    COMMAND lfs_tensor_spirv_finalize "${_extra_loss_raw}" "${_extra_loss_spv}" "${_extra_loss_manifest}"
    DEPENDS "${_extra_loss_source}" lfs_tensor_spirv_finalize
    COMMENT "slangc training/extra_loss.spv"
    VERBATIM)
set(_adam_source "${CMAKE_CURRENT_SOURCE_DIR}/shaders/adam.slang")
set(_adam_raw "${LFS_TRAINING_SPV_DIR}/adam.raw.spv")
set(_adam_spv "${LFS_TRAINING_SPV_DIR}/adam.spv")
set(_adam_manifest "${LFS_TRAINING_SPV_DIR}/adam.json")
set(_morton_source "${CMAKE_CURRENT_SOURCE_DIR}/shaders/morton.slang")
set(_morton_raw "${LFS_TRAINING_SPV_DIR}/morton.raw.spv")
set(_morton_spv "${LFS_TRAINING_SPV_DIR}/morton.spv")
set(_morton_manifest "${LFS_TRAINING_SPV_DIR}/morton.json")
set(_joint_morton_source "${CMAKE_CURRENT_SOURCE_DIR}/shaders/joint_morton.slang")
set(_joint_morton_raw "${LFS_TRAINING_SPV_DIR}/joint_morton.raw.spv")
set(_joint_morton_spv "${LFS_TRAINING_SPV_DIR}/joint_morton.spv")
set(_joint_morton_manifest "${LFS_TRAINING_SPV_DIR}/joint_morton.json")
set(_mcmc_source "${CMAKE_CURRENT_SOURCE_DIR}/shaders/mcmc.slang")
set(_mcmc_raw "${LFS_TRAINING_SPV_DIR}/mcmc.raw.spv")
set(_mcmc_spv "${LFS_TRAINING_SPV_DIR}/mcmc.spv")
set(_mcmc_manifest "${LFS_TRAINING_SPV_DIR}/mcmc.json")
set(_lpips_source "${CMAKE_CURRENT_SOURCE_DIR}/shaders/lpips.slang")
set(_lpips_raw "${LFS_TRAINING_SPV_DIR}/lpips.raw.spv")
set(_lpips_spv "${LFS_TRAINING_SPV_DIR}/lpips.spv")
set(_lpips_manifest "${LFS_TRAINING_SPV_DIR}/lpips.json")
add_custom_command(
    OUTPUT "${_morton_spv}" "${_morton_manifest}"
    BYPRODUCTS "${_morton_raw}"
    COMMAND ${LFS_TENSOR_SLANGC} "${_morton_source}"
            -entry main -stage compute -O -fp-mode precise
            -line-directive-mode none -target spirv -profile glsl_460
            -emit-spirv-directly -o "${_morton_raw}"
    COMMAND lfs_tensor_spirv_finalize "${_morton_raw}" "${_morton_spv}" "${_morton_manifest}"
    DEPENDS "${_morton_source}" lfs_tensor_spirv_finalize
    COMMENT "slangc training/morton.spv"
    VERBATIM)
add_custom_command(
    OUTPUT "${_joint_morton_spv}" "${_joint_morton_manifest}"
    BYPRODUCTS "${_joint_morton_raw}"
    COMMAND ${LFS_TENSOR_SLANGC} "${_joint_morton_source}"
            -entry main -stage compute -O -fp-mode precise
            -line-directive-mode none -target spirv -profile glsl_460
            -emit-spirv-directly
            -I "${CMAKE_SOURCE_DIR}/src/core/tensor/backend/vulkan/shaders"
            -o "${_joint_morton_raw}"
    COMMAND lfs_tensor_spirv_finalize "${_joint_morton_raw}" "${_joint_morton_spv}" "${_joint_morton_manifest}"
    DEPENDS "${_joint_morton_source}" "${CMAKE_SOURCE_DIR}/src/core/tensor/backend/vulkan/shaders/byte_read.slangh"
            "${CMAKE_SOURCE_DIR}/src/core/tensor/backend/vulkan/shaders/byte_word.slangh" lfs_tensor_spirv_finalize
    COMMENT "slangc training/joint_morton.spv"
    VERBATIM)
add_custom_command(
    OUTPUT "${_mcmc_spv}" "${_mcmc_manifest}"
    BYPRODUCTS "${_mcmc_raw}"
    COMMAND ${LFS_TENSOR_SLANGC} "${_mcmc_source}"
            -entry main -stage compute -O -fp-mode precise
            -line-directive-mode none -target spirv -profile glsl_460
            -emit-spirv-directly -o "${_mcmc_raw}"
    COMMAND lfs_tensor_spirv_finalize "${_mcmc_raw}" "${_mcmc_spv}" "${_mcmc_manifest}"
    DEPENDS "${_mcmc_source}" lfs_tensor_spirv_finalize
    COMMENT "slangc training/mcmc.spv"
    VERBATIM)
add_custom_command(
    OUTPUT "${_lpips_spv}" "${_lpips_manifest}"
    BYPRODUCTS "${_lpips_raw}"
    COMMAND ${LFS_TENSOR_SLANGC} "${_lpips_source}"
            -entry main -stage compute -O -fp-mode precise
            -line-directive-mode none -target spirv -profile glsl_460
            -emit-spirv-directly
            -I "${CMAKE_SOURCE_DIR}/src/core/tensor/backend/vulkan/shaders"
            -o "${_lpips_raw}"
    COMMAND lfs_tensor_spirv_finalize "${_lpips_raw}" "${_lpips_spv}" "${_lpips_manifest}"
    DEPENDS "${_lpips_source}" "${CMAKE_CURRENT_SOURCE_DIR}/shaders/lpips_common.slangh" lfs_tensor_spirv_finalize
            "${CMAKE_SOURCE_DIR}/src/core/tensor/backend/vulkan/shaders/half_bits.slangh"
    COMMENT "slangc training/lpips.spv"
    VERBATIM)
set(_lpips_coop_source "${CMAKE_CURRENT_SOURCE_DIR}/shaders/lpips_coop.slang")
set(_lpips_coop_raw "${LFS_TRAINING_SPV_DIR}/lpips_coop.raw.spv")
set(_lpips_coop_spv "${LFS_TRAINING_SPV_DIR}/lpips_coop.spv")
set(_lpips_coop_manifest "${LFS_TRAINING_SPV_DIR}/lpips_coop.json")
add_custom_command(
    OUTPUT "${_lpips_coop_spv}" "${_lpips_coop_manifest}"
    BYPRODUCTS "${_lpips_coop_raw}"
    COMMAND ${LFS_TENSOR_SLANGC} "${_lpips_coop_source}"
            -entry main -stage compute -O -fp-mode precise
            -line-directive-mode none -target spirv -profile glsl_460+cooperative_matrix
            -emit-spirv-directly
            -I "${CMAKE_SOURCE_DIR}/src/core/tensor/backend/vulkan/shaders"
            -o "${_lpips_coop_raw}"
    COMMAND lfs_tensor_spirv_finalize "${_lpips_coop_raw}" "${_lpips_coop_spv}" "${_lpips_coop_manifest}"
    DEPENDS "${_lpips_coop_source}" "${CMAKE_CURRENT_SOURCE_DIR}/shaders/lpips_common.slangh" lfs_tensor_spirv_finalize
            "${CMAKE_SOURCE_DIR}/src/core/tensor/backend/vulkan/shaders/half_bits.slangh"
    COMMENT "slangc training/lpips_coop.spv"
    VERBATIM)
set(_lpips_halo_source "${CMAKE_CURRENT_SOURCE_DIR}/shaders/lpips_halo.slang")
set(_lpips_halo_raw "${LFS_TRAINING_SPV_DIR}/lpips_halo.raw.spv")
set(_lpips_halo_spv "${LFS_TRAINING_SPV_DIR}/lpips_halo.spv")
set(_lpips_halo_manifest "${LFS_TRAINING_SPV_DIR}/lpips_halo.json")
add_custom_command(
    OUTPUT "${_lpips_halo_spv}" "${_lpips_halo_manifest}"
    BYPRODUCTS "${_lpips_halo_raw}"
    COMMAND ${LFS_TENSOR_SLANGC} "${_lpips_halo_source}"
            -entry main -stage compute -O -fp-mode precise
            -line-directive-mode none -target spirv -profile glsl_460+cooperative_matrix
            -emit-spirv-directly
            -I "${CMAKE_SOURCE_DIR}/src/core/tensor/backend/vulkan/shaders"
            -o "${_lpips_halo_raw}"
    COMMAND lfs_tensor_spirv_finalize "${_lpips_halo_raw}" "${_lpips_halo_spv}" "${_lpips_halo_manifest}"
    DEPENDS "${_lpips_halo_source}" "${CMAKE_CURRENT_SOURCE_DIR}/shaders/lpips_common.slangh" lfs_tensor_spirv_finalize
            "${CMAKE_SOURCE_DIR}/src/core/tensor/backend/vulkan/shaders/half_bits.slangh"
    COMMENT "slangc training/lpips_halo.spv"
    VERBATIM)
add_custom_command(
    OUTPUT "${_adam_spv}" "${_adam_manifest}"
    BYPRODUCTS "${_adam_raw}"
    COMMAND ${LFS_TENSOR_SLANGC} "${_adam_source}"
            -entry main -stage compute -O -fp-mode precise
            -line-directive-mode none -target spirv -profile glsl_460
            -emit-spirv-directly
            -o "${_adam_raw}"
    COMMAND lfs_tensor_spirv_finalize "${_adam_raw}" "${_adam_spv}" "${_adam_manifest}"
    DEPENDS "${_adam_source}" "${CMAKE_SOURCE_DIR}/src/core/include/core/rounded_math.slangh" lfs_tensor_spirv_finalize
    COMMENT "slangc training/adam.spv"
    VERBATIM)
set(_masks_source "${CMAKE_CURRENT_SOURCE_DIR}/shaders/masks.slang")
set(_masks_raw "${LFS_TRAINING_SPV_DIR}/masks.raw.spv")
set(_masks_spv "${LFS_TRAINING_SPV_DIR}/masks.spv")
set(_masks_manifest "${LFS_TRAINING_SPV_DIR}/masks.json")
add_custom_command(
    OUTPUT "${_masks_spv}" "${_masks_manifest}"
    BYPRODUCTS "${_masks_raw}"
    COMMAND ${LFS_TENSOR_SLANGC} "${_masks_source}"
            -entry main -stage compute -O -fp-mode precise
            -line-directive-mode none -target spirv -profile glsl_460
            -emit-spirv-directly
            -o "${_masks_raw}"
    COMMAND lfs_tensor_spirv_finalize "${_masks_raw}" "${_masks_spv}" "${_masks_manifest}"
    DEPENDS "${_masks_source}" lfs_tensor_spirv_finalize
    COMMENT "slangc training/masks.spv"
    VERBATIM)
set(_mesh_mask_source "${CMAKE_CURRENT_SOURCE_DIR}/shaders/mesh_mask.slang")
set(_mesh_mask_raw "${LFS_TRAINING_SPV_DIR}/mesh_mask.raw.spv")
set(_mesh_mask_spv "${LFS_TRAINING_SPV_DIR}/mesh_mask.spv")
set(_mesh_mask_manifest "${LFS_TRAINING_SPV_DIR}/mesh_mask.json")
add_custom_command(
    OUTPUT "${_mesh_mask_spv}" "${_mesh_mask_manifest}"
    BYPRODUCTS "${_mesh_mask_raw}"
    COMMAND ${LFS_TENSOR_SLANGC} "${_mesh_mask_source}"
            -entry main -stage compute -O -fp-mode precise
            -line-directive-mode none -target spirv -profile glsl_460
            -emit-spirv-directly
            -o "${_mesh_mask_raw}"
    COMMAND lfs_tensor_spirv_finalize "${_mesh_mask_raw}" "${_mesh_mask_spv}" "${_mesh_mask_manifest}"
    DEPENDS "${_mesh_mask_source}"
            "${CMAKE_SOURCE_DIR}/src/core/tensor/backend/vulkan/shaders/distortion_model.slangh"
            lfs_tensor_spirv_finalize
    COMMENT "slangc training/mesh_mask.spv"
    VERBATIM)
set(_bilateral_source "${CMAKE_CURRENT_SOURCE_DIR}/shaders/bilateral.slang")
set(_bilateral_raw "${LFS_TRAINING_SPV_DIR}/bilateral.raw.spv")
set(_bilateral_spv "${LFS_TRAINING_SPV_DIR}/bilateral.spv")
set(_bilateral_manifest "${LFS_TRAINING_SPV_DIR}/bilateral.json")
add_custom_command(
    OUTPUT "${_bilateral_spv}" "${_bilateral_manifest}"
    BYPRODUCTS "${_bilateral_raw}"
    COMMAND ${LFS_TENSOR_SLANGC} "${_bilateral_source}"
            -entry main -stage compute -O -fp-mode precise
            -line-directive-mode none -target spirv -profile glsl_460
            -emit-spirv-directly
            -o "${_bilateral_raw}"
    COMMAND lfs_tensor_spirv_finalize "${_bilateral_raw}" "${_bilateral_spv}" "${_bilateral_manifest}"
    DEPENDS "${_bilateral_source}" "${CMAKE_CURRENT_SOURCE_DIR}/shaders/ppisp_math.slangh" lfs_tensor_spirv_finalize
    COMMENT "slangc training/bilateral.spv"
    VERBATIM)
set(_controller_source "${CMAKE_CURRENT_SOURCE_DIR}/shaders/controller.slang")
set(_controller_raw "${LFS_TRAINING_SPV_DIR}/controller.raw.spv")
set(_controller_spv "${LFS_TRAINING_SPV_DIR}/controller.spv")
set(_controller_manifest "${LFS_TRAINING_SPV_DIR}/controller.json")
add_custom_command(
    OUTPUT "${_controller_spv}" "${_controller_manifest}"
    BYPRODUCTS "${_controller_raw}"
    COMMAND ${LFS_TENSOR_SLANGC} "${_controller_source}"
            -entry main -stage compute -O -fp-mode precise
            -line-directive-mode none -target spirv -profile glsl_460
            -emit-spirv-directly
            -o "${_controller_raw}"
    COMMAND lfs_tensor_spirv_finalize "${_controller_raw}" "${_controller_spv}" "${_controller_manifest}"
    DEPENDS "${_controller_source}" lfs_tensor_spirv_finalize
    COMMENT "slangc training/controller.spv"
    VERBATIM)
set(_ppisp_source "${CMAKE_CURRENT_SOURCE_DIR}/shaders/ppisp.slang")
set(_ppisp_raw "${LFS_TRAINING_SPV_DIR}/ppisp.raw.spv")
set(_ppisp_spv "${LFS_TRAINING_SPV_DIR}/ppisp.spv")
set(_ppisp_manifest "${LFS_TRAINING_SPV_DIR}/ppisp.json")
add_custom_command(
    OUTPUT "${_ppisp_spv}" "${_ppisp_manifest}"
    BYPRODUCTS "${_ppisp_raw}"
    COMMAND ${LFS_TENSOR_SLANGC} "${_ppisp_source}"
            -entry main -stage compute -O -fp-mode precise
            -line-directive-mode none -target spirv -profile glsl_460
            -emit-spirv-directly
            -o "${_ppisp_raw}"
    COMMAND lfs_tensor_spirv_finalize "${_ppisp_raw}" "${_ppisp_spv}" "${_ppisp_manifest}"
    DEPENDS "${_ppisp_source}" "${CMAKE_CURRENT_SOURCE_DIR}/shaders/ppisp_math.slangh" lfs_tensor_spirv_finalize
    COMMENT "slangc training/ppisp.spv"
    VERBATIM)
add_custom_command(
    OUTPUT "${_pair_sort_match_spv}" "${_pair_sort_match_manifest}"
    BYPRODUCTS "${_pair_sort_match_raw}"
    COMMAND ${LFS_TENSOR_SLANGC} "${_pair_sort_source}"
            -entry main -stage compute -O -fp-mode precise
            -line-directive-mode none -target spirv -profile glsl_460
            -emit-spirv-directly -D PAIR_SORT_SUBGROUP_MATCH=1
            -I "${CMAKE_CURRENT_SOURCE_DIR}/shaders"
            -I "${CMAKE_SOURCE_DIR}/src/core/tensor/backend/vulkan/shaders"
            -o "${_pair_sort_match_raw}"
    COMMAND lfs_tensor_spirv_finalize "${_pair_sort_match_raw}"
            "${_pair_sort_match_spv}" "${_pair_sort_match_manifest}"
    DEPENDS "${_pair_sort_source}" lfs_tensor_spirv_finalize
    COMMENT "slangc training/pair_sort_match.spv"
    VERBATIM)
set(_photometric_spv "${LFS_TRAINING_SPV_DIR}/photometric.spv")
add_custom_command(
    OUTPUT "${_photometric_spv}" "${LFS_TRAINING_SPV_DIR}/photometric.json"
    COMMAND ${LFS_TENSOR_SLANGC} "${CMAKE_CURRENT_SOURCE_DIR}/shaders/photometric.slang"
            -entry main -stage compute -O -fp-mode precise -line-directive-mode none
            -target spirv -profile glsl_460 -emit-spirv-directly
            -o "${LFS_TRAINING_SPV_DIR}/photometric.raw.spv"
    COMMAND lfs_tensor_spirv_finalize "${LFS_TRAINING_SPV_DIR}/photometric.raw.spv"
            "${_photometric_spv}" "${LFS_TRAINING_SPV_DIR}/photometric.json"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/photometric.slang" lfs_tensor_spirv_finalize
    VERBATIM)

# Separate filter compilations bound optimizer memory; keep their tap loops rolled.
foreach(_variant IN ITEMS photometric_fused photometric_fused_gradient)
    if(_variant STREQUAL "photometric_fused")
        set(_moments 1)
    else()
        set(_moments 0)
    endif()
    add_custom_command(
        OUTPUT "${LFS_TRAINING_SPV_DIR}/${_variant}.spv" "${LFS_TRAINING_SPV_DIR}/${_variant}.json"
        COMMAND ${LFS_TENSOR_SLANGC} "${CMAKE_CURRENT_SOURCE_DIR}/shaders/photometric_fused.slang"
                -entry main -stage compute -O -fp-mode precise -line-directive-mode none
                -target spirv -profile glsl_460 -emit-spirv-directly -D PHOTO_MOMENTS=${_moments}
                -o "${LFS_TRAINING_SPV_DIR}/${_variant}.raw.spv"
        COMMAND lfs_tensor_spirv_finalize "${LFS_TRAINING_SPV_DIR}/${_variant}.raw.spv"
                "${LFS_TRAINING_SPV_DIR}/${_variant}.spv" "${LFS_TRAINING_SPV_DIR}/${_variant}.json"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/photometric_fused.slang" lfs_tensor_spirv_finalize
        VERBATIM)
endforeach()
set(_fast_forward_spv "${LFS_TRAINING_SPV_DIR}/fast_forward.spv")
add_custom_command(
    OUTPUT "${_fast_forward_spv}" "${LFS_TRAINING_SPV_DIR}/fast_forward.json"
    COMMAND ${LFS_TENSOR_SLANGC} "${CMAKE_CURRENT_SOURCE_DIR}/shaders/fast_forward.slang"
            -entry main -stage compute -O -fp-mode precise -line-directive-mode none
            -target spirv -profile glsl_460 -emit-spirv-directly
            -o "${LFS_TRAINING_SPV_DIR}/fast_forward.raw.spv"
    COMMAND lfs_tensor_spirv_finalize "${LFS_TRAINING_SPV_DIR}/fast_forward.raw.spv"
            "${_fast_forward_spv}" "${LFS_TRAINING_SPV_DIR}/fast_forward.json"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/fast_forward.slang"
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/fast_common.slangh"
            "${CMAKE_SOURCE_DIR}/src/core/tensor/backend/vulkan/shaders/sh_decode.slangh" lfs_tensor_spirv_finalize
    VERBATIM)
foreach(_variant IN ITEMS fast_backward fast_backward_atomic)
    if(_variant STREQUAL "fast_backward_atomic")
        set(_native 1)
    else()
        set(_native 0)
    endif()
    add_custom_command(
        OUTPUT "${LFS_TRAINING_SPV_DIR}/${_variant}.spv" "${LFS_TRAINING_SPV_DIR}/${_variant}.json"
        COMMAND ${LFS_TENSOR_SLANGC} "${CMAKE_CURRENT_SOURCE_DIR}/shaders/fast_backward.slang"
                -entry main -stage compute -O -fp-mode precise -line-directive-mode none
                -target spirv -profile glsl_460 -emit-spirv-directly -D FAST_NATIVE_ATOMIC=${_native}
                -o "${LFS_TRAINING_SPV_DIR}/${_variant}.raw.spv"
        COMMAND lfs_tensor_spirv_finalize "${LFS_TRAINING_SPV_DIR}/${_variant}.raw.spv"
                "${LFS_TRAINING_SPV_DIR}/${_variant}.spv" "${LFS_TRAINING_SPV_DIR}/${_variant}.json"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/fast_backward.slang"
                "${CMAKE_CURRENT_SOURCE_DIR}/shaders/fast_common.slangh"
            "${CMAKE_SOURCE_DIR}/src/core/tensor/backend/vulkan/shaders/sh_decode.slangh"
                "${CMAKE_CURRENT_SOURCE_DIR}/shaders/fast_projection_backward.slangh"
                "${CMAKE_CURRENT_SOURCE_DIR}/shaders/effective_rank.slangh" lfs_tensor_spirv_finalize
        VERBATIM)
endforeach()
set(_gsplat_forward_spv "${LFS_TRAINING_SPV_DIR}/gsplat_forward.spv")
add_custom_command(
    OUTPUT "${_gsplat_forward_spv}" "${LFS_TRAINING_SPV_DIR}/gsplat_forward.json"
    COMMAND ${LFS_TENSOR_SLANGC} "${CMAKE_CURRENT_SOURCE_DIR}/shaders/gsplat_forward.slang"
            -entry main -stage compute -O -fp-mode precise -line-directive-mode none
            -target spirv -profile glsl_460 -emit-spirv-directly
            -o "${LFS_TRAINING_SPV_DIR}/gsplat_forward.raw.spv"
    COMMAND lfs_tensor_spirv_finalize "${LFS_TRAINING_SPV_DIR}/gsplat_forward.raw.spv"
            "${_gsplat_forward_spv}" "${LFS_TRAINING_SPV_DIR}/gsplat_forward.json"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/gsplat_forward.slang"
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/gsplat_common.slangh"
            "${CMAKE_SOURCE_DIR}/src/core/tensor/backend/vulkan/shaders/sh_decode.slangh"
            lfs_tensor_spirv_finalize
    VERBATIM)
foreach(_variant IN ITEMS gsplat_backward gsplat_backward_atomic)
    if(_variant STREQUAL "gsplat_backward_atomic")
        set(_native 1)
    else()
        set(_native 0)
    endif()
    add_custom_command(
        OUTPUT "${LFS_TRAINING_SPV_DIR}/${_variant}.spv" "${LFS_TRAINING_SPV_DIR}/${_variant}.json"
        COMMAND ${LFS_TENSOR_SLANGC} "${CMAKE_CURRENT_SOURCE_DIR}/shaders/gsplat_backward.slang"
                -entry main -stage compute -O -fp-mode precise -line-directive-mode none
                -target spirv -profile glsl_460 -emit-spirv-directly -D GSPLAT_NATIVE_ATOMIC=${_native}
                -o "${LFS_TRAINING_SPV_DIR}/${_variant}.raw.spv"
        COMMAND lfs_tensor_spirv_finalize "${LFS_TRAINING_SPV_DIR}/${_variant}.raw.spv"
                "${LFS_TRAINING_SPV_DIR}/${_variant}.spv" "${LFS_TRAINING_SPV_DIR}/${_variant}.json"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/gsplat_backward.slang"
                "${CMAKE_CURRENT_SOURCE_DIR}/shaders/gsplat_common.slangh"
                "${CMAKE_SOURCE_DIR}/src/core/tensor/backend/vulkan/shaders/sh_decode.slangh" lfs_tensor_spirv_finalize
        VERBATIM)
endforeach()
set(_structure_spv "${LFS_TRAINING_SPV_DIR}/structure.spv")
add_custom_command(
    OUTPUT "${_structure_spv}" "${LFS_TRAINING_SPV_DIR}/structure.json"
    COMMAND ${LFS_TENSOR_SLANGC} "${CMAKE_CURRENT_SOURCE_DIR}/shaders/structure.slang"
            -entry main -stage compute -O -fp-mode precise -line-directive-mode none
            -target spirv -profile glsl_460 -emit-spirv-directly
            -o "${LFS_TRAINING_SPV_DIR}/structure.raw.spv"
    COMMAND lfs_tensor_spirv_finalize "${LFS_TRAINING_SPV_DIR}/structure.raw.spv"
            "${_structure_spv}" "${LFS_TRAINING_SPV_DIR}/structure.json"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/structure.slang" lfs_tensor_spirv_finalize
    VERBATIM)
set(_flip_spv "${LFS_TRAINING_SPV_DIR}/flip.spv")
add_custom_command(
    OUTPUT "${_flip_spv}" "${LFS_TRAINING_SPV_DIR}/flip.json"
    COMMAND ${LFS_TENSOR_SLANGC} "${CMAKE_CURRENT_SOURCE_DIR}/shaders/flip.slang"
            -entry main -stage compute -O -fp-mode precise -line-directive-mode none
            -target spirv -profile glsl_460 -emit-spirv-directly
            -o "${LFS_TRAINING_SPV_DIR}/flip.raw.spv"
    COMMAND lfs_tensor_spirv_finalize "${LFS_TRAINING_SPV_DIR}/flip.raw.spv"
            "${_flip_spv}" "${LFS_TRAINING_SPV_DIR}/flip.json"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/flip.slang" lfs_tensor_spirv_finalize
    VERBATIM)
set(_blob_spv "${LFS_TRAINING_SPV_DIR}/blob.spv")
add_custom_command(
    OUTPUT "${_blob_spv}" "${LFS_TRAINING_SPV_DIR}/blob.json"
    COMMAND ${LFS_TENSOR_SLANGC} "${CMAKE_CURRENT_SOURCE_DIR}/shaders/blob.slang"
            -entry main -stage compute -O -fp-mode precise -line-directive-mode none
            -target spirv -profile glsl_460 -emit-spirv-directly
            -o "${LFS_TRAINING_SPV_DIR}/blob.raw.spv"
    COMMAND lfs_tensor_spirv_finalize "${LFS_TRAINING_SPV_DIR}/blob.raw.spv"
            "${_blob_spv}" "${LFS_TRAINING_SPV_DIR}/blob.json"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/blob.slang" lfs_tensor_spirv_finalize
    VERBATIM)
set(_training_shader_table "${LFS_TRAINING_SPV_DIR}/training_shader_table.cpp")
set(_training_image_source "${CMAKE_CURRENT_SOURCE_DIR}/shaders/training_image.slang")
set(_training_image_raw "${LFS_TRAINING_SPV_DIR}/training_image.raw.spv")
set(_training_image_spv "${LFS_TRAINING_SPV_DIR}/training_image.spv")
set(_training_image_manifest "${LFS_TRAINING_SPV_DIR}/training_image.json")
add_custom_command(
    OUTPUT "${_training_image_spv}" "${_training_image_manifest}"
    BYPRODUCTS "${_training_image_raw}"
    COMMAND ${LFS_TENSOR_SLANGC} "${_training_image_source}"
            -entry main -stage compute -O -fp-mode precise
            -line-directive-mode none -target spirv -profile glsl_460
            -emit-spirv-directly
            -I "${CMAKE_CURRENT_SOURCE_DIR}/shaders"
            -o "${_training_image_raw}"
    COMMAND lfs_tensor_spirv_finalize "${_training_image_raw}" "${_training_image_spv}" "${_training_image_manifest}"
    DEPENDS "${_training_image_source}" lfs_tensor_spirv_finalize
    COMMENT "slangc training/training_image.spv"
    VERBATIM)
set(_refine_source "${CMAKE_CURRENT_SOURCE_DIR}/shaders/refine.slang")
set(_refine_raw "${LFS_TRAINING_SPV_DIR}/refine.raw.spv")
set(_refine_spv "${LFS_TRAINING_SPV_DIR}/refine.spv")
set(_refine_manifest "${LFS_TRAINING_SPV_DIR}/refine.json")
add_custom_command(
    OUTPUT "${_refine_spv}" "${_refine_manifest}"
    BYPRODUCTS "${_refine_raw}"
    COMMAND ${LFS_TENSOR_SLANGC} "${_refine_source}"
            -entry main -stage compute -O -fp-mode precise
            -line-directive-mode none -target spirv -profile glsl_460
            -emit-spirv-directly
            -I "${CMAKE_CURRENT_SOURCE_DIR}/shaders"
            -o "${_refine_raw}"
    COMMAND lfs_tensor_spirv_finalize "${_refine_raw}" "${_refine_spv}" "${_refine_manifest}"
    DEPENDS "${_refine_source}" lfs_tensor_spirv_finalize
    COMMENT "slangc training/refine.spv"
    VERBATIM)
set(_mrnf_source "${CMAKE_CURRENT_SOURCE_DIR}/shaders/mrnf.slang")
set(_mrnf_raw "${LFS_TRAINING_SPV_DIR}/mrnf.raw.spv")
set(_mrnf_spv "${LFS_TRAINING_SPV_DIR}/mrnf.spv")
set(_mrnf_manifest "${LFS_TRAINING_SPV_DIR}/mrnf.json")
add_custom_command(
    OUTPUT "${_mrnf_spv}" "${_mrnf_manifest}"
    BYPRODUCTS "${_mrnf_raw}"
    COMMAND ${LFS_TENSOR_SLANGC} "${_mrnf_source}"
            -entry main -stage compute -O -fp-mode precise
            -line-directive-mode none -target spirv -profile glsl_460
            -emit-spirv-directly
            -I "${CMAKE_CURRENT_SOURCE_DIR}/shaders"
            -o "${_mrnf_raw}"
    COMMAND lfs_tensor_spirv_finalize "${_mrnf_raw}" "${_mrnf_spv}" "${_mrnf_manifest}"
    DEPENDS "${_mrnf_source}" lfs_tensor_spirv_finalize
    COMMENT "slangc training/mrnf.spv"
    VERBATIM)
add_custom_command(
    OUTPUT "${_training_shader_table}"
    COMMAND lfs_tensor_spirv_finalize embed-named "${_training_shader_table}"
            lfs::training::vulkan embedded_training_shaders
            structure "${_structure_spv}"
            flip "${_flip_spv}"
            blob "${_blob_spv}"
            gsplat_forward "${_gsplat_forward_spv}"
            gsplat_backward "${LFS_TRAINING_SPV_DIR}/gsplat_backward.spv"
            gsplat_backward_atomic "${LFS_TRAINING_SPV_DIR}/gsplat_backward_atomic.spv"
            pair_sort "${_pair_sort_spv}" pair_sort_match "${_pair_sort_match_spv}"
            fast_backward "${LFS_TRAINING_SPV_DIR}/fast_backward.spv"
            fast_backward_atomic "${LFS_TRAINING_SPV_DIR}/fast_backward_atomic.spv"
            fast_forward "${_fast_forward_spv}" photometric "${_photometric_spv}"
            photometric_fused "${LFS_TRAINING_SPV_DIR}/photometric_fused.spv"
            photometric_fused_gradient "${LFS_TRAINING_SPV_DIR}/photometric_fused_gradient.spv"
            sh_ops "${_sh_ops_spv}" extra_loss "${_extra_loss_spv}" adam "${_adam_spv}"
            training_image "${_training_image_spv}"
            refine "${_refine_spv}"
            mrnf "${_mrnf_spv}"
            morton "${_morton_spv}" joint_morton "${_joint_morton_spv}"
            mcmc "${_mcmc_spv}"
            lpips "${_lpips_spv}" lpips_coop "${_lpips_coop_spv}" lpips_halo "${_lpips_halo_spv}"
            masks "${_masks_spv}"
            mesh_mask "${_mesh_mask_spv}"
            bilateral "${_bilateral_spv}"
            controller "${_controller_spv}"
            ppisp "${_ppisp_spv}"
            geometry "${_geometry_spv}" geometry_fp32 "${_geometry_fp32_spv}"
    DEPENDS "${_blob_spv}" "${_structure_spv}" "${_flip_spv}" "${LFS_TRAINING_SPV_DIR}/fast_backward.spv" "${LFS_TRAINING_SPV_DIR}/fast_backward_atomic.spv" "${_gsplat_forward_spv}" "${LFS_TRAINING_SPV_DIR}/gsplat_backward.spv" "${LFS_TRAINING_SPV_DIR}/gsplat_backward_atomic.spv"
            "${LFS_TRAINING_SPV_DIR}/photometric_fused.spv" "${LFS_TRAINING_SPV_DIR}/photometric_fused_gradient.spv"
            "${_fast_forward_spv}" "${_photometric_spv}" "${_pair_sort_spv}" "${_pair_sort_match_spv}" lfs_tensor_spirv_finalize
            "${_sh_ops_spv}" "${_extra_loss_spv}" "${_adam_spv}" "${_training_image_spv}" "${_refine_spv}" "${_mrnf_spv}" "${_morton_spv}" "${_joint_morton_spv}" "${_mcmc_spv}" "${_lpips_spv}" "${_lpips_coop_spv}" "${_lpips_halo_spv}" "${_masks_spv}" "${_mesh_mask_spv}" "${_bilateral_spv}" "${_controller_spv}" "${_ppisp_spv}" "${_geometry_spv}" "${_geometry_fp32_spv}"
            "${CMAKE_CURRENT_SOURCE_DIR}/training_shader_table.hpp"
    COMMENT "embed training shaders"
    VERBATIM)
add_custom_target(lfs_training_vulkan_shaders DEPENDS "${_pair_sort_spv}" "${_pair_sort_match_spv}" "${_geometry_spv}" "${_geometry_fp32_spv}" "${_sh_ops_spv}" "${_extra_loss_spv}" "${_adam_spv}" "${_morton_spv}" "${_joint_morton_spv}" "${_mcmc_spv}" "${_lpips_spv}" "${_lpips_coop_spv}" "${_lpips_halo_spv}" "${_masks_spv}" "${_mesh_mask_spv}" "${_bilateral_spv}" "${_controller_spv}" "${_ppisp_spv}" "${_photometric_spv}" "${_fast_forward_spv}" "${_gsplat_forward_spv}" "${_training_image_spv}" "${_refine_spv}" "${_mrnf_spv}" "${_training_shader_table}")
add_dependencies(lfs_training_vulkan lfs_training_vulkan_shaders)
target_sources(lfs_training_vulkan PRIVATE "${_training_shader_table}")
target_include_directories(lfs_training_vulkan
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include
    PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/src/training
            ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(lfs_training_vulkan PRIVATE LFS_TENSOR_PRIVATE_ACCESS)
target_link_libraries(lfs_training_vulkan PUBLIC lfs_core lfs_diagnostics)
set_target_properties(lfs_training_vulkan PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON)
