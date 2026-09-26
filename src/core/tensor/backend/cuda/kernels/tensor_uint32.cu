/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "tensor_ops.hpp"

namespace lfs::core::tensor_ops {
    template LFS_CORE_API void launch_binary_op_generic<uint32_t, uint32_t, ops::add_op>(
        const uint32_t*, const uint32_t*, uint32_t*, size_t, ops::add_op, cudaStream_t);
    template LFS_CORE_API void launch_binary_op_generic<uint32_t, uint32_t, ops::sub_op>(
        const uint32_t*, const uint32_t*, uint32_t*, size_t, ops::sub_op, cudaStream_t);
    template LFS_CORE_API void launch_binary_op_generic<uint32_t, uint32_t, ops::mul_op>(
        const uint32_t*, const uint32_t*, uint32_t*, size_t, ops::mul_op, cudaStream_t);
    template LFS_CORE_API void launch_binary_op_generic<uint32_t, uint32_t, ops::div_op>(
        const uint32_t*, const uint32_t*, uint32_t*, size_t, ops::div_op, cudaStream_t);
    template LFS_CORE_API void launch_binary_op_generic<uint32_t, uint32_t, ops::pow_op>(
        const uint32_t*, const uint32_t*, uint32_t*, size_t, ops::pow_op, cudaStream_t);
    template LFS_CORE_API void launch_binary_op_generic<uint32_t, uint32_t, ops::mod_op>(
        const uint32_t*, const uint32_t*, uint32_t*, size_t, ops::mod_op, cudaStream_t);
    template LFS_CORE_API void launch_binary_op_generic<uint32_t, uint32_t, ops::maximum_op>(
        const uint32_t*, const uint32_t*, uint32_t*, size_t, ops::maximum_op, cudaStream_t);
    template LFS_CORE_API void launch_binary_op_generic<uint32_t, uint32_t, ops::minimum_op>(
        const uint32_t*, const uint32_t*, uint32_t*, size_t, ops::minimum_op, cudaStream_t);
    template LFS_CORE_API void launch_binary_op_generic<uint32_t, unsigned char, ops::equal_op>(
        const uint32_t*, const uint32_t*, unsigned char*, size_t, ops::equal_op, cudaStream_t);
    template LFS_CORE_API void launch_binary_op_generic<uint32_t, unsigned char, ops::not_equal_op>(
        const uint32_t*, const uint32_t*, unsigned char*, size_t, ops::not_equal_op, cudaStream_t);
    template LFS_CORE_API void launch_binary_op_generic<uint32_t, unsigned char, ops::less_op>(
        const uint32_t*, const uint32_t*, unsigned char*, size_t, ops::less_op, cudaStream_t);
    template LFS_CORE_API void launch_binary_op_generic<uint32_t, unsigned char, ops::less_equal_op>(
        const uint32_t*, const uint32_t*, unsigned char*, size_t, ops::less_equal_op, cudaStream_t);
    template LFS_CORE_API void launch_binary_op_generic<uint32_t, unsigned char, ops::greater_op>(
        const uint32_t*, const uint32_t*, unsigned char*, size_t, ops::greater_op, cudaStream_t);
    template LFS_CORE_API void launch_binary_op_generic<uint32_t, unsigned char, ops::greater_equal_op>(
        const uint32_t*, const uint32_t*, unsigned char*, size_t, ops::greater_equal_op, cudaStream_t);
    template LFS_CORE_API void launch_binary_op_generic<uint32_t, unsigned char, ops::logical_and_op>(
        const uint32_t*, const uint32_t*, unsigned char*, size_t, ops::logical_and_op, cudaStream_t);
    template LFS_CORE_API void launch_binary_op_generic<uint32_t, unsigned char, ops::logical_or_op>(
        const uint32_t*, const uint32_t*, unsigned char*, size_t, ops::logical_or_op, cudaStream_t);
    template LFS_CORE_API void launch_binary_op_generic<uint32_t, unsigned char, ops::logical_xor_op>(
        const uint32_t*, const uint32_t*, unsigned char*, size_t, ops::logical_xor_op, cudaStream_t);
} // namespace lfs::core::tensor_ops
