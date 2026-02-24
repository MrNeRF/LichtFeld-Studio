/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace lfs::core {

    class Tensor;
    enum class Device : uint8_t;
    enum class DataType : uint8_t;

    namespace internal {

        enum class LazyOpKind : uint8_t {
            Leaf = 0,
            Unary = 1,
            Binary = 2,
            ScalarUnary = 3,
            Permutation = 4,
            Deferred = 5,
            Reduce = 6
        };

        struct LazyExprDebugInfo {
            uint64_t node_id = 0;
            LazyOpKind op_kind = LazyOpKind::Leaf;
            std::string op_name;
            std::vector<uint64_t> input_ids;
            Device device = static_cast<Device>(0);
            DataType dtype = static_cast<DataType>(0);
            std::string shape;
            size_t buffer_bytes = 0;
        };

        LFS_CORE_API bool lazy_ir_active();
        LFS_CORE_API void clear_lazy_ir_for_testing();

        LFS_CORE_API bool tensor_has_lazy_expr(const Tensor& tensor);
        LFS_CORE_API uint64_t tensor_lazy_expr_id(const Tensor& tensor);
        LFS_CORE_API std::optional<LazyExprDebugInfo> tensor_lazy_expr_info(const Tensor& tensor);
        LFS_CORE_API std::optional<LazyExprDebugInfo> lazy_ir_node_info(uint64_t node_id);
        LFS_CORE_API std::vector<LazyExprDebugInfo> lazy_ir_collect_topological_subgraph(uint64_t root_node_id);
        // Update an existing node's dependency edges. Used to wire deferred pointwise chains.
        LFS_CORE_API bool lazy_ir_set_node_inputs(uint64_t node_id, const std::vector<uint64_t>& input_ids);

        LFS_CORE_API void lazy_ir_record_unary(const Tensor& input,
                                               const Tensor& output,
                                               std::string_view op_name);
        LFS_CORE_API void lazy_ir_record_binary(const Tensor& left,
                                                const Tensor& right,
                                                const Tensor& output,
                                                std::string_view op_name);
        LFS_CORE_API void lazy_ir_record_scalar_unary(const Tensor& input,
                                                      const Tensor& output,
                                                      std::string_view op_name);
        LFS_CORE_API void lazy_ir_record_permutation(const Tensor& input,
                                                     const Tensor& indices,
                                                     const Tensor& output,
                                                     std::string_view op_name);
        LFS_CORE_API void lazy_ir_record_reduce(const Tensor& input,
                                                const Tensor& output,
                                                std::string_view op_name);
        LFS_CORE_API uint64_t lazy_ir_record_deferred(const Tensor& output,
                                                      std::string_view op_name = "deferred_expr",
                                                      const std::vector<uint64_t>& input_ids = {});

    } // namespace internal

} // namespace lfs::core
