/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/gpu_backend_fwd.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <type_traits>

namespace lfs::core {
    class Tensor;
    enum class ReduceOp : uint8_t;
    struct StorageMeta;

    namespace ops {
        struct add_op;
        struct sub_op;
        struct mul_op;
        struct div_op;
        struct pow_op;
        struct mod_op;
        struct abs_op;
        struct neg_op;
        struct exp_op;
        struct log_op;
        struct sqrt_op;
        struct sigmoid_op;
        struct relu_op;
        struct square_op;
        struct tanh_op;
        struct rsqrt_op;
        struct sign_op;
        struct reciprocal_op;
        struct floor_op;
        struct ceil_op;
        struct round_op;
        struct maximum_op;
        struct minimum_op;
        struct equal_op;
        struct not_equal_op;
        struct less_op;
        struct less_equal_op;
        struct greater_op;
        struct greater_equal_op;
        struct logical_and_op;
        struct logical_or_op;
        struct logical_xor_op;
        struct exp2_op;
        struct log2_op;
        struct log10_op;
        struct log1p_op;
        struct sin_op;
        struct cos_op;
        struct tan_op;
        struct asin_op;
        struct acos_op;
        struct atan_op;
        struct sinh_op;
        struct cosh_op;
        struct gelu_op;
        struct swish_op;
        struct trunc_op;
        struct isnan_op;
        struct isinf_op;
        struct isfinite_op;
        struct logical_not_op;
    } // namespace ops

    namespace internal {

        enum class PointwiseOp : uint16_t {
#define LFS_POINTWISE_OP(Id, FunctorType, Name) Id,
#include "pointwise_ops.def"
#undef LFS_POINTWISE_OP
            Count
        };

        constexpr const char* pointwise_op_name(const PointwiseOp op) {
            switch (op) {
#define LFS_POINTWISE_OP(Id, FunctorType, Name) \
    case PointwiseOp::Id: return Name;
#include "pointwise_ops.def"
#undef LFS_POINTWISE_OP
            case PointwiseOp::Count: break;
            }
            return "invalid";
        }

        enum class ScalarKind : uint8_t {
            Float,
            Int32,
            Int64,
            Bool,
        };

        struct ScalarOperand {
            union Value {
                float float_value;
                int32_t int32_value;
                int64_t int64_value;
                bool bool_value;

                constexpr Value() : int64_value(0) {}
            } value;
            ScalarKind kind = ScalarKind::Float;
            bool scalar_on_right = true;
        };

        struct StorageRef {
            GpuBackend backend;
            void* data;
            size_t byte_offset;
            DataType dtype;
            const StorageMeta* meta;
            uint32_t flags = 0;
        };

        enum class AllocationClass : uint8_t {
            Pooled,
            Direct,
        };

        struct ExecContext {
            cudaStream_t cuda_stream = nullptr;
            AllocationClass allocation_class = AllocationClass::Pooled;
            const char* allocation_label = "tensor.storage";
            const char* allocation_operation = "tensor.allocate";
        };

        inline constexpr uint32_t STORAGE_REF_DIRECT_ALLOCATION = 1U << 0;
        inline constexpr uint32_t STORAGE_REF_HOST_MEMORY = 1U << 1;

        struct StridedLayout {
            size_t rank = 0;
            std::array<size_t, MAX_TENSOR_RANK> dims{};
            std::array<size_t, MAX_TENSOR_RANK> strides{};
            size_t element_count = 0;
        };

        struct PointwiseProgram {
            PointwiseOp op;
            DataType in_dtype;
            DataType out_dtype;
            ScalarOperand scalar;
        };

        struct ReduceProgram {
            ReduceOp op;
            std::array<int, MAX_TENSOR_RANK> axes{};
            size_t axis_count = 0;
            bool keepdim = false;
            DataType result_dtype;
        };

        struct SortProgram {
            size_t outer_size = 1;
            size_t dim_size = 0;
            size_t inner_size = 1;
            int dim = 0;
            bool descending = false;
        };

        struct GemmProgram {
            size_t batch = 1;
            size_t m = 0;
            size_t n = 0;
            size_t k = 0;
        };

        struct PoolProgram {
            int batch = 0;
            int channels = 0;
            int input_height = 0;
            int input_width = 0;
            int output_height = 0;
            int output_width = 0;
            int kernel_size = 0;
            int stride = 0;
            int padding = 0;
        };

        struct RandomProgram {
            size_t count = 0;
            size_t sample_count = 0;
            float first = 0.0f;
            float second = 1.0f;
            int low = 0;
            int high = 0;
            uint64_t seed = 0;
            bool replacement = false;
        };

        struct IndexProgram {
            int dim = 0;
            int boundary_mode = 0;
            int scatter_mode = 0;
            size_t input_size = 0;
            size_t index_size = 0;
            size_t total_elements = 0;
        };

        struct MaskProgram {
            size_t count = 0;
            size_t selected_count = 0;
            ScalarOperand value{};
        };

        struct CopyRequest {
            StorageRef src;
            StorageRef dst;
            size_t bytes = 0;
            bool synchronous = false;
            ExecContext context{};
            const char* operation = "tensor.copy";
        };

        struct FillRequest {
            StorageRef dst;
            size_t bytes = 0;
            uint8_t value = 0;
            bool synchronous = false;
            ExecContext context{};
            const char* operation = "tensor.fill";
        };

        struct SyncToken {
            GpuBackend backend = GpuBackend::CUDA;
            uint64_t value = 0;
            uintptr_t native = 0;
        };

        enum class PointerClass : uint8_t {
            Device,
            Host,
            Pinned,
            Unknown,
        };

        static_assert(std::is_trivially_copyable_v<ScalarOperand>);
        static_assert(std::is_trivially_copyable_v<StorageRef>);
        static_assert(std::is_trivially_copyable_v<ExecContext>);
        static_assert(std::is_trivially_copyable_v<StridedLayout>);
        static_assert(std::is_trivially_copyable_v<PointwiseProgram>);
        static_assert(std::is_trivially_copyable_v<ReduceProgram>);
        static_assert(std::is_trivially_copyable_v<SortProgram>);
        static_assert(std::is_trivially_copyable_v<GemmProgram>);
        static_assert(std::is_trivially_copyable_v<PoolProgram>);
        static_assert(std::is_trivially_copyable_v<RandomProgram>);
        static_assert(std::is_trivially_copyable_v<IndexProgram>);
        static_assert(std::is_trivially_copyable_v<MaskProgram>);
        static_assert(std::is_trivially_copyable_v<CopyRequest>);
        static_assert(std::is_trivially_copyable_v<FillRequest>);
        static_assert(std::is_trivially_copyable_v<SyncToken>);

        inline StorageRef raw_storage_ref(void* const pointer,
                                          const DataType dtype = DataType::UInt8) {
            return StorageRef{
                .backend = GpuBackend::CUDA,
                .data = pointer,
                .byte_offset = 0,
                .dtype = dtype,
                .meta = nullptr,
                .flags = STORAGE_REF_HOST_MEMORY,
            };
        }

        inline StorageRef raw_device_storage_ref(void* const pointer,
                                                 const GpuBackend backend,
                                                 const DataType dtype = DataType::UInt8) {
            return StorageRef{
                .backend = backend,
                .data = pointer,
                .byte_offset = 0,
                .dtype = dtype,
                .meta = nullptr,
                .flags = 0,
            };
        }

        inline StorageRef offset_storage_ref(StorageRef storage,
                                             const size_t byte_offset) {
            storage.byte_offset += byte_offset;
            return storage;
        }

        inline StorageRef storage_ref(const Tensor& tensor);
        inline StridedLayout strided_layout(const Tensor& tensor);

    } // namespace internal
} // namespace lfs::core
