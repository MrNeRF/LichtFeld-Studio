/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "../gpu_backend_ops.hpp"

#include "../../internal/tensor_functors.hpp"
#include "../../internal/tensor_ops.hpp"
#include "core/assert.hpp"
#include "core/cuda_error.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace lfs::core::internal {
    namespace {
        template <class T>
        T* cuda_pointer(const StorageRef storage) {
            LFS_ASSERT_MSG(storage.backend == GpuBackend::CUDA,
                           "CUDA index/mask adapter received non-CUDA storage");
            return reinterpret_cast<T*>(
                static_cast<unsigned char*>(storage.data) + storage.byte_offset);
        }

        template <class T>
        const T* cuda_const_pointer(const StorageRef storage) {
            return cuda_pointer<const T>(storage);
        }

        template <class T>
        T scalar_value(const ScalarOperand scalar) {
            switch (scalar.kind) {
            case ScalarKind::Float: return static_cast<T>(scalar.value.float_value);
            case ScalarKind::Int32: return static_cast<T>(scalar.value.int32_value);
            case ScalarKind::Int64: return static_cast<T>(scalar.value.int64_value);
            case ScalarKind::Bool: return static_cast<T>(scalar.value.bool_value);
            }
            LFS_ASSERT_MSG(false, "invalid scalar operand kind");
            return T{};
        }

        template <class Operation>
        void dispatch_index_dtype(const StorageRef storage, Operation&& operation) {
            switch (storage.dtype) {
            case DataType::Float32: operation.template operator()<float>(); break;
            case DataType::Float16: operation.template operator()<__half>(); break;
            case DataType::Int32: operation.template operator()<int32_t>(); break;
            case DataType::Int64: operation.template operator()<int64_t>(); break;
            case DataType::UInt8:
            case DataType::Bool: operation.template operator()<uint8_t>(); break;
            default: LFS_ASSERT_MSG(false, "unsupported CUDA index/mask dtype");
            }
        }
    } // namespace

    void CudaBackendOps::gather(
        const StorageRef input, const StorageRef indices, const StorageRef output,
        const StridedLayout& input_layout, const StridedLayout& index_layout,
        const IndexProgram& program, const ExecContext context) {
        if (input.dtype == DataType::Float32) {
            tensor_ops::launch_gather(
                cuda_const_pointer<float>(input), cuda_const_pointer<int>(indices),
                cuda_pointer<float>(output), input_layout.dims.data(),
                index_layout.dims.data(), input_layout.rank, program.dim,
                program.total_elements, program.boundary_mode, context.cuda_stream);
        } else if (input.dtype == DataType::Int64) {
            tensor_ops::launch_gather(
                cuda_const_pointer<int64_t>(input), cuda_const_pointer<int>(indices),
                cuda_pointer<int64_t>(output), input_layout.dims.data(),
                index_layout.dims.data(), input_layout.rank, program.dim,
                program.total_elements, program.boundary_mode, context.cuda_stream);
        } else {
            LFS_ASSERT_MSG(false, "CUDA gather supports only Float32 and Int64");
        }
        LFS_CUDA_CHECK(cudaGetLastError());
    }

    void CudaBackendOps::gather_fused_unary(
        const StorageRef input, const StorageRef indices, const StorageRef output,
        const PointwiseOp unary, const IndexProgram& program,
        const ExecContext context) {
        switch (unary) {
        case PointwiseOp::Abs:
            tensor_ops::launch_gather_fused_unary(
                cuda_const_pointer<float>(input), cuda_const_pointer<int>(indices),
                cuda_pointer<float>(output), program.input_size, program.index_size,
                ops::abs_op{}, context.cuda_stream);
            return;
        case PointwiseOp::Sqrt:
            tensor_ops::launch_gather_fused_unary(
                cuda_const_pointer<float>(input), cuda_const_pointer<int>(indices),
                cuda_pointer<float>(output), program.input_size, program.index_size,
                ops::sqrt_op{}, context.cuda_stream);
            return;
        case PointwiseOp::Neg:
            tensor_ops::launch_gather_fused_unary(
                cuda_const_pointer<float>(input), cuda_const_pointer<int>(indices),
                cuda_pointer<float>(output), program.input_size, program.index_size,
                ops::neg_op{}, context.cuda_stream);
            return;
        default:
            LFS_ASSERT_MSG(false, "unsupported fused gather unary operation");
        }
    }

    void CudaBackendOps::take(
        const StorageRef input, const StorageRef indices, const StorageRef output,
        const IndexProgram& program, const ExecContext context) {
        tensor_ops::launch_take(
            cuda_const_pointer<float>(input), cuda_const_pointer<int>(indices),
            cuda_pointer<float>(output), program.input_size, program.index_size,
            context.cuda_stream);
    }

    void CudaBackendOps::index_select(
        const StorageRef input, const StorageRef indices, const StorageRef output,
        const StridedLayout& input_layout, const IndexProgram& program,
        const ExecContext context) {
        const auto launch = [&]<typename T>() {
            tensor_ops::launch_index_select(
                cuda_const_pointer<T>(input), cuda_const_pointer<int>(indices),
                cuda_pointer<T>(output), input_layout.dims.data(), input_layout.rank,
                program.dim, program.index_size, program.boundary_mode,
                context.cuda_stream);
        };
        switch (input.dtype) {
        case DataType::Float32: launch.template operator()<float>(); break;
        case DataType::Int64: launch.template operator()<int64_t>(); break;
        case DataType::Int32: launch.template operator()<int32_t>(); break;
        case DataType::UInt8:
        case DataType::Bool: launch.template operator()<uint8_t>(); break;
        default: LFS_ASSERT_MSG(false, "unsupported CUDA index_select dtype");
        }
        LFS_CUDA_CHECK(cudaGetLastError());
    }

    void CudaBackendOps::scatter(
        const StorageRef output, const StorageRef indices, const StorageRef source,
        const StridedLayout& output_layout, const StridedLayout& source_layout,
        const IndexProgram& program, const ExecContext context) {
        const auto launch = [&]<typename T>() {
            tensor_ops::launch_scatter(
                cuda_pointer<T>(output), cuda_const_pointer<int>(indices),
                cuda_const_pointer<T>(source), output_layout.dims.data(),
                source_layout.dims.data(), output_layout.rank, program.dim,
                program.total_elements, program.scatter_mode, context.cuda_stream);
        };
        switch (output.dtype) {
        case DataType::Float32: launch.template operator()<float>(); break;
        case DataType::Int32: launch.template operator()<int32_t>(); break;
        case DataType::UInt8:
        case DataType::Bool: launch.template operator()<uint8_t>(); break;
        default: LFS_ASSERT_MSG(false, "unsupported CUDA scatter dtype");
        }
    }

    void CudaBackendOps::index_copy(
        const StorageRef output, const StorageRef indices, const StorageRef source,
        const StridedLayout& output_layout, const IndexProgram& program,
        const ExecContext context) {
        const auto launch = [&]<typename T>() {
            tensor_ops::launch_index_copy(
                cuda_pointer<T>(output), cuda_const_pointer<int>(indices),
                cuda_const_pointer<T>(source), output_layout.dims.data(),
                output_layout.rank, program.dim, program.index_size,
                context.cuda_stream);
        };
        switch (output.dtype) {
        case DataType::Float32: launch.template operator()<float>(); break;
        case DataType::Int32: launch.template operator()<int32_t>(); break;
        case DataType::UInt8:
        case DataType::Bool: launch.template operator()<uint8_t>(); break;
        default: LFS_ASSERT_MSG(false, "unsupported CUDA index_copy dtype");
        }
    }

    void CudaBackendOps::index_add(
        const StorageRef output, const StorageRef indices, const StorageRef source,
        const StridedLayout& output_layout, const IndexProgram& program,
        const ExecContext context) {
        if (output.dtype == DataType::Float32) {
            tensor_ops::launch_index_add<float>(
                cuda_pointer<float>(output), cuda_const_pointer<int>(indices),
                cuda_const_pointer<float>(source), output_layout.dims.data(),
                output_layout.rank, program.dim, program.index_size,
                context.cuda_stream);
        } else if (output.dtype == DataType::Int32) {
            tensor_ops::launch_index_add<int>(
                cuda_pointer<int>(output), cuda_const_pointer<int>(indices),
                cuda_const_pointer<int>(source), output_layout.dims.data(),
                output_layout.rank, program.dim, program.index_size,
                context.cuda_stream);
        } else {
            LFS_ASSERT_MSG(false, "CUDA index_add supports only Float32 and Int32");
        }
    }

    void CudaBackendOps::index_fill(
        const StorageRef output, const StorageRef indices,
        const StridedLayout& output_layout, const IndexProgram& program,
        const ScalarOperand value, const ExecContext context) {
        if (output.dtype == DataType::Float32) {
            tensor_ops::launch_index_fill<float>(
                cuda_pointer<float>(output), cuda_const_pointer<int>(indices),
                scalar_value<float>(value), output_layout.dims.data(),
                output_layout.rank, program.dim, program.index_size,
                context.cuda_stream);
        } else if (output.dtype == DataType::Int32) {
            tensor_ops::launch_index_fill<int>(
                cuda_pointer<int>(output), cuda_const_pointer<int>(indices),
                scalar_value<int>(value), output_layout.dims.data(),
                output_layout.rank, program.dim, program.index_size,
                context.cuda_stream);
        } else if (output.dtype == DataType::Bool || output.dtype == DataType::UInt8) {
            tensor_ops::launch_index_fill<uint8_t>(
                cuda_pointer<uint8_t>(output), cuda_const_pointer<int>(indices),
                scalar_value<uint8_t>(value), output_layout.dims.data(),
                output_layout.rank, program.dim, program.index_size,
                context.cuda_stream);
        } else {
            LFS_ASSERT_MSG(false, "unsupported CUDA index_fill dtype");
        }
    }

    void CudaBackendOps::index_put(
        const StorageRef output, const StorageRef indices, const StorageRef values,
        const IndexProgram& program, const ExecContext context) {
        LFS_ASSERT_MSG(output.dtype == DataType::Float32,
                       "CUDA index_put launcher supports only Float32");
        tensor_ops::launch_index_put(
            cuda_pointer<float>(output), cuda_const_pointer<int>(indices),
            cuda_const_pointer<float>(values), program.input_size,
            program.index_size, context.cuda_stream);
    }

    void CudaBackendOps::masked_fill(
        const StorageRef output, const StorageRef mask, const MaskProgram& program,
        const ExecContext context) {
        switch (output.dtype) {
        case DataType::Float32:
            tensor_ops::launch_masked_fill(
                cuda_pointer<float>(output), cuda_const_pointer<unsigned char>(mask),
                scalar_value<float>(program.value), program.count, context.cuda_stream);
            break;
        case DataType::Float16:
            tensor_ops::launch_masked_fill(
                cuda_pointer<__half>(output), cuda_const_pointer<unsigned char>(mask),
                __float2half(scalar_value<float>(program.value)), program.count,
                context.cuda_stream);
            break;
        case DataType::Int32:
            tensor_ops::launch_masked_fill(
                cuda_pointer<int32_t>(output), cuda_const_pointer<unsigned char>(mask),
                scalar_value<int32_t>(program.value), program.count, context.cuda_stream);
            break;
        case DataType::Int64:
            tensor_ops::launch_masked_fill(
                cuda_pointer<int64_t>(output), cuda_const_pointer<unsigned char>(mask),
                scalar_value<int64_t>(program.value), program.count, context.cuda_stream);
            break;
        case DataType::UInt8:
        case DataType::Bool:
            tensor_ops::launch_masked_fill(
                cuda_pointer<uint8_t>(output), cuda_const_pointer<unsigned char>(mask),
                scalar_value<uint8_t>(program.value), program.count, context.cuda_stream);
            break;
        default: LFS_ASSERT_MSG(false, "unsupported CUDA masked_fill dtype");
        }
    }

    size_t CudaBackendOps::masked_select(
        const StorageRef input, const StorageRef mask, const StorageRef output,
        const MaskProgram& program, const ExecContext context) {
        const auto launch = [&]<typename T>() {
            tensor_ops::launch_masked_select(
                cuda_const_pointer<T>(input), cuda_const_pointer<unsigned char>(mask),
                cuda_pointer<T>(output), program.count, program.selected_count,
                context.cuda_stream);
        };
        dispatch_index_dtype(input, launch);
        LFS_CUDA_CHECK(cudaGetLastError());
        return program.selected_count;
    }

    void CudaBackendOps::masked_scatter(
        const StorageRef output, const StorageRef mask, const StorageRef source,
        const MaskProgram& program, const ExecContext context) {
        const auto launch = [&]<typename T>() {
            tensor_ops::launch_masked_scatter(
                cuda_pointer<T>(output), cuda_const_pointer<unsigned char>(mask),
                cuda_const_pointer<T>(source), program.count,
                program.selected_count, context.cuda_stream);
        };
        dispatch_index_dtype(output, launch);
        LFS_CUDA_CHECK(cudaGetLastError());
    }

    void CudaBackendOps::and_live(
        const StorageRef mask, const StorageRef live_mask,
        const MaskProgram& program, const ExecContext context) {
        tensor_ops::launch_and_live(
            cuda_pointer<uint8_t>(mask), cuda_const_pointer<unsigned char>(live_mask),
            program.count, context.cuda_stream);
    }

    void CudaBackendOps::where(
        const StorageRef condition, const StorageRef x, const StorageRef y,
        const StorageRef output, const StridedLayout& condition_layout,
        const StridedLayout& x_layout, const StridedLayout& y_layout,
        const StridedLayout& output_layout, const ExecContext context) {
        tensor_ops::launch_where(
            cuda_const_pointer<unsigned char>(condition), cuda_const_pointer<float>(x),
            cuda_const_pointer<float>(y), cuda_pointer<float>(output),
            condition_layout.dims.data(), x_layout.dims.data(), y_layout.dims.data(),
            output_layout.dims.data(), condition_layout.rank, x_layout.rank,
            y_layout.rank, output_layout.rank, output_layout.element_count,
            context.cuda_stream);
    }

    size_t CudaBackendOps::nonzero(
        const StorageRef input, const StorageRef output, const MaskProgram& program,
        const ExecContext context) {
        return tensor_ops::launch_nonzero(
            cuda_const_pointer<float>(input), cuda_pointer<int64_t>(output),
            program.count, program.selected_count, context.cuda_stream);
    }

    size_t CudaBackendOps::nonzero_bool(
        const StorageRef input, const StorageRef output, const MaskProgram& program,
        const ExecContext context) {
        return tensor_ops::launch_nonzero_bool(
            cuda_const_pointer<unsigned char>(input), cuda_pointer<int64_t>(output),
            program.count, program.selected_count, context.cuda_stream);
    }

} // namespace lfs::core::internal
