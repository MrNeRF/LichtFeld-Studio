/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "../facade_trace.hpp"
#include "../gpu_backend_ops.hpp"

#include "../../internal/tensor_impl.hpp"
#include "core/assert.hpp"
#include "core/tensor/backend/cuda/kernels/tensor_ops.hpp"
#include "core/tensor/backend/cuda/runtime/cuda_memory_guard.hpp"

#include <vector>

namespace lfs::core::internal {
    namespace {
        template <class T>
        T* cuda_pointer(const StorageRef storage) {
            LFS_ASSERT_MSG(storage.backend == GpuBackend::CUDA,
                           "CUDA movement adapter received non-CUDA storage");
            return reinterpret_cast<T*>(
                static_cast<unsigned char*>(storage.data) + storage.byte_offset);
        }

        template <class T>
        const T* cuda_const_pointer(const StorageRef storage) {
            return cuda_pointer<const T>(storage);
        }

        std::vector<size_t> layout_vector(
            const std::array<size_t, MAX_TENSOR_RANK>& values, const size_t rank) {
            return std::vector<size_t>(values.begin(), values.begin() + rank);
        }

        struct LayoutVectorScratch {
            std::vector<size_t> dims;
            std::vector<size_t> strides;
        };

        LayoutVectorScratch& immediate_layout_vectors(const StridedLayout& layout) {
            thread_local LayoutVectorScratch scratch;
            bool matches = scratch.dims.size() == layout.rank &&
                           scratch.strides.size() == layout.rank;
            for (size_t i = 0; matches && i < layout.rank; ++i) {
                matches = scratch.dims[i] == layout.dims[i] &&
                          scratch.strides[i] == layout.strides[i];
            }
            if (!matches) {
                scratch.dims.assign(layout.dims.begin(), layout.dims.begin() + layout.rank);
                scratch.strides.assign(layout.strides.begin(),
                                       layout.strides.begin() + layout.rank);
            }
            return scratch;
        }

        std::vector<Tensor> tensor_views(
            const std::span<const StorageRef> inputs,
            const std::span<const StridedLayout> layouts,
            const ExecContext context) {
            LFS_ASSERT_MSG(inputs.size() == layouts.size(),
                           "cat storage and layout counts must match");
            std::vector<Tensor> tensors;
            tensors.reserve(inputs.size());
            for (size_t i = 0; i < inputs.size(); ++i) {
                tensors.push_back(Tensor::from_blob(
                    cuda_pointer<void>(inputs[i]),
                    TensorShape(layout_vector(layouts[i].dims, layouts[i].rank)),
                    Device::CUDA, inputs[i].dtype, context.cuda_stream));
            }
            return tensors;
        }

        void copy_metadata_to_device(
            CudaDeviceMemory<size_t>& device_shape,
            CudaDeviceMemory<size_t>& device_strides,
            const StridedLayout& layout, const ExecContext context) {
            LFS_ASSERT_MSG(device_shape.valid() && device_strides.valid(),
                           "failed to allocate CUDA shape/stride metadata");
            auto& backend = backend_ops(GpuBackend::CUDA);
            // Pageable source: cudaMemcpyAsync stages the bytes synchronously, so the
            // caller's stack layout may go out of scope after this call returns.
            backend.copy_host_to_device(CopyRequest{
                .src = raw_storage_ref(const_cast<size_t*>(layout.dims.data())),
                .dst = raw_device_storage_ref(device_shape.get(), GpuBackend::CUDA),
                .bytes = layout.rank * sizeof(size_t),
                .synchronous = false,
                .context = context,
            });
            backend.copy_host_to_device(CopyRequest{
                .src = raw_storage_ref(const_cast<size_t*>(layout.strides.data())),
                .dst = raw_device_storage_ref(device_strides.get(), GpuBackend::CUDA),
                .bytes = layout.rank * sizeof(size_t),
                .synchronous = false,
                .context = context,
            });
        }
    } // namespace

    void CudaBackendOps::strided_copy(
        const StorageRef input, const StorageRef output,
        const StridedLayout& input_layout, const ExecContext context) {
        LFS_FACADE_TRACE(strided_copy);
        CudaDeviceMemory<size_t> device_shape(input_layout.rank, context.cuda_stream);
        CudaDeviceMemory<size_t> device_strides(input_layout.rank, context.cuda_stream);
        copy_metadata_to_device(device_shape, device_strides, input_layout, context);
        tensor_ops::launch_strided_copy(
            cuda_const_pointer<void>(input), cuda_pointer<void>(output),
            device_shape.get(), device_strides.get(), input_layout.rank,
            input_layout.element_count, input.dtype, context.cuda_stream);
    }

    void CudaBackendOps::strided_copy_immediate(
        const StorageRef input, const StorageRef output,
        const StridedLayout& input_layout, const ExecContext context) {
        LFS_FACADE_TRACE(strided_copy_immediate);
        const LayoutVectorScratch& layout = immediate_layout_vectors(input_layout);
        tensor_ops::launch_strided_copy_immediate(
            cuda_const_pointer<void>(input), cuda_pointer<void>(output),
            layout.dims, layout.strides,
            input_layout.element_count, input.dtype, context.cuda_stream);
    }

    void CudaBackendOps::strided_upload(
        const StorageRef host_input, const StorageRef output,
        const StridedLayout& input_layout, const ExecContext context) {
        LFS_FACADE_TRACE(strided_upload);
        if (input_layout.rank == 3) {
            tensor_ops::launch_strided_upload(
                cuda_const_pointer<void>(host_input), cuda_pointer<void>(output),
                input_layout.dims.data(), input_layout.strides.data(),
                input_layout.rank, input_layout.element_count, output.dtype,
                context.cuda_stream);
            return;
        }

        CudaDeviceMemory<size_t> device_shape(input_layout.rank, context.cuda_stream);
        CudaDeviceMemory<size_t> device_strides(input_layout.rank, context.cuda_stream);
        copy_metadata_to_device(device_shape, device_strides, input_layout, context);
        tensor_ops::launch_strided_upload(
            cuda_const_pointer<void>(host_input), cuda_pointer<void>(output),
            device_shape.get(), device_strides.get(), input_layout.rank,
            input_layout.element_count, output.dtype, context.cuda_stream);
    }

    void CudaBackendOps::strided_scatter(
        const StorageRef input, const StorageRef output,
        const StridedLayout& output_layout, const ExecContext context) {
        LFS_FACADE_TRACE(strided_scatter);
        CudaDeviceMemory<size_t> device_shape(output_layout.rank, context.cuda_stream);
        CudaDeviceMemory<size_t> device_strides(output_layout.rank, context.cuda_stream);
        copy_metadata_to_device(device_shape, device_strides, output_layout, context);
        tensor_ops::launch_strided_scatter(
            cuda_const_pointer<void>(input), cuda_pointer<void>(output),
            device_shape.get(), device_strides.get(), output_layout.rank,
            output_layout.element_count, output.dtype, context.cuda_stream);
    }

    void CudaBackendOps::strided_scatter_immediate(
        const StorageRef input, const StorageRef output,
        const StridedLayout& output_layout, const ExecContext context) {
        LFS_FACADE_TRACE(strided_scatter_immediate);
        const LayoutVectorScratch& layout = immediate_layout_vectors(output_layout);
        tensor_ops::launch_strided_scatter_immediate(
            cuda_const_pointer<void>(input), cuda_pointer<void>(output),
            layout.dims, layout.strides,
            output_layout.element_count, output.dtype, context.cuda_stream);
    }

    void CudaBackendOps::strided_scatter_int32_to_float32(
        const StorageRef input, const StorageRef output,
        const StridedLayout& output_layout, const ExecContext context) {
        LFS_FACADE_TRACE(strided_scatter_int32_to_float32);
        // This specialized launcher reads its rank-2 metadata on the host before
        // passing scalar dimensions to the kernel.
        tensor_ops::launch_strided_scatter_int32_to_float32(
            cuda_const_pointer<void>(input), cuda_pointer<void>(output),
            output_layout.dims.data(), output_layout.strides.data(), output_layout.rank,
            output_layout.element_count, context.cuda_stream);
    }

    void CudaBackendOps::cat_last_dim(
        const StorageRef output, const std::span<const StorageRef> inputs,
        const std::span<const StridedLayout> layouts, const size_t num_rows,
        const size_t row_size, const size_t element_size,
        const ExecContext context) {
        LFS_FACADE_TRACE(cat_last_dim);
        const std::vector<Tensor> tensors = tensor_views(inputs, layouts, context);
        tensor_ops::launch_cat_last_dim(
            cuda_pointer<void>(output), tensors, num_rows, row_size, element_size,
            context.cuda_stream);
    }

    void CudaBackendOps::cat_middle_dim(
        const StorageRef output, const std::span<const StorageRef> inputs,
        const std::span<const StridedLayout> layouts, const size_t outer_size,
        const size_t inner_size, const int dim, const size_t element_size,
        const ExecContext context) {
        LFS_FACADE_TRACE(cat_middle_dim);
        const std::vector<Tensor> tensors = tensor_views(inputs, layouts, context);
        tensor_ops::launch_cat_middle_dim(
            cuda_pointer<void>(output), tensors, outer_size, inner_size, dim,
            element_size, context.cuda_stream);
    }

    void CudaBackendOps::pad(
        const StorageRef input, const StorageRef output,
        const StridedLayout& input_layout, const StridedLayout& output_layout,
        const std::array<size_t, MAX_TENSOR_RANK>& pad_before,
        const ExecContext context) {
        LFS_FACADE_TRACE(pad);
        LFS_ASSERT_MSG(input.dtype == DataType::Float32,
                       "CUDA pad supports only Float32");
        tensor_ops::launch_pad(
            cuda_const_pointer<float>(input), cuda_pointer<float>(output),
            input_layout.dims.data(), input_layout.strides.data(),
            output_layout.dims.data(), pad_before.data(), input_layout.rank,
            input_layout.element_count, context.cuda_stream);
    }

} // namespace lfs::core::internal
