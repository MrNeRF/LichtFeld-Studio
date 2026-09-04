/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/export.hpp"
#include "descriptors.hpp"

#include <array>
#include <span>

namespace lfs::core {
    class Tensor;
    class MemoryInfo;

    namespace tensor_ops {
        struct FusedPointwiseOpChain;
    }

    namespace internal {

        class LFS_CORE_API GpuBackendOps {
        public:
            virtual ~GpuBackendOps();

            // Sub-lane A: pointwise, expressions, lazy, cast and fill.
            virtual void unary(const PointwiseProgram& program,
                               StorageRef input, StorageRef output,
                               size_t count, ExecContext context) = 0;
            virtual void binary(const PointwiseProgram& program,
                                StorageRef lhs, StorageRef rhs, StorageRef output,
                                size_t count, ExecContext context) = 0;
            virtual void scalar(const PointwiseProgram& program,
                                StorageRef input, StorageRef output,
                                size_t count, ExecContext context) = 0;
            virtual void broadcast_binary(const PointwiseProgram& program,
                                          StorageRef lhs, const StridedLayout& lhs_layout,
                                          StorageRef rhs, const StridedLayout& rhs_layout,
                                          StorageRef output, const StridedLayout& output_layout,
                                          ExecContext context) = 0;
            virtual void fused_pointwise_chain(
                StorageRef input, StorageRef output, size_t count,
                const tensor_ops::FusedPointwiseOpChain& chain,
                ExecContext context) = 0;
            virtual void clamp_scalar(StorageRef data, ScalarOperand minimum,
                                      ScalarOperand maximum, size_t count,
                                      ExecContext context) = 0;
            virtual void clamp_fused(StorageRef input, StorageRef output,
                                     ScalarOperand minimum, ScalarOperand maximum,
                                     size_t count, ExecContext context) = 0;
            virtual void clamp_scalar_int(StorageRef data, ScalarOperand minimum,
                                          ScalarOperand maximum, size_t count,
                                          ExecContext context) = 0;
            virtual void convert_type(StorageRef input, StorageRef output,
                                      size_t count, ExecContext context) = 0;
            virtual void fill_strided(StorageRef output, const StridedLayout& layout,
                                      ScalarOperand value, ExecContext context) = 0;
            virtual void load_fill(StorageRef output, size_t count,
                                   ScalarOperand value, ExecContext context) = 0;

            // Sub-lane B appends reductions, scan, sort, matrix, NN and random here.
            // These four scalar reductions synchronize before returning the host value.
            virtual float sum_scalar(StorageRef input, size_t count, ExecContext context) = 0;
            virtual float mean_scalar(StorageRef input, size_t count, ExecContext context) = 0;
            virtual float max_scalar(StorageRef input, size_t count, ExecContext context) = 0;
            virtual float min_scalar(StorageRef input, size_t count, ExecContext context) = 0;
            virtual void reduce(StorageRef input, StorageRef output,
                                const StridedLayout& input_layout,
                                const ReduceProgram& program, ExecContext context) = 0;
            virtual void column_reduce(StorageRef input, StorageRef output,
                                       size_t rows, size_t columns,
                                       const ReduceProgram& program,
                                       ExecContext context) = 0;
            virtual void strided_reduce(StorageRef input, StorageRef output,
                                        size_t outer_size, size_t reduce_size,
                                        size_t inner_size, const ReduceProgram& program,
                                        ExecContext context) = 0;
            virtual void fused_transform_reduce(
                StorageRef input, StorageRef output, size_t count,
                const tensor_ops::FusedPointwiseOpChain& chain,
                const ReduceProgram& program,
                ExecContext context) = 0;
            virtual void fused_segmented_transform_reduce(
                StorageRef input, StorageRef output, size_t segment_count,
                size_t segment_size, const tensor_ops::FusedPointwiseOpChain& chain,
                const ReduceProgram& program, ExecContext context) = 0;
            // Count reductions synchronize and download the counter before returning.
            virtual size_t count_nonzero_bool(StorageRef input, size_t count,
                                              ExecContext context) = 0;
            virtual size_t count_nonzero_float(StorageRef input, size_t count,
                                               ExecContext context) = 0;
            // NaN and Inf checks synchronize and download their flag before returning.
            virtual bool has_nan(StorageRef input, size_t count, ExecContext context) = 0;
            virtual bool has_inf(StorageRef input, size_t count, ExecContext context) = 0;
            virtual void cumsum(StorageRef data, const StridedLayout& layout, int dim,
                                ExecContext context) = 0;
            virtual void sort_1d(StorageRef values, StorageRef indices, size_t count,
                                 const SortProgram& program, ExecContext context) = 0;
            virtual void sort_2d(StorageRef values, StorageRef indices,
                                 const SortProgram& program, ExecContext context) = 0;
            virtual void sgemm(StorageRef lhs, StorageRef rhs, StorageRef output,
                               const GemmProgram& program, ExecContext context) = 0;
            virtual void sgemm_tn(StorageRef lhs, StorageRef rhs, StorageRef output,
                                  const GemmProgram& program, ExecContext context) = 0;
            virtual void sgemm_batched(StorageRef lhs, StorageRef rhs, StorageRef output,
                                       const GemmProgram& program, ExecContext context) = 0;
            virtual void sgemm_bias_relu(StorageRef lhs, StorageRef rhs, StorageRef bias,
                                         StorageRef output, const GemmProgram& program,
                                         ExecContext context) = 0;
            virtual void dot_product(StorageRef lhs, StorageRef rhs, StorageRef output,
                                     size_t count, ExecContext context) = 0;
            virtual void diag(StorageRef diagonal, StorageRef output, size_t count,
                              ExecContext context) = 0;
            virtual void eye(StorageRef output, size_t rows, size_t columns,
                             ExecContext context) = 0;
            virtual void cdist(StorageRef lhs, StorageRef rhs, StorageRef output,
                               size_t lhs_rows, size_t rhs_rows, size_t columns, float p,
                               ExecContext context) = 0;
            virtual void max_pool2d(StorageRef input, StorageRef output,
                                    const PoolProgram& program, ExecContext context) = 0;
            virtual void adaptive_avg_pool2d(StorageRef input, StorageRef output,
                                             const PoolProgram& program,
                                             ExecContext context) = 0;
            virtual void bias_add(StorageRef input, StorageRef bias, StorageRef output,
                                  int count, int channels, int spatial_size,
                                  ExecContext context) = 0;
            virtual void bias_relu(StorageRef input, StorageRef bias, StorageRef output,
                                   int count, int channels, int spatial_size,
                                   ExecContext context) = 0;
            virtual void relu(StorageRef input, StorageRef output, int count,
                              ExecContext context) = 0;
            virtual void uniform(StorageRef output, const RandomProgram& program,
                                 ExecContext context) = 0;
            virtual void bernoulli(StorageRef output, const RandomProgram& program,
                                   ExecContext context) = 0;
            virtual void randint(StorageRef output, const RandomProgram& program,
                                 ExecContext context) = 0;
            virtual void multinomial(StorageRef weights, StorageRef output,
                                     const RandomProgram& program, ExecContext context) = 0;
            // Odd-sized normal generation synchronizes after copying from its scratch.
            virtual void normal(StorageRef output, StorageRef odd_count_scratch,
                                const RandomProgram& program, ExecContext context) = 0;

            // Sub-lane C: index, masking, movement, allocation, copy and sync.
            virtual void gather(StorageRef input, StorageRef indices, StorageRef output,
                                const StridedLayout& input_layout,
                                const StridedLayout& index_layout,
                                const IndexProgram& program, ExecContext context) = 0;
            virtual void gather_fused_unary(StorageRef input, StorageRef indices,
                                            StorageRef output, PointwiseOp unary,
                                            const IndexProgram& program,
                                            ExecContext context) = 0;
            virtual void take(StorageRef input, StorageRef indices, StorageRef output,
                              const IndexProgram& program, ExecContext context) = 0;
            virtual void index_select(StorageRef input, StorageRef indices,
                                      StorageRef output,
                                      const StridedLayout& input_layout,
                                      const IndexProgram& program,
                                      ExecContext context) = 0;
            virtual void scatter(StorageRef output, StorageRef indices, StorageRef source,
                                 const StridedLayout& output_layout,
                                 const StridedLayout& index_layout,
                                 const IndexProgram& program, ExecContext context) = 0;
            virtual void index_copy(StorageRef output, StorageRef indices,
                                    StorageRef source,
                                    const StridedLayout& output_layout,
                                    const IndexProgram& program,
                                    ExecContext context) = 0;
            virtual void index_add(StorageRef output, StorageRef indices,
                                   StorageRef source,
                                   const StridedLayout& output_layout,
                                   const IndexProgram& program,
                                   ExecContext context) = 0;
            virtual void index_fill(StorageRef output, StorageRef indices,
                                    const StridedLayout& output_layout,
                                    const IndexProgram& program, ScalarOperand value,
                                    ExecContext context) = 0;
            virtual void index_put(StorageRef output, StorageRef indices,
                                   StorageRef values, const IndexProgram& program,
                                   ExecContext context) = 0;
            virtual void strided_scatter(StorageRef input, StorageRef output,
                                         const StridedLayout& output_layout,
                                         ExecContext context) = 0;
            virtual void strided_scatter_immediate(
                StorageRef input, StorageRef output,
                const StridedLayout& output_layout, ExecContext context) = 0;
            virtual void strided_scatter_int32_to_float32(
                StorageRef input, StorageRef output,
                const StridedLayout& output_layout, ExecContext context) = 0;
            virtual void masked_fill(StorageRef output, StorageRef mask,
                                     const MaskProgram& program,
                                     ExecContext context) = 0;
            virtual size_t masked_select(StorageRef input, StorageRef mask,
                                         StorageRef output,
                                         const MaskProgram& program,
                                         ExecContext context) = 0;
            virtual void masked_scatter(StorageRef output, StorageRef mask,
                                        StorageRef source,
                                        const MaskProgram& program,
                                        ExecContext context) = 0;
            virtual void and_live(StorageRef mask, StorageRef live_mask,
                                  const MaskProgram& program,
                                  ExecContext context) = 0;
            virtual void where(StorageRef condition, StorageRef x, StorageRef y,
                               StorageRef output,
                               const StridedLayout& condition_layout,
                               const StridedLayout& x_layout,
                               const StridedLayout& y_layout,
                               const StridedLayout& output_layout,
                               ExecContext context) = 0;
            virtual size_t nonzero(StorageRef input, StorageRef output,
                                   const MaskProgram& program,
                                   ExecContext context) = 0;
            virtual size_t nonzero_bool(StorageRef input, StorageRef output,
                                        const MaskProgram& program,
                                        ExecContext context) = 0;
            virtual void strided_copy(StorageRef input, StorageRef output,
                                      const StridedLayout& input_layout,
                                      ExecContext context) = 0;
            virtual void strided_copy_immediate(StorageRef input, StorageRef output,
                                                const StridedLayout& input_layout,
                                                ExecContext context) = 0;
            virtual void strided_upload(StorageRef host_input, StorageRef output,
                                        const StridedLayout& input_layout,
                                        ExecContext context) = 0;
            virtual void cat_last_dim(StorageRef output,
                                      std::span<const StorageRef> inputs,
                                      std::span<const StridedLayout> layouts,
                                      size_t num_rows, size_t row_size,
                                      size_t element_size,
                                      ExecContext context) = 0;
            virtual void cat_middle_dim(StorageRef output,
                                        std::span<const StorageRef> inputs,
                                        std::span<const StridedLayout> layouts,
                                        size_t outer_size, size_t inner_size,
                                        int dim, size_t element_size,
                                        ExecContext context) = 0;
            virtual void pad(StorageRef input, StorageRef output,
                             const StridedLayout& input_layout,
                             const StridedLayout& output_layout,
                             const std::array<size_t, MAX_TENSOR_RANK>& pad_before,
                             ExecContext context) = 0;

            virtual StorageRef allocate(size_t bytes, size_t alignment,
                                        ExecContext context) = 0;
            virtual void deallocate(StorageRef storage,
                                    ExecContext context) noexcept = 0;
            virtual void record_stream(StorageRef storage, ExecContext context) = 0;
            virtual void release_stream(ExecContext context) = 0;
            virtual void rehome_stream(StorageRef storage, ExecContext context) = 0;
            virtual void trim() = 0;
            virtual void trim_if_reserved_unused_exceeds(size_t threshold_bytes) = 0;
            virtual MemoryInfo stats() = 0;
            virtual void shutdown() = 0;
            virtual void set_allocation_iteration(int iteration) = 0;
            virtual void record_tensor_allocation(StorageRef storage,
                                                  const StridedLayout& layout,
                                                  size_t bytes) = 0;

            virtual void copy_host_to_device(const CopyRequest& request) = 0;
            virtual void copy_device_to_host(const CopyRequest& request) = 0;
            virtual void copy_device_to_device(const CopyRequest& request) = 0;
            virtual void memset(const FillRequest& request) = 0;

            virtual void synchronize_stream(ExecContext context) = 0;
            virtual void synchronize_device() = 0;
            virtual void wait_for(SyncToken token) = 0;
            virtual SyncToken bridge(ExecContext producer, ExecContext consumer) = 0;
            virtual PointerClass classify_pointer(const void* pointer) = 0;
            virtual bool stream_is_capturing(ExecContext context) = 0;
        };

        class CudaBackendOps final : public GpuBackendOps {
        public:
            ~CudaBackendOps() override;

            void unary(const PointwiseProgram& program,
                       StorageRef input, StorageRef output,
                       size_t count, ExecContext context) override;
            void binary(const PointwiseProgram& program,
                        StorageRef lhs, StorageRef rhs, StorageRef output,
                        size_t count, ExecContext context) override;
            void scalar(const PointwiseProgram& program,
                        StorageRef input, StorageRef output,
                        size_t count, ExecContext context) override;
            void broadcast_binary(const PointwiseProgram& program,
                                  StorageRef lhs, const StridedLayout& lhs_layout,
                                  StorageRef rhs, const StridedLayout& rhs_layout,
                                  StorageRef output, const StridedLayout& output_layout,
                                  ExecContext context) override;
            void fused_pointwise_chain(
                StorageRef input, StorageRef output, size_t count,
                const tensor_ops::FusedPointwiseOpChain& chain,
                ExecContext context) override;
            void clamp_scalar(StorageRef data, ScalarOperand minimum,
                              ScalarOperand maximum, size_t count,
                              ExecContext context) override;
            void clamp_fused(StorageRef input, StorageRef output,
                             ScalarOperand minimum, ScalarOperand maximum,
                             size_t count, ExecContext context) override;
            void clamp_scalar_int(StorageRef data, ScalarOperand minimum,
                                  ScalarOperand maximum, size_t count,
                                  ExecContext context) override;
            void convert_type(StorageRef input, StorageRef output,
                              size_t count, ExecContext context) override;
            void fill_strided(StorageRef output, const StridedLayout& layout,
                              ScalarOperand value, ExecContext context) override;
            void load_fill(StorageRef output, size_t count,
                           ScalarOperand value, ExecContext context) override;

            // Sub-lane B: reductions, scan, sort, matrix, NN and random.
            float sum_scalar(StorageRef input, size_t count, ExecContext context) override;
            float mean_scalar(StorageRef input, size_t count, ExecContext context) override;
            float max_scalar(StorageRef input, size_t count, ExecContext context) override;
            float min_scalar(StorageRef input, size_t count, ExecContext context) override;
            void reduce(StorageRef input, StorageRef output,
                        const StridedLayout& input_layout,
                        const ReduceProgram& program, ExecContext context) override;
            void column_reduce(StorageRef input, StorageRef output,
                               size_t rows, size_t columns,
                               const ReduceProgram& program,
                               ExecContext context) override;
            void strided_reduce(StorageRef input, StorageRef output,
                                size_t outer_size, size_t reduce_size,
                                size_t inner_size, const ReduceProgram& program,
                                ExecContext context) override;
            void fused_transform_reduce(
                StorageRef input, StorageRef output, size_t count,
                const tensor_ops::FusedPointwiseOpChain& chain,
                const ReduceProgram& program,
                ExecContext context) override;
            void fused_segmented_transform_reduce(
                StorageRef input, StorageRef output, size_t segment_count,
                size_t segment_size, const tensor_ops::FusedPointwiseOpChain& chain,
                const ReduceProgram& program, ExecContext context) override;
            size_t count_nonzero_bool(StorageRef input, size_t count,
                                      ExecContext context) override;
            size_t count_nonzero_float(StorageRef input, size_t count,
                                       ExecContext context) override;
            bool has_nan(StorageRef input, size_t count, ExecContext context) override;
            bool has_inf(StorageRef input, size_t count, ExecContext context) override;
            void cumsum(StorageRef data, const StridedLayout& layout, int dim,
                        ExecContext context) override;
            void sort_1d(StorageRef values, StorageRef indices, size_t count,
                         const SortProgram& program, ExecContext context) override;
            void sort_2d(StorageRef values, StorageRef indices,
                         const SortProgram& program, ExecContext context) override;
            void sgemm(StorageRef lhs, StorageRef rhs, StorageRef output,
                       const GemmProgram& program, ExecContext context) override;
            void sgemm_tn(StorageRef lhs, StorageRef rhs, StorageRef output,
                          const GemmProgram& program, ExecContext context) override;
            void sgemm_batched(StorageRef lhs, StorageRef rhs, StorageRef output,
                               const GemmProgram& program, ExecContext context) override;
            void sgemm_bias_relu(StorageRef lhs, StorageRef rhs, StorageRef bias,
                                 StorageRef output, const GemmProgram& program,
                                 ExecContext context) override;
            void dot_product(StorageRef lhs, StorageRef rhs, StorageRef output,
                             size_t count, ExecContext context) override;
            void diag(StorageRef diagonal, StorageRef output, size_t count,
                      ExecContext context) override;
            void eye(StorageRef output, size_t rows, size_t columns,
                     ExecContext context) override;
            void cdist(StorageRef lhs, StorageRef rhs, StorageRef output,
                       size_t lhs_rows, size_t rhs_rows, size_t columns, float p,
                       ExecContext context) override;
            void max_pool2d(StorageRef input, StorageRef output,
                            const PoolProgram& program, ExecContext context) override;
            void adaptive_avg_pool2d(StorageRef input, StorageRef output,
                                     const PoolProgram& program,
                                     ExecContext context) override;
            void bias_add(StorageRef input, StorageRef bias, StorageRef output,
                          int count, int channels, int spatial_size,
                          ExecContext context) override;
            void bias_relu(StorageRef input, StorageRef bias, StorageRef output,
                           int count, int channels, int spatial_size,
                           ExecContext context) override;
            void relu(StorageRef input, StorageRef output, int count,
                      ExecContext context) override;
            void uniform(StorageRef output, const RandomProgram& program,
                         ExecContext context) override;
            void bernoulli(StorageRef output, const RandomProgram& program,
                           ExecContext context) override;
            void randint(StorageRef output, const RandomProgram& program,
                         ExecContext context) override;
            void multinomial(StorageRef weights, StorageRef output,
                             const RandomProgram& program, ExecContext context) override;
            void normal(StorageRef output, StorageRef odd_count_scratch,
                        const RandomProgram& program, ExecContext context) override;

            // Sub-lane C: index, masking, movement, allocation, copy and sync.
            void gather(StorageRef input, StorageRef indices, StorageRef output,
                        const StridedLayout& input_layout,
                        const StridedLayout& index_layout,
                        const IndexProgram& program, ExecContext context) override;
            void gather_fused_unary(StorageRef input, StorageRef indices,
                                    StorageRef output, PointwiseOp unary,
                                    const IndexProgram& program,
                                    ExecContext context) override;
            void take(StorageRef input, StorageRef indices, StorageRef output,
                      const IndexProgram& program, ExecContext context) override;
            void index_select(StorageRef input, StorageRef indices, StorageRef output,
                              const StridedLayout& input_layout,
                              const IndexProgram& program,
                              ExecContext context) override;
            void scatter(StorageRef output, StorageRef indices, StorageRef source,
                         const StridedLayout& output_layout,
                         const StridedLayout& index_layout,
                         const IndexProgram& program, ExecContext context) override;
            void index_copy(StorageRef output, StorageRef indices, StorageRef source,
                            const StridedLayout& output_layout,
                            const IndexProgram& program,
                            ExecContext context) override;
            void index_add(StorageRef output, StorageRef indices, StorageRef source,
                           const StridedLayout& output_layout,
                           const IndexProgram& program,
                           ExecContext context) override;
            void index_fill(StorageRef output, StorageRef indices,
                            const StridedLayout& output_layout,
                            const IndexProgram& program, ScalarOperand value,
                            ExecContext context) override;
            void index_put(StorageRef output, StorageRef indices, StorageRef values,
                           const IndexProgram& program,
                           ExecContext context) override;
            void strided_scatter(StorageRef input, StorageRef output,
                                 const StridedLayout& output_layout,
                                 ExecContext context) override;
            void strided_scatter_immediate(StorageRef input, StorageRef output,
                                           const StridedLayout& output_layout,
                                           ExecContext context) override;
            void strided_scatter_int32_to_float32(
                StorageRef input, StorageRef output,
                const StridedLayout& output_layout, ExecContext context) override;
            void masked_fill(StorageRef output, StorageRef mask,
                             const MaskProgram& program,
                             ExecContext context) override;
            size_t masked_select(StorageRef input, StorageRef mask, StorageRef output,
                                 const MaskProgram& program,
                                 ExecContext context) override;
            void masked_scatter(StorageRef output, StorageRef mask,
                                StorageRef source, const MaskProgram& program,
                                ExecContext context) override;
            void and_live(StorageRef mask, StorageRef live_mask,
                          const MaskProgram& program,
                          ExecContext context) override;
            void where(StorageRef condition, StorageRef x, StorageRef y,
                       StorageRef output,
                       const StridedLayout& condition_layout,
                       const StridedLayout& x_layout,
                       const StridedLayout& y_layout,
                       const StridedLayout& output_layout,
                       ExecContext context) override;
            size_t nonzero(StorageRef input, StorageRef output,
                           const MaskProgram& program,
                           ExecContext context) override;
            size_t nonzero_bool(StorageRef input, StorageRef output,
                                const MaskProgram& program,
                                ExecContext context) override;
            void strided_copy(StorageRef input, StorageRef output,
                              const StridedLayout& input_layout,
                              ExecContext context) override;
            void strided_copy_immediate(StorageRef input, StorageRef output,
                                        const StridedLayout& input_layout,
                                        ExecContext context) override;
            void strided_upload(StorageRef host_input, StorageRef output,
                                const StridedLayout& input_layout,
                                ExecContext context) override;
            void cat_last_dim(StorageRef output, std::span<const StorageRef> inputs,
                              std::span<const StridedLayout> layouts,
                              size_t num_rows, size_t row_size,
                              size_t element_size, ExecContext context) override;
            void cat_middle_dim(StorageRef output,
                                std::span<const StorageRef> inputs,
                                std::span<const StridedLayout> layouts,
                                size_t outer_size, size_t inner_size,
                                int dim, size_t element_size,
                                ExecContext context) override;
            void pad(StorageRef input, StorageRef output,
                     const StridedLayout& input_layout,
                     const StridedLayout& output_layout,
                     const std::array<size_t, MAX_TENSOR_RANK>& pad_before,
                     ExecContext context) override;

            StorageRef allocate(size_t bytes, size_t alignment,
                                ExecContext context) override;
            void deallocate(StorageRef storage,
                            ExecContext context) noexcept override;
            void record_stream(StorageRef storage, ExecContext context) override;
            void release_stream(ExecContext context) override;
            void rehome_stream(StorageRef storage, ExecContext context) override;
            void trim() override;
            void trim_if_reserved_unused_exceeds(size_t threshold_bytes) override;
            MemoryInfo stats() override;
            void shutdown() override;
            void set_allocation_iteration(int iteration) override;
            void record_tensor_allocation(StorageRef storage,
                                          const StridedLayout& layout,
                                          size_t bytes) override;
            void copy_host_to_device(const CopyRequest& request) override;
            void copy_device_to_host(const CopyRequest& request) override;
            void copy_device_to_device(const CopyRequest& request) override;
            void memset(const FillRequest& request) override;
            void synchronize_stream(ExecContext context) override;
            void synchronize_device() override;
            void wait_for(SyncToken token) override;
            SyncToken bridge(ExecContext producer, ExecContext consumer) override;
            PointerClass classify_pointer(const void* pointer) override;
            bool stream_is_capturing(ExecContext context) override;
        };

        LFS_CORE_API GpuBackendOps& backend_ops(GpuBackend backend);
        LFS_CORE_API GpuBackendOps& backend_ops_for(const Tensor& tensor);

    } // namespace internal
} // namespace lfs::core
