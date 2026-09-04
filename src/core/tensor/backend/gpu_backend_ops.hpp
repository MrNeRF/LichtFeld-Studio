/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/export.hpp"
#include "descriptors.hpp"

namespace lfs::core {
    class Tensor;

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

            // Sub-lane C appends index, masking, movement, allocation, copy and sync here.
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

            // Sub-lane C appends overrides in its owned region.
        };

        LFS_CORE_API GpuBackendOps& backend_ops(GpuBackend backend);
        LFS_CORE_API GpuBackendOps& backend_ops_for(const Tensor& tensor);

    } // namespace internal
} // namespace lfs::core
