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

            // Sub-lanes B and C append overrides in their owned regions.
        };

        LFS_CORE_API GpuBackendOps& backend_ops(GpuBackend backend);
        LFS_CORE_API GpuBackendOps& backend_ops_for(const Tensor& tensor);

    } // namespace internal
} // namespace lfs::core
