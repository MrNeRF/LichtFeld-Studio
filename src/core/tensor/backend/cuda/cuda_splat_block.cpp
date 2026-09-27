/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/splat_block.hpp"
#include "core/tensor_cuda_interop.hpp"

#include <cuda_runtime.h>

#include <format>
#include <limits>
#include <stdexcept>
#include <vector>

namespace lfs::core {
    namespace {

        class CudaSplatBlockOps final : public SplatBlockOps {
        public:
            std::size_t granularity(const int device) const override {
                return exportable_allocation_granularity(device);
            }

            std::shared_ptr<ExportableBlock> reserve(const std::size_t initial_commit,
                                                     const int device,
                                                     const std::size_t reserve_bytes) override {
                auto block = allocateExportableDeviceBlock(
                    initial_commit, device, /*track_splat_bytes=*/true, reserve_bytes);
                if (!block) {
                    throw std::runtime_error(block.error());
                }
                return std::move(*block);
            }

            void commit_range(const std::shared_ptr<ExportableBlock>& block,
                              const std::size_t offset, const std::size_t bytes) override {
                auto committed = commitExportableDeviceRange(block, offset, bytes);
                if (!committed) {
                    throw std::runtime_error(committed.error());
                }
            }

            void prepare_growth() override {
                if (const auto err = cudaDeviceSynchronize(); err != cudaSuccess) {
                    throw std::runtime_error(cudaGetErrorString(err));
                }
            }

            void init_growth_slack(const std::shared_ptr<ExportableBlock>& block,
                                   const std::size_t opacity_offset,
                                   const std::size_t rotation_offset,
                                   const std::size_t n_slack) override {
                if (!block || !block->device_ptr) {
                    throw std::runtime_error(
                        "SplatExportableStorage::grow: slack opacity init failed: null block");
                }
                std::vector<float> opacity_host(n_slack, -std::numeric_limits<float>::infinity());
                std::vector<float> rotation_host(n_slack * 4, 0.0f);
                for (std::size_t i = 0; i < n_slack; ++i) {
                    rotation_host[i * 4] = 1.0f;
                }
                auto* opacity_dst = static_cast<char*>(block->device_ptr) + opacity_offset;
                auto* rotation_dst = static_cast<char*>(block->device_ptr) + rotation_offset;
                if (const auto err = cudaMemcpyAsync(opacity_dst,
                                                     opacity_host.data(),
                                                     opacity_host.size() * sizeof(float),
                                                     cudaMemcpyHostToDevice,
                                                     getCurrentCUDAStream());
                    err != cudaSuccess) {
                    throw std::runtime_error(std::format(
                        "SplatExportableStorage::grow: slack opacity init failed: {}",
                        cudaGetErrorString(err)));
                }
                if (const auto err = cudaMemcpyAsync(rotation_dst,
                                                     rotation_host.data(),
                                                     rotation_host.size() * sizeof(float),
                                                     cudaMemcpyHostToDevice,
                                                     getCurrentCUDAStream());
                    err != cudaSuccess) {
                    throw std::runtime_error(std::format(
                        "SplatExportableStorage::grow: slack rotation init failed: {}",
                        cudaGetErrorString(err)));
                }
                if (const auto err = cudaDeviceSynchronize(); err != cudaSuccess) {
                    throw std::runtime_error(std::format(
                        "SplatExportableStorage::grow: synchronize failed: {}",
                        cudaGetErrorString(err)));
                }
            }

            Tensor bind_region(const std::shared_ptr<ExportableBlock>& block, void* const data,
                               std::size_t, TensorShape shape, const std::size_t capacity,
                               const DataType dtype, std::string external_kind) override {
                return Tensor::from_external_owner(data,
                                                   std::move(shape),
                                                   Device::GPU,
                                                   dtype,
                                                   std::shared_ptr<void>(block),
                                                   capacity,
                                                   getCurrentCUDAStream(),
                                                   std::move(external_kind));
            }
        };

    } // namespace

    SplatBlockOps& cuda_splat_block_ops() {
        static CudaSplatBlockOps ops;
        return ops;
    }

} // namespace lfs::core
