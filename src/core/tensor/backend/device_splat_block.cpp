/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/splat_block.hpp"

#include <limits>
#include <stdexcept>
#include <vector>

namespace lfs::core {
    namespace {

        class DeviceSplatBlockOps final : public SplatBlockOps {
        public:
            std::size_t granularity(int) const override { return 256; }

            std::shared_ptr<ExportableBlock> reserve(std::size_t, int,
                                                     const std::size_t reserve_bytes) override {
                if (reserve_bytes == 0) {
                    throw std::runtime_error("SplatExportableStorage::create: layout is empty");
                }
                auto backing = std::make_shared<Tensor>(
                    Tensor::zeros({reserve_bytes}, Device::GPU, DataType::UInt8));
                auto block = std::make_shared<ExportableBlock>();
                block->device_ptr = backing->data_ptr();
                block->reserved_bytes = backing->bytes();
                block->committed_bytes = 0;
                block->state = backing;
                block->chunks.push_back(ExportableChunk{0, 0, {}});
                return block;
            }

            void commit_range(const std::shared_ptr<ExportableBlock>& block, std::size_t,
                              const std::size_t bytes) override {
                if (!block || bytes == 0) {
                    return;
                }
                block->committed_bytes += bytes;
                if (block->chunks.empty()) {
                    block->chunks.push_back(ExportableChunk{0, block->committed_bytes, {}});
                } else {
                    block->chunks.front().bytes = block->committed_bytes;
                }
            }

            void prepare_growth() override {}

            void init_growth_slack(const std::shared_ptr<ExportableBlock>& block,
                                   const std::size_t opacity_offset,
                                   const std::size_t rotation_offset,
                                   const std::size_t n_slack) override {
                if (!block || n_slack == 0) {
                    return;
                }
                auto backing = std::static_pointer_cast<Tensor>(block->state);
                if (!backing || !backing->is_valid()) {
                    throw std::runtime_error(
                        "SplatExportableStorage::grow: slack init failed: backing tensor missing");
                }
                try {
                    Tensor opacity = Tensor::view_sharing_storage(
                        *backing, opacity_offset, TensorShape({n_slack, std::size_t{1}}), n_slack,
                        DataType::Float32, "splat.exportable");
                    opacity.fill_(-std::numeric_limits<float>::infinity());
                    std::vector<float> rotation_host(n_slack * 4, 0.0f);
                    for (std::size_t i = 0; i < n_slack; ++i) {
                        rotation_host[i * 4] = 1.0f;
                    }
                    Tensor rotation_src = Tensor::from_vector(
                        rotation_host, TensorShape({n_slack, std::size_t{4}}), Device::CPU);
                    Tensor rotation = Tensor::view_sharing_storage(
                        *backing, rotation_offset, TensorShape({n_slack, std::size_t{4}}), n_slack,
                        DataType::Float32, "splat.exportable");
                    rotation.copy_from(rotation_src);
                } catch (const std::exception& error) {
                    throw std::runtime_error(std::string(
                                                 "SplatExportableStorage::grow: slack init failed: ") +
                                             error.what());
                }
            }

            Tensor bind_region(const std::shared_ptr<ExportableBlock>& block, void*,
                               const std::size_t byte_offset, TensorShape shape,
                               const std::size_t capacity, const DataType dtype,
                               std::string external_kind) override {
                auto backing = std::static_pointer_cast<Tensor>(block ? block->state : nullptr);
                if (!backing || !backing->is_valid()) {
                    throw std::runtime_error(
                        "SplatExportableStorage allocator: backing tensor missing");
                }
                return Tensor::view_sharing_storage(*backing, byte_offset, std::move(shape),
                                                    capacity, dtype, std::move(external_kind));
            }
        };

    } // namespace

    SplatBlockOps& device_splat_block_ops() {
        static DeviceSplatBlockOps ops;
        return ops;
    }

} // namespace lfs::core
