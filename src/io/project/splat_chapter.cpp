/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "io/splat_chapter.hpp"

#include "core/cuda/sh_layout.cuh"
#include "core/cuda_error_typed.hpp"
#include "core/logger.hpp"
#include "core/pinned_memory_allocator.hpp"
#include "core/tensor/internal/cuda_stream_context.hpp"
#include "core/tensor/internal/memory_pool.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <cuda_runtime_api.h>
#include <exception>
#include <format>
#include <limits>
#include <ranges>
#include <streambuf>
#include <utility>

namespace lfs::io::project {

    namespace {

        constexpr std::uint32_t LFSP_MAGIC = 0x4c465350;
        constexpr std::array<std::byte, 8> RAW_MAGIC = {
            std::byte{'L'}, std::byte{'F'}, std::byte{'S'}, std::byte{'P'},
            std::byte{'L'}, std::byte{'T'}, std::byte{'2'}, std::byte{0}};
        constexpr std::size_t RAW_HEADER_BYTES = 40;
        constexpr std::size_t RAW_DESCRIPTOR_BYTES = 64;

        std::uint16_t get_u16(const std::span<const std::byte> bytes, const std::size_t at) {
            return static_cast<std::uint16_t>(std::to_integer<std::uint8_t>(bytes[at])) |
                   static_cast<std::uint16_t>(std::to_integer<std::uint8_t>(bytes[at + 1])) << 8;
        }
        std::uint32_t get_u32(const std::span<const std::byte> bytes, const std::size_t at) {
            std::uint32_t value = 0;
            for (std::size_t i = 0; i < 4; ++i)
                value |= static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(bytes[at + i])) << (8 * i);
            return value;
        }
        std::uint64_t get_u64(const std::span<const std::byte> bytes, const std::size_t at) {
            std::uint64_t value = 0;
            for (std::size_t i = 0; i < 8; ++i)
                value |= static_cast<std::uint64_t>(std::to_integer<std::uint8_t>(bytes[at + i])) << (8 * i);
            return value;
        }
        void put_u16(const std::span<std::byte> bytes, const std::size_t at, const std::uint16_t value) {
            bytes[at] = static_cast<std::byte>(value & 0xffu);
            bytes[at + 1] = static_cast<std::byte>(value >> 8);
        }
        void put_u32(const std::span<std::byte> bytes, const std::size_t at, const std::uint32_t value) {
            for (std::size_t i = 0; i < 4; ++i)
                bytes[at + i] = static_cast<std::byte>(value >> (8 * i));
        }
        void put_u64(const std::span<std::byte> bytes, const std::size_t at, const std::uint64_t value) {
            for (std::size_t i = 0; i < 8; ++i)
                bytes[at + i] = static_cast<std::byte>(value >> (8 * i));
        }

        struct RawTensor {
            std::uint32_t id = 0;
            lfs::core::DataType dtype{};
            std::vector<std::size_t> shape;
            std::uint64_t offset = 0;
            std::uint64_t length = 0;
        };

        lfs::Error splat_error(const lfs::ErrorCode code, std::string message,
                               std::string detail);

        lfs::Result<std::unique_ptr<lfs::core::SplatData>> hydrate_raw(
            const std::span<const std::byte> bytes,
            lfs::core::SplatTensorAllocator allocator) {
            if (bytes.size() < RAW_HEADER_BYTES ||
                !std::equal(RAW_MAGIC.begin(), RAW_MAGIC.end(), bytes.begin()) ||
                get_u16(bytes, 8) != 2 || get_u16(bytes, 10) != 0) {
                return splat_error(lfs::ErrorCode::DataLoss,
                                   "The raw splat payload header is invalid.",
                                   "expected LFSPLT2 version 2");
            }
            const int active = static_cast<int>(get_u32(bytes, 12));
            const int maximum = static_cast<int>(get_u32(bytes, 16));
            float scene_scale = 0.0f;
            const auto scene_bits = get_u32(bytes, 20);
            std::memcpy(&scene_scale, &scene_bits, sizeof(scene_scale));
            const auto count = get_u32(bytes, 24);
            const auto manifest_bytes = get_u64(bytes, 32);
            if (count == 0 || count > 8 || manifest_bytes > bytes.size() ||
                manifest_bytes < RAW_HEADER_BYTES + static_cast<std::uint64_t>(count) * RAW_DESCRIPTOR_BYTES + 8) {
                return splat_error(lfs::ErrorCode::DataLoss,
                                   "The raw splat payload manifest is invalid.",
                                   "tensor count or manifest length is out of range");
            }
            std::vector<RawTensor> descriptors;
            descriptors.reserve(count);
            std::uint32_t seen = 0;
            for (std::uint32_t i = 0; i < count; ++i) {
                const auto at = RAW_HEADER_BYTES + static_cast<std::size_t>(i) * RAW_DESCRIPTOR_BYTES;
                const auto id = get_u32(bytes, at);
                const auto rank = std::to_integer<std::uint8_t>(bytes[at + 5]);
                if (id >= 8 || (seen & (1u << id)) != 0 || rank > 4)
                    return splat_error(lfs::ErrorCode::DataLoss, "The raw splat tensor manifest is invalid.", "duplicate id or rank");
                seen |= 1u << id;
                const auto dtype = static_cast<lfs::core::DataType>(std::to_integer<std::uint8_t>(bytes[at + 4]));
                if (lfs::core::dtype_size(dtype) == 0)
                    return splat_error(lfs::ErrorCode::DataLoss, "The raw splat tensor manifest is invalid.", "unsupported dtype");
                std::vector<std::size_t> shape(rank);
                for (std::size_t d = 0; d < shape.size(); ++d) {
                    const auto dimension = get_u64(bytes, at + 8 + d * 8);
                    if (dimension > std::numeric_limits<std::size_t>::max())
                        return splat_error(lfs::ErrorCode::DataLoss, "The raw splat tensor manifest is invalid.", "dimension overflow");
                    shape[d] = static_cast<std::size_t>(dimension);
                }
                descriptors.push_back({id, dtype, std::move(shape), get_u64(bytes, at + 40), get_u64(bytes, at + 48)});
            }
            const auto frozen_count = get_u64(bytes, RAW_HEADER_BYTES + static_cast<std::size_t>(count) * RAW_DESCRIPTOR_BYTES);
            const auto ranges_end = manifest_bytes;
            if (frozen_count > 1'000'000 ||
                frozen_count > (ranges_end - (RAW_HEADER_BYTES + static_cast<std::size_t>(count) * RAW_DESCRIPTOR_BYTES + 8)) / 16)
                return splat_error(lfs::ErrorCode::DataLoss, "The raw splat payload manifest is invalid.", "frozen-range table overflow");
            std::vector<lfs::core::SplatData::FrozenRange> ranges;
            ranges.reserve(static_cast<std::size_t>(frozen_count));
            auto range_at = RAW_HEADER_BYTES + static_cast<std::size_t>(count) * RAW_DESCRIPTOR_BYTES + 8;
            for (std::uint64_t i = 0; i < frozen_count; ++i, range_at += 16)
                ranges.push_back({static_cast<std::size_t>(get_u64(bytes, range_at)), static_cast<std::size_t>(get_u64(bytes, range_at + 8))});
            const auto data_start = static_cast<std::size_t>(manifest_bytes);
            std::vector<lfs::core::Tensor> tensors(8);
            auto owner = std::shared_ptr<void>(
                const_cast<std::byte*>(bytes.data()), [](void*) {});
            std::uint64_t end = data_start;
            for (const auto& descriptor : descriptors) {
                std::uint64_t elements = 1;
                for (const auto dimension : descriptor.shape) {
                    if (dimension != 0 && elements > std::numeric_limits<std::uint64_t>::max() / dimension)
                        return splat_error(lfs::ErrorCode::DataLoss, "The raw splat tensor manifest is invalid.", "shape overflow");
                    elements *= dimension;
                }
                if (elements > std::numeric_limits<std::uint64_t>::max() / lfs::core::dtype_size(descriptor.dtype) ||
                    descriptor.length != elements * lfs::core::dtype_size(descriptor.dtype) ||
                    descriptor.offset < data_start || descriptor.offset > bytes.size() ||
                    descriptor.length > bytes.size() - descriptor.offset || descriptor.offset != end)
                    return splat_error(lfs::ErrorCode::DataLoss, "The raw splat tensor manifest is invalid.", "tensor range is overlapping or out of bounds");
                end = descriptor.offset + descriptor.length;
                auto tensor = lfs::core::Tensor::from_external_owner(
                    const_cast<std::byte*>(bytes.data() + descriptor.offset),
                    lfs::core::TensorShape(descriptor.shape), lfs::core::Device::CPU,
                    descriptor.dtype, owner);
                if (tensor.bytes() != descriptor.length)
                    return splat_error(lfs::ErrorCode::DataLoss, "The raw splat tensor manifest is invalid.", "tensor byte length mismatch");
                tensors[descriptor.id] = std::move(tensor);
            }
            if (end != bytes.size())
                return splat_error(lfs::ErrorCode::DataLoss, "The raw splat tensor manifest is invalid.", "tensor data does not cover the payload exactly");
            if (!tensors[0].is_valid() || !tensors[1].is_valid() || !tensors[2].is_valid() ||
                !tensors[3].is_valid() || !tensors[4].is_valid() || !tensors[5].is_valid())
                return splat_error(lfs::ErrorCode::DataLoss, "The raw splat tensor manifest is invalid.", "required tensor is missing");
            return lfs::core::SplatData::from_raw_tensors(
                active, maximum, scene_scale, std::move(tensors[0]), std::move(tensors[1]),
                std::move(tensors[2]), std::move(tensors[3]), std::move(tensors[4]),
                std::move(tensors[5]), std::move(tensors[6]), std::move(tensors[7]),
                std::move(ranges), std::move(allocator));
        }

        class SpanStreambuf final : public std::streambuf {
        public:
            explicit SpanStreambuf(const std::span<const char> bytes) {
                auto* begin = const_cast<char*>(bytes.data());
                setg(begin, begin, begin + bytes.size());
            }

        protected:
            pos_type seekoff(const off_type offset, const std::ios_base::seekdir direction,
                             const std::ios_base::openmode which = std::ios_base::in) override {
                if ((which & std::ios_base::in) == 0)
                    return pos_type(off_type(-1));
                const auto current = static_cast<off_type>(gptr() - eback());
                const auto end = static_cast<off_type>(egptr() - eback());
                off_type target = 0;
                if (direction == std::ios_base::beg) {
                    if (offset < 0 || offset > end)
                        return pos_type(off_type(-1));
                    target = offset;
                } else if (direction == std::ios_base::cur) {
                    if ((offset > 0 && offset > end - current) ||
                        (offset < 0 && offset < -current))
                        return pos_type(off_type(-1));
                    target = current + offset;
                } else if (direction == std::ios_base::end) {
                    if ((offset > 0 && offset > end) ||
                        (offset < 0 && offset < -end))
                        return pos_type(off_type(-1));
                    target = end + offset;
                } else
                    return pos_type(off_type(-1));
                setg(eback(), eback() + target, egptr());
                return pos_type(target);
            }

            pos_type seekpos(const pos_type position,
                             const std::ios_base::openmode which = std::ios_base::in) override {
                return seekoff(off_type(position), std::ios_base::beg, which);
            }
        };

        lfs::Error splat_error(const lfs::ErrorCode code, std::string message,
                               std::string detail) {
            return lfs::make_error(lfs::ErrorInit{
                .code = code,
                .domain = lfs::ErrorDomain::IO,
                .severity = lfs::Severity::Error,
                .retryability = lfs::Retryability::NotRetryable,
                .operation_id = {},
                .user_message = std::move(message),
                .detail = std::move(detail),
                .detection = LFS_SOURCE_SITE_CURRENT(),
                .fields = {},
                .native = std::nullopt,
            });
        }

    } // namespace

    namespace {

        lfs::Result<std::uint32_t> validate_lfsp(
            const std::span<const std::byte> bytes) {
            if (bytes.size() < 8) {
                return splat_error(
                    lfs::ErrorCode::DataLoss, "The embedded splat payload is truncated.",
                    std::format("SPLT payload has {} bytes; LFSP header needs 8", bytes.size()));
            }
            std::uint32_t magic = 0;
            std::uint32_t version = 0;
            std::memcpy(&magic, bytes.data(), sizeof(magic));
            std::memcpy(&version, bytes.data() + sizeof(magic), sizeof(version));
            if (magic != LFSP_MAGIC) {
                return splat_error(
                    lfs::ErrorCode::DataLoss, "The embedded splat payload has invalid magic.",
                    std::format("SPLT payload magic is 0x{:08x}, expected LFSP", magic));
            }
            if (version != 3 && version != 4) {
                return splat_error(
                    lfs::ErrorCode::Unsupported,
                    "This embedded splat payload version is not supported.",
                    std::format("LFSP version {} is unsupported", version));
            }
            return version;
        }

    } // namespace

    lfs::Result<SplatChapterPayload> SplatChapterPayload::from_lfsp(
        const std::span<const std::byte> bytes) {
        if (bytes.size() >= RAW_MAGIC.size() &&
            std::equal(RAW_MAGIC.begin(), RAW_MAGIC.end(), bytes.begin())) {
            SplatChapterPayload result;
            result.bytes_.assign(bytes.begin(), bytes.end());
            result.lfsp_version_ = 2;
            return result;
        }
        auto version = validate_lfsp(bytes);
        if (!version) {
            return std::move(version).error();
        }
        SplatChapterPayload result;
        result.bytes_.assign(bytes.begin(), bytes.end());
        result.lfsp_version_ = *version;
        return result;
    }

    lfs::Result<SplatChapterPayload> SplatChapterPayload::from_lfsp(
        std::vector<std::byte>&& bytes) {
        if (bytes.size() >= RAW_MAGIC.size() &&
            std::equal(RAW_MAGIC.begin(), RAW_MAGIC.end(), bytes.begin())) {
            SplatChapterPayload result;
            result.bytes_ = std::move(bytes);
            result.lfsp_version_ = 2;
            return result;
        }
        auto version = validate_lfsp(bytes);
        if (!version) {
            return std::move(version).error();
        }
        SplatChapterPayload result;
        result.bytes_ = std::move(bytes);
        result.lfsp_version_ = *version;
        return result;
    }

    lfs::Result<SplatChapterPayload> SplatChapterPayload::capture(
        const lfs::core::SplatData& model, const SplatSourceKind source_kind,
        const bool is_training_model) {
        const auto started = std::chrono::steady_clock::now();
        if (is_training_model) {
            return splat_error(
                lfs::ErrorCode::FailedPrecondition,
                "The training model cannot be written as a SPLT chapter.",
                "Training model state is authoritative only in CKPT");
        }
        if (must_reference_external(source_kind)) {
            return splat_error(
                lfs::ErrorCode::FailedPrecondition,
                "A live RAD node cannot be embedded in the project.",
                "Live RAD nodes remain external REFS records until explicitly baked");
        }
        try {
            struct StreamGuard {
                cudaStream_t stream = nullptr;

                ~StreamGuard() {
                    if (!stream)
                        return;
                    lfs::core::CudaMemoryPool::instance().release_stream(stream);
                    cudaStreamDestroy(stream);
                }
            } stream_guard;

            struct SourceTensor {
                std::uint32_t id;
                lfs::core::Tensor tensor;
                std::vector<std::size_t> shape;
                std::uint64_t bytes = 0;
                bool sh_range = false;
            };

            auto describe = [](const std::uint32_t id, lfs::core::Tensor tensor) {
                SourceTensor item;
                item.id = id;
                item.bytes = tensor.bytes();
                item.shape = tensor.shape().dims();
                item.tensor = std::move(tensor);
                return item;
            };
            std::vector<SourceTensor> source;
            source.reserve(8);
            source.push_back(describe(0, model.means().contiguous()));
            source.push_back(describe(1, model.sh0().contiguous()));
            const auto& resident_sh = model.shN();
            if (resident_sh.device() == lfs::core::Device::CUDA &&
                resident_sh.dtype() == lfs::core::DataType::Float32) {
                SourceTensor item;
                item.id = 2;
                item.tensor = resident_sh;
                item.shape = {static_cast<std::size_t>(model.size()),
                              model.max_sh_coeffs_rest(), 3};
                item.bytes = static_cast<std::uint64_t>(item.shape[0]) *
                             item.shape[1] * item.shape[2] * sizeof(float);
                item.sh_range = true;
                source.push_back(std::move(item));
            } else {
                source.push_back(describe(2, model.shN_canonical().contiguous()));
            }
            source.push_back(describe(3, model.scaling_raw().contiguous()));
            source.push_back(describe(4, model.rotation_raw().contiguous()));
            source.push_back(describe(5, model.opacity_raw().contiguous()));
            if (model.deleted().is_valid())
                source.push_back(describe(6, model.deleted().contiguous()));
            if (model._densification_info.is_valid())
                source.push_back(describe(7, model._densification_info.contiguous()));
            const auto source_ready = std::chrono::steady_clock::now();
            const auto descriptor_bytes = RAW_DESCRIPTOR_BYTES * source.size();
            const auto range_bytes = 8ull + 16ull * model.frozen_ranges().size();
            const auto data_start = RAW_HEADER_BYTES + descriptor_bytes + range_bytes;
            std::uint64_t data_bytes = 0;
            for (const auto& item : source)
                data_bytes += item.bytes;
            std::vector<std::byte> result(
                static_cast<std::size_t>(data_start + data_bytes));
            auto bytes = std::span<std::byte>(result);
            std::copy(RAW_MAGIC.begin(), RAW_MAGIC.end(), bytes.begin());
            put_u16(bytes, 8, 2);
            put_u16(bytes, 10, 0);
            put_u32(bytes, 12, static_cast<std::uint32_t>(model.get_active_sh_degree()));
            put_u32(bytes, 16, static_cast<std::uint32_t>(model.get_max_sh_degree()));
            std::uint32_t scene_bits = 0;
            const auto scene_scale = model.get_scene_scale();
            std::memcpy(&scene_bits, &scene_scale, sizeof(scene_bits));
            put_u32(bytes, 20, scene_bits);
            put_u32(bytes, 24, static_cast<std::uint32_t>(source.size()));
            put_u64(bytes, 32, data_start);
            put_u64(bytes, RAW_HEADER_BYTES + descriptor_bytes, model.frozen_ranges().size());
            for (std::size_t i = 0; i < model.frozen_ranges().size(); ++i) {
                const auto at = RAW_HEADER_BYTES + descriptor_bytes + 8 + i * 16;
                put_u64(bytes, at, model.frozen_ranges()[i].start);
                put_u64(bytes, at + 8, model.frozen_ranges()[i].count);
            }
            auto data_offset = static_cast<std::uint64_t>(data_start);
            for (std::size_t i = 0; i < source.size(); ++i) {
                bytes = std::span<std::byte>(result);
                const auto& item = source[i];
                const auto at = RAW_HEADER_BYTES + i * RAW_DESCRIPTOR_BYTES;
                put_u32(bytes, at, item.id);
                bytes[at + 4] = static_cast<std::byte>(item.tensor.dtype());
                bytes[at + 5] = static_cast<std::byte>(item.shape.size());
                for (std::size_t d = 0; d < item.shape.size(); ++d)
                    put_u64(bytes, at + 8 + d * 8, item.shape[d]);
                put_u64(bytes, at + 40, data_offset);
                put_u64(bytes, at + 48, item.bytes);
                data_offset += item.bytes;
            }

            constexpr std::size_t window_bytes = 64ull * 1024ull * 1024ull;
            const bool has_cuda_source = std::ranges::any_of(
                source, [](const SourceTensor& item) {
                    return item.tensor.device() == lfs::core::Device::CUDA;
                });
            if (has_cuda_source) {
                const auto status = cudaStreamCreateWithFlags(
                    &stream_guard.stream, cudaStreamNonBlocking);
                if (status != cudaSuccess) {
                    return splat_error(
                        lfs::ErrorCode::ResourceExhausted,
                        "The splat payload could not be serialized.",
                        std::format("CUDA transfer stream creation failed: {}",
                                    cudaGetErrorString(status)));
                }
            }
            struct StagingSlot {
                void* ptr = nullptr;
                cudaEvent_t ready = nullptr;
                std::byte* destination = nullptr;
                std::size_t count = 0;
                bool pending = false;
            };
            std::array<StagingSlot, 2> staging{};
            struct StagingGuard {
                std::array<StagingSlot, 2>& slots;
                const bool enabled;
                cudaStream_t stream;
                ~StagingGuard() {
                    if (enabled)
                        cudaStreamSynchronize(stream);
                    for (auto& slot : slots) {
                        if (slot.ready)
                            cudaEventDestroy(slot.ready);
                        if (slot.ptr)
                            lfs::core::PinnedMemoryAllocator::instance().deallocate(
                                slot.ptr);
                    }
                }
            } staging_guard{staging, has_cuda_source, stream_guard.stream};
            if (has_cuda_source) {
                for (auto& slot : staging) {
                    slot.ptr = lfs::core::PinnedMemoryAllocator::instance().allocate(
                        window_bytes);
                    if (!slot.ptr || cudaEventCreateWithFlags(
                                         &slot.ready, cudaEventDisableTiming) !=
                                         cudaSuccess) {
                        return splat_error(
                            lfs::ErrorCode::ResourceExhausted,
                            "The splat payload could not be serialized.",
                            "pinned staging allocation failed");
                    }
                    lfs::core::PinnedMemoryAllocator::instance().record_stream(
                        slot.ptr, stream_guard.stream);
                }
            }
            lfs::core::Tensor sh_scratch;
            if (std::ranges::any_of(source, &SourceTensor::sh_range)) {
                sh_scratch = lfs::core::Tensor::empty(
                    {window_bytes}, lfs::core::Device::CUDA,
                    lfs::core::DataType::UInt8, false);
            }
            const auto flush_slot = [&](StagingSlot& slot) -> lfs::Result<void> {
                if (!slot.pending)
                    return {};
                if (const auto status = cudaEventSynchronize(slot.ready);
                    status != cudaSuccess) {
                    return lfs::Result<void>::failure(splat_error(
                        lfs::core::cuda_status_to_error_code(status),
                        "The splat payload could not be serialized.",
                        std::format("CUDA transfer synchronization failed: {}",
                                    cudaGetErrorString(status))));
                }
                std::memcpy(slot.destination, slot.ptr, slot.count);
                slot.pending = false;
                return {};
            };
            data_offset = data_start;
            std::size_t staging_index = 0;
            for (const auto& item : source) {
                for (std::uint64_t offset = 0; offset < item.bytes;) {
                    auto count = static_cast<std::size_t>(std::min<std::uint64_t>(
                        window_bytes, item.bytes - offset));
                    if (item.sh_range && count % sizeof(float) != 0)
                        count -= count % sizeof(float);
                    auto* const destination =
                        result.data() + data_offset + offset;
                    if (item.tensor.device() == lfs::core::Device::CUDA) {
                        auto& slot = staging[staging_index++ % staging.size()];
                        if (auto flushed = flush_slot(slot); !flushed)
                            return std::move(flushed).error();
                        lfs::core::prepare_inputs_for_stream(
                            {&item.tensor}, stream_guard.stream);
                        const void* source_ptr = item.tensor.data_ptr();
                        if (item.sh_range) {
                            lfs::core::undo_reorder_sh_range_from_swizzled(
                                static_cast<const float*>(source_ptr),
                                static_cast<float*>(sh_scratch.data_ptr()),
                                offset / sizeof(float),
                                count / sizeof(float), item.shape[0],
                                static_cast<std::uint32_t>(item.shape[1]),
                                static_cast<std::uint32_t>(item.shape[1]),
                                stream_guard.stream);
                            source_ptr = sh_scratch.data_ptr();
                        } else {
                            source_ptr = static_cast<const std::byte*>(source_ptr) + offset;
                        }
                        const auto status = cudaMemcpyAsync(
                            slot.ptr, source_ptr, count,
                            cudaMemcpyDeviceToHost, stream_guard.stream);
                        if (status != cudaSuccess)
                            return splat_error(
                                lfs::core::cuda_status_to_error_code(status),
                                "The splat payload could not be serialized.",
                                std::format("CUDA tensor copy failed: {}",
                                            cudaGetErrorString(status)));
                        if (const auto recorded = cudaEventRecord(
                                slot.ready, stream_guard.stream);
                            recorded != cudaSuccess)
                            return splat_error(
                                lfs::core::cuda_status_to_error_code(recorded),
                                "The splat payload could not be serialized.",
                                std::format("CUDA transfer synchronization failed: {}",
                                            cudaGetErrorString(recorded)));
                        slot.destination = destination;
                        slot.count = count;
                        slot.pending = true;
                    } else {
                        std::memcpy(destination,
                                    static_cast<const std::byte*>(item.tensor.data_ptr()) + offset,
                                    count);
                    }
                    offset += count;
                }
                data_offset += item.bytes;
            }
            for (auto& slot : staging) {
                if (auto flushed = flush_slot(slot); !flushed)
                    return std::move(flushed).error();
            }
            const auto finished = std::chrono::steady_clock::now();
            LOG_DEBUG(
                "Project SPLT serialization stages: tensors={} bytes={} source_prepare={:.3f} ms manifest_memcpy={:.3f} ms total={:.3f} ms",
                source.size(), result.size(),
                std::chrono::duration<double, std::milli>(source_ready - started).count(),
                std::chrono::duration<double, std::milli>(finished - source_ready).count(),
                std::chrono::duration<double, std::milli>(finished - started).count());
            return from_lfsp(std::move(result));
        } catch (const std::bad_alloc& error) {
            return splat_error(
                lfs::ErrorCode::ResourceExhausted,
                "The splat payload could not be serialized.",
                std::format("SPLT allocation failed: {}", error.what()));
        } catch (const std::exception& error) {
            return splat_error(
                lfs::ErrorCode::Internal,
                "The splat payload could not be serialized.",
                std::format("SPLT serialization failed: {}", error.what()));
        }
    }

    lfs::Result<std::unique_ptr<lfs::core::SplatData>>
    SplatChapterPayload::hydrate(
        lfs::core::SplatTensorAllocator tensor_allocator) const {
        if (bytes_.size() >= RAW_MAGIC.size() &&
            std::equal(RAW_MAGIC.begin(), RAW_MAGIC.end(), bytes_.begin())) {
            return hydrate_raw(bytes_, std::move(tensor_allocator));
        }
        try {
            auto result = std::make_unique<lfs::core::SplatData>();
            const auto chars = std::span<const char>(
                reinterpret_cast<const char*>(bytes_.data()), bytes_.size());
            SpanStreambuf buffer(chars);
            std::istream stream(&buffer);
            result->deserialize(stream, std::move(tensor_allocator));
            if (!stream || stream.peek() != std::char_traits<char>::eof()) {
                return splat_error(
                    lfs::ErrorCode::DataLoss,
                    "The embedded splat payload has trailing or unread bytes.",
                    "LFSP deserializer did not consume the bounded SPLT payload exactly");
            }
            return result;
        } catch (const std::exception& error) {
            return splat_error(
                lfs::ErrorCode::DataLoss,
                "The embedded splat payload could not be restored.",
                std::format("LFSP deserialization failed: {}", error.what()));
        }
    }

    bool SplatChapterPayload::must_reference_external(
        const SplatSourceKind source_kind) noexcept {
        return source_kind == SplatSourceKind::LiveRad;
    }

} // namespace lfs::io::project
