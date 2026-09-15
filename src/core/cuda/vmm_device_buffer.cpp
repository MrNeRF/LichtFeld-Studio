/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda/vmm_device_buffer.hpp"

#include "core/crash_handler.hpp"
#include "core/logger.hpp"
#include "core/source_site.hpp"

#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#include <format>
#include <limits>
#include <utility>
#include <vector>

namespace lfs::core {

    namespace {
        [[nodiscard]] std::string driver_error_text(const CUresult status) {
            const char* name = nullptr;
            const char* description = nullptr;
            (void)cuGetErrorName(status, &name);
            (void)cuGetErrorString(status, &description);
            return std::format("{} ({})", name ? name : "unknown CUDA error",
                               description ? description : "no description");
        }

        [[nodiscard]] Error vmm_error(const ErrorCode code, const char* label,
                                      const char* operation, const CUresult status,
                                      const std::size_t bytes = 0) {
            const std::string detail = std::format(
                "{}{} failed: {}", label ? std::format("label={} ", label) : "",
                operation, driver_error_text(status));
            SmallFields fields;
            if (bytes != 0) {
                fields.add("bytes", static_cast<std::uint64_t>(bytes));
            }
            return make_error(ErrorInit{
                .code = code,
                .domain = ErrorDomain::CUDA,
                .severity = Severity::Error,
                .retryability = code == ErrorCode::ResourceExhausted
                                    ? Retryability::Retryable
                                    : Retryability::NotRetryable,
                .user_message = std::format("VMM {} failed", operation),
                .detail = detail,
                .detection = LFS_SOURCE_SITE_CURRENT(),
                .fields = std::move(fields),
                .native = NativeError{
                    .domain = ErrorDomain::CUDA,
                    .code = static_cast<std::int64_t>(status),
                    .name = driver_error_text(status),
                },
            });
        }

        [[nodiscard]] Error host_error(const ErrorCode code, const char* label,
                                       const char* operation, const std::size_t bytes = 0) {
            SmallFields fields;
            if (bytes != 0) {
                fields.add("bytes", static_cast<std::uint64_t>(bytes));
            }
            return make_error(ErrorInit{
                .code = code,
                .domain = ErrorDomain::CUDA,
                .severity = Severity::Error,
                .retryability = code == ErrorCode::ResourceExhausted
                                    ? Retryability::Retryable
                                    : Retryability::NotRetryable,
                .user_message = std::format("VMM {} failed", operation),
                .detail = label ? std::format("label={} bytes={}", label, bytes)
                                : std::format("bytes={}", bytes),
                .detection = LFS_SOURCE_SITE_CURRENT(),
                .fields = std::move(fields),
            });
        }

        [[nodiscard]] std::size_t align_up(const std::size_t value,
                                           const std::size_t alignment) {
            if (value > std::numeric_limits<std::size_t>::max() - alignment + 1) {
                return 0;
            }
            return ((value + alignment - 1) / alignment) * alignment;
        }

        [[nodiscard]] std::size_t align_down(const std::size_t value,
                                             const std::size_t alignment) {
            return (value / alignment) * alignment;
        }

        [[nodiscard]] bool is_out_of_memory(const CUresult status) {
            return status == CUDA_ERROR_OUT_OF_MEMORY;
        }
    } // namespace

    VmmDeviceBuffer::VmmDeviceBuffer(VmmDeviceBuffer&& other) noexcept
        : base_(std::exchange(other.base_, 0)),
          reservation_bytes_(std::exchange(other.reservation_bytes_, 0)),
          committed_bytes_(std::exchange(other.committed_bytes_, 0)),
          granularity_bytes_(std::exchange(other.granularity_bytes_, kGranularityBytes)),
          device_(other.device_),
          label_(std::exchange(other.label_, nullptr)),
          chunks_(std::move(other.chunks_)) {
    }

    VmmDeviceBuffer& VmmDeviceBuffer::operator=(VmmDeviceBuffer&& other) noexcept {
        if (this != &other) {
            release();
            base_ = std::exchange(other.base_, 0);
            reservation_bytes_ = std::exchange(other.reservation_bytes_, 0);
            committed_bytes_ = std::exchange(other.committed_bytes_, 0);
            granularity_bytes_ = std::exchange(other.granularity_bytes_, kGranularityBytes);
            device_ = other.device_;
            label_ = std::exchange(other.label_, nullptr);
            chunks_ = std::move(other.chunks_);
        }
        return *this;
    }

    VmmDeviceBuffer::~VmmDeviceBuffer() {
        release();
    }

    Result<VmmDeviceBuffer> VmmDeviceBuffer::create(
        const std::size_t reservation_bytes, const char* label) {
        int device = 0;
        if (const cudaError_t status = cudaGetDevice(&device); status != cudaSuccess) {
            return make_error(ErrorInit{
                .code = ErrorCode::Unavailable,
                .domain = ErrorDomain::CUDA,
                .severity = Severity::Error,
                .user_message = "Could not query the current CUDA device for VMM",
                .detail = std::format("cudaGetDevice failed: {}", cudaGetErrorString(status)),
                .detection = LFS_SOURCE_SITE_CURRENT(),
            });
        }

        CUmemAllocationProp prop{};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = device;
        std::size_t granularity = 0;
        if (cuMemGetAllocationGranularity(
                &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM) != CUDA_SUCCESS ||
            granularity == 0) {
            granularity = kGranularityBytes;
        }
        const std::size_t reserved = align_up(reservation_bytes, granularity);
        if (reserved == 0) {
            return host_error(ErrorCode::InvalidArgument, label, "address reservation", reservation_bytes);
        }

        CUdeviceptr base = 0;
        if (const CUresult status = cuMemAddressReserve(
                &base, reserved, granularity, 0, 0);
            status != CUDA_SUCCESS) {
            return vmm_error(ErrorCode::ResourceExhausted, label,
                             "cuMemAddressReserve", status, reserved);
        }

        VmmDeviceBuffer buffer;
        buffer.base_ = base;
        buffer.reservation_bytes_ = reserved;
        buffer.granularity_bytes_ = granularity;
        buffer.device_ = device;
        buffer.label_ = label;
        return buffer;
    }

    Status VmmDeviceBuffer::commit(const std::size_t required_bytes) {
        const std::size_t required = align_up(required_bytes, granularity_bytes_);
        if (required == 0 && required_bytes != 0) {
            return Status::failure(host_error(ErrorCode::InvalidArgument, label_, "commit", required_bytes));
        }
        if (required <= committed_bytes_) {
            return {};
        }
        if (required > reservation_bytes_) {
            return Status::failure(host_error(ErrorCode::ResourceExhausted, label_, "commit exceeds reservation", required));
        }

        CUmemAllocationProp prop{};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = device_;

        constexpr std::size_t kMaxChunkBytes = 128u * 1024u * 1024u;
        const std::size_t max_chunk_bytes = std::max(
            granularity_bytes_, align_down(kMaxChunkBytes, granularity_bytes_));
        while (committed_bytes_ < required) {
            const std::size_t offset = committed_bytes_;
            const std::size_t bytes = std::min(required - offset, max_chunk_bytes);
            CUmemGenericAllocationHandle handle = 0;
            if (const CUresult status = cuMemCreate(&handle, bytes, &prop, 0);
                status != CUDA_SUCCESS) {
                return Status::failure(vmm_error(
                    is_out_of_memory(status) ? ErrorCode::ResourceExhausted : ErrorCode::Internal,
                    label_, "cuMemCreate", status, bytes));
            }

            if (const CUresult status = cuMemMap(base_ + offset, bytes, 0, handle, 0);
                status != CUDA_SUCCESS) {
                (void)cuMemRelease(handle);
                return Status::failure(vmm_error(ErrorCode::Internal, label_, "cuMemMap", status, bytes));
            }

            CUmemAccessDesc access{};
            access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            access.location.id = device_;
            access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            if (const CUresult status = cuMemSetAccess(base_ + offset, bytes, &access, 1);
                status != CUDA_SUCCESS) {
                (void)cuMemUnmap(base_ + offset, bytes);
                (void)cuMemRelease(handle);
                return Status::failure(
                    vmm_error(ErrorCode::Internal, label_, "cuMemSetAccess", status, bytes));
            }

            chunks_.push_back(Chunk{
                .handle = handle,
                .offset = offset,
                .bytes = bytes,
            });
            committed_bytes_ += bytes;
        }
        return {};
    }

    Status VmmDeviceBuffer::decommit_tail(const std::size_t new_prefix_bytes) {
        const std::size_t new_prefix = align_down(new_prefix_bytes, granularity_bytes_);
        if (new_prefix >= committed_bytes_) {
            return {};
        }

        while (!chunks_.empty() && chunks_.back().offset >= new_prefix) {
            Chunk& chunk = chunks_.back();
            if (chunk.mapped) {
                if (const CUresult status = cuMemUnmap(
                        base_ + chunk.offset, chunk.bytes);
                    status != CUDA_SUCCESS) {
                    return Status::failure(
                        vmm_error(ErrorCode::Internal, label_, "cuMemUnmap decommit chunk",
                                  status, chunk.bytes));
                }
                chunk.mapped = false;
            }
            chunk.bytes = 0;
            if (const CUresult status = cuMemRelease(chunk.handle); status != CUDA_SUCCESS) {
                return Status::failure(
                    vmm_error(ErrorCode::Internal, label_, "cuMemRelease decommit chunk",
                              status));
            }
            chunks_.pop_back();
        }

        committed_bytes_ = chunks_.empty() ? 0 : chunks_.back().offset + chunks_.back().bytes;
        return {};
    }

    void VmmDeviceBuffer::release() noexcept {
        if (gpu_process_teardown_started()) {
            chunks_.clear();
            base_ = 0;
            reservation_bytes_ = 0;
            committed_bytes_ = 0;
            granularity_bytes_ = kGranularityBytes;
            label_ = nullptr;
            return;
        }
        if (base_ != 0) {
            if (const cudaError_t status = cudaDeviceSynchronize(); status != cudaSuccess) {
                LOG_ERROR("VMM CUDA cleanup device synchronization failed: {}",
                          cudaGetErrorString(status));
            }
        }
        for (const auto& chunk : chunks_) {
            if (chunk.mapped) {
                if (const CUresult status = cuMemUnmap(base_ + chunk.offset, chunk.bytes);
                    status != CUDA_SUCCESS) {
                    LOG_ERROR("VMM CUDA cleanup cuMemUnmap failed: {}",
                              driver_error_text(status));
                }
            }
            if (const CUresult status = cuMemRelease(chunk.handle); status != CUDA_SUCCESS) {
                LOG_ERROR("VMM CUDA cleanup cuMemRelease failed: {}",
                          driver_error_text(status));
            }
        }
        chunks_.clear();
        if (base_ != 0) {
            if (const CUresult status = cuMemAddressFree(base_, reservation_bytes_);
                status != CUDA_SUCCESS) {
                LOG_ERROR("VMM CUDA cleanup cuMemAddressFree failed: {}",
                          driver_error_text(status));
            }
        }
        base_ = 0;
        reservation_bytes_ = 0;
        committed_bytes_ = 0;
        granularity_bytes_ = kGranularityBytes;
        label_ = nullptr;
    }

} // namespace lfs::core
