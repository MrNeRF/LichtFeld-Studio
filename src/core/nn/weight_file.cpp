/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/nn/weight_file.hpp"

#include "core/cuda_error.hpp"

#include <cstring>
#include <format>

namespace lfs::core::nn {
    namespace {

        lfs::Error io_error(const lfs::ErrorCode code, std::string detail) {
            return lfs::make_error({
                .code = code,
                .domain = lfs::ErrorDomain::IO,
                .user_message = "Failed to read a LichtFeld weight file",
                .detail = std::move(detail),
                .detection = LFS_SOURCE_SITE_CURRENT(),
            });
        }

    } // namespace

    bool WeightFile::contains(const std::string_view name) const {
        return tensors_.find(std::string(name)) != tensors_.end();
    }

    std::vector<std::string> WeightFile::names() const {
        std::vector<std::string> out;
        out.reserve(tensors_.size());
        for (const auto& [name, _] : tensors_) {
            out.push_back(name);
        }
        return out;
    }

    const WeightFile::TensorInfo* WeightFile::info(const std::string_view name) const {
        const auto it = tensors_.find(std::string(name));
        if (it == tensors_.end()) {
            return nullptr;
        }
        return &it->second;
    }

    lfs::Result<Tensor> WeightFile::load(const std::string_view name, const Device device,
                                         const std::optional<DataType> cast) const {
        const TensorInfo* found = info(name);
        if (found == nullptr) {
            return io_error(lfs::ErrorCode::NotFound, std::format("tensor {} is not in the file", name));
        }
        const DataType dest_dtype = cast ? *cast : found->dtype;
        const void* src = mapped_.data() + payload_offset_ + found->offset;
        if (device == Device::CPU) {
            auto cpu = Tensor::empty(found->shape, Device::CPU, found->dtype, false);
            if (found->length > 0) {
                std::memcpy(cpu.data_ptr(), src, static_cast<std::size_t>(found->length));
            }
            if (dest_dtype != cpu.dtype()) {
                return cpu.to(dest_dtype);
            }
            return cpu;
        }
        if (dest_dtype == found->dtype) {
            auto gpu = Tensor::empty(found->shape, Device::CUDA, dest_dtype);
            if (found->length > 0) {
                LFS_CUDA_CHECK(cudaMemcpyAsync(gpu.data_ptr(), src,
                                               static_cast<std::size_t>(found->length),
                                               cudaMemcpyHostToDevice, gpu.stream()));
            }
            return gpu;
        }
        auto tmp = Tensor::empty(found->shape, Device::CUDA, found->dtype);
        if (found->length > 0) {
            LFS_CUDA_CHECK(cudaMemcpyAsync(tmp.data_ptr(), src, static_cast<std::size_t>(found->length),
                                           cudaMemcpyHostToDevice, tmp.stream()));
        }
        return tmp.to(dest_dtype);
    }

    lfs::Result<std::unordered_map<std::string, Tensor>>
    WeightFile::load_all(const Device device, const std::optional<DataType> cast) const {
        std::unordered_map<std::string, Tensor> out;
        out.reserve(tensors_.size());
        for (const auto& [name, _] : tensors_) {
            auto tensor = load(name, device, cast);
            if (!tensor) {
                return std::move(tensor.error());
            }
            out.emplace(name, std::move(*tensor));
        }
        if (device == Device::CUDA) {
            LFS_CUDA_CHECK(cudaDeviceSynchronize());
        }
        return out;
    }

} // namespace lfs::core::nn
