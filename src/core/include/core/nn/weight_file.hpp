/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/error.hpp"
#include "core/export.hpp"
#include "core/mapped_file.hpp"
#include "core/tensor.hpp"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace lfs::core::nn {

    // Little-endian `.lfw` weight archive:
    //   bytes 0-3:  magic "LFW1"
    //   bytes 4-7:  uint32 JSON header length
    //   bytes 8..:  UTF-8 JSON, then zero-pad so the payload starts at a
    //               64-byte file offset. Each tensor blob is 64-byte aligned
    //               relative to the payload start (and therefore to the file).
    // JSON shape:
    //   { "format": "lfw", "version": 1, "meta": {...},
    //     "tensors": { name: { "dtype", "shape", "offset", "length" } } }
    // `offset` is relative to the payload start.
    class LFS_CORE_API WeightFile {
    public:
        struct TensorInfo {
            std::string name;
            DataType dtype = DataType::Float32;
            TensorShape shape;
            std::uint64_t offset = 0;
            std::uint64_t length = 0;
        };

        WeightFile() = default;
        WeightFile(WeightFile&&) noexcept = default;
        WeightFile& operator=(WeightFile&&) noexcept = default;
        WeightFile(const WeightFile&) = delete;
        WeightFile& operator=(const WeightFile&) = delete;

        [[nodiscard]] static lfs::Result<WeightFile> open(const std::filesystem::path& path);

        [[nodiscard]] bool contains(std::string_view name) const;
        [[nodiscard]] std::vector<std::string> names() const;
        [[nodiscard]] const TensorInfo* info(std::string_view name) const;
        [[nodiscard]] const nlohmann::json& header() const { return header_; }
        [[nodiscard]] const nlohmann::json& meta() const { return meta_; }
        [[nodiscard]] std::size_t size_bytes() const { return mapped_.size(); }

        // Copies the named tensor onto `device`. Optional `cast` converts after
        // the copy (fp16 storage can be loaded as fp32, and vice versa).
        [[nodiscard]] lfs::Result<Tensor> load(std::string_view name, Device device,
                                               std::optional<DataType> cast = std::nullopt) const;

        [[nodiscard]] lfs::Result<std::unordered_map<std::string, Tensor>>
        load_all(Device device, std::optional<DataType> cast = std::nullopt) const;

    private:
        MappedFile mapped_;
        nlohmann::json header_;
        nlohmann::json meta_;
        std::unordered_map<std::string, TensorInfo> tensors_;
        std::uint64_t payload_offset_ = 0;
    };

} // namespace lfs::core::nn
