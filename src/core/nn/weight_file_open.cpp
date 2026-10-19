/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/nn/weight_file.hpp"

#include "core/path_utils.hpp"

#include <exception>
#include <format>
#include <utility>

namespace lfs::core::nn {
    namespace {

        constexpr std::uint32_t kMagic = 0x3157464c; // "LFW1" little-endian
        constexpr std::uint64_t kAlign = 64;

        lfs::Error io_error(const lfs::ErrorCode code, std::string detail) {
            return lfs::make_error({
                .code = code,
                .domain = lfs::ErrorDomain::IO,
                .user_message = "Failed to read a LichtFeld weight file",
                .detail = std::move(detail),
                .detection = LFS_SOURCE_SITE_CURRENT(),
            });
        }

        DataType dtype_from_name(const std::string_view name) {
            if (name == "float32" || name == "float" || name == "f32") {
                return DataType::Float32;
            }
            if (name == "float16" || name == "half" || name == "f16") {
                return DataType::Float16;
            }
            if (name == "int32" || name == "i32") {
                return DataType::Int32;
            }
            if (name == "int64" || name == "i64") {
                return DataType::Int64;
            }
            if (name == "uint8" || name == "u8") {
                return DataType::UInt8;
            }
            if (name == "bool") {
                return DataType::Bool;
            }
            return DataType::Float32;
        }

        bool known_dtype_name(const std::string_view name) {
            return name == "float32" || name == "float" || name == "f32" || name == "float16" ||
                   name == "half" || name == "f16" || name == "int32" || name == "i32" ||
                   name == "int64" || name == "i64" || name == "uint8" || name == "u8" ||
                   name == "bool";
        }

        std::uint32_t read_u32_le(const std::uint8_t* p) {
            return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8) |
                   (static_cast<std::uint32_t>(p[2]) << 16) | (static_cast<std::uint32_t>(p[3]) << 24);
        }

        std::uint64_t align_up(const std::uint64_t value, const std::uint64_t align) {
            return (value + align - 1) / align * align;
        }

    } // namespace

    lfs::Result<WeightFile> WeightFile::open(const std::filesystem::path& path) {
        WeightFile file;
        if (!file.mapped_.open(path, MappedFile::Advice::Random)) {
            return io_error(lfs::ErrorCode::NotFound,
                            std::format("could not map {}", lfs::core::path_to_utf8(path)));
        }
        if (file.mapped_.size() < 8) {
            return io_error(lfs::ErrorCode::DataLoss, "file is shorter than the LFW1 header");
        }
        const std::uint8_t* data = file.mapped_.data();
        if (read_u32_le(data) != kMagic) {
            return io_error(lfs::ErrorCode::DataLoss, "magic is not LFW1");
        }
        const std::uint32_t header_len = read_u32_le(data + 4);
        if (static_cast<std::uint64_t>(8) + header_len > file.mapped_.size()) {
            return io_error(lfs::ErrorCode::DataLoss, "JSON header overruns the file");
        }
        try {
            file.header_ = nlohmann::json::parse(data + 8, data + 8 + header_len);
        } catch (const std::exception& ex) {
            return lfs::make_error({
                .code = lfs::ErrorCode::DataLoss,
                .domain = lfs::ErrorDomain::IO,
                .user_message = "Failed to read a LichtFeld weight file",
                .detail = std::format("JSON header parse failed: {}", ex.what()),
                .detection = LFS_SOURCE_SITE_CURRENT(),
            });
        }
        if (!file.header_.is_object() || !file.header_.contains("tensors") ||
            !file.header_["tensors"].is_object()) {
            return io_error(lfs::ErrorCode::DataLoss, "header is missing a tensors object");
        }
        file.meta_ = file.header_.value("meta", nlohmann::json::object());
        file.payload_offset_ = align_up(8 + static_cast<std::uint64_t>(header_len), kAlign);
        if (file.payload_offset_ > file.mapped_.size()) {
            return io_error(lfs::ErrorCode::DataLoss, "payload offset overruns the file");
        }

        for (auto it = file.header_["tensors"].begin(); it != file.header_["tensors"].end(); ++it) {
            const auto& entry = it.value();
            if (!entry.is_object() || !entry.contains("dtype") || !entry.contains("shape") ||
                !entry.contains("offset") || !entry.contains("length")) {
                return io_error(lfs::ErrorCode::DataLoss,
                                std::format("tensor {} is missing required fields", it.key()));
            }
            const auto dtype_name = entry["dtype"].get<std::string>();
            if (!known_dtype_name(dtype_name)) {
                return io_error(lfs::ErrorCode::Unsupported,
                                std::format("tensor {} has unknown dtype {}", it.key(), dtype_name));
            }
            TensorInfo info;
            info.name = it.key();
            info.dtype = dtype_from_name(dtype_name);
            std::vector<std::size_t> shape;
            for (const auto& dim : entry["shape"]) {
                shape.push_back(dim.get<std::size_t>());
            }
            info.shape = TensorShape(shape);
            info.offset = entry["offset"].get<std::uint64_t>();
            info.length = entry["length"].get<std::uint64_t>();
            if (info.offset % kAlign != 0) {
                return io_error(lfs::ErrorCode::DataLoss,
                                std::format("tensor {} offset is not 64-byte aligned", info.name));
            }
            const std::uint64_t begin = file.payload_offset_ + info.offset;
            const std::uint64_t end = begin + info.length;
            if (end > file.mapped_.size()) {
                return io_error(lfs::ErrorCode::DataLoss,
                                std::format("tensor {} blob overruns the file", info.name));
            }
            const std::uint64_t expected = info.shape.elements() * dtype_size(info.dtype);
            if (info.length != expected) {
                return io_error(lfs::ErrorCode::DataLoss,
                                std::format("tensor {} length {} does not match shape ({} bytes)",
                                            info.name, info.length, expected));
            }
            file.tensors_.emplace(info.name, std::move(info));
        }
        return file;
    }

} // namespace lfs::core::nn
