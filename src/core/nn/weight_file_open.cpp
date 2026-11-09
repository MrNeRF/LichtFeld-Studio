/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/nn/weight_file.hpp"
#include "core/nn/weight_source.hpp"

#include "core/native_fs.hpp"
#include "core/path_utils.hpp"

#include <algorithm>
#include <exception>
#include <format>
#include <mutex>
#include <utility>

namespace lfs::core::nn {
    namespace {

        constexpr std::uint32_t kMagic = 0x3157464c; // "LFW1" little-endian
        constexpr std::uint64_t kAlign = 64;

        struct SourceUse {
            std::filesystem::path path;
            std::weak_ptr<const void> token;
            std::weak_ptr<std::timed_mutex> operation;
        };
        std::mutex source_mutex;
        std::vector<SourceUse> source_uses;

        // Called under source_mutex. Path equality also covers a cache file which
        // has not been created yet; equivalent recognizes existing path aliases.
        SourceUse& source_use(const std::filesystem::path& path) {
            std::erase_if(source_uses, [](const SourceUse& use) {
                return use.token.expired() && use.operation.expired();
            });
            const auto absolute = core::native_fs::absolute(path).lexically_normal();
            const auto exact = std::find_if(source_uses.begin(), source_uses.end(), [&](const SourceUse& use) { return absolute == use.path; });
            if (exact != source_uses.end())
                return *exact;
            for (auto& use : source_uses) {
                std::error_code error;
                if (core::native_fs::equivalent(path, use.path, error) && !error)
                    return use;
            }
            return source_uses.emplace_back(absolute);
        }

        std::shared_ptr<const void> retain_source_locked(const std::filesystem::path& path) {
            auto& use = source_use(path);
            auto token = use.token.lock();
            if (!token) {
                token = std::make_shared<const int>(0);
                use.token = token;
            }
            return token;
        }

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

    WeightSource::WeightSource(std::filesystem::path value)
        : path(std::move(value)), source(WeightFile::retain_source(path)) {}

    WeightFile& WeightFile::operator=(WeightFile&& other) noexcept {
        if (this != &other) {
            mapped_ = std::move(other.mapped_);
            source_ = std::move(other.source_);
            header_ = std::move(other.header_);
            meta_ = std::move(other.meta_);
            tensors_ = std::move(other.tensors_);
            payload_offset_ = other.payload_offset_;
        }
        return *this;
    }

    bool WeightFile::source_in_use(const std::filesystem::path& path) {
        std::lock_guard lock(source_mutex);
        return !source_use(path).token.expired();
    }

    std::shared_ptr<std::timed_mutex> WeightFile::source_operation(const std::filesystem::path& path) {
        std::lock_guard lock(source_mutex);
        auto& use = source_use(path);
        auto operation = use.operation.lock();
        if (!operation) {
            operation = std::make_shared<std::timed_mutex>();
            use.operation = operation;
        }
        return operation;
    }

    std::shared_ptr<const void> WeightFile::retain_source(const std::filesystem::path& path) {
        std::lock_guard lock(source_mutex);
        return retain_source_locked(path);
    }

    lfs::Result<bool> WeightFile::remove_unused_source(const std::filesystem::path& path) {
        std::lock_guard lock(source_mutex);
        if (!source_use(path).token.expired()) {
            return lfs::make_error({
                .code = lfs::ErrorCode::FailedPrecondition,
                .domain = lfs::ErrorDomain::IO,
                .user_message = "The model is in use",
                .detail = core::path_to_utf8(path),
                .detection = LFS_SOURCE_SITE_CURRENT(),
            });
        }
        std::error_code error;
        const auto status = core::native_fs::symlink_status(path, error);
        if (error == std::errc::no_such_file_or_directory || status.type() == std::filesystem::file_type::not_found)
            return false;
        if (!error && status.type() != std::filesystem::file_type::regular && status.type() != std::filesystem::file_type::symlink) {
            return lfs::make_error({
                .code = lfs::ErrorCode::Unsupported,
                .domain = lfs::ErrorDomain::IO,
                .user_message = "The cache entry is not a weight file",
                .detail = core::path_to_utf8(path),
                .detection = LFS_SOURCE_SITE_CURRENT(),
            });
        }
        const bool removed = !error && core::native_fs::remove(path, error);
        if (error) {
            return lfs::make_error({
                .code = error == std::errc::permission_denied || error == std::errc::operation_not_permitted ? lfs::ErrorCode::PermissionDenied : lfs::ErrorCode::Unavailable,
                .domain = lfs::ErrorDomain::IO,
                .user_message = "Failed to remove model weights",
                .detail = error.message(),
                .detection = LFS_SOURCE_SITE_CURRENT(),
                .native = lfs::NativeError{lfs::ErrorDomain::IO, error.value(), error.category().name()},
            });
        }
        return removed;
    }

    lfs::Result<WeightFile> WeightFile::open(const std::filesystem::path& path) {
        WeightFile file;
        {
            std::lock_guard lock(source_mutex);
            if (!file.mapped_.open(path, MappedFile::Advice::Random)) {
                return io_error(lfs::ErrorCode::NotFound,
                                std::format("could not map {}", lfs::core::path_to_utf8(path)));
            }
            file.source_ = retain_source_locked(path);
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
