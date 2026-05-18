#pragma once

#include "core/path_utils.hpp"
#include "io/error.hpp"
#include "io/exporter.hpp"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <format>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace lfs::io {

    enum class AtomicOutputTempName {
        AppendSuffix,
        PreserveExtension
    };

    inline std::filesystem::path make_atomic_temp_output_path(
        const std::filesystem::path& output_path,
        AtomicOutputTempName name_style = AtomicOutputTempName::AppendSuffix) {
        static std::atomic_uint64_t counter{0};

        const auto ticks = std::chrono::steady_clock::now().time_since_epoch().count();
        const auto unique_suffix = std::format(".{}.{}.tmp", ticks, counter.fetch_add(1, std::memory_order_relaxed));

        if (name_style == AtomicOutputTempName::PreserveExtension && output_path.has_extension()) {
            const auto temp_name = output_path.stem().string() + unique_suffix + output_path.extension().string();
            return output_path.parent_path() / temp_name;
        }

        return output_path.string() + unique_suffix;
    }

    inline Result<void> ensure_output_parent_directory(const std::filesystem::path& output_path) {
        if (output_path.parent_path().empty()) {
            return {};
        }

        std::error_code ec;
        std::filesystem::create_directories(output_path.parent_path(), ec);
        if (ec) {
            return std::unexpected(Error{
                ErrorCode::WRITE_FAILURE,
                std::format("Failed to create output directory '{}': {}", output_path.parent_path().string(), ec.message())});
        }

        return {};
    }

    inline Result<void> replace_atomic_output_file(const std::filesystem::path& temp_path,
                                                   const std::filesystem::path& output_path) {
#ifdef _WIN32
        const auto temp_w = temp_path.wstring();
        const auto output_w = output_path.wstring();
        if (!MoveFileExW(temp_w.c_str(), output_w.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
            return std::unexpected(Error{
                ErrorCode::WRITE_FAILURE,
                std::format("Failed to replace '{}' with temporary export '{}': Windows error {}",
                            output_path.string(), temp_path.string(), GetLastError())});
        }
#else
        std::error_code ec;
        std::filesystem::rename(temp_path, output_path, ec);
        if (ec) {
            return std::unexpected(Error{
                ErrorCode::WRITE_FAILURE,
                std::format("Failed to replace '{}' with temporary export '{}': {}",
                            output_path.string(), temp_path.string(), ec.message())});
        }
#endif

        return {};
    }

    class ScopedAtomicOutputFile {
    public:
        explicit ScopedAtomicOutputFile(
            std::filesystem::path output_path,
            AtomicOutputTempName name_style = AtomicOutputTempName::AppendSuffix)
            : output_path_(std::move(output_path)), temp_path_(make_atomic_temp_output_path(output_path_, name_style)) {}

        ScopedAtomicOutputFile(const ScopedAtomicOutputFile&) = delete;
        ScopedAtomicOutputFile& operator=(const ScopedAtomicOutputFile&) = delete;

        ScopedAtomicOutputFile(ScopedAtomicOutputFile&&) = delete;
        ScopedAtomicOutputFile& operator=(ScopedAtomicOutputFile&&) = delete;

        ~ScopedAtomicOutputFile() {
            if (!committed_) {
                std::error_code ec;
                std::filesystem::remove(temp_path_, ec);
            }
        }

        const std::filesystem::path& output_path() const { return output_path_; }
        const std::filesystem::path& temp_path() const { return temp_path_; }

        Result<void> commit() {
            auto result = replace_atomic_output_file(temp_path_, output_path_);
            if (!result) {
                return result;
            }

            committed_ = true;
            return {};
        }

    private:
        std::filesystem::path output_path_;
        std::filesystem::path temp_path_;
        bool committed_ = false;
    };

    inline bool report_export_progress(const ExportProgressCallback& callback, float progress, const std::string& stage) {
        if (!callback) {
            return true;
        }
        return callback(progress, stage);
    }

    inline ExportProgressCallback scale_export_progress(
        ExportProgressCallback callback,
        float start,
        float end) {
        if (!callback) {
            return {};
        }

        return [callback = std::move(callback), start, end](float progress, const std::string& stage) {
            const float scaled = start + (end - start) * progress;
            return callback(scaled, stage);
        };
    }

} // namespace lfs::io
