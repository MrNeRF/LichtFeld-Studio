/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/logger.hpp"
#include "core/user_paths.hpp"
#include "internal/expression_runtime.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <stdexcept>
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace lfs::core::internal {
    namespace {
        constexpr size_t memory_limit = 64 * 1024 * 1024;
        constexpr size_t entry_limit = 128;
        constexpr size_t disk_limit = 256 * 1024 * 1024;
        constexpr size_t artifact_limit = 16 * 1024 * 1024;
        std::mutex disk_mutex;

        uint64_t digest(const std::string_view bytes) {
            uint64_t hash = 14695981039346656037ull;
            for (const unsigned char byte : bytes)
                hash = (hash ^ byte) * 1099511628211ull;
            return hash;
        }

        std::filesystem::path artifact_path(const std::string& key) {
            const auto paths = UserPaths::resolve();
            if (!paths)
                return {};
            return paths->cacheDir() / "tensor-expressions" / std::format("{:016x}.kernel", digest(key));
        }

        std::vector<char> read_artifact(const std::filesystem::path& path, const std::string& key) {
            std::ifstream file(path, std::ios::binary | std::ios::ate);
            if (!file)
                return {};
            const auto size = file.tellg();
            const size_t prefix = 24 + key.size();
            if (size < std::streamoff(prefix) || size > std::streamoff(artifact_limit))
                return {};
            file.seekg(0);
            uint64_t header[3]{};
            file.read(reinterpret_cast<char*>(header), sizeof(header));
            if (!file || header[0] != 0x314e524b53464cull || header[1] != key.size())
                return {};
            std::string stored_key(key.size(), '\0');
            file.read(stored_key.data(), stored_key.size());
            if (stored_key != key)
                return {};
            std::vector<char> data(static_cast<size_t>(size) - prefix);
            file.read(data.data(), data.size());
            if (!file || digest({data.data(), data.size()}) != header[2])
                return {};
            std::error_code error;
            std::filesystem::last_write_time(path, std::filesystem::file_time_type::clock::now(), error);
            return data;
        }

        void write_artifact(const std::filesystem::path& path, const std::string& key,
                            const std::span<const char> data) {
            if (path.empty() || data.size() + key.size() + 24 > artifact_limit)
                return;
            std::lock_guard lock(disk_mutex);
            std::error_code error;
            std::filesystem::create_directories(path.parent_path(), error);
            if (error) {
                LOG_WARN("Expression cache directory: {}", error.message());
                return;
            }
            const uint64_t header[]{0x314e524b53464cull, key.size(), digest({data.data(), data.size()})};
            std::string payload(reinterpret_cast<const char*>(header), sizeof(header));
            payload += key;
            payload.append(data.data(), data.size());
            if (const auto written = writeTextFileAtomically(path, payload); !written) {
                LOG_WARN("Expression cache write failed: {}", path.string());
                return;
            }
            struct File {
                std::filesystem::path path;
                std::filesystem::file_time_type time;
                uintmax_t bytes;
            };
            std::vector<File> files;
            uintmax_t total = 0;
            for (std::filesystem::directory_iterator it(path.parent_path(), error), end;
                 !error && it != end; it.increment(error)) {
                if (it->path().extension() != ".kernel" || !it->is_regular_file(error))
                    continue;
                const auto bytes = it->file_size(error);
                const auto time = it->last_write_time(error);
                if (error)
                    break;
                files.push_back({it->path(), time, bytes});
                total += bytes;
            }
            std::ranges::sort(files, {}, &File::time);
            size_t remaining = files.size();
            for (const auto& file : files) {
                if (total <= disk_limit && remaining <= 1024)
                    break;
                if (std::filesystem::remove(file.path, error)) {
                    total -= file.bytes;
                    --remaining;
                }
            }
        }
    } // namespace

    std::shared_ptr<CompiledExpression> ExpressionCache::get(
        const std::string& key, const Compile& compile, const Load& load) {
        // Compilation and publication share the lock: a key has one compiler.
        std::lock_guard lock(mutex_);
        if (const auto found = entries_.find(key); found != entries_.end()) {
            lru_.splice(lru_.begin(), lru_, found->second.position);
            ++stats_.hits;
            return found->second.kernel;
        }
        const auto path = artifact_path(key);
        auto artifact = read_artifact(path, key);
        std::shared_ptr<CompiledExpression> kernel;
        if (!artifact.empty()) {
            kernel = load(artifact);
            ++stats_.disk_hits;
        }
        if (!kernel) {
            const auto begin = std::chrono::steady_clock::now();
            artifact = compile();
            stats_.compile_ms += std::chrono::duration<double, std::milli>(
                                     std::chrono::steady_clock::now() - begin)
                                     .count();
            ++stats_.compilations;
            kernel = load(artifact);
            write_artifact(path, key, artifact);
        }
        ++stats_.loads;
        kernel->bytes = artifact.size();
        while (!lru_.empty() && (entries_.size() >= entry_limit ||
                                 stats_.bytes + kernel->bytes > memory_limit)) {
            const auto found = entries_.find(lru_.back());
            stats_.bytes -= found->second.kernel->bytes;
            entries_.erase(found);
            lru_.pop_back();
            ++stats_.evictions;
        }
        if (kernel->bytes <= memory_limit) {
            lru_.push_front(key);
            entries_.emplace(key, Entry{kernel, lru_.begin()});
            stats_.bytes += kernel->bytes;
            stats_.entries = entries_.size();
        }
        return kernel;
    }

    ExpressionCacheStats ExpressionCache::stats() const {
        std::lock_guard lock(mutex_);
        return stats_;
    }

    void ExpressionCache::clear() {
        std::lock_guard lock(mutex_);
        entries_.clear();
        lru_.clear();
        stats_.entries = stats_.bytes = 0;
    }

    RuntimeLibrary::RuntimeLibrary(const char* name) {
#ifdef _WIN32
        handle_ = reinterpret_cast<void*>(LoadLibraryA(name));
#else
        handle_ = dlopen(name, RTLD_NOW | RTLD_LOCAL);
#endif
        if (!handle_)
            throw std::runtime_error(std::string("Expression driver library is unavailable: ") + name);
    }

    RuntimeLibrary::~RuntimeLibrary() {
#ifdef _WIN32
        FreeLibrary(static_cast<HMODULE>(handle_));
#else
        dlclose(handle_);
#endif
    }

    void* RuntimeLibrary::symbol(const char* name) const {
#ifdef _WIN32
        void* result = reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(handle_), name));
#else
        void* result = dlsym(handle_, name);
#endif
        if (!result)
            throw std::runtime_error(std::string("Expression driver symbol is unavailable: ") + name);
        return result;
    }
} // namespace lfs::core::internal
