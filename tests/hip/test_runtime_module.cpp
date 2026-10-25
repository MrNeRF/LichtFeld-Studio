/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "runtime_module.hpp"
#include <chrono>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include "runtime_test_directory.hpp"
using lfs::core::internal::HipRuntimeModule;

// Each invocation owns a freshly created directory. Never overwrite or
// recursively remove a pre-existing fixture directory.
class UnicodeModuleFixture {
public:
    explicit UnicodeModuleFixture(const std::filesystem::path& source) {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        for (unsigned attempt = 0; attempt < 64; ++attempt) {
            auto candidate = source.parent_path() / std::filesystem::path(u8"Unicode-\u00e9-\u6e2c\u8a66");
            candidate += "-" + std::to_string(stamp) + "-" + std::to_string(attempt);
            if (!std::filesystem::create_directory(candidate))
                continue;
            directory_ = candidate;
            path_ = directory_ / source.filename();
            std::error_code error;
            copied_ = std::filesystem::copy_file(source, path_, error);
            if (error || !copied_) {
                std::error_code cleanup_error;
                std::filesystem::remove(directory_, cleanup_error);
                throw std::runtime_error("Cannot copy Unicode module fixture: " + error.message());
            }
            return;
        }
        throw std::runtime_error("Cannot reserve a unique Unicode fixture directory");
    }
    UnicodeModuleFixture(const UnicodeModuleFixture&) = delete;
    UnicodeModuleFixture& operator=(const UnicodeModuleFixture&) = delete;
    ~UnicodeModuleFixture() {
        std::error_code error;
        if (copied_)
            std::filesystem::remove(path_, error);
        if (!directory_.empty())
            std::filesystem::remove(directory_, error);
    }
    const std::filesystem::path& path() const { return path_; }
    const std::filesystem::path& directory() const { return directory_; }

private:
    std::filesystem::path path_, directory_;
    bool copied_ = false;
};

std::string rejection(const std::filesystem::path& path) {
    try {
        HipRuntimeModule module(path);
    } catch (const std::runtime_error& error) {
        return error.what();
    }
    return {};
}

#ifdef _WIN32
int wmain(int argc, wchar_t** argv) {
#else
int main(int argc, char** argv) {
#endif
    try {
        if (argc != 3 && argc != 4)
            throw std::runtime_error("Expected module and fixture ABI (0 for real module)");
        RuntimeDirectory directory{argc == 4 ? std::filesystem::path(argv[3]) : std::filesystem::path{}};
        const int revision = std::stoi(argv[2]);
        bool rejected_relative = false;
        try {
            HipRuntimeModule invalid("relative-module");
        } catch (const std::invalid_argument&) {
            rejected_relative = true;
        }
        if (!rejected_relative)
            throw std::runtime_error("Relative module path was accepted");
        const auto path = std::filesystem::path(argv[1]);
        bool rejected_dependency = false;
        try {
            HipRuntimeModule invalid(path, "relative-startup-dependency");
        } catch (const std::invalid_argument&) {
            rejected_dependency = true;
        }
        if (!rejected_dependency)
            throw std::runtime_error("Relative startup dependency was accepted");
        if (revision) {
            const std::string expected = revision == LFS_HIP_RUNTIME_ABI_VERSION ? "Missing HIP runtime export: lfs_hip_device_count"
                                                                                 : "HIP runtime ABI mismatch: expected 6, got " + std::to_string(revision);
            std::unique_ptr<UnicodeModuleFixture> fixture;
            if (revision == 1)
                fixture = std::make_unique<UnicodeModuleFixture>(path);
            const auto actual = rejection(fixture ? fixture->path() : path);
            if (fixture) {
                const auto directory_path = fixture->directory();
                fixture.reset();
                if (std::filesystem::exists(directory_path))
                    throw std::runtime_error("Unicode module fixture cleanup failed");
            }
            if (actual != expected)
                throw std::runtime_error("Unexpected loader diagnostic: [" + actual + "] expected [" + expected + "]");
        } else {
            HipRuntimeModule module(path);
            if (module.api().runtime_abi() != LFS_HIP_RUNTIME_ABI_VERSION)
                throw std::runtime_error("Loaded wrong ABI");
#define LFS_HIP_RUNTIME_EXPORT(Name) \
    if (!module.api().Name)          \
        throw std::runtime_error("Unbound export " #Name);
#include "runtime_exports.def"
#undef LFS_HIP_RUNTIME_EXPORT
        }
        std::cout << "HIP module contract passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
