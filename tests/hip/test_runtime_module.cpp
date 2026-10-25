/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "runtime_module.hpp"
#include <iostream>
#include <stdexcept>
#include <string>

#include "runtime_test_directory.hpp"
using lfs::core::internal::HipRuntimeModule;

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
        if (revision) {
            const std::string expected = revision == 3 ? "Missing HIP runtime export: lfs_hip_device_count"
                                                       : "HIP runtime ABI mismatch: expected 3, got " + std::to_string(revision);
            auto checked_path = path;
            if (revision == 1) {
                checked_path = path.parent_path() / std::filesystem::path(u8"Unicode-\u00e9-\u6e2c\u8a66") / path.filename();
                std::filesystem::create_directories(checked_path.parent_path());
                std::filesystem::copy_file(path, checked_path, std::filesystem::copy_options::overwrite_existing);
            }
            const auto actual = rejection(checked_path);
            if (revision == 1)
                std::filesystem::remove(checked_path);
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
