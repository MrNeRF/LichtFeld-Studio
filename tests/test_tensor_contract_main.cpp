/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/crash_handler.hpp"
#include "core/tensor_backend.hpp"
#include <gtest/gtest.h>
#include <iostream>
#include <string_view>

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    int result = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        constexpr std::string_view prefix = "--require-tensor-backend=";
        if (!arg.starts_with(prefix)) {
            std::cerr << "Unknown argument: " << arg << '\n';
            result = 2;
            break;
        }
        const auto name = arg.substr(prefix.size());
        bool available = false;
        for (const auto backend : lfs::core::kCompiledGpuBackends) {
            if (name == lfs::core::gpu_backend_name(backend)) {
                available = lfs::core::gpu_backend_available(backend);
                break;
            }
        }
        if (!available) {
            std::cerr << "Required tensor backend unavailable or not compiled: " << name << '\n';
            result = 2;
            break;
        }
    }
    if (result == 0) {
        result = RUN_ALL_TESTS();
        if (!GTEST_FLAG_GET(list_tests) && ::testing::UnitTest::GetInstance()->test_to_run_count() == 0) {
            std::cerr << "No tensor contracts selected\n";
            result = 2;
        }
    }
    lfs::core::teardown_gpu_before_exit();
    lfs::core::flush_and_exit(result);
}
