/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/crash_handler.hpp"
#include "core/tensor_backend.hpp"
#include <cstddef>
#include <gtest/gtest.h>
#include <iostream>
#include <string>
#include <string_view>

namespace {
    class ContractExecution : public testing::EmptyTestEventListener {
    public:
        explicit ContractExecution(const std::string& required_backend)
            : required_suffix_(required_backend.empty() ? "" : "/" + required_backend) {}

        void OnTestEnd(const testing::TestInfo& test) override {
            if (test.result()->Skipped())
                return;
            ++executed;
            if (!required_suffix_.empty() && std::string_view(test.name()).ends_with(required_suffix_))
                ++required_executed;
        }

        size_t executed = 0;
        size_t required_executed = 0;

    private:
        std::string required_suffix_;
    };
} // namespace

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    int result = 0;
    std::string required_backend;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        constexpr std::string_view prefix = "--require-tensor-backend=";
        if (!arg.starts_with(prefix)) {
            std::cerr << "Unknown argument: " << arg << '\n';
            result = 2;
            break;
        }
        const auto name = arg.substr(prefix.size());
        if (!required_backend.empty() && required_backend != name) {
            std::cerr << "Only one tensor backend can be required\n";
            result = 2;
            break;
        }
        bool compiled = false;
        for (const auto backend : lfs::core::kCompiledGpuBackends) {
            if (name == lfs::core::gpu_backend_name(backend)) {
                compiled = true;
                // Listing must not create or probe GPU contexts.
                if (!GTEST_FLAG_GET(list_tests) && !lfs::core::gpu_backend_available(backend)) {
                    std::cerr << "Required tensor backend unavailable: " << name << '\n';
                    result = 2;
                }
                break;
            }
        }
        if (!compiled) {
            std::cerr << "Required tensor backend not compiled: " << name << '\n';
            result = 2;
        }
        if (result != 0)
            break;
        required_backend = name;
    }
    if (result == 0) {
        // GoogleTest owns the listener. Count actual, non-skipped cases across
        // repeats; selected-test counts also include a zero-repeat invocation.
        auto* execution = new ContractExecution(required_backend);
        testing::UnitTest::GetInstance()->listeners().Append(execution);
        result = RUN_ALL_TESTS();
        if (result == 0 && !GTEST_FLAG_GET(list_tests)) {
            if (execution->executed == 0) {
                std::cerr << "No tensor contracts executed\n";
                result = 2;
            } else if (!required_backend.empty() && execution->required_executed == 0) {
                std::cerr << "No tensor contracts executed for required backend: " << required_backend << '\n';
                result = 2;
            }
        }
    }
    // Discovery creates no device state. The process-wide teardown may query
    // cold backends, so keep it out of the listing-only path as well.
    if (!GTEST_FLAG_GET(list_tests))
        lfs::core::teardown_gpu_before_exit();
    lfs::core::flush_and_exit(result);
}
