/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor/backend/facade_trace.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

namespace lfs::testing {

    // Writes one JSON line per test with the facade entries it executed, so the
    // manifest generator can map cases to launcher rows by measurement instead of
    // by method-name inference. Also detects a sticky CUDA error left behind by a
    // test and stops the run naming that test, because every later case in the
    // process would otherwise fail for an unrelated reason.
    class FacadeTraceListener final : public ::testing::EmptyTestEventListener {
    public:
        explicit FacadeTraceListener(std::string path)
            : output_(path, std::ios::app) {
            lfs::core::internal::facade_trace_enable_for_testing(true);
        }

        void OnTestStart(const ::testing::TestInfo&) override {
            lfs::core::internal::facade_trace_reset_for_testing();
        }

        void OnTestEnd(const ::testing::TestInfo& info) override {
            const auto counts =
                lfs::core::internal::facade_trace_snapshot_for_testing();
            output_ << "{\"test\": \"" << info.test_suite_name() << '.' << info.name()
                    << "\", \"entries\": {";
            bool first = true;
            for (size_t index = 0; index < counts.size(); ++index) {
                if (counts[index] == 0)
                    continue;
                output_ << (first ? "" : ", ") << '"'
                        << lfs::core::internal::facade_entry_name(
                               static_cast<lfs::core::internal::FacadeEntry>(index))
                        << "\": " << counts[index];
                first = false;
            }
            output_ << '}';
            const cudaError_t sticky = cudaGetLastError();
            if (sticky != cudaSuccess) {
                output_ << ", \"cuda_sticky\": \"" << cudaGetErrorName(sticky) << '"';
            }
            output_ << "}\n";
            output_.flush();
            if (sticky != cudaSuccess && cudaDeviceSynchronize() != cudaSuccess) {
                std::fprintf(stderr,
                             "lichtfeld_tests: %s.%s left the CUDA context in sticky "
                             "error state %s; "
                             "stopping the run so later cases are not blamed for it\n",
                             info.test_suite_name(), info.name(),
                             cudaGetErrorName(sticky));
                std::fflush(stderr);
                std::_Exit(3);
            }
        }

    private:
        std::ofstream output_;
    };

} // namespace lfs::testing
