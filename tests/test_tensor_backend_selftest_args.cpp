/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor_backend.hpp"
#include "io/argument_parser.hpp"

#include <gtest/gtest.h>
#include <iterator>
#include <span>
#include <string>
#include <variant>

namespace {
    using lfs::core::GpuBackend;
    using lfs::io::args::TensorBackendSelftestMode;

    TEST(TensorBackendSelftestArgs, SelectsEachBackendInBothArgumentForms) {
        struct Case {
            const char* name;
            GpuBackend backend;
        };
        for (const auto& test : {Case{"cuda", GpuBackend::CUDA}, Case{"CuDa", GpuBackend::CUDA},
                                 Case{"vulkan", GpuBackend::Vulkan}, Case{"VuLkAn", GpuBackend::Vulkan},
                                 Case{"metal", GpuBackend::Metal}, Case{"MeTaL", GpuBackend::Metal},
                                 Case{"hip", GpuBackend::HIP}, Case{"HiP", GpuBackend::HIP}}) {
            SCOPED_TRACE(test.name);
            const char* separate[] = {"LichtFeld-Studio", "--tensor-backend-selftest", test.name};
            const std::string flag = std::string("--tensor-backend-selftest=") + test.name;
            const char* joined[] = {"LichtFeld-Studio", flag.c_str()};
            for (const auto args : {std::span<const char* const>(separate), std::span<const char* const>(joined)}) {
                const auto parsed = lfs::io::args::parse_args(static_cast<int>(args.size()), args.data());
                ASSERT_TRUE(parsed) << parsed.error();
                ASSERT_TRUE(std::holds_alternative<TensorBackendSelftestMode>(*parsed));
                EXPECT_EQ(std::get<TensorBackendSelftestMode>(*parsed).backend, test.backend);
            }
        }
    }

    TEST(TensorBackendSelftestArgs, RejectsMissingBackendAndListsHip) {
        const char* args[] = {"LichtFeld-Studio", "--tensor-backend-selftest"};
        const auto parsed = lfs::io::args::parse_args(static_cast<int>(std::size(args)), args);
        ASSERT_FALSE(parsed);
        EXPECT_NE(parsed.error().find("--tensor-backend-selftest"), std::string::npos);
        EXPECT_NE(parsed.error().find("cuda|vulkan|metal|hip"), std::string::npos);
    }

    TEST(TensorBackendSelftestArgs, RejectsInvalidAndEmptyValuesInBothArgumentForms) {
        for (const auto* value : {"", "unknown", "hip-extra", "--version"}) {
            SCOPED_TRACE(value);
            const char* separate[] = {"LichtFeld-Studio", "--tensor-backend-selftest", value};
            const std::string flag = std::string("--tensor-backend-selftest=") + value;
            const char* joined[] = {"LichtFeld-Studio", flag.c_str()};
            for (const auto args : {std::span<const char* const>(separate), std::span<const char* const>(joined)}) {
                const auto parsed = lfs::io::args::parse_args(static_cast<int>(args.size()), args.data());
                ASSERT_FALSE(parsed);
                EXPECT_NE(parsed.error().find("backend '" + std::string(value) + "'"), std::string::npos);
                EXPECT_NE(parsed.error().find("cuda, vulkan, metal, hip"), std::string::npos);
            }
        }
    }

    TEST(TensorBackendSelftestArgs, VersionModeRemainsIndependent) {
        const char* args[] = {"LichtFeld-Studio", "--version"};
        const auto parsed = lfs::io::args::parse_args(static_cast<int>(std::size(args)), args);
        ASSERT_TRUE(parsed) << parsed.error();
        EXPECT_TRUE(std::holds_alternative<lfs::io::args::VersionMode>(*parsed));
    }

    TEST(TensorBackendSelftestArgs, UncompiledBackendsReturnUnavailableWithoutFallback) {
        for (const auto backend : lfs::core::kGpuBackends) {
            if (lfs::core::gpu_backend_compiled(backend)) {
                continue;
            }
            SCOPED_TRACE(lfs::core::gpu_backend_name(backend));
            const auto status = lfs::core::tensor_backend_selftest(backend);
            ASSERT_FALSE(status);
            EXPECT_EQ(status.error().code(), lfs::ErrorCode::Unavailable);
        }
    }
} // namespace
