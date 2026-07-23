/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/error.hpp"
#include "core/error_envelope.hpp"
#include "core/error_latch.hpp"
#include "core/guarded_task.hpp"

#include <nlohmann/json.hpp>

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <optional>
#include <string>
#include <thread>

namespace {

    lfs::Error make_cuda_oom_error() {
        return lfs::make_error(lfs::ErrorInit{
                                   .code = lfs::ErrorCode::ResourceExhausted,
                                   .domain = lfs::ErrorDomain::CUDA,
                                   .operation_id = lfs::OperationId::generate(),
                                   .user_message = "Out of GPU memory during training",
                                   .detail = "raw what() with /home/user/secret path",
                                   .detection = LFS_SOURCE_SITE_CURRENT(),
                                   .fields = lfs::SmallFields{}
                                                 .add("path", std::string("/home/user/secret/scene.ply"))
                                                 .add("requested_bytes", std::uint64_t{1073741824})
                                                 .add("internal_ptr", std::string("0xdeadbeef")),
                                   .native = lfs::NativeError{lfs::ErrorDomain::CUDA, 2, "cudaErrorMemoryAllocation"},
                               })
            .with_context("train step", LFS_SOURCE_SITE_CURRENT());
    }

} // namespace

TEST(ErrorEnvelopeTest, RoundTripCarriesAllFieldsAndSanitizesPath) {
    const nlohmann::json envelope = lfs::core::to_wire_envelope(make_cuda_oom_error());

    EXPECT_EQ(envelope["code"], "ResourceExhausted");
    EXPECT_EQ(envelope["domain"], "CUDA");
    EXPECT_EQ(envelope["message"], "Out of GPU memory during training");
    EXPECT_FALSE(envelope["retryable"].get<bool>());
    EXPECT_GT(envelope["operation_id"].get<std::uint64_t>(), 0u);

    ASSERT_TRUE(envelope.contains("details"));
    const auto& details = envelope["details"];
    EXPECT_EQ(details["path"], "scene.ply");
    EXPECT_EQ(details["requested_bytes"].get<std::uint64_t>(), 1073741824u);
    EXPECT_FALSE(details.contains("internal_ptr"));
    EXPECT_EQ(details["native_name"], "cudaErrorMemoryAllocation");
    EXPECT_EQ(details["native_code"].get<std::int64_t>(), 2);
}

TEST(ErrorEnvelopeTest, NeverLeaksPathsDetailOrFrameSources) {
    const std::string body = lfs::core::to_wire_envelope(make_cuda_oom_error()).dump();

    EXPECT_EQ(body.find("/home/"), std::string::npos);
    EXPECT_EQ(body.find("raw what()"), std::string::npos);
    EXPECT_EQ(body.find("train step"), std::string::npos);
    EXPECT_NE(body.find("scene.ply"), std::string::npos);
}

TEST(ErrorEnvelopeTest, EmptyUserMessageFallsBackToGenericAndHidesDetail) {
    const lfs::Error error = [] {
        try {
            throw std::runtime_error("raw what() with /home/user/secret");
        } catch (...) {
            return lfs::core::detail::normalize_current_exception(lfs::core::TaskContext{
                .name = "test.boundary",
                .domain = lfs::ErrorDomain::TCP,
                .operation_id = lfs::OperationId::generate(),
                .site = LFS_SOURCE_SITE_CURRENT(),
            });
        }
    }();

    const nlohmann::json envelope = lfs::core::to_wire_envelope(error);
    EXPECT_EQ(envelope["message"], "Internal error");

    const std::string body = envelope.dump();
    EXPECT_EQ(body.find("raw what()"), std::string::npos);
    EXPECT_EQ(body.find("/home/"), std::string::npos);
}

TEST(ErrorEnvelopeTest, LegacyBridgePreservesMessageAndDefaultCode) {
    const lfs::Error error = lfs::make_legacy_error("No scene loaded", lfs::LegacyErrorContext{
                                                                           .code = lfs::ErrorCode::FailedPrecondition,
                                                                           .domain = lfs::ErrorDomain::MCP,
                                                                           .operation = "mcp.tool:test",
                                                                           .source = LFS_SOURCE_SITE_CURRENT(),
                                                                       });

    const nlohmann::json envelope = lfs::core::to_wire_envelope(error);
    EXPECT_EQ(envelope["code"], "FailedPrecondition");
    EXPECT_EQ(envelope["message"], "No scene loaded");
}

TEST(ErrorEnvelopeTest, InvalidUtf8MessageSerializesUnderStrictDump) {
    const std::string bad = "scene-\xff\xfe-\x80.ply could not be opened";
    const lfs::Error error = lfs::make_legacy_error(bad, lfs::LegacyErrorContext{
                                                             .code = lfs::ErrorCode::NotFound,
                                                             .domain = lfs::ErrorDomain::IO,
                                                             .operation = "io.load",
                                                             .source = LFS_SOURCE_SITE_CURRENT(),
                                                         });

    const nlohmann::json envelope = lfs::core::to_wire_envelope(error);
    const std::string message = envelope["message"].get<std::string>();

    EXPECT_EQ(message.find('\xff'), std::string::npos);
    EXPECT_NE(message.find("\xEF\xBF\xBD"), std::string::npos);
    EXPECT_NE(message.find("scene-"), std::string::npos);
    EXPECT_NO_THROW((void)envelope.dump());
}

TEST(ErrorEnvelopeTest, InvalidUtf8DetailPathSerializesUnderStrictDump) {
    const lfs::Error error = lfs::make_error(lfs::ErrorInit{
        .code = lfs::ErrorCode::NotFound,
        .domain = lfs::ErrorDomain::IO,
        .user_message = "Load failed",
        .detection = LFS_SOURCE_SITE_CURRENT(),
        .fields = lfs::SmallFields{}.add("path", std::string("/data/sc\xff\x80ene.ply")),
    });

    const nlohmann::json envelope = lfs::core::to_wire_envelope(error);
    ASSERT_TRUE(envelope.contains("details"));
    EXPECT_EQ(envelope["details"]["path"].get<std::string>().find('\xff'), std::string::npos);
    EXPECT_NO_THROW((void)envelope.dump());
}

TEST(ErrorEnvelopeTest, MaximalErrorStaysWithinSerializedCap) {
    lfs::SmallFields fields;
    fields.add("path", std::string(4096, 'p'));
    fields.add("command", std::string(4096, 'c'));
    fields.add("parameter", std::string(4096, 'a'));
    fields.add("format", std::string(4096, 'f'));
    fields.add("iteration", std::int64_t{123});
    fields.add("requested_bytes", std::uint64_t{1} << 40);
    fields.add("available_bytes", std::uint64_t{1} << 20);

    lfs::Error error = lfs::make_error(lfs::ErrorInit{
        .code = lfs::ErrorCode::Internal,
        .domain = lfs::ErrorDomain::Core,
        .user_message = std::string(8192, 'm'),
        .detection = LFS_SOURCE_SITE_CURRENT(),
        .fields = std::move(fields),
        .native = lfs::NativeError{lfs::ErrorDomain::Core, 1, std::string(4096, 'n')},
    });
    for (int i = 0; i < 20; ++i) {
        error = std::move(error).with_context("frame with fields", LFS_SOURCE_SITE_CURRENT(),
                                              lfs::SmallFields{}.add("path", std::string(512, 'x')));
    }
    for (int i = 0; i < 10; ++i) {
        error = std::move(error).with_suppressed(lfs::make_error(lfs::ErrorInit{
            .code = lfs::ErrorCode::Internal,
            .domain = lfs::ErrorDomain::Core,
            .detection = LFS_SOURCE_SITE_CURRENT(),
        }));
    }

    EXPECT_LE(lfs::core::to_wire_envelope(error).dump().size(),
              static_cast<std::size_t>(lfs::kMaxSerializedErrorBytes));
}

TEST(ErrorLatchTest, SetGetClearRoundTrip) {
    lfs::core::ErrorLatch latch;
    EXPECT_FALSE(latch.get().has_value());

    latch.set(lfs::make_error(lfs::ErrorInit{
        .code = lfs::ErrorCode::NotFound,
        .domain = lfs::ErrorDomain::IO,
        .detection = LFS_SOURCE_SITE_CURRENT(),
    }));

    const auto latched = latch.get();
    ASSERT_TRUE(latched.has_value());
    EXPECT_EQ(latched->code(), lfs::ErrorCode::NotFound);
    EXPECT_EQ(latched->domain(), lfs::ErrorDomain::IO);

    latch.clear();
    EXPECT_FALSE(latch.get().has_value());
}

TEST(ErrorLatchTest, ConcurrentSetAndGetNeverTears) {
    lfs::core::ErrorLatch latch;
    std::atomic<bool> stop{false};

    std::thread writer([&latch, &stop] {
        while (!stop.load(std::memory_order_relaxed)) {
            latch.set(lfs::make_error(lfs::ErrorInit{
                .code = lfs::ErrorCode::ResourceExhausted,
                .domain = lfs::ErrorDomain::CUDA,
                .detection = LFS_SOURCE_SITE_CURRENT(),
            }));
            latch.clear();
        }
    });

    for (int i = 0; i < 100000; ++i) {
        if (const auto snapshot = latch.get(); snapshot.has_value()) {
            EXPECT_EQ(snapshot->code(), lfs::ErrorCode::ResourceExhausted);
            EXPECT_EQ(snapshot->domain(), lfs::ErrorDomain::CUDA);
        }
    }

    stop.store(true, std::memory_order_relaxed);
    writer.join();
}
