/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/camera.hpp"
#include "core/cuda_error_typed.hpp"
#include "core/failure_report.hpp"
#include "core/logger.hpp"
#include "core/scene.hpp"
#include "core/tensor.hpp"
#include "training/trainer.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <gtest/gtest.h>
#include <mutex>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace {

    class ErrorLogCapture {
    public:
        ErrorLogCapture()
            : token_(lfs::core::Logger::get().add_log_handler(
                  [this](const lfs::core::LogLevel level,
                         const lfs::core::SourceSite&,
                         const std::string_view message) {
                      if (level != lfs::core::LogLevel::Error) {
                          return;
                      }
                      std::scoped_lock lock(mutex_);
                      messages_.emplace_back(message);
                  })) {}

        ~ErrorLogCapture() {
            lfs::core::Logger::get().remove_log_handler(token_);
        }

        [[nodiscard]] std::string joined() const {
            std::scoped_lock lock(mutex_);
            std::string result;
            for (const auto& message : messages_) {
                result += message;
                result += '\n';
            }
            return result;
        }

    private:
        lfs::core::LogHandlerToken token_;
        mutable std::mutex mutex_;
        std::vector<std::string> messages_;
    };

    class CudaErrorTypedTest : public ::testing::Test {
    protected:
        void SetUp() override {
            lfs::core::reset_cuda_diagnostics_for_testing();
        }

        void TearDown() override {
            lfs::core::reset_cuda_diagnostics_for_testing();
        }
    };

    class CudaErrorTypedDeathTest : public CudaErrorTypedTest {};

    [[nodiscard]] lfs::core::CudaCheckCompletion completion_for(
        const cudaError_t effective_error) {
        lfs::core::CudaCheckCompletion completion;
        completion.effective_error = effective_error;
        return completion;
    }

    [[nodiscard]] const lfs::SmallFields::Entry* find_field(
        const lfs::Error& error, const std::string_view key) {
        if (error.frames().empty()) {
            return nullptr;
        }
        for (const auto& entry : error.frames().front().fields.entries()) {
            if (entry.key == key) {
                return &entry;
            }
        }
        return nullptr;
    }

    [[nodiscard]] std::shared_ptr<lfs::core::Camera> make_camera(const int uid) {
        return std::make_shared<lfs::core::Camera>(
            lfs::core::Tensor::eye(3, lfs::core::Device::CPU),
            lfs::core::Tensor::zeros({3}, lfs::core::Device::CPU),
            100.0f, 100.0f, 32.0f, 32.0f,
            lfs::core::Tensor(), lfs::core::Tensor(),
            lfs::core::CameraModelType::PINHOLE,
            "camera.png", std::filesystem::path{}, std::filesystem::path{},
            64, 64, uid);
    }

    TEST_F(CudaErrorTypedTest, DirectStatusThrowsTypedCudaError) {
        try {
            lfs::core::throw_cuda_error(
                cudaErrorMemoryAllocation, {}, completion_for(cudaErrorMemoryAllocation),
                "test_call", "test message", LFS_SOURCE_SITE_CURRENT());
            FAIL() << "throw_cuda_error returned";
        } catch (const lfs::Exception& exception) {
            const lfs::Error& error = exception.error();
            EXPECT_EQ(error.domain(), lfs::ErrorDomain::CUDA);
            EXPECT_EQ(error.code(), lfs::ErrorCode::ResourceExhausted);
            ASSERT_TRUE(error.native().has_value());
            EXPECT_EQ(error.native()->code, cudaErrorMemoryAllocation);
        }
    }

    TEST_F(CudaErrorTypedTest, MapsMemoryAllocationToResourceExhausted) {
        EXPECT_EQ(lfs::core::cuda_status_to_error_code(cudaErrorMemoryAllocation),
                  lfs::ErrorCode::ResourceExhausted);
    }

    TEST_F(CudaErrorTypedTest, MapsInitializationFailureToUnavailable) {
        EXPECT_EQ(lfs::core::cuda_status_to_error_code(cudaErrorInitializationError),
                  lfs::ErrorCode::Unavailable);
    }

    TEST_F(CudaErrorTypedTest, MapsIllegalAddressToDeviceLost) {
        EXPECT_EQ(lfs::core::cuda_status_to_error_code(cudaErrorIllegalAddress),
                  lfs::ErrorCode::DeviceLost);
    }

    TEST_F(CudaErrorTypedTest, MapsInvalidValueToInvalidArgument) {
        EXPECT_EQ(lfs::core::cuda_status_to_error_code(cudaErrorInvalidValue),
                  lfs::ErrorCode::InvalidArgument);
    }

    TEST_F(CudaErrorTypedTest, MapsUnknownStatusToInternal) {
        EXPECT_EQ(lfs::core::cuda_status_to_error_code(cudaErrorNotSupported),
                  lfs::ErrorCode::Internal);
    }

    TEST_F(CudaErrorTypedTest, PreexistingStickyErrorIsSuppressed) {
        lfs::core::CudaCheckState state;
        state.pre_call_sampled = true;
        state.pre_call_error = cudaErrorLaunchFailure;

        const lfs::Error error = lfs::core::make_cuda_error(
            cudaErrorMemoryAllocation, state, completion_for(cudaErrorMemoryAllocation),
            "test_call", {}, LFS_SOURCE_SITE_CURRENT());

        EXPECT_EQ(error.code(), lfs::ErrorCode::ResourceExhausted);
        ASSERT_TRUE(error.native().has_value());
        EXPECT_EQ(error.native()->code, cudaErrorMemoryAllocation);
        ASSERT_EQ(error.suppressed().size(), 1u);
        ASSERT_TRUE(error.suppressed().front().native().has_value());
        EXPECT_EQ(error.suppressed().front().native()->code, cudaErrorLaunchFailure);
    }

    TEST_F(CudaErrorTypedTest, UnavailableLatchRemainsIndependentAndTerminal) {
        EXPECT_THROW(
            lfs::core::throw_cuda_error(
                cudaErrorInitializationError, {}, completion_for(cudaErrorInitializationError),
                "test_call", {}, LFS_SOURCE_SITE_CURRENT()),
            lfs::Exception);
        EXPECT_FALSE(lfs::core::cuda_is_unavailable());

        EXPECT_TRUE(lfs::core::latch_cuda_unavailable(cudaErrorInitializationError));
        EXPECT_FALSE(lfs::core::latch_cuda_unavailable(cudaErrorInitializationError));
        EXPECT_TRUE(lfs::core::cuda_is_unavailable());
    }

    TEST_F(CudaErrorTypedTest, TeardownFailureLogsOnceAndNeverThrows) {
        lfs::core::initialize_cuda_diagnostics();
        ErrorLogCapture capture;

        const auto log_failure = [] {
            lfs::core::log_cuda_teardown_failure(
                cudaErrorMemoryAllocation, {}, completion_for(cudaErrorMemoryAllocation),
                "test_teardown", "test teardown message", LFS_SOURCE_SITE_CURRENT());
        };
        EXPECT_NO_THROW(log_failure());
        EXPECT_NO_THROW(log_failure());

        const std::string report = capture.joined();
        EXPECT_NE(report.find("Family: CUDA"), std::string::npos);
        EXPECT_NE(report.find("cudaErrorMemoryAllocation"), std::string::npos);
        const std::string_view header = "========== LFS FAILURE REPORT ==========";
        EXPECT_EQ(report.find(header), report.rfind(header));
    }

    TEST_F(CudaErrorTypedTest, MultiStreamFieldsRemainDistinct) {
        lfs::core::CudaCheckState first_state;
        first_state.stream = 0x1234;
        lfs::core::CudaCheckState second_state;
        second_state.stream = 0x5678;

        const lfs::Error first = lfs::core::make_cuda_error(
            cudaErrorInvalidValue, first_state, completion_for(cudaErrorInvalidValue),
            "first", {}, LFS_SOURCE_SITE_CURRENT());
        const lfs::Error second = lfs::core::make_cuda_error(
            cudaErrorInvalidValue, second_state, completion_for(cudaErrorInvalidValue),
            "second", {}, LFS_SOURCE_SITE_CURRENT());

        ASSERT_NE(find_field(first, "stream"), nullptr);
        ASSERT_NE(find_field(second, "stream"), nullptr);
        EXPECT_EQ(std::get<std::int64_t>(find_field(first, "stream")->value), 0x1234);
        EXPECT_EQ(std::get<std::int64_t>(find_field(second, "stream")->value), 0x5678);
    }

    TEST_F(CudaErrorTypedTest, SyncDebugEffectiveStatusPreservesRawStatus) {
        lfs::core::CudaCheckCompletion completion;
        completion.effective_error = cudaErrorIllegalAddress;
        completion.post_sync_error = cudaErrorIllegalAddress;
        const lfs::Error error = lfs::core::make_cuda_error(
            cudaSuccess, {}, completion, "test_call", {}, LFS_SOURCE_SITE_CURRENT());

        EXPECT_EQ(error.domain(), lfs::ErrorDomain::CUDA);
        ASSERT_TRUE(error.native().has_value());
        EXPECT_EQ(error.native()->code, cudaErrorIllegalAddress);
        const auto* raw_status = find_field(error, "raw_status");
        ASSERT_NE(raw_status, nullptr);
        EXPECT_NE(std::get<std::string>(raw_status->value).find("cudaSuccess"), std::string::npos);
    }

    TEST_F(CudaErrorTypedTest, TrainerShutdownIsIdempotent) {
        lfs::core::Scene scene;
        const auto cameras = scene.addGroup("Cameras");
        scene.addCamera("camera.png", cameras, make_camera(0));
        lfs::training::Trainer trainer(scene);

        EXPECT_NO_THROW(trainer.shutdown());
        EXPECT_NO_THROW(trainer.shutdown());
    }

    TEST_F(CudaErrorTypedTest, TrainerModelReadLifecycleSucceeds) {
        lfs::core::Scene scene;
        const auto cameras = scene.addGroup("Cameras");
        scene.addCamera("camera.png", cameras, make_camera(0));
        lfs::training::Trainer trainer(scene);

        EXPECT_NO_THROW({
            trainer.beginModelRead(nullptr);
            trainer.endModelRead(nullptr);
        });
    }

    TEST_F(CudaErrorTypedTest, SeveralCamerasConstructAndDestroyCleanly) {
        EXPECT_NO_THROW({
            std::vector<std::shared_ptr<lfs::core::Camera>> cameras;
            for (int i = 0; i < 4; ++i) {
                cameras.push_back(make_camera(i));
            }
            cameras.clear();
        });
    }

    TEST_F(CudaErrorTypedDeathTest, PoisonedContextTryThrowsCatchableException) {
        EXPECT_EXIT(
            {
                try {
                    LFS_CUDA_TRY(cudaSetDevice(-1), nullptr, "poisoned direct call");
                    std::_Exit(2);
                } catch (const lfs::Exception&) {
                    std::_Exit(0);
                } catch (...) {
                    std::_Exit(3);
                }
            },
            ::testing::ExitedWithCode(0), "");
    }

    TEST_F(CudaErrorTypedDeathTest, PoisonedContextTeardownStillExitsCleanly) {
        EXPECT_EXIT(
            {
                LFS_CUDA_LOG_TEARDOWN(cudaSetDevice(-1), nullptr, "poisoned teardown call");
                std::_Exit(0);
            },
            ::testing::ExitedWithCode(0), "");
    }

} // namespace
