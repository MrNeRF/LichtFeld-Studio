/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/error.hpp"
#include "core/shareable_allocation_limit.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "io/formats/ply.hpp"
#include "io/loader.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace lfs::core;

namespace {

    class ScopedShareableAllocLimit {
    public:
        explicit ScopedShareableAllocLimit(const char* value) {
            if (const char* previous = std::getenv(kShareableAllocLimitEnvName)) {
                previous_ = previous;
            }
            set(value);
            reset_shareable_allocation_limit_for_tests();
        }

        ~ScopedShareableAllocLimit() {
            set(previous_ ? previous_->c_str() : nullptr);
            reset_shareable_allocation_limit_for_tests();
        }

        ScopedShareableAllocLimit(const ScopedShareableAllocLimit&) = delete;
        ScopedShareableAllocLimit& operator=(const ScopedShareableAllocLimit&) = delete;

    private:
        static void set(const char* value) {
#ifdef _WIN32
            (void)_putenv_s(kShareableAllocLimitEnvName, value ? value : "");
#else
            if (value) {
                (void)setenv(kShareableAllocLimitEnvName, value, 1);
            } else {
                (void)unsetenv(kShareableAllocLimitEnvName);
            }
#endif
        }

        std::optional<std::string> previous_;
    };

    void require_cuda() {
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
            GTEST_SKIP() << "CUDA device unavailable";
        }
    }

    Tensor retag_external(Tensor tensor, std::string kind) {
        const TensorShape shape = tensor.shape();
        const auto device = tensor.device();
        const auto dtype = tensor.dtype();
        const size_t capacity = tensor.capacity();
        const cudaStream_t stream = tensor.stream();
        auto owner = std::make_shared<Tensor>(std::move(tensor));
        return Tensor::from_external_owner(owner->data_ptr(),
                                           shape,
                                           device,
                                           dtype,
                                           owner,
                                           capacity,
                                           stream,
                                           std::move(kind));
    }

    struct AllocCall {
        std::string name;
        DataType dtype;
    };

    SplatTensorAllocator recording_allocator(std::vector<AllocCall>& calls) {
        return [&calls](TensorShape shape,
                        const size_t capacity,
                        const DataType dtype,
                        const std::string_view name) {
            calls.push_back(AllocCall{std::string{name}, dtype});
            Tensor backing = Tensor::zeros_direct(shape, capacity, Device::CUDA, dtype);
            return retag_external(std::move(backing), "vulkan_external_buffer");
        };
    }

    [[nodiscard]] bool called_for_float_shN(const std::vector<AllocCall>& calls) {
        return std::any_of(calls.begin(), calls.end(), [](const AllocCall& call) {
            return call.name == "SplatData.shN" && call.dtype == DataType::Float32;
        });
    }

    [[nodiscard]] bool called_for(const std::vector<AllocCall>& calls, const std::string_view name) {
        return std::any_of(calls.begin(), calls.end(), [&](const AllocCall& call) {
            return call.name == name;
        });
    }

} // namespace

TEST(ShareableAllocationLimitTest, EnvOverrideIsParsedAndCached) {
    ScopedShareableAllocLimit limit("123456789");
    EXPECT_EQ(max_shareable_allocation_bytes(), 123456789u);
    EXPECT_TRUE(shareable_allocation_limited());
    EXPECT_TRUE(shareable_allocation_limit_from_env());
    EXPECT_EQ(max_shareable_allocation_bytes(), 123456789u);
}

TEST(ShareableAllocationLimitTest, InvalidEnvFallsThroughToPlatformDefault) {
    ScopedShareableAllocLimit limit("not-a-number");
    constexpr std::size_t kCeiling = (std::size_t{1} << 32) - (std::size_t{64} << 20);
    EXPECT_EQ(max_shareable_allocation_bytes(), kCeiling);
    EXPECT_TRUE(shareable_allocation_limited());
    EXPECT_FALSE(shareable_allocation_limit_from_env());
}

TEST(ShareableAllocationLimitTest, PlatformDefaultWithoutEnv) {
    ScopedShareableAllocLimit limit(nullptr);
    constexpr std::size_t kCeiling = (std::size_t{1} << 32) - (std::size_t{64} << 20);
    EXPECT_EQ(max_shareable_allocation_bytes(), kCeiling);
    EXPECT_TRUE(shareable_allocation_limited());
    EXPECT_FALSE(shareable_allocation_limit_from_env());
}

TEST(ShareableAllocationLimitTest, ChunkEnvCapsAndRoundsToGranularity) {
    constexpr std::size_t kGran = std::size_t{2} << 20;
    {
        ScopedShareableAllocLimit limit(nullptr);
#ifdef _WIN32
        (void)_putenv_s(kShareableChunkBytesEnvName, "16777216");
#else
        ASSERT_EQ(setenv(kShareableChunkBytesEnvName, "16777216", 1), 0);
#endif
        reset_shareable_allocation_limit_for_tests();
        EXPECT_EQ(max_shareable_allocation_bytes(), 16777216u);
        EXPECT_EQ(shareable_chunk_bytes(), 16777216u);
#ifdef _WIN32
        (void)_putenv_s(kShareableChunkBytesEnvName, "");
#else
        unsetenv(kShareableChunkBytesEnvName);
#endif
        reset_shareable_allocation_limit_for_tests();
    }
    {
        ScopedShareableAllocLimit limit(nullptr);
#ifdef _WIN32
        (void)_putenv_s(kShareableChunkBytesEnvName, "1000");
#else
        ASSERT_EQ(setenv(kShareableChunkBytesEnvName, "1000", 1), 0);
#endif
        reset_shareable_allocation_limit_for_tests();
        EXPECT_EQ(shareable_chunk_bytes(), kGran);
#ifdef _WIN32
        (void)_putenv_s(kShareableChunkBytesEnvName, "");
#else
        unsetenv(kShareableChunkBytesEnvName);
#endif
        reset_shareable_allocation_limit_for_tests();
    }
}

TEST(ShareableAllocationLimitTest, DeviceLimitIsHonoredAndRounded) {
    ScopedShareableAllocLimit limit(nullptr);
    set_shareable_device_allocation_limit(8ull << 20);
    EXPECT_EQ(max_shareable_allocation_bytes(), 8ull << 20);
    EXPECT_EQ(shareable_chunk_bytes(), 8ull << 20);
    set_shareable_device_allocation_limit((8ull << 20) + 123);
    EXPECT_EQ(max_shareable_allocation_bytes(), (8ull << 20) + 123);
    EXPECT_EQ(shareable_chunk_bytes(), 8ull << 20);
}

TEST(ShareableAllocationLimitTest, ViolationMessageIncludesNumbers) {
    ScopedShareableAllocLimit limit("1000");
    EXPECT_FALSE(shareable_allocation_violation(1000, "SplatData.shN").has_value());
    const auto message = shareable_allocation_violation(1001, "SplatData.shN");
    ASSERT_TRUE(message.has_value());
    EXPECT_NE(message->find("1001"), std::string::npos) << *message;
    EXPECT_NE(message->find("1000"), std::string::npos) << *message;
    EXPECT_NE(message->find("SplatData.shN"), std::string::npos) << *message;
    EXPECT_NE(message->find("shareable GPU allocation"), std::string::npos) << *message;
    EXPECT_TRUE(is_shareable_allocation_limit_message(*message));
}

TEST(ShareableAllocationLimitTest, MigrateRequestsFloatShNRegardlessOfLimit) {
    require_cuda();
    const auto ply_path = std::filesystem::path(PROJECT_ROOT_PATH) / "gd.ply";
    if (!std::filesystem::exists(ply_path)) {
        GTEST_SKIP() << "Missing test asset: " << ply_path;
    }

    auto loaded = lfs::io::load_ply(ply_path);
    ASSERT_TRUE(loaded.has_value()) << lfs::format_for_developer(loaded.error());
    SplatData model = std::move(loaded->value);
    ASSERT_EQ(model.size(), 3000000u);
    ASSERT_EQ(model.shN_raw().dtype(), DataType::Float32);
    ASSERT_GT(model.shN_raw().bytes(), 400ull << 20);

    ScopedShareableAllocLimit limit("419430400"); // 400 MiB
    std::vector<AllocCall> calls;
    const auto result = lfs::io::migrateSplatTensorsToAllocator(model, recording_allocator(calls));
    ASSERT_TRUE(result.has_value()) << result.error().format();
    EXPECT_TRUE(called_for(calls, "SplatData.means"));
    EXPECT_TRUE(called_for(calls, "SplatData.sh0"));
    EXPECT_TRUE(called_for(calls, "SplatData.scaling"));
    EXPECT_TRUE(called_for(calls, "SplatData.rotation"));
    EXPECT_TRUE(called_for(calls, "SplatData.opacity"));
    EXPECT_TRUE(called_for_float_shN(calls));
    EXPECT_TRUE(lfs::io::splatTensorsRendererReady(model));
}

TEST(ShareableAllocationLimitTest, MigrateRequestsFloatShNWithoutLimit) {
    require_cuda();
    const auto ply_path = std::filesystem::path(PROJECT_ROOT_PATH) / "gd.ply";
    if (!std::filesystem::exists(ply_path)) {
        GTEST_SKIP() << "Missing test asset: " << ply_path;
    }

    auto loaded = lfs::io::load_ply(ply_path);
    ASSERT_TRUE(loaded.has_value()) << lfs::format_for_developer(loaded.error());
    SplatData model = std::move(loaded->value);
    ASSERT_EQ(model.shN_raw().dtype(), DataType::Float32);

    ScopedShareableAllocLimit unlimited(nullptr);
    std::vector<AllocCall> calls;
    const auto result = lfs::io::migrateSplatTensorsToAllocator(model, recording_allocator(calls));
    ASSERT_TRUE(result.has_value()) << result.error().format();
    EXPECT_TRUE(called_for_float_shN(calls));
    EXPECT_TRUE(lfs::io::splatTensorsRendererReady(model));
}
