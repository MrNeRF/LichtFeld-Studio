/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/image_io.hpp"
#include "core/nn.hpp"
#include "metrics/metrics.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
    namespace fs = std::filesystem;
    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::Tensor;
    using lfs::core::nn::models::Lpips;

    fs::path repo_root() { return PROJECT_ROOT_PATH; }

    fs::path weights_path() {
        if (const char* env = std::getenv("LFS_LPIPS_WEIGHTS"); env && env[0])
            return env;
        if (const char* home = std::getenv("HOME"); home && home[0])
            return fs::path(home) / ".lichtfeld/onnx/lpips-vgg16-v0.1.lfw";
        return {};
    }

    Tensor chw_tensor(const std::vector<float>& values, int height, int width,
                      Device device = Device::CUDA) {
        return Tensor::from_vector(
            values, lfs::core::TensorShape(std::vector<std::size_t>{1, 3, static_cast<std::size_t>(height), static_cast<std::size_t>(width)}),
            device);
    }

    std::pair<Tensor, Tensor> synthetic_pair(const int height = 64, const int width = 96) {
        const std::size_t pixels = static_cast<std::size_t>(height) * width;
        std::vector<float> a(3 * pixels), b(3 * pixels);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const std::size_t i = static_cast<std::size_t>(y) * width + x;
                a[i] = static_cast<float>(x + 2 * y) / static_cast<float>(width - 1 + 2 * (height - 1));
                a[pixels + i] = static_cast<float>(x % 17) / 16.0f;
                a[2 * pixels + i] = static_cast<float>((3 * x + 5 * y) % 29) / 28.0f;
                b[i] = 1.0f - a[i];
                b[pixels + i] = static_cast<float>((x + 3 * y) % 19) / 18.0f;
                b[2 * pixels + i] = static_cast<float>((7 * x + 2 * y) % 31) / 30.0f;
            }
        }
        return {chw_tensor(a, height, width), chw_tensor(b, height, width)};
    }

    Tensor load_rgb(const fs::path& path) {
        auto [data, width, height, channels] = lfs::core::load_image(path);
        if (!data || channels != 3)
            throw std::runtime_error("failed to load RGB image " + path.string());
        std::vector<float> chw(static_cast<std::size_t>(3) * width * height);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const std::size_t pixel = static_cast<std::size_t>(y) * width + x;
                for (int c = 0; c < 3; ++c)
                    chw[static_cast<std::size_t>(c) * width * height + pixel] =
                        static_cast<float>(data[3 * pixel + c]) / 255.0f;
            }
        }
        lfs::core::free_image(data);
        return chw_tensor(chw, height, width);
    }

    nlohmann::json fixture() {
        std::ifstream input(repo_root() / "tests/data/nn/lpips_ref_fixture.json");
        return nlohmann::json::parse(input);
    }

    std::vector<float> host_values(const Tensor& tensor) {
        return tensor.to(DataType::Float32).to(Device::CPU).contiguous().to_vector();
    }

    void expect_samples(const Tensor& tensor, const nlohmann::json& node) {
        const auto got = host_values(tensor);
        const auto& indices = node.at("indices");
        const auto& values = node.at("values");
        const auto expected_size = std::accumulate(
            node.at("shape").begin(), node.at("shape").end(), std::size_t{1},
            [](std::size_t a, const auto& b) { return a * b.template get<std::size_t>(); });
        ASSERT_EQ(got.size(), expected_size);
        for (std::size_t i = 0; i < indices.size(); ++i)
            EXPECT_NEAR(got[indices[i].get<std::size_t>()], values[i].get<float>(), 1e-4f);
    }

    fs::path bicycle_a() { return repo_root() / "data/bicycle/images_4/_DSC8679.JPG"; }
    fs::path bicycle_b() { return repo_root() / "data/bicycle/images_4/_DSC8680.JPG"; }

    std::pair<fs::path, fs::path> crop_pair_paths() {
        return {repo_root() / "tests/data/nn/lpips_crop_a.png",
                repo_root() / "tests/data/nn/lpips_crop_b.png"};
    }

    std::optional<lfs::core::nn::models::Lpips> load_model(DataType dtype = DataType::Float32) {
        const auto path = weights_path();
        if (path.empty() || !fs::is_regular_file(path))
            return std::nullopt;
        auto loaded = lfs::core::nn::models::Lpips::load(path, Device::CUDA, dtype);
        if (!loaded)
            throw std::runtime_error(std::string(loaded.error().detail()));
        return std::move(*loaded);
    }

    class LpipsTest : public ::testing::Test {
        void SetUp() override {
            if (!fs::is_regular_file(weights_path()))
                GTEST_SKIP() << "LPIPS weights absent: " << weights_path();
        }
    };

    class ScopedConvPath {
    public:
        explicit ScopedConvPath(const char* path) {
            if (const char* previous = std::getenv("LFS_LPIPS_CONV"))
                previous_ = previous;
            set(path);
        }
        ~ScopedConvPath() { set(previous_ ? previous_->c_str() : nullptr); }

    private:
        static void set(const char* value) {
#ifdef _WIN32
            _putenv_s("LFS_LPIPS_CONV", value ? value : "");
#else
            if (value)
                setenv("LFS_LPIPS_CONV", value, 1);
            else
                unsetenv("LFS_LPIPS_CONV");
#endif
        }
        std::optional<std::string> previous_;
    };
} // namespace

TEST_F(LpipsTest, FixtureParity) {
    if (!fs::is_regular_file(repo_root() / "tests/data/nn/lpips_ref_fixture.json"))
        GTEST_SKIP() << "LPIPS reference fixture is absent";
    auto reference = fixture();
    auto model = load_model();
    ASSERT_TRUE(model.has_value());
    auto [a, b] = synthetic_pair();
    for (const auto& [name, scaling] : {
             std::pair{"identity", lfs::core::nn::models::InputScaling::Identity},
             std::pair{"normalize", lfs::core::nn::models::InputScaling::Normalize}}) {
        auto taps = model->forward_with_taps(a, b, scaling);
        ASSERT_TRUE(taps.has_value()) << taps.error().detail();
        const auto& expected = reference.at("synthetic").at(name);
        for (int i = 0; i < 5; ++i) {
            expect_samples(taps->normalized_features[static_cast<std::size_t>(i)],
                           expected.at("normalized_taps").at(static_cast<std::size_t>(i)));
            EXPECT_NEAR(taps->scalars[static_cast<std::size_t>(i)],
                        expected.at("per_tap_scalars").at(static_cast<std::size_t>(i)).get<float>(), 1e-4f);
        }
        auto value = model->forward(a, b, scaling);
        ASSERT_TRUE(value.has_value()) << value.error().detail();
        EXPECT_NEAR(*value, expected.at("lpips").get<float>(), 1e-4f);
    }
}

TEST_F(LpipsTest, RealPairReference) {
    if (!fs::is_regular_file(bicycle_a()) || !fs::is_regular_file(bicycle_b()))
        GTEST_SKIP() << "data/bicycle is absent";
    auto reference = fixture().at("real");
    auto model = load_model();
    ASSERT_TRUE(model.has_value());
    auto a = load_rgb(bicycle_a());
    auto b = load_rgb(bicycle_b());
    auto identity = model->forward(a, b, lfs::core::nn::models::InputScaling::Identity);
    auto normalize = model->forward(a, b, lfs::core::nn::models::InputScaling::Normalize);
    ASSERT_TRUE(identity.has_value()) << identity.error().detail();
    ASSERT_TRUE(normalize.has_value()) << normalize.error().detail();
    EXPECT_NEAR(*identity, reference.at("identity").get<float>(), 3e-3f);
    EXPECT_NEAR(*normalize, reference.at("normalize").get<float>(), 3e-3f);
}

TEST_F(LpipsTest, LosslessCropPairMatchesReference) {
    const auto [crop_a_path, crop_b_path] = crop_pair_paths();
    if (!fs::is_regular_file(crop_a_path) || !fs::is_regular_file(crop_b_path))
        GTEST_SKIP() << "LPIPS lossless crop pair is absent";
    const auto reference = fixture().at("crop_pair");
    auto model = load_model();
    ASSERT_TRUE(model.has_value());
    auto a = load_rgb(crop_a_path);
    auto b = load_rgb(crop_b_path);
    auto taps = model->forward_with_taps(
        a, b, lfs::core::nn::models::InputScaling::Identity);
    ASSERT_TRUE(taps.has_value()) << taps.error().detail();
    for (int i = 0; i < 5; ++i) {
        EXPECT_NEAR(taps->scalars[static_cast<std::size_t>(i)],
                    reference.at("per_tap_identity").at(static_cast<std::size_t>(i)).get<float>(),
                    1e-4f);
    }
    auto identity = model->forward(
        a, b, lfs::core::nn::models::InputScaling::Identity);
    auto normalize = model->forward(a, b, lfs::core::nn::models::InputScaling::Normalize);
    ASSERT_TRUE(identity.has_value()) << identity.error().detail();
    ASSERT_TRUE(normalize.has_value()) << normalize.error().detail();
    EXPECT_NEAR(*identity, reference.at("lpips_identity").get<float>(), 1e-5f);
    EXPECT_NEAR(*normalize, reference.at("lpips_normalize").get<float>(), 1e-5f);

    auto model16 = load_model(DataType::Float16);
    ASSERT_TRUE(model16.has_value());
    auto identity16 = model16->forward(
        a, b, lfs::core::nn::models::InputScaling::Identity);
    ASSERT_TRUE(identity16.has_value()) << identity16.error().detail();
    EXPECT_NEAR(*identity16, *identity, 5e-4f);
    std::cout << "LPIPS_CROP fp32=" << *identity << " fp16=" << *identity16
              << " abs_delta=" << std::abs(*identity - *identity16) << std::endl;
}

TEST_F(LpipsTest, TiledMatchesUntiled) {
    auto [a, b] = synthetic_pair(257, 319);
    auto untiled = load_model();
    constexpr std::size_t small_budget = 64ULL * 1024ULL * 1024ULL;
    auto tiled_loaded = Lpips::load(
        weights_path(), Device::CUDA, DataType::Float32,
        lfs::core::nn::models::InputScaling::Identity, small_budget);
    ASSERT_TRUE(untiled.has_value());
    ASSERT_TRUE(tiled_loaded.has_value()) << tiled_loaded.error().detail();
    auto tiled = std::move(*tiled_loaded);
    ASSERT_LT(tiled.tile_size_for(257, 319) + 224, 319U);
    auto expected = untiled->forward(a, b);
    auto actual = tiled.forward(a, b);
    ASSERT_TRUE(expected.has_value()) << expected.error().detail();
    ASSERT_TRUE(actual.has_value()) << actual.error().detail();
    EXPECT_NEAR(*actual, *expected, 1e-6f);
    untiled->release_activations();
    tiled.release_activations();
}

TEST_F(LpipsTest, FastTilingCoversOddSyntheticImage) {
    auto [a, b] = synthetic_pair(257, 319);
    auto untiled = load_model(DataType::Float16);
    auto tiled = Lpips::load(weights_path(), Device::CUDA, DataType::Float16,
                             lfs::core::nn::models::InputScaling::Identity, 24ULL * 1024 * 1024);
    ASSERT_TRUE(untiled);
    ASSERT_TRUE(tiled);
    ASSERT_LT(tiled->tile_size_for(257, 319) + 224, 319U);
    for (const auto scaling : {lfs::core::nn::models::InputScaling::Identity,
                               lfs::core::nn::models::InputScaling::Normalize}) {
        auto expected = untiled->forward(a, b, scaling);
        auto actual = tiled->forward(a, b, scaling);
        ASSERT_TRUE(expected);
        ASSERT_TRUE(actual);
        EXPECT_NEAR(*actual, *expected, 1e-5f);
    }
}

TEST_F(LpipsTest, WideImageEstimateCoversNewDeviceAllocations) {
    auto model = load_model(DataType::Float16);
    ASSERT_TRUE(model);
    auto [a, b] = synthetic_pair(512, 6000);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    Tensor::trim_memory_pool();
    const auto required = model->estimated_peak_bytes(512, 6000);
    std::size_t free_before, free_after, total;
    ASSERT_EQ(cudaMemGetInfo(&free_before, &total), cudaSuccess);
    if (free_before < required)
        GTEST_SKIP() << "Not enough free VRAM for the allocation regression";
    auto value = model->forward(a, b);
    ASSERT_TRUE(value) << value.error().detail();
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaMemGetInfo(&free_after, &total), cudaSuccess);
    if (free_after < free_before)
        EXPECT_LE(free_before - free_after, required);
    EXPECT_LT(model->estimated_peak_bytes(512, 6000), required);
    model->release_activations();
    EXPECT_EQ(model->estimated_peak_bytes(512, 6000), required);
    Tensor::trim_memory_pool();
}

TEST_F(LpipsTest, RejectsInputsThatCannotReachAllVggStages) {
    for (const auto dtype : {DataType::Float32, DataType::Float16}) {
        auto model = load_model(dtype);
        ASSERT_TRUE(model);
        for (const auto shape : {lfs::core::TensorShape({1, 3, 8, 32}),
                                 lfs::core::TensorShape({1, 3, 32, 8}),
                                 lfs::core::TensorShape({2, 3, 32, 32})}) {
            auto image = Tensor::zeros(shape, Device::CUDA);
            auto result = model->forward(image, image);
            ASSERT_FALSE(result);
            EXPECT_EQ(result.error().code(), lfs::ErrorCode::InvalidArgument);
        }
    }
}

TEST_F(LpipsTest, TiledRealPairMatchesUntiled) {
    if (fs::is_regular_file(bicycle_a()) && fs::is_regular_file(bicycle_b())) {
        auto real_a = load_rgb(bicycle_a());
        auto real_b = load_rgb(bicycle_b());
        constexpr std::size_t real_budget = 1024ULL * 1024ULL * 1024ULL;
        auto real_tiled_loaded = Lpips::load(
            weights_path(), Device::CUDA, DataType::Float32,
            lfs::core::nn::models::InputScaling::Identity, real_budget);
        ASSERT_TRUE(real_tiled_loaded.has_value()) << real_tiled_loaded.error().detail();
        auto real_tiled = std::move(*real_tiled_loaded);
        ASSERT_LT(real_tiled.tile_size_for(822, 1237), 1237U);
        auto real_expected = load_model();
        ASSERT_TRUE(real_expected.has_value());
        auto real_untiled_value = real_expected->forward(real_a, real_b);
        auto real_tiled_value = real_tiled.forward(real_a, real_b);
        ASSERT_TRUE(real_untiled_value.has_value()) << real_untiled_value.error().detail();
        ASSERT_TRUE(real_tiled_value.has_value()) << real_tiled_value.error().detail();
        std::cout << "LPIPS_TILED real_abs_delta="
                  << std::abs(*real_tiled_value - *real_untiled_value) << std::endl;
        EXPECT_NEAR(*real_tiled_value, *real_untiled_value, 1e-6f);
        real_expected->release_activations();
        real_tiled.release_activations();
        return;
    }
    GTEST_SKIP() << "data/bicycle is absent";
}

TEST_F(LpipsTest, TiledFastMatchesUntiled) {
    if (!fs::is_regular_file(bicycle_a()) || !fs::is_regular_file(bicycle_b()))
        GTEST_SKIP() << "data/bicycle is absent";
    auto real_a = load_rgb(bicycle_a());
    auto real_b = load_rgb(bicycle_b());
    auto untiled_loaded = load_model(DataType::Float16);
    constexpr std::size_t tiled_budget = 512ULL * 1024ULL * 1024ULL;
    auto tiled_loaded = Lpips::load(
        weights_path(), Device::CUDA, DataType::Float16,
        lfs::core::nn::models::InputScaling::Identity, tiled_budget);
    ASSERT_TRUE(untiled_loaded.has_value());
    ASSERT_TRUE(tiled_loaded.has_value()) << tiled_loaded.error().detail();
    auto untiled = std::move(*untiled_loaded);
    auto tiled = std::move(*tiled_loaded);
    ASSERT_LT(tiled.tile_size_for(822, 1237), 1237U);
    auto expected = untiled.forward(real_a, real_b);
    auto actual = tiled.forward(real_a, real_b);
    ASSERT_TRUE(expected.has_value()) << expected.error().detail();
    ASSERT_TRUE(actual.has_value()) << actual.error().detail();
    std::cout << "LPIPS_TILED_FAST real_abs_delta=" << std::abs(*actual - *expected)
              << std::endl;
    EXPECT_NEAR(*actual, *expected, 1e-6f);
    untiled.release_activations();
    tiled.release_activations();
}

TEST_F(LpipsTest, IdentityIsZero) {
    if (!fs::is_regular_file(bicycle_a()))
        GTEST_SKIP() << "data/bicycle is absent";
    auto model = load_model();
    ASSERT_TRUE(model.has_value());
    auto image = load_rgb(bicycle_a());
    auto value = model->forward(image, image);
    ASSERT_TRUE(value.has_value()) << value.error().detail();
    EXPECT_LE(*value, 1e-6f);
}

TEST_F(LpipsTest, BlurIsMonotonic) {
    if (!fs::is_regular_file(bicycle_a()))
        GTEST_SKIP() << "data/bicycle is absent";
    auto model = load_model();
    ASSERT_TRUE(model.has_value());
    auto original = load_rgb(bicycle_a()).to(Device::CPU).contiguous();
    const int height = static_cast<int>(original.shape()[2]);
    const int width = static_cast<int>(original.shape()[3]);
    const auto source = original.to_vector();
    std::array<float, 3> values{};
    int out_index = 0;
    for (const int radius : {1, 2, 4}) {
        std::vector<float> blurred(source.size());
        for (int c = 0; c < 3; ++c) {
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    double sum = 0.0;
                    int count = 0;
                    for (int dy = -radius; dy <= radius; ++dy) {
                        for (int dx = -radius; dx <= radius; ++dx) {
                            const int yy = std::clamp(y + dy, 0, height - 1);
                            const int xx = std::clamp(x + dx, 0, width - 1);
                            sum += source[static_cast<std::size_t>(c) * height * width +
                                          static_cast<std::size_t>(yy) * width + xx];
                            ++count;
                        }
                    }
                    blurred[static_cast<std::size_t>(c) * height * width +
                            static_cast<std::size_t>(y) * width + x] =
                        static_cast<float>(sum / count);
                }
            }
        }
        auto value = model->forward(original.to(Device::CUDA), chw_tensor(blurred, height, width));
        ASSERT_TRUE(value.has_value()) << value.error().detail();
        values[static_cast<std::size_t>(out_index++)] = *value;
    }
    EXPECT_LT(values[0], values[1]);
    EXPECT_LT(values[1], values[2]);
}

TEST_F(LpipsTest, Float16ComputeMatchesFloat32) {
    if (!fs::is_regular_file(bicycle_a()) || !fs::is_regular_file(bicycle_b()))
        GTEST_SKIP() << "data/bicycle is absent";
    auto a = load_rgb(bicycle_a());
    auto b = load_rgb(bicycle_b());
    auto model32 = load_model(DataType::Float32);
    auto model16 = load_model(DataType::Float16);
    ASSERT_TRUE(model32.has_value());
    ASSERT_TRUE(model16.has_value());
    auto value32 = model32->forward(a, b);
    ASSERT_TRUE(value32.has_value()) << value32.error().detail();
    auto value16 = model16->forward(a, b);
    ASSERT_TRUE(value16.has_value()) << value16.error().detail();
    EXPECT_LE(std::abs(*value32 - *value16), 5e-4f);
    model32->release_activations();
    model16->release_activations();
}

TEST_F(LpipsTest, ConvPathsAgree) {
    auto fast = load_model(DataType::Float16);
    auto exact = load_model(DataType::Float32);
    ASSERT_TRUE(fast.has_value());
    ASSERT_TRUE(exact.has_value());

    const auto run_forced = [&](const char* path, const Tensor& a, const Tensor& b) {
        const ScopedConvPath forced(path);
        auto value = fast->forward(a, b);
        if (!value) {
            ADD_FAILURE() << value.error().detail();
            return 0.0f;
        }
        return *value;
    };
    const auto check_pair = [&](const Tensor& a, const Tensor& b) {
        const float tensor = run_forced("tensor", a, b);
        const float simt = run_forced("simt", a, b);
        auto exact_value = exact->forward(a, b);
        if (!exact_value) {
            ADD_FAILURE() << exact_value.error().detail();
            return;
        }
        std::cout << "LPIPS_CONV_PATHS tensor=" << tensor << " simt=" << simt
                  << " exact=" << *exact_value << " tensor_simt_delta="
                  << std::abs(tensor - simt) << " tensor_exact_delta="
                  << std::abs(tensor - *exact_value) << std::endl;
        EXPECT_LE(std::abs(tensor - simt), 1e-4f);
        EXPECT_LE(std::abs(tensor - *exact_value), 5e-4f);
        EXPECT_LE(std::abs(simt - *exact_value), 5e-4f);
    };

    const auto [crop_a_path, crop_b_path] = crop_pair_paths();
    if (fs::is_regular_file(crop_a_path) && fs::is_regular_file(crop_b_path))
        check_pair(load_rgb(crop_a_path), load_rgb(crop_b_path));
    if (fs::is_regular_file(bicycle_a()) && fs::is_regular_file(bicycle_b()))
        check_pair(load_rgb(bicycle_a()), load_rgb(bicycle_b()));
    exact->release_activations();
    fast->release_activations();
}

TEST_F(LpipsTest, FallbackDispatchMatchesReference) {
    const auto [crop_a_path, crop_b_path] = crop_pair_paths();
    if (!fs::is_regular_file(crop_a_path) || !fs::is_regular_file(crop_b_path))
        GTEST_SKIP() << "LPIPS lossless crop pair is absent";
    auto model = load_model(DataType::Float16);
    ASSERT_TRUE(model.has_value());
    auto a = load_rgb(crop_a_path);
    auto b = load_rgb(crop_b_path);
    const ScopedConvPath forced("auto75");
    auto value = model->forward(a, b, lfs::core::nn::models::InputScaling::Identity);
    ASSERT_TRUE(value.has_value()) << value.error().detail();
    EXPECT_NEAR(*value, fixture().at("crop_pair").at("lpips_identity").get<float>(), 5e-4f);
}

TEST(LpipsKernelTest, ConvKernelShapeSweep) {
    struct Case {
        int n, cin, cout, h, w;
        lfs::core::nn::ConvPadMode pad;
        lfs::core::nn::Activation act;
    };
    const Case cases[] = {
        {1, 8, 64, 5, 7, lfs::core::nn::ConvPadMode::Zeros, lfs::core::nn::Activation::None},
        {1, 64, 64, 33, 47, lfs::core::nn::ConvPadMode::Zeros, lfs::core::nn::Activation::Relu},
        {2, 64, 128, 17, 19, lfs::core::nn::ConvPadMode::Zeros, lfs::core::nn::Activation::Relu},
        {1, 128, 128, 9, 40, lfs::core::nn::ConvPadMode::Replicate, lfs::core::nn::Activation::None},
        {1, 256, 512, 3, 3, lfs::core::nn::ConvPadMode::Zeros, lfs::core::nn::Activation::Relu},
        {1, 512, 512, 1, 1, lfs::core::nn::ConvPadMode::Zeros, lfs::core::nn::Activation::None},
        {1, 16, 24, 20, 30, lfs::core::nn::ConvPadMode::Zeros, lfs::core::nn::Activation::None},
        {1, 40, 72, 12, 17, lfs::core::nn::ConvPadMode::Replicate, lfs::core::nn::Activation::Relu},
        {1, 512, 512, 77, 51, lfs::core::nn::ConvPadMode::Zeros, lfs::core::nn::Activation::Relu},
    };
    for (const auto& c : cases) {
        const auto shape = [](std::initializer_list<std::size_t> dims) {
            return lfs::core::TensorShape(std::vector<std::size_t>(dims));
        };
        const auto input = Tensor::randn(shape({static_cast<std::size_t>(c.n), static_cast<std::size_t>(c.cin),
                                                static_cast<std::size_t>(c.h), static_cast<std::size_t>(c.w)}),
                                         Device::CUDA)
                               .to(DataType::Float16);
        const auto weight = Tensor::randn(shape({static_cast<std::size_t>(c.cout), static_cast<std::size_t>(c.cin), 3, 3}),
                                          Device::CUDA)
                                .mul(0.05f)
                                .to(DataType::Float16);
        const auto bias = Tensor::randn(shape({static_cast<std::size_t>(c.cout)}), Device::CUDA).to(DataType::Float16);
        lfs::core::nn::Conv2dParams params;
        params.pad_h = 1;
        params.pad_w = 1;
        params.pad_mode = c.pad;
        params.activation = c.act;
        const auto reference = [&] {
            const ScopedConvPath forced("simt");
            return host_values(lfs::core::nn::conv2d(input, weight, &bias, params));
        }();
        const auto actual = host_values(lfs::core::nn::conv2d(input, weight, &bias, params));
        ASSERT_EQ(reference.size(), actual.size());
        float max_delta = 0.0f;
        float max_ref = 0.0f;
        for (std::size_t i = 0; i < reference.size(); ++i) {
            max_delta = std::max(max_delta, std::abs(reference[i] - actual[i]));
            max_ref = std::max(max_ref, std::abs(reference[i]));
        }
        std::cout << "LPIPS_CONV_SWEEP n=" << c.n << " cin=" << c.cin << " cout=" << c.cout
                  << " h=" << c.h << " w=" << c.w << " max_delta=" << max_delta
                  << " max_ref=" << max_ref << std::endl;
        EXPECT_LE(max_delta, 2e-3f * std::max(1.0f, max_ref))
            << "cin=" << c.cin << " cout=" << c.cout << " h=" << c.h << " w=" << c.w;
    }
}

TEST_F(LpipsTest, ForwardDoesNotMutateInputs) {
    auto [batched_pred, batched_target] = synthetic_pair();
    const auto checksum = [](const Tensor& tensor) {
        const auto values = host_values(tensor);
        return std::accumulate(values.begin(), values.end(), 0.0, [](double sum, float value) {
            return sum + static_cast<double>(value);
        });
    };

    for (const auto dtype : {DataType::Float32, DataType::Float16}) {
        auto model = load_model(dtype);
        ASSERT_TRUE(model.has_value());
        for (const auto scaling : {lfs::core::nn::models::InputScaling::Identity,
                                   lfs::core::nn::models::InputScaling::Normalize}) {
            for (const bool batched : {false, true}) {
                auto pred = batched ? batched_pred : batched_pred.squeeze(0);
                auto target = batched ? batched_target : batched_target.squeeze(0);
                const auto pred_before = checksum(pred);
                const auto target_before = checksum(target);
                auto value = model->forward(pred, target, scaling);
                ASSERT_TRUE(value.has_value()) << value.error().detail();
                EXPECT_DOUBLE_EQ(checksum(pred), pred_before)
                    << "dtype=" << static_cast<int>(dtype) << " batched=" << batched;
                EXPECT_DOUBLE_EQ(checksum(target), target_before)
                    << "dtype=" << static_cast<int>(dtype) << " batched=" << batched;
            }
        }
        model->release_activations();
    }
}

TEST(EvalMetricsTest, EvalMetricsCsvRow) {
    lfs::training::EvalMetrics without;
    without.iteration = 7;
    without.psnr = 30.0f;
    without.ssim = 0.9f;
    const auto header = lfs::training::EvalMetrics::to_csv_header();
    EXPECT_EQ(header.substr(0, header.find(",time_per_image")),
              "iteration,psnr,ssim,lpips");
    EXPECT_NE(without.to_csv_row().find("30.000000,0.900000,,"), std::string::npos);

    lfs::training::EvalMetrics with = without;
    with.lpips = 0.123456f;
    EXPECT_NE(with.to_csv_row().find("30.000000,0.900000,0.123456,"), std::string::npos);
    with.valid = true;
    EXPECT_NE(with.to_string().find("LPIPS: 0.1235"), std::string::npos);
}
