/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/gpu_device_runtime.hpp"
#include "core/gpu_kernel_module.hpp"
#include "core/nn/models/lpips.hpp"
#include "core/nn/models/romav1.hpp"
#include "core/nn/ops.hpp"
#include "core/nn/weight_file.hpp"
#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_debug.hpp"
#include "core/tensor_spatial.hpp"
#include "core/tensor_upload.hpp"
#include "core/tensor_vulkan_interop.hpp"
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>
#include <optional>
#include <set>
#include <type_traits>

namespace {
    using namespace lfs::core;
    class RestoredTensor : public testing::TestWithParam<std::optional<GpuBackend>> {
    protected:
        std::optional<GpuBackendScope> scope;
        Device device() const { return GetParam() ? Device::GPU : Device::CPU; }
        void SetUp() override {
            if (GetParam()) {
                if (!gpu_backend_available(*GetParam()))
                    GTEST_SKIP() << "Backend unavailable";
                scope.emplace(*GetParam());
            }
        }
        Tensor values(std::vector<float> data, TensorShape shape) {
            return Tensor::from_vector(data, shape, device());
        }
        void near(const Tensor& tensor, const std::vector<float>& expected, float tolerance = 1e-5f) {
            EXPECT_EQ(tensor.device(), device());
            EXPECT_EQ(gpu_backend_of(tensor), GetParam());
            const auto actual = tensor.to(DataType::Float32).to_vector();
            ASSERT_EQ(actual.size(), expected.size());
            for (size_t i = 0; i < actual.size(); ++i)
                EXPECT_NEAR(actual[i], expected[i], tolerance) << i;
        }
    };

    TEST_P(RestoredTensor, RandomSampling) {
        near(Tensor::normal({4}, 2.5f, 0.0f, device()), {2.5f, 2.5f, 2.5f, 2.5f});
        near(Tensor::bernoulli({4}, 0.0f, device()), {0, 0, 0, 0});
        near(Tensor::bernoulli({4}, 1.0f, device()), {1, 1, 1, 1});
        const auto normal = Tensor::normal({20000}, 3.0f, 2.0f, device());
        EXPECT_NEAR(normal.mean().item<float>(), 3.0f, 0.08f);
        EXPECT_NEAR(normal.sub(3.0f).square().mean().item<float>(), 4.0f, 0.2f);
        auto weights = values({0, 9, 1, 9, 0, 9}, {3, 2}).slice(1, 0, 1).squeeze();
        auto draws = Tensor::multinomial(weights, 128, true, 42);
        EXPECT_EQ(draws.to_vector_int64(), std::vector<int64_t>(128, 1));
        const auto tiny = values({0, 1e-20f, 0}, {3});
        for (uint64_t seed = 1; seed <= 12; ++seed)
            EXPECT_EQ(Tensor::multinomial(tiny, 1, false, seed).to_vector_int64(), (std::vector<int64_t>{1}));
        const auto uniform = Tensor::ones({8}, device());
        draws = Tensor::multinomial(uniform, 8, false, 43);
        const auto indices = draws.to_vector_int64();
        EXPECT_EQ(std::set<int64_t>(indices.begin(), indices.end()), (std::set<int64_t>{0, 1, 2, 3, 4, 5, 6, 7}));
        EXPECT_EQ(Tensor::multinomial(uniform, 16, true, 44).to_vector_int64(),
                  Tensor::multinomial(uniform, 16, true, 44).to_vector_int64());
        EXPECT_THROW(Tensor::multinomial(values({-1, 2}, {2}), 1, true), std::exception);
    }

    TEST_P(RestoredTensor, GeometryAndReductions) {
        near(Tensor::diag(values({2, 99, 3, 99}, {2, 2}).slice(1, 0, 1).squeeze()), {2, 0, 0, 3});
        const auto a = values({0, 0, 3, 4}, {2, 2});
        const auto b = values({0, 0}, {1, 2});
        near(a.cdist(b), {0, 5});
        near(a.cdist(b, 1), {0, 7});
        near(a.cdist(b, 0), {0, 2});
        near(a.cdist(b, std::numeric_limits<float>::infinity()), {0, 4});
        near(a.normalize(1), {0, 0, -1, 1});
        const auto infinity = values({std::numeric_limits<float>::infinity()}, {1, 1});
        near(infinity.cdist(infinity, 0), {0});
        EXPECT_TRUE(std::isnan(infinity.cdist(infinity, std::numeric_limits<float>::infinity()).item<float>()));
        near(a.reduce(ReduceOp::Sum), {7});
        near(a.reduce(ReduceOp::Mean), {1.75f});
        near(a.reduce(ReduceOp::Min), {0});
        near(a.reduce(ReduceOp::Max), {4});
        near(a.reduce(ReduceOp::Prod), {0});
        near(values({-5, 5}, {2}).mod(values({3, 3}, {2})), {-2, 2});
        const auto split = a.nonzero_split();
        ASSERT_EQ(split.size(), 2u);
        EXPECT_EQ(split[0].to_vector_int64(), (std::vector<int64_t>{1, 1}));
        EXPECT_EQ(split[1].to_vector_int64(), (std::vector<int64_t>{0, 1}));
        EXPECT_TRUE(a.all_close(a.clone()));
        EXPECT_FALSE(a.all_close(a.add(1.0f)));
    }

    TEST_P(RestoredTensor, HalfClampAndStridedMutation) {
        const float nan = std::numeric_limits<float>::quiet_NaN();
        auto input = values({-4, 99, 0.5f, 99, 4, 99, nan, 99}, {4, 2}).to(DataType::Float16);
        auto view = input.slice(1, 0, 1);
        auto result = view.clamp(-1, 1).to(DataType::Float32).to_vector();
        ASSERT_EQ(result.size(), 4u);
        EXPECT_EQ(result[0], -1);
        EXPECT_EQ(result[1], 0.5f);
        EXPECT_EQ(result[2], 1);
        EXPECT_TRUE(std::isnan(result[3]));
        view.clamp_min_(-2).clamp_(-2, 2);
        result = input.to(DataType::Float32).to_vector();
        EXPECT_EQ(result[0], -2);
        EXPECT_EQ(result[2], 0.5f);
        EXPECT_EQ(result[4], 2);
        EXPECT_TRUE(std::isnan(result[6]));
        for (size_t i = 1; i < result.size(); i += 2)
            EXPECT_EQ(result[i], 99);
        auto empty = Tensor::empty({0}, device(), DataType::Float16);
        EXPECT_EQ(empty.clamp(-1, 1).numel(), 0u);
        EXPECT_NO_THROW(empty.clamp_min_(0));
        auto positive_inf = values({std::numeric_limits<float>::infinity()}, {1});
        positive_inf.clamp_min_(0);
        EXPECT_TRUE(std::isinf(positive_inf.item<float>()));
    }

    TEST_P(RestoredTensor, WhereIntoAndMaskedConversion) {
        for (const auto dtype : {DataType::Float32, DataType::Float16}) {
            auto output = values({1, 2, 3, 4}, {4}).to(dtype);
            const auto mask = Tensor::from_vector(std::vector<bool>{true, false, true, false}, {4}, device());
            where_into(output, mask, 7.0f, output);
            near(output, {7, 2, 7, 4});
            const auto input = output.to(DataType::Float32);
            const Tensor selected = input[mask];
            near(selected, {7, 7});
            auto storage = values({1, 2, 3, 4, 5}, {5}).to(dtype);
            auto shifted = storage.slice(0, 1, 5);
            where_into(shifted, mask, 8.0f, storage.slice(0, 0, 4));
            near(storage, {1, 8, 2, 8, 4});
        }
    }

    TEST_P(RestoredTensor, NoBiasLayers) {
        const auto input = values({1, 2, 3, 4}, {2, 2});
        const auto weight = values({1, 0, 0, 2}, {2, 2});
        near(input.linear(weight), {1, 4, 3, 8});
        near(input.reshape({1, 2, 1, 2}).conv1x1(weight), {1, 2, 6, 8});
    }

    TEST_P(RestoredTensor, NeuralOperations) {
        for (const auto dtype : {DataType::Float32, DataType::Float16}) {
            auto x = values({0, 0, 3, 4}, {2, 2}).to(dtype);
            auto weight = values({1, 2}, {2}).to(dtype);
            const float eps = 1e-5f;
            near(nn::rms_norm(x, weight, eps), {0, 0, 3 / std::sqrt(12.5f + eps), 8 / std::sqrt(12.5f + eps)}, 0.003f);
            near(nn::silu(x), {0, 0, 3 / (1 + std::exp(-3.0f)), 4 / (1 + std::exp(-4.0f))}, 0.003f);
            near(nn::residual_scale(x, x, weight.reshape({2, 1})), {0, 0, 6, 12}, 0.003f);
            near(nn::softmax(x), {0.5f, 0.5f, 1 / (1 + std::exp(1.0f)), 1 / (1 + std::exp(-1.0f))}, 0.001f);
            const auto mask = values({0, -std::numeric_limits<float>::infinity()}, {1, 2}).to(dtype);
            near(nn::softmax(x, &mask), {1, 0, 1, 0}, 0.001f);
            auto sequence = values({1, 2, 3, 4, 5, 6}, {1, 1, 3, 2}).to(dtype);
            const auto windows = nn::window_partition(sequence, 2);
            near(windows, {1, 2, 3, 4, 5, 6, 0, 0});
            near(nn::window_unpartition(windows, 2, 3), {1, 2, 3, 4, 5, 6});
        }
    }

    TEST_P(RestoredTensor, LazyGatherAndUnary) {
        const auto input = values({-1, 2, -3, 4}, {4});
        const auto indices = Tensor::from_vector(std::vector<int>{2, 99, -1, 99, 0, 99}, {3, 2}, device()).slice(1, 0, 1).squeeze();
        near(input.gather_lazy(indices).eval(), {-3, 4, -1});
        near(input.gather_lazy(indices).map(ops::abs_op{}).eval(), {3, 4, 1});
        near(input.gather_lazy(indices).map(ops::neg_op{}).eval(), {3, -4, 1});
        near(input.square().gather_lazy(indices).map(ops::sqrt_op{}).eval(), {3, 4, 1});
        near(input.gather_lazy(indices).map(ops::square_op{}).eval(), {9, 16, 1});
        near(input.to(DataType::Int32).gather_lazy(indices).map(ops::abs_op{}).eval(), {3, 4, 1});
        near(input.to(DataType::Float16).gather_lazy(indices).map(ops::abs_op{}).eval(), {3, 4, 1});
        const auto predicate = input.to(DataType::Float16).gather_lazy(indices).map(ops::isfinite_op{});
        EXPECT_EQ(predicate.dtype(), DataType::Bool);
        near(predicate.eval(), {1, 1, 1});
    }

    TEST_P(RestoredTensor, InspectionUsesLogicalValues) {
        const float nan = std::numeric_limits<float>::quiet_NaN(), inf = std::numeric_limits<float>::infinity();
        const auto input = values({1, 99, nan, 99, 3, 99, inf, 99}, {4, 2}).slice(1, 0, 1);
        const auto validation = debug::validate_tensor(input);
        EXPECT_EQ(validation.nan_count, 1u);
        EXPECT_EQ(validation.inf_count, 1u);
        EXPECT_FLOAT_EQ(validation.min_val, 1);
        EXPECT_FLOAT_EQ(validation.max_val, 3);
        EXPECT_FLOAT_EQ(validation.mean_val, 2);
        const auto stats = debug::get_tensor_stats(input);
        EXPECT_FLOAT_EQ(stats.mean, 2);
        EXPECT_FLOAT_EQ(stats.std, 1);
        const float maximum = std::numeric_limits<float>::max();
        const auto wide = debug::get_tensor_stats(values({-maximum, maximum}, {2}));
        EXPECT_TRUE(std::isfinite(wide.std));
        EXPECT_NEAR(wide.std / maximum, 1.0f, 1e-5f);
        const auto finite = values({1, 99, 3, 99}, {2, 2}).slice(1, 0, 1);
        const auto diff = debug::diff_tensors(finite, finite.add(2));
        EXPECT_FLOAT_EQ(diff.max_abs_diff, 2);
        EXPECT_FLOAT_EQ(diff.mean_abs_diff, 2);
        EXPECT_EQ(diff.num_different, 2u);
        EXPECT_FALSE(debug::diff_tensors(input, input).is_close());
        const auto mixed = debug::diff_tensors(values({0, 1000}, {2}), values({0.001f, 1001}, {2}));
        EXPECT_TRUE(mixed.is_close(0.01f, 0.002f));
        EXPECT_FALSE(mixed.is_close(0.0001f, 0.0001f));
    }

    TEST_P(RestoredTensor, FormattedPrintingHandlesScalarsAndHighRanks) {
        testing::internal::CaptureStdout();
        values({2.5f}, {}).print_formatted();
        values({1, 2, 3, 4}, {1, 1, 2, 2}).print_formatted("values", 1);
        Tensor::empty({0, 3}, device()).print_formatted();
        const auto text = testing::internal::GetCapturedStdout();
        EXPECT_NE(text.find("2.5000"), std::string::npos);
        EXPECT_NE(text.find("1.0000"), std::string::npos);
        EXPECT_EQ(text.find("4.0000"), std::string::npos);
    }

    TEST_P(RestoredTensor, MutableElementAccessPreservesViewsAndStorage) {
        auto input = values({1, 2, 3, 4, 5, 6}, {2, 3});
        auto transposed = input.transpose(0, 1);
        EXPECT_FLOAT_EQ(static_cast<float>(transposed.at({2, 1})), 6);
        transposed.at({2, 1}) = 9;
        transposed.at({1, 0}) += 3;
        transposed.at({0, 0}) = transposed.at({2, 1});
        near(input, {9, 5, 3, 4, 5, 9});
        auto retained = transposed.at({1, 1});
        transposed = Tensor{};
        input = Tensor{};
        retained *= 2;
        EXPECT_FLOAT_EQ(static_cast<float>(retained), 10);
        auto scalar = values({3}, {});
        scalar.at({}) -= 1;
        scalar.at({}) /= 2;
        EXPECT_FLOAT_EQ(static_cast<float>(scalar.at({})), 1);
        EXPECT_THROW(scalar.at({0}), std::exception);
    }

    TEST_P(RestoredTensor, ConnectedComponentsOverload) {
        const auto points = values({0, 0, 0, 0.1f, 0, 0, 5, 0, 0}, {3, 3});
        const auto labels = radius_connected_components(points, 0.2f).to_vector_int();
        ASSERT_EQ(labels.size(), 3u);
        EXPECT_EQ(labels[0], labels[1]);
        EXPECT_NE(labels[0], labels[2]);
    }

    TEST_P(RestoredTensor, AllocationAndBorrowedQueue) {
        auto tensor = values({1, 2, 3, 4}, {4});
        ASSERT_TRUE(reserved_allocation_bytes(tensor));
        EXPECT_GE(*reserved_allocation_bytes(tensor), tensor.bytes());
        tensor.reserve(16);
        ASSERT_TRUE(reserved_allocation_bytes(tensor));
        EXPECT_GE(*reserved_allocation_bytes(tensor), 16 * sizeof(float));
        if (!GetParam())
            return;
        TensorWorkQueue owner(*GetParam());
        Tensor result;
        {
            TensorWorkQueue borrowed(*GetParam(), owner.native_handle());
            EXPECT_EQ(borrowed.backend(), *GetParam());
            const TensorWorkQueue::Scope bound(borrowed);
            result = tensor.add(2.0f);
            borrowed.wait();
        }
        // Destroying the adapter must leave the owned execution target usable.
        {
            const TensorWorkQueue::Scope bound(owner);
            result = result.mul(2.0f);
            owner.wait();
        }
        near(result, {6, 8, 10, 12});
    }

    TEST_P(RestoredTensor, WeightInspection) {
        const std::string header = R"({"tensors":{"weight":{"dtype":"float32","shape":[2],"offset":0,"length":8}}})";
        const auto path = std::filesystem::temp_directory_path() /
                          ("tensor_weights_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".lfw");
        struct Cleanup {
            std::filesystem::path path;
            ~Cleanup() {
                std::error_code error;
                std::filesystem::remove(path, error);
            }
        } cleanup{path};
        {
            std::ofstream file(path, std::ios::binary);
            file.write("LFW1", 4);
            const uint32_t length = header.size();
            for (unsigned i = 0; i < 4; ++i)
                file.put(static_cast<char>(length >> (8 * i)));
            file.write(header.data(), header.size());
            while (static_cast<size_t>(file.tellp()) % 64)
                file.put(0);
            const float data[]{2, 3};
            file.write(reinterpret_cast<const char*>(data), sizeof(data));
            ASSERT_TRUE(file);
        }
        auto archive = nn::WeightFile::open(path);
        ASSERT_TRUE(archive) << archive.error().detail();
        EXPECT_TRUE(archive->contains("weight"));
        EXPECT_FALSE(archive->contains("missing"));
        auto tensor = archive->load("weight", device());
        ASSERT_TRUE(tensor) << tensor.error().detail();
        near(*tensor, {2, 3});
        EXPECT_EQ(nn::models::Lpips{}.weights_bytes(), 0u);
        EXPECT_EQ(nn::models::RomaV1{}.weights_bytes(), 0u);
    }

    static_assert(std::is_convertible_v<TensorElementProxy, float>);
    static_assert(std::is_assignable_v<TensorElementProxy&, float>);
    static_assert(std::is_same_v<decltype(static_cast<TensorElementProxy (Tensor::*)(std::initializer_list<size_t>)>(&Tensor::at)),
                                 TensorElementProxy (Tensor::*)(std::initializer_list<size_t>)>);
    static_assert(std::is_same_v<decltype(&nn::softmax), Tensor (*)(const Tensor&, const Tensor*)>);
    static_assert(std::is_same_v<decltype(&nn::silu), Tensor (*)(const Tensor&)>);
    static_assert(std::is_same_v<decltype(&nn::rms_norm), Tensor (*)(const Tensor&, const Tensor&, float)>);
    static_assert(std::is_same_v<decltype(&nn::residual_scale), Tensor (*)(const Tensor&, const Tensor&, const Tensor&)>);
    static_assert(std::is_same_v<decltype(&nn::window_partition), Tensor (*)(const Tensor&, int)>);
    static_assert(std::is_same_v<decltype(&nn::window_unpartition), Tensor (*)(const Tensor&, int, int)>);
    static_assert(std::is_same_v<decltype(&where_into), void (*)(Tensor&, const Tensor&, float, const Tensor&)>);
    static_assert(std::is_same_v<decltype(&nn::WeightFile::contains), bool (nn::WeightFile::*)(std::string_view) const>);
    static_assert(std::is_same_v<decltype(&nn::models::Lpips::weights_bytes), size_t (nn::models::Lpips::*)() const>);
    static_assert(std::is_same_v<decltype(&nn::models::RomaV1::weights_bytes), size_t (nn::models::RomaV1::*)() const>);
    static_assert(std::is_same_v<decltype(&GpuKernelModule::backend), GpuBackend (GpuKernelModule::*)() const>);
    static_assert(std::is_same_v<decltype(&reserved_allocation_bytes), std::optional<size_t> (*)(const Tensor&)>);
    static_assert(std::is_constructible_v<TensorWorkQueue, GpuBackend, void*>);

    const auto backends = [] {
        std::vector<std::optional<GpuBackend>> result{std::nullopt};
        for (const auto backend : kCompiledGpuBackends)
            result.emplace_back(backend);
        return result;
    }();
    INSTANTIATE_TEST_SUITE_P(Backends, RestoredTensor, testing::ValuesIn(backends),
                             [](const testing::TestParamInfo<std::optional<GpuBackend>>& info) {
                                 return !info.param ? "Cpu" : *info.param == GpuBackend::CUDA ? "Cuda"
                                                          : *info.param == GpuBackend::Vulkan ? "Vulkan"
                                                                                              : "Metal";
                             });
} // namespace
