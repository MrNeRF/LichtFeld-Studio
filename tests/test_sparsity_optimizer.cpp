/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>

#include <sstream>
#include <vector>

#include "core/tensor.hpp"
#include "training/components/sparsity_optimizer.hpp"

using namespace lfs::core;
using namespace lfs::training;

TEST(ADMMSparsityOptimizerTest, ReinitializesWhenOpacityShapeChanges) {
    ADMMSparsityOptimizer optimizer(ADMMSparsityOptimizer::Config{
        .sparsify_steps = 100,
        .init_rho = 0.001f,
        .prune_ratio = 0.25f,
        .update_every = 10,
        .start_iteration = 0});

    auto initial = Tensor::zeros({8, 1}, Device::CUDA, DataType::Float32);
    ASSERT_TRUE(optimizer.initialize(initial).has_value());

    auto grown = Tensor::zeros({12, 1}, Device::CUDA, DataType::Float32);

    auto forward = optimizer.compute_loss_forward(grown);
    ASSERT_TRUE(forward.has_value()) << forward.error();
    EXPECT_EQ(forward->second.n, 12u);

    auto update = optimizer.update_state(grown);
    ASSERT_TRUE(update.has_value()) << update.error();

    auto prune_mask = optimizer.get_prune_mask(grown);
    ASSERT_TRUE(prune_mask.has_value()) << prune_mask.error();
    EXPECT_EQ(prune_mask->shape(), TensorShape({12}));
}

TEST(ADMMSparsityOptimizerTest, StartsAfterRegularTrainingBoundary) {
    ADMMSparsityOptimizer optimizer(ADMMSparsityOptimizer::Config{
        .sparsify_steps = 3,
        .init_rho = 0.001f,
        .prune_ratio = 0.25f,
        .update_every = 1,
        .start_iteration = 10});

    EXPECT_FALSE(optimizer.should_apply_loss(10));
    EXPECT_TRUE(optimizer.should_apply_loss(11));
    EXPECT_TRUE(optimizer.should_apply_loss(13));
    EXPECT_FALSE(optimizer.should_apply_loss(14));

    EXPECT_FALSE(optimizer.should_update(10));
    EXPECT_TRUE(optimizer.should_update(11));
    EXPECT_TRUE(optimizer.should_update(12));
    EXPECT_FALSE(optimizer.should_update(13));

    EXPECT_FALSE(optimizer.should_prune(12));
    EXPECT_TRUE(optimizer.should_prune(13));
    EXPECT_FALSE(optimizer.should_prune(14));
}

TEST(ADMMSparsityOptimizerTest, SerializeDeserializeRoundTrip) {
    const ADMMSparsityOptimizer::Config config{
        .sparsify_steps = 100,
        .init_rho = 0.001f,
        .prune_ratio = 0.25f,
        .update_every = 10,
        .start_iteration = 0};
    ADMMSparsityOptimizer source(config);

    const auto opacities = Tensor::from_vector(
        std::vector<float>{-2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f, 3.0f},
        {size_t{8}, size_t{1}}, Device::CUDA);
    ASSERT_TRUE(source.initialize(opacities).has_value());
    ASSERT_TRUE(source.update_state(opacities).has_value());

    std::stringstream stream;
    source.serialize(stream);

    ADMMSparsityOptimizer restored(config);
    restored.deserialize(stream);
    ASSERT_TRUE(restored.is_initialized());
    EXPECT_EQ(restored.state_size(), 8u);

    std::ostringstream source_bytes, restored_bytes;
    source.serialize(source_bytes);
    restored.serialize(restored_bytes);
    EXPECT_EQ(source_bytes.str(), restored_bytes.str());

    auto source_loss = source.compute_loss_forward(opacities);
    auto restored_loss = restored.compute_loss_forward(opacities);
    ASSERT_TRUE(source_loss.has_value()) << source_loss.error();
    ASSERT_TRUE(restored_loss.has_value()) << restored_loss.error();
    EXPECT_FLOAT_EQ(source_loss->first.item<float>(), restored_loss->first.item<float>());
}

TEST(ADMMSparsityOptimizerTest, DeserializeRejectsWrongMagic) {
    std::stringstream stream;
    const uint32_t wrong_magic = 0xDEADBEEF;
    const uint32_t version = 1;
    stream.write(reinterpret_cast<const char*>(&wrong_magic), sizeof(wrong_magic));
    stream.write(reinterpret_cast<const char*>(&version), sizeof(version));

    ADMMSparsityOptimizer optimizer(ADMMSparsityOptimizer::Config{});
    EXPECT_THROW(optimizer.deserialize(stream), std::runtime_error);
    EXPECT_FALSE(optimizer.is_initialized());
}

TEST(ADMMSparsityOptimizerTest, LossForwardReinitializesAfterReset) {
    ADMMSparsityOptimizer optimizer(ADMMSparsityOptimizer::Config{
        .sparsify_steps = 100,
        .init_rho = 0.001f,
        .prune_ratio = 0.25f,
        .update_every = 10,
        .start_iteration = 0});

    auto opacities = Tensor::zeros({8, 1}, Device::CUDA, DataType::Float32);
    ASSERT_TRUE(optimizer.initialize(opacities).has_value());

    optimizer.reset();
    EXPECT_FALSE(optimizer.is_initialized());

    auto forward = optimizer.compute_loss_forward(opacities);
    ASSERT_TRUE(forward.has_value()) << forward.error();
    EXPECT_TRUE(optimizer.is_initialized());
    EXPECT_EQ(forward->second.n, 8u);
}
