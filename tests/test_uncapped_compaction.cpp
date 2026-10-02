/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/scene.hpp"
#include "io/project_document.hpp"
#include "licht_test_support.hpp"
#include "training/strategies/mrnf.hpp"
#include <algorithm>
#include <cstdlib>
#include <gtest/gtest.h>
#include <sstream>

using namespace lfs::core;
using namespace lfs::training;

TEST(MRNFCheckpoint, RealRemovalPreservesOptimizerCapacityInBothModes) {
    const auto* path = std::getenv("LFS_SPLIT_TEST_MODEL");
    if (!path)
        GTEST_SKIP() << "set real-model path";
    using namespace lfs::io::project;
    using namespace lfs::test::licht;
    Scene scene;
    auto document = require_result(ProjectDocument::open(path));
    static_cast<void>(require_result(document.hydrate(scene)));
    ASSERT_NE(scene.getCombinedModel(), nullptr);
    for (const bool capped : {false, true}) {
        SCOPED_TRACE(capped);
        auto model = scene.getCombinedModel()->clone();
        param::OptimizationParameters params;
        params.strategy = "mrnf";
        params.max_cap = capped ? static_cast<int>(model.size() + 128) : 0;
        params.background_improvements = false;
        MRNF strategy(model);
        strategy.initialize(params);
        auto opacity = model.opacity_raw();
        if (opacity.ndim() == 2)
            opacity = opacity.squeeze(-1);
        const auto values = opacity.cpu().to_vector();
        const auto [low, high] = std::minmax_element(values.begin(), values.end());
        ASSERT_LT(*low, *high);
        const auto mask = opacity.lt((*low + *high) * .5f);
        const size_t removed = static_cast<size_t>(mask.to(DataType::Int32).sum().item<int>());
        ASSERT_GT(removed, 0u);
        ASSERT_LT(removed, model.size());
        const auto initial_size = model.size();
        strategy.remove_gaussians(mask);
        EXPECT_EQ(model.size(), initial_size - removed);
        std::stringstream stream;
        strategy.get_optimizer().serialize(stream);
        auto restored_model = model.clone();
        AdamOptimizer restored(restored_model, strategy.get_optimizer().get_config());
        restored.allocate_gradients();
        ASSERT_NO_THROW(restored.deserialize(stream));
        for (const auto type : {ParamType::Means, ParamType::Sh0, ParamType::ShN,
                                ParamType::Scaling, ParamType::Rotation, ParamType::Opacity}) {
            const auto* source = strategy.get_optimizer().get_state(type);
            const auto* loaded = restored.get_state(type);
            ASSERT_NE(source, nullptr);
            ASSERT_NE(loaded, nullptr);
            EXPECT_EQ(loaded->size, source->size);
            EXPECT_GE(source->capacity, source->size);
            EXPECT_GE(loaded->capacity, loaded->size);
        }
    }
}
