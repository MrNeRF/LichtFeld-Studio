/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// A model's visible and deleted counts past 2^24 deleted splats. SplatData::visible_count() subtracts a sum of the
// deleted mask from the size, and the exports' limits, the scene merge (which sizes the merged tensors from it) and
// the GUI's counts rely on it being exact. A float holds every integer only up to 2^24 = 16,777,216.
//
// Tiers by device memory: see large_model_test_support.hpp. Each test's memory is in its skip message.

#include "large_model_test_support.hpp"

#include "core/splat_data.hpp"
#include "core/tensor_backend.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>

namespace {
    using lfs::core::Device;
    using lfs::core::SplatData;
    using lfs::core::Tensor;
    namespace lm = lfs::test::large_model;

    SplatData sh0_model(const size_t rows) {
        return SplatData(0, Tensor::zeros({rows, 3}, Device::GPU), Tensor::zeros({rows, 1, 3}, Device::GPU), Tensor{},
                         Tensor::zeros({rows, 3}, Device::GPU), Tensor::full({rows, 4}, 0.5f, Device::GPU),
                         Tensor::zeros({rows, 1}, Device::GPU), 1.0f);
    }

    // Deletes every splat but the first `kept` of a model of `rows`, and checks both counts.
    void expect_counts(const size_t rows, const size_t kept) {
        if (const auto reason = lm::skip_reason(rows * 14 * sizeof(float) + 2 * rows))
            GTEST_SKIP() << *reason;
        SplatData model = sh0_model(rows);
        Tensor mask = Tensor::ones_bool({rows}, Device::GPU);
        if (kept > 0)
            mask.slice(0, 0, kept).copy_(Tensor::zeros_bool({kept}, Device::GPU));
        (void)model.soft_delete(mask);
        EXPECT_EQ(static_cast<size_t>(model.visible_count()), kept)
            << rows << " splats with " << rows - kept << " deleted";
        model.refresh_deleted_count();
        EXPECT_EQ(model.deleted_count(), rows - kept) << rows << " splats with " << rows - kept << " deleted";
    }

    // 2^24 + 3 splats, all deleted: as a float the sum rounds to 2^24 + 2 or 2^24 + 4.
    TEST(LargeModelSplatCounts, EverySplatDeletedPastTwoTo24) { expect_counts((size_t{1} << 24) + 3, 0); }

    // 100,000,001 splats, all but 1,000 deleted (the model of the SOG limit's tests).
    TEST(LargeModelSplatCounts, AThousandVisibleOfAHundredMillion) { expect_counts(100'000'001, 1000); }

} // namespace
