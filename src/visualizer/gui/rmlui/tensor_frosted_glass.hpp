/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/error.hpp"
#include "core/export.hpp"
#include "core/tensor.hpp"

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

namespace lfs::vis::gui {

    struct TensorFrostedGlassRegion {
        float x = 0.0f;
        float y = 0.0f;
        float width = 0.0f;
        float height = 0.0f;
        float radius = 0.0f;
    };

    struct TensorFrostedGlassRect {
        float left = 0.0f;
        float top = 0.0f;
        float right = 0.0f;
        float bottom = 0.0f;
    };

    [[nodiscard]] LFS_VIS_API std::vector<TensorFrostedGlassRect>
    frostedGlassClipRects(std::span<const TensorFrostedGlassRegion> regions,
                          float target_width, float target_height);

    class LFS_VIS_API TensorFrostedGlassBackdrop {
    public:
        TensorFrostedGlassBackdrop();
        ~TensorFrostedGlassBackdrop();
        TensorFrostedGlassBackdrop(const TensorFrostedGlassBackdrop&) = delete;
        TensorFrostedGlassBackdrop& operator=(const TensorFrostedGlassBackdrop&) = delete;

        [[nodiscard]] lfs::Status update(const lfs::core::Tensor& source);
        [[nodiscard]] const lfs::core::Tensor& image() const noexcept;
        [[nodiscard]] std::size_t bytes() const noexcept;
        void reset();

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace lfs::vis::gui
