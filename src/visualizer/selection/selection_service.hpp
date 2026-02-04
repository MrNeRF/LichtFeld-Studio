/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "core/tensor.hpp"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace lfs::vis {

    class SceneManager;
    class RenderingManager;

    enum class SelectionMode { Replace,
                               Add,
                               Remove };

    struct SelectionResult {
        bool success = false;
        size_t affected_count = 0;
        std::string error;
    };

    class LFS_VIS_API SelectionService {
    public:
        SelectionService(SceneManager* scene_manager, RenderingManager* rendering_manager);
        ~SelectionService();

        SelectionService(const SelectionService&) = delete;
        SelectionService& operator=(const SelectionService&) = delete;

        [[nodiscard]] SelectionResult selectRect(float x0, float y0, float x1, float y1, SelectionMode mode,
                                                 int camera_index = 0);

        [[nodiscard]] SelectionResult selectPolygon(const core::Tensor& vertices, SelectionMode mode, int camera_index = 0);

        [[nodiscard]] SelectionResult applyMask(const std::vector<uint8_t>& mask, SelectionMode mode);
        [[nodiscard]] SelectionResult applyMask(const core::Tensor& mask, SelectionMode mode);

        void beginStroke();
        [[nodiscard]] core::Tensor* getStrokeSelection();
        [[nodiscard]] SelectionResult finalizeStroke(SelectionMode mode, const std::vector<bool>& node_mask = {});
        void cancelStroke();

        [[nodiscard]] bool isStrokeActive() const { return stroke_active_; }
        [[nodiscard]] size_t getTotalGaussianCount() const;
        [[nodiscard]] bool hasScreenPositions() const;
        [[nodiscard]] std::shared_ptr<core::Tensor> getScreenPositions() const;

    private:
        static constexpr size_t LOCKED_GROUPS_SIZE = 8;

        [[nodiscard]] core::Tensor applyModeLogic(const core::Tensor& stroke, SelectionMode mode) const;

        SceneManager* scene_manager_;
        RenderingManager* rendering_manager_;

        bool stroke_active_ = false;
        core::Tensor stroke_selection_;
        std::shared_ptr<core::Tensor> selection_before_stroke_;
    };

} // namespace lfs::vis
