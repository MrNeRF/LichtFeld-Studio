/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "operation/operation.hpp"

namespace lfs::vis::op {

    class LFS_VIS_API EditDelete : public Operation {
    public:
        OperationResult execute(SceneManager& scene,
                                const OperatorProperties& props,
                                const std::any& input) override;

        [[nodiscard]] bool poll(SceneManager& scene) const override;
        [[nodiscard]] std::string id() const override { return "edit.delete"; }
        [[nodiscard]] std::string label() const override { return "Delete Selected"; }
        [[nodiscard]] ModifiesFlag modifies() const override {
            return ModifiesFlag::SELECTION | ModifiesFlag::TOPOLOGY;
        }
    };

    class LFS_VIS_API EditDuplicate : public Operation {
    public:
        OperationResult execute(SceneManager& scene,
                                const OperatorProperties& props,
                                const std::any& input) override;

        [[nodiscard]] bool poll(SceneManager& scene) const override;
        [[nodiscard]] std::string id() const override { return "edit.duplicate"; }
        [[nodiscard]] std::string label() const override { return "Duplicate Selected"; }
        [[nodiscard]] ModifiesFlag modifies() const override {
            return ModifiesFlag::TOPOLOGY;
        }
    };

} // namespace lfs::vis::op
