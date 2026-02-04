/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "operation.hpp"
#include <vector>

namespace lfs::vis::op {

    class LFS_VIS_API Pipeline {
    public:
        struct Stage {
            OperationFactory factory;
            OperatorProperties props;

            Stage() = default;
            Stage(OperationFactory f, OperatorProperties p = {})
                : factory(std::move(f)),
                  props(std::move(p)) {}
        };

        Pipeline() = default;
        explicit Pipeline(std::string name);

        Pipeline& add(OperationFactory factory, OperatorProperties props = {});
        Pipeline& add(Stage stage);

        Pipeline& operator|(Stage stage);

        OperationResult execute(SceneManager& scene);
        [[nodiscard]] bool poll(SceneManager& scene) const;
        [[nodiscard]] std::string description() const;
        [[nodiscard]] bool empty() const { return stages_.empty(); }
        [[nodiscard]] size_t size() const { return stages_.size(); }

    private:
        std::vector<Stage> stages_;
        std::string name_;

        [[nodiscard]] ModifiesFlag collectModifications() const;
    };

    inline Pipeline operator|(Pipeline::Stage a, Pipeline::Stage b) {
        Pipeline p;
        p.add(std::move(a));
        p.add(std::move(b));
        return p;
    }

    inline Pipeline& operator|(Pipeline& p, Pipeline::Stage stage) {
        return p.add(std::move(stage));
    }

} // namespace lfs::vis::op
