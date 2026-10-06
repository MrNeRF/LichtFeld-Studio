/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/splat_data.hpp"
#include "istrategy.hpp"
#include <expected>
#include <functional>
#include <memory>
#include <shared_mutex>
#include <string>
#include <unordered_map>

namespace lfs::training {

    class StrategyFactory {
    public:
        using Creator = std::function<
            std::expected<std::unique_ptr<IStrategy>, std::string>(core::SplatData&)>;

        static StrategyFactory& instance();

        [[nodiscard]] std::expected<std::unique_ptr<IStrategy>, std::string>
        create(const std::string& name, core::SplatData& model) const;

    private:
        StrategyFactory();
        void register_builtins();

        mutable std::shared_mutex mutex_;
        std::unordered_map<std::string, Creator> registry_;
    };

} // namespace lfs::training
