/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "strategy_factory.hpp"
#include "improved_gs_plus.hpp"
#include "mcmc.hpp"
#include "mrnf.hpp"
#include <format>
#include <mutex>

namespace lfs::training {

    namespace {
        [[nodiscard]] std::string canonical_strategy_key(const std::string_view name) {
            const auto canonical_name = core::param::canonical_strategy_name(name);
            return canonical_name.empty() ? std::string(name) : std::string(canonical_name);
        }
    } // namespace

    StrategyFactory& StrategyFactory::instance() {
        static StrategyFactory factory;
        return factory;
    }

    StrategyFactory::StrategyFactory() {
        register_builtins();
    }

    void StrategyFactory::register_builtins() {
        registry_[std::string(core::param::kStrategyMCMC)] = [](core::SplatData& model)
            -> std::expected<std::unique_ptr<IStrategy>, std::string> {
            return std::make_unique<MCMC>(model);
        };

        registry_[std::string(core::param::kStrategyMRNF)] = [](core::SplatData& model)
            -> std::expected<std::unique_ptr<IStrategy>, std::string> {
            return std::make_unique<MRNF>(model);
        };

        registry_[std::string(core::param::kStrategyIGSPlus)] = [](core::SplatData& model)
            -> std::expected<std::unique_ptr<IStrategy>, std::string> {
            return std::make_unique<ImprovedGSPlus>(model);
        };
    }

    std::expected<std::unique_ptr<IStrategy>, std::string>
    StrategyFactory::create(const std::string& name, core::SplatData& model) const {
        const std::string key = canonical_strategy_key(name);
        std::shared_lock lock(mutex_);
        const auto it = registry_.find(key);
        if (it == registry_.end()) {
            std::string available;
            for (const auto& [n, _] : registry_) {
                if (!available.empty()) {
                    available += ", ";
                }
                available += n;
            }
            return std::unexpected(
                std::format("Unknown strategy: '{}'. Available: {}", name, available));
        }
        return it->second(model);
    }

} // namespace lfs::training
