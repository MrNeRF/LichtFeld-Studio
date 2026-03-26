/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "strategy_factory.hpp"
#include "adc.hpp"
#include "core/logger.hpp"
#include "improved_gs_plus.hpp"
#include "lfs.hpp"
#include "mcmc.hpp"
#include <format>
#include <mutex>

namespace lfs::training {

    StrategyFactory& StrategyFactory::instance() {
        static StrategyFactory factory;
        return factory;
    }

    StrategyFactory::StrategyFactory() {
        register_builtins();
    }

    void StrategyFactory::register_builtins() {
        registry_["adc"] = [](core::SplatData& model)
            -> std::expected<std::unique_ptr<IStrategy>, std::string> {
            return std::make_unique<ADC>(model);
        };

        registry_["mcmc"] = [](core::SplatData& model)
            -> std::expected<std::unique_ptr<IStrategy>, std::string> {
            return std::make_unique<MCMC>(model);
        };

        registry_["lfs"] = [](core::SplatData& model)
            -> std::expected<std::unique_ptr<IStrategy>, std::string> {
            return std::make_unique<LFS>(model);
        };

        registry_["igs+"] = [](core::SplatData& model)
            -> std::expected<std::unique_ptr<IStrategy>, std::string> {
            return std::make_unique<ImprovedGSPlus>(model);
        };
    }

    bool StrategyFactory::register_creator(const std::string& name, Creator creator) {
        std::unique_lock lock(mutex_);
        if (registry_.contains(name)) {
            LOG_WARN("Strategy '{}' already registered", name);
            return false;
        }
        registry_[name] = std::move(creator);
        LOG_DEBUG("Registered strategy: {}", name);
        return true;
    }

    bool StrategyFactory::unregister(const std::string& name) {
        std::unique_lock lock(mutex_);
        return registry_.erase(name) > 0;
    }

    std::expected<std::unique_ptr<IStrategy>, std::string>
    StrategyFactory::create(const std::string& name, core::SplatData& model) const {
        std::shared_lock lock(mutex_);
        const auto it = registry_.find(name);
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

    bool StrategyFactory::has(const std::string& name) const {
        std::shared_lock lock(mutex_);
        return registry_.contains(name);
    }

    std::vector<std::string> StrategyFactory::list() const {
        std::shared_lock lock(mutex_);
        std::vector<std::string> names;
        names.reserve(registry_.size());
        for (const auto& [n, _] : registry_) {
            names.push_back(n);
        }
        return names;
    }

} // namespace lfs::training
