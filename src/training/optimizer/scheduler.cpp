/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "scheduler.hpp"
#include "adam_optimizer.hpp"
#include "core/logger.hpp"
#include "core/tensor_serialization.hpp"
#include <array>
#include <cmath>
#include <string>
#include <unordered_map>

namespace lfs::training {

    void ExponentialLR::step() {
        // Calculate decay factor
        double decay_factor = gamma_;

        // Map param type to name for logging
        static const std::unordered_map<ParamType, std::string> param_names = {
            {ParamType::Means, "means"},
            {ParamType::Sh0, "sh0"},
            {ParamType::ShN, "shN"},
            {ParamType::Scaling, "scaling"},
            {ParamType::Rotation, "rotation"},
            {ParamType::Opacity, "opacity"}};

        if (params_to_update_.empty()) {
            // Default behavior (MCMC): Update ONLY global LR (means uses this)
            double current_lr = optimizer_.get_lr();
            double new_lr = current_lr * decay_factor;
            LOG_DEBUG("ExponentialLR::step() - Global LR: {:.6e} → {:.6e} (gamma={:.6f})",
                      current_lr, new_lr, decay_factor);
            optimizer_.set_lr(static_cast<float>(new_lr));

            // Log other params for visibility (they stay constant)
            for (auto param_type : AdamOptimizer::all_param_types()) {
                if (optimizer_.has_param_lr(param_type)) {
                    LOG_DEBUG("  {} LR: {:.6e} (constant)",
                              param_names.at(param_type),
                              optimizer_.get_param_lr(param_type));
                }
            }
        } else {
            for (auto param_type : params_to_update_) {
                if (optimizer_.has_param_lr(param_type)) {
                    // Use double precision to match legacy behavior
                    double current_param_lr = optimizer_.get_param_lr(param_type);
                    double new_param_lr = current_param_lr * decay_factor;
                    optimizer_.set_param_lr(param_type, new_param_lr);

                    LOG_DEBUG("ExponentialLR::step() - {} LR: {:.15e} → {:.15e}",
                              param_names.at(param_type), current_param_lr, new_param_lr);
                }
            }

            // Also update global LR if it's being used by any param
            double current_lr = optimizer_.get_lr();
            double new_lr = current_lr * decay_factor;
            optimizer_.set_lr(static_cast<float>(new_lr));
        }
    }

    // ===== Serialization =====

    namespace {
        constexpr uint32_t SCHED_EXPONENTIAL_MAGIC = 0x4C465345; // "LFSE"
        constexpr uint32_t SCHED_VERSION = 1;
    } // namespace

    void ExponentialLR::serialize(std::ostream& os) const {
        os.write(reinterpret_cast<const char*>(&SCHED_EXPONENTIAL_MAGIC), sizeof(SCHED_EXPONENTIAL_MAGIC));
        os.write(reinterpret_cast<const char*>(&SCHED_VERSION), sizeof(SCHED_VERSION));
        os.write(reinterpret_cast<const char*>(&gamma_), sizeof(gamma_));

        // Write params_to_update
        uint32_t num_params = static_cast<uint32_t>(params_to_update_.size());
        os.write(reinterpret_cast<const char*>(&num_params), sizeof(num_params));
        for (const auto& param : params_to_update_) {
            uint8_t param_val = static_cast<uint8_t>(param);
            os.write(reinterpret_cast<const char*>(&param_val), sizeof(param_val));
        }

        LOG_DEBUG("Serialized ExponentialLR: gamma={}", gamma_);
    }

    void ExponentialLR::deserialize(std::istream& is) {
        uint32_t magic = 0, version = 0;
        lfs::core::serialization_detail::read_exact(is, &magic, sizeof(magic), "scheduler magic");
        lfs::core::serialization_detail::read_exact(is, &version, sizeof(version), "scheduler version");

        if (magic != SCHED_EXPONENTIAL_MAGIC) {
            throw std::runtime_error("Invalid ExponentialLR checkpoint: wrong magic");
        }
        if (version != SCHED_VERSION) {
            throw std::runtime_error("Unsupported ExponentialLR checkpoint version");
        }

        double gamma = 0.0;
        lfs::core::serialization_detail::read_exact(is, &gamma, sizeof(gamma), "scheduler gamma");
        if (!std::isfinite(gamma) || gamma <= 0.0)
            throw std::runtime_error("Invalid ExponentialLR checkpoint: gamma must be finite and positive");

        uint32_t num_params = 0;
        lfs::core::serialization_detail::read_exact(
            is, &num_params, sizeof(num_params), "scheduler parameter count");
        if (num_params > AdamOptimizer::all_param_types().size())
            throw std::runtime_error("Invalid ExponentialLR checkpoint: too many parameters");
        std::vector<ParamType> params_to_update;
        params_to_update.reserve(num_params);
        std::array<bool, 6> seen{};
        for (uint32_t i = 0; i < num_params; ++i) {
            uint8_t param_val = 0;
            lfs::core::serialization_detail::read_exact(
                is, &param_val, sizeof(param_val), "scheduler parameter id");
            if (param_val >= seen.size() || seen[param_val])
                throw std::runtime_error("Invalid ExponentialLR checkpoint: invalid parameter id");
            seen[param_val] = true;
            params_to_update.push_back(static_cast<ParamType>(param_val));
        }

        gamma_ = gamma;
        params_to_update_ = std::move(params_to_update);

        LOG_DEBUG("Deserialized ExponentialLR: gamma={}", gamma_);
    }

    void ExponentialLR::adopt_checkpoint_state(ExponentialLR& loaded) noexcept {
        std::swap(gamma_, loaded.gamma_);
        params_to_update_.swap(loaded.params_to_update_);
    }

} // namespace lfs::training
