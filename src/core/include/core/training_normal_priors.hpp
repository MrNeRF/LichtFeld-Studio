#pragma once
#include "core/parameters.hpp"
namespace lfs::training {
    inline bool training_normal_priors_enabled(const lfs::core::param::OptimizationParameters& opt) {
        return !opt.gut && opt.use_normal_loss && opt.normal_loss_weight > 0.0f;
    }

} // namespace lfs::training
