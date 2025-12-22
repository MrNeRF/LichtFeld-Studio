/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <memory>

namespace lfs::core::param {
    struct TrainingParameters;
}

namespace lfs::app {

    class Application {
    public:
        int run(std::unique_ptr<lfs::core::param::TrainingParameters> params);
    };

} // namespace lfs::app
