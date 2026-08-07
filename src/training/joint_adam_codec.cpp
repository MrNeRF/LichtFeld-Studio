/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/training/joint_adam_codec.hpp"

namespace lfs::training::joint_adam {

    void set_joint_codec_enabled_for_testing(const std::optional<bool> /*enabled*/) {
        // Joint (u, log_s) is the only Adam codec. The testing override is a no-op
        // kept for call-site compatibility while legacy tests are removed.
    }

    bool joint_codec_enabled() {
        return true;
    }

} // namespace lfs::training::joint_adam
