/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/gpu_backend_fwd.hpp"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <string_view>

namespace {
    bool drivers_are_usable() {
        constexpr std::array<std::string_view, 3> names{"CUDA", "Vulkan", "Metal"};
        for (size_t i = 0; i < names.size(); ++i) {
            if (lfs::core::gpu_backend_name(lfs::core::kGpuBackends[i]) != names[i])
                return false;
        }
        return true;
    }

    void check_exit_drivers() {
        if (!drivers_are_usable()) {
            std::fputs("Backend driver unavailable during process exit\n", stderr);
            std::_Exit(EXIT_FAILURE);
        }
        std::puts("Backend driver exit contracts passed");
    }
} // namespace

int main() {
    // Register before the first driver lookup, as a standalone Python import
    // does. GoogleTest discovery would initialize the registry too early.
    if (std::atexit(check_exit_drivers) != 0)
        return EXIT_FAILURE;
    return drivers_are_usable() ? EXIT_SUCCESS : EXIT_FAILURE;
}
