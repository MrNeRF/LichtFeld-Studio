/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/argument_parser.hpp"
#include "python_runtime_api.hpp"

namespace lfs::python {
    LFS_PYTHON_RUNTIME_API int run_plugin_command(const lfs::core::args::PluginMode& mode);
}
