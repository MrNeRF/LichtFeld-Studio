/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include "io/argument_parser.hpp"

namespace lfs::python {
    int run_plugin_command(const lfs::io::args::PluginMode& mode);
}
