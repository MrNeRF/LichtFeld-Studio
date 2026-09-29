# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

add_subdirectory(vulkan)

# Native Metal viewer stages; desktop presentation remains Vulkan until parity.
if(APPLE)
    add_subdirectory(metal)
endif()
