# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

# Embeds the Metal kernel sources, in order, as one C++ string; the Metal
# backend compiles it at runtime.
set(source "")
foreach(input IN LISTS INPUTS)
    file(READ "${input}" part)
    string(APPEND source "${part}")
endforeach()
file(WRITE "${OUTPUT}"
     "namespace lfs::core::internal::metal {\n"
     "    extern const char* const kKernelSource;\n"
     "    const char* const kKernelSource = R\"lfsmsl(${source})lfsmsl\";\n"
     "} // namespace lfs::core::internal::metal\n")
