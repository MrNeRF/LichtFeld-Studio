# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

# Embeds kernels.metal as a C++ string; the Metal backend compiles it at runtime.
file(READ "${INPUT}" source)
file(WRITE "${OUTPUT}"
     "namespace lfs::core::internal::metal {\n"
     "    extern const char* const kKernelSource;\n"
     "    const char* const kKernelSource = R\"lfsmsl(${source})lfsmsl\";\n"
     "} // namespace lfs::core::internal::metal\n")
