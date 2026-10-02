# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

# Embeds Metal kernel sources, in order, as one C++ string that the Metal
# backend compiles at runtime. NAMESPACE and SYMBOL name the string.
if(NOT NAMESPACE)
    set(NAMESPACE lfs::core::internal::metal)
endif()
if(NOT SYMBOL)
    set(SYMBOL kKernelSource)
endif()
set(source "")
foreach(input IN LISTS INPUTS)
    file(READ "${input}" part)
    string(APPEND source "${part}")
endforeach()
file(WRITE "${OUTPUT}"
     "namespace ${NAMESPACE} {\n"
     "    extern const char* const ${SYMBOL};\n"
     "    const char* const ${SYMBOL} = R\"lfsmsl(${source})lfsmsl\";\n"
     "} // namespace ${NAMESPACE}\n")
