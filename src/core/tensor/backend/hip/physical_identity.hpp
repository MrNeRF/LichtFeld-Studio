/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <cstdio>
#include <format>
#include <string>
#include <string_view>
#include <cstdint>
namespace lfs::core::internal {
inline std::string pci_physical_id(unsigned domain,unsigned bus,unsigned device,unsigned function) {
    if(bus>255||device>31||function>7)return {};
    return std::format("pci:{:04x}:{:02x}:{:02x}.{:x}",domain,bus,device,function);
}
inline std::string pci_physical_id(std::string_view raw) {
    unsigned domain=0,bus=0,device=0,function=0;int consumed=0;const std::string text(raw);
    if(std::sscanf(text.c_str(),"%x:%x:%x.%x%n",&domain,&bus,&device,&function,&consumed)!=4||size_t(consumed)!=text.size())return {};
    return pci_physical_id(domain,bus,device,function);
}
#if defined(_WIN32)
std::string luid_physical_id(const uint8_t luid[8]);
#endif
}
