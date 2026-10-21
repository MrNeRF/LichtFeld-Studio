/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <core/gpu_device_info.hpp>
#include <algorithm>
#include <vector>
namespace lfs::core::internal {
    // A driver UUID is usable as a selector only when it identifies one adapter.
    // Preserve the original inventory while detecting collisions, so changing an
    // earlier entry cannot hide a later duplicate.
    inline void disambiguate_hip_hardware(std::vector<HipHardwareDevice>& devices) {
        const auto original=devices;
        for(auto& device:devices) {
            const auto duplicates=std::ranges::count_if(original,[&](const auto& item){return item.uuid==device.uuid;});
            if(!device.uuid.empty() && duplicates==1)continue;
            const auto pci_matches=std::ranges::count_if(original,[&](const auto& item){return item.pci_bus_id==device.pci_bus_id;});
            if(!device.pci_bus_id.empty() && pci_matches==1)device.uuid="pci:"+device.pci_bus_id;
            else device.uuid="hip-index:"+std::to_string(device.index);
        }
    }
}
