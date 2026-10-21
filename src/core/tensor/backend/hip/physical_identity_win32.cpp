/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
// Keep Windows kernel display headers outside Tensor translation units.
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <winternl.h>
#include <d3dkmthk.h>
#include <cstring>
#include "physical_identity.hpp"
namespace lfs::core::internal {
std::string luid_physical_id(const uint8_t luid[8]) {
    D3DKMT_OPENADAPTERFROMLUID opened{};std::memcpy(&opened.AdapterLuid,luid,8);
    if(D3DKMTOpenAdapterFromLuid(&opened)!=0)return {};
    D3DKMT_ADAPTERADDRESS pci{};D3DKMT_QUERYADAPTERINFO query{};query.hAdapter=opened.hAdapter;query.Type=KMTQAITYPE_ADAPTERADDRESS;query.pPrivateDriverData=&pci;query.PrivateDriverDataSize=sizeof(pci);
    const auto status=D3DKMTQueryAdapterInfo(&query);D3DKMT_CLOSEADAPTER closed{};closed.hAdapter=opened.hAdapter;const auto close_status=D3DKMTCloseAdapter(&closed);
    return status==0&&close_status==0?pci_physical_id(0,pci.BusNumber,pci.DeviceNumber,pci.FunctionNumber):std::string{};
}
}
