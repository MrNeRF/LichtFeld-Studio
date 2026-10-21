/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
// Host-only adapter lookup. Keep Windows SDK headers out of HIP device code.
// clang-format off
#include <windows.h>
#include <d3dkmthk.h>
// clang-format on
#include <cstdint>
#include <cstring>

extern "C" int lfs_hip_adapter_pci(const uint8_t luid[8], uint32_t address[3]) {
    D3DKMT_OPENADAPTERFROMLUID open{};
    std::memcpy(&open.AdapterLuid, luid, sizeof(open.AdapterLuid));
    if (D3DKMTOpenAdapterFromLuid(&open) != 0)
        return 0;
    D3DKMT_ADAPTERADDRESS pci{};
    D3DKMT_QUERYADAPTERINFO query{};
    query.hAdapter = open.hAdapter;
    query.Type = KMTQAITYPE_ADAPTERADDRESS;
    query.pPrivateDriverData = &pci;
    query.PrivateDriverDataSize = sizeof(pci);
    const auto queried = D3DKMTQueryAdapterInfo(&query);
    D3DKMT_CLOSEADAPTER close{};
    close.hAdapter = open.hAdapter;
    const auto closed = D3DKMTCloseAdapter(&close);
    if (queried != 0 || closed != 0)
        return 0;
    address[0] = pci.BusNumber;
    address[1] = pci.DeviceNumber;
    address[2] = pci.FunctionNumber;
    return 1;
}
