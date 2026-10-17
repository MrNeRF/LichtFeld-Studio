/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include <filesystem>
#include <fstream>
#include <iostream>
#ifdef _WIN32
#include <windows.h>
#endif
#include <psapi.h>
namespace lfs::vis::gui {
    int nativeSelfDropContract();
    int nativeDragDataContract(const std::filesystem::path& path);
} // namespace lfs::vis::gui
int main() {
    if (auto result = lfs::vis::gui::nativeSelfDropContract())
        return result;
    const auto path = std::filesystem::temp_directory_path() / L"lichtfeld-COM-contratto-\u00e8.ply";
    std::ofstream(path) << "fixture";
    // Warm the Shell/COM allocators before measuring repeated object lifetimes.
    if (auto result = lfs::vis::gui::nativeDragDataContract(path))
        return result;
    const auto cursors_before = GetGuiResources(GetCurrentProcess(), GR_USEROBJECTS);
    const auto gdi_before = GetGuiResources(GetCurrentProcess(), GR_GDIOBJECTS);
    DWORD before = 0, after = 0;
    PROCESS_MEMORY_COUNTERS_EX first{}, last{};
    if (!GetProcessHandleCount(GetCurrentProcess(), &before))
        return 11;
    if (!GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&first), sizeof(first)))
        return 11;
    for (int i = 0; i < 5; ++i)
        if (auto result = lfs::vis::gui::nativeDragDataContract(path))
            return result;
    if (!GetProcessHandleCount(GetCurrentProcess(), &after))
        return 11;
    if (!GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&last), sizeof(last)))
        return 11;
    std::filesystem::remove(path);
    if (GetGuiResources(GetCurrentProcess(), GR_USEROBJECTS) > cursors_before + 2 || GetGuiResources(GetCurrentProcess(), GR_GDIOBJECTS) > gdi_before + 2)
        return 12;
    if (after > before + 2 || last.PrivateUsage > first.PrivateUsage + 8 * 1024 * 1024)
        return 10;
    std::cout << "PASS 6000 COM cycles: delayed provider, enumeration, independent storage, cancellation, failure; handles " << before << " -> " << after << "; private bytes " << first.PrivateUsage << " -> " << last.PrivateUsage << '\n';
}
