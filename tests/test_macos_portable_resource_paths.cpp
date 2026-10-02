/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "internal/resource_paths.hpp"
#include <cassert>
#include <fstream>
int main() {
    namespace fs = std::filesystem;
    const auto assets = lfs::core::getExecutableDir().parent_path() / "share/LichtFeld-Studio/assets";
    fs::create_directories(assets);
    std::ofstream(assets / "bundled.txt") << "bundle";
    fs::create_directories(VISUALIZER_ASSET_PATH);
    std::ofstream(fs::path(VISUALIZER_ASSET_PATH) / "outside.txt") << "outside";
    assert(lfs::vis::getAssetPath("bundled.txt") == assets / "bundled.txt");
    const fs::path configured_python = fs::path(VISUALIZER_ASSET_PATH) / "python3";
    std::ofstream(configured_python) << "external interpreter";
#ifdef LFS_MACOS_PORTABLE_APP
    assert(lfs::core::getEmbeddedPython().empty());
#else
    assert(lfs::core::getEmbeddedPython() == configured_python);
#endif
    const auto bundled_python = lfs::core::getExecutableDir() / "python3";
    std::ofstream(bundled_python) << "bundled interpreter";
    assert(lfs::core::getEmbeddedPython() == bundled_python);
    fs::remove(bundled_python);

#ifdef LFS_MACOS_PORTABLE_APP
    try {
        (void)lfs::vis::getAssetPath("outside.txt");
        return 1;
    } catch (const std::runtime_error& error) {
        assert(std::string(error.what()).find(VISUALIZER_ASSET_PATH) == std::string::npos);
    }
#else
    assert(lfs::vis::getAssetPath("outside.txt") == fs::path(VISUALIZER_ASSET_PATH) / "outside.txt");
#endif
}
