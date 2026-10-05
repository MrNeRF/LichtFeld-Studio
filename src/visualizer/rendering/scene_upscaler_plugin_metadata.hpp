/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <locale>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <sstream>
#include <string>
#include <vector>

namespace lfs::vis {
    // Metadata is read without executing the module, including in safe mode.
    // The bound prevents a malformed file from allocating unbounded memory.
    [[nodiscard]] inline std::optional<std::string> readSceneUpscalerProviderId(
        const std::filesystem::path& file) {
        std::ifstream input(file, std::ios::binary);
        if (!input)
            return std::nullopt;
        std::array<char, 131> bytes{};
        input.read(bytes.data(), bytes.size());
        std::string id(bytes.data(), static_cast<std::size_t>(input.gcount()));
        if (id.ends_with('\n'))
            id.pop_back();
        if (id.ends_with('\r'))
            id.pop_back();
        if (id.empty() || id.size() > 128 ||
            !std::ranges::all_of(id, [](const unsigned char c) {
                return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '-' || c == '_';
            }) ||
            id == "native" || id == "spatial" || id == "temporal")
            return std::nullopt;
        return id;
    }

    struct SceneUpscalerProviderFolder {
        std::string id;
        std::filesystem::path directory;
        std::string library;
        std::array<float, 3> input_scales;
    };

    [[nodiscard]] inline std::string sceneUpscalerPluginFilename(const std::string_view stem) {
#if defined(_WIN32)
        return std::string(stem) + ".dll";
#elif defined(__APPLE__)
        return "lib" + std::string(stem) + ".dylib";
#else
        return "lib" + std::string(stem) + ".so";
#endif
    }

    [[nodiscard]] inline std::optional<std::string> readSceneUpscalerModuleStem(const std::filesystem::path& file) {
        std::ifstream input(file, std::ios::binary);
        if (!input)
            return std::nullopt;
        std::array<char, 131> bytes{};
        input.read(bytes.data(), bytes.size());
        std::string stem(bytes.data(), static_cast<std::size_t>(input.gcount()));
        if (stem.ends_with('\n'))
            stem.pop_back();
        if (stem.ends_with('\r'))
            stem.pop_back();
        if (stem.empty() || stem.size() > 128 || !std::ranges::all_of(stem, [](const unsigned char c) {
                return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                       (c >= '0' && c <= '9') || c == '-' || c == '_';
            }))
            return std::nullopt;
        return stem;
    }

    [[nodiscard]] inline std::optional<std::array<float, 3>> readSceneUpscalerPresetScales(const std::filesystem::path& file) {
        std::ifstream input(file, std::ios::binary);
        if (!input)
            return std::nullopt;
        std::array<char, 257> bytes{};
        input.read(bytes.data(), bytes.size());
        if (input.gcount() > 256)
            return std::nullopt;
        std::istringstream values(std::string(bytes.data(), static_cast<std::size_t>(input.gcount())));
        values.imbue(std::locale::classic());
        std::array<float, 3> result{};
        for (auto& scale : result) {
            if (!(values >> scale) || !std::isfinite(scale) || scale <= 0.0f || scale > 1.0f)
                return std::nullopt;
        }
        values >> std::ws;
        if (!values.eof())
            return std::nullopt;
        return result;
    }

    [[nodiscard]] inline std::vector<SceneUpscalerProviderFolder> discoverSceneUpscalerProviderFolders(
        const std::span<const std::filesystem::path> roots) {
        std::map<std::string, SceneUpscalerProviderFolder> providers;
        std::set<std::string> duplicates;
        std::set<std::filesystem::path> directories;
        for (const auto& root : roots) {
            std::error_code error;
            const auto folder = root / "scene_upscalers";
            auto entries = std::filesystem::directory_iterator(folder, error);
            if (error)
                continue;
            const std::filesystem::directory_iterator end;
            for (; entries != end; entries.increment(error)) {
                if (error)
                    break;
                const auto& entry = *entries;
                if (!entry.is_directory(error) || error || entry.is_symlink(error) || error)
                    continue;
                const auto directory = entry.path().filename();
                auto library = readSceneUpscalerModuleStem(entry.path() / "provider-module.txt");
                if (!library || !std::filesystem::is_regular_file(entry.path() / sceneUpscalerPluginFilename(*library), error) || error)
                    continue;
                if (!directories.insert(directory).second)
                    continue;
                auto id = readSceneUpscalerProviderId(entry.path() / "provider-id.txt");
                auto scales = readSceneUpscalerPresetScales(entry.path() / "provider-presets.txt");
                if (!id || !scales)
                    continue;
                auto [existing, inserted] = providers.emplace(*id, SceneUpscalerProviderFolder{*id, directory, *library, *scales});
                if (!inserted && existing->second.directory != directory)
                    duplicates.insert(*id);
            }
        }
        std::vector<SceneUpscalerProviderFolder> result;
        for (const auto& [id, provider] : providers) {
            if (!duplicates.contains(id))
                result.push_back(provider);
        }
        return result;
    }
} // namespace lfs::vis
