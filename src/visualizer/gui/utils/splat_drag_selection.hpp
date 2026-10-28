/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <optional>
#include <span>
#include <unordered_set>
#include <vector>

namespace lfs::vis::gui {
    enum class SplatDragKind { Splat,
                               Group,
                               Other };
    enum class SplatDragError { None,
                                MissingNode,
                                UnsupportedRoot,
                                UnloadedSplat,
                                Cycle };
    template <class Id>
    struct SplatDragNode {
        SplatDragKind kind;
        bool loaded;
        std::span<const Id> children;
    };
    template <class Id>
    struct SplatDragSelection {
        std::vector<Id> splats;
        SplatDragError error = SplatDragError::None;
        std::optional<Id> failed_node = std::nullopt;
    };
    // Resolve live hierarchy on the UI thread. An iterative walk handles deeply
    // nested groups, including collapsed UI rows, without retaining scene nodes.
    // Explicit unsupported roots reject the whole gesture; non-splat descendants
    // inside a group are traversed but never exported.
    template <class Id, class Lookup>
    SplatDragSelection<Id> collectSplatDragSelection(std::span<const Id> roots, Lookup lookup) {
        SplatDragSelection<Id> result;
        struct Entry {
            Id id;
            bool exit = false;
        };
        std::vector<Entry> pending;
        std::unordered_set<Id> visited, active;
        for (auto it = roots.rbegin(); it != roots.rend(); ++it) {
            const auto node = lookup(*it);
            if (!node)
                return {{}, SplatDragError::MissingNode, *it};
            if (node->kind == SplatDragKind::Other)
                return {{}, SplatDragError::UnsupportedRoot, *it};
            pending.push_back({*it});
        }
        while (!pending.empty()) {
            const auto [id, exit] = pending.back();
            pending.pop_back();
            if (exit) {
                active.erase(id);
                continue;
            }
            if (active.contains(id))
                return {{}, SplatDragError::Cycle, id};
            if (!visited.insert(id).second)
                continue;
            const auto node = lookup(id);
            if (!node)
                return {{}, SplatDragError::MissingNode, id};
            if (node->kind == SplatDragKind::Splat) {
                if (!node->loaded)
                    return {{}, SplatDragError::UnloadedSplat, id};
                result.splats.push_back(id);
            }
            active.insert(id);
            pending.push_back({id, true});
            for (auto it = node->children.rbegin(); it != node->children.rend(); ++it)
                pending.push_back({*it});
        }
        return result;
    }
} // namespace lfs::vis::gui
