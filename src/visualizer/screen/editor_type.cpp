/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "screen/editor_type.hpp"

#include <algorithm>
#include <utility>

namespace lfs::vis::screen {

    void EditorTypeRegistry::add(EditorType type) {
        const auto it = std::find_if(types_.begin(), types_.end(),
                                     [&](const EditorType& existing) { return existing.id == type.id; });
        if (it != types_.end())
            *it = std::move(type);
        else
            types_.push_back(std::move(type));
    }

    void EditorTypeRegistry::setDynamicProvider(DynamicProvider provider, DynamicLister lister) {
        dynamic_provider_ = std::move(provider);
        dynamic_lister_ = std::move(lister);
    }

    std::optional<EditorType> EditorTypeRegistry::find(const std::string_view id) const {
        const auto it = std::find_if(types_.begin(), types_.end(),
                                     [id](const EditorType& type) { return type.id == id; });
        if (it != types_.end())
            return *it;
        if (dynamic_provider_)
            return dynamic_provider_(id);
        return std::nullopt;
    }

    std::vector<EditorType> EditorTypeRegistry::list() const {
        std::vector<EditorType> out = types_;
        if (dynamic_lister_) {
            for (auto& type : dynamic_lister_()) {
                const bool shadowed = std::any_of(out.begin(), out.end(),
                                                  [&](const EditorType& t) { return t.id == type.id; });
                if (!shadowed)
                    out.push_back(std::move(type));
            }
        }
        return out;
    }

    std::unique_ptr<SpaceData> EditorTypeRegistry::createSpace(const std::string_view id) const {
        const auto type = find(id);
        if (!type || !type->create_space)
            return nullptr;
        return type->create_space();
    }

} // namespace lfs::vis::screen
