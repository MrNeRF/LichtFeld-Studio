/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "screen/screen_service.hpp"

#include <cassert>

namespace lfs::vis::screen {

    namespace {
        EditorTypeRegistry makeRegistry() {
            EditorTypeRegistry registry;
            registerBuiltinEditorTypes(registry);
            return registry;
        }
    } // namespace

    ScreenService::ScreenService()
        : editor_types_(makeRegistry()),
          screen_(Screen::makeDefault(editor_types_)) {}

    void ScreenService::replace(Screen screen) {
        std::lock_guard lock(mutex_);
        assert(&screen.registry() == &editor_types_);
        screen_ = std::move(screen);
        ++epoch_;
    }

    void ScreenService::resetToDefault() {
        std::lock_guard lock(mutex_);
        screen_ = Screen::makeDefault(editor_types_);
        ++epoch_;
    }

    View3DSpace& ScreenService::activeView3D() {
        auto* view = screen_.view(screen_.activeView());
        assert(view);
        return *view;
    }

    const View3DSpace& ScreenService::activeView3D() const {
        const auto* view = screen_.view(screen_.activeView());
        assert(view);
        return *view;
    }

    ViewId ScreenService::activeView() const {
        std::lock_guard lock(mutex_);
        return screen_.activeView().value;
    }

    std::optional<ViewSettings> ScreenService::viewSettings(const ViewId view) const {
        std::lock_guard lock(mutex_);
        if (const auto* area = screen_.area(AreaId{view})) {
            if (const auto* space = dynamic_cast<const View3DSpace*>(area->space(editors::kView3D)))
                return space->settings;
        }
        return std::nullopt;
    }

    bool ScreenService::editViewSettings(const ViewId view, const std::function<void(ViewSettings&)>& edit) {
        std::lock_guard lock(mutex_);
        auto* area = screen_.area(AreaId{view});
        auto* space = area ? dynamic_cast<View3DSpace*>(area->space(editors::kView3D)) : nullptr;
        if (!space)
            return false;
        edit(space->settings);
        sanitizeDepthViewSettings(space->settings);
        if (splitViewEnabled(space->settings.split_view_mode)) {
            for (const auto id : screen_.areas()) {
                if (id.value != view) {
                    if (auto* other = dynamic_cast<View3DSpace*>(screen_.area(id)->space(editors::kView3D)))
                        other->settings.split_view_mode = SplitViewMode::Disabled;
                }
            }
        }
        return true;
    }

} // namespace lfs::vis::screen
