/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "rendering/view_source.hpp"
#include "screen/screen.hpp"
#include "screen/view3d_space.hpp"

#include <core/export.hpp>
#include <mutex>
#include <utility>

namespace lfs::vis::screen {

    // Owns the editor types and the window's screen. The GUI thread reads and
    // edits spaces directly (camera motion every frame, as the single
    // viewport did). Structural changes (areas added, removed, switched) go
    // through edit(), and other threads read through read(); both hold the
    // lock, so a thread never sees a space being destroyed under it.
    class LFS_VIS_API ScreenService final : public ViewSource {
    public:
        ScreenService();

        ScreenService(const ScreenService&) = delete;
        ScreenService& operator=(const ScreenService&) = delete;

        [[nodiscard]] EditorTypeRegistry& editorTypes() { return editor_types_; }
        [[nodiscard]] const EditorTypeRegistry& editorTypes() const { return editor_types_; }

        // GUI thread only.
        [[nodiscard]] Screen& screen() { return screen_; }
        [[nodiscard]] const Screen& screen() const { return screen_; }

        template <typename Fn>
        decltype(auto) edit(Fn&& fn) {
            std::lock_guard lock(mutex_);
            return std::forward<Fn>(fn)(screen_);
        }

        template <typename Fn>
        decltype(auto) read(Fn&& fn) const {
            std::lock_guard lock(mutex_);
            return std::forward<Fn>(fn)(std::as_const(screen_));
        }

        void replace(Screen screen);
        void resetToDefault();

        // The active view's space. Never null on a valid screen.
        [[nodiscard]] View3DSpace& activeView3D();
        [[nodiscard]] const View3DSpace& activeView3D() const;
        [[nodiscard]] View3DSpace* view3D(ViewId view) const { return screen_.view(AreaId{view}); }

        // ViewSource
        [[nodiscard]] ViewId activeView() const override;
        [[nodiscard]] std::optional<ViewSettings> viewSettings(ViewId view) const override;
        bool editViewSettings(ViewId view, const std::function<void(ViewSettings&)>& edit) override;

    private:
        EditorTypeRegistry editor_types_;
        Screen screen_;
        mutable std::recursive_mutex mutex_;
    };

} // namespace lfs::vis::screen
