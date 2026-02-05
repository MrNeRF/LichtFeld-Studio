/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "operator/poll_dependency.hpp"

#include <core/export.hpp>

#include <cassert>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace lfs::vis {
    class Scene;
}

namespace lfs::vis::gui {

    struct UIContext;
    struct ViewportLayout;

    enum class PanelSpace : uint8_t {
        SidePanel,
        Floating,
        ViewportOverlay,
        Dockable,
        MainPanelTab,
        SceneHeader,
        StatusBar
    };

    enum class PanelOption : uint32_t {
        DEFAULT_CLOSED = 1 << 0,
        HIDE_HEADER = 1 << 1,
        SELF_MANAGED = 1 << 2,
    };

    using PollDependency = lfs::vis::op::PollDependency;

    struct PanelDrawContext {
        const UIContext* ui = nullptr;
        const ViewportLayout* viewport = nullptr;
        Scene* scene = nullptr;
        bool ui_hidden = false;
        uint64_t scene_generation = 0;
        bool has_selection = false;
        bool is_training = false;
    };

    class IPanel {
    public:
        virtual ~IPanel() = default;
        virtual void draw(const PanelDrawContext& ctx) = 0;
        virtual bool poll(const PanelDrawContext& ctx) {
            (void)ctx;
            return true;
        }
    };

    struct PanelInfo {
        std::shared_ptr<IPanel> panel;
        std::string label;
        std::string idname;
        PanelSpace space = PanelSpace::Floating;
        int order = 100;
        bool enabled = true;
        uint32_t options = 0;
        PollDependency poll_deps = PollDependency::ALL;
        bool is_native = true;
        int consecutive_errors = 0;
        bool error_disabled = false;
        static constexpr int MAX_CONSECUTIVE_ERRORS = 3;

        bool has_option(PanelOption opt) const {
            return (options & static_cast<uint32_t>(opt)) != 0;
        }
    };

    struct PanelSummary {
        std::string label;
        std::string idname;
        PanelSpace space;
        int order;
        bool enabled;
    };

    struct PanelSnapshot {
        size_t index;
        IPanel* panel;
        std::string label;
        std::string idname;
        uint32_t options;
        bool is_native;
        PollDependency poll_deps;

        bool has_option(PanelOption opt) const {
            return (options & static_cast<uint32_t>(opt)) != 0;
        }
    };

    struct PollCacheEntry {
        bool result;
        uint64_t scene_generation;
        bool has_selection;
        bool is_training;
        PollDependency deps;
    };

    class LFS_VIS_API PanelRegistry {
    public:
        static PanelRegistry& instance();

        void register_panel(PanelInfo info);
        void unregister_panel(const std::string& idname);
        void unregister_all_non_native();

        void draw_panels(PanelSpace space, const PanelDrawContext& ctx);
        void draw_single_panel(const std::string& idname, const PanelDrawContext& ctx);
        bool has_panels(PanelSpace space) const;

        std::vector<PanelSummary> get_panels_for_space(PanelSpace space);
        std::vector<std::string> get_panel_names(PanelSpace space) const;
        std::optional<PanelSummary> get_panel(const std::string& idname);
        void set_panel_enabled(const std::string& idname, bool enabled);
        bool is_panel_enabled(const std::string& idname) const;
        bool set_panel_label(const std::string& idname, const std::string& new_label);
        bool set_panel_order(const std::string& idname, int new_order);
        bool set_panel_space(const std::string& idname, PanelSpace new_space);
        void invalidate_poll_cache(PollDependency changed = PollDependency::ALL);

    private:
        PanelRegistry() = default;
        ~PanelRegistry() = default;
        PanelRegistry(const PanelRegistry&) = delete;
        PanelRegistry& operator=(const PanelRegistry&) = delete;

        bool check_poll(const PanelSnapshot& snap, const PanelDrawContext& ctx);

        mutable std::mutex mutex_;
        mutable std::mutex poll_mutex_;
        std::vector<PanelInfo> panels_;
        mutable std::unordered_map<std::string, PollCacheEntry> poll_cache_;
    };

} // namespace lfs::vis::gui
