/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/scene.hpp"
#include "gui/panel_registry.hpp"
#include "gui/rmlui/rml_panel_host.hpp"
#include <RmlUi/Core/EventListener.h>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <mutex>
#include <thread>

namespace lfs::vis::gui {
    class CollisionExportPanel final : public IPanel, private Rml::EventListener {
    public:
        explicit CollisionExportPanel(RmlUIManager* manager);
        ~CollisionExportPanel() override;
        void draw(const PanelDrawContext& ctx) override;
        void preload(const PanelDrawContext& ctx) override;
        PanelRenderCapabilities renderCapabilities() const override { return {.direct = true}; }
        PanelDirectRenderResult renderDirect(const PanelDirectRenderRequest&, const PanelDrawContext&) override;
        bool needsAnimationFrame() const override { return busy_ || host_.needsAnimationFrame(); }
        void reloadRmlResources() override;
        void releaseRendererResources() override { host_.releaseRendererResources(); }

    private:
        void ProcessEvent(Rml::Event&) override;
        bool prepare(const PanelDrawContext&);
        void chooseExport(bool voxels);
        void startExport(bool voxels, std::filesystem::path path);
        void showError(const std::string& message);
        void text(const char* id, const std::string& value);
        RmlPanelHost host_;
        Rml::ElementDocument* document_ = nullptr;
        core::Scene* scene_ = nullptr;
        core::Uuid target_;
        std::jthread worker_;
        std::atomic<bool> busy_{false}, done_{false};
        std::mutex mutex_;
        std::string progress_, error_, destination_;
        float progress_value_ = 0;
        std::chrono::steady_clock::time_point started_, next_update_;
    };
} // namespace lfs::vis::gui
