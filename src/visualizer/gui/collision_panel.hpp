/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/collision.hpp"
#include "core/scene.hpp"
#include "gui/panel_registry.hpp"
#include "gui/rmlui/rml_panel_host.hpp"
#include <RmlUi/Core/EventListener.h>
#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>

namespace lfs::vis::gui {
    class CollisionPanel final : public IPanel, private Rml::EventListener {
    public:
        explicit CollisionPanel(RmlUIManager* manager);
        ~CollisionPanel() override;
        void draw(const PanelDrawContext& ctx) override;
        void preload(const PanelDrawContext& ctx) override;
        PanelRenderCapabilities renderCapabilities() const override { return {.direct = true}; }
        PanelDirectRenderResult renderDirect(const PanelDirectRenderRequest& request, const PanelDrawContext& ctx) override;
        bool needsAnimationFrame() const override { return busy_ || host_.needsAnimationFrame(); }
        void reloadRmlResources() override;
        void releaseRendererResources() override { host_.releaseRendererResources(); }

    private:
        void ProcessEvent(Rml::Event& event) override;
        bool prepare(const PanelDrawContext& ctx);
        void start(bool new_node);
        void buildSurface();
        void updateDisplay();
        void updateSummary();
        void applyPreset();
        void adoptResult();
        void loadSettings(const core::SceneNode& node);
        void writeInputs();
        core::CollisionSettings readInputs() const;
        float number(const char* id) const;
        bool checked(const char* id) const;
        std::string value(const char* id) const;
        void setValue(const char* id, std::string value);
        void status(std::string text);
        RmlPanelHost host_;
        Rml::ElementDocument* document_ = nullptr;
        core::Scene* scene_ = nullptr;
        core::Uuid target_;
        bool training_ = false;
        core::CollisionSettings settings_;
        core::CollisionDisplay display_;
        bool advanced_ = false;
        bool updating_ = false;
        std::string last_stats_;
        std::weak_ptr<const core::CollisionVolume> counted_volume_;
        uint64_t occupied_count_ = 0;
        glm::vec3 world_seed_{0};
        std::jthread worker_;
        std::atomic<bool> busy_{false};
        std::atomic<bool> done_{false};
        std::mutex mutex_;
        std::string progress_;
        float progress_value_ = 0;
        int progress_step_ = 0, progress_total_ = 1;
        std::chrono::steady_clock::time_point job_started_;
        std::string error_;
        struct Result {
            std::shared_ptr<const core::CollisionVolume> volume;
            std::shared_ptr<core::MeshData> mesh;
            glm::mat4 world_transform{1};
            std::vector<core::Uuid> sources;
            core::Uuid target;
            uint64_t scene_generation = 0;
            uint64_t graph_epoch = 0;
            core::CollisionDisplay display;
            std::string stats;
        };
        std::optional<Result> result_;
    };
} // namespace lfs::vis::gui
