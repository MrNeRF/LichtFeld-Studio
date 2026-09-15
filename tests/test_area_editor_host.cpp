/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/area_editor_host.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

    using lfs::vis::SplitAxis;
    using lfs::vis::ViewId;
    using lfs::vis::ViewportWorkspace;
    using lfs::vis::gui::AreaEditorHost;
    using lfs::vis::gui::IPanel;
    using lfs::vis::gui::PanelInfo;
    using lfs::vis::gui::PanelRegistry;
    using lfs::vis::gui::PanelSpace;

    class AreaPanel final : public IPanel {
    public:
        inline static std::vector<std::shared_ptr<AreaPanel>> instances;

        std::string instance_id;
        std::string applied_chrome;
        bool available = true;
        bool fail = false;
        int preloads = 0;
        bool poll(const lfs::vis::gui::PanelDrawContext&) override { return available; }
        lfs::vis::gui::PanelDirectRenderResult renderDirect(
            const lfs::vis::gui::PanelDirectRenderRequest& request,
            const lfs::vis::gui::PanelDrawContext&) override {
            EXPECT_EQ(request.mode, lfs::vis::gui::PanelDirectRenderMode::Preload);
            EXPECT_EQ(request.input, nullptr);
            if (fail)
                throw std::runtime_error("bad plugin");
            ++preloads;
            return {.handled = true, .height = request.height};
        }

        void draw(const lfs::vis::gui::PanelDrawContext&) override {}
        [[nodiscard]] bool supportsAreaInstances() const override { return true; }
        [[nodiscard]] std::shared_ptr<IPanel> createAreaInstance(
            const std::string_view id) const override {
            auto instance = std::make_shared<AreaPanel>();
            instance->instance_id = id;
            instances.push_back(instance);
            return instance;
        }
        [[nodiscard]] std::string captureChromeJson() const override {
            return std::string("{\"instance\":\"") + instance_id + "\"}";
        }
        void applyChromeJson(const std::string_view json) override {
            applied_chrome = json;
        }
    };

    class AreaEditorHostTest : public ::testing::Test {
    protected:
        void SetUp() override {
            PanelRegistry::instance().unregister_all_non_native();
            AreaPanel::instances.clear();

            PanelInfo info;
            info.id = "test.area";
            info.label = "Test area";
            info.space = PanelSpace::MainPanelTab;
            info.is_native = false;
            info.panel = std::make_shared<AreaPanel>();
            ASSERT_TRUE(PanelRegistry::instance().register_panel(std::move(info)));
        }

        void TearDown() override {
            PanelRegistry::instance().unregister_all_non_native();
            AreaPanel::instances.clear();
        }

        static auto snapshot(const ViewportWorkspace& workspace) {
            return workspace.snapshot({0, 0, 800, 600});
        }
    };

    TEST_F(AreaEditorHostTest, PreSceneSyncUsesVisibleInstancesAndRespectsPanelAvailability) {
        ViewportWorkspace workspace;
        ASSERT_TRUE(workspace.setAreaEditor(workspace.primaryView(), "test.area"));
        const auto second = workspace.split(workspace.primaryView(), SplitAxis::Horizontal);
        ASSERT_TRUE(second);
        AreaEditorHost host(PanelRegistry::instance(), workspace, nullptr);
        host.preloadVisiblePanels(snapshot(workspace), {});
        ASSERT_EQ(AreaPanel::instances.size(), 2u);
        EXPECT_EQ(AreaPanel::instances[0]->preloads, 1);
        EXPECT_EQ(AreaPanel::instances[1]->preloads, 1);

        lfs::vis::gui::PanelDrawContext suppressed;
        suppressed.suppress_non_native_panels = true;
        host.preloadVisiblePanels(snapshot(workspace), suppressed);
        EXPECT_EQ(AreaPanel::instances[0]->preloads, 1);
        AreaPanel::instances[0]->available = false;
        host.preloadVisiblePanels(snapshot(workspace), {});
        EXPECT_EQ(AreaPanel::instances[0]->preloads, 1);
        EXPECT_EQ(AreaPanel::instances[1]->preloads, 2);

        AreaPanel::instances[0]->available = true;
        AreaPanel::instances[0]->fail = true;
        EXPECT_NO_THROW(host.preloadVisiblePanels(snapshot(workspace), {}));
        EXPECT_EQ(AreaPanel::instances[1]->preloads, 3);
        ASSERT_TRUE(workspace.maximize(*second));
        host.preloadVisiblePanels(snapshot(workspace), {});
        EXPECT_EQ(AreaPanel::instances[1]->preloads, 4);
    }

    TEST_F(AreaEditorHostTest, CreatesIndependentInstancesAndRestoresChrome) {
        PanelRegistry& registry = PanelRegistry::instance();
        ViewportWorkspace workspace;
        const ViewId id = workspace.primaryView();
        ASSERT_TRUE(workspace.setAreaEditor(id, "test.area"));

        AreaEditorHost host(registry, workspace, nullptr);
        host.synchronize(snapshot(workspace));
        ASSERT_EQ(AreaPanel::instances.size(), 1u);
        const std::string first_id = AreaPanel::instances.front()->instance_id;
        EXPECT_NE(first_id, "");
        EXPECT_EQ(host.activeAreaCount(), 1u);

        host.captureChrome();
        ASSERT_NE(workspace.findView(id), nullptr);
        EXPECT_EQ(workspace.findView(id)->chrome.at("test.area"),
                  std::string("{\"instance\":\"") + first_id + "\"}");

        ASSERT_TRUE(workspace.setAreaEditor(id, "viewport"));
        host.synchronize(snapshot(workspace));
        EXPECT_EQ(host.activeAreaCount(), 1u);
        EXPECT_EQ(host.retiredAreaCount(), 1u);

        ASSERT_TRUE(workspace.setAreaEditor(id, "test.area"));
        host.synchronize(snapshot(workspace));
        ASSERT_EQ(AreaPanel::instances.size(), 2u);
        EXPECT_NE(AreaPanel::instances[1]->instance_id, first_id);
        EXPECT_EQ(AreaPanel::instances[1]->applied_chrome,
                  std::string("{\"instance\":\"") + first_id + "\"}");

        host.retireAll();
        workspace.findView(id)->chrome["test.area"] = "{\"restored\":true}";
        host.synchronize(snapshot(workspace));
        ASSERT_EQ(AreaPanel::instances.size(), 3u);
        EXPECT_EQ(AreaPanel::instances[2]->applied_chrome, "{\"restored\":true}");
    }

    TEST_F(AreaEditorHostTest, RetainsHiddenLiveAreasAndDefersRetirement) {
        PanelRegistry& registry = PanelRegistry::instance();
        ViewportWorkspace workspace({800, 600});
        const ViewId first = workspace.primaryView();
        const auto second_result = workspace.split(first, SplitAxis::Horizontal);
        ASSERT_TRUE(second_result);
        const ViewId second = *second_result;
        ASSERT_TRUE(workspace.setAreaEditor(first, "test.area"));

        AreaEditorHost host(registry, workspace, nullptr);
        host.synchronize(snapshot(workspace));
        ASSERT_EQ(host.activeAreaCount(), 2u);
        ASSERT_TRUE(workspace.maximize(second));
        const auto maximized = snapshot(workspace);
        ASSERT_EQ(maximized.areas.size(), 1u);
        ASSERT_EQ(maximized.live_ids.size(), 2u);

        host.synchronize(maximized);
        EXPECT_EQ(host.activeAreaCount(), 2u);
        EXPECT_EQ(host.retiredAreaCount(), 0u);

        host.retireAll();
        EXPECT_EQ(host.activeAreaCount(), 0u);
        // The hidden viewport area has no retained panel when the RML manager
        // is null, so only the fake editor instance needs retirement here.
        EXPECT_EQ(host.retiredAreaCount(), 1u);
        host.collectRetired();
        EXPECT_EQ(host.retiredAreaCount(), 0u);
    }

    TEST_F(AreaEditorHostTest, SameEditorInTwoAreasGetsIndependentInstances) {
        PanelRegistry& registry = PanelRegistry::instance();
        ViewportWorkspace workspace({800, 600});
        const ViewId first = workspace.primaryView();
        const auto second_result = workspace.split(first, SplitAxis::Horizontal);
        ASSERT_TRUE(second_result);
        const ViewId second = *second_result;
        ASSERT_TRUE(workspace.setAreaEditor(first, "test.area"));
        ASSERT_TRUE(workspace.setAreaEditor(second, "test.area"));

        AreaEditorHost host(registry, workspace, nullptr);
        host.synchronize(snapshot(workspace));
        ASSERT_EQ(AreaPanel::instances.size(), 2u);
        EXPECT_NE(AreaPanel::instances[0]->instance_id,
                  AreaPanel::instances[1]->instance_id);
        AreaPanel::instances[0]->applied_chrome = "first";
        AreaPanel::instances[1]->applied_chrome = "second";
        EXPECT_EQ(AreaPanel::instances[0]->applied_chrome, "first");
        EXPECT_EQ(AreaPanel::instances[1]->applied_chrome, "second");
    }

    TEST_F(AreaEditorHostTest, RetriesUnavailableFactoryAfterRegistryChanges) {
        ViewportWorkspace workspace;
        const ViewId id = workspace.primaryView();
        ASSERT_TRUE(workspace.setAreaEditor(id, "late.area"));
        AreaEditorHost host(PanelRegistry::instance(), workspace, nullptr);
        host.synchronize(snapshot(workspace));
        EXPECT_TRUE(AreaPanel::instances.empty());

        PanelInfo late;
        late.id = "late.area";
        late.label = "Late area";
        late.space = PanelSpace::MainPanelTab;
        late.is_native = false;
        late.panel = std::make_shared<AreaPanel>();
        ASSERT_TRUE(PanelRegistry::instance().register_panel(std::move(late)));

        host.synchronize(snapshot(workspace));
        ASSERT_EQ(AreaPanel::instances.size(), 1u);
        EXPECT_EQ(host.activeAreaCount(), 1u);
    }

    TEST_F(AreaEditorHostTest, DefaultActionsUpdateWorkspaceState) {
        PanelRegistry& registry = PanelRegistry::instance();
        ViewportWorkspace workspace;
        const ViewId id = workspace.primaryView();
        AreaEditorHost host(registry, workspace, nullptr);

        host.handleAction(id, lfs::vis::gui::AreaEditorAction::ToggleProjection, {});
        ASSERT_NE(workspace.findView(id), nullptr);
        EXPECT_TRUE(workspace.findView(id)->projection.orthographic);
        host.handleAction(id, lfs::vis::gui::AreaEditorAction::SetEditor, "test.area");
        EXPECT_EQ(workspace.findView(id)->editor_id, "test.area");
    }

} // namespace
