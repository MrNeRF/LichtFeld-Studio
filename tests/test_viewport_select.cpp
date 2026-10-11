/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/rmlui/elements/viewport_select.hpp"
#include <RmlUi/Core.h>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <gtest/gtest.h>
#include <iostream>

namespace {
    class StubRenderInterface final : public Rml::RenderInterface {
    public:
        Rml::CompiledGeometryHandle CompileGeometry(Rml::Span<const Rml::Vertex>, Rml::Span<const int>) override {
            return 1;
        }
        void RenderGeometry(Rml::CompiledGeometryHandle, Rml::Vector2f, Rml::TextureHandle) override {}
        void ReleaseGeometry(Rml::CompiledGeometryHandle) override {}
        Rml::TextureHandle LoadTexture(Rml::Vector2i& dimensions, const Rml::String&) override {
            dimensions = {16, 16};
            return 1;
        }
        Rml::TextureHandle GenerateTexture(Rml::Span<const Rml::byte>, Rml::Vector2i) override { return 1; }
        void ReleaseTexture(Rml::TextureHandle) override {}
        void EnableScissorRegion(bool) override {}
        void SetScissorRegion(Rml::Rectanglei) override {}
    };

    class ViewportSelectTest : public ::testing::Test {
    protected:
        void SetUp() override {
            ASSERT_TRUE(Rml::Initialise());
            ASSERT_TRUE(Rml::LoadFontFace((std::filesystem::path(PROJECT_ROOT_PATH) /
                                           "src/visualizer/gui/assets/fonts/Inter-Regular.ttf")
                                              .string()));
            Rml::Factory::RegisterElementInstancer("select", &instancer_);
            context_ = Rml::CreateContext("select_test", {1218, 1006}, &renderer_);
            ASSERT_NE(context_, nullptr);
        }
        void TearDown() override {
            Rml::RemoveContext("select_test");
            Rml::Shutdown();
        }
        void load(float top = 817, int count = 5) {
            std::string options;
            for (int i = 0; i < count; ++i)
                options += "<option value='" + std::to_string(i) + "'>Option " + std::to_string(i) + "</option>";
            document_ = context_->LoadDocumentFromMemory(
                "<rml><head><link type='text/rcss' href='components.rcss'/><style>"
                "body {font-family:Inter; width:100%;height:100%;}"
                "select {position:absolute;left:143px;top:" +
                    std::to_string(top) + "px;width:180px;height:24px;}"
                                          "option {line-height:22px;}</style></head><body><select id='select'>" +
                    options + "</select></body></rml>",
                (std::filesystem::path(PROJECT_ROOT_PATH) / "src/visualizer/gui/rmlui/resources/test.rml").string());
            ASSERT_NE(document_, nullptr);
            document_->Show();
            context_->Update();
            select_ = dynamic_cast<Rml::ElementFormControlSelect*>(document_->GetElementById("select"));
            ASSERT_NE(select_, nullptr);
            render();
        }
        void render(float top = 38, float bottom = 938) {
            lfs::vis::gui::SelectViewportScope viewport(top, bottom);
            context_->Update();
            context_->Render();
        }
        Rml::Element* box() { return select_->GetOption(0)->GetParentNode(); }
        void click(Rml::Element* element) {
            const auto p = element->GetAbsoluteOffset(Rml::BoxArea::Border);
            context_->ProcessMouseMove(static_cast<int>(p.x + 8), static_cast<int>(p.y + element->GetOffsetHeight() / 2), 0);
            context_->ProcessMouseButtonDown(0, 0);
            context_->ProcessMouseButtonUp(0, 0);
            render();
        }
        StubRenderInterface renderer_;
        Rml::ElementInstancerGeneric<lfs::vis::gui::ViewportSelect> instancer_;
        Rml::Context* context_ = nullptr;
        Rml::ElementDocument* document_ = nullptr;
        Rml::ElementFormControlSelect* select_ = nullptr;
    };

    TEST_F(ViewportSelectTest, BottomEdgeFlipsBeforeFirstRenderAndEveryOptionIsClickable) {
        load();
        for (int i = 0; i < 5; ++i) {
            select_->ShowSelectBox();
            render();
            const auto pos = box()->GetAbsoluteOffset(Rml::BoxArea::Border);
            ASSERT_GE(pos.y, 38);
            ASSERT_LE(pos.y + box()->GetOffsetHeight(), select_->GetAbsoluteOffset(Rml::BoxArea::Border).y);
            for (int frame = 0; frame < 5; ++frame) {
                render();
                EXPECT_EQ(box()->GetAbsoluteOffset(Rml::BoxArea::Border), pos);
            }
            click(select_->GetOption(i));
            EXPECT_EQ(select_->GetSelection(), i);
            EXPECT_FALSE(select_->IsSelectBoxVisible());
        }
    }

    TEST_F(ViewportSelectTest, DropdownWithRoomBelowKeepsNativeGeometry) {
        load(500);
        select_->ShowSelectBox();
        context_->Update();
        context_->Render();
        const auto pos = box()->GetAbsoluteOffset(Rml::BoxArea::Border);
        const auto size = box()->GetBox().GetSize(Rml::BoxArea::Border);
        render();
        EXPECT_EQ(box()->GetAbsoluteOffset(Rml::BoxArea::Border), pos);
        EXPECT_EQ(box()->GetBox().GetSize(Rml::BoxArea::Border), size);
        EXPECT_GT(pos.y, select_->GetAbsoluteOffset(Rml::BoxArea::Border).y);
        click(select_->GetOption(4));
        EXPECT_EQ(select_->GetSelection(), 4);
    }

    TEST_F(ViewportSelectTest, TallWindowKeepsDownwardOpening) {
        load();
        select_->ShowSelectBox();
        render(0, 1006);
        EXPECT_GT(box()->GetAbsoluteOffset(Rml::BoxArea::Border).y, select_->GetAbsoluteOffset(Rml::BoxArea::Border).y);
        click(select_->GetOption(4));
        EXPECT_EQ(select_->GetSelection(), 4);
    }

    TEST_F(ViewportSelectTest, ClippedTopOpensBelow) {
        load(300);
        select_->ShowSelectBox();
        render(280, 500);
        EXPECT_GE(box()->GetAbsoluteOffset(Rml::BoxArea::Border).y, 280);
        EXPECT_LE(box()->GetAbsoluteOffset(Rml::BoxArea::Border).y + box()->GetOffsetHeight(), 500);
    }

    TEST_F(ViewportSelectTest, OversizedListFitsAndCanScrollToLastOption) {
        load(400, 40);
        select_->ShowSelectBox();
        render(280, 600);
        ASSERT_GE(box()->GetAbsoluteOffset(Rml::BoxArea::Border).y, 280);
        ASSERT_LE(box()->GetAbsoluteOffset(Rml::BoxArea::Border).y + box()->GetOffsetHeight(), 600);
        box()->SetScrollTop(box()->GetScrollHeight());
        render(280, 600);
        auto* last = select_->GetOption(39);
        EXPECT_GE(last->GetAbsoluteOffset(Rml::BoxArea::Border).y, 280);
        EXPECT_LE(last->GetAbsoluteOffset(Rml::BoxArea::Border).y + last->GetOffsetHeight(), 600);
    }

    TEST_F(ViewportSelectTest, KeyboardSelectionAndCancelStillWork) {
        load();
        select_->SetSelection(2);
        select_->Focus();
        select_->ShowSelectBox();
        render();
        context_->ProcessKeyDown(Rml::Input::KI_DOWN, 0);
        context_->ProcessKeyDown(Rml::Input::KI_DOWN, 0);
        context_->ProcessKeyDown(Rml::Input::KI_RETURN, 0);
        render();
        EXPECT_EQ(select_->GetSelection(), 4);
        EXPECT_FALSE(select_->IsSelectBoxVisible());
        select_->ShowSelectBox();
        render();
        context_->ProcessKeyDown(Rml::Input::KI_UP, 0);
        select_->CancelSelectBox();
        EXPECT_EQ(select_->GetSelection(), 4);
    }

    TEST_F(ViewportSelectTest, NativeUpwardOpeningKeepsGeometry) {
        load(950);
        select_->ShowSelectBox();
        context_->Update();
        context_->Render();
        const auto pos = box()->GetAbsoluteOffset(Rml::BoxArea::Border);
        const auto size = box()->GetBox().GetSize(Rml::BoxArea::Border);
        render(0, 1006);
        EXPECT_EQ(box()->GetAbsoluteOffset(Rml::BoxArea::Border), pos);
        EXPECT_EQ(box()->GetBox().GetSize(Rml::BoxArea::Border), size);
    }

    TEST_F(ViewportSelectTest, ResizingRestoresOriginalMaximumHeight) {
        load(400, 40);
        box()->SetProperty("max-height", "500px");
        select_->ShowSelectBox();
        render(280, 600);
        ASSERT_LT(box()->GetOffsetHeight(), 500);
        render(0, 1006);
        EXPECT_EQ(box()->GetLocalProperty("max-height")->Get<float>(), 500);
        EXPECT_GE(box()->GetOffsetHeight(), 500);
        select_->HideSelectBox();
        render();
        select_->ShowSelectBox();
        render(0, 1006);
        EXPECT_GE(box()->GetOffsetHeight(), 500);
    }

    TEST_F(ViewportSelectTest, BorderBoxListStaysInsideVisibleArea) {
        load(400, 40);
        box()->SetProperty("box-sizing", "border-box");
        select_->ShowSelectBox();
        render(280, 600);
        EXPECT_GE(box()->GetAbsoluteOffset(Rml::BoxArea::Border).y, 280);
        EXPECT_LE(box()->GetAbsoluteOffset(Rml::BoxArea::Border).y + box()->GetOffsetHeight(), 600);
    }

    TEST_F(ViewportSelectTest, ScaledControlStaysInsideVisibleArea) {
        context_->SetDimensions({1827, 1509});
        context_->SetDensityIndependentPixelRatio(1.5f);
        load(1225);
        select_->ShowSelectBox();
        render(57, 1407);
        EXPECT_GE(box()->GetAbsoluteOffset(Rml::BoxArea::Border).y, 57);
        EXPECT_LE(box()->GetAbsoluteOffset(Rml::BoxArea::Border).y + box()->GetOffsetHeight(), 1407);
    }

    TEST_F(ViewportSelectTest, UpdateCost) {
        if (!std::getenv("LFS_SELECT_BENCH"))
            GTEST_SKIP() << "Opt-in update timing";
        Rml::ElementInstancerGeneric<Rml::ElementFormControlSelect> native_instancer;
        for (bool opened : {false, true}) {
            for (int repeat = 0; repeat < 5; ++repeat) {
                for (bool native : {true, false}) {
                    if (document_) {
                        document_->Close();
                        context_->Update();
                    }
                    Rml::Factory::RegisterElementInstancer("select", native
                                                                         ? static_cast<Rml::ElementInstancer*>(&native_instancer)
                                                                         : static_cast<Rml::ElementInstancer*>(&instancer_));
                    load();
                    if (opened)
                        select_->ShowSelectBox();
                    for (int i = 0; i < 100; ++i)
                        render();
                    const auto start = std::chrono::steady_clock::now();
                    for (int i = 0; i < 10000; ++i)
                        render();
                    const auto ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start).count() / 10000;
                    std::cout << "SELECT_UPDATE native=" << native << " opened=" << opened << " ns=" << ns << '\n';
                }
            }
        }
        Rml::Factory::RegisterElementInstancer("select", &instancer_);
    }
} // namespace
