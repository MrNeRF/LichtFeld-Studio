/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <RmlUi/Core.h>
#include <RmlUi/Core/Context.h>
#include <RmlUi/Core/Element.h>
#include <RmlUi/Core/ElementDocument.h>
#include <RmlUi/Core/Event.h>
#include <RmlUi/Core/EventListener.h>
#include <RmlUi/Core/RenderInterface.h>

#include <gtest/gtest.h>

#include <string>

#include <visualizer/gui/rmlui/rml_context_owner.hpp>

namespace {

    class NullRenderInterface final : public Rml::RenderInterface {
    public:
        Rml::CompiledGeometryHandle CompileGeometry(Rml::Span<const Rml::Vertex>,
                                                    Rml::Span<const int>) override {
            return 1;
        }
        void RenderGeometry(Rml::CompiledGeometryHandle, Rml::Vector2f,
                            Rml::TextureHandle) override {}
        void ReleaseGeometry(Rml::CompiledGeometryHandle) override {}

        Rml::TextureHandle LoadTexture(Rml::Vector2i& texture_dimensions,
                                       const Rml::String&) override {
            texture_dimensions = {};
            return {};
        }
        Rml::TextureHandle GenerateTexture(Rml::Span<const Rml::byte>,
                                           Rml::Vector2i) override {
            return {};
        }
        void ReleaseTexture(Rml::TextureHandle) override {}

        void EnableScissorRegion(bool) override {}
        void SetScissorRegion(Rml::Rectanglei) override {}
    };

    class CountingClickListener final : public Rml::EventListener {
    public:
        void ProcessEvent(Rml::Event&) override {
            ++count;
        }

        int count = 0;
    };

    bool belongsToDocument(Rml::Element* element,
                           Rml::ElementDocument* const document) {
        while (element) {
            if (element == document)
                return true;
            element = element->GetParentNode();
        }
        return false;
    }

    class RmlSharedContextContractTest : public ::testing::Test {
    protected:
        static void SetUpTestSuite() {
            ASSERT_TRUE(Rml::Initialise());
        }

        static void TearDownTestSuite() {
            Rml::Shutdown();
        }

        void SetUp() override {
            context_ = Rml::CreateContext("gui-shared-context-contract",
                                          Rml::Vector2i(320, 240),
                                          &render_interface_);
            ASSERT_NE(context_, nullptr);
        }

        void TearDown() override {
            if (context_)
                Rml::RemoveContext(context_->GetName());
        }

        Rml::Context* context_ = nullptr;
        NullRenderInterface render_interface_;
    };

    Rml::ElementDocument* loadSurface(Rml::Context* const context,
                                      const char* const id,
                                      const char* const colour) {
        const Rml::String document = Rml::String("<rml><head><style>"
                                                 "body { margin: 0px; }"
                                                 "#") +
                                     id +
                                     " { position: absolute; left: 24px; top: 24px; "
                                     "width: 180px; height: 100px; background-color: " +
                                     colour +
                                     "; }</style></head><body><div id='" +
                                     id + "'></div></body></rml>";
        auto* const document_handle = context->LoadDocumentFromMemory(document, id);
        if (document_handle)
            document_handle->Show(Rml::ModalFlag::None, Rml::FocusFlag::None);
        return document_handle;
    }

    Rml::ElementDocument* loadSurfaceAt(Rml::Context* const context,
                                         const char* const id,
                                         const char* const colour,
                                         const int left, const int top,
                                         const int width, const int height) {
        const Rml::String document = Rml::String("<rml><head><style>"
                                                 "body { margin: 0px; width: 100%; height: 100%; }"
                                                 "#") +
                                     id +
                                     " { width: 100%; height: 100%; background-color: " +
                                     colour + "; }</style></head><body><div id='" +
                                     id + "'></div></body></rml>";
        auto* const document_handle = context->LoadDocumentFromMemory(document, id);
        if (document_handle) {
            document_handle->SetProperty("position", "absolute");
            document_handle->SetProperty("left", std::to_string(left) + "px");
            document_handle->SetProperty("top", std::to_string(top) + "px");
            document_handle->SetProperty("width", std::to_string(width) + "px");
            document_handle->SetProperty("height", std::to_string(height) + "px");
            document_handle->Show(Rml::ModalFlag::None, Rml::FocusFlag::None);
        }
        return document_handle;
    }

    TEST_F(RmlSharedContextContractTest,
           DocumentsShareNativeStackingHitTestingAndInput) {
        auto* const back = loadSurface(context_, "back-surface", "#1b5e20");
        auto* const front = loadSurface(context_, "front-surface", "#b71c1c");
        ASSERT_NE(back, nullptr);
        ASSERT_NE(front, nullptr);
        ASSERT_TRUE(context_->Update());

        EXPECT_EQ(context_->GetNumDocuments(), 2);
        EXPECT_EQ(back->GetContext(), context_);
        EXPECT_EQ(front->GetContext(), context_);

        auto* const back_surface = back->GetElementById("back-surface");
        auto* const front_surface = front->GetElementById("front-surface");
        ASSERT_NE(back_surface, nullptr);
        ASSERT_NE(front_surface, nullptr);

        CountingClickListener front_clicks;
        front_surface->AddEventListener("click", &front_clicks);

        EXPECT_EQ(context_->GetElementAtPoint(Rml::Vector2f(40.0f, 40.0f)), front_surface);
        context_->ProcessMouseMove(40, 40, 0);
        context_->ProcessMouseButtonDown(0, 0);
        context_->ProcessMouseButtonUp(0, 0);
        EXPECT_EQ(front_clicks.count, 1);

        front->PushToBack();
        ASSERT_TRUE(context_->Update());
        EXPECT_EQ(context_->GetElementAtPoint(Rml::Vector2f(40.0f, 40.0f)), back_surface);

        front_surface->RemoveEventListener("click", &front_clicks);
    }

    TEST_F(RmlSharedContextContractTest, UnloadingOneDocumentKeepsTheOtherDocumentLive) {
        auto* const first = loadSurface(context_, "first-surface", "#0d47a1");
        auto* const second = loadSurface(context_, "second-surface", "#4a148c");
        ASSERT_NE(first, nullptr);
        ASSERT_NE(second, nullptr);
        ASSERT_TRUE(context_->Update());

        context_->UnloadDocument(second);
        ASSERT_TRUE(context_->Update());

        EXPECT_EQ(context_->GetNumDocuments(), 1);
        EXPECT_EQ(context_->GetDocument(0), first);
        EXPECT_EQ(context_->GetElementAtPoint(Rml::Vector2f(40.0f, 40.0f)),
                  first->GetElementById("first-surface"));
    }

    TEST_F(RmlSharedContextContractTest, DocumentsUseWindowSpaceBoundsForNativeHitTesting) {
        auto* const back = loadSurfaceAt(context_, "back-surface", "#1b5e20",
                                         40, 50, 180, 100);
        auto* const front = loadSurfaceAt(context_, "front-surface", "#b71c1c",
                                          80, 85, 160, 80);
        ASSERT_NE(back, nullptr);
        ASSERT_NE(front, nullptr);
        ASSERT_TRUE(context_->Update());

        auto* const back_target =
            context_->GetElementAtPoint(Rml::Vector2f(50.0f, 60.0f));
        auto* const front_target =
            context_->GetElementAtPoint(Rml::Vector2f(100.0f, 100.0f));
        ASSERT_NE(back_target, nullptr);
        ASSERT_NE(front_target, nullptr);
        EXPECT_TRUE(belongsToDocument(back_target, back));
        EXPECT_TRUE(belongsToDocument(front_target, front));

        front->SetProperty("left", "220px");
        ASSERT_TRUE(context_->Update());
        EXPECT_TRUE(belongsToDocument(
            context_->GetElementAtPoint(Rml::Vector2f(100.0f, 100.0f)), back));
        EXPECT_TRUE(belongsToDocument(
            context_->GetElementAtPoint(Rml::Vector2f(230.0f, 100.0f)), front));
    }

    TEST_F(RmlSharedContextContractTest, BorrowedContextOwnerDoesNotDestroySharedContext) {
        {
            lfs::vis::gui::RmlContextOwner owner(context_);
            EXPECT_EQ(owner.get(), context_);
        }

        EXPECT_EQ(Rml::GetContext("gui-shared-context-contract"), context_);

        auto* const document = loadSurface(context_, "remaining-surface", "#263238");
        ASSERT_NE(document, nullptr);
        EXPECT_TRUE(context_->Update());
        EXPECT_EQ(context_->GetNumDocuments(), 1);
    }

} // namespace
