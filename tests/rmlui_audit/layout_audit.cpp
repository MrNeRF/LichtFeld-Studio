// SPDX-License-Identifier: GPL-3.0-or-later
// Standalone differential probe: identical staged resources run against each RmlUi version.
#include <RmlUi/Core.h>
#include <RmlUi/Core/ElementDocument.h>
#include <RmlUi/Core/ElementInstancer.h>
#include <RmlUi/Core/Factory.h>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
#include <unordered_map>
#include <unordered_set>
#ifdef LFS_RMLUI_AUDIT_USE_APP_CONTROLS
#include "gui/rmlui/elements/viewport_select.hpp"
#endif

using nlohmann::json;
namespace fs = std::filesystem;
struct System final : Rml::SystemInterface {
    json messages = json::array();
    double GetElapsedTime() override { return 1.0; }
    bool LogMessage(Rml::Log::Type type, const Rml::String& message) override {
        messages.push_back({{"type", int(type)}, {"message", message}});
        return true;
    }
};
struct Renderer final : Rml::RenderInterface {
    struct Mesh {
        std::vector<Rml::Vertex> vertices;
        size_t indices;
    };
    std::unordered_map<Rml::CompiledGeometryHandle, Mesh> meshes;
    Rml::CompiledGeometryHandle next_handle = 1;
    Rml::Matrix4f transform = Rml::Matrix4f::Identity();
    Rml::Rectanglei scissor;
    bool scissoring = false;
    json draws = json::object();
    Rml::CompiledGeometryHandle CompileGeometry(Rml::Span<const Rml::Vertex> vertices, Rml::Span<const int> indices) override {
        const auto handle = next_handle++;
        meshes.emplace(handle, Mesh{{vertices.begin(), vertices.end()}, indices.size()});
        return handle;
    }
    void RenderGeometry(Rml::CompiledGeometryHandle handle, Rml::Vector2f translation, Rml::TextureHandle texture) override {
        const auto& mesh = meshes.at(handle);
        if (mesh.vertices.empty() || mesh.indices == 0)
            return;
        float left = std::numeric_limits<float>::max(), top = left;
        float right = -left, bottom = -left;
        for (const auto& vertex : mesh.vertices) {
            const auto p = transform * Rml::Vector4f(vertex.position.x + translation.x, vertex.position.y + translation.y, 0.f, 1.f);
            left = std::min(left, p.x / p.w);
            top = std::min(top, p.y / p.w);
            right = std::max(right, p.x / p.w);
            bottom = std::max(bottom, p.y / p.w);
        }
        draws[std::to_string(draws.size())] = {{"tag", "paint"}, {"id", ""}, {"class", ""}, {"rect", {left, top, right - left, bottom - top}}, {"indices", mesh.indices}, {"textured", texture != 0}, {"scissoring", scissoring}, {"scissor", {scissor.Left(), scissor.Top(), scissor.Right(), scissor.Bottom()}}};
    }
    void ReleaseGeometry(Rml::CompiledGeometryHandle handle) override { meshes.erase(handle); }
    void SetTransform(const Rml::Matrix4f* value) override { transform = value ? *value : Rml::Matrix4f::Identity(); }
    Rml::TextureHandle LoadTexture(Rml::Vector2i& size, const Rml::String&) override {
        size = {32, 32};
        return 1;
    }
    Rml::TextureHandle GenerateTexture(Rml::Span<const Rml::byte>, Rml::Vector2i) override { return 1; }
    void ReleaseTexture(Rml::TextureHandle) override {}
    void EnableScissorRegion(bool value) override { scissoring = value; }
    void SetScissorRegion(Rml::Rectanglei value) override { scissor = value; }
};
json describe(Rml::Element* element) {
    auto offset = element->GetAbsoluteOffset(Rml::BoxArea::Border);
    auto size = element->GetBox().GetSize(Rml::BoxArea::Border);
    return {{"tag", element->GetTagName()}, {"id", element->GetId()}, {"class", element->GetClassNames()}, {"visible", element->IsVisible(true)}, {"rect", {offset.x, offset.y, size.x, size.y}}, {"client", {element->GetClientWidth(), element->GetClientHeight()}}, {"scroll", {element->GetScrollWidth(), element->GetScrollHeight(), element->GetScrollLeft(), element->GetScrollTop()}}};
}
void snapshot(Rml::Element* element, const std::string& path, json& result) {
    result[path] = describe(element);
    for (int i = 0; i < element->GetNumChildren(); ++i)
        snapshot(element->GetChild(i), path + "/" + std::to_string(i), result);
}
json focusTrace(Rml::Context* context, int modifiers) {
    json trace = json::object();
    std::unordered_set<Rml::Element*> visited;
    for (int step = 0; step < 512; ++step) {
        context->ProcessKeyDown(Rml::Input::KI_TAB, modifiers);
        context->ProcessKeyUp(Rml::Input::KI_TAB, modifiers);
        context->Update();
        auto* focus = context->GetFocusElement();
        if (!focus || !visited.insert(focus).second)
            break;
        int depth = 0;
        for (auto* e = focus; e; e = e->GetParentNode())
            trace[std::to_string(step) + "/" + std::to_string(depth++)] = describe(e);
    }
    return trace;
}
int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: rmlui_layout_audit STAGED_RESOURCES FONT_DIR OUTPUT.json\n";
        return 2;
    }
    System system;
    Renderer renderer;
    Rml::SetSystemInterface(&system);
    Rml::SetRenderInterface(&renderer);
    if (!Rml::Initialise())
        return 3;
#ifdef LFS_RMLUI_AUDIT_USE_APP_CONTROLS
    using namespace lfs::vis::gui;
    Rml::ElementInstancerGeneric<FormControlLayout<Rml::ElementFormControlInput>> input;
    Rml::ElementInstancerGeneric<FormControlLayout<Rml::ElementFormControlTextArea>> textarea;
    Rml::ElementInstancerGeneric<ViewportSelect> select;
    Rml::Factory::RegisterElementInstancer("input", &input);
    Rml::Factory::RegisterElementInstancer("textarea", &textarea);
    Rml::Factory::RegisterElementInstancer("select", &select);
#endif
    Rml::LoadFontFace((fs::path(argv[2]) / "Inter-Regular.ttf").generic_string());
    Rml::LoadFontFace((fs::path(argv[2]) / "Inter-SemiBold.ttf").generic_string());
    Rml::ElementInstancerGeneric<Rml::Element> generic;
    for (const char* tag : {"chromaticity-diagram", "color-picker", "colour-offset", "node-curve", "crf-curve", "loss-graph", "node-canvas", "vram-timeline", "python-editor-view", "scene-graph", "terminal-view"})
        Rml::Factory::RegisterElementInstancer(tag, &generic);
    json output = {{"cases", json::object()}};
    std::vector<fs::path> files;
    for (auto& entry : fs::directory_iterator(argv[1]))
        if (entry.path().extension() == ".rml")
            files.push_back(entry.path());
    std::sort(files.begin(), files.end());
    int failures = 0;
    for (float dpi : {1.0f, 1.5f, 2.0f})
        for (Rml::Vector2i viewport : {Rml::Vector2i{320, 480}, Rml::Vector2i{480, 720}, Rml::Vector2i{1280, 800}}) {
            auto* context = Rml::CreateContext("audit", viewport);
            context->SetDensityIndependentPixelRatio(dpi);
            for (const auto& file : files) {
                std::ifstream in(file);
                std::string source{std::istreambuf_iterator<char>(in), {}};
                if (source.find("<template ") != std::string::npos)
                    continue;
                const std::string key = file.filename().string() + "@" + std::to_string(viewport.x) + "x" + std::to_string(viewport.y) + "/" + std::to_string(dpi);
                auto* doc = context->LoadDocument(file.generic_string());
                if (!doc) {
                    output["cases"][key] = {{"error", "load failed"}};
                    ++failures;
                    continue;
                }
                doc->Show(Rml::ModalFlag::None, Rml::FocusFlag::None);
                context->Update();
                context->Update();
                json tree = json::object();
                snapshot(doc, "body", tree);
                output["cases"][key]["initial"] = std::move(tree);
                // Exercise end-of-scroll layout, then focus traversal and render geometry generation.
                std::vector<Rml::Element*> elements{doc};
                for (size_t i = 0; i < elements.size(); ++i) {
                    auto* e = elements[i];
                    for (int j = 0; j < e->GetNumChildren(); ++j)
                        elements.push_back(e->GetChild(j));
                    e->SetScrollTop(e->GetScrollHeight());
                    e->SetScrollLeft(e->GetScrollWidth());
                }
                context->Update();
                tree = json::object();
                snapshot(doc, "body", tree);
                output["cases"][key]["scrolled"] = std::move(tree);
                doc->Focus();
                for (int i = 0; i < 8; ++i) {
                    context->ProcessKeyDown(Rml::Input::KI_TAB, 0);
                    context->ProcessKeyUp(Rml::Input::KI_TAB, 0);
                    context->Update();
                }
                tree = json::object();
                snapshot(doc, "body", tree);
                output["cases"][key]["tabbed"] = std::move(tree);
                output["cases"][key]["focus_forward"] = focusTrace(context, 0);
                output["cases"][key]["focus_backward"] = focusTrace(context, Rml::Input::KM_SHIFT);
                renderer.draws = json::object();
                context->Render();
                output["cases"][key]["paint"] = renderer.draws;
                doc->Close();
                context->Update();
            }
            Rml::RemoveContext("audit");
        }
    output["messages"] = system.messages;
    output["load_failures"] = failures;
    Rml::Shutdown();
    std::ofstream(argv[3]) << output.dump(2);
    std::cout << "cases=" << output["cases"].size() << " load_failures=" << failures << " messages=" << system.messages.size() << "\n";
    return failures ? 1 : 0;
}
