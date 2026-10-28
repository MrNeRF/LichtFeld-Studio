/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "visualizer/ipc/render_settings_convert.hpp"
#include "visualizer/preferences.hpp"
#include "visualizer/project/session_state.hpp"
#include "visualizer/rendering/viewport_request_builder.hpp"
#include <iostream>
#include <stdexcept>

static void require(bool value, const char* message) {
    if (!value)
        throw std::runtime_error(message);
}

int main() {
    try {
        using namespace lfs::vis;
        Viewport viewport(640, 480);
        FrameContext context{.viewport = viewport, .render_size = {640, 480}};
        auto& scene = context.scene_state;
        auto& settings = context.settings;
        auto request = [&] { return buildViewportRenderRequest(context, {640, 480}); };
        scene.selected_node_mask = {true, false};
        require(!request().overlay.markers.show_object_outline, "Outline must be opt-in");
        settings.selection_outline = true;
        const auto selected = request();
        require(selected.overlay.markers.show_object_outline, "Selected nodes did not enable the outline");
        require(selected.overlay.emphasis.emphasized_node_mask == scene.selected_node_mask,
                "Outline lost the object mask when desaturation is disabled");
        require(!selected.overlay.emphasis.dim_non_emphasized, "Outline enabled desaturation");
        context.gaussian_selection_visible = false;
        require(request().overlay.markers.show_object_outline, "Hidden point selection hid object outline");
        scene.selected_node_mask = {false, false};
        require(!request().overlay.markers.show_object_outline, "Empty selection retained outline work");
        scene.selected_node_mask = {false, true};
        context.training_active = true;
        require(!request().overlay.markers.show_object_outline, "Training retained outline work");
        context.training_active = false;
        settings.point_cloud_mode = true;
        require(!request().overlay.markers.show_object_outline, "Point mode retained splat outline work");
        settings.point_cloud_mode = false;
        settings.depth_view = true;
        require(!request().overlay.markers.show_object_outline, "Depth view was colorized by outline");
        settings.depth_view = false;
        settings.selection_outline_color = {0.2f, 0.4f, 0.6f};
        settings.selection_outline_opacity = 0.3f;
        settings.selection_outline_width = 1.5f;
        settings.selection_outline_threshold = 0.7f;
        settings.selection_outline_holes = false;
        const auto appearance = request().overlay.markers;
        require(appearance.outline_color == settings.selection_outline_color &&
                    appearance.outline_opacity == 0.3f && appearance.outline_width == 1.5f &&
                    appearance.outline_threshold == 0.7f && !appearance.outline_holes,
                "Render request lost outline appearance");
        auto proxy = to_proxy(settings);
        require(proxy.selection_outline, "Proxy lost the option");
        RenderSettings restored;
        apply_proxy(restored, proxy);
        require(restored.selection_outline, "Proxy round trip lost the option");
        require(restored.selection_outline_color == settings.selection_outline_color &&
                    restored.selection_outline_opacity == 0.3f && restored.selection_outline_width == 1.5f &&
                    restored.selection_outline_threshold == 0.7f && !restored.selection_outline_holes,
                "Proxy round trip lost outline appearance");
        auto json = project::renderSettingsToProjectJson(settings);
        require(json["selection_outline"] == true, "Project did not write the option");
        auto read = project::renderSettingsFromProjectJson(json);
        require(read && read->selection_outline, "Project round trip lost the option");
        require(read->selection_outline_color == settings.selection_outline_color &&
                    read->selection_outline_opacity == 0.3f && read->selection_outline_width == 1.5f &&
                    read->selection_outline_threshold == 0.7f && !read->selection_outline_holes,
                "Project round trip lost outline appearance");
        for (const auto* key : {"selection_outline_color", "selection_outline_opacity", "selection_outline_width", "selection_outline_threshold", "selection_outline_holes"})
            json.erase(key);
        read = project::renderSettingsFromProjectJson(json);
        require(read && read->selection_outline_opacity == 1.0f && read->selection_outline_width == 2.0f &&
                    read->selection_outline_threshold == 0.5f && read->selection_outline_holes,
                "Earlier projects lost outline appearance defaults");
        json["selection_color_center_marker"] = {1.0f, 0.0f, 0.0f};
        read = project::renderSettingsFromProjectJson(json);
        require(read && read->selection_outline_color == glm::vec3(1.0f, 0.0f, 0.0f),
                "Earlier projects lost their custom outline color");
        json.erase("selection_outline");
        read = project::renderSettingsFromProjectJson(json);
        require(read && !read->selection_outline, "Earlier project files must retain the disabled default");
        const auto defaults = renderingDefaultsFromSettings(settings);
        require(bool(applyRenderingDefaults(restored, defaults)) && restored.selection_outline,
                "Viewport defaults lost the option");
        require(!applyRenderingDefaults(restored, R"({"selection_outline":"invalid"})") && restored.selection_outline,
                "Invalid defaults changed the current settings");
        require(restored.selection_outline_color == settings.selection_outline_color && restored.selection_outline_width == 1.5f && !restored.selection_outline_holes,
                "Viewport defaults lost outline appearance");
        for (const auto* invalid : {R"({"selection_outline_opacity":-0.1})", R"({"selection_outline_opacity":1.1})",
                                    R"({"selection_outline_width":13})", R"({"selection_outline_threshold":0})",
                                    R"({"selection_outline_holes":"false"})", R"({"selection_outline_color":[1,0]})",
                                    R"({"selection_outline_color":[1,-0.1,0]})"}) {
            require(!applyRenderingDefaults(restored, invalid), "Invalid outline defaults accepted");
            require(restored.selection_outline_opacity == 0.3f,
                    "Invalid defaults partially changed outline appearance");
        }
        require(bool(applyRenderingDefaults(restored, R"({"selection_outline_width":12})")) && restored.selection_outline_width == 12.0f, "Extended thickness rejected");
        std::cout << "PASS outline gating, object masks, proxy, project compatibility and defaults\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
