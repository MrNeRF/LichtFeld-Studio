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
        auto proxy = to_proxy(settings);
        require(proxy.selection_outline, "Proxy lost the option");
        RenderSettings restored;
        apply_proxy(restored, proxy);
        require(restored.selection_outline, "Proxy round trip lost the option");
        auto json = project::renderSettingsToProjectJson(settings);
        require(json["selection_outline"] == true, "Project did not write the option");
        auto read = project::renderSettingsFromProjectJson(json);
        require(read && read->selection_outline, "Project round trip lost the option");
        json.erase("selection_outline");
        read = project::renderSettingsFromProjectJson(json);
        require(read && !read->selection_outline, "Earlier project files must retain the disabled default");
        const auto defaults = renderingDefaultsFromSettings(settings);
        require(bool(applyRenderingDefaults(restored, defaults)) && restored.selection_outline,
                "Viewport defaults lost the option");
        require(!applyRenderingDefaults(restored, R"({"selection_outline":"invalid"})") && restored.selection_outline,
                "Invalid defaults changed the current settings");
        std::cout << "PASS outline gating, object masks, proxy, project compatibility and defaults\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
