#version 450

// Generates thick frustum-edge quads for each camera instance.

layout(location = 0) out vec2 ScreenPos;
layout(location = 1) out vec2 P0;
layout(location = 2) out vec2 P1;
layout(location = 3) out vec4 Color;
layout(location = 4) out vec4 Params;
layout(location = 5) out float ViewDepth;

struct FrustumInstance {
    mat4 model;
    vec4 color;
};

layout(std430, set = 1, binding = 0) readonly buffer FrustumInstances {
    FrustumInstance items[];
} frustums;

layout(push_constant) uniform FrustumPush {
    vec4 viewport_rect;
    // x: depth available, y: flip-y depth UV, z: line thickness, w: projection mode.
    vec4 params;
    mat4 view;
    vec4 viewport_panel;
    vec4 projection;
} pc;

const float PI = 3.14159265358979323846;

const vec3 PTS[5] = vec3[5](
    vec3(-0.5, -0.5, -1.0),
    vec3( 0.5, -0.5, -1.0),
    vec3( 0.5,  0.5, -1.0),
    vec3(-0.5,  0.5, -1.0),
    vec3( 0.0,  0.0,  0.0));

const ivec2 EDGES[8] = ivec2[8](
    ivec2(0, 1), ivec2(1, 2), ivec2(2, 3), ivec2(3, 0),
    ivec2(0, 4), ivec2(1, 4), ivec2(2, 4), ivec2(3, 4));

const vec2 QUAD[6] = vec2[6](
    vec2(0.0,  1.0), vec2(1.0,  1.0), vec2(1.0, -1.0),
    vec2(0.0,  1.0), vec2(1.0, -1.0), vec2(0.0, -1.0));

void emitDegenerate() {
    ScreenPos = vec2(0.0);
    P0 = vec2(0.0);
    P1 = vec2(0.0);
    Color = vec4(0.0);
    Params = vec4(0.0, 1.0, 0.0, 1.0);
    ViewDepth = 0.0;
    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
}

bool segmentOutsidePanel(vec2 a, vec2 b, vec2 panel_pos, vec2 panel_size, float extent) {
    vec2 panel_min = panel_pos - vec2(extent);
    vec2 panel_max = panel_pos + panel_size + vec2(extent);
    vec2 seg_min = min(a, b);
    vec2 seg_max = max(a, b);
    return seg_max.x < panel_min.x || seg_max.y < panel_min.y ||
           seg_min.x > panel_max.x || seg_min.y > panel_max.y;
}

bool projectEquirect(vec3 view, vec2 panel_pos, vec2 panel_size, out vec2 screen, out float depth) {
    depth = length(view);
    if (depth <= 1e-6) {
        return false;
    }
    vec3 dir = view / depth;
    float ndc_x = atan(dir.x, -dir.z) / PI;
    float ndc_y = -asin(clamp(dir.y, -1.0, 1.0)) / (PI * 0.5);
    screen = panel_pos + vec2((ndc_x * 0.5 + 0.5) * panel_size.x,
                              (ndc_y * 0.5 + 0.5) * panel_size.y);
    return true;
}

void main() {
    uint edge = uint(gl_VertexIndex) / 6u;
    uint corner = uint(gl_VertexIndex) % 6u;
    FrustumInstance inst = frustums.items[gl_InstanceIndex];

    ivec2 e = EDGES[edge];
    vec4 world_a = inst.model * vec4(PTS[e.x], 1.0);
    vec4 world_b = inst.model * vec4(PTS[e.y], 1.0);
    vec3 view_a = (pc.view * world_a).xyz;
    vec3 view_b = (pc.view * world_b).xyz;

    vec2 panel_pos = pc.viewport_panel.xy;
    vec2 panel_size = max(pc.viewport_panel.zw, vec2(1.0));
    vec2 render_size = max(pc.projection.xy, vec2(1.0));
    float cx = render_size.x * 0.5;
    float cy = render_size.y * 0.5;
    vec2 scale = panel_size / render_size;
    bool equirectangular = pc.params.w > 1.5;
    bool ortho = pc.params.w > 0.5 && pc.params.w < 1.5;

    if (!equirectangular) {
        const float kMinViewZ = -1e-4;
        bool a_behind = view_a.z >= kMinViewZ;
        bool b_behind = view_b.z >= kMinViewZ;
        if (a_behind && b_behind) {
            emitDegenerate();
            return;
        }
        if (a_behind) {
            float t = (kMinViewZ - view_a.z) / (view_b.z - view_a.z);
            view_a = mix(view_a, view_b, clamp(t, 0.0, 1.0));
            view_a.z = kMinViewZ;
        } else if (b_behind) {
            float t = (kMinViewZ - view_b.z) / (view_a.z - view_b.z);
            view_b = mix(view_b, view_a, clamp(t, 0.0, 1.0));
            view_b.z = kMinViewZ;
        }
    }

    vec2 screen_a;
    vec2 screen_b;
    float depth_a = 0.0;
    float depth_b = 0.0;
    if (equirectangular) {
        if (!projectEquirect(view_a, panel_pos, panel_size, screen_a, depth_a) ||
            !projectEquirect(view_b, panel_pos, panel_size, screen_b, depth_b)) {
            emitDegenerate();
            return;
        }
        float normalized_ax = (screen_a.x - panel_pos.x) / panel_size.x;
        float normalized_bx = (screen_b.x - panel_pos.x) / panel_size.x;
        if (abs(normalized_ax - normalized_bx) > 0.5) {
            emitDegenerate();
            return;
        }
    } else if (ortho) {
        float ortho_scale = max(pc.projection.z, 1.0e-6);
        screen_a = panel_pos + vec2(cx + view_a.x * ortho_scale, cy - view_a.y * ortho_scale) * scale;
        screen_b = panel_pos + vec2(cx + view_b.x * ortho_scale, cy - view_b.y * ortho_scale) * scale;
        depth_a = max(-view_a.z, 0.0);
        depth_b = max(-view_b.z, 0.0);
    } else {
        float perspective_depth_a = max(-view_a.z, 1.0e-6);
        float perspective_depth_b = max(-view_b.z, 1.0e-6);
        screen_a = panel_pos + vec2(cx + view_a.x * pc.projection.z / perspective_depth_a,
                                    cy - view_a.y * pc.projection.w / perspective_depth_a) * scale;
        screen_b = panel_pos + vec2(cx + view_b.x * pc.projection.z / perspective_depth_b,
                                    cy - view_b.y * pc.projection.w / perspective_depth_b) * scale;
        depth_a = max(-view_a.z, 0.0);
        depth_b = max(-view_b.z, 0.0);
    }

    float thickness = max(pc.params.z, 1.0);
    float extent = thickness * 0.5 + 2.0;
    if (segmentOutsidePanel(screen_a, screen_b, panel_pos, panel_size, extent)) {
        emitDegenerate();
        return;
    }
    vec2 delta = screen_b - screen_a;
    float len = length(delta);
    vec2 dir = (len > 1e-4) ? (delta / len) : vec2(1.0, 0.0);
    vec2 nrm = vec2(-dir.y, dir.x);

    float end_sel = QUAD[corner].x;
    float side = QUAD[corner].y;
    vec2 base = (end_sel < 0.5) ? (screen_a - dir * extent)
                                : (screen_b + dir * extent);
    vec2 screen = base + nrm * (side * extent);

    ScreenPos = screen;
    P0 = screen_a;
    P1 = screen_b;
    Color = inst.color;
    Params = vec4(0.0, thickness, 0.0, 1.0);
    ViewDepth = (end_sel < 0.5) ? depth_a : depth_b;

    vec2 ndc = vec2((screen.x - panel_pos.x) / panel_size.x,
                    (screen.y - panel_pos.y) / panel_size.y) * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
}
