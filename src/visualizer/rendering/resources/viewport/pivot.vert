#version 450
layout(location = 0) out vec2 v_uv;
layout(push_constant) uniform PivotPush {
    vec4 center_size;
    vec4 color_opacity;
} u;

const vec2 CORNERS[4] = vec2[4](
    vec2(-1.0, -1.0), vec2(1.0, -1.0),
    vec2(1.0, 1.0), vec2(-1.0, 1.0)
);
const int INDICES[6] = int[6](0, 1, 2, 0, 2, 3);

void main() {
    vec2 corner = CORNERS[INDICES[gl_VertexIndex]];
    v_uv = corner;
    gl_Position = vec4(u.center_size.xy + corner * u.center_size.zw, 0.0, 1.0);
}
