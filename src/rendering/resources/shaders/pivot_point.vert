#version 330 core

uniform mat4 u_view;
uniform mat4 u_projection;
uniform vec3 u_pivot_pos;
uniform float u_screen_size;
uniform vec2 u_viewport_size;

out vec2 v_uv;

const vec2 CORNERS[4] = vec2[4](
    vec2(-1.0, -1.0), vec2(1.0, -1.0),
    vec2(1.0, 1.0), vec2(-1.0, 1.0)
);
const int INDICES[6] = int[6](0, 1, 2, 0, 2, 3);

void main() {
    vec2 corner = CORNERS[INDICES[gl_VertexID]];
    v_uv = corner;

    vec4 clip_pos = u_projection * u_view * vec4(u_pivot_pos, 1.0);
    vec2 ndc = clip_pos.xy / clip_pos.w;
    vec2 size_ndc = (u_screen_size / u_viewport_size) * 2.0;

    gl_Position = vec4(ndc + corner * size_ndc, 0.0, 1.0);
}
