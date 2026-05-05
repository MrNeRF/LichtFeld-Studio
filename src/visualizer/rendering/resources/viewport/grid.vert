#version 450
layout(location = 0) in vec2 vertex_position;
layout(location = 0) out vec3 worldFar;
layout(location = 1) out vec3 worldNear;
struct GridUniform {
    mat4 view_projection;
    vec4 view_position_plane;
    vec4 opacity_padding;
    vec4 near_origin;
    vec4 near_x;
    vec4 near_y;
    vec4 far_origin;
    vec4 far_x;
    vec4 far_y;
};
layout(std430, set = 0, binding = 0) readonly buffer GridUniforms {
    GridUniform grids[];
} grid_buffer;
layout(push_constant) uniform GridPush {
    int grid_index;
} push;

void main() {
    GridUniform u = grid_buffer.grids[push.grid_index];
    gl_Position = vec4(vertex_position, 0.0, 1.0);
    vec2 p = vec2(vertex_position.x * 0.5 + 0.5, -vertex_position.y * 0.5 + 0.5);
    worldNear = (u.near_origin + u.near_x * p.x + u.near_y * p.y).xyz;
    worldFar = (u.far_origin + u.far_x * p.x + u.far_y * p.y).xyz;
}
