#version 450
layout(location = 0) in vec2 vertex_position;
layout(location = 0) out vec3 rayOrigin;
layout(location = 1) out vec3 rayDir;
struct GridUniform {
    mat4 view_projection;
    vec4 view_position_plane;
    vec4 opacity_ortho;
    vec4 pivot_radius;
    vec4 ray_origin;
    vec4 ray_origin_x;
    vec4 ray_origin_y;
    vec4 ray_dir;
    vec4 ray_dir_x;
    vec4 ray_dir_y;
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
    rayOrigin = (u.ray_origin + u.ray_origin_x * p.x + u.ray_origin_y * p.y).xyz;
    rayDir = (u.ray_dir + u.ray_dir_x * p.x + u.ray_dir_y * p.y).xyz;
}
