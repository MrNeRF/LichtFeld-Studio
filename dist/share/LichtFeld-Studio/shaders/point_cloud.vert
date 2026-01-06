#version 430 core

layout(location = 0) in vec3 a_vertex_position;
layout(location = 1) in vec3 a_instance_position;
layout(location = 2) in vec3 a_instance_color;
layout(location = 3) in float a_transform_index;

uniform mat4 u_view;
uniform mat4 u_projection;
uniform float u_voxel_size;

#define MAX_TRANSFORMS 64
uniform mat4 u_model_transforms[MAX_TRANSFORMS];
uniform int u_num_transforms;

out vec3 v_color;
out vec3 v_normal;
out vec3 v_frag_pos;

void main() {
    mat4 model = mat4(1.0);
    int idx = int(a_transform_index);
    if (u_num_transforms > 0 && idx >= 0 && idx < u_num_transforms) {
        model = u_model_transforms[idx];
    }

    vec3 world_position = (model * vec4(a_instance_position, 1.0)).xyz;
    world_position += a_vertex_position * u_voxel_size;

    vec4 view_pos = u_view * vec4(world_position, 1.0);
    gl_Position = u_projection * view_pos;

    v_color = a_instance_color;
    v_normal = normalize(mat3(model) * normalize(a_vertex_position));
    v_frag_pos = view_pos.xyz;
}
