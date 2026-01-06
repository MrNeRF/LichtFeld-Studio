#version 430 core

layout (location = 0) in vec3 a_position;

uniform mat4 u_mvp;
uniform vec3 u_box_center;
uniform vec3 u_view_dir;

out float v_backface_factor;

void main() {
    gl_Position = u_mvp * vec4(a_position, 1.0);

    // Calculate how much this vertex is on the back side of the box
    // by checking if the vertex-to-center direction aligns with view direction
    vec3 to_center = normalize(u_box_center - a_position);
    float alignment = dot(to_center, u_view_dir);

    // If alignment < 0, vertex is on the far side from camera (back-facing)
    v_backface_factor = clamp(-alignment * 2.0, 0.0, 1.0);
}