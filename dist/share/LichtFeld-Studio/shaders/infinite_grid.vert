#version 430 core

layout(location = 0) in vec2 vertex_position;

uniform vec3 near_origin;
uniform vec3 near_x;
uniform vec3 near_y;

uniform vec3 far_origin;
uniform vec3 far_x;
uniform vec3 far_y;

out vec3 worldFar;
out vec3 worldNear;

void main() {
    gl_Position = vec4(vertex_position, 0.0, 1.0);
    
    // Map from NDC [-1, 1] to [0, 1]
    vec2 p = vertex_position * 0.5 + 0.5;
    
    // Calculate world space positions on near and far planes
    worldNear = near_origin + near_x * p.x + near_y * p.y;
    worldFar = far_origin + far_x * p.x + far_y * p.y;
}

