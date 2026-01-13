#version 330 core

in vec3 g_color;

out vec4 FragColor;

void main() {
    FragColor = vec4(g_color, 1.0);
}
