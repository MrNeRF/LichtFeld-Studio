#version 330 core
out vec4 FragColor;

in vec2 TexCoord;

uniform sampler2D texture0;

void main() {
    // Sample texture with correct coordinates
    vec4 color = texture(texture0, TexCoord);
    
    // Ensure full opacity
    FragColor = vec4(color.rgb, 1.0);
}
