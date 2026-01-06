#version 330 core
out vec4 FragColor;

in vec2 TexCoord;

uniform sampler2D leftTexture;
uniform sampler2D rightTexture;
uniform float splitPosition;
uniform bool showDivider;
uniform vec4 dividerColor;
uniform float dividerWidth;

void main() {
    vec2 uv = TexCoord;
    
    // Sample BOTH textures at the SAME coordinates
    // This ensures they're perfectly aligned
    vec4 leftColor = texture(leftTexture, uv);
    vec4 rightColor = texture(rightTexture, uv);
    
    // Choose which color to use based on split position
    vec4 color;
    if (uv.x < splitPosition) {
        // Show left texture on the left side
        color = leftColor;
    } else {
        // Show right texture on the right side
        color = rightColor;
    }
    
    // Draw divider if enabled
    if (showDivider) {
        float halfWidth = dividerWidth * 0.5;
        if (abs(uv.x - splitPosition) < halfWidth) {
            color = dividerColor;
        }
    }
    
    FragColor = color;
}
