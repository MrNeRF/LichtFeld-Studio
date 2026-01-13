#version 430 core

in vec3 g_FragPos;
in vec4 g_vertexColor;
in vec2 g_TexCoord;
flat in int g_instanceID;
flat in uint g_textureID;
flat in uint g_isValidation;
flat in uint g_isEquirectangular;

out vec4 FragColor;

uniform vec3 viewPos;
uniform int highlightIndex = -1;
uniform vec3 trainHighlightColor = vec3(1.0, 0.55, 0.0);
uniform vec3 valHighlightColor = vec3(0.9, 0.75, 0.0);
uniform bool pickingMode = false;
uniform float minimumPickDistance = 0.5;
uniform bool showImages = false;
uniform float imageOpacity = 0.7;
uniform sampler2DArray cameraTextures;

void main() {
    if (pickingMode) {
        float distance = length(viewPos - g_FragPos);
        if (distance < minimumPickDistance) {
            discard;
        }
        FragColor = g_vertexColor;
        return;
    }

    // textureID stores layer index + 1 (0 means no texture)
    if (showImages && g_textureID > 0u) {
        vec4 imageColor = texture(cameraTextures, vec3(g_TexCoord, float(g_textureID - 1u)));
        vec4 finalColor = vec4(imageColor.rgb, imageOpacity);

        if (g_instanceID == highlightIndex) {
            vec3 highlightTint = (g_isValidation > 0u) ? valHighlightColor : trainHighlightColor;
            finalColor.rgb = mix(finalColor.rgb, highlightTint, 0.3);
        }

        FragColor = finalColor;
        return;
    }

    // Wireframe: use vertex color (contains per-camera color with alpha)
    vec4 finalColor = g_vertexColor;

    // Apply highlight when hovered - different color for train vs validation
    if (g_instanceID == highlightIndex) {
        finalColor.rgb = (g_isValidation > 0u) ? valHighlightColor : trainHighlightColor;
        finalColor.a = min(1.0, finalColor.a + 0.3);
    }

    FragColor = finalColor;
}
