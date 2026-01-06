#version 430 core

// Inputs from vertex shader
in vec3 FragPos;
in vec4 vertexColor;
in vec2 TexCoord;
flat in int instanceID;
flat in uint textureID;
flat in uint isValidation;

// Output
out vec4 FragColor;

// Uniforms
uniform vec3 viewPos;
uniform int highlightIndex = -1;
uniform vec3 trainHighlightColor = vec3(1.0, 0.55, 0.0);   // Dark orange for training
uniform vec3 valHighlightColor = vec3(0.9, 0.75, 0.0);     // Dark yellow for validation
uniform bool pickingMode = false;
uniform float minimumPickDistance = 0.5;
uniform bool showImages = false;
uniform float imageOpacity = 0.7;
uniform sampler2DArray cameraTextures;

void main() {
    if (pickingMode) {
        float distance = length(viewPos - FragPos);
        if (distance < minimumPickDistance) {
            discard;
        }
        FragColor = vertexColor;
        return;
    }

    // textureID stores layer index + 1 (0 means no texture)
    if (showImages && textureID > 0u) {
        vec4 imageColor = texture(cameraTextures, vec3(TexCoord, float(textureID - 1u)));
        vec4 finalColor = vec4(imageColor.rgb, imageOpacity);

        if (instanceID == highlightIndex) {
            vec3 highlightTint = (isValidation > 0u) ? valHighlightColor : trainHighlightColor;
            finalColor.rgb = mix(finalColor.rgb, highlightTint, 0.3);
        }

        FragColor = finalColor;
        return;
    }

    // Wireframe: use vertex color (contains per-camera color with alpha)
    vec4 finalColor = vertexColor;

    // Apply highlight when hovered - different color for train vs validation
    if (instanceID == highlightIndex) {
        finalColor.rgb = (isValidation > 0u) ? valHighlightColor : trainHighlightColor;
        finalColor.a = min(1.0, finalColor.a + 0.3);
    }

    FragColor = finalColor;
}
