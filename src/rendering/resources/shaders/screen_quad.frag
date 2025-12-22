#version 430 core

in vec2 TexCoord;
uniform sampler2D screenTexture;
uniform sampler2D depthTexture;
uniform float near_plane = 0.1;
uniform float far_plane = 100000.0;
uniform bool has_depth = false;
uniform bool orthographic = false;
uniform bool depth_is_ndc = false;  // True if depth is already NDC (OpenGL), skip conversion
out vec4 FragColor;

// Convert view-space depth to NDC depth (0 to 1 range for OpenGL)
float viewDepthToNDC(float z_view) {
    // Handle no-hit case (large depth values)
    if (z_view > 1e9) {
        return 1.0;  // Far plane
    }

    if (orthographic) {
        // Orthographic: linear mapping
        return (z_view - near_plane) / (far_plane - near_plane);
    } else {
        // Perspective projection depth conversion
        // NDC z = (f + n) / (f - n) + (2 * f * n) / ((f - n) * -z_view)
        // Then convert from [-1, 1] to [0, 1]
        float A = (far_plane + near_plane) / (far_plane - near_plane);
        float B = (2.0 * far_plane * near_plane) / (far_plane - near_plane);
        float ndc_z = A - B / z_view;
        return ndc_z * 0.5 + 0.5;
    }
}

void main()
{
    vec2 uv = vec2(TexCoord.x, 1.0 - TexCoord.y);
    FragColor = texture(screenTexture, uv);

    if (has_depth) {
        float depth_value = texture(depthTexture, uv).r;
        if (depth_is_ndc) {
            // Depth is already in NDC format (0-1), use directly
            gl_FragDepth = depth_value;
        } else {
            // Depth is view-space, convert to NDC
            gl_FragDepth = viewDepthToNDC(depth_value);
        }
    } else {
        // No depth available - use fixed far depth to not interfere
        gl_FragDepth = 1.0;
    }
}