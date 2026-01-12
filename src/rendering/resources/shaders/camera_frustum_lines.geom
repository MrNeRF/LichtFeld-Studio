#version 430 core

layout(lines) in;
layout(line_strip, max_vertices = 2) out;

in vec3 FragPos[];
in vec4 vertexColor[];
in vec2 TexCoord[];
in float ndcX[];
flat in int instanceID[];
flat in uint textureID[];
flat in uint isValidation[];
flat in uint isEquirectangular[];
flat in int equiView[];

out vec3 g_FragPos;
out vec4 g_vertexColor;
out vec2 g_TexCoord;
flat out int g_instanceID;
flat out uint g_textureID;
flat out uint g_isValidation;
flat out uint g_isEquirectangular;

void main() {
    if (equiView[0] == 1 && abs(ndcX[0] - ndcX[1]) > 1.0)
        return;

    for (int i = 0; i < 2; i++) {
        gl_Position = gl_in[i].gl_Position;
        g_FragPos = FragPos[i];
        g_vertexColor = vertexColor[i];
        g_TexCoord = TexCoord[i];
        g_instanceID = instanceID[i];
        g_textureID = textureID[i];
        g_isValidation = isValidation[i];
        g_isEquirectangular = isEquirectangular[i];
        EmitVertex();
    }
    EndPrimitive();
}
