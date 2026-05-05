#version 450
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec4 inTangent; // xyz + handedness
layout(location = 3) in vec2 inTexcoord;
layout(location = 4) in vec4 inColor;

layout(push_constant) uniform MeshPush {
    mat4 mvp;
    mat4 model;
} push;

layout(location = 0) out vec3 outWorldPos;
layout(location = 1) out vec3 outNormal;
layout(location = 2) out vec2 outTexcoord;
layout(location = 3) out vec4 outColor;
layout(location = 4) out mat3 outTBN; // 3 vec3 attributes -> locations 4,5,6

void main() {
    vec4 wpos = push.model * vec4(inPosition, 1.0);
    outWorldPos = wpos.xyz;

    // Normal transform — uses transpose(inverse(mat3(model))) to handle non-uniform
    // scale correctly. For rigid transforms this collapses to mat3(model).
    mat3 normal_matrix = transpose(inverse(mat3(push.model)));
    outNormal = normalize(normal_matrix * inNormal);

    outTexcoord = inTexcoord;
    outColor = inColor;

    if (length(inTangent.xyz) > 0.0) {
        vec3 T = normalize(normal_matrix * inTangent.xyz);
        vec3 N = outNormal;
        T = normalize(T - dot(T, N) * N);
        vec3 B = cross(N, T) * inTangent.w;
        outTBN = mat3(T, B, N);
    } else {
        outTBN = mat3(1.0);
    }

    gl_Position = push.mvp * vec4(inPosition, 1.0);
}
