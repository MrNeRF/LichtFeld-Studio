#version 450
// VR hands, drawn over an eye image. Positions are in the tracking space.
layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;

layout(push_constant) uniform HandPush {
    mat4 view_projection;
    vec4 eye;   // xyz: the eye, in the tracking space
    vec4 color; // rgb: linear color of the light, a: strength
} pc;

layout(location = 0) out vec3 vNormal;
layout(location = 1) out vec3 vToEye;

// The depth pass and the color pass must agree to the last bit.
invariant gl_Position;

void main() {
    gl_Position = pc.view_projection * vec4(aPosition, 1.0);
    vNormal = aNormal;
    vToEye = pc.eye.xyz - aPosition;
}
