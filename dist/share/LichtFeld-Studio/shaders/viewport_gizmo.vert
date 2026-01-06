#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
uniform mat4 uMVP;
uniform mat4 uModel;
uniform vec3 uColor;
uniform float uAlpha;
out vec3 vNormal;
out vec3 vColor;
out float vAlpha;
void main() {
    vNormal = mat3(uModel) * aNormal;
    vColor = uColor;
    vAlpha = uAlpha;
    gl_Position = uMVP * vec4(aPos, 1.0);
}
