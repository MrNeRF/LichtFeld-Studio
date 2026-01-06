#version 330 core
in vec3 vNormal;
in vec3 vColor;
in float vAlpha;
uniform bool uUseLighting;
out vec4 FragColor;
void main() {
    float light = uUseLighting ? 0.4 + 0.6 * max(dot(normalize(vNormal), normalize(vec3(0.5, 0.7, 0.3))), 0.0) : 1.0;
    FragColor = vec4(vColor * light, vAlpha);
}
