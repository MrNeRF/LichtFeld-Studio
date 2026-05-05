#version 450
layout(location = 0) in vec2 TexCoord;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D u_environment;

layout(push_constant) uniform EnvPush {
    mat4 cam_to_world;       // upper-left mat3 = camera rotation; w col unused
    vec4 intrinsics;         // focal_x, focal_y, cx, cy
    vec4 viewport_exposure;  // viewport.xy, exposure, rotation_radians
    vec4 flags;              // x = is_equirectangular_view
} push;

const float PI = 3.14159265358979323846;

vec3 rotate_y(vec3 v, float angle) {
    float c = cos(angle);
    float s = sin(angle);
    return vec3(c * v.x + s * v.z, v.y, -s * v.x + c * v.z);
}

vec3 aces_tonemap(vec3 x) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

vec3 ray_direction_local() {
    bool equirect = push.flags.x > 0.5;
    if (equirect) {
        float lon = (TexCoord.x - 0.5) * (2.0 * PI);
        float lat = (TexCoord.y - 0.5) * PI;
        float cos_lat = cos(lat);
        return normalize(vec3(sin(lon) * cos_lat, sin(lat), -cos(lon) * cos_lat));
    }
    vec2 viewport = push.viewport_exposure.xy;
    vec2 pixel = TexCoord * viewport;
    vec2 centered = vec2(
        (pixel.x - push.intrinsics.z) / max(push.intrinsics.x, 1e-6),
        (pixel.y - push.intrinsics.w) / max(push.intrinsics.y, 1e-6));
    return normalize(vec3(centered, -1.0));
}

vec2 environment_uv(vec3 dir) {
    float longitude = atan(dir.x, -dir.z);
    float latitude = asin(clamp(dir.y, -1.0, 1.0));
    return vec2(longitude / (2.0 * PI) + 0.5,
                0.5 - latitude / PI);
}

void main() {
    vec3 local_dir = ray_direction_local();
    mat3 rot = mat3(push.cam_to_world);
    vec3 world_dir = normalize(rot * local_dir);
    world_dir = normalize(rotate_y(world_dir, push.viewport_exposure.w));

    vec2 uv = environment_uv(world_dir);
    uv.x = fract(uv.x);
    uv.y = clamp(uv.y, 0.0, 1.0);
    vec3 color = textureLod(u_environment, uv, 0.0).rgb;
    color *= exp2(push.viewport_exposure.z);
    color = aces_tonemap(color);
    color = pow(color, vec3(1.0 / 2.2));
    outColor = vec4(color, 1.0);
}
