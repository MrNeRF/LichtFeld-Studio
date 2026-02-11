#version 430 core

in vec3 v_world_pos;
in vec3 v_normal;
in vec2 v_texcoord;
in vec4 v_color;
in mat3 v_tbn;

uniform vec3 u_camera_pos;
uniform vec3 u_light_dir;
uniform float u_light_intensity;
uniform float u_ambient;

uniform vec4 u_base_color;
uniform float u_metallic;
uniform float u_roughness;
uniform vec3 u_emissive;

uniform bool u_has_albedo_tex;
uniform bool u_has_normal_tex;
uniform bool u_has_metallic_roughness_tex;

uniform sampler2D u_albedo_tex;
uniform sampler2D u_normal_tex;
uniform sampler2D u_metallic_roughness_tex;

uniform bool u_has_vertex_colors;

uniform bool u_shadow_enabled;
uniform sampler2DShadow u_shadow_map;
uniform mat4 u_light_vp;

layout(location = 0) out vec4 frag_color;

const float PI = 3.14159265359;

float distribution_ggx(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;
    float denom = NdotH2 * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

float geometry_schlick_ggx(float NdotV, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

float geometry_smith(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    return geometry_schlick_ggx(NdotV, roughness) * geometry_schlick_ggx(NdotL, roughness);
}

vec3 fresnel_schlick(float cos_theta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

float calculate_shadow(vec3 world_pos) {
    vec4 light_space = u_light_vp * vec4(world_pos, 1.0);
    vec3 proj = light_space.xyz / light_space.w;
    proj = proj * 0.5 + 0.5;

    if (proj.z > 1.0)
        return 1.0;

    float shadow = 0.0;
    vec2 texel_size = 1.0 / textureSize(u_shadow_map, 0);
    for (int x = -1; x <= 1; ++x) {
        for (int y = -1; y <= 1; ++y) {
            vec3 sample_coord = vec3(proj.xy + vec2(x, y) * texel_size, proj.z);
            shadow += texture(u_shadow_map, sample_coord);
        }
    }
    return shadow / 9.0;
}

void main() {
    vec4 albedo = u_base_color;
    if (u_has_albedo_tex) {
        albedo *= texture(u_albedo_tex, v_texcoord);
    }
    if (u_has_vertex_colors) {
        albedo *= v_color;
    }

    float metallic = u_metallic;
    float roughness = u_roughness;
    float ao = 1.0;
    if (u_has_metallic_roughness_tex) {
        vec3 orm = texture(u_metallic_roughness_tex, v_texcoord).rgb;
        ao = orm.r;
        roughness *= orm.g;
        metallic *= orm.b;
    }
    roughness = max(roughness, 0.04);

    vec3 N = normalize(v_normal);
    if (u_has_normal_tex) {
        vec3 normal_sample = texture(u_normal_tex, v_texcoord).rgb * 2.0 - 1.0;
        N = normalize(v_tbn * normal_sample);
    }

    vec3 V = normalize(u_camera_pos - v_world_pos);
    vec3 L = normalize(u_light_dir);
    vec3 H = normalize(V + L);

    float NdotL = max(dot(N, L), 0.0);

    float shadow = 1.0;
    if (u_shadow_enabled)
        shadow = calculate_shadow(v_world_pos);

    vec3 F0 = mix(vec3(0.04), albedo.rgb, metallic);
    float NDF = distribution_ggx(N, H, roughness);
    float G = geometry_smith(N, V, L, roughness);
    vec3 F = fresnel_schlick(max(dot(H, V), 0.0), F0);

    vec3 kS = F;
    vec3 kD = (1.0 - kS) * (1.0 - metallic);

    vec3 diffuse = kD * albedo.rgb / PI;
    vec3 spec = (NDF * G * F) / (4.0 * max(dot(N, V), 0.0) * NdotL + 0.0001);

    vec3 Lo = (diffuse + spec) * NdotL * u_light_intensity * shadow;
    vec3 ambient = albedo.rgb * u_ambient * ao;
    vec3 color = ambient + Lo + u_emissive;

    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0 / 2.2));

    frag_color = vec4(color, albedo.a);
}
