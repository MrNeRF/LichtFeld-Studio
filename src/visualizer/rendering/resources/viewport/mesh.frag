#version 450
layout(location = 0) in vec3 inWorldPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexcoord;
layout(location = 3) in vec4 inColor;
layout(location = 4) in mat3 inTBN;

layout(location = 0) out vec4 outColor;

// Set 0 = per-frame light/camera state + shadow map.
layout(set = 0, binding = 0) uniform LightUbo {
    vec4 camera_pos;          // xyz = camera position, w unused
    vec4 light_dir;           // xyz = world-space light direction (normalized)
    vec4 params;              // x = intensity, y = ambient, z = shadow_enabled, w unused
    mat4 light_vp;            // light view-projection (for sampling shadow map)
} u_light;
layout(set = 0, binding = 1) uniform sampler2DShadow u_shadow_map;

// Set 1 = per-material state + textures.
layout(set = 1, binding = 0) uniform MaterialUbo {
    vec4 base_color;
    vec4 emissive_metallic;     // xyz = emissive, w = metallic
    vec4 roughness_flags;       // x roughness, y has_albedo, z has_normal, w has_metallic_roughness
    vec4 vertex_color_emphasis; // x has_vertex_colors, y is_emphasized, z dim_non_emphasized, w flash_intensity
} u_material;

layout(set = 1, binding = 1) uniform sampler2D u_albedo;
layout(set = 1, binding = 2) uniform sampler2D u_normal;
layout(set = 1, binding = 3) uniform sampler2D u_metallic_roughness;

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
    vec4 light_space = u_light.light_vp * vec4(world_pos, 1.0);
    vec3 proj = light_space.xyz / light_space.w;
    // Vulkan NDC y is flipped vs GL. The shadow map was rendered in the same Vulkan
    // pipeline so light_proj is already Vulkan-NDC; we only need to remap xy from
    // [-1,1] to [0,1] (z stays 0..1 in Vulkan).
    vec2 uv = proj.xy * 0.5 + 0.5;
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || proj.z > 1.0)
        return 1.0;

    float shadow = 0.0;
    vec2 texel_size = 1.0 / vec2(textureSize(u_shadow_map, 0));
    for (int x = -1; x <= 1; ++x) {
        for (int y = -1; y <= 1; ++y) {
            vec3 sample_coord = vec3(uv + vec2(x, y) * texel_size, proj.z);
            shadow += texture(u_shadow_map, sample_coord);
        }
    }
    return shadow / 9.0;
}

void main() {
    vec4 albedo = u_material.base_color;
    if (u_material.roughness_flags.y > 0.5) {
        albedo *= texture(u_albedo, inTexcoord);
    }
    if (u_material.vertex_color_emphasis.x > 0.5) {
        albedo *= inColor;
    }

    float metallic = u_material.emissive_metallic.w;
    float roughness = u_material.roughness_flags.x;
    float ao = 1.0;
    if (u_material.roughness_flags.w > 0.5) {
        vec3 orm = texture(u_metallic_roughness, inTexcoord).rgb;
        ao = orm.r;
        roughness *= orm.g;
        metallic *= orm.b;
    }
    roughness = max(roughness, 0.04);

    vec3 N = normalize(inNormal);
    if (u_material.roughness_flags.z > 0.5) {
        vec3 normal_sample = texture(u_normal, inTexcoord).rgb * 2.0 - 1.0;
        N = normalize(inTBN * normal_sample);
    }

    vec3 V = normalize(u_light.camera_pos.xyz - inWorldPos);
    vec3 L = normalize(u_light.light_dir.xyz);
    vec3 H = normalize(V + L);

    float NdotL = max(dot(N, L), 0.0);

    float shadow = 1.0;
    if (u_light.params.z > 0.5)
        shadow = calculate_shadow(inWorldPos);

    vec3 F0 = mix(vec3(0.04), albedo.rgb, metallic);
    float NDF = distribution_ggx(N, H, roughness);
    float G = geometry_smith(N, V, L, roughness);
    vec3 F = fresnel_schlick(max(dot(H, V), 0.0), F0);

    vec3 kS = F;
    vec3 kD = (1.0 - kS) * (1.0 - metallic);

    vec3 diffuse = kD * albedo.rgb / PI;
    vec3 spec = (NDF * G * F) / (4.0 * max(dot(N, V), 0.0) * NdotL + 0.0001);

    float light_intensity = u_light.params.x;
    float ambient_amount = u_light.params.y;

    vec3 Lo = (diffuse + spec) * NdotL * light_intensity * shadow;
    vec3 ambient = albedo.rgb * ambient_amount * ao;
    vec3 color = ambient + Lo + u_material.emissive_metallic.xyz;

    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0 / 2.2));

    bool is_emphasized = u_material.vertex_color_emphasis.y > 0.5;
    bool dim_non_emphasized = u_material.vertex_color_emphasis.z > 0.5;
    float flash_intensity = u_material.vertex_color_emphasis.w;

    if (dim_non_emphasized && !is_emphasized) {
        float luma = dot(color, vec3(0.2126, 0.7152, 0.0722));
        color = mix(color, vec3(luma), 0.6);
    }
    if (is_emphasized && flash_intensity > 0.0) {
        vec3 flash_color = vec3(1.0, 0.95, 0.6);
        color = mix(color, flash_color, flash_intensity * 0.5);
    }

    outColor = vec4(color, albedo.a);
}
