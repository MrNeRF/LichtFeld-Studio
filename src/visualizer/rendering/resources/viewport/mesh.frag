#version 450
layout(location = 0) in vec3 inWorldPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexcoord;
layout(location = 3) in vec4 inColor;
layout(location = 4) in mat3 inTBN;

layout(location = 0) out vec4 outColor;

// Set 0 = per-frame light/camera state + shadow map.
layout(set = 0, binding = 0) uniform LightUbo {
    vec4 camera_pos; // xyz = camera position, w unused
    vec4 light_dir;  // xyz = world-space light direction (normalized)
    vec4 params;     // x = intensity, y = ambient, z = shadow_enabled, w unused
    vec4 selection;  // x = emphasized, y = dim others, z = panorama, w = reserved
    mat4 light_vp;   // light view-projection (for sampling shadow map)
    vec4 camera_forward; // xyz = world-space view direction
    vec4 depth_view; // x = enabled, y = near, z = far, w = grayscale
    mat4 view;
    mat4 projection;
} u_light;
layout(set = 0, binding = 1) uniform sampler2DShadow u_shadow_map;

// Set 1 = per-material state + textures.
layout(set = 1, binding = 0) uniform MaterialUbo {
    vec4 base_color;
    vec4 emissive_metallic;     // xyz = emissive, w = metallic
    vec4 roughness_flags;       // x roughness, y has_albedo, z has_normal, w has_metallic_roughness
    vec4 vertex_color_flags; // x = has_vertex_colors, yzw reserved
} u_material;

layout(set = 1, binding = 1) uniform sampler2D u_albedo;
layout(set = 1, binding = 2) uniform sampler2D u_normal;
layout(set = 1, binding = 3) uniform sampler2D u_metallic_roughness;

const float PI = 3.14159265359;

float normalizedDepth(float depth, float lo, float hi) {
    lo = max(lo, 1.0e-4);
    hi = max(hi, lo + 1.0e-4);
    depth = clamp(depth, lo, hi);

    const float linear_t = clamp((depth - lo) / max(hi - lo, 1.0e-5), 0.0, 1.0);
    const float log_span = max(log2(hi / lo), 1.0e-4);
    const float log_t = clamp(log2(depth / lo) / log_span, 0.0, 1.0);
    const float log_weight = smoothstep(1.75, 24.0, hi / lo);
    return smoothstep(0.0, 1.0, mix(linear_t, log_t, log_weight));
}

vec3 depthPalette(float near_t) {
    near_t = clamp(near_t, 0.0, 1.0);
    const vec3 far_0 = vec3(0.050, 0.040, 0.150);
    const vec3 far_1 = vec3(0.060, 0.195, 0.500);
    const vec3 mid_0 = vec3(0.000, 0.500, 0.650);
    const vec3 mid_1 = vec3(0.360, 0.735, 0.410);
    const vec3 near_0 = vec3(0.965, 0.820, 0.300);
    const vec3 near_1 = vec3(0.985, 0.430, 0.125);
    if (near_t < 0.20) return mix(far_0, far_1, smoothstep(0.00, 0.20, near_t));
    if (near_t < 0.43) return mix(far_1, mid_0, smoothstep(0.20, 0.43, near_t));
    if (near_t < 0.67) return mix(mid_0, mid_1, smoothstep(0.43, 0.67, near_t));
    if (near_t < 0.86) return mix(mid_1, near_0, smoothstep(0.67, 0.86, near_t));
    return mix(near_0, near_1, smoothstep(0.86, 1.00, near_t));
}

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
    if (u_light.depth_view.x > 0.5) {
        float hi = u_light.depth_view.z;
        const float lo = u_light.depth_view.y;
        if (hi <= lo + 1.0e-5) hi = lo + 1.0;
        const float depth = dot(inWorldPos - u_light.camera_pos.xyz, u_light.camera_forward.xyz);
        const float near_t = 1.0 - normalizedDepth(depth, lo, hi);
        const vec3 color = u_light.depth_view.w > 0.5 ? vec3(near_t) : depthPalette(near_t);
        outColor = vec4(color, 1.0);
        return;
    }
    vec4 albedo = u_material.base_color;
    if (u_material.roughness_flags.y > 0.5) {
        albedo *= texture(u_albedo, inTexcoord);
    }
    if (u_material.vertex_color_flags.x > 0.5) {
        albedo *= inColor;
    }
    if (u_material.vertex_color_flags.y > 0.5) {
        if (albedo.a < u_material.vertex_color_flags.z) {
            discard;
        }
        albedo.a = 1.0;
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

    bool is_emphasized = u_light.selection.x > 0.5;
    bool dim_non_emphasized = u_light.selection.y > 0.5;

    if (dim_non_emphasized && !is_emphasized) {
        float luma = dot(color, vec3(0.2126, 0.7152, 0.0722));
        color = mix(color, vec3(luma), 0.6);
    }

    outColor = vec4(color, albedo.a);
}
