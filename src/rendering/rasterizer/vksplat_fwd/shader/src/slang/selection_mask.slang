#include "config.slang"

import utils;

static const uint SELECTION_MODE_BRUSH = 0;
static const uint SELECTION_MODE_RECTANGLE = 1;

struct SelectionUniforms
{
    uint num_splats;
    uint primitive_count;
    uint mode;
    uint transform_indices_enabled;
    uint node_visibility_enabled;
    uint node_visibility_count;
    uint num_model_transforms;
    uint image_height;
    uint image_width;
    uint camera_model;
    uint pad0;
    uint pad1;
    float fx;
    float fy;
    float cx;
    float cy;
    float4 dist_coeffs;
    float4x4 world_view_transform;
}

layout(binding=0) StructuredBuffer<float> xyz_ws;
layout(binding=1) ByteAddressBuffer transform_indices;
layout(binding=2) ByteAddressBuffer node_mask;
layout(binding=3) StructuredBuffer<float4> primitives;
layout(binding=4) StructuredBuffer<float4> model_transforms;
layout(binding=5) StructuredBuffer<float4> rotations;
layout(binding=6) StructuredBuffer<float4> scales_opacs;
layout(binding=7) RWByteAddressBuffer selection_out;

[ForceInline]
bool node_visible(uint splat_idx, SelectionUniforms uniforms)
{
    if (uniforms.transform_indices_enabled == 0 || uniforms.node_visibility_enabled == 0)
        return true;

    int node_idx = read_int32(transform_indices, splat_idx);
    if (node_idx < 0 || uint(node_idx) >= uniforms.node_visibility_count)
        return false;

    return read_byte(node_mask, uint(node_idx)) != 0;
}

[ForceInline]
uint model_transform_base(uint splat_idx, SelectionUniforms uniforms)
{
    if (uniforms.num_model_transforms == 0)
        return 0u;

    int transform_idx = uniforms.transform_indices_enabled != 0
                            ? read_int32(transform_indices, splat_idx)
                            : 0;
    transform_idx = clamp(transform_idx, 0, int(uniforms.num_model_transforms) - 1);
    return uint(transform_idx) * 4u;
}

[ForceInline]
float3 selection_rotation_column(float3x3 rotation, uint column)
{
    if (column == 0u)
        return float3(rotation[0][0], rotation[1][0], rotation[2][0]);
    if (column == 1u)
        return float3(rotation[0][1], rotation[1][1], rotation[2][1]);
    return float3(rotation[0][2], rotation[1][2], rotation[2][2]);
}

[ForceInline]
bool valid_gut_selection_point(float3 projected, Camera cam)
{
    return projected.z > 0.0f &&
           isfinite(dot(projected, projected)) &&
           image_point_in_image_bounds_margin(projected.xy, cam, 0.1f);
}

[ForceInline]
bool project_gut_splat(
    uint splat_idx,
    SelectionUniforms uniforms,
    Camera cam,
    float3 mean_ws,
    uint transform_base,
    bool transform_enabled,
    out float2 position)
{
    position = float2(0.0);
    float3 mean_vs = project_point(mean_ws, cam);
    if (mean_vs.z <= NEAR_CLIP || !isfinite(dot(mean_vs, mean_vs)))
        return false;

    static const uint D = 3u;
    static const uint NUM_SIGMA_POINTS = 7u;
    static const float UT_ALPHA = 0.1f;
    static const float UT_BETA = 2.0f;
    static const float UT_KAPPA = 0.0f;
    float lambda = UT_ALPHA * UT_ALPHA * (float(D) + UT_KAPPA) - float(D);
    float sigma_scale = sqrt(float(D) + lambda);
    float weight_mean_0 = lambda / (float(D) + lambda);
    float weight_sigma = 1.0f / (2.0f * (float(D) + lambda));

    float3 source_mean = read_t3_float3(splat_idx, xyz_ws);
    float3 scales = scales_opacs[splat_idx].xyz;
    float3x3 rotation = quat_to_rotmat(normalize(rotations[splat_idx]));

    float2 mean2d = float2(0.0);
    for (uint i = 0u; i < NUM_SIGMA_POINTS; ++i) {
        float3 sigma_point = source_mean;
        if (i > 0u) {
            uint axis = (i - 1u) % D;
            float scale_i = axis == 0u ? scales.x : (axis == 1u ? scales.y : scales.z);
            float sign_i = i <= D ? 1.0f : -1.0f;
            sigma_point += sign_i * sigma_scale * scale_i * selection_rotation_column(rotation, axis);
        }
        if (transform_enabled) {
            sigma_point = transform_point_rows(model_transforms, transform_base, sigma_point);
        }

        float3 projected = project_point(sigma_point, cam);
        if (!valid_gut_selection_point(projected, cam))
            return false;

        float weight = i == 0u ? weight_mean_0 : weight_sigma;
        mean2d += weight * projected.xy;
    }

    position = mean2d - 0.5f;
    return true;
}

[ForceInline]
bool project_splat(uint splat_idx, SelectionUniforms uniforms, out float2 position)
{
    position = float2(0.0);
    if (!node_visible(splat_idx, uniforms))
        return false;

    Camera cam = load_camera(uniforms.world_view_transform,
        uniforms.fx, uniforms.fy, uniforms.cx, uniforms.cy,
        int(base_camera_model(uniforms.camera_model)), uniforms.dist_coeffs,
        uniforms.image_height, uniforms.image_width);
    float3 mean = read_t3_float3(splat_idx, xyz_ws);
    uint transform_base = model_transform_base(splat_idx, uniforms);
    bool transform_enabled = uniforms.num_model_transforms > 0;
    if (transform_enabled) {
        mean = transform_point_rows(model_transforms, transform_base, mean);
    }
    if (is_gut_projection_mode(uniforms.camera_model)) {
        return project_gut_splat(splat_idx, uniforms, cam, mean, transform_base, transform_enabled, position);
    }

    float3 projected = project_point(mean, cam);
    if (projected.z <= NEAR_CLIP)
        return false;

    position = projected.xy - 0.5;
    return true;
}

[ForceInline]
bool primitive_hit(float2 position, float4 primitive, uint mode)
{
    if (mode == SELECTION_MODE_RECTANGLE) {
        return position.x >= primitive.x && position.x <= primitive.z &&
               position.y >= primitive.y && position.y <= primitive.w;
    }

    float2 delta = position - primitive.xy;
    return dot(delta, delta) <= primitive.z;
}

[shader("compute")]
[numthreads(SUBGROUP_SIZE, 1, 1)]
void main(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform SelectionUniforms uniforms)
{
    uint word_idx = dispatchThreadID.x;
    uint base_idx = word_idx * 4u;
    if (base_idx >= uniforms.num_splats)
        return;

    uint packed = 0u;
    for (uint lane = 0u; lane < 4u; ++lane) {
        uint splat_idx = base_idx + lane;
        if (splat_idx >= uniforms.num_splats)
            break;

        float2 position;
        bool selected = false;
        if (project_splat(splat_idx, uniforms, position)) {
            for (uint primitive_idx = 0u; primitive_idx < uniforms.primitive_count; ++primitive_idx) {
                if (primitive_hit(position, primitives[primitive_idx], uniforms.mode)) {
                    selected = true;
                    break;
                }
            }
        }

        if (selected)
            packed |= 1u << (lane * 8u);
    }

    selection_out.Store(word_idx * 4u, packed);
}
