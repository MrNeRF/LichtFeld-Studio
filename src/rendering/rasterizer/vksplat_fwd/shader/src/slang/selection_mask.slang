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
    uint pad0;
    uint pad1;
    uint pad2;
    float fx;
    float fy;
    float cx;
    float cy;
    float4x4 world_view_transform;
}

layout(binding=0) StructuredBuffer<float> xyz_ws;
layout(binding=1) ByteAddressBuffer transform_indices;
layout(binding=2) ByteAddressBuffer node_mask;
layout(binding=3) StructuredBuffer<float4> primitives;
layout(binding=4) StructuredBuffer<float4> model_transforms;
layout(binding=5) RWByteAddressBuffer selection_out;

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
bool project_splat(uint splat_idx, SelectionUniforms uniforms, out float2 position)
{
    position = float2(0.0);
    if (!node_visible(splat_idx, uniforms))
        return false;

    Camera cam = load_camera(uniforms.world_view_transform,
        uniforms.fx, uniforms.fy, uniforms.cx, uniforms.cy,
        int(CameraModel::PINHOLE), float4(0.0),
        uniforms.image_height, uniforms.image_width);
    float3 mean = read_t3_float3(splat_idx, xyz_ws);
    if (uniforms.num_model_transforms > 0) {
        int transform_idx = uniforms.transform_indices_enabled != 0
                                ? read_int32(transform_indices, splat_idx)
                                : 0;
        transform_idx = clamp(transform_idx, 0, int(uniforms.num_model_transforms) - 1);
        mean = transform_point_rows(model_transforms, uint(transform_idx) * 4u, mean);
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
