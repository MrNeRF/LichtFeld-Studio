#include "config.slang"

import utils;

static const uint SELECTION_MODE_BRUSH = 0;
static const uint SELECTION_MODE_RECTANGLE = 1;
static const uint SELECTION_MODE_POLYGON_MASK = 2;
static const uint SELECTION_MODE_RING = 3;
static const uint RING_PICK_PHASE_NONE = 0;
static const uint RING_PICK_PHASE_FIND_MIN = 1;
static const uint RING_PICK_PHASE_WRITE_PICK = 2;

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
    uint aabb_x0;
    uint aabb_y0;
    uint aabb_w;
    uint aabb_h;
    float ring_width;
    uint mip_filter;
    uint ring_pick_phase;
    uint pad2;
}

layout(binding=0) StructuredBuffer<float> xyz_ws;
layout(binding=1) ByteAddressBuffer transform_indices;
layout(binding=2) ByteAddressBuffer node_mask;
layout(binding=3) StructuredBuffer<float4> primitives;
layout(binding=4) StructuredBuffer<float4> model_transforms;
layout(binding=5) StructuredBuffer<float4> rotations;
layout(binding=6) StructuredBuffer<float> scaling_raw;
layout(binding=7) RWByteAddressBuffer selection_out;
layout(binding=8) ByteAddressBuffer polygon_mask;
layout(binding=9) StructuredBuffer<float> opacity_raw;
layout(binding=10) RWByteAddressBuffer ring_pick_out;

static const uint RING_PICK_NO_HIT = 0xFFFFFFFFu;

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
    float3 scales = exp(read_t3_float3(splat_idx, scaling_raw));
    float3x3 rotation = quat_to_rotmat(normalize(rotations[splat_idx]));

    float2 image_points[NUM_SIGMA_POINTS];
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

        image_points[i] = projected.xy;
    }

    unwrap_image_points_x(image_points, cam);

    float2 mean2d = float2(0.0);
    for (uint i = 0u; i < NUM_SIGMA_POINTS; ++i) {
        float weight = i == 0u ? weight_mean_0 : weight_sigma;
        mean2d += weight * image_points[i];
    }
    if (is_equirectangular_camera(cam)) {
        mean2d.x = wrap_image_x(mean2d.x, cam);
    }

    position = mean2d - 0.5f;
    return true;
}

[ForceInline]
bool project_gut_ring_splat(
    uint splat_idx,
    SelectionUniforms uniforms,
    Camera cam,
    float3 mean_ws,
    uint transform_base,
    bool transform_enabled,
    out float2 position,
    out float3 inv_cov_vs,
    out float opacity,
    out float depth)
{
    position = float2(0.0);
    inv_cov_vs = float3(0.0);
    opacity = 0.0f;
    depth = 0.0f;

    float3 mean_vs = project_point(mean_ws, cam);
    if (mean_vs.z <= NEAR_CLIP || !isfinite(dot(mean_vs, mean_vs)))
        return false;
    depth = mean_vs.z;

    static const uint D = 3u;
    static const uint NUM_SIGMA_POINTS = 7u;
    static const float UT_ALPHA = 0.1f;
    static const float UT_BETA = 2.0f;
    static const float UT_KAPPA = 0.0f;
    float lambda = UT_ALPHA * UT_ALPHA * (float(D) + UT_KAPPA) - float(D);
    float sigma_scale = sqrt(float(D) + lambda);
    float weight_mean_0 = lambda / (float(D) + lambda);
    float weight_cov_0 = weight_mean_0 + (1.0f - UT_ALPHA * UT_ALPHA + UT_BETA);
    float weight_sigma = 1.0f / (2.0f * (float(D) + lambda));

    float3 source_mean = read_t3_float3(splat_idx, xyz_ws);
    float3 raw_scaling = min(read_t3_float3(splat_idx, scaling_raw), float3(MAX_RAW_SCALE));
    float3 scales = exp(raw_scaling);
    float3x3 rotation = quat_to_rotmat(safe_quat_normalize(rotations[splat_idx]));

    float3 sigma_points[NUM_SIGMA_POINTS];
    sigma_points[0] = mean_ws;
    for (uint i = 0u; i < D; ++i) {
        float scale_i = i == 0u ? scales.x : (i == 1u ? scales.y : scales.z);
        float3 delta = sigma_scale * scale_i * selection_rotation_column(rotation, i);
        float3 source_plus = source_mean + delta;
        float3 source_minus = source_mean - delta;
        sigma_points[1u + i] = transform_enabled
                                    ? transform_point_rows(model_transforms, transform_base, source_plus)
                                    : source_plus;
        sigma_points[1u + D + i] = transform_enabled
                                        ? transform_point_rows(model_transforms, transform_base, source_minus)
                                        : source_minus;
    }

    float2 image_points[NUM_SIGMA_POINTS];
    for (uint i = 0u; i < NUM_SIGMA_POINTS; ++i) {
        float3 projected = project_point(sigma_points[i], cam);
        if (!valid_gut_selection_point(projected, cam))
            return false;
        image_points[i] = projected.xy;
    }

    unwrap_image_points_x(image_points, cam);

    float2 mean2d = float2(0.0);
    for (uint i = 0u; i < NUM_SIGMA_POINTS; ++i) {
        float weight = i == 0u ? weight_mean_0 : weight_sigma;
        mean2d += weight * image_points[i];
    }
    float2 covariance_mean2d = mean2d;
    if (is_equirectangular_camera(cam)) {
        mean2d.x = wrap_image_x(mean2d.x, cam);
    }

    float3 cov2d = float3(0.0);
    for (uint i = 0u; i < NUM_SIGMA_POINTS; ++i) {
        float2 d = image_points[i] - covariance_mean2d;
        float weight = i == 0u ? weight_cov_0 : weight_sigma;
        cov2d += weight * float3(d.x * d.x, d.x * d.y, d.y * d.y);
    }

    float4 cov_vs = add_blur_to_covariance2d(cov2d, (uniforms.mip_filter & 1u) != 0u, false);
    float det = compute_det(cov_vs.xyz);
    if (det <= 0.0f || !isfinite(det))
        return false;

    opacity = sigmoid_activation(opacity_raw[splat_idx]) * cov_vs.w;
    if (opacity <= ALPHA_THRESHOLD)
        return false;

    position = mean2d - 0.5f;
    inv_cov_vs = compute_inverse(cov_vs.xyz);
    return isfinite(dot(inv_cov_vs, inv_cov_vs));
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

    if (is_equirectangular_camera(cam)) {
        projected.x = wrap_image_x(projected.x, cam);
    }
    position = projected.xy - 0.5;
    return true;
}

[ForceInline]
bool project_ring_splat(
    uint splat_idx,
    SelectionUniforms uniforms,
    out float2 position,
    out float3 inv_cov_vs,
    out float opacity,
    out float depth)
{
    position = float2(0.0);
    inv_cov_vs = float3(0.0);
    opacity = 0.0f;
    depth = 0.0f;
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
        return project_gut_ring_splat(
            splat_idx, uniforms, cam, mean, transform_base, transform_enabled, position, inv_cov_vs, opacity, depth);
    }

    float3 projected = project_point(mean, cam);
    if (projected.z <= NEAR_CLIP ||
        !isfinite(dot(projected, projected)) ||
        !image_point_in_image_bounds_margin(projected.xy, cam, 0.2f)) {
        return false;
    }
    if (is_equirectangular_camera(cam)) {
        projected.x = wrap_image_x(projected.x, cam);
    }
    position = projected.xy - 0.5f;
    depth = projected.z;

    float3 raw_scaling = min(read_t3_float3(splat_idx, scaling_raw), float3(MAX_RAW_SCALE));
    float3 scales = exp(raw_scaling);
    float3x3 cov_ws = get_covariance_from_quat_scales(
        safe_quat_normalize(rotations[splat_idx]), scales);
    if (transform_enabled) {
        cov_ws = transform_covariance_rows(model_transforms, transform_base, cov_ws);
    }
    float4 cov_vs = covariance_3d_to_2d(cam, mean, cov_ws, (uniforms.mip_filter & 1u) != 0u, false);
    opacity = sigmoid_activation(opacity_raw[splat_idx]) * cov_vs.w;
    if (opacity <= ALPHA_THRESHOLD) {
        return false;
    }

    float3 cov_projected = clamp_covariance_2d_extent(cov_vs.xyz, opacity);
    float det = compute_det(cov_projected);
    if (det <= 0.0f || !isfinite(det)) {
        return false;
    }
    inv_cov_vs = compute_inverse(cov_projected);
    return isfinite(dot(inv_cov_vs, inv_cov_vs));
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

[ForceInline]
float ring_sample_line_distance(
    float2 sample,
    float2 position,
    float3 inv_cov,
    float opacity,
    SelectionUniforms uniforms,
    out bool hit)
{
    hit = false;
    float2 delta = sample - position;
    float sigma_over_2 =
        0.5f * (inv_cov.x * delta.x * delta.x + inv_cov.z * delta.y * delta.y) +
        inv_cov.y * delta.x * delta.y;
    if (sigma_over_2 < 0.0f || !isfinite(sigma_over_2)) {
        return 1.0e20f;
    }

    float gaussian = exp(-sigma_over_2);
    float boundary_gaussian = ALPHA_THRESHOLD / max(opacity, eps);
    float ring_band = max(uniforms.ring_width, 0.0f) * 10.0f;
    float ring_outer = boundary_gaussian * (1.0f + ring_band);
    float ring_inner = boundary_gaussian * max(0.0f, 1.0f - ring_band);
    hit = gaussian < ring_outer && gaussian > ring_inner;
    return abs(sigma_over_2 + log(max(boundary_gaussian, eps)));
}

[ForceInline]
bool ring_primitive_hit(
    float2 position,
    float3 inv_cov,
    float opacity,
    float4 primitive,
    SelectionUniforms uniforms)
{
    bool sample_hit;
    float2 sample = primitive.xy;
    ring_sample_line_distance(sample, position, inv_cov, opacity, uniforms, sample_hit);
    if (sample_hit) {
        return true;
    }

    float padding = max(primitive.z, 0.0f);
    if (padding > 0.0f) {
        ring_sample_line_distance(sample + float2(padding, 0.0f), position, inv_cov, opacity, uniforms, sample_hit);
        if (sample_hit) {
            return true;
        }
        ring_sample_line_distance(sample - float2(padding, 0.0f), position, inv_cov, opacity, uniforms, sample_hit);
        if (sample_hit) {
            return true;
        }
        ring_sample_line_distance(sample + float2(0.0f, padding), position, inv_cov, opacity, uniforms, sample_hit);
        if (sample_hit) {
            return true;
        }
        ring_sample_line_distance(sample - float2(0.0f, padding), position, inv_cov, opacity, uniforms, sample_hit);
        if (sample_hit) {
            return true;
        }
    }

    return false;
}

[ForceInline]
uint ring_depth_pick_key(float depth)
{
    if (!isfinite(depth) || depth <= 0.0f) {
        return RING_PICK_NO_HIT;
    }
    return asuint(depth);
}

[ForceInline]
bool update_ring_pick(uint depth_key, uint splat_idx, SelectionUniforms uniforms)
{
    if (depth_key == RING_PICK_NO_HIT) {
        return false;
    }

    if (uniforms.ring_pick_phase == RING_PICK_PHASE_NONE) {
        return true;
    }

    if (uniforms.ring_pick_phase == RING_PICK_PHASE_FIND_MIN) {
        uint old_key;
        ring_pick_out.InterlockedMin(0u, depth_key, old_key);
        return false;
    }

    if (uniforms.ring_pick_phase == RING_PICK_PHASE_WRITE_PICK) {
        if (ring_pick_out.Load(0u) != depth_key) {
            return false;
        }
        uint old_id;
        ring_pick_out.InterlockedMin(4u, splat_idx, old_id);
        return true;
    }

    return false;
}

[ForceInline]
bool polygon_mask_hit(float2 position, SelectionUniforms uniforms)
{
    int ix = int(floor(position.x)) - int(uniforms.aabb_x0);
    int iy = int(floor(position.y)) - int(uniforms.aabb_y0);
    if (ix < 0 || iy < 0 || uint(ix) >= uniforms.aabb_w || uint(iy) >= uniforms.aabb_h)
        return false;
    uint idx = uint(iy) * uniforms.aabb_w + uint(ix);
    return read_byte(polygon_mask, idx) != 0u;
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

        bool selected = false;
        if (uniforms.mode == SELECTION_MODE_RING) {
            float2 position;
            float3 inv_cov_vs;
            float opacity;
            float depth;
            if (project_ring_splat(splat_idx, uniforms, position, inv_cov_vs, opacity, depth)) {
                for (uint primitive_idx = 0u; primitive_idx < uniforms.primitive_count; ++primitive_idx) {
                    if (ring_primitive_hit(position, inv_cov_vs, opacity, primitives[primitive_idx], uniforms)) {
                        selected = true;
                        break;
                    }
                }
                if (selected) {
                    selected = update_ring_pick(ring_depth_pick_key(depth), splat_idx, uniforms);
                }
            }
        } else {
            float2 position;
            if (project_splat(splat_idx, uniforms, position)) {
                if (uniforms.mode == SELECTION_MODE_POLYGON_MASK) {
                    selected = polygon_mask_hit(position, uniforms);
                } else {
                    for (uint primitive_idx = 0u; primitive_idx < uniforms.primitive_count; ++primitive_idx) {
                        if (primitive_hit(position, primitives[primitive_idx], uniforms.mode)) {
                            selected = true;
                            break;
                        }
                    }
                }
            }
        }

        if (selected)
            packed |= 1u << (lane * 8u);
    }

    selection_out.Store(word_idx * 4u, packed);
}
