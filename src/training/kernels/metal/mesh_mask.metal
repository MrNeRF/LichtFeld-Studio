// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later

// Mesh coverage for evaluation masks: port of mesh_mask.cu. mesh_coverage_prepare rasterizes
// small faces and queues large ones; mesh_coverage_large drains that queue with one threadgroup
// per face. The library builds with fast math, so finiteness is tested on the bits.

constant constexpr int kMeshCoverageMaxVertices = 8;
constant constexpr int kMeshCoverageSmallBoxPixels = 64;
constant constexpr float kMeshCoverageDistortedMargin = 2.0f;
constant constexpr int kMeshCoverageMaxEdgeSamples = 65536;

// MeshMaskCameraBlock in ops/mesh_mask_camera.hpp.
struct MeshCoverageCamera {
    float rows[12];
    float fx, fy, cx, cy;
    int width, height;
    float guard_min_x, guard_max_x, guard_min_y, guard_max_y;
    int distorted, model, num_distortion;
    float distortion[12];
    float src_fx, src_fy, src_cx, src_cy, dst_fx, dst_fy;
};

struct MeshCoverageParams {
    device const float* vertices;
    device const int* indices;
    device const float* samples; // [H,W,2], distorted cameras only
    device int* large_faces;
    device int* counters; // [0] queued large faces, [1] next to claim
    device uchar* mask;
    int vertex_count;
    int face_count;
    float z_near;
    int padding;
    MeshCoverageCamera camera;
};

struct MeshCoveragePolygon {
    float2 vertices[kMeshCoverageMaxVertices];
    int count;
    int min_x, max_x, min_y, max_y;
};

static bool mesh_coverage_finite(const float v) { return (as_type<uint>(v) & 0x7f800000u) != 0x7f800000u; }

// Forward lens models of core/cuda/undistort/distortion_model.cuh.
static float2 mesh_coverage_pinhole(const float x, const float y, constant float* dist, const int n) {
    const float r2 = x * x + y * y, r4 = r2 * r2, r6 = r4 * r2;
    const float k1 = n > 0 ? dist[0] : 0.0f, k2 = n > 1 ? dist[1] : 0.0f, k3 = n > 2 ? dist[2] : 0.0f;
    const float numerator = 1.0f + k1 * r2 + k2 * r4 + k3 * r6;
    const float radial = n >= 6 ? numerator / (1.0f + dist[3] * r2 + dist[4] * r4 + dist[5] * r6) : numerator;
    const int tangential = n >= 6 ? 6 : 3;
    const float p1 = n > tangential ? dist[tangential] : 0.0f;
    const float p2 = n > tangential + 1 ? dist[tangential + 1] : 0.0f;
    return float2(x * radial + 2.0f * p1 * x * y + p2 * (r2 + 2.0f * x * x),
                  y * radial + p1 * (r2 + 2.0f * y * y) + 2.0f * p2 * x * y);
}

static float2 mesh_coverage_fisheye(const float x, const float y, constant float* dist, const int n) {
    const float r = sqrt(x * x + y * y);
    if (r < 1e-8f)
        return float2(x, y);
    const float theta = atan(r), theta2 = theta * theta, theta4 = theta2 * theta2, theta6 = theta4 * theta2,
                theta8 = theta4 * theta4;
    const float k1 = n > 0 ? dist[0] : 0.0f, k2 = n > 1 ? dist[1] : 0.0f, k3 = n > 2 ? dist[2] : 0.0f,
                k4 = n > 3 ? dist[3] : 0.0f;
    const float scale = theta * (1.0f + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8) / r;
    return float2(x * scale, y * scale);
}

static float2 mesh_coverage_thin_prism_fisheye(const float x, const float y, constant float* dist, const int n) {
    const float r = sqrt(x * x + y * y);
    if (r < 1e-8f)
        return float2(x, y);
    const float scale = atan(r) / r;
    const float uu = x * scale, vv = y * scale;
    const float u2 = uu * uu, uv = uu * vv, v2 = vv * vv;
    const float r2 = u2 + v2, r4 = r2 * r2, r6 = r4 * r2, r8 = r6 * r2;
    const float k1 = n > 0 ? dist[0] : 0.0f, k2 = n > 1 ? dist[1] : 0.0f;
    const float k3 = n > 2 ? dist[2] : 0.0f, k4 = n > 3 ? dist[3] : 0.0f;
    const float p1 = n > 4 ? dist[4] : 0.0f, p2 = n > 5 ? dist[5] : 0.0f;
    const float sx1 = n > 6 ? dist[6] : 0.0f, sx2 = n > 7 ? dist[7] : 0.0f;
    const float sy1 = n > 8 ? dist[8] : 0.0f, sy2 = n > 9 ? dist[9] : 0.0f;
    const float radial = k1 * r2 + k2 * r4 + k3 * r6 + k4 * r8;
    return float2(uu + uu * radial + 2.0f * p1 * uv + p2 * (r2 + 2.0f * u2) + sx1 * r2 + sx2 * r4,
                  vv + vv * radial + 2.0f * p2 * uv + p1 * (r2 + 2.0f * v2) + sy1 * r2 + sy2 * r4);
}

static float2 mesh_coverage_distort(constant MeshCoverageCamera& c, const float x, const float y) {
    if (c.model == 0)
        return mesh_coverage_pinhole(x, y, c.distortion, c.num_distortion);
    if (c.model == 2)
        return mesh_coverage_fisheye(x, y, c.distortion, c.num_distortion);
    if (c.model == 4)
        return mesh_coverage_thin_prism_fisheye(x, y, c.distortion, c.num_distortion);
    return float2(x, y);
}

static float3 mesh_coverage_transform(constant MeshCoverageCamera& c, const float3 p) {
    return float3(c.rows[0] * p.x + c.rows[1] * p.y + c.rows[2] * p.z + c.rows[3],
                  c.rows[4] * p.x + c.rows[5] * p.y + c.rows[6] * p.z + c.rows[7],
                  c.rows[8] * p.x + c.rows[9] * p.y + c.rows[10] * p.z + c.rows[11]);
}

static float mesh_coverage_plane_distance(constant MeshCoverageCamera& c, const float3 p, const int plane,
                                          const float z_near) {
    switch (plane) {
    case 0:
        return p.z - z_near;
    case 1:
        return p.x - c.guard_min_x * p.z;
    case 2:
        return c.guard_max_x * p.z - p.x;
    case 3:
        return p.y - c.guard_min_y * p.z;
    default:
        return c.guard_max_y * p.z - p.y;
    }
}

static int mesh_coverage_clip(constant MeshCoverageCamera& c, thread const float3* input, const int input_count,
                              thread float3* output, const int plane, const float z_near) {
    int output_count = 0;
    float3 previous = input[input_count - 1];
    float previous_distance = mesh_coverage_plane_distance(c, previous, plane, z_near);
    bool previous_inside = previous_distance >= 0.0f;
    for (int i = 0; i < input_count; ++i) {
        const float3 current = input[i];
        const float current_distance = mesh_coverage_plane_distance(c, current, plane, z_near);
        const bool current_inside = current_distance >= 0.0f;
        if (current_inside != previous_inside) {
            const float t = previous_distance / (previous_distance - current_distance);
            output[output_count++] = previous + t * (current - previous);
        }
        if (current_inside)
            output[output_count++] = current;
        previous = current;
        previous_distance = current_distance;
        previous_inside = current_inside;
    }
    return output_count;
}

static bool mesh_coverage_bounds(constant MeshCoverageCamera& c, thread MeshCoveragePolygon& polygon) {
    bool found = false;
    float min_x = FLT_MAX, max_x = -FLT_MAX, min_y = FLT_MAX, max_y = -FLT_MAX;
    if (c.distorted == 0) {
        for (int i = 0; i < polygon.count; ++i) {
            const float px = polygon.vertices[i].x * c.fx + c.cx;
            const float py = polygon.vertices[i].y * c.fy + c.cy;
            if (!mesh_coverage_finite(px) || !mesh_coverage_finite(py))
                return false;
            min_x = fmin(min_x, px);
            max_x = fmax(max_x, px);
            min_y = fmin(min_y, py);
            max_y = fmax(max_y, py);
            found = true;
        }
    } else {
        const float sample_fx = fmax(fabs(c.src_fx), fabs(c.dst_fx));
        const float sample_fy = fmax(fabs(c.src_fy), fabs(c.dst_fy));
        for (int edge = 0; edge < polygon.count; ++edge) {
            const float2 a = polygon.vertices[edge];
            const float2 b = polygon.vertices[(edge + 1) % polygon.count];
            const float edge_pixels = fmax(fabs(b.x - a.x) * sample_fx, fabs(b.y - a.y) * sample_fy);
            const int steps = max(1, min(kMeshCoverageMaxEdgeSamples, int(ceil(edge_pixels))));
            for (int step = 0; step <= steps; ++step) {
                const float t = float(step) / float(steps);
                const float2 d = mesh_coverage_distort(c, a.x + t * (b.x - a.x), a.y + t * (b.y - a.y));
                const float px = d.x * c.src_fx + c.src_cx;
                const float py = d.y * c.src_fy + c.src_cy;
                if (!mesh_coverage_finite(px) || !mesh_coverage_finite(py))
                    continue;
                min_x = fmin(min_x, px);
                max_x = fmax(max_x, px);
                min_y = fmin(min_y, py);
                max_y = fmax(max_y, py);
                found = true;
            }
        }
    }
    if (!found)
        return false;
    const float margin = c.distorted != 0 ? kMeshCoverageDistortedMargin : 0.0f;
    const float first_x = min_x - 0.5f - margin;
    const float last_x = max_x - 0.5f + margin;
    const float first_y = min_y - 0.5f - margin;
    const float last_y = max_y - 0.5f + margin;
    if (last_x < 0.0f || last_y < 0.0f || first_x > float(c.width - 1) || first_y > float(c.height - 1))
        return false;
    polygon.min_x = max(0, int(floor(fmax(first_x, -1.0f))));
    polygon.max_x = min(c.width - 1, int(ceil(fmin(last_x, float(c.width)))));
    polygon.min_y = max(0, int(floor(fmax(first_y, -1.0f))));
    polygon.max_y = min(c.height - 1, int(ceil(fmin(last_y, float(c.height)))));
    return polygon.min_x <= polygon.max_x && polygon.min_y <= polygon.max_y;
}

static bool mesh_coverage_prepare_face(constant MeshCoverageParams& p, const int face,
                                       thread MeshCoveragePolygon& polygon) {
    constant MeshCoverageCamera& c = p.camera;
    float3 clipped_a[kMeshCoverageMaxVertices];
    float3 clipped_b[kMeshCoverageMaxVertices];
    for (int corner = 0; corner < 3; ++corner) {
        const int index = p.indices[3 * face + corner];
        if (index < 0 || index >= p.vertex_count)
            return false;
        clipped_a[corner] = mesh_coverage_transform(
            c, float3(p.vertices[3 * index], p.vertices[3 * index + 1], p.vertices[3 * index + 2]));
    }
    thread float3* input = clipped_a;
    thread float3* output = clipped_b;
    int count = 3;
    for (int plane = 0; plane < 5; ++plane) {
        count = mesh_coverage_clip(c, input, count, output, plane, p.z_near);
        if (count < 3)
            return false;
        thread float3* swapped = input;
        input = output;
        output = swapped;
    }

    polygon.count = count;
    for (int i = 0; i < count; ++i) {
        const float inverse_z = 1.0f / input[i].z;
        polygon.vertices[i] = float2(input[i].x * inverse_z, input[i].y * inverse_z);
    }
    float twice_area = 0.0f;
    for (int i = 0; i < count; ++i) {
        const float2 a = polygon.vertices[i];
        const float2 b = polygon.vertices[(i + 1) % count];
        twice_area += a.x * b.y - a.y * b.x;
    }
    if (!mesh_coverage_finite(twice_area) || twice_area == 0.0f)
        return false;
    if (twice_area < 0.0f) {
        for (int i = 0; i < count / 2; ++i) {
            const float2 swapped = polygon.vertices[i];
            polygon.vertices[i] = polygon.vertices[count - 1 - i];
            polygon.vertices[count - 1 - i] = swapped;
        }
    }
    return mesh_coverage_bounds(c, polygon);
}

static float mesh_coverage_edge(const float2 a, const float2 b, const float2 point) {
    return fma(b.x - a.x, point.y - a.y, -(b.y - a.y) * (point.x - a.x));
}

static bool mesh_coverage_inside_edge(const float2 a, const float2 b, const float2 point) {
    const float dx = b.x - a.x, dy = b.y - a.y;
    const float px = point.x - a.x, py = point.y - a.y;
    const float tolerance = 4.0f * 1.1920928955078125e-7f * (fabs(dx * py) + fabs(dy * px));
    return fma(dx, py, -dy * px) >= -tolerance;
}

static bool mesh_coverage_inside(thread const MeshCoveragePolygon& polygon, const float2 point) {
    const float2 first = polygon.vertices[0];
    for (int triangle = 1; triangle + 1 < polygon.count; ++triangle) {
        const float2 second = polygon.vertices[triangle];
        const float2 third = polygon.vertices[triangle + 1];
        if (mesh_coverage_edge(first, second, third) == 0.0f)
            continue;
        if (mesh_coverage_inside_edge(first, second, point) && mesh_coverage_inside_edge(second, third, point) &&
            mesh_coverage_inside_edge(third, first, point))
            return true;
    }
    return false;
}

// Every write stores 1, so racing writers need no atomics.
static void mesh_coverage_pixel(constant MeshCoverageParams& p, thread const MeshCoveragePolygon& polygon,
                                const int x, const int y) {
    constant MeshCoverageCamera& c = p.camera;
    float2 ray;
    if (c.distorted != 0) {
        const int index = 2 * (y * c.width + x);
        ray = float2(p.samples[index], p.samples[index + 1]);
    } else {
        ray = float2((float(x) + 0.5f - c.cx) / c.fx, (float(y) + 0.5f - c.cy) / c.fy);
    }
    if (mesh_coverage_finite(ray.x) && mesh_coverage_finite(ray.y) && mesh_coverage_inside(polygon, ray))
        p.mask[y * c.width + x] = 1;
}

kernel void mesh_coverage_prepare(constant MeshCoverageParams& p [[buffer(0)]],
                                  uint face [[thread_position_in_grid]],
                                  uint threads [[threads_per_grid]]) {
    device atomic_int* counters = (device atomic_int*)p.counters;
    for (int f = int(face); f < p.face_count; f += int(threads)) {
        MeshCoveragePolygon polygon;
        if (!mesh_coverage_prepare_face(p, f, polygon))
            continue;
        const int pixels = (polygon.max_x - polygon.min_x + 1) * (polygon.max_y - polygon.min_y + 1);
        if (pixels <= kMeshCoverageSmallBoxPixels) {
            for (int y = polygon.min_y; y <= polygon.max_y; ++y)
                for (int x = polygon.min_x; x <= polygon.max_x; ++x)
                    mesh_coverage_pixel(p, polygon, x, y);
        } else {
            const int slot = atomic_fetch_add_explicit(&counters[0], 1, memory_order_relaxed);
            p.large_faces[slot] = f;
        }
    }
}

kernel void mesh_coverage_large(constant MeshCoverageParams& p [[buffer(0)]],
                                uint lane [[thread_index_in_threadgroup]],
                                uint width [[threads_per_threadgroup]]) {
    threadgroup float2 shared_vertices[kMeshCoverageMaxVertices];
    threadgroup int shared_shape[7]; // face, prepared, count, min_x, max_x, min_y, max_y
    device atomic_int* counters = (device atomic_int*)p.counters;
    while (true) {
        if (lane == 0) {
            const int claimed = atomic_fetch_add_explicit(&counters[1], 1, memory_order_relaxed);
            const int queued = atomic_load_explicit(&counters[0], memory_order_relaxed);
            const int face = claimed < queued ? p.large_faces[claimed] : -1;
            MeshCoveragePolygon polygon;
            const bool prepared = face >= 0 && mesh_coverage_prepare_face(p, face, polygon);
            shared_shape[0] = face;
            shared_shape[1] = prepared ? 1 : 0;
            if (prepared) {
                for (int i = 0; i < polygon.count; ++i)
                    shared_vertices[i] = polygon.vertices[i];
                shared_shape[2] = polygon.count;
                shared_shape[3] = polygon.min_x;
                shared_shape[4] = polygon.max_x;
                shared_shape[5] = polygon.min_y;
                shared_shape[6] = polygon.max_y;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (shared_shape[0] < 0)
            return;
        if (shared_shape[1] != 0) {
            MeshCoveragePolygon polygon;
            polygon.count = shared_shape[2];
            for (int i = 0; i < polygon.count; ++i)
                polygon.vertices[i] = shared_vertices[i];
            polygon.min_x = shared_shape[3];
            polygon.max_x = shared_shape[4];
            polygon.min_y = shared_shape[5];
            polygon.max_y = shared_shape[6];
            const int box_width = polygon.max_x - polygon.min_x + 1;
            for (int y = polygon.min_y; y <= polygon.max_y; ++y)
                for (int box_x = int(lane); box_x < box_width; box_x += int(width))
                    mesh_coverage_pixel(p, polygon, polygon.min_x + box_x, y);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
