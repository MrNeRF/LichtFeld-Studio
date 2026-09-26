// ---------------------------------------------------------------------------
// Point-cloud rasterization, ported from the renderer's point_cloud_raster.cu.
// Each point splats a disk of pixels; CUDA packs (depth, color) into one
// 64-bit atomic min, which here takes two 32-bit passes with the same winner:
// kOp 1 keeps each pixel's nearest depth, kOp 2 the smallest color among the
// points at that depth. kOp 0 clears and kOp 3 writes the image and depth.

struct PointRasterParams {
    device const float* positions;
    device const float* colors;
    device const float* parameters;
    device uint* scratch;
    device float* image;
    device float* depth;
    device const float* transforms;
    device const int* indices;
    device const uchar* visibility;
    device const uchar* deleted;
    uint count, width, height, channels, transform_count, visibility_count, flags;
    float ortho_scale, focal_y, voxel_size, far_plane;
    uint padding;
};

constant uint kRasterCropBox = 1u, kRasterCropEllipsoid = 2u, kRasterCropInverse = 4u, kRasterCropDesaturate = 8u,
              kRasterEquirectangular = 16u, kRasterOrthographic = 32u, kRasterTransparent = 64u;
constant uint kRasterEmpty = 0xffffffffu;

// Row `row` of a column-major 4x4 matrix times (x, y, z, w).
static float raster_row(device const float* m, int row, float x, float y, float z, float w) {
    return m[row] * x + m[row + 4] * y + m[row + 8] * z + m[row + 12] * w;
}

struct RasterSplat {
    int x, y, radius;
    uint depth, color;
};

// Projects point i; false when it is hidden, cropped out or off screen.
static bool raster_splat(constant PointRasterParams& p, uint i, thread RasterSplat& splat) {
    if (p.deleted != nullptr && p.deleted[i] != 0)
        return false;
    const int transform = p.indices != nullptr ? clamp(p.indices[i], 0, max(int(p.transform_count) - 1, 0)) : 0;
    if (p.visibility != nullptr && uint(transform) < p.visibility_count && p.visibility[transform] == 0)
        return false;
    float x = p.positions[i * 3], y = p.positions[i * 3 + 1], z = p.positions[i * 3 + 2];
    if (p.transforms != nullptr && p.transform_count > 0) {
        device const float* m = p.transforms + transform * 16;
        const float nx = raster_row(m, 0, x, y, z, 1.0f), ny = raster_row(m, 1, x, y, z, 1.0f);
        const float nz = raster_row(m, 2, x, y, z, 1.0f), nw = raster_row(m, 3, x, y, z, 1.0f);
        const float scale = fabs(nw) > 1e-6f ? 1.0f / nw : 1.0f;
        x = nx * scale;
        y = ny * scale;
        z = nz * scale;
    }
    device const float* view = p.parameters;
    device const float* view_projection = p.parameters + 16;
    device const float* to_local = p.parameters + 32;
    device const float* crop_min = p.parameters + 48;
    device const float* crop_max = p.parameters + 51;
    bool desaturate = false;
    if ((p.flags & (kRasterCropBox | kRasterCropEllipsoid)) != 0) {
        const float lx = raster_row(to_local, 0, x, y, z, 1.0f), ly = raster_row(to_local, 1, x, y, z, 1.0f);
        const float lz = raster_row(to_local, 2, x, y, z, 1.0f);
        bool inside;
        if ((p.flags & kRasterCropBox) != 0) {
            inside = lx >= crop_min[0] && lx <= crop_max[0] && ly >= crop_min[1] && ly <= crop_max[1] &&
                     lz >= crop_min[2] && lz <= crop_max[2];
        } else {
            const float rx = max(fabs(crop_min[0]), 1e-8f), ry = max(fabs(crop_min[1]), 1e-8f);
            const float rz = max(fabs(crop_min[2]), 1e-8f);
            inside = (lx * lx) / (rx * rx) + (ly * ly) / (ry * ry) + (lz * lz) / (rz * rz) <= 1.0f;
        }
        if (inside == ((p.flags & kRasterCropInverse) != 0)) {
            if ((p.flags & kRasterCropDesaturate) == 0)
                return false;
            desaturate = true;
        }
    }
    const float view_x = raster_row(view, 0, x, y, z, 1.0f), view_y = raster_row(view, 1, x, y, z, 1.0f);
    const float view_z = raster_row(view, 2, x, y, z, 1.0f);
    const bool orthographic = (p.flags & kRasterOrthographic) != 0;
    float pixel_x, pixel_y, depth;
    if ((p.flags & kRasterEquirectangular) != 0) {
        const float length = sqrt(view_x * view_x + view_y * view_y + view_z * view_z);
        if (length <= 1e-6f)
            return false;
        const float pi = 3.14159265358979323846f;
        const float u = 0.5f + atan2(view_x / length, -view_z / length) / (2.0f * pi);
        const float v = 0.5f - asin(clamp(view_y / length, -1.0f, 1.0f)) / pi;
        pixel_x = u * float(p.width - 1);
        pixel_y = v * float(p.height - 1);
        if (!isfinite(pixel_x) || !isfinite(pixel_y) || pixel_x < 0.0f || pixel_x >= float(p.width) ||
            pixel_y < 0.0f || pixel_y >= float(p.height))
            return false;
        depth = length;
    } else {
        const float cx = raster_row(view_projection, 0, x, y, z, 1.0f), cy = raster_row(view_projection, 1, x, y, z, 1.0f);
        const float cz = raster_row(view_projection, 2, x, y, z, 1.0f), cw = raster_row(view_projection, 3, x, y, z, 1.0f);
        if (fabs(cw) <= 1e-6f)
            return false;
        const float ndc_x = cx / cw, ndc_y = cy / cw, ndc_z = cz / cw;
        if (!isfinite(ndc_x) || !isfinite(ndc_y) || !isfinite(ndc_z) || ndc_x < -1.0f || ndc_x > 1.0f ||
            ndc_y < -1.0f || ndc_y > 1.0f || ndc_z < 0.0f || ndc_z > 1.0f)
            return false;
        pixel_x = (ndc_x * 0.5f + 0.5f) * float(p.width - 1);
        pixel_y = (0.5f - ndc_y * 0.5f) * float(p.height - 1);
        depth = orthographic ? -view_z : max(-view_z, 0.0f);
        if (depth <= 0.0f && !orthographic)
            return false;
    }
    const float voxel = max(p.voxel_size, 1e-5f);
    splat.radius = orthographic ? max(1, int(ceil(voxel * float(p.height) / max(p.ortho_scale, 1e-5f) * 0.5f)))
                                : max(1, int(ceil(voxel * p.focal_y / max(depth, 1e-4f))));
    float3 color = clamp(float3(p.colors[i * 3], p.colors[i * 3 + 1], p.colors[i * 3 + 2]), 0.0f, 1.0f);
    if (desaturate)
        color += (dot(color, float3(0.299f, 0.587f, 0.114f)) - color) * 0.75f;
    const uint3 bytes = uint3(clamp(color, 0.0f, 1.0f) * 255.0f + 0.5f);
    splat.color = bytes.x | (bytes.y << 8) | (bytes.z << 16);
    splat.depth = as_type<uint>(max(depth, 0.0f));
    splat.x = int(rint(pixel_x));
    splat.y = int(rint(pixel_y));
    return true;
}

kernel void point_raster(constant PointRasterParams& p [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    const uint pixels = p.width * p.height;
    device atomic_uint* nearest = (device atomic_uint*)p.scratch;
    device atomic_uint* colors = (device atomic_uint*)(p.scratch + pixels);
    if (kOp == 0 || kOp == 3) {
        if (i >= pixels)
            return;
        if (kOp == 0) {
            p.scratch[i] = kRasterEmpty;
            p.scratch[pixels + i] = kRasterEmpty;
            return;
        }
        device const float* background = p.parameters + 54;
        const uint depth_bits = p.scratch[i];
        const bool empty = depth_bits == kRasterEmpty;
        const uint color = p.scratch[pixels + i];
        for (uint c = 0; c < 3; ++c)
            p.image[c * pixels + i] = empty ? background[c] : float((color >> (8 * c)) & 0xffu) / 255.0f;
        if (p.channels == 4)
            p.image[3 * pixels + i] = empty && (p.flags & kRasterTransparent) != 0 ? 0.0f : 1.0f;
        p.depth[i] = empty ? p.far_plane : as_type<float>(depth_bits);
        return;
    }
    RasterSplat splat;
    if (i >= p.count || !raster_splat(p, i, splat))
        return;
    const int radius_squared = splat.radius * splat.radius;
    for (int dy = -splat.radius; dy <= splat.radius; ++dy) {
        const int y = splat.y + dy;
        if (y < 0 || y >= int(p.height))
            continue;
        for (int dx = -splat.radius; dx <= splat.radius; ++dx) {
            const int x = splat.x + dx;
            if (x < 0 || x >= int(p.width) || dx * dx + dy * dy > radius_squared)
                continue;
            const uint pixel = uint(y) * p.width + uint(x);
            if (kOp == 1)
                atomic_fetch_min_explicit(&nearest[pixel], splat.depth, memory_order_relaxed);
            else if (atomic_load_explicit(&nearest[pixel], memory_order_relaxed) == splat.depth)
                atomic_fetch_min_explicit(&colors[pixel], splat.color, memory_order_relaxed);
        }
    }
}
