#version 450
#extension GL_GOOGLE_include_directive : require
#define LFS_EQUIRECT_GLSL
#include "../../../../rendering/rasterizer/vulkan/shader/src/slang/equirect_projection.inc"

layout(location = 0) in vec2 ndc;
layout(location = 0) out vec4 color;
layout(set = 0, binding = 0, std430) readonly buffer Volume {
    ivec4 dimensions_edge;
    uint words[];
} volume;
layout(push_constant) uniform Projection {
    mat4 clip_to_grid;
    vec4 pixel;
    vec4 panorama;
} projection;

uint traversal_path[22];
int traversal_level;

// Retain ancestors across ray steps; only descend from the changed coordinate bit.
bool occupied(ivec3 cell, out ivec3 lo, out ivec3 hi) {
    uint index = traversal_path[traversal_level];
    int edge = 1 << traversal_level;
    lo = cell & ~(edge - 1);
    while (true) {
        uint word = volume.words[index];
        hi = min(lo + edge, volume.dimensions_edge.xyz);
        if (word <= 1u)
            return word == 1u;
        if ((word & 0x80000000u) != 0u) {
            ivec3 p = cell - lo;
            uint bit = uint(p.x + 4 * p.y + 16 * p.z);
            uint mask = volume.words[(word & 0x7fffffffu) + bit / 32u];
            lo = cell;
            hi = cell + 1;
            return (mask & (1u << (bit % 32u))) != 0u;
        }
        edge /= 2;
        --traversal_level;
        ivec3 child = ivec3(greaterThanEqual(cell, lo + edge));
        index = word - 2u + uint(child.x + 2 * child.y + 4 * child.z);
        traversal_path[traversal_level] = index;
        lo += child * edge;
    }
}

void main() {
    vec4 near_h = projection.clip_to_grid * vec4(ndc, 0.0, 1.0);
    vec4 far_h = projection.clip_to_grid * vec4(ndc, 1.0, 1.0);
    vec3 origin = near_h.xyz / near_h.w;
    vec3 direction = normalize(far_h.xyz * near_h.w - near_h.xyz * far_h.w);
    if (projection.panorama.x != 0.0) {
        vec3 camera_ray = lfs_equirect_direction(ndc * 0.5 + 0.5);
        vec3 grid_ray = mat3(projection.clip_to_grid) * camera_ray;
        origin = (projection.clip_to_grid * vec4(camera_ray * projection.panorama.y, 1.0)).xyz;
        direction = normalize(grid_ray);
    }
    // Derivatives must be evaluated before divergent traversal and discards.
    vec3 origin_dx = dFdx(origin), origin_dy = dFdy(origin);
    vec3 direction_dx = dFdx(direction), direction_dy = dFdy(direction);
    float enter = 0.0;
    float leave = 3.402823e38;
    int face = 0;
    for (int axis = 0; axis < 3; ++axis) {
        if (direction[axis] == 0.0) {
            if (origin[axis] < 0.0 || origin[axis] >= volume.dimensions_edge[axis])
                discard;
        } else {
            float a = -origin[axis] / direction[axis];
            float b = (float(volume.dimensions_edge[axis]) - origin[axis]) / direction[axis];
            float start = min(a, b);
            if (start > enter) {
                enter = start;
                face = axis;
            }
            leave = min(leave, max(a, b));
        }
    }
    if (enter >= leave)
        discard;
    traversal_level = findMSB(volume.dimensions_edge.w);
    traversal_path[traversal_level] = 0u;
    ivec3 cell = clamp(ivec3(floor(origin + direction * enter)), ivec3(0), volume.dimensions_edge.xyz - 1);
    while (enter < leave && all(greaterThanEqual(cell, ivec3(0))) &&
           all(lessThan(cell, volume.dimensions_edge.xyz))) {
        ivec3 lo, hi;
        if (occupied(cell, lo, hi) != (projection.panorama.w != 0.0)) {
            vec3 hit = origin + direction * enter;
            vec3 f = fract(hit);
            vec3 distance_to_edge = min(f, 1.0 - f);
            distance_to_edge[face] = 1.0;
            float edge = min(distance_to_edge.x, min(distance_to_edge.y, distance_to_edge.z));
            float width = projection.pixel.y != 0.0 || projection.panorama.x != 0.0
                              ? max(length(origin_dx + enter * direction_dx), length(origin_dy + enter * direction_dy))
                              : enter * projection.pixel.x;
            float interior = smoothstep(0.0, clamp(width * 1.5, 0.01, 0.5), edge);
            float shade = face == 0 ? 0.85 : (face == 1 ? 0.65 : 0.45);
            vec3 tint = projection.panorama.w != 0.0 ? vec3(1.0, 0.35, 0.08) : vec3(1.0);
            float opacity = projection.panorama.z;
            color = vec4(tint * mix(0.2, shade, interior), mix(min(2.0 * opacity, 1.0), opacity, interior));
            return;
        }
        vec3 exit_time = vec3(3.402823e38);
        for (int axis = 0; axis < 3; ++axis) {
            if (direction[axis] != 0.0)
                exit_time[axis] = (float(direction[axis] > 0.0 ? hi[axis] : lo[axis]) - origin[axis]) / direction[axis];
        }
        enter = min(exit_time.x, min(exit_time.y, exit_time.z));
        ivec3 previous = cell;
        cell = ivec3(floor(origin + direction * enter));
        for (int axis = 0; axis < 3; ++axis) {
            if (exit_time[axis] == enter) {
                cell[axis] = direction[axis] > 0.0 ? hi[axis] : lo[axis] - 1;
                face = axis;
            } else {
                cell[axis] = clamp(cell[axis], lo[axis], hi[axis] - 1);
            }
        }
        uvec3 changed = uvec3(previous ^ cell);
        traversal_level = max(traversal_level, findMSB(changed.x | changed.y | changed.z) + 1);
    }
    discard;
}
