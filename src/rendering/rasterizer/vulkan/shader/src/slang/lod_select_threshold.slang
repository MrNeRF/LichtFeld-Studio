#include "config.slang"

struct Uniforms {
    uint32_t node_count;
    uint32_t output_capacity;
    uint32_t chunk_splats;
    uint32_t invalid_page;
    float pixel_scale_limit;
    float object_scale;
    float behind_camera_penalty;
    float cone_foveation;
    float cone_dot0;
    float cone_dot;
    float cone_blend_denominator;
    float cone_tail_valid;
    float4 view_row0;
    float4 view_row1;
    float4 view_row2;
    float outside_view_foveation;
    float viewport_half_tan_x;
    float viewport_half_tan_y;
    float ortho_half_width;
    float ortho_half_height;
    uint32_t viewport_foveation;
    uint32_t orthographic;
    uint32_t physical_node_count;
    uint32_t logical_chunk_count;
    uint32_t current_frame;
    uint32_t fade_frames;
    uint32_t pad0;
};

// Quantized sidecar records: bounds 2 words/node (RadMetaBoundsQ u16 x4),
// links 3 words/node (RadMetaLinksQ); dequant frames live per page in
// page_frames float4 slots [1..2]. logical derives from page_to_chunk.
layout(binding = 0) StructuredBuffer<uint32_t> node_bounds;
layout(binding = 1) StructuredBuffer<uint32_t> node_links;
layout(binding = 2) StructuredBuffer<uint32_t> chunk_to_page;
layout(binding = 3) RWStructuredBuffer<uint32_t> out_counts;
layout(binding = 4) RWStructuredBuffer<uint32_t> out_indices;
layout(binding = 5) RWStructuredBuffer<uint32_t> out_logical_indices;
layout(binding = 6) RWStructuredBuffer<float> out_weights;
// Per-logical-chunk residency interest, cleared each frame. Priority entries
// hold the float bits of the touching node's pixel scale (orderable as uint
// for non-negative floats); CHUNK_TOUCH_PROTECTED marks chunks the current
// cut renders from.
layout(binding = 7) RWStructuredBuffer<uint32_t> chunk_touch;
layout(binding = 8) RWStructuredBuffer<uint32_t> out_levels;
// Frame stamp of each pool page's last publish; 0 = always fully visible.
layout(binding = 9) StructuredBuffer<uint32_t> page_age;
layout(binding = 10) StructuredBuffer<float4> page_frames;
layout(binding = 11) StructuredBuffer<uint32_t> page_to_chunk;

static const float LOD_TRANSITION_START = 1.0f;
static const float LOD_TRANSITION_END = 1.18f;
static const uint32_t CHUNK_TOUCH_PROTECTED = 0xffffffffu;

uint32_t physical_index_for_logical(uint32_t logical, Uniforms uniforms) {
    const uint32_t chunk_splats = max(uniforms.chunk_splats, 1u);
    const uint32_t chunk = logical / chunk_splats;
    const uint32_t offset = logical - chunk * chunk_splats;
    const uint32_t page = chunk_to_page[chunk];
    return page == uniforms.invalid_page
               ? uniforms.invalid_page
               : page * chunk_splats + offset;
}

float3 view_position_for_center(float3 center, Uniforms uniforms) {
    return float3(dot(uniforms.view_row0.xyz, center) + uniforms.view_row0.w,
                  dot(uniforms.view_row1.xyz, center) + uniforms.view_row1.w,
                  dot(uniforms.view_row2.xyz, center) + uniforms.view_row2.w);
}

float node_pixel_scale(uint32_t physical, Uniforms uniforms) {
    const uint32_t page = physical / max(uniforms.chunk_splats, 1u);
    const uint32_t w0 = node_bounds[physical * 2u + 0u];
    const uint32_t w1 = node_bounds[physical * 2u + 1u];
    const float4 frame_min = page_frames[page * 4u + 1u];
    const float4 frame_ext = page_frames[page * 4u + 2u];
    const float k_inv = 1.0f / 65535.0f;
    const float3 q = float3(float(w0 & 0xffffu), float(w0 >> 16u), float(w1 & 0xffffu));
    const float3 center = frame_min.xyz + q * k_inv * frame_ext.xyz;
    const float size = exp(frame_min.w + float(w1 >> 16u) * k_inv * frame_ext.w);
    const float radius = max(size * uniforms.object_scale, 1.0e-6f);
    const float3 view_center = view_position_for_center(center, uniforms);
    const float distance = max(length(view_center), 1.0e-6f);
    const float forward_dot = -view_center.z;
    float distance_for_scale = distance;

    const float behind_camera_penalty =
        uniforms.viewport_foveation != 0u
            ? min(uniforms.behind_camera_penalty, uniforms.outside_view_foveation)
            : uniforms.behind_camera_penalty;
    float foveate = behind_camera_penalty;
    if (forward_dot > 0.0f) {
        const float view_dot = forward_dot / distance;
        if (view_dot >= uniforms.cone_dot0) {
            foveate = 1.0f;
        } else if (view_dot >= uniforms.cone_dot) {
            if (uniforms.cone_blend_denominator >= 1.0e-6f) {
                const float t = (view_dot - uniforms.cone_dot) / uniforms.cone_blend_denominator;
                foveate = uniforms.cone_foveation + (1.0f - uniforms.cone_foveation) * saturate(t);
            } else {
                foveate = 1.0f;
            }
        } else if (uniforms.cone_tail_valid > 0.5f) {
            const float t = view_dot / uniforms.cone_dot;
            foveate = behind_camera_penalty +
                      (uniforms.cone_foveation - behind_camera_penalty) * saturate(t);
        }
    }

    if (uniforms.viewport_foveation != 0u) {
        const float outside_floor = saturate(uniforms.outside_view_foveation);
        float viewport_foveate = 1.0f;
        if (uniforms.orthographic != 0u) {
            const float half_width = uniforms.ortho_half_width;
            const float half_height = uniforms.ortho_half_height;
            if (half_width > 0.0f && half_height > 0.0f) {
                const float overflow_x = abs(view_center.x) - (half_width + radius);
                const float overflow_y = abs(view_center.y) - (half_height + radius);
                const float overflow = max(overflow_x, overflow_y);
                if (overflow > 0.0f) {
                    const float blend_width = max(max(half_width, half_height) * 0.08f, radius * 0.5f);
                    const float t = 1.0f - saturate(overflow / blend_width);
                    viewport_foveate = outside_floor + (1.0f - outside_floor) * t;
                }
            }
        } else if (uniforms.viewport_half_tan_x > 0.0f && uniforms.viewport_half_tan_y > 0.0f) {
            const float view_depth = -view_center.z;
            if (view_depth + radius <= 1.0e-6f) {
                viewport_foveate = outside_floor;
            } else {
                const float depth_for_extent = max(view_depth, 1.0e-6f);
                const float half_width = depth_for_extent * uniforms.viewport_half_tan_x;
                const float half_height = depth_for_extent * uniforms.viewport_half_tan_y;
                const float overflow_x = abs(view_center.x) - (half_width + radius);
                const float overflow_y = abs(view_center.y) - (half_height + radius);
                const float overflow = max(overflow_x, overflow_y);
                if (overflow > 0.0f) {
                    const float blend_width = max(max(half_width, half_height) * 0.08f, radius * 0.5f);
                    const float t = 1.0f - saturate(overflow / blend_width);
                    viewport_foveate = outside_floor + (1.0f - outside_floor) * t;
                }
                if (view_depth > 1.0e-6f) {
                    distance_for_scale = view_depth;
                }
            }
        }
        foveate = min(foveate, viewport_foveate);
    }

    return (radius / max(distance_for_scale, 1.0e-6f)) * foveate;
}

bool child_range_resident(uint32_t child_start, uint32_t child_count, Uniforms uniforms) {
    for (uint32_t c = 0u; c < child_count; ++c) {
        if (physical_index_for_logical(child_start + c, uniforms) == uniforms.invalid_page)
            return false;
    }
    return true;
}

float child_transition_weight(float parent_pixel_scale, Uniforms uniforms) {
    const float limit = max(uniforms.pixel_scale_limit, 1.0e-12f);
    const float start = limit * LOD_TRANSITION_START;
    const float end = limit * LOD_TRANSITION_END;
    return saturate((parent_pixel_scale - start) / max(end - start, 1.0e-12f));
}

// Smoothstep fade-in for freshly streamed pages so refinement appears
// gradually instead of popping. Stamp 0 marks pages resident since load.
float chunk_fade(uint32_t chunk, Uniforms uniforms) {
    if (uniforms.fade_frames == 0u) {
        return 1.0f;
    }
    const uint32_t page = chunk_to_page[chunk];
    if (page == uniforms.invalid_page) {
        return 0.0f;
    }
    const uint32_t stamp = page_age[page];
    if (stamp == 0u || uniforms.current_frame <= stamp) {
        return stamp == 0u ? 1.0f : 0.0f;
    }
    const float t = saturate(float(uniforms.current_frame - stamp) / float(uniforms.fade_frames));
    return t * t * (3.0f - 2.0f * t);
}

float child_range_fade(uint32_t child_start, uint32_t child_count, Uniforms uniforms) {
    const uint32_t chunk_splats = max(uniforms.chunk_splats, 1u);
    const uint32_t first_chunk = child_start / chunk_splats;
    const uint32_t last_chunk = (child_start + max(child_count, 1u) - 1u) / chunk_splats;
    float fade = 1.0f;
    for (uint32_t chunk = first_chunk; chunk <= last_chunk; ++chunk) {
        fade = min(fade, chunk_fade(chunk, uniforms));
    }
    return fade;
}

void touch_chunk_index(uint32_t chunk, uint32_t value, Uniforms uniforms) {
    if (uniforms.logical_chunk_count == 0u || chunk >= uniforms.logical_chunk_count)
        return;
    uint32_t ignored = 0u;
    InterlockedMax(chunk_touch[chunk], value, ignored);
}

void touch_chunk(uint32_t logical, uint32_t value, Uniforms uniforms) {
    touch_chunk_index(logical / max(uniforms.chunk_splats, 1u), value, uniforms);
}

uint32_t touch_priority(float pixel_scale) {
    return clamp(asuint(max(pixel_scale, 0.0f)), 1u, CHUNK_TOUCH_PROTECTED - 1u);
}

void append_node(uint32_t physical, uint32_t logical, float weight, uint32_t level, Uniforms uniforms) {
    touch_chunk(logical, CHUNK_TOUCH_PROTECTED, uniforms);
    uint32_t append_index = 0u;
    InterlockedAdd(out_counts[0], 1u, append_index);
    if (append_index >= uniforms.output_capacity) {
        uint32_t ignored = 0u;
        InterlockedAdd(out_counts[1], 1u, ignored);
        return;
    }

    out_indices[append_index] = physical;
    out_logical_indices[append_index] = logical;
    out_weights[append_index] = saturate(weight);
    out_levels[append_index] = level;
}

[shader("compute")]
    [numthreads(128, 1, 1)] void
    main(
        uint3 dispatchThreadID
        : SV_DispatchThreadID,
          uniform Uniforms uniforms) {
        const uint32_t physical = dispatchThreadID.x;
        if (physical >= uniforms.physical_node_count)
            return;

        const uint32_t chunk_splats = max(uniforms.chunk_splats, 1u);
        const uint32_t page = physical / chunk_splats;
        const uint32_t chunk = page_to_chunk[page];
        if (chunk == uniforms.invalid_page)
            return;
        const uint32_t logical = chunk * chunk_splats + (physical - page * chunk_splats);
        // Covers the last chunk's slack nodes: their derived logical lands
        // past the tree.
        if (logical >= uniforms.node_count)
            return;

        const uint32_t links_base = physical * 3u;
        const uint32_t child_start = node_links[links_base + 0u];
        const uint32_t packed = node_links[links_base + 1u];
        const uint32_t child_count = packed & 0xffffu;
        const uint32_t level = (packed >> 16u) & 0xffu;
        const uint32_t parent_logical = node_links[links_base + 2u];

        const float pixel_scale = node_pixel_scale(physical, uniforms);

        bool parent_above_limit = true;
        bool parent_waiting_for_residency = false;
        if (parent_logical != uniforms.invalid_page) {
            const uint32_t parent_physical = physical_index_for_logical(parent_logical, uniforms);
            if (parent_physical == uniforms.invalid_page) {
                // Parent chunk evicted: the nearest resident ancestor sees a
                // non-resident child range and renders this region itself.
                // Appending here too would double-cover and flood the output
                // with threshold-blind nodes; just request the parent back.
                touch_chunk(parent_logical, touch_priority(pixel_scale), uniforms);
                return;
            }
            const uint32_t parent_links_base = parent_physical * 3u;
            const uint32_t parent_child_start = node_links[parent_links_base + 0u];
            const uint32_t parent_packed = node_links[parent_links_base + 1u];
            const uint32_t parent_child_count = parent_packed & 0xffffu;
            parent_above_limit = node_pixel_scale(parent_physical, uniforms) > uniforms.pixel_scale_limit;
            parent_waiting_for_residency =
                parent_above_limit &&
                parent_child_count != 0u &&
                !child_range_resident(parent_child_start, parent_child_count, uniforms);
        }

        if (!parent_above_limit || parent_waiting_for_residency)
            return;

        const bool leaf = child_count == 0u;
        const bool children_resident =
            leaf || child_range_resident(child_start, child_count, uniforms);

        if (!leaf && pixel_scale > uniforms.pixel_scale_limit) {
            if (!children_resident) {
                // Expansion blocked on residency: ask the pager for the child
                // chunks at this node's priority and render the parent meanwhile.
                const uint32_t first_chunk = child_start / chunk_splats;
                const uint32_t last_chunk = (child_start + child_count - 1u) / chunk_splats;
                const uint32_t priority = touch_priority(pixel_scale);
                for (uint32_t chunk = first_chunk; chunk <= last_chunk; ++chunk) {
                    touch_chunk_index(chunk, priority, uniforms);
                }
                append_node(physical, logical, 1.0f, level, uniforms);
                return;
            }
            // Expanded interior nodes stay LRU-fresh: their links/bounds are read
            // by every child's cut test.
            touch_chunk(logical, touch_priority(pixel_scale), uniforms);
            // The parent keeps rendering its fading complement until freshly
            // streamed children are fully visible — even past the transition
            // band, or the hand-off pops.
            const float fade = child_range_fade(child_start, child_count, uniforms);
            const float parent_weight = 1.0f - child_transition_weight(pixel_scale, uniforms) * fade;
            if (parent_weight > 1.0f / 255.0f) {
                append_node(physical, logical, parent_weight, level, uniforms);
            }
            return;
        }

        float weight = 1.0f;
        if (parent_logical != uniforms.invalid_page) {
            // Parent residency was established above.
            const uint32_t parent_physical = physical_index_for_logical(parent_logical, uniforms);
            const float parent_pixel_scale = node_pixel_scale(parent_physical, uniforms);
            weight = child_transition_weight(parent_pixel_scale, uniforms);
        }
        // Unconditional append even at weight 0: append_node marks the chunk
        // PROTECTED, which a fading-in chunk must stay throughout the fade.
        weight *= chunk_fade(chunk, uniforms);
        append_node(physical, logical, weight, level, uniforms);
    }
