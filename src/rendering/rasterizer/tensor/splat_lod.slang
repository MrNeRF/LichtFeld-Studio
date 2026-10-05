// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
// GPU threshold/reachability/fade/budget LOD selection on tensor buffers.
#include "../slang/config.slang"

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
    // 0: initial selection; 1..16: budget gate; 17: indirect retry.
    uint32_t budget_pass;
};

// Quantized sidecar records: bounds 2 words/node (RadMetaBoundsQ u16 x4),
// links 3 words/node (RadMetaLinksQ); dequant frames live per page in
// page_frames float4 slots [1..2]. logical derives from page_to_chunk.
struct Parameters {
    Ptr<uint, Access.Read> bounds;
    Ptr<uint, Access.Read> links;
    Ptr<uint, Access.Read> chunk_to_page;
    Ptr<uint, Access.ReadWrite> counts;
    Ptr<uint, Access.ReadWrite> indices;
    Ptr<uint, Access.ReadWrite> logical_indices;
    Ptr<float, Access.ReadWrite> weights;
    Ptr<uint, Access.ReadWrite> touches;
    Ptr<uint, Access.ReadWrite> levels;
    Ptr<uint, Access.Read> page_age;
    Ptr<float4, Access.Read> page_frames;
    Ptr<uint, Access.Read> page_to_chunk;
    Ptr<Uniforms, Access.Read> uniforms;
    uint budget_pass;
    uint padding;
};
[[vk::push_constant]] ConstantBuffer<Parameters> p;

static const float LOD_TRANSITION_START = 1.0f;
static const float LOD_TRANSITION_END = 1.18f;
static const uint32_t CHUNK_TOUCH_PROTECTED = 0xffffffffu;

uint32_t physical_index_for_logical(uint32_t logical, Uniforms uniforms) {
    if (logical >= uniforms.node_count)
        return uniforms.invalid_page;
    const uint32_t chunk_splats = max(uniforms.chunk_splats, 1u);
    const uint32_t chunk = logical / chunk_splats;
    const uint32_t offset = logical - chunk * chunk_splats;
    if (chunk >= uniforms.logical_chunk_count)
        return uniforms.invalid_page;
    const uint32_t page = p.chunk_to_page[chunk];
    if (page == uniforms.invalid_page || page >= 1u + (uniforms.physical_node_count - 1u) / chunk_splats)
        return uniforms.invalid_page;
    const uint physical = page * chunk_splats + offset;
    return physical < uniforms.physical_node_count ? physical : uniforms.invalid_page;
}

float3 view_position_for_center(float3 center, Uniforms uniforms) {
    return float3(dot(uniforms.view_row0.xyz, center) + uniforms.view_row0.w,
                  dot(uniforms.view_row1.xyz, center) + uniforms.view_row1.w,
                  dot(uniforms.view_row2.xyz, center) + uniforms.view_row2.w);
}

float node_pixel_scale(uint32_t physical, Uniforms uniforms) {
    const uint32_t page = physical / max(uniforms.chunk_splats, 1u);
    const uint32_t w0 = p.bounds[physical * 2u + 0u];
    const uint32_t w1 = p.bounds[physical * 2u + 1u];
    const float4 frame_min = p.page_frames[page * 4u + 1u];
    const float4 frame_ext = p.page_frames[page * 4u + 2u];
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
    if (child_start >= uniforms.node_count || child_count > uniforms.node_count - child_start)
        return false;
    if (child_count == 0u)
        return true;
    // Residency is page-granular. Check both ends of each covered chunk,
    // including partial pool pages, instead of repeating the same lookup for
    // every descendant. This preserves the source selector's cut exactly.
    const uint chunk_splats = max(uniforms.chunk_splats, 1u);
    const uint last = child_start + child_count - 1u;
    const uint last_chunk = last / chunk_splats;
    for (uint chunk = child_start / chunk_splats; chunk <= last_chunk; ++chunk) {
        const uint first_node = max(child_start, chunk * chunk_splats);
        const uint last_node = chunk == last_chunk ? last : (chunk + 1u) * chunk_splats - 1u;
        if (physical_index_for_logical(first_node, uniforms) == uniforms.invalid_page ||
            physical_index_for_logical(last_node, uniforms) == uniforms.invalid_page)
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
    if (chunk >= uniforms.logical_chunk_count)
        return 0.0f;
    const uint32_t page = p.chunk_to_page[chunk];
    if (page == uniforms.invalid_page ||
        page >= 1u + (uniforms.physical_node_count - 1u) / max(uniforms.chunk_splats, 1u)) {
        return 0.0f;
    }
    const uint32_t stamp = p.page_age[page];
    if (stamp == 0u || uniforms.current_frame <= stamp) {
        return stamp == 0u ? 1.0f : 0.0f;
    }
    const float t = saturate(float(uniforms.current_frame - stamp) / float(uniforms.fade_frames));
    return t * t * (3.0f - 2.0f * t);
}

float child_range_fade(uint32_t child_start, uint32_t child_count, Uniforms uniforms) {
    if (child_start >= uniforms.node_count || child_count > uniforms.node_count - child_start)
        return 0.0f;
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
    InterlockedMax(p.touches[chunk], value, ignored);
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
    InterlockedAdd(p.counts[0], 1u, append_index);
    if (append_index >= uniforms.output_capacity) {
        uint32_t ignored = 0u;
        InterlockedAdd(p.counts[1], 1u, ignored);
        return;
    }

    p.indices[append_index] = physical;
    p.logical_indices[append_index] = logical;
    p.weights[append_index] = saturate(weight);
    p.levels[append_index] = level;
}

[shader("compute")]
[numthreads(128, 1, 1)]
void select_lod(uint3 dispatchThreadID : SV_DispatchThreadID) {
        Uniforms uniforms = p.uniforms[0];
        uniforms.budget_pass = p.budget_pass;
        // counts: candidate count, overflow count, threshold multiplier, then
        // VkDispatchIndirectCommand. A failed cut is never sent to projection.
        if (uniforms.budget_pass >= 1u && uniforms.budget_pass <= 16u) {
            if (dispatchThreadID.x != 0u)
                return;
            const uint32_t count = p.counts[0];
            const bool retry = count > uniforms.output_capacity;
            p.counts[3] = retry ? 1u + (uniforms.physical_node_count - 1u) / 128u : 0u;
            p.counts[4] = 1u;
            p.counts[5] = 1u;
            if (retry) {
                // Interests from the discarded selection must not pin pages
                // that its replacement no longer uses.
                for (uint32_t chunk = 0u; chunk < uniforms.logical_chunk_count; ++chunk)
                    p.touches[chunk] = 0u;
                const float ratio = float(count) / float(max(uniforms.output_capacity, 1u));
                const float scale = asfloat(p.counts[2]) * clamp(pow(ratio / 0.9f, 0.85f), 1.25f, 4.0f);
                // The final retry stops at the root, even for pathological
                // non-monotone trees. All preceding attempts retain detail.
                p.counts[2] = asuint(uniforms.budget_pass == 16u ? 3.402823e38f : scale);
                p.counts[0] = 0u;
                p.counts[1] = 0u;
            }
            return;
        }
        if (uniforms.budget_pass == 0u) {
            if (dispatchThreadID.x == 0u)
                p.counts[2] = asuint(1.0f);
        } else {
            const float scale = asfloat(p.counts[2]);
            uniforms.pixel_scale_limit = scale >= 3.402823e38f ? 3.402823e38f : uniforms.pixel_scale_limit * scale;
        }
        const uint32_t physical = dispatchThreadID.x;
        if (physical >= uniforms.physical_node_count)
            return;

        const uint32_t chunk_splats = max(uniforms.chunk_splats, 1u);
        const uint32_t page = physical / chunk_splats;
        const uint32_t chunk = p.page_to_chunk[page];
        if (chunk == uniforms.invalid_page || chunk >= uniforms.logical_chunk_count)
            return;
        const uint32_t logical = chunk * chunk_splats + (physical - page * chunk_splats);
        // Covers the last chunk's slack nodes: their derived logical lands
        // past the tree.
        if (logical >= uniforms.node_count)
            return;

        const uint32_t links_base = physical * 3u;
        const uint32_t child_start = p.links[links_base + 0u];
        const uint32_t packed = p.links[links_base + 1u];
        const uint32_t child_count = packed & 0xffffu;
        if (child_count != 0u &&
            (child_start <= logical || child_start >= uniforms.node_count || child_count > uniforms.node_count - child_start))
            return;
        const uint32_t level = (packed >> 16u) & 0xffu;
        const uint32_t parent_logical = p.links[links_base + 2u];

        if (uniforms.budget_pass == 17u && asfloat(p.counts[2]) >= 3.402823e38f) {
            if (parent_logical == uniforms.invalid_page)
                append_node(physical, logical, 1.0f, level, uniforms);
            return;
        }

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
            const uint32_t parent_child_start = p.links[parent_links_base + 0u];
            const uint32_t parent_packed = p.links[parent_links_base + 1u];
            const uint32_t parent_child_count = parent_packed & 0xffffu;
            parent_above_limit = node_pixel_scale(parent_physical, uniforms) > uniforms.pixel_scale_limit;
            parent_waiting_for_residency =
                parent_above_limit &&
                parent_child_count != 0u &&
                !child_range_resident(parent_child_start, parent_child_count, uniforms);
        }

        if (!parent_above_limit || parent_waiting_for_residency)
            return;

        // A flat dispatch must enforce the same root-to-node reachability as
        // tree traversal. Projected size is not monotone along a branch (the
        // centers, view depth and foveation differ), so checking just the
        // immediate parent also selects descendants below an already stopped
        // ancestor. Those disconnected cuts overdraw the ancestor and exhaust
        // the output budget, leaving holes elsewhere in the scene.
        uint32_t ancestor = parent_logical;
        while (ancestor != uniforms.invalid_page) {
            const uint32_t ancestor_physical = physical_index_for_logical(ancestor, uniforms);
            if (ancestor_physical == uniforms.invalid_page) {
                touch_chunk(ancestor, touch_priority(pixel_scale), uniforms);
                return;
            }
            const uint32_t ancestor_links = ancestor_physical * 3u;
            const uint32_t next = p.links[ancestor_links + 2u];
            if (next == uniforms.invalid_page)
                break;
            // RAD trees are ordered parent before child. Reject malformed
            // links instead of permitting an unbounded shader loop.
            if (next >= ancestor)
                return;
            const uint32_t next_physical = physical_index_for_logical(next, uniforms);
            if (next_physical == uniforms.invalid_page) {
                touch_chunk(next, touch_priority(pixel_scale), uniforms);
                return;
            }
            const uint32_t next_links = next_physical * 3u;
            if (node_pixel_scale(next_physical, uniforms) <= uniforms.pixel_scale_limit ||
                !child_range_resident(p.links[next_links], p.links[next_links + 1u] & 0xffffu, uniforms))
                return;
            ancestor = next;
        }

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
