// Position-only cull prepass for the HiGS viewer projection. Reads just the
// splat centers (plus node/crop state and the LoD indirection) for the whole
// render domain and emits the compact survivor list the survivor projection
// runs over. Every test here is an exact mirror of a reject inside the
// projection shader — the survivor set is a strict superset of the
// finally-visible set, never smaller, so no conservative margin is needed.

#define NO_MAIN 1
#include "vertex_shader.slang"
#include "indirect_layout.inc"

#define CULL_ENTRY_CULL 1
#define CULL_ENTRY_PREPARE 2

#ifndef CULL_ENTRY
#define CULL_ENTRY 0
#endif

#if CULL_ENTRY == CULL_ENTRY_CULL

layout(binding=0) StructuredBuffer<float> xyz_ws;
layout(binding=1) ByteAddressBuffer transform_indices;
layout(binding=2) ByteAddressBuffer node_mask;
layout(binding=3) StructuredBuffer<float4> overlay_params;
layout(binding=4) StructuredBuffer<float4> model_transforms;
layout(binding=5) StructuredBuffer<uint32_t> lod_indices;
layout(binding=6) StructuredBuffer<uint32_t> lod_logical_indices;
layout(binding=7) StructuredBuffer<uint32_t> lod_counts;
layout(binding=8) RWStructuredBuffer<int32_t> out_survivors;
// Count and projection-command offsets are defined by LFS_VK_SURVIVOR_STATE_*.
layout(binding=9) RWStructuredBuffer<uint32_t> out_survivor_state;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform Uniforms uniforms
) {
    const uint render_idx = dispatchThreadID.x;
    bool survives = false;

    if (render_idx < uniforms.num_splats) {
        const bool lod_active = (uniforms.lod_enabled & 1u) != 0u;
        uint g_idx = render_idx;
        uint logical_idx = render_idx;
        bool valid = true;
        if (lod_active) {
            if ((uniforms.lod_enabled & 32u) != 0u) {
                const uint valid_count = min(lod_counts[0], uniforms.lod_count);
                valid = render_idx < valid_count;
            }
            if (valid) {
                g_idx = lod_indices[render_idx];
                if (g_idx == 0xffffffffu || g_idx >= uniforms.model_num_splats) {
                    valid = false;
                } else {
                    logical_idx = (uniforms.lod_enabled & 8u) != 0u
                        ? lod_logical_indices[render_idx]
                        : g_idx;
                }
            }
        }

        if (valid) {
            const float3 xyz = read_t3_float3(int32_t(g_idx), xyz_ws);
            const SplatActiveState state = compute_splat_active_state(
                logical_idx, xyz, uniforms.step,
                transform_indices, node_mask, overlay_params, model_transforms);
            if (state.active) {
                const uint camera_height =
                    uniforms.camera_height == 0 ? uniforms.image_height : uniforms.camera_height;
                const uint camera_width =
                    uniforms.camera_width == 0 ? uniforms.image_width : uniforms.camera_width;
                const Camera cam = load_camera(uniforms.world_view_transform,
                    uniforms.fx, uniforms.fy, uniforms.cx, uniforms.cy,
                    base_camera_model(uniforms.camera_model), uniforms.dist_coeffs,
                    camera_height, camera_width);
              #if LFS_3DGUT_PROJECTION
                survives = splat_center_potentially_visible_gut(state.mean_ws, cam);
              #else
                survives = splat_center_potentially_visible(state.mean_ws, cam);
              #endif
            }
        }
    }

    const uint survivors_in_wave = WaveActiveCountBits(survives);
    const uint lane_offset = WavePrefixCountBits(survives);
    uint wave_base = 0;
    if (WaveIsFirstLane() && survivors_in_wave > 0u) {
        InterlockedAdd(out_survivor_state[LFS_VK_SURVIVOR_STATE_COUNT_WORD_OFFSET],
                       survivors_in_wave,
                       wave_base);
    }
    wave_base = WaveReadLaneFirst(wave_base);
    if (survives) {
        out_survivors[wave_base + lane_offset] = int32_t(render_idx);
    }
}

#elif CULL_ENTRY == CULL_ENTRY_PREPARE

layout(binding=0) RWStructuredBuffer<uint32_t> survivor_state;
layout(binding=1) RWStructuredBuffer<uint32_t> out_emit_count;

[shader("compute")]
[numthreads(1, 1, 1)]
void main() {
    // Survivor projection groups: SUBGROUP_SIZE threads each, matching the
    // projection shader's numthreads. Also zeroes the projection's compact-slot
    // append counter so the host does not need a separate fill.
    const uint count = survivor_state[LFS_VK_SURVIVOR_STATE_COUNT_WORD_OFFSET];
    const uint projection = LFS_VK_SURVIVOR_STATE_PROJECTION_WORD_OFFSET;
    survivor_state[projection + LFS_VK_DISPATCH_X_WORD_OFFSET] =
        max((count + SUBGROUP_SIZE - 1u) / SUBGROUP_SIZE, 1u);
    survivor_state[projection + LFS_VK_DISPATCH_Y_WORD_OFFSET] = 1u;
    survivor_state[projection + LFS_VK_DISPATCH_Z_WORD_OFFSET] = 1u;
    out_emit_count[0] = 0u;
}

#endif
