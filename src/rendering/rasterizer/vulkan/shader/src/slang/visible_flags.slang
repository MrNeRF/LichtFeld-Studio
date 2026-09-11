#include "config.slang"
#include "visible_mask.inc"

struct Uniforms {
    uint32_t num_splats;
    uint32_t pad0;
    uint32_t pad1;
    uint32_t pad2;
};

layout(binding=0) StructuredBuffer<int32_t> tiles_touched;
layout(binding=1) RWStructuredBuffer<uint32_t> out_visible_mask;
layout(binding=2) RWStructuredBuffer<int32_t> out_visible_block_counts;

[shader("compute")]
[numthreads(LFS_VK_VISIBLE_MASK_WORKGROUP_SIZE, 1, 1)]
void main(
    uint3 groupID : SV_GroupID,
    uint3 groupThreadID : SV_GroupThreadID,
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform Uniforms uniforms
) {
    const uint32_t lane = groupThreadID.x;
    const uint32_t group = groupID.x;
    const uint32_t idx = dispatchThreadID.x;
    const bool visible = idx < uniforms.num_splats && tiles_touched[idx] > 0;

    // Clear the four words and the block count before the ballot writes. The
    // barrier keeps those initial stores ahead of all participating atomics,
    // including the partially occupied final workgroup.
    if (lane < LFS_VK_VISIBLE_MASK_WORDS_PER_WORKGROUP)
        out_visible_mask[group * LFS_VK_VISIBLE_MASK_WORDS_PER_WORKGROUP + lane] = 0u;
    if (lane == 0u)
        out_visible_block_counts[group] = 0;
    GroupMemoryBarrierWithGroupSync();

    if (visible) {
        const uint32_t word = group * LFS_VK_VISIBLE_MASK_WORDS_PER_WORKGROUP +
                              lane / LFS_VK_VISIBLE_MASK_WORD_BITS;
        const uint32_t bit = 1u << (lane % LFS_VK_VISIBLE_MASK_WORD_BITS);
        InterlockedOr(out_visible_mask[word], bit);
    }

    const uint32_t wave_count = WaveActiveCountBits(visible);
    if (WaveIsFirstLane())
        InterlockedAdd(out_visible_block_counts[group], int32_t(wave_count));
}
