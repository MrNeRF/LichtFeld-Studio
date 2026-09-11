#include "config.slang"
#include "visible_mask.inc"

struct Uniforms {
    uint32_t num_splats;
    uint32_t pad0;
    uint32_t pad1;
    uint32_t pad2;
};

layout(binding=0) StructuredBuffer<uint32_t> visible_mask;
layout(binding=1) StructuredBuffer<int32_t> visible_block_prefix;
layout(binding=2) StructuredBuffer<uint32_t> primitive_depth_keys;

layout(binding=3) RWStructuredBuffer<sortingKey_t> out_compact_depth_keys;
layout(binding=4) RWStructuredBuffer<int32_t> out_compact_indices;

[shader("compute")]
[numthreads(LFS_VK_VISIBLE_MASK_WORKGROUP_SIZE, 1, 1)]
void main(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform Uniforms uniforms
) {
    uint32_t primitive_idx = dispatchThreadID.x;
    if (primitive_idx >= uniforms.num_splats)
        return;
    const uint32_t group = primitive_idx / LFS_VK_VISIBLE_MASK_WORKGROUP_SIZE;
    const uint32_t lane = primitive_idx % LFS_VK_VISIBLE_MASK_WORKGROUP_SIZE;
    const uint32_t word = visible_mask[group * LFS_VK_VISIBLE_MASK_WORDS_PER_WORKGROUP +
                                       lane / LFS_VK_VISIBLE_MASK_WORD_BITS];
    const uint32_t bit_in_word = lane % LFS_VK_VISIBLE_MASK_WORD_BITS;
    const uint32_t bit = 1u << bit_in_word;
    if ((word & bit) == 0u)
        return;

    // visible_block_prefix is inclusive. Its preceding entry is therefore
    // the exclusive base for this workgroup; the loop adds all set bits in
    // earlier words and countbits(bits below lane) adds the in-word rank.
    const uint32_t block_base = group == 0u
                                    ? 0u
                                    : uint32_t(max(visible_block_prefix[group - 1u], 0));
    uint32_t below = 0u;
    for (uint32_t prior_word = 0u;
         prior_word < lane / LFS_VK_VISIBLE_MASK_WORD_BITS;
         ++prior_word) {
        below += countbits(visible_mask[group * LFS_VK_VISIBLE_MASK_WORDS_PER_WORKGROUP +
                                        prior_word]);
    }
    below += countbits(word & ((1u << bit_in_word) - 1u));

    uint32_t compact_idx = block_base + below;
    // Host sizes compact outs to num_splats (sort capacity); clamp data-dependent
    // write index the same way lod_* guards output_capacity / protected_capacity.
    if (compact_idx >= uniforms.num_splats)
        return;
    out_compact_depth_keys[compact_idx] = sortingKey_t(primitive_depth_keys[primitive_idx]);
    out_compact_indices[compact_idx] = int32_t(primitive_idx);
}
