// HiGS chain fan-out: converts the survivor projection's compact-slot append
// counter into everything the visible-bounded chain dispatches with — the
// clamped visible count, the depth radix-sort args, the generic per-element
// args, and the indirect cumsum level counts/args. One 1x1 dispatch replaces
// the visible_flags / visible_prefix / compact passes of the N-wide path.

#include "config.slang"
#include "indirect_layout.inc"

struct Uniforms {
    uint32_t visible_capacity;
    uint32_t sort_partition_size;
    uint32_t pad0;
    uint32_t pad1;
};

static const uint32_t CUMSUM_BLOCK = 1024u;

layout(binding=0) StructuredBuffer<uint32_t> emit_count;
// [0] = bounded visible count; [1] = raw emit total for the host-side loud
// overflow check.
layout(binding=1) RWStructuredBuffer<uint32_t> out_visible_count;
// Layout is defined by LFS_VK_VISIBLE_CHAIN_DISPATCH_* in indirect_layout.inc.
layout(binding=2) RWStructuredBuffer<uint32_t> out_dispatch;
// Indirect cumsum element counts per level: [0]=visible, [1]=ceil(v/1024),
// [2]=ceil([1]/1024), [3]=1.
layout(binding=3) RWStructuredBuffer<int32_t> out_cumsum_counts;

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uniform Uniforms uniforms) {
    const uint32_t raw = emit_count[0];
    const uint32_t visible = min(raw, uniforms.visible_capacity);
    out_visible_count[0] = visible;
    out_visible_count[1] = raw;

    const uint32_t partition_size = max(uniforms.sort_partition_size, 1u);
    const uint32_t radix = LFS_VK_VISIBLE_CHAIN_DISPATCH_RADIX_WORD_OFFSET;
    out_dispatch[radix + LFS_VK_DISPATCH_X_WORD_OFFSET] =
        max((visible + partition_size - 1u) / partition_size, 1u);
    out_dispatch[radix + LFS_VK_DISPATCH_Y_WORD_OFFSET] = 1u;
    out_dispatch[radix + LFS_VK_DISPATCH_Z_WORD_OFFSET] = 1u;

    const uint32_t per_element = LFS_VK_VISIBLE_CHAIN_DISPATCH_PER_ELEMENT_WORD_OFFSET;
    out_dispatch[per_element + LFS_VK_DISPATCH_X_WORD_OFFSET] =
        max((visible + 63u) / 64u, 1u);
    out_dispatch[per_element + LFS_VK_DISPATCH_Y_WORD_OFFSET] = 1u;
    out_dispatch[per_element + LFS_VK_DISPATCH_Z_WORD_OFFSET] = 1u;

    const uint32_t c1 = (visible + CUMSUM_BLOCK - 1u) / CUMSUM_BLOCK;
    const uint32_t c2 = (c1 + CUMSUM_BLOCK - 1u) / CUMSUM_BLOCK;
    const uint32_t cumsum_l0 = LFS_VK_VISIBLE_CHAIN_DISPATCH_CUMSUM_L0_WORD_OFFSET;
    out_dispatch[cumsum_l0 + LFS_VK_DISPATCH_X_WORD_OFFSET] = max(c1, 1u);
    out_dispatch[cumsum_l0 + LFS_VK_DISPATCH_Y_WORD_OFFSET] = 1u;
    out_dispatch[cumsum_l0 + LFS_VK_DISPATCH_Z_WORD_OFFSET] = 1u;
    const uint32_t cumsum_l1 = LFS_VK_VISIBLE_CHAIN_DISPATCH_CUMSUM_L1_WORD_OFFSET;
    out_dispatch[cumsum_l1 + LFS_VK_DISPATCH_X_WORD_OFFSET] = max(c2, 1u);
    out_dispatch[cumsum_l1 + LFS_VK_DISPATCH_Y_WORD_OFFSET] = 1u;
    out_dispatch[cumsum_l1 + LFS_VK_DISPATCH_Z_WORD_OFFSET] = 1u;

    out_cumsum_counts[0] = int32_t(visible);
    out_cumsum_counts[1] = int32_t(c1);
    out_cumsum_counts[2] = int32_t(c2);
    out_cumsum_counts[3] = 1;
}
