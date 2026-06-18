#include "strategy_utils.slang"

#define MORTON_SORT_PHASE_ComputeStats 0
#define MORTON_SORT_PHASE_GenerateKeys 1
#define MORTON_SORT_PHASE_ApplyIndices 2
#define MORTON_SORT_PHASE_ApplyIndicesSH 3
#define MORTON_SORT_PHASE_UpdateBuffer 4
#define MORTON_SORT_PHASE_UpdateBufferSH 5

#ifndef MORTON_SORT_PHASE
#define MORTON_SORT_PHASE -1
#endif


#if MORTON_SORT_PHASE == MORTON_SORT_PHASE_ComputeStats

layout(binding=0) StructuredBuffer<float> means;  // (N, 3)
layout(binding=1) RWByteAddressBuffer stats;  // (2, 3), sum of x and x^2

groupshared float3 shared_mean[SUBGROUP_SIZE], shared_mean2[SUBGROUP_SIZE];

[numthreads(SUBGROUP_SIZE*SUBGROUP_SIZE, 1, 1)]
void main(
    uint3 threadID : SV_DispatchThreadID,
    uint3 groupThreadID : SV_GroupThreadID,
    uniform uint32_t num_splats
) {
    const uint32_t tid = threadID.x;

    float3 mean = tid >= num_splats ? float3(0) :
        float3(means[3*tid], means[3*tid+1], means[3*tid+2]);
    float3 mean2 = mean * mean;
    if (!isfinite(dot(mean2, mean2)))
        mean = mean2 = float3(0);

    // for numerical stability, i.e. prevent overflow/underflow
    mean /= sqrt(float(num_splats));
    mean2 /= sqrt(float(num_splats));

    // block reduce before atomic
    uint subgroup_id = groupThreadID.x / SUBGROUP_SIZE;
    uint lane_id = groupThreadID.x % SUBGROUP_SIZE;
    mean = WaveActiveSum(mean);
    mean2 = WaveActiveSum(mean2);
    if (lane_id == 0) {
        shared_mean[subgroup_id] = mean;
        shared_mean2[subgroup_id] = mean2;
    }
    GroupMemoryBarrierWithGroupSync();

    if (subgroup_id == 0) {
        mean = shared_mean[lane_id];
        mean2 = shared_mean2[lane_id];
        mean = WaveActiveSum(mean);
        mean2 = WaveActiveSum(mean2);
        if (lane_id == 0) {
          #if USE_EMULATED_F32_ATOMIC
            AtomicAddFloat(stats, 0*sizeof(float), mean.x);
            AtomicAddFloat(stats, 1*sizeof(float), mean.y);
            AtomicAddFloat(stats, 2*sizeof(float), mean.z);
            AtomicAddFloat(stats, 3*sizeof(float), mean2.x);
            AtomicAddFloat(stats, 4*sizeof(float), mean2.y);
            AtomicAddFloat(stats, 5*sizeof(float), mean2.z);
          #else
            stats.InterlockedAddF32(0*sizeof(float), mean.x);
            stats.InterlockedAddF32(1*sizeof(float), mean.y);
            stats.InterlockedAddF32(2*sizeof(float), mean.z);
            stats.InterlockedAddF32(3*sizeof(float), mean2.x);
            stats.InterlockedAddF32(4*sizeof(float), mean2.y);
            stats.InterlockedAddF32(5*sizeof(float), mean2.z);
          #endif
        }
    }
    

}

#endif  // MORTON_SORT_PHASE == MORTON_SORT_PHASE_ComputeMeanStd



#if MORTON_SORT_PHASE == MORTON_SORT_PHASE_GenerateKeys

#ifndef USE_64_BIT_MORTON_KEYS
#define USE_64_BIT_MORTON_KEYS 0
#endif

layout(binding=0) StructuredBuffer<float> means;  // (N, 3)
layout(binding=1) StructuredBuffer<float> stats;  // (2, 3), sum of x and x^2

#if USE_64_BIT_MORTON_KEYS
layout(binding=2) RWStructuredBuffer<uint64_t> out_unsorted_keys;  // (N,)
#else
layout(binding=2) RWStructuredBuffer<sortingKey_t> out_unsorted_keys;  // (N,)
#endif
layout(binding=3) RWStructuredBuffer<int32_t> out_unsorted_gauss_idx;  // (N,)

[ForceInline]
float erf(float x) {
    float t = abs(x);
    float d = (((0.078108*t+0.000972)*t+0.230389)*t+0.278393)*t+1.0;
    return sign(x) * (1.0 - 1.0/(d*d*d*d));
}

[ForceInline]
float normcdf(float x) {
    return clamp(0.5 + 0.5 * erf(x * sqrt(0.5)), 0.0, 1.0);
}

#if USE_64_BIT_MORTON_KEYS

[ForceInline]
uint64_t insert_2_zeros_between_bits(uint64_t x) {
    x = (x | (x << 32)) & (uint64_t)0xFFFF00000000FFFFull;
    x = (x | (x << 16)) & (uint64_t)0x00FF0000FF0000FFull;
    x = (x | (x << 8))  & (uint64_t)0x100F00F00F00F00Full;
    x = (x | (x << 4))  & (uint64_t)0x10C30C30C30C30C3ull;
    x = (x | (x << 2))  & (uint64_t)0x1249249249249249ull;
    return x;
}

#else

[ForceInline]
uint32_t insert_2_zeros_between_bits(uint32_t x) {
    x = (x | (x << 16)) & (uint32_t)0xFF0000FFu;
    x = (x | (x << 8))  & (uint32_t)0x0F00F00Fu;
    x = (x | (x << 4))  & (uint32_t)0xC30C30C3u;
    x = (x | (x << 2))  & (uint32_t)0x49249249u;
    return x;
}

#endif

[numthreads(256, 1, 1)]
void main(
    uint3 threadID : SV_DispatchThreadID,
    uniform uint32_t num_splats
) {
    const uint32_t tid = threadID.x;
    if (tid >= num_splats)
        return;

    // normalize position to between 0 and 1
    float3 pos_mean = float3(stats[0], stats[1], stats[2]) / sqrt(float(num_splats));
    float3 pos_varp = float3(stats[3], stats[4], stats[5]) / sqrt(float(num_splats)) - pos_mean*pos_mean;
    float3 pos_stdp = sqrt(max(pos_varp, float3(1e-6)));

    float3 pos = float3(means[3*tid], means[3*tid+1], means[3*tid+2]);
    pos = (pos - pos_mean) / pos_stdp;
    pos = float3(normcdf(pos.x), normcdf(pos.y), normcdf(pos.z));

    // generate morton keys
#if USE_64_BIT_MORTON_KEYS
    static constexpr uint kMortonBitsPerDim = 21;
    static constexpr uint64_t mask_comp = ((uint64_t)1 << (uint64_t)kMortonBitsPerDim) - 1;
    uint64_t x = uint64_t(pos.x * (float)(1 << kMortonBitsPerDim) + 0.5f) & mask_comp;
    uint64_t y = uint64_t(pos.y * (float)(1 << kMortonBitsPerDim) + 0.5f) & mask_comp;
    uint64_t z = uint64_t(pos.z * (float)(1 << kMortonBitsPerDim) + 0.5f) & mask_comp;
    static constexpr uint64_t mask_full = (((uint64_t)1 << (uint64_t)(3*kMortonBitsPerDim)) - 1);
    x = insert_2_zeros_between_bits(x) & mask_full;
    y = insert_2_zeros_between_bits(y) & mask_full;
    z = insert_2_zeros_between_bits(z) & mask_full;
    uint64_t morton = (x * 2 + y) * 2 + z;
#else
    static constexpr uint kMortonBitsPerDim = 10;
    static constexpr uint32_t mask_comp = ((uint32_t)1 << (uint32_t)kMortonBitsPerDim) - 1;
    uint32_t x = uint32_t(pos.x * (float)(1 << kMortonBitsPerDim) + 0.5f) & mask_comp;
    uint32_t y = uint32_t(pos.y * (float)(1 << kMortonBitsPerDim) + 0.5f) & mask_comp;
    uint32_t z = uint32_t(pos.z * (float)(1 << kMortonBitsPerDim) + 0.5f) & mask_comp;
    static constexpr uint32_t mask_full = (((uint32_t)1 << (uint32_t)(3*kMortonBitsPerDim)) - 1);
    x = insert_2_zeros_between_bits(x) & mask_full;
    y = insert_2_zeros_between_bits(y) & mask_full;
    z = insert_2_zeros_between_bits(z) & mask_full;
    sortingKey_t morton = (sortingKey_t)((x * 2 + y) * 2 + z);
#endif

    out_unsorted_keys[tid] = morton;
    out_unsorted_gauss_idx[tid] = tid;
}

#endif  // MORTON_SORT_PHASE == MORTON_SORT_PHASE_GenerateKeys




#if MORTON_SORT_PHASE == MORTON_SORT_PHASE_ApplyIndices

layout(binding=0) RWStructuredBuffer<int32_t> sorted_gauss_idx;  // (N,)
layout(binding=1) StructuredBuffer<float> buffer_in;
layout(binding=2) RWStructuredBuffer<float> buffer_out;

[numthreads(1024, 1, 1)]
void main(
    uint3 threadID : SV_DispatchThreadID,
    uniform uint32_t[2] uniforms
) {
    uint32_t num_splats = uniforms[0];
    uint32_t stride = uniforms[1];
    const uint32_t tid = threadID.x;
    if (tid >= num_splats*stride)
        return;

    int32_t idx = sorted_gauss_idx[tid / stride];
    float value = buffer_in[idx * stride + (tid % stride)];
    buffer_out[tid] = value;
}

#endif  // MORTON_SORT_PHASE == MORTON_SORT_PHASE_ApplyIndices


#if MORTON_SORT_PHASE == MORTON_SORT_PHASE_UpdateBuffer

layout(binding=0) StructuredBuffer<float> buffer_temp;
layout(binding=1) RWStructuredBuffer<float> buffer_in;

[numthreads(1024, 1, 1)]
void main(
    uint3 threadID : SV_DispatchThreadID,
    uniform uint[2] uniforms
) {
    uint32_t num_splats = uniforms[0];
    uint32_t stride = uniforms[1];
    const uint32_t tid = threadID.x;
    if (tid >= num_splats*stride)
        return;

    buffer_in[tid] = buffer_temp[tid];
}

#endif  // MORTON_SORT_PHASE == MORTON_SORT_PHASE_UpdateBuffer



#if MORTON_SORT_PHASE == MORTON_SORT_PHASE_ApplyIndicesSH

layout(binding=0) RWStructuredBuffer<int32_t> sorted_gauss_idx;  // (N,)
layout(binding=1) StructuredBuffer<float4> buffer_in;
layout(binding=2) RWStructuredBuffer<float4> buffer_out;

[numthreads(1024, 1, 1)]
void main(
    uint3 threadID : SV_DispatchThreadID,
    uniform uint[3] uniforms
) {
    uint num_splats = uniforms[0];
    uint offset = uniforms[1];

    const uint32_t tid = threadID.x;
    if (tid >= num_splats)
        return;

    int32_t id_dst = tid;
    int32_t id_src = sorted_gauss_idx[id_dst];

    buffer_out[id_dst] = buffer_in[shAt(id_src, offset)];
}

#endif  // MORTON_SORT_PHASE == MORTON_SORT_PHASE_ApplyIndicesSH


#if MORTON_SORT_PHASE == MORTON_SORT_PHASE_UpdateBufferSH

layout(binding=0) StructuredBuffer<float4> buffer_temp;
layout(binding=1) RWStructuredBuffer<float4> buffer_in;

[numthreads(1024, 1, 1)]
void main(
    uint3 threadID : SV_DispatchThreadID,
    uniform uint[3] uniforms
) {
    uint num_splats = uniforms[0];
    uint offset = uniforms[1];

    const uint32_t tid = threadID.x;
    if (tid >= num_splats)
        return;

    buffer_in[shAt(tid, offset)] = buffer_temp[tid];
}

#endif  // MORTON_SORT_PHASE == MORTON_SORT_PHASE_UpdateBufferSH
