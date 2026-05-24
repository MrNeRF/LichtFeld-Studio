#define GROUP_SIZE 256
#define GROUP_SIZE_SPARSE 64

#define MCMC_PHASE_InjectNoise 0
#define MCMC_PHASE_Probabilities 1

#define MCMC_PHASE_RelocationIndexMap 10
#define MCMC_PHASE_ComputeRelocation 11
#define MCMC_PHASE_UpdateRelocation 12

#define MCMC_PHASE_AddIndexMap 20
#define MCMC_PHASE_ComputeAdd 21
#define MCMC_PHASE_UpdateAdd 22

#ifndef MCMC_PHASE
#define MCMC_PHASE -1
#endif

#include "strategy_utils.slang"


struct InjectNoiseUniforms {
    uint32_t array_size;
    uint32_t seed;
    uint32_t image_width;
    uint32_t image_height;
    float noise_lr;
};

struct Uniforms {
    uint32_t array_size;
    uint32_t num_add;
    uint32_t seed;
    float min_opacity;
};




#if MCMC_PHASE == MCMC_PHASE_InjectNoise

layout(binding=0) StructuredBuffer<int32_t> radii;  // (N,)
layout(binding=1) StructuredBuffer<float4> rotations;  // (N, 4)
layout(binding=2) RWStructuredBuffer<float> means;  // (N, 3)
layout(binding=3) RWStructuredBuffer<float4> scales_opacs;  // (N, 4)

#define injectNoise main

float op_sigmoid(float x) {
    static const float kK = 100.0;
    static const float kX0 = 0.995;
    return 1.0 / (1.0 + exp(-kK * (x - kX0)));
}

[numthreads(GROUP_SIZE, 1, 1)]
void injectNoise(
    int32_t3 thread_id : SV_DispatchThreadID,
    uniform InjectNoiseUniforms uniforms
) {
    const uint32_t num_splats = uniforms.array_size;
    const uint32_t tid = thread_id.x;
    if (tid >= num_splats)
        return;
    
    float4 quat = normalize(rotations[tid]);
    float4 scale_opac = scales_opacs[tid];
    float3 scale = scale_opac.xyz;
    float opac = scale_opac.w;

    // inject noise
    float3x3 cov = sqrt_cov_from_quat_scales(quat, scale);
    cov = mul(cov, transpose(cov));
    float3 noise = randn3(tid, uniforms.seed);
    noise = mul(cov, noise);
    noise *= op_sigmoid(1.0-opac) * uniforms.noise_lr;

    write_mean(read_mean(tid, means) + noise, tid, means);
}

#endif  // MCMC_PHASE == MCMC_PHASE_InjectNoise



#if MCMC_PHASE == MCMC_PHASE_Probabilities

layout(binding=0) StructuredBuffer<float4> scales_opacs;  // (N,)
layout(binding=1) RWStructuredBuffer<uint32_t> sample_probs;  // (N,)
#define computeProbabilities main

[numthreads(GROUP_SIZE, 1, 1)]
void computeProbabilities(
    int32_t3 thread_id : SV_DispatchThreadID,
    uniform Uniforms uniforms
) {
    const uint32_t num_splats = uniforms.array_size;
    const uint32_t tid = thread_id.x;
    if (tid >= num_splats)
        return;

    float opac = scales_opacs[tid].w;
    opac = clamp(opac, 1e-6, 1-1e-6);

    // low opacity -> 0; high opacity -> nonzero, prevent overflow
    const float kMinOpacity = uniforms.min_opacity;
    float max_opac = float(0x70000000) / float(num_splats);
    sample_probs[tid] = (opac <= kMinOpacity || !isfinite(opac)) ? 0 :
        max(uint32_t(round(opac * max_opac)), 1u);
}

#endif  // MCMC_PHASE == MCMC_PHASE_Probabilities



/*
To relocate splats:

1) Compute relocation probabilities for all Gaussians
2) Prefix sum the probability array
3) Compute relocation index map
4) Compute attributes for Gaussians relocated to
5) Update attributes for Gaussians relocated from

*/

[ForceInline]
float log_factorial(float x) {
    // least squares for x = [0, 1... 50]
    // absolute error: max 4.6e-5, L1 9.3e-6
    // notebook: https://www.desmos.com/calculator/mik6cz7h5v
    static const float kA = 0.996763591291;
    static const float kH = 2.73507778369;
    static const float kB = 0.978531458767;
    static const float kC2 = -0.0400131099548;
    static const float kC1 = -0.596969060666;
    static const float kD3 = 0.00251628691783;
    static const float kD2 = 0.0959712063178;
    static const float kD1 = 0.803207449544;
    return (kA * log(x + kH) - kB +
        (kC2 * x + kC1) / (((kD3 * x + kD2) * x + kD1) * x + 1.0)
    ) * x;
}

[ForceInline]
float binom(float n, float k) {
    return exp(log_factorial(n) - log_factorial(k) - log_factorial(n-k));
}


void relocation(inout float opacity, inout float3 scale, int n_idx) {
    n_idx = min(n_idx, 50);  // log_factorial only fits to 50

    float new_opacity = 1.0 - pow(1.0-opacity, 1.0 / n_idx);

    float denom_sum = 0.0;
    for (int i = 1; i <= n_idx; ++i) {
        for (int k = 0; k <= (i - 1); ++k) {
            denom_sum += binom(i-1, k) * 
                (cos(M_PI*k) / sqrt(k+1)) *  // (-1)^k / sqrt(k+1)
                pow(new_opacity, k+1);
        }
    }
    float coeff = (opacity / denom_sum);

    opacity = new_opacity;
    scale = coeff * scale;
}



#if MCMC_PHASE == MCMC_PHASE_RelocationIndexMap

layout(binding=0) StructuredBuffer<uint32_t> sample_probs;  // (N,)
layout(binding=1) StructuredBuffer<uint32_t> sample_probs_cumsum;  // (N,)
layout(binding=2) RWStructuredBuffer<uint32_t> index_map;  // (N,)
layout(binding=3) RWByteAddressBuffer n_idx_buffer;  // (N,)
#define computeRelocationIndexMap main

[numthreads(GROUP_SIZE, 1, 1)]
void computeRelocationIndexMap(
    int32_t3 thread_id : SV_DispatchThreadID,
    uniform Uniforms uniforms
) {
    const uint32_t array_size = uniforms.array_size;
    const uint32_t tid = thread_id.x;
    if (tid >= array_size)
        return;
    
    uint32_t prob = sample_probs[tid];

    // not relocated, put same ID
    if (prob != 0) {
        index_map[tid] = tid;
        return;
    }

    // binary search for random index
    // find the min idx such that sample_probs_cumsum[idx] >= rand
    uint32_t rand_max = sample_probs_cumsum[array_size - 1];
  #if USE_EMULATED_INT64
    uint32_t rand = mod64(hash_u32_u322(uniforms.seed, tid), rand_max) + 1u;
  #else
    uint32_t rand = uint32_t(hash_u32_u64(uniforms.seed, tid) % rand_max + 1);
  #endif
    uint32_t lo = 0;
    uint32_t hi = array_size - 1;
    while (lo < hi) {
        uint32_t mid = lo + ((hi - lo) >> 1);
        if (sample_probs_cumsum[mid] < rand)
            lo = mid + 1;
        else
            hi = mid;
    }
    index_map[tid] = lo;

    static const uint sizeof_int = 4;
    n_idx_buffer.InterlockedAdd(lo*sizeof_int, 1);
}

#endif  // MCMC_PHASE == MCMC_PHASE_RelocationIndexMap



#if MCMC_PHASE == MCMC_PHASE_ComputeRelocation

layout(binding=0) StructuredBuffer<uint32_t> n_idx_buffer;  // (N,)

layout(binding=1) RWStructuredBuffer<float4> scales_opacs;  // (N, 4)

layout(binding=2) RWStructuredBuffer<float> g_means;  // (N, 2, 3)
layout(binding=3) RWStructuredBuffer<float4> g_sh_coeffs_1;  // (N, 16, 3) = (N, 12, 4)
layout(binding=4) RWStructuredBuffer<float4> g_sh_coeffs_2;  // (N, 16, 3) = (N, 12, 4)
layout(binding=5) RWStructuredBuffer<float4> g_rotations;  // (N, 2, 4)
layout(binding=6) RWStructuredBuffer<float4> g_scales_opacs;  // (N, 2, 4)

#define computeRelocation main

[numthreads(GROUP_SIZE_SPARSE, 1, 1)]
void computeRelocation(
    int32_t3 thread_id : SV_DispatchThreadID,
    uniform Uniforms uniforms
) {
    uint32_t num_splats = uniforms.array_size;
    const uint32_t cur_idx = thread_id.x;
    if (cur_idx >= num_splats)
        return;

    // only compute on indices being relocated to
    uint n_idx = n_idx_buffer[cur_idx] + 1;
    if (n_idx == 1)
        return;

    // compute relocation
    float4 scale_opac = scales_opacs[cur_idx];
    float3 scale = scale_opac.xyz;
    float opac = scale_opac.w;
    relocation(opac, scale, int(n_idx));
    opac = clamp(opac, uniforms.min_opacity, 0.999999);
    scales_opacs[cur_idx] = float4(scale, opac);

    // set grad to zero
    #if 1
    _FOR_LOOP(i, 2) write_mean(float3(0.0), cur_idx, g_means, 2, i);
    _FOR_LOOP(i, 12) g_sh_coeffs_1[shAt(cur_idx, i)] = float4(0.0);
    _FOR_LOOP(i, 12) g_sh_coeffs_2[shAt(cur_idx, i)] = float4(0.0);
    _FOR_LOOP(i, 2) g_rotations[cur_idx*2+i] = float4(0.0);
    _FOR_LOOP(i, 2) g_scales_opacs[cur_idx*2+i] = float4(0.0);
    #endif

}

#endif  // MCMC_PHASE == MCMC_PHASE_ComputeRelocation



#if MCMC_PHASE == MCMC_PHASE_UpdateRelocation

layout(binding=0) StructuredBuffer<uint32_t> index_map;  // (N,)

layout(binding=1) RWStructuredBuffer<float> means;  // (N, 3)
layout(binding=2) RWStructuredBuffer<float4> sh_coeffs;  // (N, 16, 3) = (N, 12, 4)
layout(binding=3) RWStructuredBuffer<float4> rotations;  // (N, 4)
layout(binding=4) RWStructuredBuffer<float4> scales_opacs;  // (N, 4)

layout(binding=5) RWStructuredBuffer<float> g_means;  // (N, 2, 3)
layout(binding=6) RWStructuredBuffer<float4> g_sh_coeffs_1;  // (N, 16, 3) = (N, 12, 4)
layout(binding=7) RWStructuredBuffer<float4> g_sh_coeffs_2;  // (N, 16, 3) = (N, 12, 4)
layout(binding=8) RWStructuredBuffer<float4> g_rotations;  // (N, 2, 4)
layout(binding=9) RWStructuredBuffer<float4> g_scales_opacs;  // (N, 2, 4)

#define updateRelocation main

[numthreads(GROUP_SIZE_SPARSE, 1, 1)]
void updateRelocation(
    int32_t3 thread_id : SV_DispatchThreadID,
    uniform Uniforms uniforms
) {
    uint32_t num_splats = uniforms.array_size;
    const uint32_t id_dst = thread_id.x;
    if (id_dst >= num_splats)
        return;

    // only update on those relocated from
    uint32_t id_src = index_map[id_dst];
    if (id_src == id_dst)
        return;

    // copy components
    write_mean(read_mean(id_src, means), id_dst, means);
    _FOR_LOOP(i, 12) sh_coeffs[shAt(id_dst,i)] = sh_coeffs[shAt(id_src,i)];
    rotations[id_dst] = rotations[id_src];
    scales_opacs[id_dst] = scales_opacs[id_src];

    // set grad to zero (not in gsplat)
    #if 0
    _FOR_LOOP(i, 2) write_mean(float3(0.0), id_dst, g_means, 2, i);
    _FOR_LOOP(i, 12) g_sh_coeffs_1[shAt(id_dst,i)] = float4(0.0);
    _FOR_LOOP(i, 12) g_sh_coeffs_2[shAt(id_dst,i)] = float4(0.0);
    _FOR_LOOP(i, 2) g_rotations[id_dst*2+i] = float4(0.0);
    _FOR_LOOP(i, 2) g_scales_opacs[id_dst*2+i] = float4(0.0);
    #endif
}

#endif  // MCMC_PHASE == MCMC_PHASE_UpdateRelocation



/*
To add more splats:

1) Compute relocation probabilities for all Gaussians
2) Prefix sum the probability array
3) [dN] Compute add index map
4) [N] Compute attributes for Gaussians added to
5) [dN] Update attributes for Gaussians added from

*/


#if MCMC_PHASE == MCMC_PHASE_AddIndexMap

layout(binding=0) StructuredBuffer<uint32_t> sample_probs_cumsum;  // (N,)
layout(binding=1) RWStructuredBuffer<uint32_t> index_map;  // (dN,)
layout(binding=2) RWByteAddressBuffer n_idx_buffer;  // (N,)
#define computeAddIndexMap main

[numthreads(GROUP_SIZE, 1, 1)]
void computeAddIndexMap(
    int32_t3 thread_id : SV_DispatchThreadID,
    uniform Uniforms uniforms
) {
    const uint32_t tid = thread_id.x;
    if (tid >= uniforms.num_add)
        return;

    const uint32_t num_splats = uniforms.array_size;

    // binary search for random index
    // find the min idx such that sample_probs_cumsum[idx] >= rand
    uint32_t rand_max = sample_probs_cumsum[num_splats - 1];
  #if USE_EMULATED_INT64
    uint32_t rand = mod64(hash_u32_u322(uniforms.seed+1, tid), rand_max) + 1u;
  #else
    uint32_t rand = uint32_t(hash_u32_u64(uniforms.seed+1, tid) % rand_max + 1);
  #endif
    uint32_t lo = 0;
    uint32_t hi = num_splats - 1;
    while (lo < hi) {
        uint32_t mid = lo + ((hi - lo) >> 1);
        if (sample_probs_cumsum[mid] < rand)
            lo = mid + 1;
        else
            hi = mid;
    }
    index_map[tid] = lo;

    static const uint sizeof_int = 4;
    n_idx_buffer.InterlockedAdd(lo*sizeof_int, 1);
}

#endif  // MCMC_PHASE == MCMC_PHASE_AddIndexMap



#if MCMC_PHASE == MCMC_PHASE_ComputeAdd

layout(binding=0) StructuredBuffer<uint32_t> n_idx_buffer;  // (N,)

layout(binding=1) RWStructuredBuffer<float4> scales_opacs;  // (N, 4)

#define computeAdd main

[numthreads(GROUP_SIZE_SPARSE, 1, 1)]
void computeAdd(
    int32_t3 thread_id : SV_DispatchThreadID,
    uniform Uniforms uniforms
) {
    uint32_t num_splats = uniforms.array_size;
    const uint32_t cur_idx = thread_id.x;
    if (cur_idx >= num_splats)
        return;

    // only compute on indices being relocated to
    uint n_idx = n_idx_buffer[cur_idx] + 1;
    if (n_idx == 1)
        return;

    // compute relocation
    float4 scale_opac = scales_opacs[cur_idx];
    float3 scale = scale_opac.xyz;
    float opac = scale_opac.w;
    relocation(opac, scale, int(n_idx));
    opac = clamp(opac, uniforms.min_opacity, 0.999999);
    scales_opacs[cur_idx] = float4(scale, opac);

    // don't change grad
}

#endif  // MCMC_PHASE == MCMC_PHASE_ComputeAdd



#if MCMC_PHASE == MCMC_PHASE_UpdateAdd

layout(binding=0) StructuredBuffer<uint32_t> index_map;  // (dN,)

layout(binding=1) RWStructuredBuffer<float> means;  // (N+dN, 3)
layout(binding=2) RWStructuredBuffer<float4> sh_coeffs;  // (N+dN, 16, 3) = (N+dN, 12, 4)
layout(binding=3) RWStructuredBuffer<float4> rotations;  // (N+dN, 4)
layout(binding=4) RWStructuredBuffer<float4> scales_opacs;  // (N+dN, 4)

layout(binding=5) RWStructuredBuffer<float> g_means;  // (N+dN, 2, 3)
layout(binding=6) RWStructuredBuffer<float4> g_sh_coeffs_1;  // (N+dN, 16, 3) = (N+dN, 12, 4)
layout(binding=7) RWStructuredBuffer<float4> g_sh_coeffs_2;  // (N+dN, 16, 3) = (N+dN, 12, 4)
layout(binding=8) RWStructuredBuffer<float4> g_rotations;  // (N+dN, 2, 4)
layout(binding=9) RWStructuredBuffer<float4> g_scales_opacs;  // (N+dN, 2, 4)

#define updateAdd main

[numthreads(GROUP_SIZE, 1, 1)]
void updateAdd(
    int32_t3 thread_id : SV_DispatchThreadID,
    uniform Uniforms uniforms
) {
    const uint32_t tid = thread_id.x;
    if (tid >= uniforms.num_add)
        return;

    const uint32_t num_splats = uniforms.array_size;
    uint32_t id_dst = tid + num_splats;
    uint32_t id_src = index_map[tid];

    // copy components
    write_mean(read_mean(id_src, means), id_dst, means);
    _FOR_LOOP(i, 12) sh_coeffs[shAt(id_dst,i)] = sh_coeffs[shAt(id_src,i)];
    rotations[id_dst] = rotations[id_src];
    scales_opacs[id_dst] = scales_opacs[id_src];

    // set grad to zero
    _FOR_LOOP(i, 2) write_mean(float3(0.0), id_dst, g_means, 2, i);
    _FOR_LOOP(i, 12) g_sh_coeffs_1[shAt(id_dst,i)] = float4(0.0);
    _FOR_LOOP(i, 12) g_sh_coeffs_2[shAt(id_dst,i)] = float4(0.0);
    _FOR_LOOP(i, 2) g_rotations[id_dst*2+i] = float4(0.0);
    _FOR_LOOP(i, 2) g_scales_opacs[id_dst*2+i] = float4(0.0);
}

#endif  // MCMC_PHASE == MCMC_PHASE_UpdateAdd


