#ifndef SUBGROUP_SIZE
#define SUBGROUP_SIZE 32
#endif

// Matches the CUDA reference forward kernel (min_alpha_threshold = 0.5/255).
// At 1/255 the per-splat extent (extend_from_opacity) shrinks ~7%, which leaves
// uncovered sub-pixel gaps at object boundaries that show as small black
// triangles against the background.
#define ALPHA_THRESHOLD (0.5f/255.f)

// Matches the CUDA viewer/training rasterizer: the non-mip path dilates by
// 0.3 px, while mip filtering uses less dilation and compensates opacity by
// the covariance determinant ratio.
#define DILATION_3DGS 0.3
#define DILATION_MIP_FILTER 0.1

#define TILE_HEIGHT 16
#define TILE_WIDTH 16
#define TILE_SIZE (TILE_HEIGHT*TILE_WIDTH)

#define RASTER_BATCH_SIZE 1024u
#define RASTER_DENSE_TILE_THRESHOLD RASTER_BATCH_SIZE

// HiGS macro-tile pipeline. Mirrors config.h; the tile count per macro tile
// must equal SUBGROUP_SIZE (ballot transpose carries one tile per lane).
// The HiGS chain rasterizes 8x8 px render tiles (the paper's preferred
// geometry; macro tile = 64x32 px) independent of the legacy 16x16 grid.
#define HIGS_MACRO_TILE_WIDTH_TILES 8u
#define HIGS_MACRO_TILE_HEIGHT_TILES 4u
#define HIGS_MACRO_TILE_SIZE_TILES (HIGS_MACRO_TILE_WIDTH_TILES * HIGS_MACRO_TILE_HEIGHT_TILES)
#define HIGS_TILE_WIDTH 8u
#define HIGS_TILE_HEIGHT 8u
#define HIGS_TILE_SIZE (HIGS_TILE_WIDTH * HIGS_TILE_HEIGHT)
// Macro-tile extent in legacy 16px-tile units (projection rects use that grid).
#define HIGS_MACRO_T16_W ((HIGS_MACRO_TILE_WIDTH_TILES * HIGS_TILE_WIDTH) / TILE_WIDTH)
#define HIGS_MACRO_T16_H ((HIGS_MACRO_TILE_HEIGHT_TILES * HIGS_TILE_HEIGHT) / TILE_HEIGHT)
#define HIGS_RASTER_WAVE_BATCHES 16384u
#define HIGS_RASTER_MAX_WAVES 16u
#define HIGS_DEPTH_WAVE_INSTANCES 4194304u
#define HIGS_DEPTH_MAX_WAVES 64u
#define HIGS_DEPTH_MAX_WAVES_FALLBACK 16u
// Drain ring buffer: per-warp shared queue of compacted splat indices,
// blended in dense batches of this size (reference Bd = 64).
#define HIGS_DRAIN_BATCH 64u

// 0: none
// 1: gsplat
// 2: all nonzero
#define TILE_CULLING_MODE 2

// 0.2 in Inria, 0.01 in GSplat
#define NEAR_CLIP 0.01f


#define EXPORT_MODE_VULKAN_FORWARD 0
#define EXPORT_MODE_VULKAN_BACKWARD 1

#define sortingKey_t uint32_t
// #define sortingKey_t int64_t

// SH ordering
// Degree 3 SH gives (3+1)^2=16 coefficients, or 16×3=48 floats
// 16×float3: [000111222...15]×N
// 12×float4: [000011112222...11]×N (default)
// Reordered 12×float4 coalesced memory access:
// [[0000...×W][1111...×W]×12]×ceil(N/W)

#define SH_REORDER_SIZE SUBGROUP_SIZE
// #define SH_REORDER_SIZE 1


// 0.99 for Inria, 0.999 for GSplat
#define ALPHA_BLEND_MAX_OPACITY 0.999f

// Matches CUDA reference (config::max_blend_color). SH evaluation can produce
// per-channel values well above 1.0; clipping caps blow-ups while leaving
// typical HDR-ish values (<= ~2) untouched.
#define MAX_BLEND_COLOR 4.0f

// Matches CUDA reference (config::max_power_threshold = 0.5 * max_stddev^2,
// max_stddev = sqrt(8)). Floors power_threshold_for_opacity so low-opacity
// splats still get a finite extent for the covariance clamp.
#define MAX_POWER_THRESHOLD 4.0f

// Matches CUDA reference (config::max_raw_scale = 20.0f). Clamps log-scale
// before activation so a misbehaving splat with raw_scale >> 20 cannot
// explode into a screen-spanning haze floater (exp(2*20) ≈ 2.4e17, the
// largest variance that stays finite in fp32 with margin). This is the
// primary haze-floater killer at the projection stage.
#define MAX_RAW_SCALE 20.0f

// subtile culling in rasterization
#define SUBTILE_HEIGHT 0
#define SUBTILE_WIDTH 0
#define SUBTILE_SIZE (SUBTILE_HEIGHT*SUBTILE_WIDTH)


// optional, not in Inria or GSplat
#define NORMALIZE_QUATS_IN_OPTIMIZER 0


static const float M_PI = 3.14159265358979;

#ifndef USE_EMULATED_INT64
#define USE_EMULATED_INT64 0
#endif

#ifndef USE_EMULATED_F32_ATOMIC
#define USE_EMULATED_F32_ATOMIC 0
#endif

#if USE_EMULATED_F32_ATOMIC
void AtomicAddFloat(RWByteAddressBuffer buffer, uint byteAddress, float valueToAdd) {
    uint oldInt = buffer.Load(byteAddress);
    uint newInt;
    uint assumed;

    do {
        assumed = oldInt;
        float fAssumed = asfloat(assumed);
        float fNew = fAssumed + valueToAdd;
        newInt = asuint(fNew);

        uint resultInt;
        buffer.InterlockedCompareExchange(byteAddress, assumed, newInt, resultInt);
        oldInt = resultInt;
    } while (assumed != oldInt);
}
#endif
