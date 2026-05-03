// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "config.slang"

static const float SH_C0 = float(0.28209479177387814);
static const float SH_C1 = float(0.4886025119029199);
static const float SH_C2[] = {
	float(1.0925484305920792),
	float(-1.0925484305920792),
	float(0.31539156525252005),
	float(-1.0925484305920792),
	float(0.5462742152960396)
};
static const float SH_C3[] = {
	float(-0.5900435899266435),
	float(2.890611442640554),
	float(-0.4570457994644658),
	float(0.3731763325901154),
	float(-0.4570457994644658),
	float(1.445305721320277),
	float(-0.5900435899266435)
};

struct SpherHarmCoeffs : IDifferentiable
{
    // Band 0
    float3 coeff0;

    // Band 1
    float3 coeff1;
    float3 coeff2;
    float3 coeff3;

    // Band2
    float3 coeff4;
    float3 coeff5;
    float3 coeff6;
    float3 coeff7;
    float3 coeff8;

    // Band3
    float3 coeff9;
    float3 coeff10;
    float3 coeff11;
    float3 coeff12;
    float3 coeff13;
    float3 coeff14;
    float3 coeff15;
};

#if SH_REORDER_SIZE > 1
  #define USE_FLOAT4_BATCH_READ 1
  // #define USE_FLOAT4_BATCH_READ 0
  #define USE_FLOAT4_BATCH_WRITE 1
#else
  #define USE_FLOAT4_BATCH_READ 0
  // #define USE_FLOAT4_BATCH_READ 1
  #define USE_FLOAT4_BATCH_WRITE 1
#endif

#if !USE_FLOAT4_BATCH_READ

#define _READ_F3(i) \
  g_sh_coeffs.coeff##i = float3(float(sh_coeffs[48*g_idx+3*i+0]), float(sh_coeffs[48*g_idx+3*i+1]), float(sh_coeffs[48*g_idx+3*i+2]))

SpherHarmCoeffs read_spherical_harmonics_coeffs(uint32_t g_idx, IArray<float> sh_coeffs, uint32_t active_sh) {

    SpherHarmCoeffs g_sh_coeffs;
    _READ_F3(0);
    
    if (active_sh > 0) {
      _READ_F3(1); _READ_F3(2); _READ_F3(3);
      
      if (active_sh > 1) {
        _READ_F3(4); _READ_F3(5); _READ_F3(6); _READ_F3(7); _READ_F3(8);

        if (active_sh > 2) {
          _READ_F3(9); _READ_F3(10); _READ_F3(11); _READ_F3(12); _READ_F3(13); _READ_F3(14); _READ_F3(15);
        }
      }
    }
    return g_sh_coeffs;
}

#else  // USE_FLOAT4_BATCH_READ


SpherHarmCoeffs read_spherical_harmonics_coeffs(uint32_t g_idx, IArray<float4> sh_coeffs, uint32_t active_sh) {
    SpherHarmCoeffs g_sh_coeffs;

  #if SH_REORDER_SIZE <= 1
    uint32_t base = 12*g_idx;
    #define _SH_COEFF(idx) float4(sh_coeffs[base+idx])
  #else
    uint32_t base = (g_idx/SH_REORDER_SIZE) * (12*SH_REORDER_SIZE);
    uint32_t offset = g_idx%SH_REORDER_SIZE;
    #define _SH_COEFF(idx) float4(sh_coeffs[base+SH_REORDER_SIZE*idx+offset])
  #endif

    float4 temp0, temp1;

    temp0 = _SH_COEFF(0);
    g_sh_coeffs.coeff0 = temp0.xyz;
    temp1 = _SH_COEFF(1);
    g_sh_coeffs.coeff1  = float3(temp0.w, temp1.xy);
    temp0 = _SH_COEFF(2);
    g_sh_coeffs.coeff2  = float3(temp1.zw, temp0.x);
    g_sh_coeffs.coeff3  = temp0.yzw;

    temp0 = _SH_COEFF(3);
    g_sh_coeffs.coeff4 = temp0.xyz;
    temp1 = _SH_COEFF(4);
    g_sh_coeffs.coeff5  = float3(temp0.w, temp1.xy);
    temp0 = _SH_COEFF(5);
    g_sh_coeffs.coeff6  = float3(temp1.zw, temp0.x);
    g_sh_coeffs.coeff7  = temp0.yzw;

    temp0 = _SH_COEFF(6);
    g_sh_coeffs.coeff8 = temp0.xyz;
    temp1 = _SH_COEFF(7);
    g_sh_coeffs.coeff9  = float3(temp0.w, temp1.xy);
    temp0 = _SH_COEFF(8);
    g_sh_coeffs.coeff10  = float3(temp1.zw, temp0.x);
    g_sh_coeffs.coeff11  = temp0.yzw;

    temp0 = _SH_COEFF(9);
    g_sh_coeffs.coeff12 = temp0.xyz;
    temp1 = _SH_COEFF(10);
    g_sh_coeffs.coeff13  = float3(temp0.w, temp1.xy);
    temp0 = _SH_COEFF(11);
    g_sh_coeffs.coeff14  = float3(temp1.zw, temp0.x);
    g_sh_coeffs.coeff15  = temp0.yzw;

    #undef _SH_COEFF

    return g_sh_coeffs;
}

#endif  // USE_FLOAT4_BATCH_READ


[ForceInline]
uint get_num_sh(uint active_sh) {
  // return active_sh == 0 ? 1 : active_sh == 1 ? 3 : active_sh == 2 ? 7 : 12;
  return (3*active_sh*(active_sh+2)+6) >> 2;
}

[ForceInline]
bool is_sh_skipped(uint32_t step, uint cid, uint num_sh) {
    return cid%12 >= num_sh;
}

[ForceInline]
float4 get_sh_coeff_4(in SpherHarmCoeffs coeffs, uint idx) {
    if (idx == 0) return float4(coeffs.coeff0.xyz, coeffs.coeff1.x);
    if (idx == 1) return float4(coeffs.coeff1.yz, coeffs.coeff2.xy);
    if (idx == 2) return float4(coeffs.coeff2.z, coeffs.coeff3.xyz);
    if (idx == 3) return float4(coeffs.coeff4.xyz, coeffs.coeff5.x);
    if (idx == 4) return float4(coeffs.coeff5.yz, coeffs.coeff6.xy);
    if (idx == 5) return float4(coeffs.coeff6.z, coeffs.coeff7.xyz);
    if (idx == 6) return float4(coeffs.coeff8.xyz, coeffs.coeff9.x);
    if (idx == 7) return float4(coeffs.coeff9.yz, coeffs.coeff10.xy);
    if (idx == 8) return float4(coeffs.coeff10.z, coeffs.coeff11.xyz);
    if (idx == 9) return float4(coeffs.coeff12.xyz, coeffs.coeff13.x);
    if (idx == 10) return float4(coeffs.coeff13.yz, coeffs.coeff14.xy);
    if (idx == 11) return float4(coeffs.coeff14.z, coeffs.coeff15.xyz);
    return float4(0);
}

#if !USE_FLOAT4_BATCH_WRITE

#define _WRITE_F3(i) \
  buffer[48*g_idx+3*i+0] = float(coeffs.coeff##i.x), \
  buffer[48*g_idx+3*i+1] = float(coeffs.coeff##i.y), \
  buffer[48*g_idx+3*i+2] = float(coeffs.coeff##i.z)

void write_spherical_harmonics_coeffs_grads(SpherHarmCoeffs coeffs, uint32_t g_idx, RWStructuredBuffer<float> buffer, uint active_sh, uint step) {

    _WRITE_F3(0);
    
    if (active_sh > 0) {
      _WRITE_F3(1); _WRITE_F3(2); _WRITE_F3(3);
      
      if (active_sh > 1) {
        _WRITE_F3(4); _WRITE_F3(5); _WRITE_F3(6); _WRITE_F3(7); _WRITE_F3(8);

        if (active_sh > 2) {
          _WRITE_F3(9); _WRITE_F3(10); _WRITE_F3(11); _WRITE_F3(12); _WRITE_F3(13); _WRITE_F3(14); _WRITE_F3(15);
        }
      }
    }
}

#else  // USE_FLOAT4_BATCH_WRITE

void write_spherical_harmonics_coeffs_grads(SpherHarmCoeffs coeffs, uint32_t g_idx, RWStructuredBuffer<float4> buffer, uint active_sh, uint step) {

    uint num_sh = get_num_sh(active_sh);
    #define PREFIX_COND(idx) if (!is_sh_skipped(step+1, idx, num_sh))

  #if SH_REORDER_SIZE <= 1
    uint32_t base = 12*g_idx;
    #define _SH_COEFF(idx) PREFIX_COND(idx) buffer[base+idx]
  #else
    uint32_t base = (g_idx/SH_REORDER_SIZE) * (12*SH_REORDER_SIZE);
    uint32_t offset = g_idx%SH_REORDER_SIZE;
    #define _SH_COEFF(idx) PREFIX_COND(idx) buffer[base+SH_REORDER_SIZE*idx+offset]
  #endif

    [ForceUnroll]
    for (uint i = 0; i < 12; i++)
        _SH_COEFF(i)  = get_sh_coeff_4(coeffs, i);

    #undef _SH_COEFF
}

#endif  // USE_FLOAT4_BATCH_WRITE


[Differentiable]
float3 compute_color_from_sh_coeffs(SpherHarmCoeffs sh, float3 dir, uint32_t active_sh) {
    float3 rgb = SH_C0 * (sh.coeff0);
    static const float _2 = float(2.0), _3 = float(3.0), _4 = float(4.0);
    if (active_sh > 0) {
        rgb = rgb - SH_C1 * dir.y * (sh.coeff1) + SH_C1 * dir.z * (sh.coeff2) - SH_C1 * dir.x * (sh.coeff3);
        if (active_sh > 1)
        {
            float xx = dir.x * dir.x, yy = dir.y * dir.y, zz = dir.z * dir.z;
            float xy = dir.x * dir.y, yz = dir.y * dir.z, xz = dir.x * dir.z;
            rgb = rgb +
                  SH_C2[0] * xy * (sh.coeff4) +
                  SH_C2[1] * yz * (sh.coeff5) +
                  SH_C2[2] * (_2 * zz - xx - yy) * (sh.coeff6) +
                  SH_C2[3] * xz * (sh.coeff7) +
                  SH_C2[4] * (xx - yy) * (sh.coeff8);

            if (active_sh > 2)
            {
                rgb = rgb +
                      SH_C3[0] * dir.y * (_3 * xx - yy) * (sh.coeff9) +
                      SH_C3[1] * xy * dir.z * (sh.coeff10) +
                      SH_C3[2] * dir.y * (_4 * zz - xx - yy) * (sh.coeff11) +
                      SH_C3[3] * dir.z * (_2 * zz - _3 * xx - _3 * yy) * (sh.coeff12) +
                      SH_C3[4] * dir.x * (_4 * zz - xx - yy) * (sh.coeff13) +
                      SH_C3[5] * dir.z * (xx - yy) * (sh.coeff14) +
                      SH_C3[6] * dir.x * (xx - _3 * yy) * (sh.coeff15);
            }
        }
    }

    rgb += float(0.5);
    rgb = max(rgb, float(0.0));

    return rgb;
}