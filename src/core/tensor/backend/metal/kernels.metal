// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later

// Tensor kernels of the Metal backend, ported from the Vulkan Slang shaders so
// both backends compute identical results. The host compiles this source at
// runtime with safe math and precise functions and prepends the LFS_OP_*,
// LFS_DT_*, LFS_REDUCE_* and LFS_FILTER_* ids generated from the C++ enums.
// Operands are bound whole and addressed by byte offsets, so any element
// offset is legal.

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#pragma METAL fp contract(off)
using namespace metal;

constant uint kOp [[function_constant(0)]];
constant uint kInputDType [[function_constant(1)]];
constant uint kOutputDType [[function_constant(2)]];
constant uint kArity [[function_constant(3)]];
constant uint kVectorized [[function_constant(4)]];
constant uint kElementSize [[function_constant(5)]];
constant uint kReduce [[function_constant(6)]];
constant uint kScatter [[function_constant(7)]];
constant uint kTransposeB [[function_constant(14)]];
constant uint kBiasRelu [[function_constant(15)]];
// Matrix-unit GEMMs compute 64x64 tiles instead of 32x32 ones.
constant uint kWideTile [[function_constant(27)]];

constant uint kReduceThreads = 256;

// ---------------------------------------------------------------------------
// IEEE helpers shared with the Vulkan shaders.

static float ieee_maximum(float lhs, float rhs) {
    const uint lhs_bits = as_type<uint>(lhs);
    const uint rhs_bits = as_type<uint>(rhs);
    if ((lhs_bits & 0x7fffffffu) > 0x7f800000u)
        return lhs;
    if ((rhs_bits & 0x7fffffffu) > 0x7f800000u)
        return rhs;
    if ((lhs_bits & 0x7fffffffu) == 0u && (rhs_bits & 0x7fffffffu) == 0u)
        return as_type<float>((lhs_bits & rhs_bits) & 0x80000000u);
    return lhs < rhs ? rhs : lhs;
}

static float ieee_minimum(float lhs, float rhs) {
    const uint lhs_bits = as_type<uint>(lhs);
    const uint rhs_bits = as_type<uint>(rhs);
    if ((lhs_bits & 0x7fffffffu) > 0x7f800000u)
        return lhs;
    if ((rhs_bits & 0x7fffffffu) > 0x7f800000u)
        return rhs;
    if ((lhs_bits & 0x7fffffffu) == 0u && (rhs_bits & 0x7fffffffu) == 0u)
        return as_type<float>((lhs_bits | rhs_bits) & 0x80000000u);
    return rhs < lhs ? rhs : lhs;
}

static float ieee_round(float value) {
    if (!isfinite(value))
        return value;
    const float lower = floor(value);
    const float fraction = value - lower;
    const float rounded = fraction < 0.5f ? lower
                          : fraction > 0.5f ? lower + 1.0f
                                            : (fmod(lower, 2.0f) == 0.0f ? lower : lower + 1.0f);
    if (rounded == 0.0f && (as_type<uint>(value) & 0x80000000u) != 0u)
        return as_type<float>(0x80000000u);
    return rounded;
}

static float accurate_log1p(float value) {
    if (value == 0.0f)
        return value;
    if (value <= -1.0f || !isfinite(value))
        return log(1.0f + value);
    const float sum = 1.0f + value;
    const int exponent_bits = (as_type<int>(sum) - 1061158912) & -8388608;
    const float reduced_value = as_type<float>(as_type<int>(value) - exponent_bits);
    const float scale = as_type<float>(1082130432 - exponent_bits);
    const float x = reduced_value + fma(0.25f, scale, -1.0f);
    const float exponent = float(exponent_bits) * 1.1920928955078125e-7f;
    float polynomial = fma(-0.04534861445426941f, x, 0.10546888411045074f);
    polynomial = fma(polynomial, x, -0.13229703903198242f);
    polynomial = fma(polynomial, x, 0.14491446316242218f);
    polynomial = fma(polynomial, x, -0.16641564667224884f);
    polynomial = fma(polynomial, x, 0.199888676404953f);
    polynomial = fma(polynomial, x, -0.2500019669532776f);
    polynomial = fma(polynomial, x, 0.33333510160446167f);
    polynomial = fma(polynomial, x, -0.5f);
    const float reduced_log = fma(polynomial * x, x, x);
    return fma(exponent, 0.6931471824645996f, reduced_log);
}

static float accurate_asin(float value) {
    const float bounded = clamp(value, -1.0f, 1.0f);
    return atan2(bounded, sqrt(max(0.0f, 1.0f - bounded * bounded)));
}

static float accurate_acos(float value) {
    const float bounded = clamp(value, -1.0f, 1.0f);
    return atan2(sqrt(max(0.0f, 1.0f - bounded * bounded)), bounded);
}

// Remainders have the dividend's sign, including INT_MIN operands.
static int signed_remainder(int lhs, int rhs) {
    const uint a = lhs < 0 ? 0u - uint(lhs) : uint(lhs);
    const uint b = rhs < 0 ? 0u - uint(rhs) : uint(rhs);
    const uint remainder = b != 0u ? a % b : 0u;
    return int(lhs < 0 ? 0u - remainder : remainder);
}

// Wrap-around integer powers; negative exponents follow integer division.
static int int_pow(int base, int exponent) {
    if (exponent < 0)
        return base == 1 ? 1 : base == -1 ? ((exponent & 1) != 0 ? -1 : 1) : 0;
    int result = 1;
    int factor = base;
    for (int remaining = exponent; remaining != 0; remaining >>= 1) {
        if ((remaining & 1) != 0)
            result *= factor;
        factor *= factor;
    }
    return result;
}

static long int64_pow(long base, long exponent) {
    if (exponent < 0)
        return base == 1 ? 1 : base == -1 ? ((exponent & 1) != 0 ? -1 : 1) : 0;
    long result = 1;
    long factor = base;
    for (long remaining = exponent; remaining != 0; remaining >>= 1) {
        if ((remaining & 1) != 0)
            result *= factor;
        factor *= factor;
    }
    return result;
}

// ---------------------------------------------------------------------------
// Pointwise ops, keyed by the kOp function constant.

static float float_binary(float lhs, float rhs) {
    if (kOp == LFS_OP_AddScalar || kOp == LFS_OP_AddTensor) return lhs + rhs;
    if (kOp == LFS_OP_SubScalar || kOp == LFS_OP_SubTensor) return lhs - rhs;
    if (kOp == LFS_OP_MulScalar || kOp == LFS_OP_MulTensor) return lhs * rhs;
    if (kOp == LFS_OP_DivScalar || kOp == LFS_OP_DivTensor) return lhs / rhs;
    if (kOp == LFS_OP_PowScalar || kOp == LFS_OP_PowTensor) return rhs == 2.0f ? lhs * lhs : pow(lhs, rhs);
    if (kOp == LFS_OP_ModScalar || kOp == LFS_OP_ModTensor) return fmod(lhs, rhs);
    if (kOp == LFS_OP_MaximumTensor || kOp == LFS_OP_MaximumScalar) return ieee_maximum(lhs, rhs);
    if (kOp == LFS_OP_MinimumTensor || kOp == LFS_OP_MinimumScalar) return ieee_minimum(lhs, rhs);
    return lhs;
}

static bool float_predicate(float lhs, float rhs) {
    if (kOp == LFS_OP_EqualTensor || kOp == LFS_OP_EqualScalar) return lhs == rhs;
    if (kOp == LFS_OP_NotEqualTensor || kOp == LFS_OP_NotEqualScalar) return lhs != rhs;
    if (kOp == LFS_OP_LessTensor || kOp == LFS_OP_LessScalar) return lhs < rhs;
    if (kOp == LFS_OP_LessEqualTensor || kOp == LFS_OP_LessEqualScalar) return lhs <= rhs;
    if (kOp == LFS_OP_GreaterTensor || kOp == LFS_OP_GreaterScalar) return lhs > rhs;
    if (kOp == LFS_OP_GreaterEqualTensor || kOp == LFS_OP_GreaterEqualScalar) return lhs >= rhs;
    if (kOp == LFS_OP_LogicalAndTensor) return lhs != 0.0f && rhs != 0.0f;
    if (kOp == LFS_OP_LogicalOrTensor) return lhs != 0.0f || rhs != 0.0f;
    if (kOp == LFS_OP_LogicalXorTensor) return (lhs != 0.0f) != (rhs != 0.0f);
    if (kOp == LFS_OP_IsNan) return isnan(lhs);
    if (kOp == LFS_OP_IsInf) return isinf(lhs);
    if (kOp == LFS_OP_IsFinite) return isfinite(lhs);
    if (kOp == LFS_OP_LogicalNot) return lhs == 0.0f;
    return false;
}

static bool is_scalar_binary_op() {
    return kOp <= LFS_OP_ModScalar || (kOp >= LFS_OP_MaximumScalar && kOp <= LFS_OP_GreaterEqualScalar);
}

static float float_unary(float value, float scalar, bool scalar_on_right) {
    if (is_scalar_binary_op())
        return scalar_on_right ? float_binary(value, scalar) : float_binary(scalar, value);
    if (kOp == LFS_OP_Abs) return abs(value);
    if (kOp == LFS_OP_Neg) return -value;
    if (kOp == LFS_OP_Exp) return exp(value);
    if (kOp == LFS_OP_Log) return log(value);
    if (kOp == LFS_OP_Sqrt) return sqrt(value);
    if (kOp == LFS_OP_Sigmoid) return 1.0f / (1.0f + exp(-value));
    if (kOp == LFS_OP_Relu) return isnan(value) ? value : ieee_maximum(value, 0.0f);
    if (kOp == LFS_OP_Square) return value * value;
    if (kOp == LFS_OP_Tanh) return tanh(value);
    if (kOp == LFS_OP_Rsqrt) return rsqrt(value);
    if (kOp == LFS_OP_Sign) return float(int(value > 0.0f) - int(value < 0.0f));
    if (kOp == LFS_OP_Reciprocal) return 1.0f / value;
    if (kOp == LFS_OP_Floor) return floor(value);
    if (kOp == LFS_OP_Ceil) return ceil(value);
    if (kOp == LFS_OP_Round) return ieee_round(value);
    if (kOp == LFS_OP_Exp2) return exp2(value);
    if (kOp == LFS_OP_Log2) return log2(value);
    if (kOp == LFS_OP_Log10) return log10(value);
    if (kOp == LFS_OP_Log1p) return accurate_log1p(value);
    if (kOp == LFS_OP_Sin) return sin(value);
    if (kOp == LFS_OP_Cos) return cos(value);
    if (kOp == LFS_OP_Tan) return tan(value);
    if (kOp == LFS_OP_Asin) return accurate_asin(value);
    if (kOp == LFS_OP_Acos) return accurate_acos(value);
    if (kOp == LFS_OP_Atan) return atan(value);
    if (kOp == LFS_OP_Sinh) return sinh(value);
    if (kOp == LFS_OP_Cosh) return cosh(value);
    if (kOp == LFS_OP_Gelu) {
        const float inner = sqrt(2.0f / 3.14159265358979323846f) * (value + 0.044715f * value * value * value);
        return 0.5f * value * (1.0f + tanh(inner));
    }
    if (kOp == LFS_OP_Swish) return value / (1.0f + exp(-value));
    if (kOp == LFS_OP_Trunc) return trunc(value);
    if (kOp == LFS_OP_ExpThenMulScalar) return exp(value) * scalar;
    if (kOp == LFS_OP_MulScalarThenAbs) return abs(value * scalar);
    if (kOp == LFS_OP_MulScalarThenRelu) return ieee_maximum(value * scalar, 0.0f);
    return value;
}

static int int_binary(int lhs, int rhs) {
    if (kOp == LFS_OP_AddScalar || kOp == LFS_OP_AddTensor) return lhs + rhs;
    if (kOp == LFS_OP_SubScalar || kOp == LFS_OP_SubTensor) return lhs - rhs;
    if (kOp == LFS_OP_MulScalar || kOp == LFS_OP_MulTensor) return lhs * rhs;
    if (kOp == LFS_OP_DivScalar || kOp == LFS_OP_DivTensor) return rhs == 0 ? 0 : rhs == -1 ? -lhs : lhs / rhs;
    if (kOp == LFS_OP_PowScalar || kOp == LFS_OP_PowTensor) return int_pow(lhs, rhs);
    if (kOp == LFS_OP_ModScalar || kOp == LFS_OP_ModTensor) return signed_remainder(lhs, rhs);
    if (kOp == LFS_OP_MaximumTensor || kOp == LFS_OP_MaximumScalar) return max(lhs, rhs);
    if (kOp == LFS_OP_MinimumTensor || kOp == LFS_OP_MinimumScalar) return min(lhs, rhs);
    return lhs;
}

static bool int_predicate(int lhs, int rhs) {
    if (kOp == LFS_OP_EqualTensor || kOp == LFS_OP_EqualScalar) return lhs == rhs;
    if (kOp == LFS_OP_NotEqualTensor || kOp == LFS_OP_NotEqualScalar) return lhs != rhs;
    if (kOp == LFS_OP_LessTensor || kOp == LFS_OP_LessScalar) return lhs < rhs;
    if (kOp == LFS_OP_LessEqualTensor || kOp == LFS_OP_LessEqualScalar) return lhs <= rhs;
    if (kOp == LFS_OP_GreaterTensor || kOp == LFS_OP_GreaterScalar) return lhs > rhs;
    if (kOp == LFS_OP_GreaterEqualTensor || kOp == LFS_OP_GreaterEqualScalar) return lhs >= rhs;
    if (kOp == LFS_OP_LogicalAndTensor) return lhs != 0 && rhs != 0;
    if (kOp == LFS_OP_LogicalOrTensor) return lhs != 0 || rhs != 0;
    if (kOp == LFS_OP_LogicalXorTensor) return (lhs != 0) != (rhs != 0);
    if (kOp == LFS_OP_IsNan || kOp == LFS_OP_IsInf) return false;
    if (kOp == LFS_OP_IsFinite) return true;
    if (kOp == LFS_OP_LogicalNot) return lhs == 0;
    return false;
}

static int int_unary(int value, int scalar, bool scalar_on_right) {
    if (is_scalar_binary_op())
        return scalar_on_right ? int_binary(value, scalar) : int_binary(scalar, value);
    if (kOp == LFS_OP_Abs) return abs(value);
    if (kOp == LFS_OP_Neg) return -value;
    if (kOp == LFS_OP_Relu) return max(value, 0);
    if (kOp == LFS_OP_Square) return value * value;
    if (kOp == LFS_OP_Sign) return int(value > 0) - int(value < 0);
    if (kOp == LFS_OP_Floor || kOp == LFS_OP_Ceil || kOp == LFS_OP_Round || kOp == LFS_OP_Trunc) return value;
    return int(float_unary(float(value), float(scalar), scalar_on_right));
}

static long int64_binary(long lhs, long rhs) {
    if (kOp == LFS_OP_AddTensor) return lhs + rhs;
    if (kOp == LFS_OP_SubTensor) return lhs - rhs;
    if (kOp == LFS_OP_MulTensor) return lhs * rhs;
    if (kOp == LFS_OP_DivTensor) return rhs == 0 ? 0 : rhs == -1 ? -lhs : lhs / rhs;
    if (kOp == LFS_OP_PowTensor) return int64_pow(lhs, rhs);
    if (kOp == LFS_OP_ModTensor) return rhs == 0 || rhs == -1 ? 0 : lhs % rhs;
    if (kOp == LFS_OP_MaximumTensor) return lhs < rhs ? rhs : lhs;
    if (kOp == LFS_OP_MinimumTensor) return rhs < lhs ? rhs : lhs;
    return lhs;
}

static bool int64_predicate(long lhs, long rhs) {
    if (kOp == LFS_OP_EqualTensor) return lhs == rhs;
    if (kOp == LFS_OP_NotEqualTensor) return lhs != rhs;
    if (kOp == LFS_OP_LessTensor) return lhs < rhs;
    if (kOp == LFS_OP_LessEqualTensor) return lhs <= rhs;
    if (kOp == LFS_OP_GreaterTensor) return lhs > rhs;
    if (kOp == LFS_OP_GreaterEqualTensor) return lhs >= rhs;
    if (kOp == LFS_OP_LogicalAndTensor) return lhs != 0 && rhs != 0;
    if (kOp == LFS_OP_LogicalOrTensor) return lhs != 0 || rhs != 0;
    if (kOp == LFS_OP_LogicalXorTensor) return (lhs != 0) != (rhs != 0);
    return false;
}

static bool uint_predicate(uint lhs, uint rhs) {
    if (kOp == LFS_OP_EqualTensor) return lhs == rhs;
    if (kOp == LFS_OP_NotEqualTensor) return lhs != rhs;
    if (kOp == LFS_OP_LessTensor) return lhs < rhs;
    if (kOp == LFS_OP_LessEqualTensor) return lhs <= rhs;
    if (kOp == LFS_OP_GreaterTensor) return lhs > rhs;
    if (kOp == LFS_OP_GreaterEqualTensor) return lhs >= rhs;
    if (kOp == LFS_OP_LogicalAndTensor) return lhs != 0u && rhs != 0u;
    if (kOp == LFS_OP_LogicalOrTensor) return lhs != 0u || rhs != 0u;
    return (lhs != 0u) != (rhs != 0u);
}

static uint uint_binary(uint lhs, uint rhs) {
    if (kOp == LFS_OP_AddTensor) return lhs + rhs;
    if (kOp == LFS_OP_SubTensor) return lhs - rhs;
    if (kOp == LFS_OP_MulTensor) return lhs * rhs;
    if (kOp == LFS_OP_DivTensor) return rhs == 0u ? 0u : lhs / rhs;
    if (kOp == LFS_OP_ModTensor) return rhs == 0u ? 0u : lhs % rhs;
    if (kOp == LFS_OP_MaximumTensor) return max(lhs, rhs);
    if (kOp == LFS_OP_MinimumTensor) return min(lhs, rhs);
    if (kOp == LFS_OP_PowTensor) {
        uint result = 1u;
        uint factor = lhs;
        for (uint exponent = rhs; exponent != 0u; exponent >>= 1u) {
            if ((exponent & 1u) != 0u)
                result *= factor;
            factor *= factor;
        }
        return result;
    }
    return lhs;
}

static float load_float(device const uchar* base, uint index) {
    if (kInputDType == LFS_DT_Float16)
        return float(((device const half*)base)[index]);
    return ((device const float*)base)[index];
}

static void store_float(device uchar* base, uint index, float value) {
    if (kOutputDType == LFS_DT_Float16)
        ((device half*)base)[index] = half(value);
    else
        ((device float*)base)[index] = value;
}

struct PointwiseParams {
    ulong lhs_offset;
    ulong rhs_offset;
    ulong output_offset;
    long scalar_int64;
    float scalar_float;
    uint count;
    uint scalar_kind;
    uint flags;
};

static uchar evaluate_byte(device const uchar* lhs, device const uchar* rhs,
                           constant PointwiseParams& params, uint lhs_index, uint rhs_index) {
    if (kInputDType == LFS_DT_Float32 || kInputDType == LFS_DT_Float16) {
        const float a = load_float(lhs, lhs_index);
        const float b = kArity == 2 ? load_float(rhs, rhs_index) : params.scalar_float;
        return float_predicate(a, b) ? 1 : 0;
    }
    if (kInputDType == LFS_DT_Int64) {
        const long a = ((device const long*)lhs)[lhs_index];
        const long b = ((device const long*)rhs)[rhs_index];
        return int64_predicate(a, b) ? 1 : 0;
    }
    if (kInputDType == LFS_DT_UInt32)
        return uint_predicate(((device const uint*)lhs)[lhs_index], ((device const uint*)rhs)[rhs_index]) ? 1 : 0;
    const int a = kInputDType == LFS_DT_Int32 ? ((device const int*)lhs)[lhs_index] : int(lhs[lhs_index]);
    const int b = kArity != 2 ? int(params.scalar_int64)
                  : kInputDType == LFS_DT_Int32 ? ((device const int*)rhs)[rhs_index]
                                                : int(rhs[rhs_index]);
    return kOutputDType == LFS_DT_Bool ? (int_predicate(a, b) ? 1 : 0) : uchar(uint(int_binary(a, b)) & 255u);
}

// output[index] = op(lhs[lhs_index], rhs[rhs_index]), or of the scalar when
// kArity is 1.
static void evaluate(device const uchar* lhs, device const uchar* rhs, device uchar* output,
                     constant PointwiseParams& params, uint lhs_index, uint rhs_index, uint index) {
    if (kOutputDType == LFS_DT_UInt8 || kOutputDType == LFS_DT_Bool) {
        output[index] = evaluate_byte(lhs, rhs, params, lhs_index, rhs_index);
        return;
    }
    const bool scalar_on_right = (params.flags & 1u) != 0u;
    if (kInputDType == LFS_DT_Float32 || kInputDType == LFS_DT_Float16) {
        const float a = load_float(lhs, lhs_index);
        const float b = kArity == 2 ? load_float(rhs, rhs_index) : params.scalar_float;
        store_float(output, index, kArity == 2 ? float_binary(a, b) : float_unary(a, b, scalar_on_right));
    } else if (kInputDType == LFS_DT_Int32) {
        const int a = ((device const int*)lhs)[lhs_index];
        const int b = kArity == 2 ? ((device const int*)rhs)[rhs_index] : int(params.scalar_int64);
        ((device int*)output)[index] = kArity == 2 ? int_binary(a, b) : int_unary(a, b, scalar_on_right);
    } else if (kInputDType == LFS_DT_Int64) {
        ((device long*)output)[index] =
            int64_binary(((device const long*)lhs)[lhs_index], ((device const long*)rhs)[rhs_index]);
    } else if (kInputDType == LFS_DT_UInt32) {
        ((device uint*)output)[index] =
            uint_binary(((device const uint*)lhs)[lhs_index], ((device const uint*)rhs)[rhs_index]);
    }
}

kernel void pointwise(device const uchar* lhs_buffer [[buffer(0)]],
                      device const uchar* rhs_buffer [[buffer(1)]],
                      device uchar* output_buffer [[buffer(2)]],
                      constant PointwiseParams& params [[buffer(3)]],
                      uint index [[thread_position_in_grid]]) {
    device const uchar* lhs = lhs_buffer + params.lhs_offset;
    device const uchar* rhs = rhs_buffer + params.rhs_offset;
    device uchar* output = output_buffer + params.output_offset;
    if (kVectorized != 0) {
        if (index * 4u >= params.count)
            return;
        const float4 a = ((device const float4*)lhs)[index];
        const float4 b = kArity == 2 ? ((device const float4*)rhs)[index] : float4(params.scalar_float);
        if (kOutputDType == LFS_DT_Float32) {
            float4 result;
            for (uint lane = 0; lane < 4; ++lane)
                result[lane] = kArity == 2 ? float_binary(a[lane], b[lane]) : float_unary(a[lane], b[lane], true);
            ((device float4*)output)[index] = result;
        } else {
            uchar4 result;
            for (uint lane = 0; lane < 4; ++lane)
                result[lane] = float_predicate(a[lane], b[lane]) ? 1 : 0;
            ((device uchar4*)output)[index] = result;
        }
        return;
    }
    if (index < params.count)
        evaluate(lhs, rhs, output, params, index, index, index);
}

// Clamps Float32 (NaN passes through) or Int32 elements to [minimum, maximum].
struct ClampParams {
    ulong input_offset;
    ulong output_offset;
    float float_minimum;
    float float_maximum;
    int int_minimum;
    int int_maximum;
    uint count;
    uint padding;
};

kernel void clamp_values(device const uchar* input_buffer [[buffer(0)]],
                         device uchar* output_buffer [[buffer(1)]],
                         constant ClampParams& params [[buffer(2)]],
                         uint index [[thread_position_in_grid]]) {
    if (index >= params.count)
        return;
    if (kInputDType == LFS_DT_Float32) {
        const float value = ((device const float*)(input_buffer + params.input_offset))[index];
        ((device float*)(output_buffer + params.output_offset))[index] =
            isnan(value) ? value : min(max(value, params.float_minimum), params.float_maximum);
    } else {
        const int value = ((device const int*)(input_buffer + params.input_offset))[index];
        ((device int*)(output_buffer + params.output_offset))[index] =
            min(max(value, params.int_minimum), params.int_maximum);
    }
}

// ---------------------------------------------------------------------------
// Lazily fused Float32 chains, specialized per chain shape: the op kinds (5
// bits each) are function constants, so the chain compiles to straight-line
// code. Tensor operands are passed by address and read at the element's index;
// with every operand 16-byte aligned, each thread handles four elements.

constant uint kChainLength [[function_constant(8)]];
constant uint kChainKinds0 [[function_constant(9)]];
constant uint kChainKinds1 [[function_constant(10)]];
constant uint kChainKinds2 [[function_constant(11)]];

static uint chain_kind(uint op) {
    const uint word = op < 6 ? kChainKinds0 : op < 12 ? kChainKinds1 : kChainKinds2;
    return (word >> (5 * (op % 6))) & 31u;
}

struct ChainOp {
    float scalar;
    uint padding;
    device const float* rhs;
};

struct ChainParams {
    ulong input_offset;
    ulong output_offset;
    uint count;
    uint padding;
    ChainOp ops[16];
};

static float chain_unary(float value, uint kind) {
    switch (kind) {
    case 10: return abs(value);
    case 11: return -value;
    case 12: return exp(value);
    case 13: return log(value);
    case 14: return sqrt(value);
    case 15: return 1.0f / (1.0f + exp(-value));
    case 16: return isnan(value) ? value : ieee_maximum(value, 0.0f);
    case 17: return value * value;
    case 18: return tanh(value);
    case 19: return rsqrt(value);
    case 20: return float(int(value > 0.0f) - int(value < 0.0f));
    case 21: return 1.0f / value;
    case 22: return floor(value);
    case 23: return ceil(value);
    case 24: return ieee_round(value);
    default: return value;
    }
}

static float chain_step(float value, uint kind, float scalar, float rhs) {
    if (kind <= 3u)
        return kind == 0u ? value + scalar : kind == 1u ? value - scalar : kind == 2u ? value * scalar : value / scalar;
    if (kind <= 7u)
        return kind == 4u ? value + rhs : kind == 5u ? value - rhs : kind == 6u ? value * rhs : value / rhs;
    return chain_unary(value, kind);
}

static bool chain_reads_tensor(uint kind) {
    return kind >= 4u && kind <= 7u;
}

kernel void pointwise_chain(device const uchar* input_buffer [[buffer(0)]],
                            device uchar* output_buffer [[buffer(1)]],
                            constant ChainParams& params [[buffer(2)]],
                            uint index [[thread_position_in_grid]]) {
    device const float* input = (device const float*)(input_buffer + params.input_offset);
    device float* output = (device float*)(output_buffer + params.output_offset);
    const uint width = kVectorized != 0 ? 4u : 1u;
    const uint first = index * width;
    if (first >= params.count)
        return;
    if (kVectorized != 0 && first + 4u <= params.count) {
        float4 value = ((device const float4*)input)[index];
        for (uint op = 0; op < kChainLength; ++op) {
            const uint kind = chain_kind(op);
            const float4 rhs = chain_reads_tensor(kind) ? ((device const float4*)params.ops[op].rhs)[index] : float4(0.0f);
            for (uint lane = 0; lane < 4; ++lane)
                value[lane] = chain_step(value[lane], kind, params.ops[op].scalar, rhs[lane]);
        }
        ((device float4*)output)[index] = value;
        return;
    }
    for (uint element = first; element < min(first + width, params.count); ++element) {
        float value = input[element];
        for (uint op = 0; op < kChainLength; ++op) {
            const uint kind = chain_kind(op);
            const float rhs = chain_reads_tensor(kind) ? params.ops[op].rhs[element] : 0.0f;
            value = chain_step(value, kind, params.ops[op].scalar, rhs);
        }
        output[element] = value;
    }
}

// ---------------------------------------------------------------------------
// Dtype conversion, following the torch casts of convert.slang.

struct ConvertParams {
    ulong input_offset;
    ulong output_offset;
    uint count;
    uint padding;
};

static float load_as_float(device const uchar* input, uint index) {
    if (kInputDType == LFS_DT_Float32) return ((device const float*)input)[index];
    if (kInputDType == LFS_DT_Float16) return float(((device const half*)input)[index]);
    if (kInputDType == LFS_DT_Int32) return float(((device const int*)input)[index]);
    if (kInputDType == LFS_DT_Int64) return float(((device const long*)input)[index]);
    if (kInputDType == LFS_DT_UInt32) return float(((device const uint*)input)[index]);
    return float(input[index]);
}

static long load_as_int64(device const uchar* input, uint index) {
    if (kInputDType == LFS_DT_Int32) return long(((device const int*)input)[index]);
    if (kInputDType == LFS_DT_Int64) return ((device const long*)input)[index];
    if (kInputDType == LFS_DT_UInt32) return long(((device const uint*)input)[index]);
    if (kInputDType == LFS_DT_UInt8 || kInputDType == LFS_DT_Bool) return long(input[index]);
    return long(load_as_float(input, index));
}

static uint torch_uint8_cast(float value) {
    if (!isfinite(value))
        return 0u;
    float wrapped = fmod(trunc(value), 256.0f);
    if (wrapped < 0.0f)
        wrapped += 256.0f;
    return uint(wrapped);
}

kernel void convert(device const uchar* input_buffer [[buffer(0)]],
                    device uchar* output_buffer [[buffer(1)]],
                    constant ConvertParams& params [[buffer(2)]],
                    uint index [[thread_position_in_grid]]) {
    if (index >= params.count)
        return;
    device const uchar* input = input_buffer + params.input_offset;
    device uchar* output = output_buffer + params.output_offset;
    if (kOutputDType == LFS_DT_UInt8 || kOutputDType == LFS_DT_Bool) {
        uint value;
        if (kInputDType == kOutputDType)
            value = kInputDType == LFS_DT_Bool ? (input[index] != 0 ? 1u : 0u) : uint(input[index]);
        else if (kOutputDType == LFS_DT_UInt8)
            value = kInputDType == LFS_DT_Float32 || kInputDType == LFS_DT_Float16
                        ? torch_uint8_cast(load_as_float(input, index))
                        : uint(load_as_int64(input, index)) & 255u;
        else
            value = load_as_float(input, index) != 0.0f ? 1u : 0u;
        output[index] = uchar(value);
        return;
    }
    if (kOutputDType == LFS_DT_Float32)
        ((device float*)output)[index] = load_as_float(input, index);
    else if (kOutputDType == LFS_DT_Float16)
        ((device half*)output)[index] = kInputDType == LFS_DT_Float16 ? ((device const half*)input)[index]
                                                                     : half(load_as_float(input, index));
    else if (kOutputDType == LFS_DT_Int32)
        ((device int*)output)[index] = int(load_as_int64(input, index));
    else if (kOutputDType == LFS_DT_Int64)
        ((device long*)output)[index] = load_as_int64(input, index);
    else
        ((device uint*)output)[index] = uint(load_as_int64(input, index));
}

// ---------------------------------------------------------------------------
// Fill, arange and byte copies. kElementSize is 1, 2, 4, 8 or 16 bytes; the
// 16-byte form replicates the low 32 bits of the pattern. Fill with kOp 1
// writes a strided view, whose elements the index enumerates row-major.

struct FillParams {
    ulong output_offset;
    ulong pattern;
    ulong count;
    uint dims[8];
    uint strides[8];
    uint rank;
    uint padding;
};

kernel void fill(device uchar* output_buffer [[buffer(0)]],
                 constant FillParams& params [[buffer(1)]],
                 uint thread_index [[thread_position_in_grid]]) {
    if (thread_index >= params.count)
        return;
    device uchar* output = output_buffer + params.output_offset;
    ulong index = thread_index;
    if (kOp == 1) {
        ulong remaining = index;
        index = 0;
        for (int axis = int(params.rank) - 1; axis >= 0; --axis) {
            index += ulong(remaining % params.dims[axis]) * params.strides[axis];
            remaining /= params.dims[axis];
        }
    }
    if (kElementSize == 16)
        ((device uint4*)output)[index] = uint4(uint(params.pattern));
    else if (kElementSize == 8)
        ((device ulong*)output)[index] = params.pattern;
    else if (kElementSize == 4)
        ((device uint*)output)[index] = uint(params.pattern);
    else if (kElementSize == 2)
        ((device ushort*)output)[index] = ushort(params.pattern);
    else
        output[index] = uchar(params.pattern);
}

struct ArangeParams {
    ulong output_offset;
    float start;
    float step;
    uint count;
    uint padding;
};

kernel void arange(device uchar* output_buffer [[buffer(0)]],
                   constant ArangeParams& params [[buffer(1)]],
                   uint index [[thread_position_in_grid]]) {
    if (index >= params.count)
        return;
    const float value = params.start + float(index) * params.step;
    if (kOutputDType == LFS_DT_Int32)
        ((device int*)(output_buffer + params.output_offset))[index] = int(value);
    else
        ((device float*)(output_buffer + params.output_offset))[index] = value;
}

struct CopyParams {
    ulong source_offset;
    ulong destination_offset;
    ulong count;
};

kernel void copy_bytes(device const uchar* source_buffer [[buffer(0)]],
                       device uchar* destination_buffer [[buffer(1)]],
                       constant CopyParams& params [[buffer(2)]],
                       uint index [[thread_position_in_grid]]) {
    if (index >= params.count)
        return;
    device const uchar* source = source_buffer + params.source_offset;
    device uchar* destination = destination_buffer + params.destination_offset;
    if (kElementSize == 16)
        ((device uint4*)destination)[index] = ((device const uint4*)source)[index];
    else if (kElementSize == 4)
        ((device uint*)destination)[index] = ((device const uint*)source)[index];
    else
        destination[index] = source[index];
}

// ---------------------------------------------------------------------------
// Strided gather and scatter over up to eight dimensions, and broadcast
// selection. Elements move as kElementSize bytes, up to 16-byte chunks of
// several elements; the one converting form is the Int32 to Float32 scatter.

static void copy_element(device const uchar* source, ulong source_index,
                         device uchar* destination, ulong destination_index) {
    if (kInputDType != kOutputDType)
        ((device float*)destination)[destination_index] = float(((device const int*)source)[source_index]);
    else if (kElementSize == 16)
        ((device uint4*)destination)[destination_index] = ((device const uint4*)source)[source_index];
    else if (kElementSize == 8)
        ((device ulong*)destination)[destination_index] = ((device const ulong*)source)[source_index];
    else if (kElementSize == 4)
        ((device uint*)destination)[destination_index] = ((device const uint*)source)[source_index];
    else if (kElementSize == 2)
        ((device ushort*)destination)[destination_index] = ((device const ushort*)source)[source_index];
    else
        destination[destination_index] = source[source_index];
}

struct StridedParams {
    ulong input_offset;
    ulong output_offset;
    uint dims[8];
    uint strides[8];
    uint rank;
    uint count;
};

kernel void strided_copy(device const uchar* input_buffer [[buffer(0)]],
                         device uchar* output_buffer [[buffer(1)]],
                         constant StridedParams& params [[buffer(2)]],
                         uint index [[thread_position_in_grid]]) {
    if (index >= params.count)
        return;
    ulong strided = 0;
    uint remaining = index;
    for (int axis = int(params.rank) - 1; axis >= 0; --axis) {
        strided += ulong(remaining % params.dims[axis]) * params.strides[axis];
        remaining /= params.dims[axis];
    }
    copy_element(input_buffer + params.input_offset, kScatter != 0 ? ulong(index) : strided,
                 output_buffer + params.output_offset, kScatter != 0 ? strided : ulong(index));
}

// Output axes of a broadcast launch, outermost first. The host merges
// adjacent axes that every operand broadcasts alike; component i of a stride
// serves operand i and is 0 on the axes that operand broadcasts along.
struct BroadcastAxes {
    uint dims[8];
    uint4 strides[8];
    uint rank;
    uint padding[3];
};

static uint4 broadcast_indices(uint index, constant BroadcastAxes& axes) {
    uint4 indices = 0;
    for (int axis = int(axes.rank) - 1; axis > 0; --axis) {
        const uint quotient = index / axes.dims[axis];
        indices += (index - quotient * axes.dims[axis]) * axes.strides[axis];
        index = quotient;
    }
    return indices + index * axes.strides[0];
}

struct WhereParams {
    ulong condition_offset;
    ulong x_offset;
    ulong y_offset;
    ulong output_offset;
    uint count;
    uint padding[3];
    BroadcastAxes axes;
};

kernel void where_select(device const uchar* condition [[buffer(0)]],
                         device const uchar* x [[buffer(1)]],
                         device const uchar* y [[buffer(2)]],
                         device uchar* output [[buffer(3)]],
                         constant WhereParams& params [[buffer(4)]],
                         uint index [[thread_position_in_grid]]) {
    if (index >= params.count)
        return;
    const uint4 source = broadcast_indices(index, params.axes);
    const bool selected = (condition + params.condition_offset)[source.x] != 0;
    copy_element(selected ? x + params.x_offset : y + params.y_offset, selected ? source.y : source.z,
                 output + params.output_offset, index);
}

// Binary pointwise ops over broadcast operands.
struct BroadcastParams {
    PointwiseParams pointwise;
    BroadcastAxes axes;
};

kernel void broadcast_binary(device const uchar* lhs_buffer [[buffer(0)]],
                             device const uchar* rhs_buffer [[buffer(1)]],
                             device uchar* output_buffer [[buffer(2)]],
                             constant BroadcastParams& params [[buffer(3)]],
                             uint index [[thread_position_in_grid]]) {
    if (index >= params.pointwise.count)
        return;
    const uint4 source = broadcast_indices(index, params.axes);
    evaluate(lhs_buffer + params.pointwise.lhs_offset, rhs_buffer + params.pointwise.rhs_offset,
             output_buffer + params.pointwise.output_offset, params.pointwise, source.x, source.y, index);
}

// ---------------------------------------------------------------------------
// cat (kOp 0) copies a contiguous input into its column block of each output
// row; pad (kOp 1) copies a strided input into the padded output. Elements
// move as kElementSize bytes (with equal dtypes), as in cat_pad.slang.

struct CatPadParams {
    ulong input_offset;
    ulong output_offset;
    uint input_dims[8];
    uint input_strides[8];
    uint output_strides[8];
    uint pad_before[8];
    uint count;
    uint rank;
    uint input_block;
    uint output_block;
    uint output_column;
    uint padding;
};

kernel void cat_pad(device const uchar* input_buffer [[buffer(0)]],
                    device uchar* output_buffer [[buffer(1)]],
                    constant CatPadParams& params [[buffer(2)]],
                    uint index [[thread_position_in_grid]]) {
    if (index >= params.count)
        return;
    device const uchar* input = input_buffer + params.input_offset;
    device uchar* output = output_buffer + params.output_offset;
    if (kOp == 0) {
        const uint row = index / params.input_block;
        copy_element(input, index, output,
                     ulong(row) * params.output_block + params.output_column + (index - row * params.input_block));
        return;
    }
    uint remaining = index;
    ulong input_index = 0;
    ulong output_index = 0;
    for (int axis = int(params.rank) - 1; axis >= 0; --axis) {
        const uint coordinate = remaining % params.input_dims[axis];
        remaining /= params.input_dims[axis];
        input_index += ulong(coordinate) * params.input_strides[axis];
        output_index += ulong(coordinate + params.pad_before[axis]) * params.output_strides[axis];
    }
    copy_element(input, input_index, output, output_index);
}

// ---------------------------------------------------------------------------
// Gathers a transposed 2D view through 32x32 tiles in threadgroup memory, so
// both the reads and the writes are coalesced.

struct TransposeParams {
    ulong input_offset;
    ulong output_offset;
    uint rows;
    uint columns;
    uint input_stride;
    uint padding;
};

template <typename T>
static void transpose_tile(device const uchar* input_buffer, device uchar* output_buffer,
                           constant TransposeParams& params, threadgroup T* tile,
                           uint2 local, uint2 group) {
    device const T* input = (device const T*)(input_buffer + params.input_offset);
    device T* output = (device T*)(output_buffer + params.output_offset);
    const uint row0 = group.x * 32, column0 = group.y * 32;
    for (uint step = 0; step < 32; step += 8) {
        const uint row = row0 + local.x, column = column0 + local.y + step;
        if (row < params.rows && column < params.columns)
            tile[(local.y + step) * 33 + local.x] = input[ulong(column) * params.input_stride + row];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = 0; step < 32; step += 8) {
        const uint row = row0 + local.y + step, column = column0 + local.x;
        if (row < params.rows && column < params.columns)
            output[ulong(row) * params.columns + column] = tile[local.x * 33 + local.y + step];
    }
}

kernel void transpose_2d(device const uchar* input_buffer [[buffer(0)]],
                         device uchar* output_buffer [[buffer(1)]],
                         constant TransposeParams& params [[buffer(2)]],
                         uint2 local [[thread_position_in_threadgroup]],
                         uint2 group [[threadgroup_position_in_grid]]) {
    if (kElementSize == 8) {
        threadgroup ulong tile[32 * 33];
        transpose_tile(input_buffer, output_buffer, params, tile, local, group);
    } else if (kElementSize == 4) {
        threadgroup uint tile[32 * 33];
        transpose_tile(input_buffer, output_buffer, params, tile, local, group);
    } else if (kElementSize == 2) {
        threadgroup ushort tile[32 * 33];
        transpose_tile(input_buffer, output_buffer, params, tile, local, group);
    } else {
        threadgroup uchar tile[32 * 33];
        transpose_tile(input_buffer, output_buffer, params, tile, local, group);
    }
}

// ---------------------------------------------------------------------------
// Reductions, ported from reduce.slang. kReduce follows ReduceOp (Sum,
// Mean, Max, Min, Prod, Any, All); element codes follow DataType, and
// LFS_DT_Pair marks float2 (sum, compensation) partials. Sums carry a
// Neumaier compensation through every step; max and min propagate the first
// NaN. A fused chain (kChainLength > 0) transforms Float32 elements first.
// Modes: 0 partial, each threadgroup folds a grid-stride slice of count
// elements into output[group]; 2 segmented, one threadgroup per contiguous
// segment of reduce elements; 3 strided, one thread per (outer, inner)
// output over the reduce range of its split (grid y), writing
// output[split][outer][inner]; 4 rows, one SIMD group per contiguous segment.

constant uint kReduceMode [[function_constant(16)]];

constant bool kFloatInput = kInputDType == LFS_DT_Float32 || kInputDType == LFS_DT_Pair;
constant bool kLogical = kReduce == LFS_REDUCE_ANY || kReduce == LFS_REDUCE_ALL;
constant bool kFloatAccumulator = kFloatInput && !kLogical;
constant bool kSumLike = kReduce == LFS_REDUCE_SUM || kReduce == LFS_REDUCE_MEAN;

struct ReduceParams {
    ulong input_offset;
    ulong output_offset;
    uint outer;
    uint reduce;
    uint inner;
    uint count;
    uint split_chunk;
    float mean_scale;
    ChainOp ops[16];
};

static float reduce_identity() {
    if (kReduce == LFS_REDUCE_MAX) return -INFINITY;
    if (kReduce == LFS_REDUCE_MIN) return INFINITY;
    if (kReduce == LFS_REDUCE_PROD) return 1.0f;
    return 0.0f;
}

static long int_identity() {
    if (kReduce == LFS_REDUCE_MAX) return -0x7fffffffffffffffl - 1;
    if (kReduce == LFS_REDUCE_MIN) return 0x7fffffffffffffffl;
    if (kReduce == LFS_REDUCE_PROD || kReduce == LFS_REDUCE_ALL) return 1;
    return 0;
}

static void combine_value(thread float& accumulator, thread float& compensation, float value) {
    if (kReduce == LFS_REDUCE_MAX) {
        accumulator = ieee_maximum(accumulator, value);
    } else if (kReduce == LFS_REDUCE_MIN) {
        accumulator = ieee_minimum(accumulator, value);
    } else if (kReduce == LFS_REDUCE_PROD) {
        accumulator *= value;
    } else {
        const float total = accumulator + value;
        compensation += abs(accumulator) >= abs(value) ? (accumulator - total) + value
                                                       : (value - total) + accumulator;
        accumulator = total;
    }
}

// TwoSum keeps both compensations when pairs meet.
static void combine_pair(thread float& accumulator, thread float& compensation, float2 pair) {
    if (!kSumLike) {
        combine_value(accumulator, compensation, pair.x);
        return;
    }
    const float total = accumulator + pair.x;
    const float carried = total - accumulator;
    const float error = (accumulator - (total - carried)) + (pair.x - carried);
    accumulator = total;
    compensation += error + pair.y;
}

static void combine_int(thread long& accumulator, long value) {
    if (kSumLike)
        accumulator += value;
    else if (kReduce == LFS_REDUCE_MAX)
        accumulator = max(accumulator, value);
    else if (kReduce == LFS_REDUCE_MIN)
        accumulator = min(accumulator, value);
    else if (kReduce == LFS_REDUCE_PROD)
        accumulator *= value;
    else if (kReduce == LFS_REDUCE_ANY)
        accumulator = accumulator != 0 || value != 0 ? 1 : 0;
    else
        accumulator = accumulator != 0 && value != 0 ? 1 : 0;
}

// Folds the (value, compensation) pairs of a SIMD group through shuffles.
static float2 reduce_simdgroup(float2 pair) {
    for (ushort offset = 16; offset > 0; offset /= 2) {
        float accumulator = pair.x, compensation = pair.y;
        combine_pair(accumulator, compensation, float2(simd_shuffle_down(pair.x, offset), simd_shuffle_down(pair.y, offset)));
        pair = float2(accumulator, compensation);
    }
    return pair;
}

static long reduce_simdgroup(long value) {
    for (ushort offset = 16; offset > 0; offset /= 2) {
        const uint2 halves = as_type<uint2>(value);
        combine_int(value, as_type<long>(uint2(simd_shuffle_down(halves.x, offset), simd_shuffle_down(halves.y, offset))));
    }
    return value;
}

// Folds a threadgroup's values; the result is valid in thread 0.
template <typename T>
static T reduce_threadgroup(T value, T identity, threadgroup T* shared, ushort lane, ushort simdgroup) {
    const T folded = reduce_simdgroup(value);
    if (lane == 0)
        shared[simdgroup] = folded;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return reduce_simdgroup(simdgroup == 0 && lane < kReduceThreads / 32 ? shared[lane] : identity);
}

static float load_reduce_float(device const uchar* input, ulong index, constant ReduceParams& params) {
    float value = ((device const float*)input)[index];
    for (uint op = 0; op < kChainLength; ++op) {
        const uint kind = chain_kind(op);
        const float rhs = chain_reads_tensor(kind) ? params.ops[op].rhs[index] : 0.0f;
        value = chain_step(value, kind, params.ops[op].scalar, rhs);
    }
    return value;
}

static long load_reduce_int(device const uchar* input, ulong index) {
    if (kInputDType == LFS_DT_Int32)
        return long(((device const int*)input)[index]);
    if (kInputDType == LFS_DT_Int64)
        return ((device const long*)input)[index];
    return long(input[index]);
}

struct Accumulator {
    float value;
    float compensation;
    long integer;
};

static Accumulator start_accumulator() {
    return Accumulator{reduce_identity(), 0.0f, int_identity()};
}

static void accumulate(thread Accumulator& accumulator, device const uchar* input, ulong index,
                       constant ReduceParams& params) {
    if (kFloatAccumulator) {
        if (kInputDType == LFS_DT_Pair)
            combine_pair(accumulator.value, accumulator.compensation, ((device const float2*)input)[index]);
        else
            combine_value(accumulator.value, accumulator.compensation, load_reduce_float(input, index, params));
    } else if (kLogical) {
        const bool nonzero = kFloatInput ? load_reduce_float(input, index, params) != 0.0f : load_reduce_int(input, index) != 0;
        combine_int(accumulator.integer, nonzero ? 1 : 0);
    } else {
        combine_int(accumulator.integer, load_reduce_int(input, index));
    }
}

// Only sums carry a compensation; adding a zero term to a max or min would
// turn -0 into +0.
static void store_reduction(device uchar* output, ulong index, Accumulator result, constant ReduceParams& params) {
    if (kOutputDType == LFS_DT_Pair) {
        ((device float2*)output)[index] = float2(result.value, result.compensation);
    } else if (kOutputDType == LFS_DT_Float32) {
        float value = float(result.integer);
        if (kFloatAccumulator)
            value = kSumLike && isfinite(result.compensation) ? result.value + result.compensation : result.value;
        if (kReduce == LFS_REDUCE_MEAN)
            value *= params.mean_scale;
        ((device float*)output)[index] = value;
    } else if (kOutputDType == LFS_DT_Int32) {
        ((device int*)output)[index] = int(result.integer);
    } else if (kOutputDType == LFS_DT_Int64) {
        ((device long*)output)[index] = result.integer;
    } else {
        output[index] = result.integer != 0 ? 1 : 0;
    }
}

kernel void reduce(device const uchar* input_buffer [[buffer(0)]],
                   device uchar* output_buffer [[buffer(1)]],
                   constant ReduceParams& params [[buffer(2)]],
                   uint thread_index [[thread_index_in_threadgroup]],
                   uint2 group [[threadgroup_position_in_grid]],
                   uint2 groups [[threadgroups_per_grid]],
                   ushort lane [[thread_index_in_simdgroup]],
                   ushort simdgroup [[simdgroup_index_in_threadgroup]]) {
    threadgroup float2 shared_pairs[kReduceThreads / 32];
    threadgroup long shared_integers[kReduceThreads / 32];
    device const uchar* input = input_buffer + params.input_offset;
    device uchar* output = output_buffer + params.output_offset;
    if (kReduceMode == 4) {
        const uint row = group.x * (kReduceThreads / 32) + simdgroup;
        if (row >= params.outer)
            return;
        const ulong base = ulong(row) * params.reduce;
        Accumulator accumulator = start_accumulator();
        for (uint element = lane; element < params.reduce; element += 32)
            accumulate(accumulator, input, base + element, params);
        if (kFloatAccumulator) {
            const float2 total = reduce_simdgroup(float2(accumulator.value, accumulator.compensation));
            accumulator.value = total.x;
            accumulator.compensation = total.y;
        } else {
            accumulator.integer = reduce_simdgroup(accumulator.integer);
        }
        if (lane == 0)
            store_reduction(output, row, accumulator, params);
        return;
    }
    if (kReduceMode == 3) {
        const uint outputs = params.outer * params.inner;
        const uint output_index = group.x * kReduceThreads + thread_index;
        if (output_index >= outputs)
            return;
        const uint outer_index = output_index / params.inner;
        const ulong base = ulong(outer_index) * params.reduce * params.inner + (output_index - outer_index * params.inner);
        const uint end = min(params.reduce, (group.y + 1) * params.split_chunk);
        Accumulator accumulator = start_accumulator();
        // Unrolled so the strided loads issue ahead of the in-order combines.
#pragma unroll(8)
        for (uint element = group.y * params.split_chunk; element < end; ++element)
            accumulate(accumulator, input, base + ulong(element) * params.inner, params);
        store_reduction(output, ulong(group.y) * outputs + output_index, accumulator, params);
        return;
    }
    Accumulator accumulator = start_accumulator();
    if (kReduceMode == 0) {
        for (ulong index = group.x * kReduceThreads + thread_index; index < params.count; index += groups.x * kReduceThreads)
            accumulate(accumulator, input, index, params);
    } else {
        const ulong base = ulong(group.x) * params.reduce;
        for (uint element = thread_index; element < params.reduce; element += kReduceThreads)
            accumulate(accumulator, input, base + element, params);
    }
    if (kFloatAccumulator) {
        const float2 total = reduce_threadgroup(float2(accumulator.value, accumulator.compensation),
                                                float2(reduce_identity(), 0.0f), shared_pairs, lane, simdgroup);
        accumulator.value = total.x;
        accumulator.compensation = total.y;
    } else {
        accumulator.integer = reduce_threadgroup(accumulator.integer, int_identity(), shared_integers, lane, simdgroup);
    }
    if (thread_index == 0)
        store_reduction(output, group.x, accumulator, params);
}

// ---------------------------------------------------------------------------
// Indexed extrema along the middle axis of an (outer, reduce, inner) view.
// Each candidate becomes a 64-bit key whose largest value is the CPU loop's
// choice: the high word orders values (inverted for the minimum), a NaN at
// position 0 sits above every number and later NaNs above that; the low word
// is the inverted position, so ties keep the first. kOp bit 0 picks the
// maximum; kOp >> 1 picks the pass: 0 a thread folds a chunk of one output's
// line, 1 a SIMD group folds a chunk of one contiguous line, 2 a SIMD group
// folds one output's chunk keys. With one chunk the first passes finish the output themselves.

struct ArgExtremeParams {
    uint outer;
    uint reduce;
    uint inner;
    uint chunk;
    uint chunks;
    uint padding[3];
};

static ulong arg_extreme_key(float value, uint position) {
    uint bits = as_type<uint>(value);
    uint order;
    if ((bits & 0x7fffffffu) > 0x7f800000u) {
        order = position == 0 ? 0xfffffffeu : 0xffffffffu;
    } else {
        if ((bits & 0x7fffffffu) == 0u)
            bits = 0u; // -0 ties +0, as the CPU's strict comparison does.
        order = (bits & 0x80000000u) != 0u ? ~bits : bits | 0x80000000u;
        if ((kOp & 1u) == 0u)
            order = ~order;
    }
    return (ulong(order) << 32) | ulong(~position);
}

static ulong arg_extreme_simdgroup(ulong key) {
    for (ushort offset = 16; offset > 0; offset /= 2) {
        const uint2 halves = as_type<uint2>(key);
        key = max(key, as_type<ulong>(uint2(simd_shuffle_down(halves.x, offset), simd_shuffle_down(halves.y, offset))));
    }
    return key;
}

static void arg_extreme_finish(device const float* input, device float* values, device long* indices,
                               constant ArgExtremeParams& params, uint output, ulong key) {
    const uint position = ~uint(key);
    const uint outer_index = output / params.inner;
    const ulong base = ulong(outer_index) * params.reduce * params.inner + (output - outer_index * params.inner);
    values[output] = input[base + ulong(position) * params.inner];
    indices[output] = long(position);
}

kernel void arg_extreme(device const float* input [[buffer(0)]],
                        device ulong* keys [[buffer(1)]],
                        device float* values [[buffer(2)]],
                        device long* indices [[buffer(3)]],
                        constant ArgExtremeParams& params [[buffer(4)]],
                        uint thread_index [[thread_index_in_threadgroup]],
                        uint2 group [[threadgroup_position_in_grid]],
                        ushort lane [[thread_index_in_simdgroup]],
                        ushort simdgroup [[simdgroup_index_in_threadgroup]]) {
    const uint pass = kOp >> 1;
    const uint outputs = params.outer * params.inner;
    if (pass == 1) {
        const uint row = group.x * (kReduceThreads / 32) + simdgroup;
        if (row >= params.outer)
            return;
        const ulong base = ulong(row) * params.reduce;
        const uint end = min(params.reduce, (group.y + 1) * params.chunk);
        ulong best = 0;
        for (uint position = group.y * params.chunk + lane; position < end; position += 32)
            best = max(best, arg_extreme_key(input[base + position], position));
        best = arg_extreme_simdgroup(best);
        if (lane != 0)
            return;
        if (params.chunks == 1)
            arg_extreme_finish(input, values, indices, params, row, best);
        else
            keys[ulong(group.y) * outputs + row] = best;
        return;
    }
    if (pass == 2) {
        const uint output = group.x * (kReduceThreads / 32) + simdgroup;
        if (output >= outputs)
            return;
        ulong best = 0;
        for (uint chunk = lane; chunk < params.chunks; chunk += 32)
            best = max(best, keys[ulong(chunk) * outputs + output]);
        best = arg_extreme_simdgroup(best);
        if (lane == 0)
            arg_extreme_finish(input, values, indices, params, output, best);
        return;
    }
    const uint output = group.x * kReduceThreads + thread_index;
    if (output >= outputs)
        return;
    ulong best = 0;
    const uint outer_index = output / params.inner;
    const ulong base = ulong(outer_index) * params.reduce * params.inner + (output - outer_index * params.inner);
    const uint end = min(params.reduce, (group.y + 1) * params.chunk);
#pragma unroll(8)
    for (uint position = group.y * params.chunk; position < end; ++position)
        best = max(best, arg_extreme_key(input[base + ulong(position) * params.inner], position));
    if (params.chunks == 1)
        arg_extreme_finish(input, values, indices, params, output, best);
    else
        keys[ulong(group.y) * outputs + output] = best;
}

// ---------------------------------------------------------------------------
// Counts over a threadgroup grid-stride, ported from count.slang. kOp picks
// the match: 0 nonzero bytes, 1 nonzero floats, 2 NaN (flag), 3 infinity
// (flag). Threadgroups add their tallies to one zeroed counter.

struct CountParams {
    ulong input_offset;
    uint count;
    uint padding;
};

kernel void count_matches(device const uchar* input_buffer [[buffer(0)]],
                          device atomic_uint* result [[buffer(1)]],
                          constant CountParams& params [[buffer(2)]],
                          uint thread_index [[thread_position_in_threadgroup]],
                          uint group [[threadgroup_position_in_grid]],
                          uint groups [[threadgroups_per_grid]],
                          ushort lane [[thread_index_in_simdgroup]],
                          ushort simdgroup [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint shared[kReduceThreads / 32];
    device const uchar* input = input_buffer + params.input_offset;
    uint matches = 0;
    for (ulong index = group * kReduceThreads + thread_index; index < params.count; index += groups * kReduceThreads) {
        if (kOp == 0) {
            matches += input[index] != 0 ? 1u : 0u;
        } else {
            const uint magnitude = as_type<uint>(((device const float*)input)[index]) & 0x7fffffffu;
            matches += (kOp == 1 ? magnitude != 0u : kOp == 2 ? magnitude > 0x7f800000u : magnitude == 0x7f800000u) ? 1u : 0u;
        }
    }
    const uint folded = simd_sum(matches);
    if (lane == 0)
        shared[simdgroup] = folded;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint total = simd_sum(simdgroup == 0 && lane < kReduceThreads / 32 ? shared[lane] : 0u);
    if (thread_index == 0 && total != 0) {
        if (kOp <= 1)
            atomic_fetch_add_explicit(result, total, memory_order_relaxed);
        else
            atomic_fetch_or_explicit(result, 1u, memory_order_relaxed);
    }
}

// ---------------------------------------------------------------------------
// In-place inclusive prefix sums along one axis of an (outer, dim, inner)
// view, ported from scan.slang for Float32 and Int32. kOp picks the pass:
// 0 short lines, one thread per line; 1 single-block lines, one threadgroup
// per line; 2 independent blocks, one threadgroup per (line, block), each
// writing its total; 3 adds the scanned totals of preceding blocks.

struct ScanParams {
    ulong data_offset;
    ulong totals_offset;
    uint outer;
    uint dim;
    uint inner;
    uint lines;
};

static ulong scan_line_base(uint line, constant ScanParams& params) {
    const uint outer_index = line / params.inner;
    return ulong(outer_index) * params.dim * params.inner + (line - outer_index * params.inner);
}

// Inclusive scan of a threadgroup's values in thread order.
template <typename T>
static T scan_block(T value, threadgroup T* totals, ushort lane, ushort simdgroup) {
    const T inclusive = simd_prefix_inclusive_sum(value);
    if (lane == 31)
        totals[simdgroup] = inclusive;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0) {
        const T preceding = simd_prefix_exclusive_sum(lane < kReduceThreads / 32 ? totals[lane] : T(0));
        if (lane < kReduceThreads / 32)
            totals[lane] = preceding;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return inclusive + totals[simdgroup];
}

// Passes 1 and 2 run a threadgroup per (block x, line y).
template <typename T>
static void scan_lines(device T* data, device T* totals, constant ScanParams& params, uint thread_index,
                       uint2 group, ushort lane, ushort simdgroup, threadgroup T* shared) {
    const uint blocks = (params.dim + kReduceThreads - 1) / kReduceThreads;
    if (kOp == 0) {
        const uint line = group.x * kReduceThreads + thread_index;
        if (line >= params.lines)
            return;
        const ulong base = scan_line_base(line, params);
        T running = 0;
        for (uint element = 0; element < params.dim; ++element) {
            running += data[base + ulong(element) * params.inner];
            data[base + ulong(element) * params.inner] = running;
        }
        return;
    }
    if (kOp == 3) {
        const ulong index = ulong(group.x) * kReduceThreads + thread_index;
        const uint element = uint(index / params.inner % params.dim);
        if (index >= ulong(params.lines) * params.dim || element < kReduceThreads)
            return;
        const ulong line = index / (ulong(params.dim) * params.inner) * params.inner + index % params.inner;
        data[index] += totals[line * blocks + element / kReduceThreads - 1];
        return;
    }
    const uint element = group.x * kReduceThreads + thread_index;
    const bool live = element < params.dim;
    const ulong index = scan_line_base(group.y, params) + ulong(element) * params.inner;
    const T scanned = scan_block(live ? data[index] : T(0), shared, lane, simdgroup);
    if (live)
        data[index] = scanned;
    if (kOp == 2 && thread_index == kReduceThreads - 1)
        totals[ulong(group.y) * blocks + group.x] = scanned;
}

kernel void scan(device uchar* data_buffer [[buffer(0)]],
                 device uchar* totals_buffer [[buffer(1)]],
                 constant ScanParams& params [[buffer(2)]],
                 uint thread_index [[thread_index_in_threadgroup]],
                 uint2 group [[threadgroup_position_in_grid]],
                 ushort lane [[thread_index_in_simdgroup]],
                 ushort simdgroup [[simdgroup_index_in_threadgroup]]) {
    threadgroup float shared_floats[kReduceThreads / 32];
    threadgroup int shared_ints[kReduceThreads / 32];
    if (kInputDType == LFS_DT_Float32)
        scan_lines((device float*)(data_buffer + params.data_offset), (device float*)(totals_buffer + params.totals_offset),
                   params, thread_index, group, lane, simdgroup, shared_floats);
    else
        scan_lines((device int*)(data_buffer + params.data_offset), (device int*)(totals_buffer + params.totals_offset),
                   params, thread_index, group, lane, simdgroup, shared_ints);
}

// ---------------------------------------------------------------------------
// Indexing with Int32 indices, ported from index.slang. kOp is the mode:
// 0 gather (output shaped like the index tensor), 1 take (flat, clamped),
// 2 index_select, 3 scatter assign, where the last position of each target
// wins, 4 scatter add, 5 index_put (flat, clamped), 6 index_fill, 7 winners
// (the last position of each target, for mode 3), 8 checked Int64-to-Int32
// index conversion. kBoundary is 0 assert (records a device fault and writes
// zero), 1 clamp or 2 wrap; kUnary applies abs (1), sqrt (2) or neg (3) after
// a take. Elements move as kElementSize bytes; adds read kInputDType.

constant uint kBoundary [[function_constant(17)]];
constant uint kUnary [[function_constant(18)]];

struct IndexParams {
    ulong input_offset;
    ulong index_offset;
    ulong value_offset;
    ulong winner_offset;
    uint outer;
    uint dim_size;
    uint inner;
    uint index_size;
    uint total;
    uint rank;
    uint index_rank;
    uint dim;
    uint fill_low;
    uint fill_high;
    uint input_size;
    uint op_id;
    uint input_dims[8];
    uint index_dims[8];
};

// Each batch binds its own fault record at buffer 5. The first fault of the
// batch wins; the host reads the record once the batch completed.
static void record_fault(device atomic_uint* fault, uint code, uint word1, uint word2, uint word3) {
    uint expected = 0;
    while (!atomic_compare_exchange_weak_explicit(fault, &expected, code, memory_order_relaxed,
                                                  memory_order_relaxed)) {
        if (expected != 0)
            return;
    }
    device uint* words = (device uint*)fault;
    words[1] = word1;
    words[2] = word2;
    words[3] = word3;
}

static void record_index_fault(device atomic_uint* fault, constant IndexParams& params, int index, uint extent) {
    record_fault(fault, 1, as_type<uint>(index), extent, params.op_id);
}

static void store_zero(device uchar* destination, ulong index) {
    if (kElementSize == 8)
        ((device ulong*)destination)[index] = 0;
    else if (kElementSize == 4)
        ((device uint*)destination)[index] = 0;
    else if (kElementSize == 2)
        ((device ushort*)destination)[index] = 0;
    else
        destination[index] = 0;
}

static void store_fill(device uchar* destination, ulong index, uint low, uint high) {
    if (kElementSize == 8)
        ((device ulong*)destination)[index] = (ulong(high) << 32) | low;
    else if (kElementSize == 4)
        ((device uint*)destination)[index] = low;
    else if (kElementSize == 2)
        ((device ushort*)destination)[index] = ushort(low);
    else
        destination[index] = uchar(low);
}

// Negative indices count from the end, then the result is clamped.
static uint clamp_flat(int index, uint size) {
    if (index < 0)
        index += int(size);
    return index < 0 ? 0u : index >= int(size) ? size - 1u : uint(index);
}

static int wrap_index(int index, uint extent) {
    if (index >= 0)
        return int(uint(index) % extent);
    const uint remainder = uint(-index) % extent;
    return remainder == 0u ? 0 : int(extent - remainder);
}

// Applies the boundary mode; false when an asserted index is out of range.
static bool bound_index(thread int& index, uint extent, constant IndexParams& params, device atomic_uint* fault) {
    if (kBoundary == 1) {
        index = max(0, min(int(extent) - 1, index));
    } else if (kBoundary == 2) {
        index = wrap_index(index, extent);
    } else if (index < 0 || index >= int(extent)) {
        record_index_fault(fault, params, index, extent);
        return false;
    }
    return true;
}

static bool gather_source(uint tid, device const int* indices, constant IndexParams& params, device atomic_uint* fault,
                          thread ulong& source) {
    if (kOp == 1) {
        source = clamp_flat(indices[tid], params.input_size);
        return true;
    }
    if (kOp == 2) {
        const uint outer_index = tid / (params.index_size * params.inner);
        const uint position = tid / params.inner % params.index_size;
        int selected = indices[position];
        if (!bound_index(selected, params.dim_size, params, fault))
            return false;
        source = (ulong(outer_index) * params.dim_size + uint(selected)) * params.inner + tid % params.inner;
        return true;
    }
    int gathered = indices[tid];
    if (!bound_index(gathered, params.input_dims[params.dim], params, fault))
        return false;
    // Output coordinates follow the index tensor; the input is contiguous.
    uint remaining = tid;
    uint coordinates[8];
    for (int axis = int(params.index_rank) - 1; axis >= 0; --axis) {
        coordinates[axis] = remaining % params.index_dims[axis];
        remaining /= params.index_dims[axis];
    }
    source = 0;
    for (uint axis = 0; axis < params.rank; ++axis) {
        const uint coordinate = axis == params.dim ? uint(gathered) : axis < params.index_rank ? coordinates[axis] : 0u;
        if (coordinate >= params.input_dims[axis])
            return false;
        source = source * params.input_dims[axis] + coordinate;
    }
    return true;
}

static void add_element(device uchar* base, ulong byte_offset, ulong index, device const uchar* values, uint tid) {
    if (kInputDType == LFS_DT_Float32) {
        atomic_fetch_add_explicit((device atomic_float*)(base + byte_offset) + index,
                                  ((device const float*)values)[tid], memory_order_relaxed);
    } else if (kInputDType == LFS_DT_Int32) {
        atomic_fetch_add_explicit((device atomic_int*)(base + byte_offset) + index,
                                  ((device const int*)values)[tid], memory_order_relaxed);
    } else {
        // Bytes add through their aligned word.
        const ulong position = byte_offset + index;
        device atomic_uint* word = (device atomic_uint*)base + (position >> 2);
        const uint shift = uint(position & 3) * 8;
        uint expected = atomic_load_explicit(word, memory_order_relaxed);
        uint replacement;
        do {
            const uint sum = ((expected >> shift) + values[tid]) & 255u;
            replacement = (expected & ~(255u << shift)) | (sum << shift);
        } while (!atomic_compare_exchange_weak_explicit(word, &expected, replacement, memory_order_relaxed,
                                                        memory_order_relaxed));
    }
}

kernel void index_op(device uchar* input_buffer [[buffer(0)]],
                     device const uchar* index_buffer [[buffer(1)]],
                     device uchar* value_buffer [[buffer(2)]],
                     device uchar* winner_buffer [[buffer(3)]],
                     constant IndexParams& params [[buffer(4)]],
                     device atomic_uint* fault [[buffer(5)]],
                     uint tid [[thread_position_in_grid]]) {
    if (tid >= params.total)
        return;
    device uchar* input = input_buffer + params.input_offset;
    device const int* indices = (device const int*)(index_buffer + params.index_offset);
    device uchar* values = value_buffer + params.value_offset;
    device atomic_int* winners = (device atomic_int*)(winner_buffer + params.winner_offset);
    if (kOp <= 2) {
        ulong source = 0;
        if (!gather_source(tid, indices, params, fault, source))
            store_zero(values, tid);
        else if (kUnary == 0)
            copy_element(input, source, values, tid);
        else
            ((device float*)values)[tid] = kUnary == 1 ? abs(((device const float*)input)[source])
                                           : kUnary == 2 ? sqrt(((device const float*)input)[source])
                                                         : -((device const float*)input)[source];
        return;
    }
    if (kOp == 5) {
        copy_element(values, tid, input, clamp_flat(indices[tid], params.input_size));
        return;
    }
    if (kOp == 7) {
        const int target = indices[tid];
        if (target < 0 || target >= int(params.dim_size))
            record_index_fault(fault, params, target, params.dim_size);
        else
            atomic_fetch_max_explicit(winners + target, int(tid), memory_order_relaxed);
        return;
    }
    if (kOp == 8) {
        const long index = ((device const long*)input)[tid];
        if (index < 0 || index >= long(params.dim_size)) {
            // Code 2 stores the signed index in words 1-2 and the extent in word 3.
            record_fault(fault, 2, uint(index), uint(ulong(index) >> 32), params.dim_size);
            ((device int*)values)[tid] = -1;
        } else {
            ((device int*)values)[tid] = int(index);
        }
        return;
    }
    // Modes 3, 4 and 6 cover outer * index_size * inner source elements.
    const uint position = tid / params.inner % params.index_size;
    const int target = indices[position];
    if (target < 0 || target >= int(params.dim_size)) {
        if (kOp != 3)
            record_index_fault(fault, params, target, params.dim_size);
        return;
    }
    const ulong destination = (ulong(tid / (params.index_size * params.inner)) * params.dim_size + uint(target)) * params.inner +
                              tid % params.inner;
    if (kOp == 3) {
        if (atomic_load_explicit(winners + target, memory_order_relaxed) == int(position))
            copy_element(values, tid, input, destination);
    } else if (kOp == 6) {
        store_fill(input, destination, params.fill_low, params.fill_high);
    } else {
        add_element(input_buffer, params.input_offset, destination, values, tid);
    }
}

// ---------------------------------------------------------------------------
// Masked ops, ported from mask.slang. kOp: 0 masked_fill, 1 and_live (keep a
// mask byte only where the live mask is set), 2 compact select, 3 compact
// scatter, 4 nonzero positions (Int64), 5 predicate values for an inclusive
// scan, 6 where_into (the fill where selected, else the source); compacted
// slots are scan[i] - 1. kPredicate reads a byte mask (0) or nonzero Float32
// elements (1).

constant uint kPredicate [[function_constant(19)]];

struct MaskParams {
    ulong data_offset;
    ulong mask_offset;
    ulong source_offset;
    ulong scan_offset;
    uint count;
    uint fill_low;
    uint fill_high;
    uint padding;
};

kernel void mask_op(device uchar* data_buffer [[buffer(0)]],
                    device const uchar* mask_buffer [[buffer(1)]],
                    device uchar* source_buffer [[buffer(2)]],
                    device uchar* scan_buffer [[buffer(3)]],
                    constant MaskParams& params [[buffer(4)]],
                    uint index [[thread_position_in_grid]]) {
    if (index >= params.count)
        return;
    device uchar* data = data_buffer + params.data_offset;
    device const uchar* mask = mask_buffer + params.mask_offset;
    device uchar* source = source_buffer + params.source_offset;
    device uint* scan = (device uint*)(scan_buffer + params.scan_offset);
    const bool selected = kPredicate == 0 ? mask[index] != 0 : ((device const float*)mask)[index] != 0.0f;
    if (kOp == 0) {
        if (selected)
            store_fill(data, index, params.fill_low, params.fill_high);
    } else if (kOp == 1) {
        if (!selected)
            data[index] = 0;
    } else if (kOp == 2) {
        if (selected)
            copy_element(data, index, source, scan[index] - 1);
    } else if (kOp == 3) {
        if (selected)
            copy_element(source, scan[index] - 1, data, index);
    } else if (kOp == 4) {
        if (selected)
            ((device long*)source)[scan[index] - 1] = long(index);
    } else if (kOp == 5) {
        scan[index] = selected ? 1u : 0u;
    } else if (selected) {
        store_fill(data, index, params.fill_low, params.fill_high);
    } else {
        copy_element(source, index, data, index);
    }
}

// ---------------------------------------------------------------------------
// Sorting Float32 lines with Int64 source positions, ported from sort.slang
// and radix.slang. Keys map float bits to an order-preserving unsigned value
// (both zeros equal, NaN after every number), inverted for descending order;
// positions break ties, so both sorts are stable. A line of an (outer, dim,
// inner) view is lines = outer * inner, element e at base + e * inner.

constant uint kSortCapacity = 2048;
constant uint kRadixDigits = 16;
constant uint kRadixPerThread = 8;
constant uint kRadixBlock = kReduceThreads * kRadixPerThread;

struct SortParams {
    ulong values_offset;
    ulong indices_offset;
    ulong keys_a_offset;
    ulong keys_b_offset;
    ulong positions_a_offset;
    ulong positions_b_offset;
    ulong histogram_offset;
    uint lines;
    uint dim_size;
    uint inner;
    uint blocks_per_line;
    uint shift;
    uint parity;
    uint descending;
    uint total;
};

static uint sortable_key(float value, bool descending) {
    uint bits = as_type<uint>(value);
    if ((bits & 0x7fffffffu) == 0u)
        bits = 0u;
    const uint key = (bits & 0x7fffffffu) > 0x7f800000u ? 0xffffffffu
                     : (bits & 0x80000000u) != 0u       ? ~bits
                                                        : bits | 0x80000000u;
    return descending ? ~key : key;
}

// Decodes a sort key into its value. Zeros, whose sign the key drops, and
// NaNs, whose payload it drops, return false and are read back instead.
static bool value_from_key(uint key, bool descending, thread float& value) {
    if (descending)
        key = ~key;
    if (key == 0x80000000u || key == 0xffffffffu)
        return false;
    value = as_type<float>((key & 0x80000000u) != 0u ? key & 0x7fffffffu : ~key);
    return true;
}

static ulong sort_line_base(uint line, constant SortParams& params) {
    const uint outer_index = line / params.inner;
    return ulong(outer_index) * params.dim_size * params.inner + (line - outer_index * params.inner);
}

// One threadgroup sorts kSortCapacity / kSortWidth lines of at most
// kSortWidth (a power of two) elements each with bitonic networks in
// threadgroup memory; a network's partners never leave its line's segment,
// and the final merge of every line runs ascending.
constant uint kSortWidth [[function_constant(29)]];

kernel void sort_shared(device uchar* values_buffer [[buffer(0)]],
                        device uchar* indices_buffer [[buffer(1)]],
                        constant SortParams& params [[buffer(2)]],
                        uint thread_index [[thread_index_in_threadgroup]],
                        uint group [[threadgroup_position_in_grid]]) {
    threadgroup uint keys[kSortCapacity];
    threadgroup uint positions[kSortCapacity];
    device float* values = (device float*)(values_buffer + params.values_offset);
    device long* indices = (device long*)(indices_buffer + params.indices_offset);
    constexpr uint per_thread = kSortCapacity / kReduceThreads;
    const uint first_line = group * (kSortCapacity / kSortWidth);
    // Element e holds position e % kSortWidth of line first_line + e / kSortWidth.
    const auto live = [&](const uint element) {
        return first_line + element / kSortWidth < params.lines && element % kSortWidth < params.dim_size;
    };
    const auto address = [&](const uint element, const uint position) {
        return sort_line_base(first_line + element / kSortWidth, params) + ulong(position) * params.inner;
    };
    for (uint slot = 0; slot < per_thread; ++slot) {
        const uint element = thread_index + slot * kReduceThreads;
        const bool present = live(element);
        keys[element] = present ? sortable_key(values[address(element, element % kSortWidth)], params.descending != 0)
                                : 0xffffffffu;
        positions[element] = present ? element % kSortWidth : 0xffffffffu;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint size = 2; size <= kSortWidth; size <<= 1) {
        for (uint span = size >> 1; span > 0; span >>= 1) {
            for (uint slot = 0; slot < per_thread; ++slot) {
                const uint element = thread_index + slot * kReduceThreads;
                const uint partner = element ^ span;
                if (partner <= element)
                    continue;
                const uint key_a = keys[element], key_b = keys[partner];
                const uint position_a = positions[element], position_b = positions[partner];
                const bool greater = key_a > key_b || (key_a == key_b && position_a > position_b);
                // The direction follows the position within the line.
                if (greater == ((element % kSortWidth & size) == 0u)) {
                    keys[element] = key_b;
                    keys[partner] = key_a;
                    positions[element] = position_b;
                    positions[partner] = position_a;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    // Values permute through threadgroup memory, so a line is never read
    // after it was partly overwritten.
    for (uint slot = 0; slot < per_thread; ++slot) {
        const uint element = thread_index + slot * kReduceThreads;
        if (live(element)) {
            indices[address(element, element % kSortWidth)] = long(positions[element]);
            keys[element] = as_type<uint>(values[address(element, positions[element])]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint slot = 0; slot < per_thread; ++slot) {
        const uint element = thread_index + slot * kReduceThreads;
        if (live(element))
            values[address(element, element % kSortWidth)] = as_type<float>(keys[element]);
    }
}

// Packed 16-bit digit counters: digit d lives in component d / 2, half d % 2.
static void add_packed(thread uint4& low, thread uint4& high, uint digit, uint amount) {
    const uint lane = digit >> 1, value = amount << ((digit & 1u) * 16u);
    if (lane < 4)
        low[lane] += value;
    else
        high[lane - 4] += value;
}

static uint read_packed(uint4 low, uint4 high, uint digit) {
    const uint lane = digit >> 1;
    return ((lane < 4 ? low[lane] : high[lane - 4]) >> ((digit & 1u) * 16u)) & 0xffffu;
}

// Least-significant-digit radix sort over 4-bit digits for longer lines.
// kOp is the phase: 0 extract keys and positions, 1 block histograms, 3
// stable scatter from A to B or B to A by pass parity, 4 decode the sorted
// keys into values in keys B, 5 write values and positions. Between 1 and 3
// the host scans each line's digit-major histogram inclusively.
kernel void radix_sort(device uchar* values_buffer [[buffer(0)]],
                       device uchar* indices_buffer [[buffer(1)]],
                       device uchar* scratch [[buffer(2)]],
                       constant SortParams& params [[buffer(3)]],
                       uint thread_index [[thread_index_in_threadgroup]],
                       uint group [[threadgroup_position_in_grid]],
                       uint position [[thread_position_in_grid]]) {
    threadgroup atomic_uint shared_histogram[kRadixDigits];
    threadgroup uint4 simd_low[kReduceThreads / 32], simd_high[kReduceThreads / 32];
    device float* values = (device float*)(values_buffer + params.values_offset);
    device uint* keys_a = (device uint*)(scratch + params.keys_a_offset);
    device uint* keys_b = (device uint*)(scratch + params.keys_b_offset);
    device uint* positions_a = (device uint*)(scratch + params.positions_a_offset);
    device uint* positions_b = (device uint*)(scratch + params.positions_b_offset);
    device uint* histogram = (device uint*)(scratch + params.histogram_offset);
    if (kOp == 0 || kOp == 4 || kOp == 5) {
        if (position >= params.total)
            return;
        const uint line = position / params.dim_size;
        const uint element = position - line * params.dim_size;
        const ulong base = sort_line_base(line, params);
        if (kOp == 0) {
            keys_a[position] = sortable_key(values[base + ulong(element) * params.inner], params.descending != 0);
            positions_a[position] = element;
        } else if (kOp == 4) {
            float value;
            if (!value_from_key(keys_a[position], params.descending != 0, value))
                value = values[base + ulong(positions_a[position]) * params.inner];
            keys_b[position] = as_type<uint>(value);
        } else {
            values[base + ulong(element) * params.inner] = as_type<float>(keys_b[position]);
            ((device long*)(indices_buffer + params.indices_offset))[base + ulong(element) * params.inner] =
                long(positions_a[position]);
        }
        return;
    }
    const bool from_a = params.parity == 0u;
    const uint line = group / params.blocks_per_line;
    const uint block = group - line * params.blocks_per_line;
    const ulong line_offset = ulong(line) * params.dim_size;
    const uint first = block * kRadixBlock + thread_index * kRadixPerThread;
    uint digits[kRadixPerThread];
    uint keys[kRadixPerThread];
    uint origins[kRadixPerThread];
    uint4 low = uint4(0), high = uint4(0);
    for (uint slot = 0; slot < kRadixPerThread; ++slot) {
        const uint element = first + slot;
        digits[slot] = kRadixDigits;
        if (element < params.dim_size) {
            keys[slot] = from_a ? keys_a[line_offset + element] : keys_b[line_offset + element];
            if (kOp == 3)
                origins[slot] = from_a ? positions_a[line_offset + element] : positions_b[line_offset + element];
            digits[slot] = (keys[slot] >> params.shift) & (kRadixDigits - 1u);
            add_packed(low, high, digits[slot], 1u);
        }
    }
    if (kOp == 1) {
        if (thread_index < kRadixDigits)
            atomic_store_explicit(&shared_histogram[thread_index], 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint digit = 0; digit < kRadixDigits; ++digit) {
            const uint count = read_packed(low, high, digit);
            if (count != 0u)
                atomic_fetch_add_explicit(&shared_histogram[digit], count, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (thread_index < kRadixDigits)
            histogram[(ulong(line) * kRadixDigits + thread_index) * params.blocks_per_line + block] =
                atomic_load_explicit(&shared_histogram[thread_index], memory_order_relaxed);
        return;
    }
    // Ranks within the block come from a prefix over the packed counters:
    // within each SIMD group, then across the groups' totals.
    uint4 exclusive_low = simd_prefix_exclusive_sum(low), exclusive_high = simd_prefix_exclusive_sum(high);
    const uint simd = thread_index / 32, lane = thread_index % 32;
    if (lane == 31) {
        simd_low[simd] = exclusive_low + low;
        simd_high[simd] = exclusive_high + high;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint earlier = 0; earlier < simd; ++earlier) {
        exclusive_low += simd_low[earlier];
        exclusive_high += simd_high[earlier];
    }
    // The line's histogram holds inclusive sums; the exclusive one is the entry before.
    const ulong line_base = ulong(line) * kRadixDigits * params.blocks_per_line;
    uint4 seen_low = uint4(0), seen_high = uint4(0);
    for (uint slot = 0; slot < kRadixPerThread; ++slot) {
        if (digits[slot] == kRadixDigits)
            continue;
        const uint rank = read_packed(exclusive_low, exclusive_high, digits[slot]) +
                          read_packed(seen_low, seen_high, digits[slot]);
        add_packed(seen_low, seen_high, digits[slot], 1u);
        const uint entry = digits[slot] * params.blocks_per_line + block;
        const ulong destination = line_offset + (entry == 0 ? 0u : histogram[line_base + entry - 1]) + rank;
        if (from_a) {
            keys_b[destination] = keys[slot];
            positions_b[destination] = origins[slot];
        } else {
            keys_a[destination] = keys[slot];
            positions_a[destination] = origins[slot];
        }
    }
}

// ---------------------------------------------------------------------------
// Random draws, ported from random.slang: element i takes Philox4x32-10 with
// counter i and the seed as key, so every element draws an independent,
// reproducible 128-bit block. kOp: 0 uniform [first, second), 1
// bernoulli(first), 2 randint [low, high), 3 normal(first, second), 4
// multinomial with replacement over kOp 8's running sums, 5 Gumbel keys for
// sampling without replacement (the host sorts them), 7 weight statistics
// (scaled sum, invalid flag, scale) in one threadgroup, 8 the running sums.

struct RandomParams {
    ulong output_offset;
    ulong weights_offset;
    ulong keys_offset;
    ulong seed;
    uint count;
    uint sample_count;
    int low;
    int high;
    float first;
    float second;
    float total;
    uint padding;
};

constant float kUnitScale = 1.0f / 16777216.0f;
constant float kTwoPi = 6.28318530717958647692f;

static uint4 philox_draw(ulong index, ulong seed) {
    uint4 counter = uint4(uint(index), uint(index >> 32), 0u, 0u);
    uint2 key = uint2(uint(seed), uint(seed >> 32));
    for (uint round = 0; round < 10; ++round) {
        const uint high0 = mulhi(0xD2511F53u, counter.x), low0 = 0xD2511F53u * counter.x;
        const uint high1 = mulhi(0xCD9E8D57u, counter.z), low1 = 0xCD9E8D57u * counter.z;
        counter = uint4(high1 ^ counter.y ^ key.x, low1, high0 ^ counter.w ^ key.y, low0);
        key += uint2(0x9E3779B9u, 0xBB67AE85u);
    }
    return counter;
}

// A 24-bit mantissa draw in [0, 1), as the CUDA kernels build it.
static float unit_interval(uint word) {
    return float(word >> 8) * kUnitScale;
}

static float float_below(float value) {
    const uint bits = as_type<uint>(value);
    return value > 0.0f ? as_type<float>(bits - 1u) : value == 0.0f ? as_type<float>(0x80000001u) : as_type<float>(bits + 1u);
}

kernel void random_op(device uchar* output_buffer [[buffer(0)]],
                      device const uchar* weights_buffer [[buffer(1)]],
                      device uchar* keys_buffer [[buffer(2)]],
                      constant RandomParams& params [[buffer(3)]],
                      uint index [[thread_position_in_grid]],
                      uint thread_index [[thread_index_in_threadgroup]],
                      ushort lane [[thread_index_in_simdgroup]],
                      ushort simdgroup [[simdgroup_index_in_threadgroup]]) {
    threadgroup float shared_values[kReduceThreads / 32];
    threadgroup uint shared_flags[kReduceThreads / 32];
    device uchar* output = output_buffer + params.output_offset;
    device const float* weights = (device const float*)(weights_buffer + params.weights_offset);
    device float* keys = (device float*)(keys_buffer + params.keys_offset);
    if (kOp == 7) {
        float maximum = 0.0f;
        uint invalid = 0u;
        for (uint i = thread_index; i < params.count; i += kReduceThreads) {
            const float weight = weights[i];
            invalid |= !(weight >= 0.0f) || isinf(weight) ? 1u : 0u;
            maximum = max(maximum, weight);
        }
        maximum = simd_max(maximum);
        invalid = simd_or(invalid);
        if (lane == 0) {
            shared_values[simdgroup] = maximum;
            shared_flags[simdgroup] = invalid;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        maximum = simd_max(lane < kReduceThreads / 32 ? shared_values[lane] : 0.0f);
        invalid = simd_or(lane < kReduceThreads / 32 ? shared_flags[lane] : 0u);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Scale the weights so their maximum sits near 2^0 before summing.
        const uint exponent = min(max((as_type<uint>(maximum) >> 23) & 255u, 1u), 253u);
        const float scale = as_type<float>((254u - exponent) << 23);
        float sum = 0.0f;
        for (uint i = thread_index; i < params.count; i += kReduceThreads)
            sum += weights[i] * scale;
        sum = simd_sum(sum);
        if (lane == 0)
            shared_values[simdgroup] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sum = simd_sum(lane < kReduceThreads / 32 ? shared_values[lane] : 0.0f);
        if (thread_index == 0) {
            ((device float*)output)[0] = sum;
            ((device uint*)output)[1] = invalid;
            ((device float*)output)[2] = scale;
        }
        return;
    }
    if (kOp == 8) {
        // The scaled weights' running sum into keys, in category order, so
        // each draw's search sees exactly the sums a linear scan adds up. One
        // SIMD group loads 32 weights at a time, the next 32 in flight.
        if (index >= 32)
            return;
        float cumulative = 0.0f;
        float next = lane < params.count ? weights[lane] * params.first : 0.0f;
        for (uint base = 0; base < params.count; base += 32) {
            const float scaled = next;
            const uint ahead = base + 32 + lane;
            next = ahead < params.count ? weights[ahead] * params.first : 0.0f;
            // Each lane adds the chunk's weights in order up to its own;
            // lanes past the end add zeros, which leave the sum unchanged.
            float running = cumulative;
#pragma unroll
            for (ushort j = 0; j < 32; ++j) {
                const float weight = simd_shuffle(scaled, j);
                running = j <= lane ? running + weight : running;
            }
            if (base + lane < params.count)
                keys[base + lane] = running;
            cumulative = simd_shuffle(running, ushort(31));
        }
        return;
    }
    if (index >= (kOp == 4 ? params.sample_count : params.count))
        return;
    const uint4 words = philox_draw(index, params.seed);
    if (kOp == 0) {
        float value = params.first;
        if (params.first != params.second) {
            value = fma(unit_interval(words.x), params.second - params.first, params.first);
            if (!(value < params.second))
                value = float_below(params.second);
        }
        ((device float*)output)[index] = value;
    } else if (kOp == 1) {
        ((device float*)output)[index] = unit_interval(words.x) < params.first ? 1.0f : 0.0f;
    } else if (kOp == 2) {
        const ulong range = ulong(long(params.high) - long(params.low));
        ((device int*)output)[index] = int(long(params.low) + long((ulong(words.x) * range) >> 32));
    } else if (kOp == 3) {
        const float radius = sqrt(-2.0f * log(float((words.x >> 8) + 1u) * kUnitScale));
        ((device float*)output)[index] = params.first + params.second * (radius * cos(kTwoPi * unit_interval(words.y)));
    } else if (kOp == 4) {
        // The first category whose running sum (kOp 8) exceeds the draw; the
        // sums never decrease, so a binary search finds it.
        const float u = unit_interval(words.x) * params.total;
        uint low = 0, high = params.count;
        while (low < high) {
            const uint middle = (low + high) / 2;
            if (u < keys[middle])
                high = middle;
            else
                low = middle + 1;
        }
        ((device long*)output)[index] = long(min(low, params.count - 1));
    } else {
        const float u = min(max(unit_interval(words.x), 1e-10f), 1.0f - 1e-10f);
        keys[index] = log(max(weights[index], 1e-10f)) - log(-log(u));
    }
}

// ---------------------------------------------------------------------------
// Spatial selection, ported from radius_neighbors.slang, point_region.slang
// and project_points.slang. Operands are GPU addresses in the parameters.
// Every operation rounds on its own (no contraction), as the rounded_math
// helpers of the Slang shaders do.

static float dot_rounded(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// kOp 0 links reference points into hashed cells, 1 marks the points within
// the radius of a reference.
struct RadiusParams {
    device const float* points;
    device const uchar* references;
    device int* heads;
    device int* next;
    device uchar* output;
    uint count;
    uint bucket_mask;
    float radius;
    uint padding;
};

static float3 radius_point(constant RadiusParams& params, uint i) {
    return float3(params.points[3 * ulong(i)], params.points[3 * ulong(i) + 1], params.points[3 * ulong(i) + 2]);
}

static int3 radius_cell(float3 point, float radius) {
    return int3(clamp(floor(point / radius * 0.5f), -268435456.0f, 268435456.0f));
}

static uint radius_bucket(int3 cell, uint mask) {
    return ((uint(cell.x) * 73856093u) ^ (uint(cell.y) * 19349663u) ^ (uint(cell.z) * 83492791u)) & mask;
}

static bool within_radius(float3 a, float3 b, float radius) {
    float3 d = a - b;
    float r2 = radius * radius;
    if (r2 < 1.17549435e-38f || !isfinite(r2)) {
        d = d / radius;
        r2 = 1.0f;
    }
    return dot_rounded(d, d) <= r2;
}

kernel void radius_neighbors(constant RadiusParams& params [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i >= params.count)
        return;
    const float3 point = radius_point(params, i);
    const bool finite = all(isfinite(point));
    if (kOp == 0) {
        if (params.references[i] != 0 && finite) {
            const uint bucket = radius_bucket(radius_cell(point, params.radius), params.bucket_mask);
            params.next[i] = atomic_exchange_explicit((device atomic_int*)&params.heads[bucket], int(i), memory_order_relaxed);
        }
        return;
    }
    bool found = finite && params.references[i] != 0;
    const int3 center = radius_cell(point, params.radius);
    for (int z = -1; finite && !found && z <= 1; ++z) {
        for (int y = -1; !found && y <= 1; ++y) {
            for (int x = -1; !found && x <= 1; ++x) {
                for (int j = params.heads[radius_bucket(center + int3(x, y, z), params.bucket_mask)]; j >= 0 && !found;
                     j = params.next[j])
                    found = within_radius(point, radius_point(params, uint(j)), params.radius);
            }
        }
    }
    params.output[i] = found ? 1 : 0;
}

// kOp is the region: 0 disk, 1 rectangle (edges included), 2 even-odd polygon,
// 3 any of several disks. Points inside set their mask byte; others keep it.
struct RegionParams {
    device uchar* mask;
    device const float* points;
    device const float* geometry;
    uint count;
    uint geometry_count;
    float4 bounds;
    float radius_sq;
    float minimum_coordinate;
};

static bool in_disk(float2 point, float2 center, float radius_sq) {
    const float dx = point.x - center.x, dy = point.y - center.y;
    return dx * dx + dy * dy <= radius_sq;
}

kernel void mark_points_2d(constant RegionParams& params [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i >= params.count)
        return;
    const float2 point = float2(params.points[2 * ulong(i)], params.points[2 * ulong(i) + 1]);
    if (!(point.x >= params.minimum_coordinate && point.y >= params.minimum_coordinate))
        return;
    bool inside = false;
    if (kOp == 0) {
        inside = in_disk(point, params.bounds.xy, params.radius_sq);
    } else if (kOp == 1) {
        inside = point.x >= params.bounds.x && point.y >= params.bounds.y && point.x <= params.bounds.z &&
                 point.y <= params.bounds.w;
    } else if (kOp == 3) {
        for (uint k = 0; k < params.geometry_count && !inside; ++k)
            inside = in_disk(point, float2(params.geometry[2 * k], params.geometry[2 * k + 1]), params.radius_sq);
    } else {
        for (uint k = 0, j = params.geometry_count - 1; k < params.geometry_count; j = k++) {
            const float xi = params.geometry[2 * k], yi = params.geometry[2 * k + 1];
            const float xj = params.geometry[2 * j], yj = params.geometry[2 * j + 1];
            if ((yi > point.y) != (yj > point.y) && point.x < (xj - xi) * (point.y - yi) / (yj - yi) + xi)
                inside = !inside;
        }
    }
    if (inside)
        params.mask[i] = 1;
}

// kOp is the PointProjectionModel; kProjectionInputs flags transforms (1),
// indices (2) and visibility (4).
constant uint kProjectionInputs [[function_constant(20)]];

struct ProjectionParams {
    float4 row0;
    float4 row1;
    float4 row2;
    device const float* points;
    device float2* output;
    device const uchar* transforms;
    device const int* indices;
    device const uchar* visibility;
    float2 scale;
    float2 center;
    float invalid_value;
    float near_distance;
    uint count;
    uint transform_count;
    uint visibility_count;
    uint padding;
};

static float4 view_position(constant ProjectionParams& params, uint i) {
    int node = (kProjectionInputs & 2u) != 0 ? params.indices[i] : 0;
    if ((kProjectionInputs & 6u) == 6u &&
        (node < 0 || uint(node) >= params.visibility_count || params.visibility[node] == 0))
        return float4(0.0f);
    float3 position = float3(params.points[3 * ulong(i)], params.points[3 * ulong(i) + 1], params.points[3 * ulong(i) + 2]);
    if ((kProjectionInputs & 1u) != 0 && params.transform_count > 0) {
        device const float* m = (device const float*)(params.transforms + ulong(clamp(node, 0, int(params.transform_count - 1))) * 64);
        position = float3(dot_rounded(float3(m[0], m[1], m[2]), position) + m[3],
                          dot_rounded(float3(m[4], m[5], m[6]), position) + m[7],
                          dot_rounded(float3(m[8], m[9], m[10]), position) + m[11]);
    }
    const float3 d = position - float3(params.row0.w, params.row1.w, params.row2.w);
    const float3 view = float3(dot_rounded(params.row0.xyz, d), dot_rounded(params.row1.xyz, d), dot_rounded(params.row2.xyz, d));
    return all(isfinite(view)) ? float4(view, 1.0f) : float4(0.0f);
}

kernel void project_points(constant ProjectionParams& params [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i >= params.count)
        return;
    const float4 view = view_position(params, i);
    const float2 invalid = float2(params.invalid_value);
    float2 pixel = invalid;
    if (kOp == 3) {
        const float3 direction = float3(view.x, -view.y, -view.z);
        const float length = view.w == 0.0f ? 0.0f : sqrt(dot_rounded(direction, direction));
        if (isfinite(length) && length > params.near_distance) {
            const float3 unit = direction / length;
            constexpr float pi = 3.14159265358979323846f;
            pixel = float2((atan2(unit.x, unit.z) / (2.0f * pi) + 0.5f) * params.scale.x,
                           (asin(clamp(unit.y, -1.0f, 1.0f)) / pi + 0.5f) * params.scale.y);
        }
    } else if (view.w != 0.0f && view.z < -params.near_distance) {
        if (kOp == 1) {
            if (isfinite(params.scale.x) && params.scale.x > 0.0f)
                pixel = float2(view.x * params.scale.x + params.center.x, -view.y * params.scale.y + params.center.y);
        } else {
            const float depth = -view.z;
            pixel = float2(view.x * params.scale.x / depth + params.center.x,
                           -view.y * params.scale.y / depth + params.center.y);
        }
    }
    params.output[i] = pixel;
}

// Clears the mask bytes of points outside the filters, ported from
// filter_points.slang. kFilter holds the LFS_FILTER_* flags and kOp the
// window's PointProjectionModel. The window block is the one Vulkan uploads:
// rotation, translation, focal, center, ortho scale, size, half extents,
// center and depth range.
constant uint kFilter [[function_constant(21)]];

struct FilterParams {
    device uchar* mask;
    device const float* points;
    device const float* transforms;
    device const int* indices;
    device const uchar* allowed;
    device const float* box_transform;
    device const float* box_min;
    device const float* box_max;
    device const float* ellipsoid_transform;
    device const float* radii;
    float window[25];
    uint count;
    uint transform_count;
    uint allowed_count;
    uint padding;
};

static float3 filter_vector(device const float* values, uint i) {
    return float3(values[3 * ulong(i)], values[3 * ulong(i) + 1], values[3 * ulong(i) + 2]);
}

// Applies a column-major world-to-local 4x4 matrix.
static float3 filter_local(float3 position, device const float* m) {
    return float3(dot_rounded(position, float3(m[0], m[4], m[8])) + m[12],
                  dot_rounded(position, float3(m[1], m[5], m[9])) + m[13],
                  dot_rounded(position, float3(m[2], m[6], m[10])) + m[14]);
}

static bool filter_keep(constant FilterParams& p, uint i) {
    int node = p.indices != nullptr ? p.indices[i] : 0;
    if ((kFilter & LFS_FILTER_NODES) != 0 && (node < 0 || uint(node) >= p.allowed_count || p.allowed[node] == 0))
        return false;
    if ((kFilter & LFS_FILTER_GEOMETRY) == 0)
        return true;
    float3 position = filter_vector(p.points, i);
    if (p.transforms != nullptr && p.transform_count > 0) {
        device const float* m = p.transforms + 16 * ulong(clamp(node, 0, int(p.transform_count - 1)));
        position = float3(dot_rounded(position, float3(m[0], m[1], m[2])) + m[3],
                          dot_rounded(position, float3(m[4], m[5], m[6])) + m[7],
                          dot_rounded(position, float3(m[8], m[9], m[10])) + m[11]);
    }
    if ((kFilter & LFS_FILTER_BOX) != 0) {
        const float3 local = filter_local(position, p.box_transform);
        const bool inside = all(local >= filter_vector(p.box_min, 0)) && all(local <= filter_vector(p.box_max, 0));
        if (inside == ((kFilter & LFS_FILTER_INVERSE_BOX) != 0))
            return false;
    }
    if ((kFilter & LFS_FILTER_ELLIPSOID) != 0) {
        const float3 local = filter_local(position, p.ellipsoid_transform), r = filter_vector(p.radii, 0);
        const float norm = local.x * local.x / (r.x * r.x) + local.y * local.y / (r.y * r.y) + local.z * local.z / (r.z * r.z);
        if ((norm <= 1.0f) == ((kFilter & LFS_FILTER_INVERSE_ELLIPSOID) != 0))
            return false;
    }
    if ((kFilter & LFS_FILTER_WINDOW) == 0)
        return true;
    constant float* w = p.window;
    const float3 d = position - float3(w[9], w[10], w[11]);
    const float vx = dot_rounded(d, float3(w[0], w[1], w[2]));
    const float vy = -dot_rounded(d, float3(w[3], w[4], w[5]));
    const float vz = -dot_rounded(d, float3(w[6], w[7], w[8]));
    float px, py, depth = vz;
    if (kOp == 0) {
        px = vx * w[12] / vz + w[14];
        py = vy * w[13] / vz + w[15];
    } else if (kOp == 1) {
        px = vx * w[16] + 0.5f * w[17];
        py = vy * w[16] + 0.5f * w[18];
    } else {
        const float len = sqrt(dot_rounded(float3(vx, vy, vz), float3(vx, vy, vz)));
        if (len <= 1.0e-6f || !isfinite(len))
            return false;
        constexpr float pi = 3.14159265358979323846f;
        px = (atan2(vx / len, vz / len) / (2.0f * pi) + 0.5f) * w[17];
        py = (asin(clamp(vy / len, -1.0f, 1.0f)) / pi + 0.5f) * w[18];
        depth = len;
    }
    return abs(px - w[21]) <= w[19] && abs(py - w[22]) <= w[20] && depth >= w[23] && depth <= w[24] && depth > 0.0f;
}

kernel void filter_points(constant FilterParams& params [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i < params.count && params.mask[i] != 0 && !filter_keep(params, i))
        params.mask[i] = 0;
}

// Label edits, ported from update_labels.slang. kOp is the phase: 0 updates
// every label in place, 3 copies the existing labels, then 1 clears and 2
// applies the labels of indexed rows. Racing rows of one index store the
// same byte, so plain stores suffice.
struct LabelParams {
    device uchar* output;
    device const uchar* selected;
    device const uchar* existing;
    device const uchar* locked;
    device const int* indices;
    device const int* categories;
    device const uchar* allowed;
    uint count;
    uint output_count;
    uint allowed_count;
    uint label;
    uint mode;
    uint padding;
};

static bool label_eligible(constant LabelParams& p, uint row) {
    if (p.categories == nullptr)
        return true;
    const int category = p.categories[row];
    return category >= 0 && uint(category) < p.allowed_count && p.allowed[category] != 0;
}

static uint label_old(constant LabelParams& p, uint row) {
    return p.existing != nullptr ? uint(p.existing[row]) : 0u;
}

static uint label_updated(constant LabelParams& p, uint old, bool selected) {
    if (!selected)
        return p.mode == 2 && old == p.label ? 0u : old;
    if (p.mode == 1)
        return old == p.label ? 0u : old;
    if (old != 0 && old != p.label && p.locked != nullptr && p.locked[old] != 0)
        return old;
    return p.label;
}

kernel void update_labels(constant LabelParams& p [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (kOp == 0 || kOp == 3) {
        if (i >= p.output_count)
            return;
        const uint old = label_old(p, i);
        const bool edit = kOp == 0 && label_eligible(p, i);
        p.output[i] = uchar(edit ? label_updated(p, old, p.selected[i] != 0) : old);
        return;
    }
    if (i >= p.count)
        return;
    const int id = p.indices[i];
    if (id < 0 || uint(id) >= p.output_count || !label_eligible(p, i))
        return;
    const bool selected = p.selected[i] != 0;
    const uint old = label_old(p, uint(id));
    if (kOp == 1) {
        if (!selected && old == p.label)
            p.output[id] = 0;
    } else if (selected) {
        // A selected row also restores its label after the replacement clear.
        const uint next = label_updated(p, old, true);
        if (next != old || (p.mode == 2 && old == p.label))
            p.output[id] = uchar(next);
    }
}

// ---------------------------------------------------------------------------
// Counts of the nonzero byte values into 256 bins, ported from histogram_u8.slang.

struct HistogramParams {
    device const uchar* input;
    device atomic_uint* output;
    uint size;
    uint padding;
};

kernel void histogram_u8(constant HistogramParams& params [[buffer(0)]],
                         uint index [[thread_position_in_grid]],
                         uint lane [[thread_index_in_threadgroup]],
                         uint threads [[threads_per_grid]]) {
    threadgroup atomic_uint counts[256];
    atomic_store_explicit(&counts[lane], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = index; i < params.size; i += threads) {
        const uint value = params.input[i];
        if (value != 0u)
            atomic_fetch_add_explicit(&counts[value], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint count = atomic_load_explicit(&counts[lane], memory_order_relaxed);
    if (count != 0u)
        atomic_fetch_add_explicit(&params.output[lane], count, memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// PPISP camera response, ported from ppisp_apply.slang (NVIDIA PPISP,
// Apache-2.0): vignetting, a homography in RG/intensity space, then
// per-channel response curves over a Float RGB CHW image.

struct PpispParams {
    float exposure_factor;
    float vignetting[15];
    float color_matrix[9];
    float crf[15];
    int y_offset;
    int full_height;
};

struct PpispApplyParams {
    device const float* input;
    device float* output;
    int width;
    int height;
    PpispParams settings;
};

kernel void ppisp_apply(constant PpispApplyParams& args [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    constant PpispParams& p = args.settings;
    const int width = args.width, height = args.height, plane = width * height;
    if (i >= uint(plane))
        return;
    const int full_height = p.full_height > 0 ? p.full_height : height;
    const float resolution = float(max(width, full_height));
    const float x = (float(i % uint(width)) + 0.5f - width * 0.5f) / resolution;
    const float y = (float(int(i / uint(width)) + p.y_offset) + 0.5f - full_height * 0.5f) / resolution;
    float3 rgb;
    for (int c = 0; c < 3; ++c) {
        const int k = c * 5;
        const float dx = x - p.vignetting[k], dy = y - p.vignetting[k + 1], r2 = fma(dx, dx, dy * dy), r4 = r2 * r2,
                    r6 = r4 * r2;
        const float falloff =
            max(0.0f, min(1.0f, fma(p.vignetting[k + 4], r6, fma(p.vignetting[k + 3], r4, fma(p.vignetting[k + 2], r2, 1.0f)))));
        rgb[c] = max(args.input[c * plane + i] * p.exposure_factor * falloff, 0.0f);
    }
    const float intensity = rgb.x + rgb.y + rgb.z;
    float3 rgi;
    for (int c = 0; c < 3; ++c) {
        const int k = c * 3;
        rgi[c] = fma(p.color_matrix[k], rgb.x, fma(p.color_matrix[k + 1], rgb.y, p.color_matrix[k + 2] * intensity));
    }
    const float norm = intensity / (max(rgi.z, 0.0f) + 1e-5f);
    rgb.x = rgi.x * norm;
    rgb.y = rgi.y * norm;
    rgb.z = rgi.z * norm - rgb.x - rgb.y;
    for (int c = 0; c < 3; ++c) {
        const int k = c * 5;
        const float value = clamp(rgb[c], 0.0f, 1.0f), mid = p.crf[k + 3], a = p.crf[k + 4];
        const float curve = value <= mid ? a * pow(value / mid, p.crf[k])
                                         : 1.0f - (1.0f - a) * pow((1.0f - value) / (1.0f - mid), p.crf[k + 1]);
        args.output[c * plane + i] = pow(max(0.0f, curve), p.crf[k + 2]);
    }
}

// ---------------------------------------------------------------------------
// Composites a Float CHW render over an equirectangular environment into
// u8 HWC, ported from environment_composite.slang. kOp 1 renders a panorama.

struct EnvironmentCompositeParams {
    float rotation[9];
    int full_width, full_height, band_width, band_height, y_offset;
    float focal_x, focal_y, center_x, center_y;
    int equirect_view;
    float exposure_factor, env_rotation_radians;
    int env_width, env_height;
};

struct CompositeParams {
    device const float* rgb;
    device const float* alpha;
    device const float* environment;
    device uchar* output;
    EnvironmentCompositeParams p;
};

static float3 environment_normalized(float3 v) {
    const float len = sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    return len <= 0 ? v : v * (1.0f / len);
}

static float3 environment_fetch(constant CompositeParams& params, int x, int y) {
    const uint i = uint(y * params.p.env_width + x) * 3;
    return float3(params.environment[i], params.environment[i + 1], params.environment[i + 2]);
}

kernel void environment_composite(constant CompositeParams& params [[buffer(0)]], uint idx [[thread_position_in_grid]]) {
    constant EnvironmentCompositeParams& p = params.p;
    const uint count = uint(p.band_width * p.band_height);
    if (idx >= count)
        return;
    constexpr float pi = 3.14159265358979323846f;
    const int x = int(idx) % p.band_width, y = int(idx) / p.band_width + p.y_offset;
    const float u = (float(x) + 0.5f) / p.full_width, v = 1.0f - (float(y) + 0.5f) / p.full_height;
    float3 local;
    if (kOp != 0) {
        const float lon = (u - 0.5f) * (2.0f * pi), lat = (v - 0.5f) * pi, c = cos(lat);
        local = environment_normalized(float3(sin(lon) * c, sin(lat), -cos(lon) * c));
    } else {
        local = environment_normalized(float3((u * p.full_width - p.center_x) / max(p.focal_x, 1e-6f),
                                              (v * p.full_height - p.center_y) / max(p.focal_y, 1e-6f), -1));
    }
    float3 dir = environment_normalized(float3(p.rotation[0] * local.x + p.rotation[3] * local.y + p.rotation[6] * local.z,
                                               p.rotation[1] * local.x + p.rotation[4] * local.y + p.rotation[7] * local.z,
                                               p.rotation[2] * local.x + p.rotation[5] * local.y + p.rotation[8] * local.z));
    const float c = cos(p.env_rotation_radians), s = sin(p.env_rotation_radians);
    dir = environment_normalized(float3(c * dir.x + s * dir.z, dir.y, -s * dir.x + c * dir.z));
    float eu = atan2(dir.x, -dir.z) / (2.0f * pi) + 0.5f;
    const float ev = clamp(0.5f - asin(clamp(dir.y, -1.0f, 1.0f)) / pi, 0.0f, 1.0f);
    eu -= floor(eu);
    const float ex = eu * (p.env_width - 1), ey = ev * (p.env_height - 1);
    const int x0 = clamp(int(floor(ex)), 0, p.env_width - 1), y0 = clamp(int(floor(ey)), 0, p.env_height - 1);
    const int x1 = (x0 + 1) % p.env_width, y1 = min(y0 + 1, p.env_height - 1);
    const float3 top = mix(environment_fetch(params, x0, y0), environment_fetch(params, x1, y0), ex - x0);
    const float3 bottom = mix(environment_fetch(params, x0, y1), environment_fetch(params, x1, y1), ex - x0);
    const float3 hdr = mix(top, bottom, ey - y0) * p.exposure_factor;
    float3 background = clamp(hdr * (2.51f * hdr + 0.03f) / (hdr * (2.43f * hdr + 0.59f) + 0.14f), 0.0f, 1.0f);
    background = clamp(pow(background, float3(1.0f / 2.2f)), 0.0f, 1.0f);
    const float3 render = float3(params.rgb[idx], params.rgb[count + idx], params.rgb[2 * count + idx]);
    const uint3 color = uint3(clamp(mix(background, render, params.alpha[idx]), 0.0f, 1.0f) * 255.0f + 0.5f);
    for (uint channel = 0; channel < 3; ++channel)
        params.output[3 * idx + channel] = uchar(color[channel]);
}

// ---------------------------------------------------------------------------
// Applies a linear map to splat log-scales and rotations, ported from
// affine_splat_geometry.slang. Safe math with contraction off keeps the
// TwoSum error terms that the Vulkan shader guards with an opaque zero.

struct AffineSplatParams {
    device const float* scales;
    device const float* rotations;
    device float* out_scales;
    device float* out_rotations;
    float linear[9];
    uint count;
};

static float2 two_sum(float a, float b) {
    const float total = a + b, part = total - a;
    return float2(total, (a - (total - part)) + (b - part));
}

static float2 add_pair(float2 a, float2 b) {
    const float2 sum = two_sum(a.x, b.x);
    return two_sum(sum.x, (a.y + b.y) + sum.y);
}

static float exp_difference(float a, float b) {
    if (a == -INFINITY)
        return 0;
    const float2 delta = two_sum(a, -b);
    if (delta.x < -104.0f)
        return 0;
    const int k = int(round(delta.x * 1.4426950408889634f));
    const float r = fma(-float(k), 0.693147182464599609375f, delta.x) + (delta.y + float(k) * 1.904654323148236e-9f);
    // Range reduction bounds abs(r) by ln(2)/2.
    float poly = 1.0f / 40320.0f;
    poly = fma(poly, r, 1.0f / 5040.0f);
    poly = fma(poly, r, 1.0f / 720.0f);
    poly = fma(poly, r, 1.0f / 120.0f);
    poly = fma(poly, r, 1.0f / 24.0f);
    poly = fma(poly, r, 1.0f / 6.0f);
    poly = fma(poly, r, 0.5f);
    poly = fma(poly, r, 1.0f);
    poly = fma(poly, r, 1.0f);
    return ldexp(poly, k);
}

static float2 log_pair(float value) {
    if (value == 0)
        return float2(-INFINITY, 0);
    const uint bits = as_type<uint>(value);
    int exponent = int((bits >> 23) & 255u) - 127;
    float m = as_type<float>((bits & 0x7fffffu) | 0x3f800000u);
    if (m > 1.41421356237f) {
        m *= 0.5f;
        ++exponent;
    }
    const float z = (m - 1) / (m + 1), z2 = z * z;
    float poly = 1.0f / 13.0f;
    poly = fma(poly, z2, 1.0f / 11.0f);
    poly = fma(poly, z2, 1.0f / 9.0f);
    poly = fma(poly, z2, 1.0f / 7.0f);
    poly = fma(poly, z2, 1.0f / 5.0f);
    poly = fma(poly, z2, 1.0f / 3.0f);
    const float fraction = 2 * z * fma(poly, z2, 1.0f);
    const float high = float(exponent) * 0.693147182464599609375f;
    const float low =
        fma(float(exponent), 0.693147182464599609375f, -high) + float(exponent) * (-1.904654323148236e-9f) + fraction;
    return float2(high, low);
}

static float restore_log_scale(float length, float2 linear_log, float largest_log) {
    if (length == 0)
        return -INFINITY;
    return add_pair(add_pair(log_pair(length), linear_log), float2(largest_log, 0)).x;
}

static float scaled_length(float3 v) {
    const float scale = max(abs(v.x), max(abs(v.y), abs(v.z)));
    return scale == 0 ? 0 : scale * length(v / scale);
}

static bool orthogonalize(thread float3& a, thread float3& b) {
    const float aa = dot(a, a), bb = dot(b, b), ab = dot(a, b);
    if (abs(ab) <= 2e-7f * sqrt(aa) * sqrt(bb))
        return false;
    const float delta = 0.5f * (bb - aa);
    const float scale = max(abs(delta), abs(ab));
    const float d = delta / scale, e = ab / scale;
    const float h = sqrt(d * d + e * e);
    const float t = e / (d + (delta < 0 ? -h : h));
    const float c = rsqrt(1 + t * t), s = t * c;
    const float3 old = a;
    a = c * old - s * b;
    b = s * old + c * b;
    return true;
}

static void sort_axes(thread float3& a, thread float3& b, thread float& la, thread float& lb) {
    if (la < lb) {
        const float3 v = a;
        a = b;
        b = v;
        const float l = la;
        la = lb;
        lb = l;
    }
}

kernel void affine_splat_geometry(constant AffineSplatParams& p [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.count)
        return;
    device float* const out_scale = p.out_scales + 3 * i;
    device float* const out_rotation = p.out_rotations + 4 * i;
    const float3 log_scale(p.scales[3 * i], p.scales[3 * i + 1], p.scales[3 * i + 2]);
    float4 q(p.rotations[4 * i], p.rotations[4 * i + 1], p.rotations[4 * i + 2], p.rotations[4 * i + 3]);
    const float qm = max(max(abs(q.x), abs(q.y)), max(abs(q.z), abs(q.w)));
    q = qm > 0 ? normalize(q / qm) : float4(1, 0, 0, 0);
    const float w = q.x, x = q.y, y = q.z, z = q.w;
    // Difference of squares preserves zero diagonal entries at equal quaternion components.
    const float3 r0((w * w + x * x) - (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y));
    const float3 r1(2 * (x * y - w * z), (w * w + y * y) - (x * x + z * z), 2 * (y * z + w * x));
    const float3 r2(2 * (x * z + w * y), 2 * (y * z - w * x), (w * w + z * z) - (x * x + y * y));
    const float largest_log = max(log_scale.x, max(log_scale.y, log_scale.z));
    float linear_scale = 0;
    for (int k = 0; k < 9; ++k)
        linear_scale = max(linear_scale, abs(p.linear[k]));
    if (linear_scale == 0 || largest_log == -INFINITY) {
        for (int k = 0; k < 3; ++k)
            out_scale[k] = -INFINITY;
        for (int k = 0; k < 4; ++k)
            out_rotation[k] = k == 0 ? 1 : 0;
        return;
    }
    // Common scale factors leave singular vectors unchanged and bound matrix entries.
    const float3 a0 = float3(p.linear[0], p.linear[1], p.linear[2]) / linear_scale;
    const float3 a1 = float3(p.linear[3], p.linear[4], p.linear[5]) / linear_scale;
    const float3 a2 = float3(p.linear[6], p.linear[7], p.linear[8]) / linear_scale;
    float3 b0 = float3(dot(a0, r0), dot(a1, r0), dot(a2, r0)) * exp_difference(log_scale.x, largest_log);
    float3 b1 = float3(dot(a0, r1), dot(a1, r1), dot(a2, r1)) * exp_difference(log_scale.y, largest_log);
    float3 b2 = float3(dot(a0, r2), dot(a1, r2), dot(a2, r2)) * exp_difference(log_scale.z, largest_log);
    for (int sweep = 0; sweep < 8; ++sweep) {
        bool changed = orthogonalize(b0, b1);
        changed = orthogonalize(b0, b2) || changed;
        changed = orthogonalize(b1, b2) || changed;
        if (!changed)
            break;
    }
    float l0 = scaled_length(b0), l1 = scaled_length(b1), l2 = scaled_length(b2);
    sort_axes(b0, b1, l0, l1);
    sort_axes(b0, b2, l0, l2);
    sort_axes(b1, b2, l1, l2);
    const float2 linear_log = log_pair(linear_scale);
    out_scale[0] = restore_log_scale(l0, linear_log, largest_log);
    out_scale[1] = restore_log_scale(l1, linear_log, largest_log);
    out_scale[2] = restore_log_scale(l2, linear_log, largest_log);
    const float3 u = l0 > 0 ? b0 / l0 : float3(1, 0, 0);
    float3 v = l1 > 0 ? b1 / l1 : float3(0);
    // The null-space basis is arbitrary; the completed frame must remain orthonormal.
    v -= u * dot(u, v);
    if (dot(v, v) < 1e-12f) {
        const float3 axis = abs(u.x) <= abs(u.y) && abs(u.x) <= abs(u.z) ? float3(1, 0, 0)
                            : abs(u.y) <= abs(u.z)                        ? float3(0, 1, 0)
                                                                          : float3(0, 0, 1);
        v = axis - u * dot(u, axis);
    }
    v = normalize(v);
    const float3 t = cross(u, v);
    // A right-handed frame represents reflections without changing the covariance.
    const float trace = u.x + v.y + t.z;
    if (trace > 0) {
        const float s = 2 * sqrt(1 + trace);
        q = float4(s / 4, (v.z - t.y) / s, (t.x - u.z) / s, (u.y - v.x) / s);
    } else if (u.x > v.y && u.x > t.z) {
        const float s = 2 * sqrt(1 + u.x - v.y - t.z);
        q = float4((v.z - t.y) / s, s / 4, (u.y + v.x) / s, (t.x + u.z) / s);
    } else if (v.y > t.z) {
        const float s = 2 * sqrt(1 + v.y - u.x - t.z);
        q = float4((t.x - u.z) / s, (u.y + v.x) / s, s / 4, (v.z + t.y) / s);
    } else {
        const float s = 2 * sqrt(1 + t.z - u.x - v.y);
        q = float4((u.y - v.x) / s, (t.x + u.z) / s, (v.z + t.y) / s, s / 4);
    }
    q = normalize(q);
    if (q.x < 0)
        q = -q;
    for (int k = 0; k < 4; ++k)
        out_rotation[k] = q[k];
}

// ---------------------------------------------------------------------------
// Image resampling, ported from image_resample.slang. kOp 0 undistorts CHW
// images or HW masks with the COLMAP camera models (BSD-3 formulas of
// sensor/models.h); 1 resizes depth and 2 normal priors over valid taps only.

struct ResampleParams {
    device const float* input;
    device float* output;
    float src_fx, src_fy, src_cx, src_cy, dst_fx, dst_fy, dst_cx, dst_cy;
    int sw, sh, dw, dh;
    float distortion[12];
    int model, num_distortion, channels, padding;
};

static float2 distort_pinhole(float x, float y, constant float* dist, int n) {
    const float r2 = x * x + y * y, r4 = r2 * r2, r6 = r4 * r2;
    const float k1 = n > 0 ? dist[0] : 0.0f, k2 = n > 1 ? dist[1] : 0.0f, k3 = n > 2 ? dist[2] : 0.0f;
    const float radial = 1.0f + k1 * r2 + k2 * r4 + k3 * r6;
    const float p1 = n > 3 ? dist[3] : 0.0f, p2 = n > 4 ? dist[4] : 0.0f;
    return float2(x * radial + 2.0f * p1 * x * y + p2 * (r2 + 2.0f * x * x),
                  y * radial + p1 * (r2 + 2.0f * y * y) + 2.0f * p2 * x * y);
}

static float2 distort_fisheye(float x, float y, constant float* dist, int n) {
    const float r = sqrt(x * x + y * y);
    if (r < 1e-8f)
        return float2(x, y);
    const float theta = atan(r), theta2 = theta * theta, theta4 = theta2 * theta2, theta6 = theta4 * theta2,
                theta8 = theta4 * theta4;
    const float k1 = n > 0 ? dist[0] : 0.0f, k2 = n > 1 ? dist[1] : 0.0f, k3 = n > 2 ? dist[2] : 0.0f,
                k4 = n > 3 ? dist[3] : 0.0f;
    const float scale = theta * (1.0f + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8) / r;
    return float2(x * scale, y * scale);
}

static float2 distort_thin_prism_fisheye(float x, float y, constant float* dist, int n) {
    if (sqrt(x * x + y * y) < 1e-8f)
        return float2(x, y);
    const float2 fisheye = distort_fisheye(x, y, dist, n);
    float xd = fisheye.x, yd = fisheye.y;
    const float p1 = n > 4 ? dist[4] : 0.0f, p2 = n > 5 ? dist[5] : 0.0f;
    const float r2 = xd * xd + yd * yd;
    xd += 2.0f * p1 * xd * yd + p2 * (r2 + 2.0f * xd * xd);
    yd += p1 * (r2 + 2.0f * yd * yd) + 2.0f * p2 * xd * yd;
    const float s1 = n > 6 ? dist[6] : 0.0f, s2 = n > 7 ? dist[7] : 0.0f, s3 = n > 8 ? dist[8] : 0.0f,
                s4 = n > 9 ? dist[9] : 0.0f;
    const float r2d = xd * xd + yd * yd, r4d = r2d * r2d;
    return float2(xd + (s1 * r2d + s2 * r4d), yd + (s3 * r2d + s4 * r4d));
}

static float resample_tap(constant ResampleParams& p, int base, int x, int y) {
    return x >= 0 && y >= 0 && x < p.sw && y < p.sh ? p.input[base + y * p.sw + x] : 0.0f;
}

static bool prior_valid(constant ResampleParams& p, int i, int plane) {
    const float a = p.input[i];
    if (!isfinite(a))
        return false;
    if (kOp == 1)
        return a > 0;
    const float b = p.input[plane + i], c = p.input[2 * plane + i];
    return isfinite(b) && isfinite(c) && a * a + b * b + c * c >= 0.25f;
}

kernel void image_resample(constant ResampleParams& p [[buffer(0)]], uint idx [[thread_position_in_grid]]) {
    const int sp = p.sw * p.sh, dp = p.dw * p.dh;
    if (idx >= uint(dp))
        return;
    const int x = int(idx) % p.dw, y = int(idx) / p.dw;
    if (kOp == 0) {
        float2 d = float2((float(x) + 0.5f - p.dst_cx) / p.dst_fx, (float(y) + 0.5f - p.dst_cy) / p.dst_fy);
        if (p.model == 0)
            d = distort_pinhole(d.x, d.y, p.distortion, p.num_distortion);
        else if (p.model == 2)
            d = distort_fisheye(d.x, d.y, p.distortion, p.num_distortion);
        else if (p.model == 4)
            d = distort_thin_prism_fisheye(d.x, d.y, p.distortion, p.num_distortion);
        const float sx = d.x * p.src_fx + p.src_cx - 0.5f, sy = d.y * p.src_fy + p.src_cy - 0.5f;
        const int x0 = int(floor(sx)), y0 = int(floor(sy));
        const float fx = sx - floor(sx), fy = sy - floor(sy);
        for (int c = 0; c < p.channels; ++c) {
            const float v00 = resample_tap(p, c * sp, x0, y0), v01 = resample_tap(p, c * sp, x0 + 1, y0);
            const float v10 = resample_tap(p, c * sp, x0, y0 + 1), v11 = resample_tap(p, c * sp, x0 + 1, y0 + 1);
            p.output[c * dp + int(idx)] = (1.0f - fy) * ((1.0f - fx) * v00 + fx * v01) + fy * ((1.0f - fx) * v10 + fx * v11);
        }
        return;
    }
    const float sx = max(0.0f, min(p.sw - 1.0f, (float(x) + 0.5f) * p.sw / p.dw - 0.5f));
    const float sy = max(0.0f, min(p.sh - 1.0f, (float(y) + 0.5f) * p.sh / p.dh - 0.5f));
    const int x0 = int(sx), y0 = int(sy);
    float3 value = 0;
    float weight = 0;
    if (prior_valid(p, int(sy + 0.5f) * p.sw + int(sx + 0.5f), sp)) {
        for (int j = 0; j < 2; ++j) {
            for (int i = 0; i < 2; ++i) {
                const int index = min(y0 + j, p.sh - 1) * p.sw + min(x0 + i, p.sw - 1);
                if (!prior_valid(p, index, sp))
                    continue;
                const float w = (i != 0 ? sx - x0 : 1.0f - (sx - x0)) * (j != 0 ? sy - y0 : 1.0f - (sy - y0));
                weight += w;
                for (int c = 0; c < p.channels; ++c)
                    value[c] += w * p.input[c * sp + index];
            }
        }
    }
    if (kOp == 2)
        weight = sqrt(value.x * value.x + value.y * value.y + value.z * value.z);
    for (int c = 0; c < p.channels; ++c)
        p.output[c * dp + int(idx)] = weight > 1e-8f ? value[c] / weight : 0.0f;
}

// ---------------------------------------------------------------------------
// Spherical-harmonics storage conversion, ported from sh_codec.slang and
// sh_encode.slang. Formats are ShFormat: 0 canonical [N,K,3], 1 Float32 and
// 2 Float16 in float4 cell groups of 32-row tiles, 3 u16 codes in cell columns
// of 32-row tiles with Float32 bounds per 256 rows. kShIndices is 0 without
// indices, 1 for Int32 and 2 for Int64.
constant uint kShSource [[function_constant(22)]];
constant uint kShDestination [[function_constant(23)]];
constant uint kShIndices [[function_constant(24)]];
constant uint kShSourceRest [[function_constant(25)]];
constant uint kShDestinationRest [[function_constant(26)]];

struct ShParams {
    device const uchar* source;
    device uchar* destination;
    device const uchar* indices;
    device const float* source_bounds;
    device float* destination_bounds;
    uint source_rows, destination_rows, count;
    uint source_offset, destination_offset, padding;
};

static uint sh_offset(uint row, uint cell, uint rest, bool q16) {
    if (q16)
        return (row / 32 * (rest * 3) + cell) * 32 + row % 32;
    return ((row / 32 * ((rest * 3 + 3) / 4) + cell / 4) * 32 + row % 32) * 4 + cell % 4;
}

static uint sh_index(constant ShParams& p, uint i) {
    return kShIndices == 2 ? uint(((device const long*)p.indices)[i]) : uint(((device const int*)p.indices)[i]);
}

static float sh_read(constant ShParams& p, uint row, uint c) {
    const uint width = kShSource == 1 || kShSource == 2 ? (kShSourceRest * 3 + 3) / 4 * 4 : kShSourceRest * 3;
    if (row >= p.source_rows || c >= width)
        return 0;
    const uint offset = kShSource == 0 ? row * kShSourceRest * 3 + c : sh_offset(row, c, kShSourceRest, kShSource == 3);
    if (kShSource == 3) {
        const float lo = p.source_bounds[row / 256 * 2], hi = p.source_bounds[row / 256 * 2 + 1];
        const uint code = ((device const ushort*)p.source)[offset];
        return fma(hi - lo, float(code) * (1.0f / 65535.0f), lo);
    }
    if (kShSource == 2)
        return float(((device const half*)p.source)[offset]);
    return ((device const float*)p.source)[offset];
}

kernel void sh_codec(constant ShParams& p [[buffer(0)]], uint index [[thread_position_in_grid]],
                     uint threads [[threads_per_grid]]) {
    const uint width = kShDestination == 0 ? kShDestinationRest * 3 : (kShDestinationRest * 3 + 3) / 4 * 4;
    uint rows = p.count;
    // A range that ends the destination also zeroes the tail lanes of its last tile.
    if (kScatter == 0 && p.destination_offset + p.count == p.destination_rows && kShDestination != 0)
        rows += (32 - p.destination_rows % 32) % 32;
    for (uint i = index; i < (rows + 31) / 32 * 32 * width; i += threads) {
        // Destinations are enumerated in memory order.
        const uint row = kShDestination == 0 ? i / width : i / (width * 32) * 32 + i / 4 % 32;
        const uint c = kShDestination == 0 ? i % width : i / 128 % (width / 4) * 4 + i % 4;
        if (row >= rows)
            continue;
        const uint source_row = kShIndices != 0 && kScatter == 0 && row < p.count ? sh_index(p, row) : p.source_offset + row;
        const uint destination_row = kScatter != 0 ? sh_index(p, row) : p.destination_offset + row;
        const float value =
            row < p.count && !(c >= kShDestinationRest * 3 && kShSource != 1) ? sh_read(p, source_row, c) : 0;
        const uint offset = kShDestination == 0 ? destination_row * width + c
                                                : sh_offset(destination_row, c, kShDestinationRest, false);
        if (kShDestination == 2)
            ((device half*)p.destination)[offset] = half(value);
        else
            ((device float*)p.destination)[offset] = value;
    }
}

// Q16 encoding: every threadgroup quantizes 256 rows against their bounds.
kernel void sh_encode(constant ShParams& p [[buffer(0)]], uint group [[threadgroup_position_in_grid]],
                      uint lane [[thread_index_in_threadgroup]], uint simd [[simdgroup_index_in_threadgroup]],
                      uint simds [[simdgroups_per_threadgroup]]) {
    threadgroup float lows[32], highs[32];
    const uint row = group * 256 + lane;
    float cells[45];
    float lo = 1e30f, hi = -1e30f;
    if (row < p.count) {
        const uint source_row = kShIndices != 0 ? sh_index(p, row) : p.source_offset + row;
        for (uint c = 0; c < kShDestinationRest * 3; ++c) {
            cells[c] = c < kShSourceRest * 3 ? sh_read(p, source_row, c) : 0;
            lo = fmin(lo, cells[c]);
            hi = fmax(hi, cells[c]);
        }
    }
    lo = simd_min(lo);
    hi = simd_max(hi);
    if (simd_is_first()) {
        lows[simd] = lo;
        highs[simd] = hi;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = 0; k < simds; ++k) {
        lo = min(lo, lows[k]);
        hi = max(hi, highs[k]);
    }
    if (lo > hi)
        lo = hi = 0;
    if (lane == 0) {
        p.destination_bounds[group * 2] = lo;
        p.destination_bounds[group * 2 + 1] = hi;
    }
    if (row >= (p.count + 31) / 32 * 32)
        return;
    for (uint c = 0; c < kShDestinationRest * 3; ++c) {
        uint code = 0;
        if (row < p.count) {
            const float scaled = 65535.0f * (cells[c] - lo) / fmax(hi - lo, 1e-20f);
            const float base = floor(scaled);
            code = uint(fmin(fmax(base + (scaled - base >= 0.5f ? 1.0f : 0.0f), 0.0f), 65535.0f));
        }
        ((device ushort*)p.destination)[sh_offset(row, c, kShDestinationRest, true)] = ushort(code);
    }
}

// ---------------------------------------------------------------------------
// RAD LOD pages, ported from rad_page.slang with the radmath formulas of
// rad_dequant_math.hpp. The pool keeps float16 halves, so each step that can
// move a value across a half rounding boundary follows the Slang arithmetic:
// halves flush to zero, byte fractions multiply by rounded reciprocals and
// rotation divisions multiply by a correctly rounded reciprocal. kOp is the
// phase: 0 dequantizes a packed page, 1 clears the SH maxima of a resident
// page, 2 reduces them and 3 quantizes the resident page.

struct RadPackedProperty {
    uint kind, encoding, plane_offset, plane_bytes;
    float min_val, max_val, scale;
};

struct RadPagePackedDesc {
    uint count, sh_coeffs_rest, lod_opacity, property_count;
    uint meta_bounds_offset, meta_links_offset, meta_node_count, used_bytes, chunk, reserved[3];
    float bbox_min[3], bbox_extent[3], log_size_min, log_size_range;
    RadPackedProperty props[8];
};

struct RadSources {
    device const float* means;
    device const float* sh0;
    device const uchar* shN;
    device const float* rotation;
    device const float* scaling;
    device const float* opacity;
    device const float* sh_bounds;
    uint offset, count, rest, half_sh, quant_sh, padding;
};

struct RadPageParams {
    device uint* means;
    device uint* sh0;
    device uint* shN;
    device uint* rotation;
    device uint* scaling;
    device uint* opacity;
    device uint* frames;
    device uint* bounds;
    device uint* links;
    device const uchar* packed;
    device const RadPagePackedDesc* descriptor;
    uint page, page_splats, slots, padding;
    RadSources sources;
};

static float rad_half_to_float(uint value) {
    const uint sign = (value >> 15) & 1u;
    uint exponent = (value >> 10) & 0x1fu;
    uint mantissa = value & 0x3ffu;
    uint bits;
    if (exponent == 0u) {
        if (mantissa == 0u) {
            bits = sign << 31;
        } else {
            int shift = 0;
            while ((mantissa & 0x400u) == 0u) {
                mantissa <<= 1;
                ++shift;
            }
            exponent = uint(1 - shift);
            bits = (sign << 31) | ((exponent + 127u - 15u) << 23) | ((mantissa & 0x3ffu) << 13);
        }
    } else if (exponent == 0x1fu) {
        bits = (sign << 31) | (0xffu << 23) | (mantissa << 13);
    } else {
        bits = (sign << 31) | ((exponent + 127u - 15u) << 23) | (mantissa << 13);
    }
    return as_type<float>(bits);
}

// Round to nearest even with flush to zero, like the RAD file encoder.
static uint rad_float_to_half(float value) {
    const uint bits = as_type<uint>(value);
    const uint sign = (bits >> 31) & 1u;
    const uint exponent = (bits >> 23) & 0xffu;
    const uint mantissa = bits & 0x7fffffu;
    if (exponent == 0u)
        return sign << 15;
    if (exponent == 0xffu)
        return (sign << 15) | 0x7c00u | (mantissa >> 13);
    const int biased = int(exponent) - 127 + 15;
    if (biased >= 31)
        return (sign << 15) | 0x7c00u;
    if (biased <= 0)
        return sign << 15;
    uint half_mantissa = mantissa >> 13;
    if ((mantissa & 0x1fffu) > 0x1000u || ((mantissa & 0x1fffu) == 0x1000u && (half_mantissa & 1u) != 0u))
        ++half_mantissa;
    return (sign << 15) + (uint(biased) << 10) + half_mantissa;
}

static uint rad_pair(float a, float b) { return rad_float_to_half(a) | (rad_float_to_half(b) << 16); }

static float rad_at_least(float v, float lo) { return v < lo ? lo : v; }

static float rad_byte_fraction(uint value, uint denominator) {
    return float(value) * as_type<float>(denominator == 255u ? 0x3b808081u : 0x3c010204u);
}

// The comparisons of the shared formulas, so NaN ranges resolve the same way.
static float rad_max_abs(RadPackedProperty p) {
    float m = abs(p.min_val);
    if (abs(p.max_val) > m)
        m = abs(p.max_val);
    if (abs(p.scale) > m)
        m = abs(p.scale);
    return m > 1e-6f ? m : 1e-6f;
}

static float rad_rotation_reciprocal(float value) {
    float reciprocal = 1.0f / value;
    reciprocal = fma(fma(-value, reciprocal, 1.0f), reciprocal, reciprocal);
    const float error = fma(-value, reciprocal, 1.0f);
    if (error == 0.0f)
        return reciprocal;
    const uint bits = as_type<uint>(reciprocal);
    const float adjacent = as_type<float>(error > 0.0f ? bits + 1u : bits - 1u);
    const float midpoint = value * (abs(adjacent - reciprocal) * 0.5f);
    return abs(error) > midpoint || (abs(error) == midpoint && (bits & 1u) != 0u) ? adjacent : reciprocal;
}

static float4 rad_quat_oct88(uint b0, uint b1, uint b2) {
    float oct_x = rad_byte_fraction(b0, 255u) * 2.0f - 1.0f;
    float oct_y = rad_byte_fraction(b1, 255u) * 2.0f - 1.0f;
    const float oct_z = 1.0f - abs(oct_x) - abs(oct_y);
    if (oct_z < 0.0f) {
        const float x = oct_x;
        oct_x = (1.0f - abs(oct_y)) * (oct_x >= 0.0f ? 1.0f : -1.0f);
        oct_y = (1.0f - abs(x)) * (oct_y >= 0.0f ? 1.0f : -1.0f);
    }
    float3 axis = float3(oct_x, oct_y, oct_z);
    const float length = sqrt(axis.x * axis.x + axis.y * axis.y + axis.z * axis.z);
    if (length > 0.0f)
        axis *= rad_rotation_reciprocal(length);
    const float half_theta = rad_byte_fraction(b2, 255u) * M_PI_F * 0.5f;
    float4 q = float4(axis * sin(half_theta), cos(half_theta));
    const float norm = sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    if (norm > 0.0f)
        q *= rad_rotation_reciprocal(norm);
    if ((rad_float_to_half(q.x) & 0x7fffu) == 0u)
        q.x = as_type<float>(b0 < 128u ? 0x80000000u : 0u);
    if ((rad_float_to_half(q.y) & 0x7fffu) == 0u)
        q.y = as_type<float>(b1 < 128u ? 0x80000000u : 0u);
    return q;
}

static uint rad_property(device const RadPagePackedDesc& d, uint kind) {
    for (uint p = 0; p < d.property_count; ++p) {
        if (d.props[p].kind == kind)
            return p;
    }
    return 8;
}

static float rad_plane(constant RadPageParams& p, device const RadPagePackedDesc& d, uint prop, uint dims,
                       uint component, uint i) {
    if (prop == 8 || i >= d.count)
        return 0.0f;
    const RadPackedProperty a = d.props[prop];
    device const uchar* base = p.packed + a.plane_offset;
    const uint e = component * d.count + i, stride = d.count * dims;
    switch (a.encoding) {
    case 0: return ((device const float*)base)[e];
    case 1: return as_type<float>(uint(base[e]) | uint(base[stride + e]) << 8 | uint(base[2 * stride + e]) << 16 |
                                  uint(base[3 * stride + e]) << 24);
    case 2: return rad_half_to_float(((device const ushort*)base)[e]);
    case 3: return rad_half_to_float(uint(base[e]) | uint(base[stride + e]) << 8);
    case 4: return fma(rad_byte_fraction(base[e], 255u), a.max_val - a.min_val, a.min_val);
    case 5: {
        const int v = base[e] >= 128 ? int(base[e]) - 256 : int(base[e]);
        return (v < 0 ? -rad_byte_fraction(uint(-v), 127u) : rad_byte_fraction(uint(v), 127u)) * rad_max_abs(a);
    }
    case 6: return base[e] == 0 ? 0.0f : exp(a.min_val + float(base[e] - 1) * (a.max_val - a.min_val) / 254.0f);
    case 7: return exp(rad_half_to_float(((device const ushort*)base)[e]));
    default: return 0.0f;
    }
}

static uint rad_quantize(float v, float max_abs) { return uint(int(clamp(rint(v / max_abs * 127.0f), -127.0f, 127.0f))) & 255u; }

static ulong rad_sh_index(constant RadPageParams& p, uint i, uint slot) {
    return ((ulong(p.page) * p.page_splats + i) / 32 * p.slots + slot) * 32 + i % 32;
}

static float rad_resident_sh(constant RadSources& s, uint i, uint c) {
    const uint splat = s.offset + i;
    if (s.quant_sh != 0) {
        const float lo = s.sh_bounds[splat / 256 * 2], hi = s.sh_bounds[splat / 256 * 2 + 1];
        const uint cell = splat / 32 * (s.rest * 3 * 32) + c * 32 + splat % 32;
        return lo + (hi - lo) * (float(((device const ushort*)s.shN)[cell]) * (1.0f / 65535.0f));
    }
    const uint index = ((splat / 32 * ((s.rest * 3 + 3) / 4) + c / 4) * 32 + splat % 32) * 4 + c % 4;
    return s.half_sh != 0 ? rad_half_to_float(((device const ushort*)s.shN)[index])
                          : ((device const float*)s.shN)[index];
}

static uint rad_band(uint c) { return c < 9 ? 0 : (c < 24 ? 1 : 2); }

kernel void rad_page(constant RadPageParams& p [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.page_splats)
        return;
    const ulong dst = ulong(p.page) * p.page_splats + i;
    device uint* frame = p.frames + ulong(p.page) * 16;
    if (kOp != 0) {
        constant RadSources& s = p.sources;
        if (kOp == 1) {
            if (i < 4)
                frame[i] = 0;
            return;
        }
        if (kOp == 2) {
            if (i >= s.count || !s.shN)
                return;
            float maxima[3] = {0, 0, 0};
            for (uint c = 0; c < s.rest * 3; ++c)
                maxima[rad_band(c)] = max(maxima[rad_band(c)], abs(rad_resident_sh(s, i, c)));
            for (uint b = 0; b < 3; ++b)
                atomic_fetch_max_explicit((device atomic_uint*)frame + b, as_type<uint>(maxima[b]),
                                          memory_order_relaxed);
            return;
        }
        const bool live = i < s.count;
        const uint row = s.offset + i;
        for (uint c = 0; c < 3; ++c)
            p.means[dst * 3 + c] = live ? as_type<uint>(s.means[row * 3 + c]) : 0;
        const float3 rgb = live ? float3(s.sh0[row * 3], s.sh0[row * 3 + 1], s.sh0[row * 3 + 2]) : 0;
        const float3 scale = live ? float3(s.scaling[row * 3], s.scaling[row * 3 + 1], s.scaling[row * 3 + 2]) : 0;
        p.sh0[dst * 2] = rad_pair(rgb.x, rgb.y);
        p.sh0[dst * 2 + 1] = rad_pair(rgb.z, 0);
        p.scaling[dst * 2] = rad_pair(scale.x, scale.y);
        p.scaling[dst * 2 + 1] = rad_pair(scale.z, 0);
        float4 q = float4(1, 0, 0, 0);
        if (live)
            q = float4(s.rotation[row * 4], s.rotation[row * 4 + 1], s.rotation[row * 4 + 2], s.rotation[row * 4 + 3]);
        p.rotation[dst * 2] = rad_pair(q.x, q.y);
        p.rotation[dst * 2 + 1] = rad_pair(q.z, q.w);
        // One thread owns a whole opacity word, odd tails included.
        if ((i & 1) == 0)
            p.opacity[dst / 2] = rad_pair(live ? s.opacity[row] : 0, i + 1 < s.count ? s.opacity[row + 1] : 0);
        for (uint slot = 0; slot < p.slots; ++slot) {
            uint packed = 0;
            for (uint b = 0; b < 4; ++b) {
                const uint c = slot * 4 + b;
                if (live && s.shN && c < s.rest * 3)
                    packed |= rad_quantize(rad_resident_sh(s, i, c), max(as_type<float>(frame[rad_band(c)]), 1e-6f))
                              << (b * 8);
            }
            p.shN[rad_sh_index(p, i, slot)] = packed;
        }
        return;
    }
    device const RadPagePackedDesc& d = *p.descriptor;
    const bool live = i < d.count;
    const uint xyz = rad_property(d, 0), alpha = rad_property(d, 1), rgb = rad_property(d, 2);
    const uint scales = rad_property(d, 3), rot = rad_property(d, 4);
    if (i == 0) {
        for (uint b = 0; b < 16; ++b)
            frame[b] = 0;
        for (uint b = 0; b < 3; ++b) {
            const uint prop = rad_property(d, 5 + b);
            frame[b] = as_type<uint>(prop < 8 ? rad_max_abs(d.props[prop]) : 0.0f);
            frame[4 + b] = as_type<uint>(d.bbox_min[b]);
            frame[8 + b] = as_type<uint>(d.bbox_extent[b]);
        }
        frame[7] = as_type<uint>(d.log_size_min);
        frame[11] = as_type<uint>(d.log_size_range);
    }
    for (uint c = 0; c < 3; ++c)
        p.means[dst * 3 + c] = as_type<uint>(rad_plane(p, d, xyz, 3, c, i));
    float sh[3] = {0, 0, 0};
    uint sc[3] = {0, 0, 0};
    if (live) {
        for (uint c = 0; c < 3; ++c) {
            if (rgb < 8)
                sh[c] = (rad_plane(p, d, rgb, 3, c, i) - 0.5f) / 0.28209479177387814f;
            if (scales < 8)
                sc[c] = d.props[scales].encoding == 7
                            ? ((device const ushort*)(p.packed + d.props[scales].plane_offset))[c * d.count + i]
                            : rad_float_to_half(log(rad_at_least(rad_plane(p, d, scales, 3, c, i), 1.0e-8f)));
        }
    }
    p.sh0[dst * 2] = rad_pair(sh[0], sh[1]);
    p.sh0[dst * 2 + 1] = rad_pair(sh[2], 0);
    p.scaling[dst * 2] = sc[0] | (sc[1] << 16);
    p.scaling[dst * 2 + 1] = sc[2];
    if ((i & 1) == 0) {
        uint packed = 0;
        for (uint b = 0; b < 2; ++b) {
            float v = 0;
            if (i + b < d.count && alpha < 8) {
                const float raw = rad_plane(p, d, alpha, 1, 0, i + b);
                if (d.lod_opacity != 0) {
                    v = rad_at_least(raw, 0.0f);
                } else {
                    const float a = raw > 1.0f - 1.0e-6f ? 1.0f - 1.0e-6f : rad_at_least(raw, 1.0e-6f);
                    v = log(a / (1.0f - a));
                }
            }
            packed |= rad_float_to_half(v) << (b * 16);
        }
        p.opacity[dst / 2] = packed;
    }
    float4 q = float4(0, 0, 0, 1);
    if (live && rot < 8) {
        if (d.props[rot].encoding == 8) {
            device const uchar* base = p.packed + d.props[rot].plane_offset;
            q = rad_quat_oct88(base[i * 3], base[i * 3 + 1], base[i * 3 + 2]);
        } else {
            q.xyz = float3(rad_plane(p, d, rot, 3, 0, i), rad_plane(p, d, rot, 3, 1, i), rad_plane(p, d, rot, 3, 2, i));
            const float w2 = 1.0f - q.x * q.x - q.y * q.y - q.z * q.z;
            q.w = sqrt(w2 > 0.0f ? w2 : 0.0f);
        }
    }
    p.rotation[dst * 2] = rad_pair(q.w, q.x);
    p.rotation[dst * 2 + 1] = rad_pair(q.y, q.z);
    for (uint slot = 0; slot < p.slots; ++slot) {
        uint packed = 0;
        for (uint b = 0; b < 4; ++b) {
            const uint c = slot * 4 + b;
            if (!live || c >= d.sh_coeffs_rest * 3)
                continue;
            const uint band = rad_band(c), local = c - (band == 0 ? 0 : (band == 1 ? 9 : 24));
            const uint prop = rad_property(d, 5 + band);
            if (prop < 8) {
                const RadPackedProperty a = d.props[prop];
                const uint v = a.encoding == 5 ? uint(p.packed[a.plane_offset + local * d.count + i])
                                               : rad_quantize(rad_plane(p, d, prop, 9 + band * 6, local, i), rad_max_abs(a));
                packed |= v << (b * 8);
            }
        }
        p.shN[rad_sh_index(p, i, slot)] = packed;
    }
    if (p.bounds && p.links) {
        device const uint* bounds = (device const uint*)(p.packed + d.meta_bounds_offset);
        device const uint* links = (device const uint*)(p.packed + d.meta_links_offset);
        for (uint c = 0; c < 2; ++c)
            p.bounds[dst * 2 + c] = i < d.meta_node_count ? bounds[i * 2 + c] : 0;
        for (uint c = 0; c < 3; ++c)
            p.links[dst * 3 + c] = i < d.meta_node_count ? links[i * 3 + c] : 0xffffffffu;
    }
}

// ---------------------------------------------------------------------------
// Kernels of the portable neural-network ops, ported from inference.slang.
// kOp is the InferenceKernel: im2col, col2im, resize, pool, activation, grid.

struct InferenceGeometry {
    int channels, height, width, out_height, out_width;
    int kernel_h, kernel_w, stride_h, stride_w;
    int pad_h, pad_w, dilation_h, dilation_w;
    int offset, columns, mode, coord, include_pad;
    float u0, u1, v0, v1;
};

// Operands are Float32 or Float16 (kInputDType, kOutputDType); the math is FP32.
struct InferenceParams {
    device const uchar* input;
    device uchar* output;
    uint total, step;
    InferenceGeometry p;
};

static float inference_sample(constant InferenceParams& params, int plane, int y, int x) {
    constant InferenceGeometry& p = params.p;
    return load_float(params.input, (plane * p.height + clamp(y, 0, p.height - 1)) * p.width + clamp(x, 0, p.width - 1));
}

static float resize_coordinate(int i, int in_size, int out_size, int mode) {
    if (out_size == 1)
        return 0.0f;
    if (mode == 2)
        return float(i) * (in_size - 1) / (out_size - 1);
    if (mode == 1)
        return float(i) * in_size / out_size;
    return (float(i) + 0.5f) * in_size / out_size - 0.5f;
}

static float cubic_weight(float x) {
    x = abs(x);
    if (x <= 1.0f)
        return ((1.25f * x - 2.25f) * x) * x + 1.0f;
    if (x < 2.0f)
        return ((-0.75f * x + 3.75f) * x - 6.0f) * x + 3.0f;
    return 0.0f;
}

// The erf approximation the Vulkan kernel evaluates in fp32.
static float erf_approximation(float x) {
    const float a = abs(x), t = 1.0f / (1.0f + 0.3275911f * a);
    const float r =
        1.0f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t *
                   exp(-a * a);
    return x < 0.0f ? -r : r;
}

// An nn::Activation of x, as the Vulkan kernels evaluate it in fp32.
static float nn_activation(float x, int activation) {
    if (activation == 1)
        return max(x, 0.0f);
    if (activation == 2)
        return 0.5f * x * (1.0f + tanh(0.7978845608028654f * (x + 0.044715f * x * x * x)));
    if (activation == 3)
        return 0.5f * x * (1.0f + erf_approximation(x * 0.7071067811865475f));
    if (activation == 4)
        return x / (1.0f + exp(-x));
    return x;
}

kernel void inference(constant InferenceParams& params [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i >= params.total)
        return;
    constant InferenceGeometry& p = params.p;
    const int index = int(i);
    float value = 0.0f;
    if (kOp == 0) {
        const int column = index % p.columns + p.offset, tap = index / p.columns;
        const int x = (column % p.out_width) * p.stride_w - p.pad_w + (tap % p.kernel_w) * p.dilation_w;
        const int y = (column / p.out_width) * p.stride_h - p.pad_h + ((tap / p.kernel_w) % p.kernel_h) * p.dilation_h;
        const int c = tap / (p.kernel_h * p.kernel_w);
        if (p.mode == 1 || (x >= 0 && x < p.width && y >= 0 && y < p.height))
            value = inference_sample(params, c, y, x);
    } else if (kOp == 1) {
        const int x = index % p.out_width, y = (index / p.out_width) % p.out_height;
        const int c = index / (p.out_width * p.out_height);
        for (int ky = 0; ky < p.kernel_h; ++ky) {
            for (int kx = 0; kx < p.kernel_w; ++kx) {
                int iy = y + p.pad_h - ky * p.dilation_h, ix = x + p.pad_w - kx * p.dilation_w;
                if (iy < 0 || ix < 0 || iy % p.stride_h != 0 || ix % p.stride_w != 0)
                    continue;
                iy /= p.stride_h;
                ix /= p.stride_w;
                if (iy < p.height && ix < p.width)
                    value += load_float(params.input, ((c * p.kernel_h + ky) * p.kernel_w + kx) * p.height * p.width +
                                                          iy * p.width + ix);
            }
        }
    } else if (kOp == 2) {
        const int x = index % p.out_width, y = (index / p.out_width) % p.out_height;
        const int plane = index / (p.out_width * p.out_height);
        const float fx = resize_coordinate(x, p.width, p.out_width, p.coord);
        const float fy = resize_coordinate(y, p.height, p.out_height, p.coord);
        const int ix = int(floor(fx)), iy = int(floor(fy));
        if (p.mode == 0) {
            value = inference_sample(params, plane, iy, ix);
        } else if (p.mode == 1) {
            const float dx = fx - ix, dy = fy - iy;
            const float a = inference_sample(params, plane, iy, ix) * (1.0f - dx) + inference_sample(params, plane, iy, ix + 1) * dx;
            const float b =
                inference_sample(params, plane, iy + 1, ix) * (1.0f - dx) + inference_sample(params, plane, iy + 1, ix + 1) * dx;
            value = a * (1.0f - dy) + b * dy;
        } else {
            for (int yy = -1; yy <= 2; ++yy) {
                for (int xx = -1; xx <= 2; ++xx)
                    value += inference_sample(params, plane, iy + yy, ix + xx) * cubic_weight(fy - (iy + yy)) *
                             cubic_weight(fx - (ix + xx));
            }
        }
    } else if (kOp == 3) {
        const int x = (index % p.out_width) * p.stride_w - p.pad_w;
        const int y = ((index / p.out_width) % p.out_height) * p.stride_h - p.pad_h;
        const int plane = index / (p.out_width * p.out_height);
        value = p.mode == 0 ? -INFINITY : 0.0f;
        int count = 0;
        for (int ky = 0; ky < p.kernel_h; ++ky) {
            for (int kx = 0; kx < p.kernel_w; ++kx) {
                if (y + ky >= 0 && y + ky < p.height && x + kx >= 0 && x + kx < p.width) {
                    const float v = inference_sample(params, plane, y + ky, x + kx);
                    value = p.mode == 0 ? max(value, v) : value + v;
                    ++count;
                }
            }
        }
        if (p.mode != 0)
            value /= max(1, p.include_pad != 0 ? p.kernel_h * p.kernel_w : count);
    } else if (kOp == 4) {
        value = nn_activation(load_float(params.input, index), p.mode);
    } else {
        const int pixel = index % (p.height * p.width);
        if (index < p.height * p.width)
            value = p.width == 1 ? p.u0 : p.u0 + (p.u1 - p.u0) * (pixel % p.width) / (p.width - 1);
        else
            value = p.height == 1 ? p.v0 : p.v0 + (p.v1 - p.v0) * (pixel / p.width) / (p.height - 1);
    }
    store_float(params.output, i, value);
}

// ---------------------------------------------------------------------------
// Float32 GEMM on the matrix units through Metal Performance Primitives:
// C[m][n] = A[m][k] * B, with B stored as [k][n] or, with kTransposeB, as
// [n][k]; batches are packed. kBiasRelu applies max(value + bias[row], 0) to
// the tile in registers before it is stored. A threadgroup of four SIMD
// groups computes one 32x32 tile, or 64x64 with kWideTile, and the matmul
// checks the matrix edges.

struct GemmParams {
    ulong lhs_offset;
    ulong rhs_offset;
    ulong output_offset;
    ulong bias_offset;
    ulong lhs_stride;
    ulong rhs_stride;
    ulong output_stride;
    uint m;
    uint n;
    uint k;
    uint padding;
};

using Matrix = tensor<device float, dextents<int32_t, 2>, tensor_inline>;

template <bool TransposeB, int Tile>
static void gemm_tile(device float* lhs, device float* rhs, device float* output, device const float* bias,
                      constant GemmParams& params, uint2 group) {
    const int m = int(params.m), n = int(params.n), k = int(params.k);
    // Extents list the innermost dimension first.
    Matrix a(lhs, dextents<int32_t, 2>(k, m));
    Matrix b(rhs, TransposeB ? dextents<int32_t, 2>(k, n) : dextents<int32_t, 2>(n, k));
    Matrix c(output, dextents<int32_t, 2>(n, m));
    constexpr auto descriptor =
        mpp::tensor_ops::matmul2d_descriptor(Tile, Tile, static_cast<int>(dynamic_extent), false, TransposeB);
    mpp::tensor_ops::matmul2d<descriptor, execution_simdgroups<4>> matmul;
    const int row = int(group.y) * Tile, column = int(group.x) * Tile;
    auto a_tile = a.slice(0, row);
    auto b_tile = TransposeB ? b.slice(0, column) : b.slice(column, 0);
    auto c_tile = c.slice(column, row);
    if (kBiasRelu == 0) {
        matmul.run(a_tile, b_tile, c_tile);
        return;
    }
    auto result = matmul.template get_destination_cooperative_tensor<decltype(a_tile), decltype(b_tile), float>();
    for (uint16_t i = 0; i < result.get_capacity(); ++i) {
        if (result.is_valid_element(i))
            result[i] = 0.0f;
    }
    matmul.run(a_tile, b_tile, result);
    for (uint16_t i = 0; i < result.get_capacity(); ++i) {
        const int element_row = row + result.get_multidimensional_index(i)[1];
        if (result.is_valid_element(i) && element_row < m) {
            const float value = result[i] + bias[element_row];
            result[i] = value > 0.0f ? value : 0.0f;
        }
    }
    result.store(c_tile);
}

kernel void gemm(device const uchar* lhs_buffer [[buffer(0)]],
                 device const uchar* rhs_buffer [[buffer(1)]],
                 device uchar* output_buffer [[buffer(2)]],
                 device const uchar* bias_buffer [[buffer(3)]],
                 constant GemmParams& params [[buffer(4)]],
                 uint3 group [[threadgroup_position_in_grid]]) {
    device float* lhs = (device float*)(lhs_buffer + params.lhs_offset) + group.z * params.lhs_stride;
    device float* rhs = (device float*)(rhs_buffer + params.rhs_offset) + group.z * params.rhs_stride;
    device float* output = (device float*)(output_buffer + params.output_offset) + group.z * params.output_stride;
    device const float* bias = kBiasRelu != 0 ? (device const float*)(bias_buffer + params.bias_offset) : nullptr;
    if (kTransposeB != 0 && kWideTile != 0)
        gemm_tile<true, 64>(lhs, rhs, output, bias, params, group.xy);
    else if (kTransposeB != 0)
        gemm_tile<true, 32>(lhs, rhs, output, bias, params, group.xy);
    else if (kWideTile != 0)
        gemm_tile<false, 64>(lhs, rhs, output, bias, params, group.xy);
    else
        gemm_tile<false, 32>(lhs, rhs, output, bias, params, group.xy);
}

// ---------------------------------------------------------------------------
// Neural-network helpers, ported from nn.slang. kOp picks the kind:
// 0 max_pool2d and 1 adaptive_avg_pool2d over [N, C, H, W], 2 bias_add and
// 3 bias_relu with the bias indexed by (index / spatial) % channels, 4 relu.

struct NnParams {
    ulong input_offset;
    ulong bias_offset;
    ulong output_offset;
    uint total;
    uint channels;
    uint input_height;
    uint input_width;
    uint output_height;
    uint output_width;
    uint window;
    uint stride;
    int padding;
    uint spatial;
};

kernel void nn(device const uchar* input_buffer [[buffer(0)]],
               device const uchar* bias_buffer [[buffer(1)]],
               device uchar* output_buffer [[buffer(2)]],
               constant NnParams& params [[buffer(3)]],
               uint index [[thread_position_in_grid]]) {
    if (index >= params.total)
        return;
    device const float* input = (device const float*)(input_buffer + params.input_offset);
    device float* output = (device float*)(output_buffer + params.output_offset);
    if (kOp == 4) {
        const float value = input[index];
        output[index] = value > 0.0f ? value : 0.0f;
        return;
    }
    if (kOp == 2 || kOp == 3) {
        device const float* bias = (device const float*)(bias_buffer + params.bias_offset);
        const float value = input[index] + bias[(index / params.spatial) % params.channels];
        output[index] = kOp == 3 && !(value > 0.0f) ? 0.0f : value;
        return;
    }
    const uint w_out = index % params.output_width;
    const uint h_out = (index / params.output_width) % params.output_height;
    const uint plane_index = index / (params.output_width * params.output_height);
    device const float* plane = input + ulong(plane_index) * params.input_height * params.input_width;
    if (kOp == 0) {
        const int h_start = int(h_out * params.stride) - params.padding;
        const int w_start = int(w_out * params.stride) - params.padding;
        float best = -INFINITY;
        for (uint kh = 0; kh < params.window; ++kh) {
            const int h_in = h_start + int(kh);
            if (h_in < 0 || h_in >= int(params.input_height))
                continue;
            for (uint kw = 0; kw < params.window; ++kw) {
                const int w_in = w_start + int(kw);
                if (w_in >= 0 && w_in < int(params.input_width))
                    best = ieee_maximum(best, plane[ulong(h_in) * params.input_width + uint(w_in)]);
            }
        }
        output[index] = best;
        return;
    }
    const uint h_begin = h_out * params.input_height / params.output_height;
    const uint h_end = ((h_out + 1) * params.input_height + params.output_height - 1) / params.output_height;
    const uint w_begin = w_out * params.input_width / params.output_width;
    const uint w_end = ((w_out + 1) * params.input_width + params.output_width - 1) / params.output_width;
    float sum = 0.0f;
    uint count = 0;
    for (uint h = h_begin; h < h_end; ++h) {
        for (uint w = w_begin; w < w_end; ++w) {
            sum += plane[ulong(h) * params.input_width + w];
            ++count;
        }
    }
    output[index] = count > 0 ? sum / float(count) : 0.0f;
}

// ---------------------------------------------------------------------------
// eye (kOp 0) and diag (kOp 1): a [rows][columns] Float32 matrix that is zero
// off the diagonal and one, or the diagonal vector's element, on it.

struct MatrixFillParams {
    ulong diagonal_offset;
    ulong output_offset;
    uint columns;
    uint count;
};

kernel void matrix_fill(device const uchar* diagonal_buffer [[buffer(0)]],
                        device uchar* output_buffer [[buffer(1)]],
                        constant MatrixFillParams& params [[buffer(2)]],
                        uint index [[thread_position_in_grid]]) {
    if (index >= params.count)
        return;
    const uint row = index / params.columns;
    float value = 0.0f;
    if (row == index - row * params.columns)
        value = kOp == 0 ? 1.0f : ((device const float*)(diagonal_buffer + params.diagonal_offset))[row];
    ((device float*)(output_buffer + params.output_offset))[index] = value;
}

// ---------------------------------------------------------------------------
// out[i][j] = p-norm distance between row i of a and row j of b, with the
// p == 0 (count of differing features) and p == infinity (largest absolute
// difference) conventions of cdist.slang.

struct CdistParams {
    ulong lhs_offset;
    ulong rhs_offset;
    ulong output_offset;
    uint rows;
    uint columns;
    uint features;
    float p;
};

kernel void cdist(device const uchar* lhs_buffer [[buffer(0)]],
                  device const uchar* rhs_buffer [[buffer(1)]],
                  device uchar* output_buffer [[buffer(2)]],
                  constant CdistParams& params [[buffer(3)]],
                  uint index [[thread_position_in_grid]]) {
    if (index >= params.rows * params.columns)
        return;
    const uint i = index / params.columns;
    const uint j = index - i * params.columns;
    device const float* a = (device const float*)(lhs_buffer + params.lhs_offset) + ulong(i) * params.features;
    device const float* b = (device const float*)(rhs_buffer + params.rhs_offset) + ulong(j) * params.features;
    const float p = params.p;
    float distance = 0.0f;
    for (uint d = 0; d < params.features; ++d) {
        const float difference = a[d] - b[d];
        if (p == 2.0f)
            distance += difference * difference;
        else if (p == 1.0f)
            distance += abs(difference);
        else if (p == 0.0f)
            distance += difference != 0.0f ? 1.0f : 0.0f;
        else if (isinf(p))
            distance = max(distance, abs(difference));
        else
            distance += pow(abs(difference), p);
    }
    if (p == 2.0f)
        distance = sqrt(distance);
    else if (p != 1.0f && p != 0.0f && !isinf(p))
        distance = pow(distance, 1.0f / p);
    ((device float*)(output_buffer + params.output_offset))[index] = distance;
}
