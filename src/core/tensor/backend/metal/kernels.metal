// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later

// Tensor kernels of the Metal backend, ported from the Vulkan Slang shaders so
// both backends compute identical results. The host compiles this source at
// runtime with safe math and precise functions and prepends the LFS_OP_*,
// LFS_DT_* and LFS_REDUCE_* ids generated from the C++ enums. Operands are
// bound whole and addressed by byte offsets, so any element offset is legal.

#include <metal_stdlib>
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
                           constant PointwiseParams& params, uint index) {
    if (kInputDType == LFS_DT_Float32 || kInputDType == LFS_DT_Float16) {
        const float a = load_float(lhs, index);
        const float b = kArity == 2 ? load_float(rhs, index) : params.scalar_float;
        return float_predicate(a, b) ? 1 : 0;
    }
    if (kInputDType == LFS_DT_Int64) {
        const long a = ((device const long*)lhs)[index];
        const long b = ((device const long*)rhs)[index];
        return int64_predicate(a, b) ? 1 : 0;
    }
    if (kInputDType == LFS_DT_UInt32)
        return uint_predicate(((device const uint*)lhs)[index], ((device const uint*)rhs)[index]) ? 1 : 0;
    const int a = kInputDType == LFS_DT_Int32 ? ((device const int*)lhs)[index] : int(lhs[index]);
    const int b = kArity != 2 ? int(params.scalar_int64)
                  : kInputDType == LFS_DT_Int32 ? ((device const int*)rhs)[index]
                                                : int(rhs[index]);
    return kOutputDType == LFS_DT_Bool ? (int_predicate(a, b) ? 1 : 0) : uchar(uint(int_binary(a, b)) & 255u);
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
    if (index >= params.count)
        return;
    if (kOutputDType == LFS_DT_UInt8 || kOutputDType == LFS_DT_Bool) {
        output[index] = evaluate_byte(lhs, rhs, params, index);
        return;
    }
    const bool scalar_on_right = (params.flags & 1u) != 0u;
    if (kInputDType == LFS_DT_Float32 || kInputDType == LFS_DT_Float16) {
        const float a = load_float(lhs, index);
        const float b = kArity == 2 ? load_float(rhs, index) : params.scalar_float;
        store_float(output, index, kArity == 2 ? float_binary(a, b) : float_unary(a, b, scalar_on_right));
    } else if (kInputDType == LFS_DT_Int32) {
        const int a = ((device const int*)lhs)[index];
        const int b = kArity == 2 ? ((device const int*)rhs)[index] : int(params.scalar_int64);
        ((device int*)output)[index] = kArity == 2 ? int_binary(a, b) : int_unary(a, b, scalar_on_right);
    } else if (kInputDType == LFS_DT_Int64) {
        ((device long*)output)[index] = int64_binary(((device const long*)lhs)[index], ((device const long*)rhs)[index]);
    } else if (kInputDType == LFS_DT_UInt32) {
        ((device uint*)output)[index] = uint_binary(((device const uint*)lhs)[index], ((device const uint*)rhs)[index]);
    }
}

// ---------------------------------------------------------------------------
// Lazily fused Float32 chains. Tensor operands of ops 4..7 are bound to rhs
// slots 0..15 and addressed by the element's index.

struct ChainOp {
    uint kind;
    float scalar;
    uint slot;
    uint padding;
    ulong rhs_offset;
};

struct ChainParams {
    ulong input_offset;
    ulong output_offset;
    uint count;
    uint op_count;
    ChainOp ops[16];
};

kernel void pointwise_chain(device const uchar* input_buffer [[buffer(0)]],
                            device uchar* output_buffer [[buffer(1)]],
                            constant ChainParams& params [[buffer(2)]],
                            device const uchar* rhs0 [[buffer(3)]], device const uchar* rhs1 [[buffer(4)]],
                            device const uchar* rhs2 [[buffer(5)]], device const uchar* rhs3 [[buffer(6)]],
                            device const uchar* rhs4 [[buffer(7)]], device const uchar* rhs5 [[buffer(8)]],
                            device const uchar* rhs6 [[buffer(9)]], device const uchar* rhs7 [[buffer(10)]],
                            device const uchar* rhs8 [[buffer(11)]], device const uchar* rhs9 [[buffer(12)]],
                            device const uchar* rhs10 [[buffer(13)]], device const uchar* rhs11 [[buffer(14)]],
                            device const uchar* rhs12 [[buffer(15)]], device const uchar* rhs13 [[buffer(16)]],
                            device const uchar* rhs14 [[buffer(17)]], device const uchar* rhs15 [[buffer(18)]],
                            uint index [[thread_position_in_grid]]) {
    if (index >= params.count)
        return;
    const device uchar* const slots[16] = {rhs0, rhs1, rhs2, rhs3, rhs4, rhs5, rhs6, rhs7,
                                           rhs8, rhs9, rhs10, rhs11, rhs12, rhs13, rhs14, rhs15};
    float value = ((device const float*)(input_buffer + params.input_offset))[index];
    for (uint i = 0; i < params.op_count; ++i) {
        const ChainOp op = params.ops[i];
        if (op.kind <= 3u) {
            value = op.kind == 0u ? value + op.scalar
                    : op.kind == 1u ? value - op.scalar
                    : op.kind == 2u ? value * op.scalar
                                    : value / op.scalar;
        } else if (op.kind <= 7u) {
            const float rhs = ((device const float*)(slots[op.slot] + op.rhs_offset))[index];
            value = op.kind == 4u ? value + rhs
                    : op.kind == 5u ? value - rhs
                    : op.kind == 6u ? value * rhs
                                    : value / rhs;
        } else if (op.kind == 10u) value = abs(value);
        else if (op.kind == 11u) value = -value;
        else if (op.kind == 12u) value = exp(value);
        else if (op.kind == 13u) value = log(value);
        else if (op.kind == 14u) value = sqrt(value);
        else if (op.kind == 15u) value = 1.0f / (1.0f + exp(-value));
        else if (op.kind == 16u) value = isnan(value) ? value : ieee_maximum(value, 0.0f);
        else if (op.kind == 17u) value = value * value;
        else if (op.kind == 18u) value = tanh(value);
        else if (op.kind == 19u) value = rsqrt(value);
        else if (op.kind == 20u) value = float(int(value > 0.0f) - int(value < 0.0f));
        else if (op.kind == 21u) value = 1.0f / value;
        else if (op.kind == 22u) value = floor(value);
        else if (op.kind == 23u) value = ceil(value);
        else if (op.kind == 24u) value = ieee_round(value);
    }
    ((device float*)(output_buffer + params.output_offset))[index] = value;
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
// 16-byte form replicates the low 32 bits of the pattern.

struct FillParams {
    ulong output_offset;
    ulong pattern;
    ulong count;
};

kernel void fill(device uchar* output_buffer [[buffer(0)]],
                 constant FillParams& params [[buffer(1)]],
                 uint index [[thread_position_in_grid]]) {
    if (index >= params.count)
        return;
    device uchar* output = output_buffer + params.output_offset;
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
// selection. Elements move as kElementSize bytes; the one converting form is
// the Int32 to Float32 scatter.

static void copy_element(device const uchar* source, ulong source_index,
                         device uchar* destination, ulong destination_index) {
    if (kInputDType != kOutputDType)
        ((device float*)destination)[destination_index] = float(((device const int*)source)[source_index]);
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

struct WhereParams {
    ulong condition_offset;
    ulong x_offset;
    ulong y_offset;
    ulong output_offset;
    uint condition_dims[8];
    uint x_dims[8];
    uint y_dims[8];
    uint output_dims[8];
    uint condition_rank;
    uint x_rank;
    uint y_rank;
    uint output_rank;
    uint count;
    uint padding;
};

// Right-aligned broadcast index, as in where.slang and the CUDA backend.
static ulong broadcast_index(ulong index, constant uint* source_dims, uint source_rank,
                             constant uint* output_dims, uint output_rank) {
    ulong source_index = 0;
    ulong source_stride = 1;
    for (int axis = int(output_rank) - 1; axis >= 0; --axis) {
        const ulong coordinate = axis == 0 ? index : index % output_dims[axis];
        if (axis != 0)
            index /= output_dims[axis];
        const int source_axis = axis - int(output_rank - source_rank);
        if (source_axis >= 0) {
            const uint extent = source_dims[source_axis];
            source_index += (extent == 1u ? 0 : coordinate) * source_stride;
            source_stride *= extent;
        }
    }
    return source_index;
}

kernel void where_select(device const uchar* condition [[buffer(0)]],
                         device const uchar* x [[buffer(1)]],
                         device const uchar* y [[buffer(2)]],
                         device uchar* output [[buffer(3)]],
                         constant WhereParams& params [[buffer(4)]],
                         uint index [[thread_position_in_grid]]) {
    if (index >= params.count)
        return;
    const bool selected = (condition + params.condition_offset)[broadcast_index(
                              index, params.condition_dims, params.condition_rank,
                              params.output_dims, params.output_rank)] != 0;
    const ulong source_index = selected
        ? broadcast_index(index, params.x_dims, params.x_rank, params.output_dims, params.output_rank)
        : broadcast_index(index, params.y_dims, params.y_rank, params.output_dims, params.output_rank);
    copy_element(selected ? x + params.x_offset : y + params.y_offset, source_index,
                 output + params.output_offset, index);
}

// ---------------------------------------------------------------------------
// Scalar Float32 reductions in two deterministic passes. Sums carry a
// Neumaier compensation through every step, like reduce.slang; max and min
// propagate the first NaN.

struct ReduceParams {
    ulong input_offset;
    uint count;
    uint partial_count;
    float mean_scale;
    uint padding;
};

static float reduce_identity() {
    if (kReduce == LFS_REDUCE_MAX) return -INFINITY;
    if (kReduce == LFS_REDUCE_MIN) return INFINITY;
    return 0.0f;
}

static void combine_value(thread float& accumulator, thread float& compensation, float value) {
    if (kReduce == LFS_REDUCE_MAX) {
        accumulator = ieee_maximum(accumulator, value);
    } else if (kReduce == LFS_REDUCE_MIN) {
        accumulator = ieee_minimum(accumulator, value);
    } else {
        const float total = accumulator + value;
        compensation += abs(accumulator) >= abs(value) ? (accumulator - total) + value
                                                       : (value - total) + accumulator;
        accumulator = total;
    }
}

static void combine_pair(thread float& accumulator, thread float& compensation, float2 pair) {
    if (kReduce == LFS_REDUCE_MAX) {
        accumulator = ieee_maximum(accumulator, pair.x);
    } else if (kReduce == LFS_REDUCE_MIN) {
        accumulator = ieee_minimum(accumulator, pair.x);
    } else {
        const float total = accumulator + pair.x;
        const float carried = total - accumulator;
        const float error = (accumulator - (total - carried)) + (pair.x - carried);
        accumulator = total;
        compensation += error + pair.y;
    }
}

static float2 reduce_threadgroup(float2 pair, threadgroup float2* shared, uint thread_index) {
    shared[thread_index] = pair;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint span = kReduceThreads / 2; span > 0; span /= 2) {
        if (thread_index < span) {
            float accumulator = shared[thread_index].x;
            float compensation = shared[thread_index].y;
            combine_pair(accumulator, compensation, shared[thread_index + span]);
            shared[thread_index] = float2(accumulator, compensation);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return shared[0];
}

kernel void reduce_partial(device const uchar* input_buffer [[buffer(0)]],
                           device float2* partials [[buffer(1)]],
                           constant ReduceParams& params [[buffer(2)]],
                           uint thread_index [[thread_position_in_threadgroup]],
                           uint group [[threadgroup_position_in_grid]],
                           uint groups [[threadgroups_per_grid]]) {
    threadgroup float2 shared[kReduceThreads];
    device const float* input = (device const float*)(input_buffer + params.input_offset);
    float accumulator = reduce_identity();
    float compensation = 0.0f;
    for (uint index = group * kReduceThreads + thread_index; index < params.count; index += groups * kReduceThreads)
        combine_value(accumulator, compensation, input[index]);
    const float2 result = reduce_threadgroup(float2(accumulator, compensation), shared, thread_index);
    if (thread_index == 0)
        partials[group] = result;
}

kernel void reduce_final(device const float2* partials [[buffer(0)]],
                         device float* result [[buffer(1)]],
                         constant ReduceParams& params [[buffer(2)]],
                         uint thread_index [[thread_position_in_threadgroup]]) {
    threadgroup float2 shared[kReduceThreads];
    float accumulator = reduce_identity();
    float compensation = 0.0f;
    for (uint index = thread_index; index < params.partial_count; index += kReduceThreads)
        combine_pair(accumulator, compensation, partials[index]);
    const float2 total = reduce_threadgroup(float2(accumulator, compensation), shared, thread_index);
    if (thread_index == 0) {
        float value = total.x;
        if (kReduce != LFS_REDUCE_MAX && kReduce != LFS_REDUCE_MIN && isfinite(total.y))
            value += total.y;
        if (kReduce == LFS_REDUCE_MEAN)
            value *= params.mean_scale;
        result[0] = value;
    }
}
