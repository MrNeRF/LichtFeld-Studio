// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later

// Tensor kernels of the Metal backend, ported from the Vulkan Slang shaders so
// both backends compute identical results. The host compiles this source at
// runtime with safe math and precise functions and prepends the LFS_OP_*,
// LFS_DT_* and LFS_REDUCE_* ids generated from the C++ enums. Operands are
// bound whole and addressed by byte offsets, so any element offset is legal.

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
// Scalar Float32 reductions, deterministic: each threadgroup folds its share
// and the host folds the partials. Sums carry a Neumaier compensation through
// every step, like reduce.slang; max and min propagate the first NaN.

struct ReduceParams {
    ulong input_offset;
    uint count;
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

// Folds the (value, compensation) pairs of a SIMD group through shuffles.
static float2 reduce_simdgroup(float2 pair) {
    for (ushort offset = 16; offset > 0; offset /= 2) {
        float accumulator = pair.x, compensation = pair.y;
        combine_pair(accumulator, compensation, float2(simd_shuffle_down(pair.x, offset), simd_shuffle_down(pair.y, offset)));
        pair = float2(accumulator, compensation);
    }
    return pair;
}

// Folds a threadgroup's pairs; the result is valid in thread 0.
static float2 reduce_threadgroup(float2 pair, threadgroup float2* shared, ushort lane, ushort simdgroup) {
    const float2 folded = reduce_simdgroup(pair);
    if (lane == 0)
        shared[simdgroup] = folded;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return reduce_simdgroup(simdgroup == 0 && lane < kReduceThreads / 32 ? shared[lane]
                                                                          : float2(reduce_identity(), 0.0f));
}

kernel void reduce_partial(device const uchar* input_buffer [[buffer(0)]],
                           device float2* partials [[buffer(1)]],
                           constant ReduceParams& params [[buffer(2)]],
                           uint thread_index [[thread_position_in_threadgroup]],
                           uint group [[threadgroup_position_in_grid]],
                           uint groups [[threadgroups_per_grid]],
                           ushort lane [[thread_index_in_simdgroup]],
                           ushort simdgroup [[simdgroup_index_in_threadgroup]]) {
    threadgroup float2 shared[kReduceThreads / 32];
    device const float* input = (device const float*)(input_buffer + params.input_offset);
    float accumulator = reduce_identity();
    float compensation = 0.0f;
    for (uint index = group * kReduceThreads + thread_index; index < params.count; index += groups * kReduceThreads)
        combine_value(accumulator, compensation, input[index]);
    const float2 result = reduce_threadgroup(float2(accumulator, compensation), shared, lane, simdgroup);
    if (thread_index == 0)
        partials[group] = result;
}

// Dot products stay on the GPU: per-threadgroup compensated sums of products,
// then one threadgroup folds the partials into the output.

struct DotParams {
    ulong lhs_offset;
    ulong rhs_offset;
    ulong output_offset;
    uint count;
    uint padding;
};

kernel void dot_partial(device const uchar* lhs_buffer [[buffer(0)]],
                        device const uchar* rhs_buffer [[buffer(1)]],
                        device float2* partials [[buffer(2)]],
                        constant DotParams& params [[buffer(3)]],
                        uint thread_index [[thread_position_in_threadgroup]],
                        uint group [[threadgroup_position_in_grid]],
                        uint groups [[threadgroups_per_grid]],
                        ushort lane [[thread_index_in_simdgroup]],
                        ushort simdgroup [[simdgroup_index_in_threadgroup]]) {
    threadgroup float2 shared[kReduceThreads / 32];
    device const float* lhs = (device const float*)(lhs_buffer + params.lhs_offset);
    device const float* rhs = (device const float*)(rhs_buffer + params.rhs_offset);
    float accumulator = 0.0f;
    float compensation = 0.0f;
    for (uint index = group * kReduceThreads + thread_index; index < params.count; index += groups * kReduceThreads)
        combine_value(accumulator, compensation, lhs[index] * rhs[index]);
    const float2 result = reduce_threadgroup(float2(accumulator, compensation), shared, lane, simdgroup);
    if (thread_index == 0)
        partials[group] = result;
}

kernel void fold_pairs(device const float2* partials [[buffer(0)]],
                       device uchar* output_buffer [[buffer(1)]],
                       constant DotParams& params [[buffer(2)]],
                       uint thread_index [[thread_position_in_threadgroup]],
                       ushort lane [[thread_index_in_simdgroup]],
                       ushort simdgroup [[simdgroup_index_in_threadgroup]]) {
    threadgroup float2 shared[kReduceThreads / 32];
    float accumulator = 0.0f;
    float compensation = 0.0f;
    for (uint index = thread_index; index < params.count; index += kReduceThreads)
        combine_pair(accumulator, compensation, partials[index]);
    const float2 total = reduce_threadgroup(float2(accumulator, compensation), shared, lane, simdgroup);
    if (thread_index == 0)
        *(device float*)(output_buffer + params.output_offset) = isfinite(total.y) ? total.x + total.y : total.x;
}

// ---------------------------------------------------------------------------
// Float32 GEMM on the matrix units through Metal Performance Primitives:
// C[m][n] = A[m][k] * B, with B stored as [k][n] or, with kTransposeB, as
// [n][k]; batches are packed. kBiasRelu applies max(value + bias[row], 0) to
// the tile in registers before it is stored. A threadgroup of four SIMD
// groups computes one 32x32 tile, and the matmul checks the matrix edges.

constant int kGemmTile = 32;

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

template <bool TransposeB>
static void gemm_tile(device float* lhs, device float* rhs, device float* output, device const float* bias,
                      constant GemmParams& params, uint2 group) {
    const int m = int(params.m), n = int(params.n), k = int(params.k);
    // Extents list the innermost dimension first.
    Matrix a(lhs, dextents<int32_t, 2>(k, m));
    Matrix b(rhs, TransposeB ? dextents<int32_t, 2>(k, n) : dextents<int32_t, 2>(n, k));
    Matrix c(output, dextents<int32_t, 2>(n, m));
    constexpr auto descriptor = mpp::tensor_ops::matmul2d_descriptor(
        kGemmTile, kGemmTile, static_cast<int>(dynamic_extent), false, TransposeB);
    mpp::tensor_ops::matmul2d<descriptor, execution_simdgroups<4>> matmul;
    const int row = int(group.y) * kGemmTile, column = int(group.x) * kGemmTile;
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
    if (kTransposeB != 0)
        gemm_tile<true>(lhs, rhs, output, bias, params, group.xy);
    else
        gemm_tile<false>(lhs, rhs, output, bias, params, group.xy);
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
