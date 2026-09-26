// ---------------------------------------------------------------------------
// Dedicated neural-network kernels. Operands are Float16 or Float32
// (kInputDType) and every kernel accumulates in FP32.

// Layer norm, or RMS norm without a bias, over rows of `cols` values. A SIMD
// group normalizes one row.
struct NnNormParams {
    device const uchar* input;
    device const uchar* weight;
    device const uchar* bias;
    device uchar* output;
    uint rows, cols;
    float eps;
    uint has_bias;
};

kernel void nn_norm(constant NnNormParams& p [[buffer(0)]], uint group [[threadgroup_position_in_grid]],
                    ushort simd [[simdgroup_index_in_threadgroup]], ushort simds [[simdgroups_per_threadgroup]],
                    ushort lane [[thread_index_in_simdgroup]]) {
    const uint row = group * simds + simd;
    if (row >= p.rows)
        return;
    const uint base = row * p.cols;
    float mean = 0;
    if (p.has_bias != 0) {
        float sum = 0;
        for (uint c = lane; c < p.cols; c += 32)
            sum += load_float(p.input, base + c);
        mean = simd_sum(sum) / float(p.cols);
    }
    float squares = 0;
    for (uint c = lane; c < p.cols; c += 32) {
        const float x = load_float(p.input, base + c) - mean;
        squares = fma(x, x, squares);
    }
    const float deviation = sqrt(simd_sum(squares) / float(p.cols) + p.eps);
    for (uint c = lane; c < p.cols; c += 32) {
        float y = (load_float(p.input, base + c) - mean) / deviation * load_float(p.weight, c);
        if (p.has_bias != 0)
            y += load_float(p.bias, c);
        store_float(p.output, base + c, y);
    }
}

// Linear layers on the matrix units: out[b][m][n] = act(a[b][m] . w + bias[n])
// * scale[n] + residual[b][m][n], with w as [k][n] or, with kTransposeB,
// [n][k]. A threadgroup of four SIMD groups computes one 32x32 tile.
struct NnLinearParams {
    device const uchar* a;
    device const uchar* w;
    device const uchar* bias;
    device const uchar* scale;
    device const uchar* residual;
    device uchar* output;
    ulong a_stride, w_stride, output_stride;
    uint m, n, k;
    int activation;
    uint has_bias, has_scale, has_residual, padding;
};

constant int kNnTile = 32;

template <typename T, bool TransposeW>
static void nn_linear_tile(constant NnLinearParams& p, uint3 group) {
    using Matrix = tensor<device T, dextents<int32_t, 2>, tensor_inline>;
    const int m = int(p.m), n = int(p.n), k = int(p.k);
    // Extents list the innermost dimension first.
    Matrix a((device T*)p.a + group.z * p.a_stride, dextents<int32_t, 2>(k, m));
    Matrix w((device T*)p.w + group.z * p.w_stride, TransposeW ? dextents<int32_t, 2>(k, n) : dextents<int32_t, 2>(n, k));
    constexpr auto descriptor =
        mpp::tensor_ops::matmul2d_descriptor(kNnTile, kNnTile, static_cast<int>(dynamic_extent), false, TransposeW);
    mpp::tensor_ops::matmul2d<descriptor, execution_simdgroups<4>> matmul;
    const int row = int(group.y) * kNnTile, column = int(group.x) * kNnTile;
    auto a_tile = a.slice(0, row);
    auto w_tile = TransposeW ? w.slice(0, column) : w.slice(column, 0);
    auto result = matmul.template get_destination_cooperative_tensor<decltype(a_tile), decltype(w_tile), float>();
    for (uint16_t i = 0; i < result.get_capacity(); ++i) {
        if (result.is_valid_element(i))
            result[i] = 0.0f;
    }
    matmul.run(a_tile, w_tile, result);
    device T* output = (device T*)p.output + group.z * p.output_stride;
    device const T* residual = (device const T*)p.residual + group.z * p.output_stride;
    for (uint16_t i = 0; i < result.get_capacity(); ++i) {
        const auto index = result.get_multidimensional_index(i);
        const int r = row + index[1], c = column + index[0];
        if (!result.is_valid_element(i) || r >= m || c >= n)
            continue;
        float value = result[i];
        if (p.has_bias != 0)
            value += float(((device const T*)p.bias)[c]);
        value = nn_activation(value, p.activation);
        if (p.has_scale != 0)
            value *= float(((device const T*)p.scale)[c]);
        if (p.has_residual != 0)
            value += float(residual[r * n + c]);
        output[r * n + c] = T(value);
    }
}

kernel void nn_linear(constant NnLinearParams& p [[buffer(0)]], uint3 group [[threadgroup_position_in_grid]]) {
    if (kInputDType == LFS_DT_Float16) {
        if (kTransposeB != 0)
            nn_linear_tile<half, true>(p, group);
        else
            nn_linear_tile<half, false>(p, group);
    } else {
        if (kTransposeB != 0)
            nn_linear_tile<float, true>(p, group);
        else
            nn_linear_tile<float, false>(p, group);
    }
}

// Attention: softmax(q . k^T * scale + mask) . v with an online softmax in
// FP32, as the CUDA kernel computes it. A SIMD group owns 8 queries of one
// group (a head of one batch), and a threadgroup's four SIMD groups share the
// key and value blocks that stream through threadgroup memory. The head dim
// pads to kNnDimBlocks blocks of 8; half weights meet half values.
constant uint kNnDimBlocks [[function_constant(12)]];

struct NnAttentionParams {
    device const uchar* q;
    device const uchar* k;
    device const uchar* v;
    device const uchar* mask;
    device uchar* output;
    long mask_batch, mask_head, mask_query, mask_key;
    uint heads, queries, keys, dim;
    float scale;
    uint has_mask;
};

constant uint kNnAttentionQueries = 32;

template <typename T, uint Keys>
static void nn_attention_block(constant NnAttentionParams& p, uint query_block, uint group, ushort simd, ushort lane,
                               threadgroup T* staging, threadgroup float* scores, threadgroup T* weights,
                               threadgroup float* diagonal) {
    const uint dim = p.dim, stride = kNnDimBlocks * 8, worker = simd * 32u + lane;
    device const T* q = (device const T*)p.q + ulong(group) * p.queries * dim;
    device const T* k = (device const T*)p.k + ulong(group) * p.keys * dim;
    device const T* v = (device const T*)p.v + ulong(group) * p.keys * dim;
    const uint first_query = query_block * kNnAttentionQueries;
    for (uint i = worker; i < kNnAttentionQueries * stride; i += 128) {
        const uint query = first_query + i / stride, column = i % stride;
        staging[i] = query < p.queries && column < dim ? q[ulong(query) * dim + column] : T(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_matrix<T, 8, 8> queries[16];
    simdgroup_float8x8 output[16];
    for (uint b = 0; b < kNnDimBlocks; ++b) {
        simdgroup_load(queries[b], staging + simd * 8u * stride + b * 8u, stride);
        output[b] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    threadgroup float* s = scores + simd * 8u * Keys;
    threadgroup T* w = weights + simd * 8u * Keys;
    threadgroup float* d = diagonal + simd * 64u;
    for (uint i = lane; i < 64; i += 32)
        d[i] = 0.0f;
    // A quad of lanes owns each row: lane / 4 is the row, lane % 4 its first column.
    const uint row = lane / 4u, query = first_query + simd * 8u + row;
    const bool live = query < p.queries;
    const long mask_row = long(group / p.heads) * p.mask_batch + long(group % p.heads) * p.mask_head +
                          long(query) * p.mask_query;
    float row_max = -INFINITY, row_sum = 0.0f;
    threadgroup T* keys = staging;
    threadgroup T* values = staging + Keys * stride;
    for (uint key0 = 0; key0 < p.keys; key0 += Keys) {
        // The previous block, or the queries, have been read.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = worker; i < Keys * stride; i += 128) {
            const uint key = key0 + i / stride, column = i % stride;
            const bool present = key < p.keys && column < dim;
            keys[i] = present ? k[ulong(key) * dim + column] : T(0);
            values[i] = present ? v[ulong(key) * dim + column] : T(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint c = 0; c < Keys / 8; ++c) {
            simdgroup_float8x8 product = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (uint b = 0; b < kNnDimBlocks; ++b) {
                simdgroup_matrix<T, 8, 8> key_block;
                simdgroup_load(key_block, keys + c * 8u * stride + b * 8u, stride, ulong2(0), true);
                simdgroup_multiply_accumulate(product, queries[b], key_block, product);
            }
            simdgroup_store(product, s + c * 8u, Keys);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        float scaled[Keys / 4];
        float block_max = -INFINITY;
        for (uint j = 0; j < Keys / 4; ++j) {
            const uint column = lane % 4u + 4u * j, key = key0 + column;
            float score = -INFINITY;
            if (key < p.keys) {
                score = s[row * Keys + column] * p.scale;
                if (p.has_mask != 0 && live)
                    score += float(((device const T*)p.mask)[mask_row + long(key) * p.mask_key]);
            }
            scaled[j] = score;
            block_max = max(block_max, score);
        }
        block_max = max(block_max, simd_shuffle_xor(block_max, 1));
        block_max = max(block_max, simd_shuffle_xor(block_max, 2));
        const float new_max = max(row_max, block_max);
        const float alpha = row_max == -INFINITY ? 0.0f : exp(row_max - new_max);
        float block_sum = 0.0f;
        for (uint j = 0; j < Keys / 4; ++j) {
            const float weight = scaled[j] == -INFINITY ? 0.0f : exp(scaled[j] - new_max);
            w[row * Keys + lane % 4u + 4u * j] = T(weight);
            block_sum += weight;
        }
        block_sum += simd_shuffle_xor(block_sum, 1);
        block_sum += simd_shuffle_xor(block_sum, 2);
        row_sum = row_sum * alpha + block_sum;
        row_max = new_max;
        if (lane % 4u == 0)
            d[row * 9u] = alpha;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 rescale;
        simdgroup_load(rescale, d, 8);
        for (uint b = 0; b < kNnDimBlocks; ++b) {
            simdgroup_multiply(output[b], rescale, output[b]);
            for (uint c = 0; c < Keys / 8; ++c) {
                simdgroup_matrix<T, 8, 8> weight_block, value_block;
                simdgroup_load(weight_block, w + c * 8u, Keys);
                simdgroup_load(value_block, values + c * 8u * stride + b * 8u, stride);
                simdgroup_multiply_accumulate(output[b], weight_block, value_block, output[b]);
            }
        }
    }
    // Rows whose keys were all masked out stay zero.
    if (lane % 4u == 0)
        d[row * 9u] = row_sum > 0.0f ? 1.0f / row_sum : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 inverse_sum;
    simdgroup_load(inverse_sum, d, 8);
    threadgroup float* result = (threadgroup float*)staging + simd * 8u * stride;
    for (uint b = 0; b < kNnDimBlocks; ++b) {
        simdgroup_multiply(output[b], inverse_sum, output[b]);
        simdgroup_store(output[b], result + b * 8u, stride);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    device T* out = (device T*)p.output + ulong(group) * p.queries * dim;
    for (uint i = lane; i < 8u * dim; i += 32) {
        const uint r = i / dim, column = i % dim, target = first_query + simd * 8u + r;
        if (target < p.queries)
            out[ulong(target) * dim + column] = T(result[r * stride + column]);
    }
}

kernel void nn_attention(constant NnAttentionParams& p [[buffer(0)]], uint2 group [[threadgroup_position_in_grid]],
                         ushort simd [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    // Holds 32 queries or two key and value blocks of 128 dims, and at the
    // end the four SIMD groups' outputs in FP32.
    threadgroup float staging[4096];
    threadgroup float scores[4 * 8 * 32];
    threadgroup float weights[4 * 8 * 32];
    threadgroup float diagonal[4 * 64];
    if (kInputDType == LFS_DT_Float16)
        nn_attention_block<half, 32>(p, group.x, group.y, simd, lane, (threadgroup half*)staging, scores,
                                     (threadgroup half*)weights, diagonal);
    else
        nn_attention_block<float, 16>(p, group.x, group.y, simd, lane, staging, scores, weights, diagonal);
}
