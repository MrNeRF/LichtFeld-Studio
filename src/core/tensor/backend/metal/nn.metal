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
// [n][k]. Convolutions index bias and scale by row instead and write rows
// out_pitch apart. A threadgroup of four SIMD groups computes one 32x32 tile,
// or 64x64 with kWideTile.
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
    uint has_bias, has_scale, has_residual, row_bias;
    uint out_pitch, padding;
};

template <typename T, bool TransposeW, int Tile>
static void nn_linear_tile(constant NnLinearParams& p, uint3 group) {
    using Matrix = tensor<device T, dextents<int32_t, 2>, tensor_inline>;
    const int m = int(p.m), n = int(p.n), k = int(p.k);
    // Extents list the innermost dimension first.
    Matrix a((device T*)p.a + group.z * p.a_stride, dextents<int32_t, 2>(k, m));
    Matrix w((device T*)p.w + group.z * p.w_stride, TransposeW ? dextents<int32_t, 2>(k, n) : dextents<int32_t, 2>(n, k));
    constexpr auto descriptor =
        mpp::tensor_ops::matmul2d_descriptor(Tile, Tile, static_cast<int>(dynamic_extent), false, TransposeW);
    mpp::tensor_ops::matmul2d<descriptor, execution_simdgroups<4>> matmul;
    const int row = int(group.y) * Tile, column = int(group.x) * Tile;
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
    const uint pitch = p.out_pitch;
    for (uint16_t i = 0; i < result.get_capacity(); ++i) {
        const auto index = result.get_multidimensional_index(i);
        const int r = row + index[1], c = column + index[0];
        if (!result.is_valid_element(i) || r >= m || c >= n)
            continue;
        const int channel = p.row_bias != 0 ? r : c;
        const uint at = uint(r) * pitch + uint(c);
        float value = result[i];
        if (p.has_bias != 0)
            value += float(((device const T*)p.bias)[channel]);
        value = nn_activation(value, p.activation);
        if (p.has_scale != 0)
            value *= float(((device const T*)p.scale)[channel]);
        if (p.has_residual != 0)
            value += float(residual[at]);
        output[at] = T(value);
    }
}

template <typename T>
static void nn_linear_typed(constant NnLinearParams& p, uint3 group) {
    if (kTransposeB != 0 && kWideTile != 0)
        nn_linear_tile<T, true, 64>(p, group);
    else if (kTransposeB != 0)
        nn_linear_tile<T, true, 32>(p, group);
    else if (kWideTile != 0)
        nn_linear_tile<T, false, 64>(p, group);
    else
        nn_linear_tile<T, false, 32>(p, group);
}

kernel void nn_linear(constant NnLinearParams& p [[buffer(0)]], uint3 group [[threadgroup_position_in_grid]]) {
    if (kInputDType == LFS_DT_Float16)
        nn_linear_typed<half>(p, group);
    else
        nn_linear_typed<float>(p, group);
}

// Attention: softmax(q . k^T * scale + mask) . v with an online softmax in
// FP32, as the CUDA kernel computes it. Each SIMD group owns 16 queries of
// one group (a head of one batch) and streams 32-key blocks through the
// matrix units: scores into registers, their row maxima and sums through
// reduce_rows, and the weights, staged in threadgroup memory, times the
// values into the output. kNnHeadTile, 64 or 128, bounds the head dim.
// When few queries meet many keys, kNnAttentionSplit spreads the keys over
// splits of key_split keys that write unnormalized partial outputs with
// their row maxima and sums; nn_attention_combine merges them.
constant uint kNnHeadTile [[function_constant(12)]];
constant uint kNnAttentionSplit [[function_constant(28)]];

struct NnAttentionParams {
    device const uchar* q;
    device const uchar* k;
    device const uchar* v;
    device const uchar* mask;
    device uchar* output;
    long mask_batch, mask_head, mask_query, mask_key;
    device float* partial;
    uint heads, queries, keys, dim;
    float scale;
    uint has_mask, key_split, groups;
};

constant int kNnAttentionQueries = 16, kNnAttentionKeys = 32, kNnAttentionSimds = 4;

template <typename T, int HeadTile>
static void nn_attention_block(constant NnAttentionParams& p, uint query_block, uint group, uint split, ushort lane,
                               threadgroup T* weights, threadgroup float* stats) {
    using Matrix = tensor<device T, dextents<int32_t, 2>, tensor_inline>;
    using Staged = tensor<threadgroup T, dextents<int32_t, 2>, tensor_inline>;
    constexpr int kQ = kNnAttentionQueries, kK = kNnAttentionKeys;
    const int dim = int(p.dim), queries = int(p.queries), keys = int(p.keys);
    const int first = int(query_block) * kQ;
    device T* q = (device T*)p.q + ulong(group) * queries * dim;
    device T* k = (device T*)p.k + ulong(group) * keys * dim;
    device T* v = (device T*)p.v + ulong(group) * keys * dim;
    // Extents list the innermost dimension first.
    Matrix query_tile(q + ulong(first) * dim, dextents<int32_t, 2>(dim, queries - first));
    Matrix key_tensor(k, dextents<int32_t, 2>(dim, keys));
    constexpr auto score_descriptor =
        mpp::tensor_ops::matmul2d_descriptor(kQ, kK, static_cast<int>(dynamic_extent), false, true);
    mpp::tensor_ops::matmul2d<score_descriptor, execution_simdgroup> score_matmul;
    constexpr auto value_descriptor = mpp::tensor_ops::matmul2d_descriptor(
        kQ, HeadTile, static_cast<int>(dynamic_extent), false, false, false,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<value_descriptor, execution_simdgroup> value_matmul;
    auto key_block = key_tensor.slice(0, 0);
    auto scores = score_matmul.template get_destination_cooperative_tensor<decltype(query_tile), decltype(key_block), float>();
    auto reduced =
        score_matmul.template get_row_reduction_destination_cooperative_tensor<decltype(query_tile), decltype(key_block), float>();
    Staged staged(weights, dextents<int32_t, 2>(kK, kQ));
    Matrix value_block(v, dextents<int32_t, 2>(dim, kK));
    auto output = value_matmul.template get_destination_cooperative_tensor<decltype(staged), decltype(value_block), float>();
    for (uint16_t i = 0; i < output.get_capacity(); ++i) {
        if (output.is_valid_element(i))
            output[i] = 0.0f;
    }
    // Per query row: the running maximum and sum, and each block's rescale.
    threadgroup float* row_max = stats;
    threadgroup float* row_sum = stats + kQ;
    threadgroup float* rescale = stats + 2 * kQ;
    tensor<threadgroup float, dextents<int32_t, 1>, tensor_inline> rescale_tensor(rescale, dextents<int32_t, 1>(kQ));
    if (lane < kQ) {
        row_max[lane] = -INFINITY;
        row_sum[lane] = 0.0f;
    }
    const long mask_base = long(group / p.heads) * p.mask_batch + long(group % p.heads) * p.mask_head;
    const int key_begin = kNnAttentionSplit != 0 ? int(split * p.key_split) : 0;
    const int key_end = kNnAttentionSplit != 0 ? min(keys, key_begin + int(p.key_split)) : keys;
    for (int key0 = key_begin; key0 < key_end; key0 += kK) {
        const int count = min(kK, key_end - key0);
        for (uint16_t i = 0; i < scores.get_capacity(); ++i) {
            if (scores.is_valid_element(i))
                scores[i] = 0.0f;
        }
        key_block = key_tensor.slice(0, key0);
        score_matmul.run(query_tile, key_block, scores);
        for (uint16_t i = 0; i < scores.get_capacity(); ++i) {
            if (!scores.is_valid_element(i))
                continue;
            const auto index = scores.get_multidimensional_index(i);
            const int key = key0 + index[0], query = first + index[1];
            float score = -INFINITY;
            if (key < key_end) {
                score = scores[i] * p.scale;
                if (p.has_mask != 0 && query < queries)
                    score += float(((device const T*)p.mask)[mask_base + long(query) * p.mask_query + long(key) * p.mask_key]);
            }
            scores[i] = score;
        }
        mpp::tensor_ops::reduce_rows(scores, reduced, mpp::tensor_ops::reduction_operation::max, -INFINITY);
        reduced.store(rescale_tensor);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < kQ) {
            const float previous = row_max[lane], next = max(previous, rescale[lane]);
            rescale[lane] = previous == -INFINITY ? 0.0f : exp(previous - next);
            row_max[lane] = next;
            row_sum[lane] *= rescale[lane];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint16_t i = 0; i < scores.get_capacity(); ++i) {
            if (!scores.is_valid_element(i))
                continue;
            const auto index = scores.get_multidimensional_index(i);
            const float weight = scores[i] == -INFINITY ? 0.0f : exp(scores[i] - row_max[index[1]]);
            scores[i] = weight;
            weights[index[1] * kK + index[0]] = T(weight);
        }
        for (uint16_t i = 0; i < output.get_capacity(); ++i) {
            if (output.is_valid_element(i))
                output[i] *= rescale[output.get_multidimensional_index(i)[1]];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        mpp::tensor_ops::reduce_rows(scores, reduced, mpp::tensor_ops::reduction_operation::sum, 0.0f);
        reduced.store(rescale_tensor);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < kQ)
            row_sum[lane] += rescale[lane];
        // The inner dimension of the product is this block's key count.
        Staged block_weights(weights, dextents<int32_t, 2>(count, kQ), array<int32_t, 2>{1, kK});
        value_block = Matrix(v + ulong(key0) * dim, dextents<int32_t, 2>(dim, count));
        value_matmul.run(block_weights, value_block, output);
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (kNnAttentionSplit != 0) {
        // Unnormalized outputs [split][group][query][dim], then the row
        // maxima and sums, each [split][group][query].
        const ulong row0 = (ulong(split) * p.groups + group) * queries;
        const ulong stats_at = ulong((p.keys + p.key_split - 1) / p.key_split) * p.groups * queries;
        device float* partial = p.partial + row0 * dim;
        for (uint16_t i = 0; i < output.get_capacity(); ++i) {
            if (!output.is_valid_element(i))
                continue;
            const auto index = output.get_multidimensional_index(i);
            const int query = first + index[1], column = index[0];
            if (query < queries && column < dim)
                partial[ulong(query) * dim + column] = output[i];
        }
        device float* maxima = p.partial + stats_at * dim + row0;
        if (lane < kQ && first + int(lane) < queries) {
            maxima[first + lane] = row_max[lane];
            maxima[stats_at + first + lane] = row_sum[lane];
        }
        return;
    }
    // Rows whose keys were all masked out stay zero.
    device T* out = (device T*)p.output + ulong(group) * queries * dim;
    for (uint16_t i = 0; i < output.get_capacity(); ++i) {
        if (!output.is_valid_element(i))
            continue;
        const auto index = output.get_multidimensional_index(i);
        const int query = first + index[1], column = index[0];
        if (query < queries && column < dim) {
            const float sum = row_sum[index[1]];
            out[ulong(query) * dim + column] = T(sum > 0.0f ? output[i] / sum : 0.0f);
        }
    }
}

kernel void nn_attention(constant NnAttentionParams& p [[buffer(0)]], uint3 group [[threadgroup_position_in_grid]],
                         ushort simd [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float weights[kNnAttentionSimds][kNnAttentionQueries * kNnAttentionKeys];
    threadgroup float stats[kNnAttentionSimds][3 * kNnAttentionQueries];
    // SIMD groups own consecutive query blocks; one past the end has no work.
    const uint query_block = group.x * kNnAttentionSimds + simd;
    if (query_block * kNnAttentionQueries >= p.queries)
        return;
    if (kInputDType == LFS_DT_Float16) {
        if (kNnHeadTile == 128)
            nn_attention_block<half, 128>(p, query_block, group.y, group.z, lane, (threadgroup half*)weights[simd], stats[simd]);
        else
            nn_attention_block<half, 64>(p, query_block, group.y, group.z, lane, (threadgroup half*)weights[simd], stats[simd]);
    } else {
        if (kNnHeadTile == 128)
            nn_attention_block<float, 128>(p, query_block, group.y, group.z, lane, weights[simd], stats[simd]);
        else
            nn_attention_block<float, 64>(p, query_block, group.y, group.z, lane, weights[simd], stats[simd]);
    }
}

// Merges the key splits of nn_attention: a thread takes one output value.
kernel void nn_attention_combine(constant NnAttentionParams& p [[buffer(0)]], uint3 id [[thread_position_in_grid]]) {
    const uint column = id.x, query = id.y, group = id.z;
    if (column >= p.dim)
        return;
    const uint splits = (p.keys + p.key_split - 1) / p.key_split;
    const ulong rows = ulong(p.groups) * p.queries, stats_at = ulong(splits) * rows;
    device const float* maxima = p.partial + stats_at * p.dim;
    float peak = -INFINITY;
    for (uint split = 0; split < splits; ++split)
        peak = max(peak, maxima[split * rows + group * p.queries + query]);
    float sum = 0.0f, value = 0.0f;
    if (peak != -INFINITY) {
        for (uint split = 0; split < splits; ++split) {
            const ulong row = split * rows + group * p.queries + query;
            const float split_max = maxima[row];
            if (split_max == -INFINITY)
                continue;
            const float weight = exp(split_max - peak);
            sum += maxima[stats_at + row] * weight;
            value += p.partial[row * p.dim + column] * weight;
        }
    }
    store_float(p.output, (group * p.queries + query) * p.dim + column, sum > 0.0f ? value / sum : 0.0f);
}

// Convolution patches in the input's precision: columns[b][tap][column] for
// the output pixels [first, first + count) of one group, where a tap is a
// (channel, ky, kx) triple. A thread takes one pixel of one channel and
// writes its kernel taps. Replicate padding (mode 1) clamps to the edge.
struct NnIm2colParams {
    device const uchar* input;
    device uchar* columns;
    uint input_batch, count, taps, first, channel0, padding;
    InferenceGeometry p;
};

kernel void nn_im2col(constant NnIm2colParams& params [[buffer(0)]], uint3 id [[thread_position_in_grid]]) {
    if (id.x >= params.count)
        return;
    constant InferenceGeometry& p = params.p;
    const int pixel = int(params.first + id.x);
    const int x0 = (pixel % p.out_width) * p.stride_w - p.pad_w, y0 = (pixel / p.out_width) * p.stride_h - p.pad_h;
    const uint plane = id.z * params.input_batch + (params.channel0 + id.y) * uint(p.height * p.width);
    uint at = (id.z * params.taps + id.y * uint(p.kernel_h * p.kernel_w)) * params.count + id.x;
    for (int ky = 0; ky < p.kernel_h; ++ky) {
        const int y = y0 + ky * p.dilation_h;
        for (int kx = 0; kx < p.kernel_w; ++kx, at += params.count) {
            const int x = x0 + kx * p.dilation_w;
            float value = 0.0f;
            if (p.mode == 1 || (x >= 0 && x < p.width && y >= 0 && y < p.height))
                value = load_float(params.input,
                                   plane + uint(clamp(y, 0, p.height - 1) * p.width + clamp(x, 0, p.width - 1)));
            store_float(params.columns, at, value);
        }
    }
}

// Transposed convolution outputs gathered from the GEMM's columns
// [image][channel][ky][kx][input pixel], with the bias and activation. A
// thread computes one output pixel of one channel.
struct NnCol2imParams {
    device const uchar* columns;
    device const uchar* bias;
    device uchar* output;
    uint pixels, channel0, out_batch, columns_batch;
    int activation;
    uint has_bias;
    InferenceGeometry p;
};

kernel void nn_col2im(constant NnCol2imParams& params [[buffer(0)]], uint3 id [[thread_position_in_grid]]) {
    if (id.x >= params.pixels)
        return;
    constant InferenceGeometry& p = params.p;
    const int ox = int(id.x) % p.out_width, oy = int(id.x) / p.out_width;
    const uint plane = uint(p.height * p.width);
    const uint row0 = id.z * params.columns_batch + id.y * uint(p.kernel_h * p.kernel_w) * plane;
    float value = 0.0f;
    for (int ky = 0; ky < p.kernel_h; ++ky) {
        int iy = oy + p.pad_h - ky * p.dilation_h;
        if (iy < 0 || iy % p.stride_h != 0 || (iy /= p.stride_h) >= p.height)
            continue;
        for (int kx = 0; kx < p.kernel_w; ++kx) {
            int ix = ox + p.pad_w - kx * p.dilation_w;
            if (ix < 0 || ix % p.stride_w != 0 || (ix /= p.stride_w) >= p.width)
                continue;
            value += load_float(params.columns, row0 + uint(ky * p.kernel_w + kx) * plane + uint(iy * p.width + ix));
        }
    }
    const uint channel = params.channel0 + id.y;
    if (params.has_bias != 0)
        value += load_float(params.bias, channel);
    store_float(params.output, id.z * params.out_batch + channel * params.pixels + id.x,
                nn_activation(value, params.activation));
}
