
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "convert.h"
#include "type_utils.h"
#include "static_switch.h"
#include "surface.h"
#include "color_space_conversion_impl.h"
#include "sample_format_utils.h"
#include <stdexcept>
#include <string>

namespace nvimgcodec {

#define IMG_TYPES uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, float

template <typename Out, typename In>
struct RGB_to_Gray_Converter {
  static constexpr int out_pixel_sz = 1;
  static constexpr int in_pixel_sz = 3;
  static NVIMGCODEC_HOST_DEV NVIMGCODEC_FORCEINLINE vec<out_pixel_sz, Out> convert(vec<in_pixel_sz, In> rgb) {
    return rgb_to_gray<Out>(rgb);
  }
};

template <typename Out, typename In>
struct RGB_to_BGR_Converter {
  static constexpr int out_pixel_sz = 3;
  static constexpr int in_pixel_sz = 3;
  static NVIMGCODEC_HOST_DEV NVIMGCODEC_FORCEINLINE vec<out_pixel_sz, Out> convert(vec<in_pixel_sz, In> rgb) {
    vec<out_pixel_sz, Out> out;
    out[0] = ConvertSatNorm<Out>(rgb[2]);
    out[1] = ConvertSatNorm<Out>(rgb[1]);
    out[2] = ConvertSatNorm<Out>(rgb[0]);
    return out;
  }
};

template <typename Out, typename In>
struct Gray_to_RGB_Converter {
  static constexpr int out_pixel_sz = 3;
  static constexpr int in_pixel_sz = 1;
  static NVIMGCODEC_HOST_DEV NVIMGCODEC_FORCEINLINE vec<out_pixel_sz, Out> convert(vec<in_pixel_sz, In> gray) {
    return vec<out_pixel_sz, Out>(ConvertNorm<Out>(gray[0]));
  }
};

template <typename Out, typename In>
struct BGR_to_Gray_Converter {
  static constexpr int out_pixel_sz = 1;
  static constexpr int in_pixel_sz = 3;
  static NVIMGCODEC_HOST_DEV NVIMGCODEC_FORCEINLINE vec<out_pixel_sz, Out> convert(vec<in_pixel_sz, In> bgr) {
    return RGB_to_Gray_Converter<Out, In>::convert(
      RGB_to_BGR_Converter<In, In>::convert(bgr));
  }
};

template <typename Out, typename In>
struct Gray_to_YCbCr_Converter {
  static constexpr int out_pixel_sz = 3;
  static constexpr int in_pixel_sz = 1;
  static NVIMGCODEC_HOST_DEV NVIMGCODEC_FORCEINLINE vec<out_pixel_sz, Out> convert(vec<in_pixel_sz, In> gray) {
    vec<out_pixel_sz, Out> out;
    out[0] = itu_r_bt_601::gray_to_y<Out>(gray[0]);
    out[1] = ConvertNorm<Out>(0.5f);
    out[2] = ConvertNorm<Out>(0.5f);
    return out;
  }
};

template <typename Out, typename In>
struct YCbCr_to_Gray_Converter {
  static constexpr int out_pixel_sz = 1;
  static constexpr int in_pixel_sz = 3;
  static NVIMGCODEC_HOST_DEV NVIMGCODEC_FORCEINLINE vec<out_pixel_sz, Out> convert(vec<in_pixel_sz, In> ycbcr) {
    vec<out_pixel_sz, Out> out;
    out[0] = itu_r_bt_601::y_to_gray<Out>(ycbcr[0]);
    return out;
  }
};

template <typename Out, typename In>
struct RGB_to_YCbCr_Converter {
  static constexpr int out_pixel_sz = 3;
  static constexpr int in_pixel_sz = 3;
  static NVIMGCODEC_HOST_DEV NVIMGCODEC_FORCEINLINE vec<out_pixel_sz, Out> convert(vec<in_pixel_sz, In> rgb) {
    vec<out_pixel_sz, Out> out;
    out[0] = itu_r_bt_601::rgb_to_y<Out>(rgb);
    out[1] = itu_r_bt_601::rgb_to_cb<Out>(rgb);
    out[2] = itu_r_bt_601::rgb_to_cr<Out>(rgb);
    return out;
  }
};

template <typename Out, typename In>
struct BGR_to_YCbCr_Converter {
  static constexpr int out_pixel_sz = 3;
  static constexpr int in_pixel_sz = 3;
  static NVIMGCODEC_HOST_DEV NVIMGCODEC_FORCEINLINE vec<out_pixel_sz, Out> convert(vec<in_pixel_sz, In> bgr) {
    return RGB_to_YCbCr_Converter<Out, In>::convert(
      RGB_to_BGR_Converter<In, In>::convert(bgr));
  }
};

template <typename Out, typename In>
struct YCbCr_to_RGB_Converter {
  static constexpr int out_pixel_sz = 3;
  static constexpr int in_pixel_sz = 3;
  static NVIMGCODEC_HOST_DEV NVIMGCODEC_FORCEINLINE vec<out_pixel_sz, Out> convert(vec<in_pixel_sz, In> ycbcr) {
    return itu_r_bt_601::ycbcr_to_rgb<Out>(ycbcr);
  }
};

template <typename Out, typename In>
struct YCbCr_to_BGR_Converter {
  static constexpr int out_pixel_sz = 3;
  static constexpr int in_pixel_sz = 3;
  static NVIMGCODEC_HOST_DEV NVIMGCODEC_FORCEINLINE vec<out_pixel_sz, Out> convert(vec<in_pixel_sz, In> ycbcr) {
    return RGB_to_BGR_Converter<Out, Out>::convert(
      YCbCr_to_RGB_Converter<Out, In>::convert(ycbcr));
  }
};


template <typename Out, typename In, typename Converter>
__global__ void ConvertKernel(Surface2D<Out> output, const Surface2D<In> input) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= output.size.x || y >= output.size.y)
      return;

    vec<Converter::in_pixel_sz, In> pixel;
#pragma unroll
  for (int c = 0; c < Converter::in_pixel_sz; c++) {
    pixel[c] = input(x, y, c);
  }
  auto out_pixel = Converter::convert(pixel);
#pragma unroll
  for (int c = 0; c < Converter::out_pixel_sz; c++) {
    output(x, y, c) = out_pixel[c];
  }
}

template <typename Out, typename In, typename Converter>
__global__ void ConvertNormKernel(Surface2D<Out> output, const Surface2D<In> input, float multiplier) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= output.size.x || y >= output.size.y)
      return;

    vec<Converter::in_pixel_sz, float> pixel;
#pragma unroll
  for (int c = 0; c < Converter::in_pixel_sz; c++) {
    pixel[c] = multiplier * input(x, y, c);
  }
  auto out_pixel = Converter::convert(pixel);
#pragma unroll
  for (int c = 0; c < Converter::out_pixel_sz; c++) {
    output(x, y, c) = out_pixel[c];
  }
}

template <typename Out, typename In>
__global__ void PassthroughKernel(Surface2D<Out> output, const Surface2D<In> input) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= output.size.x || y >= output.size.y)
      return;
    for (int c = 0; c < output.channels; c++) {
        output(x, y, c) = ConvertNorm<Out, In>(input(x, y, c));
    }
}

template <typename Out, typename In>
__global__ void PassthroughNormKernel(Surface2D<Out> output, const Surface2D<In> input, float multiplier) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= output.size.x || y >= output.size.y)
      return;

    for (int c = 0; c < output.channels; c++) {
        output(x, y, c) = ConvertNorm<Out, float>(multiplier * input(x, y, c));
    }
}

template <typename Out, typename In>
void LaunchConvertNormKernelImpl(const nvimgcodecImageInfo_t& out_info, const nvimgcodecImageInfo_t &in_info, cudaStream_t stream)
{
    const int w = out_info.plane_info[0].width;
    const int h = out_info.plane_info[0].height;
    const int out_c = NumberOfChannels(out_info);
    const int in_c = NumberOfChannels(in_info);
    const auto out_format = out_info.sample_format;
    const auto in_format = in_info.sample_format;
    const size_t out_bytes_per_sample = out_info.plane_info[0].sample_type >> 11;
    const size_t in_bytes_per_sample = in_info.plane_info[0].sample_type >> 11;
    const int out_precision = out_info.plane_info[0].precision;
    const int in_precision = in_info.plane_info[0].precision;
    const auto out_dtype = out_info.plane_info[0].sample_type;
    const auto in_dtype = in_info.plane_info[0].sample_type;
    const dim3 block(32, 32, 1);
    const dim3 grid((w+31)/32, (h+31)/32, 1);

    Surface2D<const In> in;
    in.data = reinterpret_cast<const In*>(in_info.buffer);
    in.size = {w, h};
    in.channels = in_c;
    if (IsPlanar(in_format)) {
        in.strides = {1, in_info.plane_info[0].row_stride / in_bytes_per_sample};
        in.channel_stride = in_info.plane_info[0].row_stride * h / in_bytes_per_sample;
    } else {
        in.strides = {in_c, in_info.plane_info[0].row_stride / in_bytes_per_sample};
        in.channel_stride = 1;
    }

    Surface2D<Out> out;
    out.data = reinterpret_cast<Out*>(out_info.buffer);
    out.size = {w, h};
    out.channels = out_c;
    if (IsPlanar(out_format)) {
        out.strides = {1, out_info.plane_info[0].row_stride / out_bytes_per_sample};
        out.channel_stride = out_info.plane_info[0].row_stride * h / out_bytes_per_sample;
    } else {
        out.strides = {out_c, out_info.plane_info[0].row_stride / out_bytes_per_sample};
        out.channel_stride = 1;
    }

    // Multiplier is such so that when converted to the output type we use as many positive bits as out_precision
    float multiplier = NeedDynamicRangeScaling(out_precision, out_dtype, in_precision, in_dtype)
                           ? DynamicRangeMultiplier(out_precision, out_dtype, in_precision, in_dtype) / MaxValue(in_dtype)
                           : 1.0f;

#define CONV_KERNEL(CONVERTER) \
    if (multiplier != 1.0f) { \
        ConvertNormKernel<Out, In, CONVERTER<Out, float>><<<grid, block, 0, stream>>>(out, in, multiplier); \
    } else { \
        ConvertKernel<Out, In, CONVERTER<Out, In>><<<grid, block, 0, stream>>>(out, in); \
    }

#define PASSTHROUGH_KERNEL() \
    if (multiplier != 1.0f) { \
        PassthroughNormKernel<Out, In><<<grid, block, 0, stream>>>(out, in, multiplier); \
    } else { \
        PassthroughKernel<Out, In><<<grid, block, 0, stream>>>(out, in); \
    }

    if (IsRgb(in_format) && IsGray(out_format)) {
        CONV_KERNEL(RGB_to_Gray_Converter);
    } else if (IsBgr(in_format) && IsGray(out_format)) {
        CONV_KERNEL(BGR_to_Gray_Converter);
    } else if ((IsBgr(in_format) && IsRgb(out_format)) || (IsRgb(in_format) && IsBgr(out_format))) {
        CONV_KERNEL(RGB_to_BGR_Converter);
    } else if (IsGray(in_format) && (IsRgb(out_format) || IsBgr(out_format))) {
        CONV_KERNEL(Gray_to_RGB_Converter);
    } else if (in_c >= out_c) {
        PASSTHROUGH_KERNEL();
    } else {
        throw std::runtime_error("Invalid conversion");
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
      throw std::runtime_error("CUDA Runtime failure: '#" + std::to_string(err) + "'");

#undef CONV_KERNEL
#undef PASSTHROUGH_KERNEL
}


void LaunchConvertNormKernel(const nvimgcodecImageInfo_t& out_info, const nvimgcodecImageInfo_t &in_info, cudaStream_t stream) {
    TYPE_SWITCH(out_info.plane_info[0].sample_type, type2id, Output, (IMG_TYPES),
        (TYPE_SWITCH(in_info.plane_info[0].sample_type, type2id, Input, (IMG_TYPES),
            (LaunchConvertNormKernelImpl<Output, Input>(out_info, in_info, stream)),
            std::runtime_error("Unsupported input type"))),
        std::runtime_error("Unsupported output type"))

}


} // namespace nvimgcodec