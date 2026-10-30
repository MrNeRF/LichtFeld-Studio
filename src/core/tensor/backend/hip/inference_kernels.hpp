/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "slang_prelude.hpp"
#pragma clang fp contract(off)
namespace hip_inference_0_f32 {
    using namespace hip_slang;
#include "inference_0_f32.cu"
} // namespace hip_inference_0_f32
namespace hip_inference_1_f32 {
    using namespace hip_slang;
#include "inference_1_f32.cu"
} // namespace hip_inference_1_f32
namespace hip_inference_2_f32 {
    using namespace hip_slang;
#include "inference_2_f32.cu"
} // namespace hip_inference_2_f32
namespace hip_inference_4_f32 {
    using namespace hip_slang;
#include "inference_4_f32.cu"
} // namespace hip_inference_4_f32
namespace hip_inference_5_f32 {
    using namespace hip_slang;
#include "inference_5_f32.cu"
} // namespace hip_inference_5_f32
namespace hip_inference_0_f16 {
    using namespace hip_slang;
#include "inference_0_f16.cu"
} // namespace hip_inference_0_f16
namespace hip_inference_1_f16 {
    using namespace hip_slang;
#include "inference_1_f16.cu"
} // namespace hip_inference_1_f16
namespace hip_inference_2_f16 {
    using namespace hip_slang;
#include "inference_2_f16.cu"
} // namespace hip_inference_2_f16
namespace hip_inference_4_f16 {
    using namespace hip_slang;
#include "inference_4_f16.cu"
} // namespace hip_inference_4_f16
namespace hip_inference_5_f16 {
    using namespace hip_slang;
#include "inference_5_f16.cu"
} // namespace hip_inference_5_f16

#pragma clang fp contract(fast)
