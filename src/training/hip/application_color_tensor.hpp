/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "application_tensor.hpp"
#include "application_color_api.h"
#include <climits>
namespace lfs::training::hip {
struct ColorCall {
    ApplicationCall owner;LfsHipColorCall p{};
    explicit ColorCall(unsigned operation){p.struct_bytes=sizeof(p);p.operation=operation;p.device=owner.queue.device;p.queue=owner.queue.native_queue;}
    void bind(unsigned i,const core::Tensor& tensor){p.bindings[i]=owner.bind(tensor,core::DataType::Float32);}
    void submit(const char* symbol) {
        const auto fn=owner.module->symbol<decltype(&lfs_hip_application_bilateral)>(symbol);
        LfsHipError error{};if(int code=fn(&p,&error))throw TrainingError(code,error);
    }
};
inline int color_count(size_t n){if(n>INT_MAX)throw std::invalid_argument("HIP color count exceeds int32 range");return static_cast<int>(n);}
inline void color_require(bool value,const char* message){if(!value)throw std::invalid_argument(message);}
inline void color_shape(const core::Tensor& t,size_t count){color_require(t.is_valid()&&t.numel()==count,"HIP color operand shape mismatch");}
}
