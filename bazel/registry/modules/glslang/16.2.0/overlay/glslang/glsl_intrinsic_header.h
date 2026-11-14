// Pre-generated for the Bazel overlay of glslang 16.2.0; only used by the
// glslang_validator binary (StandAlone/StandAlone.cpp).
// Generated from the release tarball with:
//   python gen_extension_headers.py -i glslang/ExtensionHeaders -o glslang/glsl_intrinsic_header.h
// Regenerate when glslang/ExtensionHeaders/*.glsl changes.

/***************************************************************************
 *
 * Copyright (c) 2015-2021 The Khronos Group Inc.
 * Copyright (c) 2015-2021 Valve Corporation
 * Copyright (c) 2015-2021 LunarG, Inc.
 * Copyright (c) 2015-2021 Google Inc.
 * Copyright (c) 2021 Advanced Micro Devices, Inc.All rights reserved.
 *
 ****************************************************************************/
#pragma once

#ifndef _INTRINSIC_EXTENSION_HEADER_H_
#define _INTRINSIC_EXTENSION_HEADER_H_

std::string GL_EXT_shader_realtime_clock_GLSL = R"(
//
// Copyright (C) 2002-2005  3Dlabs Inc. Ltd.
// Copyright (C) 2013-2016 LunarG, Inc.
// Copyright (C) 2016-2020 Google, Inc.
// Modifications Copyright(C) 2021 Advanced Micro Devices, Inc.All rights reserved.
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
//    Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//
//    Redistributions in binary form must reproduce the above
//    copyright notice, this list of conditions and the following
//    disclaimer in the documentation and/or other materials provided
//    with the distribution.
//
//    Neither the name of 3Dlabs Inc. Ltd. nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//


)";


std::string getIntrinsic(const char* const* shaders, int n) {
	std::string shaderString = "";
	for (int i = 0; i < n; i++) {
		if (strstr(shaders[i], "GL_EXT_shader_realtime_clock") != nullptr) {
		    shaderString.append(GL_EXT_shader_realtime_clock_GLSL);
		}
	}
	return shaderString;
}

#endif
