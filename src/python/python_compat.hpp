/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

// On Windows debug builds, MSVC defines _DEBUG which causes Python.h to
// enable Py_DEBUG, requiring debug-only symbols like Py_NegativeRefcount
// and Py_DECREF_DecRefTotal. Since vcpkg provides only release Python,
// we need to temporarily undefine _DEBUG before including Python.h.
#if defined(_WIN32) && defined(_DEBUG)
#pragma push_macro("_DEBUG")
#undef _DEBUG
#include <Python.h>
#pragma pop_macro("_DEBUG")
#else
#include <Python.h>
#endif
