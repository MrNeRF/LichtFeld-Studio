/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../src/python/lfs/py_tensor.hpp"
#include "core/tensor_backend.hpp"
NB_MODULE(_tensor_binding_contracts, m) {
    lfs::python::register_tensor(m);
}
