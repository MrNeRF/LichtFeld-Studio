# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Compile the native pointwise catalog contracts without a GPU or SDK."""
from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class HipPointwiseCatalog(unittest.TestCase):
    def test_catalog_modes_predicates_and_half_subset(self):
        compiler = os.environ.get("CXX") or shutil.which("c++") or shutil.which("clang++") or shutil.which("cl")
        self.assertIsNotNone(compiler, "A C++ compiler is required for the catalog contract")
        source = r'''
#include "pointwise_catalog.hpp"
#include <iterator>
using namespace lfs::core;
using namespace lfs::core::internal;
using hip_pointwise::Lowering;
constexpr unsigned modes[] = {
#define LFS_POINTWISE_OP(Id, F, Name) Lowering<F>::mode,
#include "pointwise_ops.def"
#undef LFS_POINTWISE_OP
};
constexpr bool predicates[] = {
#define LFS_POINTWISE_OP(Id, F, Name) Lowering<F>::predicate,
#include "pointwise_ops.def"
#undef LFS_POINTWISE_OP
};
static_assert(std::size(modes) == LFS_HIP_OP_COUNT && LFS_HIP_OP_COUNT > 0);
static_assert(modes[LFS_HIP_OP_AddScalar] == 1);
static_assert(modes[LFS_HIP_OP_AddTensor] == 2);
static_assert(modes[LFS_HIP_OP_ExpThenMulScalar] == 0);
static_assert(predicates[LFS_HIP_OP_EqualScalar] && predicates[LFS_HIP_OP_LogicalNot]);
static_assert(!predicates[LFS_HIP_OP_Abs]);
constexpr unsigned half_count() {
    unsigned count = 0;
    for (unsigned i = 0; i < LFS_HIP_OP_COUNT; ++i)
        count += hip_pointwise::half_supported(i);
    return count;
}
static_assert(half_count() == 17);
static_assert(hip_pointwise::half_supported(LFS_HIP_OP_AddTensor));
static_assert(hip_pointwise::half_supported(LFS_HIP_OP_SubTensor));
static_assert(hip_pointwise::half_supported(LFS_HIP_OP_MulTensor));
static_assert(hip_pointwise::half_supported(LFS_HIP_OP_DivTensor));
static_assert(hip_pointwise::half_supported(LFS_HIP_OP_EqualTensor));
static_assert(hip_pointwise::half_supported(LFS_HIP_OP_NotEqualTensor));
static_assert(hip_pointwise::half_supported(LFS_HIP_OP_LessTensor));
static_assert(hip_pointwise::half_supported(LFS_HIP_OP_LessEqualTensor));
static_assert(hip_pointwise::half_supported(LFS_HIP_OP_GreaterTensor));
static_assert(hip_pointwise::half_supported(LFS_HIP_OP_GreaterEqualTensor));
static_assert(hip_pointwise::half_supported(LFS_HIP_OP_LogicalAndTensor));
static_assert(hip_pointwise::half_supported(LFS_HIP_OP_LogicalOrTensor));
static_assert(hip_pointwise::half_supported(LFS_HIP_OP_LogicalXorTensor));
static_assert(hip_pointwise::half_supported(LFS_HIP_OP_IsNan));
static_assert(hip_pointwise::half_supported(LFS_HIP_OP_IsInf));
static_assert(hip_pointwise::half_supported(LFS_HIP_OP_IsFinite));
static_assert(hip_pointwise::half_supported(LFS_HIP_OP_LogicalNot));
int main() {
    const auto mul_abs = Lowering<pointwise_mul_abs_functor>::make(-3.f);
    const auto max_scalar = Lowering<pointwise_maximum_scalar_functor>::make(0.f);
    return mul_abs(2.f) != 6.f || max_scalar(0.5f, 1.5f) != 1.5f;
}
'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "catalog.cpp").write_text(source)
            executable = path / ("catalog.exe" if os.name == "nt" else "catalog")
            adapter = path / "tensor/backend/hip"
            detail = path / "include/core/detail"
            adapter.mkdir(parents=True)
            detail.mkdir(parents=True)
            for header in ("pointwise_catalog.hpp", "runtime_api.h"):
                shutil.copyfile(ROOT / "src/core/tensor/backend/hip" / header, adapter / header)
            catalog = (ROOT / "src/core/include/core/detail/pointwise_ops.def").read_text()
            (detail / "pointwise_ops.def").write_text(catalog)
            includes = [adapter, detail, ROOT / "src/core/include", ROOT / "src/core/include/core/detail",
                        ROOT / "src/core/tensor/backend/hip"]
            if Path(compiler).stem.lower() in ("cl", "clang-cl"):
                command = [compiler, "/nologo", "/std:c++20", "/EHsc",
                           *("/I" + str(p) for p in includes), str(path / "catalog.cpp"),
                           "/Fe:" + str(executable), "/Fo:" + str(path / "catalog.obj")]
            else:
                command = [compiler, "-std=c++20", *("-I" + str(p) for p in includes),
                           str(path / "catalog.cpp"), "-o", str(executable)]
            result = subprocess.run(command, capture_output=True, text=True, timeout=45)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            subprocess.run([str(executable)], check=True, timeout=10)
            invalid_catalogs = {
                "duplicate": catalog + '\nLFS_POINTWISE_OP(AddScalar, ignored, "duplicate")\n',
                "empty": "",
                "malformed": "unknown preprocessor directive",
                "missing_scalar": catalog.replace("AddScalar", "UnknownScalar", 1),
                "unsupported_functor": catalog.replace("pointwise_add_scalar_functor", "unknown_binding", 1),
            }
            for name, invalid in invalid_catalogs.items():
                with self.subTest(catalog=name):
                    (detail / "pointwise_ops.def").write_text(invalid)
                    rejected = subprocess.run(command, capture_output=True, text=True, timeout=45)
                    self.assertNotEqual(rejected.returncode, 0, "Invalid catalog compiled: " + name)


if __name__ == "__main__":
    unittest.main()
