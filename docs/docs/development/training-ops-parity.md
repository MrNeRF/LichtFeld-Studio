# Training ops parity

Each trainer GPU backend has a table of function pointers, one struct per family, filled in `training_ops()` (`src/training/include/lfs/training/ops/registry.hpp`). The CUDA table is the reference. A Vulkan or Metal slot stays null until that backend implements the family. A null slot means the trainer refuses to start when the family is required. It does not fall through to CUDA.

The parity suite in `tests/test_training_ops_parity.cpp` builds the same deterministic inputs the family's exact-byte fixture uses, runs the CUDA table, copies those inputs onto the second backend, and runs that backend's table. Integer and bookkeeping results compare byte for byte. Float results use the absolute and relative tolerances named next to each case in the test. `TrainingOpsParity.SelfCheck` runs every family twice on CUDA through that same comparison and requires identical bytes. `TrainingOpsParity.ComparatorRejectsPerturbation` checks that a value outside the tolerance is rejected. `TrainingOpsLossCurveParity.SyntheticLossCurve` trains the synthetic 32×32 scene for 200 steps on CUDA and on the second backend, and compares each step's loss within 1e-2 absolute plus 1e-2 relative. It runs only when every family that configuration requires is present. Two CUDA runs of this scene already drift by a few thousandths, because the raster blend is an atomic reduction.

Each family test runs once per second backend, named `Backends/TrainingOpsFamilyParity.MatchesCuda/<Family>_Vulkan` and `..._Metal`. A null slot or a missing device skips that case with the family or backend name. The loss curve runs for `Cuda` (CUDA against itself), `Vulkan` and `Metal`.

From the repository root, with the test binary's runtime libraries on the library path:

```bash
./build/tests/lichtfeld_tests --tensor-backend=cuda --gtest_filter='TrainingOpsParity.*:*TrainingOps*Parity*'
./build/tests/lichtfeld_tests --tensor-backend=cuda --gtest_filter='*TrainingOps*Parity*Vulkan*'
./build/tests/lichtfeld_tests --tensor-backend=cuda --gtest_filter='*TrainingOps*Parity*Metal*'
```

Vulkan and Metal skip a family until its slot is non-null.

A slot implements every function pointer on its struct. Outputs are the tensors and scalars the CUDA fixture produces for those inputs. The calls do not synchronize for the caller; the test waits before it reads. Raster results use `RasterResult`. Session calls must be safe to invoke, and `allocation_bytes` must be exact. Seeded stochastic ops (MCMC noise and sampling, MRNF noise and Gumbel, random backgrounds) are exact for the same seed. Fast backward passes a zero image gradient into the fused Adam update and checks that means, scales, and opacity stay unchanged. A nonzero blend is reduced with atomics, so two launches of the same CUDA kernel are not byte-identical.

| Family | Struct | Header | Test |
| --- | --- | --- | --- |
| Session | `SessionOps` | `ops/session.hpp` | `TrainingOpsFamilyParity.MatchesCuda/Session_*` |
| Fast | `FastRasterOps` | `ops/raster.hpp` | `TrainingOpsFamilyParity.MatchesCuda/Fast_*` |
| Gsplat | `GsplatRasterOps` | `ops/gsplat.hpp` | `TrainingOpsFamilyParity.MatchesCuda/Gsplat_*` |
| Photometric | `PhotometricOps` | `ops/loss.hpp` | `TrainingOpsFamilyParity.MatchesCuda/Photometric_*` |
| Geometry | `GeometryLossOps` | `ops/geometry.hpp` | `TrainingOpsFamilyParity.MatchesCuda/Geometry_*` |
| Masks | `MaskOps` | `ops/masks.hpp` | `TrainingOpsFamilyParity.MatchesCuda/Masks_*` |
| ExtraLoss | `ExtraLossOps` | `ops/extra_loss.hpp` | `TrainingOpsFamilyParity.MatchesCuda/ExtraLoss_*` |
| Adam | `AdamOps` | `ops/adam.hpp` | `TrainingOpsFamilyParity.MatchesCuda/Adam_*` |
| Mcmc | `McmcOps` | `ops/mcmc.hpp` | `TrainingOpsFamilyParity.MatchesCuda/Mcmc_*` |
| Mrnf | `MrnfOps` | `ops/mrnf.hpp` | `TrainingOpsFamilyParity.MatchesCuda/Mrnf_*` |
| Refine | `RefineOps` | `ops/refine.hpp` | `TrainingOpsFamilyParity.MatchesCuda/Refine_*` |
| Bilateral | `BilateralOps` | `ops/bilateral.hpp` | `TrainingOpsFamilyParity.MatchesCuda/Bilateral_*` |
| PPISP | `PPISPOps` | `ops/ppisp.hpp` | `TrainingOpsFamilyParity.MatchesCuda/PPISP_*` |
| Controller | `ControllerOps` | `ops/ppisp.hpp` | `TrainingOpsFamilyParity.MatchesCuda/Controller_*` |
| Morton | `MortonOps` | `ops/morton.hpp` | `TrainingOpsFamilyParity.MatchesCuda/Morton_*` |
| Sh | `ShOps` | `ops/sh.hpp` | `TrainingOpsFamilyParity.MatchesCuda/Sh_*` |
| TrainingImage | `TrainingImageOps` | `ops/training_image.hpp` | `TrainingOpsFamilyParity.MatchesCuda/TrainingImage_*` |
| SharedImage | `SharedImageOps` | `core/shared_image_ops.hpp` | `TrainingOpsFamilyParity.MatchesCuda/SharedImage_*` |
| Lpips | `LpipsOps` | `ops/lpips.hpp` | `TrainingOpsFamilyParity.MatchesCuda/Lpips_*` |

Headers live under `src/training/include/lfs/training/` except `SharedImageOps`, which is shared with image I/O and lives in `src/core/include/core/shared_image_ops.hpp`. The CUDA definitions are the `*_cuda` translation units next to those headers. Filling a slot is the kernel work for that family. This harness does not add kernels.
