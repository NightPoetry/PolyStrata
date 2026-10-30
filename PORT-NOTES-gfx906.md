# gfx906 (Instinct MI50 / MI60, Radeon VII) — HIP Port and Qwen3.6-35B-A3B Deployment Notes

## 1. Overview

This branch enables the strata-poly q36 engine (Qwen3.6-35B-A3B) on the
wave64 gfx906 backend, including full 262144-token context operation with
hybrid expert residency (VRAM hot set, pinned host pool, CPU fallback).

The work consists of two parts:

1. Build-system and kernel-interface corrections required to compile and load
   the engine on gfx906 with ROCm 6.3.3.
2. A verified deployment configuration with measured throughput.

All changes were validated on an Instinct MI50 16 GB (gfx906, wave64) with
ROCm 6.3.3; the upstream gfx906 documentation in `docs/AMD_HIP.md` describes
the same target compiled against HIP 7.14.

## 2. Build corrections

### 2.1 Expert format tables (`src/kernels/cuda/iq_kernels.cu`)

The Qwen3.6-35B-A3B release artifacts use expert tensor layouts that were not
present in the gfx906 dispatch tables:

- `STRATA_GU_FMTS`: added `X(20)` (IQ2_XS) for gate/up projections.
- `STRATA_D_FMTS`: added `X(18)` (IQ3_XXS) and `X(22)` (IQ2_S) for down
  projections.

Without these entries the loader rejects the model with
`native expert kernels do not take (types)`. The corresponding `Fmt<N>`
device templates already exist in the same translation unit; only the
dispatch tables required extension. Note that the internal type enumeration
used by these tables is the Strata project enumeration and differs from the
standard `GGMLQuantizationType` numbering; when extending the tables, derive
the numeric identifiers from the model file rather than from printed type
names.

### 2.2 HIP clang compatibility (`src/qwen35/q36_engine.cu`)

Two constructs accepted by nvcc and by recent HIP clang are rejected by
ROCm 6.3.3's clang:

- `__ldg(const int32_t*)` has no overload. Replaced with a plain load;
  equivalent on gfx906, where `__ldg` lowers to a regular global load.
- `__half_as_ushort` is device-only on this toolchain. The host-side use
  (a diagnostic dump) is replaced with a `__builtin_memcpy` bit-cast.

### 2.3 cuBLAS interface mapping
(`include/strata/platform/hip_compat/cublas_v2.h`)

The f32 projection path of the q36 engine calls `cublasSgemm` and
`cublasGemmStridedBatchedEx`. Both names are now mapped to their hipBLAS
equivalents in the compatibility header, consistent with the existing
mappings in that file.

### 2.4 Blackwell-only translation unit
(`CMakeLists.txt`)

`strata_mmq_w4a4` compiles `mmq_nvfp4_w4a4.cu` for the `sm_120a` target
family and has no gfx906 kernel image; on this target the enclosing library
is skipped (`if(NOT STRATA_HIP_GFX906)`) and the dependent link is guarded
with `if(TARGET strata_mmq_w4a4)`.

### 2.5 Missing translation unit (`src/prefill/moe_f16_kq.cu`)

`moe_fused_iq.cu` references the `f16kq::*` symbols provided by
`moe_f16_kq.cu`; the file is added to the `strata_mmq` source list. Its
device code paths are guarded by `STRATA_F16_DEVICE` and compile as
host-side stubs on gfx906.

## 3. Build

`scripts/build-gfx906.sh` in this branch reproduces the verified build:

```sh
cmake -S . -B build-906 -G "Unix Makefiles" -DSTRATA_HIP_GFX906=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx906 -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_GGML_DIR=<llama.cpp checkout at the pinned commit>
cmake --build build-906 -j$(nproc) strata-poly
```

On ROCm releases that no longer ship gfx906 libraries, the community image
`mixa3607/rocm-gfx906:7.14-complete` (HIP 7.14, rocBLAS with gfx906
kernels) provides a compatible toolchain, as described in `docs/AMD_HIP.md`.

## 4. Verified configuration

| Item | Value |
| --- | --- |
| Device | Instinct MI50 16 GB (gfx906, wave64) |
| ROCm | 6.3.3 (system HIP clang) |
| Model | Qwen3.6-35B-A3B, UD-Q2_K_XL (12.57 GB, GGUF) |
| Context | 262144 tokens (KV cache resident in VRAM) |
| Expert residency | Hybrid: VRAM hot set, pinned host pool, CPU fallback |
| Decode | 26.3 tok/s sustained, short-prompt measurement |

## 5. Known limitations

1. **KV cache residency.** The q36 engine does not implement KV streaming or
   reduced-precision KV variants for this path. At 262144 tokens the KV cache
   occupies approximately 12.7 GB of VRAM, which constrains the expert cache
   on 16 GB devices (approximately 11 of 256 expert slots per layer). A
   32 GB device, or upstream KV streaming support for the q36 engine,
   removes this constraint.
2. **MTP draft block.** The blk.40 draft layer of the release artifacts uses
   Q2_K (gate/up) and Q3_K_S (down) expert tensors, for which native decode
   kernels are not implemented (`Fmt<10>`/`Fmt<11>` templates do not exist in
   the translation unit). Two options: remove the draft block at pack time
   and run without MTP, or implement the corresponding grouped kernels.
3. **Toolchain sensitivity.** The fixes in §2.2 target the ROCm 6.3.3 clang;
   builds against HIP 7.14 have not been revalidated with the extended type
   tables and should be re-benchmarked.

## 6. References

- Upstream project: VecSzn/PolyStrata.
- gfx906 groundwork: Niko1221/Strata, `docs/AMD_HIP.md`
  (wave64 compat layer; attention switches `STRATA_GFX906_ATTN_QUERY_SWIZZLE`,
  `STRATA_GFX906_ATTN_REDUCE12`, PRs #1661 and #1718 by 0FL01).
- Community toolchain image: `mixa3607/rocm-gfx906:7.14-complete`.
