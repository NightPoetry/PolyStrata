# gfx906 (Instinct MI50 / MI60, Radeon VII) — HIP 移植与 Qwen3.6 实战笔记

> 本分支：`gfx906-hip-port` — 在 wave64 的 gfx906 上编译并实跑 strata-poly（Qwen3.6-35B-A3B，262144 满上下文 + 动态热专家缓存）的移植补丁与实战记录。
> 实测环境：Instinct MI50 16GB / ROCm 6.3.3 / Ubuntu（内核 6.x）/ 单卡。262K 满窗 26.3 tok/s 稳定（Q2_K_XL 件）。

## 为什么需要这个分支

- `STRATA_ENABLE_HIP`（wave32 路线）在 gfx906 上拒绝启动：gfx906 是 wave64、无 WMMA、无 packed byte arithmetic。
- 社区的 Qwen3.6 成绩全部来自 wave32 卡；gfx906 的内核类型表从没被真实需求逼过——本分支首次补齐并在 MI50 上实跑成功。

## 补丁内容（五处，全部已在本分支）

1. **`src/kernels/cuda/iq_kernels.cu` — 内核类型表扩表**：
   - `STRATA_GU_FMTS` 补 `X(20)`（IQ2_XS gate/up）
   - `STRATA_D_FMTS` 补 `X(18)`（IQ3_XXS down）与 `X(22)`（IQ2_S down）
   - Qwen3.6-35B-A3B 的 UD-Q2_K_XL 官方件 = Q6_K 主干专家 + IQ2_XS/IQ3_XXS 混合件，缺这几个枚举直接 `native expert kernels do not take` 拒载。
2. **`src/qwen35/q36_engine.cu` — HIP clang 6.x 兼容**：
   - `__ldg(const int32_t*)` 无重载 → 改普通解引用（gfx906 上 __ldg 本就是普通 load）
   - `__half_as_ushort` 无 host 版 → `__builtin_memcpy` 位拷贝 lambda
3. **`include/strata/platform/hip_compat/cublas_v2.h`**：补 `cublasSgemm → hipblasSgemm`、`cublasGemmStridedBatchedEx → hipblasGemmStridedBatchedEx` 映射（q36 引擎的 f32mm 路径要用）。
4. **`CMakeLists.txt`**：`strata_mmq_w4a4`（Blackwell NVFP4 W4A4 专用）在 gfx906 上无内核必炸 → `if(NOT STRATA_HIP_GFX906)` 包裹 + `if(TARGET ...)` 守卫链接。
5. **`src/prefill/moe_f16_kq.cu` 入库**（strata_mmq 源清单）：moe_fused_iq.cu 引用 f16kq 符号，缺它链接失败。

## 构建（MI50 实测配方）

```bash
scripts/build-gfx906.sh   # 见脚本内注释；要点:
# -DSTRATA_HIP_GFX906=ON -DCMAKE_HIP_ARCHITECTURES=gfx906
# 老 ROCm(6.3.3) 的 HIP clang 对 __ldg/alignas 较严——本分支已兼容
# 新 ROCm(7.14) 用户沿用 docs/AMD_HIP.md 的 docker 路线亦可
```

## 实跑记录（MI50 16GB）

```
Qwen3.6-35B-A3B UD-Q2_K_XL (12.57 GB, MTP block 内嵌):
  262144 满上下文装载 ✓ (262K KV 全驻 VRAM 12.7G)
  动态热专家缓存 ✓ (hybrid: VRAM 热集 + pinned host 内存池 + 8 线程 CPU 兜底)
  26.3 tok/s decode 稳定 (短 prompt; 起始装载 30-40s)
```

## 已知边界

- **262K KV 全驻 VRAM**：q36 引擎暂无 KV streaming/k8v4——16G 卡上专家缓存被 KV 挤压（热专家仅 ~11/256 层）。32G 卡（MI50 32G）或上游给 q36 补 KV streaming 可解。
- **MTP**：UD-Q4/Q3 官方件的 blk.40 草稿层（Q2_K/Q3_K_S 专家）暂无内核——用 --mtp 需为这两个类型补核，或摘除草稿层（本分支实测摘除可跑）。
- gfx906 内核类型表历史上无人维护——遇到新的 "kernels do not take (type)" 报错，同法扩 `STRATA_GU_FMTS`/`STRATA_D_FMTS` 即可（模板 `Fmt<N>` 多数已存在）。

## 致谢与上游

基于 VecSzn/PolyStrata（其上承 Niko1221/Strata 的 gfx906 先驱工作，PR #1661/#1718 by 0FL01）。
本分支的五处补丁与 MI50 实跑数据欢迎回流上游。
