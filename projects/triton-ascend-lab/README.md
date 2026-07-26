# Triton-Ascend Lab

> Phase 4 动手项目：8 个递增 Triton kernel，覆盖 5 种算子模式

## 8 个 kernel 递进关系

```
kernel-1: vector_add      ← 最简：理解 block/grid + gm↔ub
kernel-2: softmax         ← 归约：多阶段算法
kernel-3: matmul_tile     ← 经典：tile 切分 + Cube Unit
kernel-4: fused_layernorm ← 融合：减少搬运
kernel-5: sigmoid         ← 逐元素激活：一行公式
kernel-6: gelu            ← 逐元素激活：多项式近似
kernel-7: rms_norm        ← 归一化 (新)：LLaMA 标配
kernel-8: group_norm      ← 归一化 (进阶)：2D grid + 双维度归约
```

## 5 种模式覆盖

| 模式 | kernel | 状态 |
|------|--------|:--:|
| 逐元素 | 1(vector_add), 5(sigmoid), 6(gelu) | ✅ 3/3 |
| 归约 | 2(softmax) | ✅ 1/1 |
| 矩阵乘 | 3(matmul_tile) | ✅ 1/1 |
| 归一化 | 4(layernorm), 7(rms_norm), 8(group_norm) | ✅ 3/3 |
| 融合 | 4(fused_layernorm) | ✅ 1/1 |

## 参考项目

- [FlagGems](https://github.com/FlagOpen/FlagGems) — 150+ Triton 算子参考实现
- kernel-5~8 均标注了对应的 FlagGems 源文件

## 运行

```bash
# 一键运行全部
bash run-all.sh

# 单个 kernel
cd kernel-1-vector-add
python3 kernel.py
```

## 每个 kernel 的三层讲解

| 层 | 内容 | 文件 |
|----|------|------|
| Python 源码 | 逐行注释 Triton kernel | `kernel.py` |
| TTIR dump | Triton MLIR IR 预期输出 | `expected.ttir` |
| Ascend 对照 | 该 kernel 在 Ascend NPU 上的映射 | `README.md` |

## 硬件对照速查

| Ascend 概念 | 影响你的代码 |
|------------|------------|
| gm (HBM) | 数据从 gm load 到 ub 才能算 |
| ub (L1 1MB) | BLOCK_SIZE 不能太大（受 ub 容量限制） |
| Cube Unit | `tl.dot()` 走 Cube，仅 matmul 类使用 |
| Vector Unit | 其他操作（add/mul/exp/reduce）走 Vector |
| SIMT/SIMD 双模 | 分支多的 kernel 可能有 divergence 开销 |

## 前置知识

- Phase 3 MLIR（理解 TTIR 格式）
- Phase 4 文档（Triton 编程模型 + Ascend NPU 概念）
