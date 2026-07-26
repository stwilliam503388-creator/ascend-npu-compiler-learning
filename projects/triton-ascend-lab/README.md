# Triton-Ascend Lab

> Phase 4 动手项目：4 个递增 Triton kernel，从入门到融合优化

## 4 个 kernel 递进关系

```
kernel-1: vector_add        ← 最简：理解 block/grid + gm↔ub
    ↓
kernel-2: softmax           ← 归约：多阶段算法 + warp reduce
    ↓
kernel-3: matmul_tile       ← 经典：tile 切分 + Cube Unit 利用
    ↓
kernel-4: fused_layernorm   ← 融合：减少 gm↔ub 搬运 + SIMT/SIMD 双模
```

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
