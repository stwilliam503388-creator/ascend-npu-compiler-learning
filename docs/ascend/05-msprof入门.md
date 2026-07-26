# 05 — msprof 性能分析入门

> 目标：理解 Ascend NPU kernel 性能分析的基本方法
> 前置：[Phase 4 Triton-Ascend](../triton-ascend/README.md)
> 预估时间：10 分钟

## msprof 是什么？

昇腾 NPU 的性能分析工具（类比 NVIDIA nsight / nsys）。

```
msprof --application="python my_triton_kernel.py"
  → 输出: 时间线、算子耗时、内存搬运、带宽利用率
```

## 核心指标

| 指标 | 含义 | 健康值 |
|------|------|--------|
| **aicore_time** | 算子实际计算时间 | 越短越好 |
| **dma_time** | gm↔ub 数据搬运时间 | < aicore_time |
| **bandwidth_utilization** | HBM 带宽利用率 | > 60% |
| **cube_utilization** | Cube Unit 利用率 | matmul: > 80%, 其他: ~0% |
| **vec_utilization** | Vector Unit 利用率 | element-wise: > 60% |

## 典型优化场景

### 场景 1：dma_time 远大于 aicore_time

```
诊断：数据搬运是瓶颈（受限于 HBM 带宽，~1.2 TB/s）
方程：
  - 融合算子：减少 kernel 启动 + gm↔ub 次数
  - 增大 BLOCK_SIZE：每次搬运更多数据，摊销启动开销
```

### 场景 2：cube_utilization 低，但 kernel 有 matmul

```
诊断：tile 太小，Cube Unit 空闲等数据
方程：
  - 增大 BLOCK_M/BLOCK_N/BLOCK_K
  - 确保 tile 大小足够喂饱 16×16 Cube Unit
```

### 场景 3：vec_utilization 低

```
诊断：Vector Unit 没跑满（分支多、归约同步开销大）
方程：
  - 减少 kernel 内的 if/else 分支
  - 归约类算子：分 tile 做 partial reduction，减少同步
```

## 与 Triton 开发流程的对应

```
30 个任务书的三段式：

开发（写出正确 kernel）
  → 验证：数值精度 (torch.allclose)
  → 不需要 msprof

迁移（GPU→Ascend 适配）
  → 验证：IR 对照 (TTIR → hivm.hir)
  → 轻量 msprof：确认 Tile 大小合理

优化（性能调优）
  → 重 msprof：aicore/dma/bandwidth 全分析
  → autotune：遍历 BLOCK_SIZE 组合找最优
```

## autotune 示例

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=16),
    ],
    key=['n_elements'],
)
@triton.jit
def my_kernel(..., BLOCK_SIZE: tl.constexpr):
    ...
```

## 快速参考

```
# 性能分析
msprof --output=./prof_data --application="python kernel.py"

# 查看结果
msprof --export=on --output=./prof_data --iteration-id=1

# 关键输出文件
#   prof_data/summary.csv    ← 各算子耗时汇总
#   prof_data/timeline.json  ← 时间线（可视化）
```

> 📖 [术语表](../glossary.md)
> 📖 [04 — Triton vs AscendC 对比](./04-AscendC对比.md)
