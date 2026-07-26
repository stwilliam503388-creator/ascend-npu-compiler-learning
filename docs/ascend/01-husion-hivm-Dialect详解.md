# 01 — hivm 和 hacc Dialect 详解

> 目标：理解 AscendNPU-IR 的真实 Dialect 设计
> 前置：[00 — Ascend NPU 硬件概述](./00-Ascend-NPU硬件概述.md)
> 预估时间：25 分钟

## 1. AscendNPU-IR 的真实 Dialect 结构

从源码和测试用例中提取的实际 Dialect：

```
linalg.generic (MLIR 标准 — 框架层描述)
     │  ConvertLinalgToHivm (conversion pass)
     ▼
hivm.hir.load / hivm.hir.vadd / hivm.hir.store (Huawei IR)
     │  ConvertHivmToLLVM
     ▼
llvm.func (最终代码)
```

**和常见 MLIR 教程不同**：AscendNPU-IR 没有中间的"husion"融合方言。融合优化在 `linalg` 层完成（`linalg.elemwise_binary`、`linalg.reduce`），然后直接 Lowering 到 `hivm.hir`。

## 2. hivm Dialect — 华为虚拟指令集

**全称**：Huawei Instruction Virtual Machine

### 核心操作（从真实测试用例提取）

```mlir
// 从 Global Memory 加载到 Unified Buffer
%ub = memref.alloc() : memref<16xf16, #hivm.address_space<ub>>
hivm.hir.load ins(%gm_buf : memref<16xf16, #hivm.address_space<gm>>)
              outs(%ub : memref<16xf16, #hivm.address_space<ub>>)

// 向量加法
%result = memref.alloc() : memref<16xf16, #hivm.address_space<ub>>
hivm.hir.vadd ins(%ub_a, %ub_b : memref<16xf16, #hivm.address_space<ub>>,
                           memref<16xf16, #hivm.address_space<ub>>)
              outs(%result : memref<16xf16, #hivm.address_space<ub>>)

// 存回 Global Memory
hivm.hir.store ins(%result : memref<16xf16, #hivm.address_space<ub>>)
               outs(%gm_out : memref<16xf16, #hivm.address_space<gm>>)
```

### 地址空间

| 空间 | 缩写 | 含义 | 厨房类比 |
|------|------|------|---------|
| `#hivm.address_space<gm>` | Global Memory | HBM 显存 | 冰箱 |
| `#hivm.address_space<ub>` | Unified Buffer | 片上缓存 | 案板 |

**关键点**：数据搬运不是自动的。编译器必须显式用 `hivm.hir.load`/`hivm.hir.store` 在 gm ↔ ub 之间搬数据。

### 完整 Lowering 示例（真实代码）

**输入**（linalg）：
```mlir
func.func @add(%A: tensor<16xf16>, %B: tensor<16xf16>) -> tensor<16xf16> {
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%A, %B : tensor<16xf16>, tensor<16xf16>)
    outs(%A : tensor<16xf16>) {
  ^bb0(%a: f16, %b: f16, %c: f16):
    %add = arith.addf %a, %b : f16
    linalg.yield %add : f16
  } -> tensor<16xf16>
  func.return %0 : tensor<16xf16>
}
```

**输出**（hivm，来源：`bishengir/test/Integration/HIVM/VecAdd/add.mlir`）：
```mlir
func.func @add(%A: memref<16xf16, #hivm.address_space<gm>>,
               %B: memref<16xf16, #hivm.address_space<gm>>,
               %C: memref<16xf16, #hivm.address_space<gm>>)
    attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %ub_a = memref.alloc() : memref<16xf16, #hivm.address_space<ub>>
  hivm.hir.load ins(%A) outs(%ub_a)
  %ub_b = memref.alloc() : memref<16xf16, #hivm.address_space<ub>>
  hivm.hir.load ins(%B) outs(%ub_b)
  %ub_c = memref.alloc() : memref<16xf16, #hivm.address_space<ub>>
  hivm.hir.vadd ins(%ub_a, %ub_b) outs(%ub_c)
  hivm.hir.store ins(%ub_c) outs(%C)
  return
}
```

**关键变化**：
- `tensor` → `memref`（显式内存管理）
- 隐式数据搬运 → 显式 `load` + `store` + `alloc`
- 新增 `hacc.entry` + `hacc.function_kind` kernel 属性

## 3. hacc Dialect — 华为加速器属性

| 属性 | 含义 |
|------|------|
| `hacc.entry` | 标记为 kernel 入口函数 |
| `hacc.function_kind<DEVICE>` | 在 NPU 上执行 |
| `hacc.function_kind<HOST>` | 在 Host CPU 上执行 |
| `hacc.host_func_type<host_entry>` | Host 端入口函数类型 |

## 4. 融合在哪发生？

融合优化在 `linalg` 层完成，使用标准 MLIR 操作：

```mlir
// linalg.reduce — 规约（sum over dimension）
%1 = linalg.reduce {arith.addf} ins(%A: tensor<?x5xf16>)
    outs(%init: tensor<5xf16>) dimensions = [0]

// linalg.elemwise_binary — 逐元素二元操作
%2 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
    ins(%1, %2: tensor<5xf16>, tensor<5xf16>)
    outs(%init: tensor<5xf16>) -> tensor<5xf16>
```

这些操作会被 `ConvertLinalgToHivm` pass 识别并 Lowering 为 `hivm.hir.*` 指令。

## 5. 和 Phase 3/4 的衔接

| 学过的 | AscendNPU-IR 怎么用 |
|--------|-------------------|
| LLVM Pass（hello-pass） | ConvertLinalgToHivm 也是 Pass，思路一样 |
| MLIR Pass（mlir-hello） | 同样使用 `OpRewritePattern` 匹配 → 替换 |
| MLIR Dialect | hivm 和 hacc 都是 Dialect |
| SSA | hivm IR 全部 SSA |

## 验证

- [ ] 能说出 hivm.hir.load/store 的作用
- [ ] 知道 `gm` 和 `ub` 两个地址空间的含义
- [ ] 能画出 linalg → hivm.hir 的 Lowering 路径
- [ ] 知道 hacc.entry 是干什么的

> 📖 [术语表](../glossary.md)
> **下一步**：[02 — 一个 Ascend Pass 详解](./02-一个Ascend-Pass详解.md)
