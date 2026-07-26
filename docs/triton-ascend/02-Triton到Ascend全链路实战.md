# 02 — Triton 到 Ascend 全链路实战

> 目标：跟踪一个 vector_add kernel 从 Python 到 hivm.hir 的完整路径
> 前置：[01 — Triton 编程模型与 Ascend NPU](./01-Triton编程模型与Ascend NPU.md)
> 预估时间：40 分钟

## 1. 起点：Python Triton Kernel

```python
# kernel-1-vector-add/kernel.py
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
```

**13 行代码。** 现在跟踪它如何编译。

## 2. 阶段一：Python AST → Triton IR (TTIR)

```
triton.jit 装饰器捕获 Python AST
     ↓
生成 Triton Dialect MLIR (TTIR)
```

```mlir
// dump: TRITON_DUMP_IR=1 python kernel.py
module {
  tt.func public @add_kernel(
    %arg0: !tt.ptr<f32>,
    %arg1: !tt.ptr<f32>,
    %arg2: !tt.ptr<f32>,
    %arg3: i32
  ) {
    %c64_i32 = arith.constant 64 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c64_i32 : i32
    %2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %3 = arith.addi %1, %2 : tensor<64xi32>
    %4 = tt.splat %arg3 : i32 -> tensor<64xi32>
    %5 = arith.cmpi slt, %3, %4 : tensor<64xi32>
    %6 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %7 = tt.addptr %6, %3 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %8 = tt.load %7, %5 : tensor<64x!tt.ptr<f32>>
    %9 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %10 = tt.addptr %9, %3 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %11 = tt.load %10, %5 : tensor<64x!tt.ptr<f32>>
    %12 = arith.addf %8, %11 : tensor<64xf32>
    %13 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %14 = tt.addptr %13, %3 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    tt.store %14, %12, %5 : tensor<64x!tt.ptr<f32>>
    tt.return
  }
}
```

**对照 Python 源码：**

| Python | TTIR | 说明 |
|--------|------|------|
| `pid = tl.program_id(0)` | `tt.get_program_id x` | 获取 program ID |
| `block_start = pid * BLOCK_SIZE` | `arith.muli %0, %c64_i32` | 计算起始偏移 |
| `offsets = block_start + tl.arange(0, BLOCK_SIZE)` | `tt.make_range` + `arith.addi` | 生成偏移数组 |
| `mask = offsets < n_elements` | `arith.cmpi slt` | 边界掩码 |
| `x = tl.load(x_ptr + offsets, mask=mask)` | `tt.addptr` + `tt.load` | 加载 |
| `output = x + y` | `arith.addf` | 实际计算 |
| `tl.store(output_ptr + offsets, output, mask=mask)` | `tt.addptr` + `tt.store` | 存储 |

**关键观察**：TTIR 是硬件无关的。没有任何 Ascend 特定操作。

## 3. 阶段二：TTIR → Ascend NPU IR

```
TTIR (硬件无关)
     ↓  Ascend Backend Passes
hivm.hir (Ascend 虚拟指令)
```

这是 triton-ascend 后端做的工作。核心转换：

| TTIR | hivm.hir | 含义 |
|------|---------|------|
| `tt.load` (从 HBM 读) | `hivm.hir.load` ins(gm) outs(ub) | **必须显式搬运 gm→ub** |
| `arith.addf` | `hivm.hir.vadd` | 向量加法（在 ub 上） |
| `tt.store` (写回 HBM) | `hivm.hir.store` ins(ub) outs(gm) | **必须显式搬运 ub→gm** |

```mlir
// hivm.hir 输出（简化）
func.func @add_kernel(%A: memref<64xf32, #hivm.address_space<gm>>,
                     %B: memref<64xf32, #hivm.address_space<gm>>,
                     %C: memref<64xf32, #hivm.address_space<gm>>) {
  // Step 1: gm → ub
  %ub_a = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
  hivm.hir.load ins(%A) outs(%ub_a)
  %ub_b = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
  hivm.hir.load ins(%B) outs(%ub_b)

  // Step 2: 在 ub 上计算
  %ub_c = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
  hivm.hir.vadd ins(%ub_a, %ub_b) outs(%ub_c)

  // Step 3: ub → gm
  hivm.hir.store ins(%ub_c) outs(%C)
  return
}
```

### 三阶段数据流

```
  HBM (gm)          L1 Buffer (ub)         Vector Unit
     │                    │                     │
  [A] ──load──→ [ub_a] ──┤                     │
  [B] ──load──→ [ub_b] ──┤                     │
                          ├─ vadd(ub_a,ub_b) ──→ [ub_c]
                          │                     │
  [C] ←──store── [ub_c] ←─┘                     │
```

**为什么比 GPU 多一步？** GPU 有三级缓存（Global → L2 → L1/Shared → Reg），Ascend 是 gm ↔ ub 直接搬运。更简单，但需要更精心的 tile 设计。

## 4. 阶段三：hivm.hir → LLVM IR → 机器码

```
hivm.hir  →  ConvertHivmToLLVM  →  llvm.func  →  毕昇编译器  →  .o
```

这是 Phase 5 的领域，暂时知道路径即可。

---

## 5. 对照：你自己的 kernel 怎么验证？

因为没有 Ascend 硬件，我们验证到 TTIR 层：

```bash
# 只 dump TTIR
TRITON_DUMP_IR=1 python kernel-1-vector-add/kernel.py 2>&1 | grep -A 50 "tt.func"

# 或者用 triton-ascend-lab 的一键脚本
cd projects/triton-ascend-lab
bash run-all.sh
```

对比 `expected.ttir` 和实际输出。

---

## 验证

- [ ] 能画出 Python → TTIR → hivm.hir → LLVM IR 四个阶段
- [ ] 知道 TTIR 中各操作的含义（tt.load / arith.addf / tt.store）
- [ ] 知道 hivm.hir.load 和 hivm.hir.vadd 的数据流向
- [ ] 知道为什么 Ascend 比 GPU 多显式搬运

> 📖 [术语表](../glossary.md)
> **下一步**：[03 — Ascend NPU 编程模式](./03-Ascend NPU编程模式.md)
