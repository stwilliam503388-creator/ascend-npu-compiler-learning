# 02 — ascendnpu-ir 快速上手

> 目标：从 Toy Tutorial 过渡到真实的 MLIR 编译器项目
> 前置：[01 — Toy Tutorial 导读](./01-Toy-Tutorial导读.md)
> 预估时间：30 分钟

## 1. ascendnpu-ir 是什么？

一个基于 MLIR 的 **Ascend NPU 编译器后端**。它做的事情：

```
linalg.generic (高级矩阵运算)
       ↓  Lowering
husion.elemwise_binary (昇腾融合 IR)
       ↓  Lowering
hivm.vadd (昇腾虚拟指令)
       ↓  代码生成
Ascend NPU 可执行文件
```

和 Toy Tutorial 的对比：

| | Toy Tutorial | ascendnpu-ir |
|---|-------------|-------------|
| 规模 | 教学项目，~1000 行 | 工业项目，750+ 源文件 |
| Dialect | 1 个 (`toy`) | 多个 (`husion`, `hivm` 等) |
| 目标 | CPU (JIT) | Ascend NPU |
| 真实度 | 玩具 | 华为生产环境在用 |

## 2. 获取代码

```bash
git clone https://github.com/Ascend/AscendNPU-IR.git
# 或用户 fork
git clone https://github.com/stwilliam503388-creator/ascendnpu-ir.git
```

## 3. 项目结构

```
ascendnpu-ir/
├── bishengir/               ← 核心编译器代码
│   ├── include/hir/         ← Dialect 定义 (TableGen)
│   ├── lib/Conversion/      ← Lowering Pass 实现
│   └── test/                ← 131 个测试用例
├── docs/                    ← 官方文档（中英文）
└── third-party/             ← LLVM/MLIR 依赖
```

## 4. 核心 Lowering 路径

```
linalg.generic (框架看到的)
     │  ConvertLinalgToHusion
     ▼
husion.elemwise_binary (融合优化，减少数据搬运)
     │  ConvertHusionToHIVM
     ▼
hivm.vadd / hivm.load / hivm.store (接近硬件)
     │  代码生成
     ▼
Ascend NPU 指令
```

**每一步的职责**：

| 步骤 | 做什么 | 类比 |
|------|--------|------|
| linalg → husion | 识别可融合的操作，合并 | "鸡肉+花生可以一起备料" |
| husion → hivm | 拆成 NPU 指令 | "切菜机怎么切，炒锅炒多久" |
| hivm → 可执行 | 分配寄存器、计算地址 | "哪锅先热，哪盘先出锅" |

## 5. 在哪里继续深入？

| 看什么 | 去哪 |
|--------|------|
| 完整 5 步 roadmap | [MLIR-L00](./MLIR-L00-速通与AscendNPU-IR实战.md) |
| 手写 MLIR Pass | [MLIR-L03](./MLIR-L03-自定义AscendNPU-IR-Pass实战.md) |
| 一键运行脚本 | [MLIR-L08](./MLIR-L08-ascendnpu-ir-demo可运行流水线.md) |
| 构建 ascendnpu-ir | [Phase 5 构建指南](../ascend/03-构建与调试指南.md) |

## 验证

- [ ] 知道 ascendnpu-ir 项目在哪、怎么看
- [ ] 能画出 linalg → husion → hivm 的路径
- [ ] 知道 husion 的核心概念是"融合"

> 📖 [术语表](../glossary.md)
> **下一步**：[Phase 4 — Triton-Ascend 编程实战](../triton-ascend/README.md)
