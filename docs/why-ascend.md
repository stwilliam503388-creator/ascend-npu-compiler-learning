# 为什么学 Ascend NPU 编译器？

> 5 分钟读完，帮你判断这个项目值不值得投入时间

---

## 学编译器后端能干什么？

很多人觉得编译器是"造轮子的人"才需要学的东西。其实只要你在做 AI 相关的工程工作，编译器知识每天都在帮你——只是你可能没意识到。

### 场景 1：看懂 PyTorch 报错里的 IR dump

```python
# 模型训练崩了，PyTorch 吐出一大段 IR：
# %4 = linalg.matmul ins(%2, %3) -> ...
# RuntimeError: lowering failed at linalg.generic
```

如果你不懂 IR，只能复制粘贴问 ChatGPT。如果你懂 IR，一眼看出是 `matmul` 的 lower 出了问题。

### 场景 2：给 AI 框架写自定义算子

模型里有个 PyTorch 没有的算子？你要用 Triton/CUDA 自己写。写完后，编译器得能把它正确 lower 到硬件指令。这时候你写的不只是算子，是"编译器能理解的算子描述"。

### 场景 3：看懂 Triton kernel 的编译流程

```python
@triton.jit
def add_kernel(x, y, z):
    # 这 5 行 Python，Triton 编译器生成了几百行 MLIR + LLVM IR
```

Triton 是目前最火的 AI 编译器之一。看懂它的编译流程，才能写出真正高效的 kernel。

### 场景 4：为自家硬件写编译器后端

越来越多的 AI 芯片公司需要"从 MLIR 到自家 NPU"的 Lowering。这不像写 Web 后端——整个市场上能做编译器后端的人非常少，供需严重失衡。

### 场景 5：参与开源编译器项目

LLVM/MLIR 是业界最活跃的开源编译器基础设施。一个 LLVM Pass 的 PR 可能被 Clang/Rust/Swift 等十几个语言用上——影响力远超普通业务代码。

---

## 为什么选 Ascend NPU？

| 对比维度 | CUDA 生态 | Ascend 生态 |
|---------|----------|------------|
| 硬件覆盖 | NVIDIA GPU 一家 | 华为昇腾系列（推理+训练） |
| 市场规模 | 全球最大 | 国内增长最快 |
| 学习曲线 | 陡峭（PTX 汇编难） | 相对平缓（MLIR 抽象层更高） |
| 社区成熟度 | 极其成熟 | 快速成长中 |
| 技能可迁移性 | 局限于 NVIDIA | MLIR 知识可用于任何硬件后端 |
| 岗位竞争 | 极其激烈 | 相对蓝海 |

**关键优势**：Ascend 编译器后端基于 MLIR。MLIR 是 LLVM 的作者 Chris Lattner 设计的下一代编译器框架，是编译器领域的通用基础设施。**学到的 MLIR 知识可以迁移到任何其他硬件后端**，而学 CUDA 的经验只有 NVIDIA GPU 能用。

---

## 这个项目能带你走到哪？

```
你现在                  Phase 1+2 后              Phase 3+4 后              Phase 5 后
  ↓                        ↓                          ↓                         ↓
 零基础          能写 LLVM Pass          能写 Triton kernel      能开发 Ascend 编译器后端
             能读懂 + 修改 IR          能写自定义 MLIR Dialect
             理解 Pass 运作机制         能参与开源编译器项目
```

**对标岗位**：
- 编译器开发工程师（LLVM/MLIR）
- AI 框架开发工程师（PyTorch/TensorFlow 底层）
- NPU 算子开发工程师（CANN/TBE）
- 高性能计算工程师（HPC）

---

## 学习建议

1. **先建立信心** — 直接跳到 `projects/hello-pass/`，跑通你的第一个 Pass
2. **概念不用一次全懂** — IR、Pass、Lowering 这些词第一次看不懂没关系，回头查 [术语表](./glossary.md)
3. **动手 > 阅读** — 每学一个概念就写代码验证，不要只看文档
4. **预计时间** — Phase 1（45 分钟）+ Phase 2（60 分钟）+ Phase 3（120 分钟）+ Phase 4（110 分钟）≈ 5.5 小时入门

---

## 开始学习

→ **[快速入门](./quickstart.md)** — 从环境检查到跑通 HelloPass，一条路走到底

→ **[Primer 入门](./primer/README.md)** — 先学概念？从这里开始
