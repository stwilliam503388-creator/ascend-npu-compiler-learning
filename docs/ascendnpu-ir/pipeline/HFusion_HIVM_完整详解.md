# 昇腾 BiSheng Compiler 编译器管线详解（完整版）

> 基于 AscendNPU-IR 源码深度分析，覆盖 HFusion（融合优化）+ HIVM（硬件映射）两条核心管线的完整实现。
> 所有文件路径均为工程绝对路径，根目录: `AscendNPU-IR`

---

## 项目概述：AscendNPU-IR

### 项目简介

AscendNPU IR（AscendNPU Intermediate Representation）是华为基于 MLIR 构建的、面向昇腾 AI 处理器的中间表示编译器。项目代号 "BiSheng"（毕昇），专为昇腾 910 系列 NPU（Da Vinci 架构）设计，提供从高层算子融合到底层硬件指令映射的完整编译管线。

- **仓库**: [GitCode](https://gitcode.com/Ascend/AscendNPU-IR) | [GitHub](https://github.com/Ascend/AscendNPU-IR)
- **许可**: Apache License 2.0
- **代码量**: ~223,000 行（不含 third-party/LLVM），714 个源文件
- **主语言**: C++（编译器核心）、TableGen（MLIR 方言定义）、Python（绑定/测试）、CMake（构建系统）

### 编译定位

```
Torch / Linalg / GPU IR (生态框架)
        │
        ▼ AscendNPU-IR (本项目)
   ┌─────────────────────────────┐
   │  HFusion Dialect  (融合层)   │  ← 算子融合 + AutoSchedule
   │  HIVM Dialect     (硬件层)   │  ← 内存规划 + 同步求解 + Tiling
   │  Template Lib     (模板库)   │  ← 向量运算模板 (C++ 模板元编程)
   │  ExecutionEngine  (执行引擎) │  ← JIT 编译 + 运行时
   └──────────┬──────────────────┘
              │
              ▼
   CANN / Ascend Runtime (昇腾运行时)
              │
              ▼
   Ascend 910 NPU (Da Vinci 架构)
```

### 仓库结构 (完整路径)

```

├── CMakeLists.txt                          # 顶层构建文件
├── bishengir/                              # ★ 核心源码 (~216,000 行)
│   ├── cmake/                              # CMake 模块
│   ├── include/bishengir/                  # 公共头文件
│   │   ├── Dialect/HFusion/                #   HFusion Dialect 定义
│   │   │   ├── IR/                         #     Ops, Types, Attrs (TableGen)
│   │   │   ├── Transforms/                 #     优化 Pass 头文件
│   │   │   │   ├── AutoSchedule/           #       自动调度器头文件
│   │   │   │   └── OpFusion/               #       算子融合头文件
│   │   │   └── Pipelines/                  #     管线头文件
│   │   ├── Dialect/HIVM/                   #   HIVM Dialect 定义
│   │   │   ├── IR/                         #     Ops, Interfaces (TableGen)
│   │   │   ├── Transforms/                 #     优化 Pass 头文件
│   │   │   │   ├── GraphSyncSolver/        #       图同步求解器头文件 (8 文件)
│   │   │   │   └── InjectSync/             #       同步注入头文件 (8 文件)
│   │   │   └── Pipelines/                  #     管线头文件
│   │   ├── Dialect/Tensor/                 #   Tensor Dialect
│   │   ├── Conversion/                     #   转换 Pass 头文件
│   │   └── ...
│   ├── lib/                                # 源码实现
│   │   ├── Dialect/HFusion/                #   HFusion 实现 (133 文件, ~50K 行)
│   │   │   ├── IR/                         #     Dialect 注册 + Ops 实现
│   │   │   ├── Transforms/                 #     优化 Pass 实现 (94 个文件)
│   │   │   │   ├── Normalize.cpp           #       9,479 行 — 核心标准化
│   │   │   │   ├── OpFusion.cpp            #       融合入口 (322 行)
│   │   │   │   ├── OpFusion/               #       融合子模块 (5 文件)
│   │   │   │   ├── AutoSchedule/           #       调度器实现 (8 文件)
│   │   │   │   └── ...                     #       80+ 其他 Pass
│   │   │   └── Pipelines/                  #     管线编排 (325 行)
│   │   ├── Dialect/HIVM/                   #   HIVM 实现 (241 文件, ~85K 行)
│   │   │   ├── IR/                         #     Dialect 注册 + Ops 实现
│   │   │   ├── Transforms/                 #     优化 Pass 实现 (60+ Pass)
│   │   │   │   ├── PlanMemory.cpp          #       2,884 行 — 内存规划
│   │   │   │   ├── GraphSyncSolver/        #       图同步求解器 (8 文件)
│   │   │   │   ├── InjectSync/             #       同步注入 (8 文件)
│   │   │   │   └── ...                     #       50+ 其他 Pass
│   │   │   └── Pipelines/                  #     管线编排 (428 行)
│   │   ├── Conversion/                     #   Dialect 间转换实现 (~80 文件)
│   │   ├── Template/                       #   向量模板库 (~60 文件)
│   │   └── ExecutionEngine/                #   执行引擎
│   ├── tools/                              # 二进制工具
│   │   └── bishengir-compile/              #   主编译器入口
│   ├── python/                             # Python 绑定 (PyBind11)
│   ├── test/                               # 测试用例
│   └── unittests/                          # 单元测试
├── bishengir-demo/                         # 演示项目 (独立子项目)
├── build-tools/                            # 构建脚本 + LLVM 补丁
├── docs/                                   # Sphinx 文档工程
└── third-party/                            # 第三方依赖 (llvm-project/torch-mlir/shmem)
```

### 核心 Dialect 层级关系

```
┌──────────────────────────────────────────────────────┐
│                  高层生态 Dialect                       │
│   linalg, tosa, torch, tensor, arith, math, gpu       │
└──────────────────────┬───────────────────────────────┘
                       │ Conversion Pass (7 个输入)
                       ▼
┌──────────────────────────────────────────────────────┐
│            HFusion Dialect (融合优化层)                │
│   hfusion.elemwise_binary, hfusion.matmul,            │
│   hfusion.reduce, hfusion.broadcast, ...              │
│   ─────────────────────────────────────────           │
│   Pass 管线: lower-hfusion-pipeline (7 阶段)           │
│   核心算法: OpFusion + AutoSchedule (4 调度策略)       │
│   输出: 轮廓化 kernel 函数                             │
└──────────────────────┬───────────────────────────────┘
                       │ HFusionToHIVM Conversion
                       ▼
┌──────────────────────────────────────────────────────┐
│             HIVM Dialect (硬件映射层)                  │
│   hivm.v_add, hivm.mmad, hivm.v_reduce,              │
│   hivm.set_wait, hivm.set_flag, hivm.pipe_barrier     │
│   ─────────────────────────────────────────           │
│   Pass 管线: convert-to-hivm + optimize-hivm (2 条)   │
│   核心算法: PlanMemory + GraphSyncSolver + Bufferize  │
│   输出: Standard MLIR + 硬件同步 intrinsic             │
└──────────────────────┬───────────────────────────────┘
                       │ HIVMToStandard
                       ▼
┌──────────────────────────────────────────────────────┐
│    Standard MLIR + Hardware Intrinsic                  │
│   scf.for + memref.load/store + arith                 │
│   + hivm.set_wait/set_flag (保留)                     │
└──────────────────────┬───────────────────────────────┘
                       │
                       ▼
              LLVM IR / Ascend Binary
```

### 编译工具入口

编译入口位于 `bishengir/lib/Tools/bishengir-compile/PassPipeline.cpp`，主调用链:

```
main() → runBishengirCompile91095()
  ├─ buildHFusionPipelines()         # HFusion 优化管线
  │    └─ "lower-hfusion-pipeline"   # 7 阶段, 30+ Pass
  ├─ buildConvertToHIVMPipeline()    # HFusion → HIVM 转换
  │    └─ "convert-to-hivm-pipeline" # 4 Pass
  └─ buildOptimizeHIVMPipeline()    # HIVM 优化管线
       └─ "optimize-hivm-pipeline"   # 5 阶段, 60+ Pass
```

### 代码规模统计

| 模块 | 文件数 | 代码行数 | 占比 | 说明 |
|------|--------|---------|------|------|
| HIVM Dialect | 241 | ~85,500 | 38% | 硬件映射层 (最大子模块) |
| HFusion Dialect | 133 | ~50,300 | 23% | 融合优化层 |
| Conversion | ~80 | ~35,000 | 16% | 7 个输入 + 2 个输出转换桥 |
| Template Lib | ~60 | ~25,000 | 11% | 向量运算 C++ 模板 |
| 工具/测试/Python/文档 | ~200 | ~20,000 | 12% | 构建/测试/绑定 |
| **总计 (bishengir/)** | **~714** | **~216,000** | **100%** | 不含 third-party |

---

## 目录

- [全局数据流图](#零全局数据流图)
- [Part A: HFusion 融合优化管线](#part-a-hfusion-融合优化管线)
  - [A1. 管线入口与 Pass 注册](#a1-管线入口与-pass-注册)
  - [A2. 核心类型系统](#a2-核心类型系统)
  - [A3. 融合推断算法](#a3-融合推断算法)
  - [A4. 融合块分析器](#a4-融合块分析器)
  - [A5. 算子轮廓化](#a5-算子轮廓化)
  - [A6. 自动调度器](#a6-自动调度器)
  - [A7. HFusion 数据流链路](#a7-hfusion-数据流链路)
- [Part B: HIVM 硬件映射管线](#part-b-hivm-硬件映射管线)
  - [B1. 双管线架构](#b1-双管线架构)
  - [B2. 内存规划算法](#b2-内存规划算法)
  - [B3. 图同步求解器](#b3-图同步求解器)
  - [B4. 自动同步注入](#b4-自动同步注入)
  - [B5. Lowering 到 Standard MLIR](#b5-lowering-到-standard-mlir)
  - [B6. HIVM 数据流链路](#b6-hivm-数据流链路)
- [附录：完整循环链路](#附录完整循环链路)

---

## 零、全局数据流图

```
                    ┌─────────────────────────────────────┐
                    │       bishengir-compile 工具入口       │
                    │   PassPipeline.cpp:46-85             │
                    └──────────┬──────────────────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        ▼                      ▼                      ▼
  ┌──────────┐    ┌──────────────────┐    ┌───────────────────┐
  │ 高层 IR   │───▶│ lower-hfusion    │───▶│ optimize-hivm     │
  │ Torch/   │    │ -pipeline        │    │ -pipeline         │
  │ Linalg/  │    │ (7 阶段, 30+ Pass)│    │ (5 阶段, 60+ Pass)│
  │ GPU      │    └────────┬─────────┘    └────────┬──────────┘
  └──────────┘             │                       │
                           ▼                       ▼
                    ┌─────────────┐         ┌──────────────┐
                    │  HFusion IR │────────▶│   HIVM IR    │
                    │ (虚拟Tensor) │  转换    │ (指令级抽象)  │
                    └─────────────┘         └──────┬───────┘
                                                   │ 最终 lowering
                                                   ▼
                                           ┌───────────────┐
                                           │ Standard MLIR │
                                           │ + hw intrinsic│
                                           └───────┬───────┘
                                                   │
                                                   ▼
                                           ┌───────────────┐
                                           │  LLVM IR /    │
                                           │ Ascend Binary │
                                           └───────────────┘

数据形态演变:
  linalg.generic ─→ hfusion.elemwise_binary ─→ hivm.v_add ─→ scf.for + memref.load/store
       ↑                    ↑                       ↑                  ↑
   高层语义           融合优化层              硬件指令层          标准 MLIR
  (无内存概念)      (Tensor SSA)          (MemRef 地址)     (完全 lowering)
```

---

# Part A: HFusion 融合优化管线

## A1. 管线入口与 Pass 注册

### 入口函数

**文件**: `bishengir/lib/Dialect/HFusion/Pipelines/HFusionPipelines.cpp:279`

```cpp
void buildHFusionPipelines(OpPassManager &pm,
                           const HFusionPipelineOptions &options) {
  // 阶段1: 预处理
  preProcess(pm, options);                         // → L119
  canonicalizationPipeline(pm, options);

  if (!options.enableTritonKernelCompile) {
    if (!options.enableMultiKernel) {
      // 阶段2-3: 标准单 kernel 路径
      preFlattenPass(pm, options);                  // → L117
      flattenAndFold(pm, options);                  // → L153
    }
    // 阶段4: 推断 + 融合 + 轮廓化
    inferAndOutlineOp(pm, options);                 // → L177
    postProcessOutlinedKernel(pm);                   // → L143

    if (options.enableMultiKernel) {
      preFlattenPass(pm, options);
      flattenAndFold(pm, options);
    }
    // 阶段6: 自动调度
    hfusionAutoSchedulePipeline(pm, options);       // → L217
  } else {
    // Triton 特殊路径 (跳过融合)
    // ...
  }

  canonicalizationPipeline(pm, options, AfterAutoSchedule);
  // 阶段7: 后处理
  postProcess(pm, options);                         // → L257
}
```

### Pass 注册

```cpp
// HFusionPipelines.cpp:316-322
void registerLowerHFusionPipelines() {
  PassPipelineRegistration<HFusionPipelineOptions>(
      "lower-hfusion-pipeline",           // ← 管线名
      "lower hfusion pipeline",
      buildHFusionPipelines               // ← 入口函数指针
  );
}
```

### 调用链

```
bishengir-compile (PassPipeline.cpp:76)
  └─ buildHFusionPipelines(pm, options)
       ├─ preProcess(pm, options)
       │    ├─ createArithToHFusionConversionPass()
       │    ├─ createMathToHFusionConversionPass()
       │    ├─ createLinalgToHFusionConversionPass()
       │    ├─ createNormalizeOpsPass()           ← 9,479行
       │    └─ createDecomposePass()
       ├─ preFlattenPass(pm, options)
       │    └─ createComposeMultiReduce()
       ├─ flattenAndFold(pm, options)
       │    └─ createFlattenOpsPass()
       ├─ inferAndOutlineOp(pm, options)
       │    ├─ createInferFuncFusionKind()        ← 推断策略
       │    └─ createHFusionOpFusionPass()        ← 执行融合
       ├─ hfusionAutoSchedulePipeline(pm, options)
       │    └─ createHFusionAutoSchedulePass()    ← 调度器分发
       └─ postProcess(pm, options)
```

---

## A2. 核心类型系统

### FusionKind 枚举定义

**文件**: `bishengir/include/bishengir/Dialect/HFusion/IR/HFusionEnums.td:203-215`

```tablegen
def HFusion_FusionKindEnum :
  HFusion_I32Enum<"FusionKind", "HFusion fused kernel kind", [
  HFUSION_FUSION_KIND_PURE_ELEMWISE,     // 1: 纯逐元素
  HFUSION_FUSION_KIND_ANY_PB,            // 2: 任意 Parallel+Broadcast
  HFUSION_FUSION_KIND_LAST_AXIS_PBR,     // 3: 末轴 P+B+Reduce
  HFUSION_FUSION_KIND_ANY_PBR,           // 4: 任意 P+B+Reduce
  HFUSION_FUSION_KIND_SINGLE_CUBE,       // 5: 单 Cube 算子
  HFUSION_FUSION_KIND_SHALLOW_CV,        // 6: 浅层 Cube+Vector
  HFUSION_FUSION_KIND_SHALLOW_VV,        // 7: 浅层 Vector+Vector
  HFUSION_FUSION_KIND_MIX_CV,            // 8: 混合 Cube+Vector
  HFUSION_FUSION_KIND_MIX_C2,            // 9: 混合双 Cube
  HFUSION_FUSION_KIND_UNKNOWN,           // 10: 未知
]> {}
```

C++ 端对应类型: `mlir::hfusion::FusionKind`

### OpPattern 枚举

**文件**: `bishengir/include/bishengir/Dialect/HFusion/Transforms/OpFusion/FusibleHelper.h:44-68`

```cpp
enum class OpPattern : uint8_t {
  kAuxiliary = 0,         // 辅助 op (arith 等)
  kBuffer = 1,            // buffer 分配 op
  kOpaque = 2,            // 不可穿透 op
  kZeroRankElemwise = 3,  // 0阶逐元素
  kReshape = 50,          // reshape
  kLoadStore = 51,        // load/store
  kInsertSlice = 52,      // insert_slice
  kMidFusionAuxiliary = 54,

  // >= 100 称为 "important pattern"，可单独轮廓化
  kElementWise = 100,
  kLastAxisReduce = 101,
  kLastAxisBroadcast = 102,
  kOtherReduce = 103,
  kOtherBroadcast = 104,
  kMatmul = 105,
  kAllReduce = 106,
  kAllGather = 107,
  kReduceScatter = 108,
  kMidFusionImportantAux = 109,
  kInterleave = 110,
  kTranspose = 111,
  kExtractSlice = 112,
};
```

### TypePattern 枚举

```cpp
enum class TypePattern : uint8_t {
  kPureElementWise = 0,  // 纯逐元素块
  kPureMatmul = 1,       // 纯 matmul 块
  kSuffixElementWise = 2,// 后缀逐元素块
  kOpaque = 3,           // 不透明块
};
```

### 融合策略与 OpPattern 映射关系

```
OpPattern                 →  FusionKind
─────────────────────────────────────────
kElementWise / kTranspose  →  PureElemwise
kLastAxisReduce            →  LastAxisPBR
kOtherReduce               →  AnyPBR
kMatmul                    →  SingleCube
kLastAxisBroadcast         →  AnyPB

多算子场景 (InferFuncFusionKind):
  for i = 1..max:
    if tryFusionKind(func, FusionKind(i)):
      return FusionKind(i)
```

### KernelInfo 数据结构族

**文件**: `bishengir/include/bishengir/Dialect/HFusion/Transforms/AutoSchedule/KernelInfo.h:28-101`

```cpp
struct OpInfo {
  size_t idx;                                    // 拓扑序号
  SmallVector<BitVector> inputsAnchorDimension;  // 输入维度锚点
  SmallVector<BitVector> resultsAnchorDimension; // 输出维度锚点
  SmallVector<SmallVector<int64_t>> inputsInterchange;
  SmallVector<SmallVector<int64_t>> resultsInterchange;
};

struct MatmulInfo : public OpInfo {
  size_t tensorAId{0};
  size_t tensorBId{0};
  bool transposeA{false};
  bool transposeB{false};
  int64_t numParallel{0};    // 并行轴数
  int64_t numReduction{0};   // 规约轴数
};

struct ReduceInfo : public OpInfo {
  int64_t numLoops{0};
  SetVector<int64_t> reductionDims;  // 规约轴索引
  int64_t numResults{0};
};

struct BroadcastInfo : public OpInfo {
  int64_t numLoops{0};
  SetVector<int64_t> broadcastDims;  // 广播轴索引
};

struct TransposeInfo : public OpInfo {
  int64_t numLoops{0};
  std::pair<int64_t, int64_t> permuteDims;  // 置换轴
  int64_t elemBitwidth;
  bool transposeLastDim;
};

struct StoreOpInfo : public OpInfo {
  SetVector<int64_t> looselyReductionDims;   // 松散规约维度
  SetVector<int64_t> strictlyParallelDims;   // 严格并行维度
};

struct ConcatInfo : public OpInfo {
  int64_t rank{0};
  int64_t concatDim;
  int64_t elemBitwidth;
};
```

### 数据类型流转图

```
TorchOp / LinalgOp
        │
        ▼ ArithToHFusion / LinalgToHFusion / TorchToHFusion
        │
   linalg::ElemwiseBinaryOp / hfusion::MatmulOp / hfusion::ReduceOp
        │
        ▼ NormalizeOps (Normalize.cpp)
        │
   标准化 HFusion Op (满足硬件对齐约束)
        │
        ▼ InferFuncFusionKind
        │
   FusionKind 标记 (PureElemwise / SingleCube / ShallowCV / ...)
        │
        ▼ OpFusion → FusibleBlockAnalyzer
        │
   FusibleBlock { ops_[], ins_[], outs_[] }
        │
        ▼ FusibleBlockOutliner
        │
   func.func @kernel(%in: tensor<...>) → tensor<...> {
     hfusion.elemwise_binary ...  // 融合后的 kernel 体
   }
        │
        ▼ AutoSchedule → SchedulerBase
        │
   KernelInfo 收集 → TilingUtils 决策 → 生成 scf.forall/for
```

---

## A3. 融合推断算法

### 单 fusionKind 尝试

**文件**: `bishengir/lib/Dialect/HFusion/Transforms/InferFuncFusionKind.cpp:69-80`

```cpp
bool tryFusionKind(func::FuncOp func, const FusionKind &fusionKind) {
  // 只有单 block 函数才可融合
  if (!llvm::hasSingleElement(func.getBody()))
    return false;

  auto *funcBlock = getSingleFuncBlock(func);

  // 用指定 fusionKind 构造融合规则引擎
  opfusion::FusibleHelper fusibleHelper(fusionKind,
                                        /*bufferToOut=*/false,
                                        /*maxHorizontalFusionSize=*/-1);
  // 构造分析器并检查
  opfusion::FusibleBlockAnalyzer analyzer(*funcBlock, fusibleHelper);
  return analyzer.isFusible();  // 返回 true/false
}
```

### 全策略遍历推断

**文件**: `bishengir/lib/Dialect/HFusion/Transforms/InferFuncFusionKind.cpp:82-104`（简化）

```cpp
FusionKind inferFuncFusionKind(func::FuncOp func) {
  // Step 1: 统计 important op 数量
  auto [opCount, ops] = getImportantOpInformation(func);

  if (opCount == 0)
    return FusionKind::AnyPB;    // 空函数默认

  // Step 2: 单算子直接推断
  if (opCount == 1)
    return FusibleHelper::getSingleFusionKind(ops.front());

  // Step 3: 遍历所有策略 (PureElemwise=1 → Unknown=10)
  for (uint32_t i = 1; i < getMaxEnumValForFusionKind(); i++) {
    auto fusionKind = symbolizeFusionKind(i).value();
    if (tryFusionKind(func, fusionKind))
      return fusionKind;  // 返回第一个可行的
  }

  return FusionKind::Unknown;
}
```

### getSingleFusionKind 映射表

**文件**: `bishengir/lib/Dialect/HFusion/Transforms/OpFusion/FusibleHelper.cpp:50-74`

```
OpPattern                 → FusionKind
─────────────────────────────────────
kElementWise              → PureElemwise
kZeroRankElemwise         → PureElemwise
kInterleave               → PureElemwise
kMidFusionImportantAux    → PureElemwise
kTranspose                → PureElemwise
kLastAxisReduce           → LastAxisPBR
kOtherReduce              → AnyPBR
kMatmul                   → SingleCube
kExtractSlice             → AnyPB
kLastAxisBroadcast        → AnyPB
kOtherBroadcast           → AnyPB
```

---

## A4. 融合块分析器

### 核心数据结构

**文件**: `bishengir/lib/Dialect/HFusion/Transforms/OpFusion/FusibleBlockAnalyzer.cpp:48-84`

```cpp
class FusibleBlockAnalyzer {
private:
  const FusibleHelper *fusibleHelper_;
  SmallVector<Operation *> ops_;           // 所有 op
  DenseMap<Operation *, size_t> opIdx_;    // op → 索引映射

  // Union-Find 数据结构
  SmallVector<int32_t> disjointSet_;       // 并查集父节点
  SmallVector<TypePattern> setType_;       // 集合类型
  SmallVector<int64_t> opMaxRank_;         // 最大规约秩
  SmallVector<SetVector<int64_t>> opReduceDim_;  // 规约维度
  SmallVector<int> importantSize_;         // important op 数量
  SmallVector<int> shapePivot_;            // shape 锚点
  SmallVector<SmallVector<int32_t>> edge_;       // 前向边
  SmallVector<SmallVector<int32_t>> revEdge_;    // 反向边
};
```

### 融合规则验证 (verifyRulesAndJoin)

**文件**: `bishengir/lib/Dialect/HFusion/Transforms/OpFusion/FusibleBlockAnalyzer.cpp:86-147`

```cpp
bool FusibleBlockAnalyzer::verifyRulesAndJoin(int nodeU, int nodeV,
                                              bool horizontal = false) {
  int parentU = find(nodeU);  // Union-Find find
  int parentV = find(nodeV);

  if (parentU == parentV)
    return false;  // 已在同一集合

  // ── 规则1: 规约秩检查 ──
  if (fusibleHelper_->isRestrictedByReduceRank(
          opMaxRank_[parentU], opMaxRank_[parentV]))
    return false;

  // ── 规则2: 规约维度检查 ──
  if (fusibleHelper_->isRestrictedByReduceDim(
          opReduceDim_[parentU], opReduceDim_[parentV]))
    return false;

  // ── 规则3: 节点类型兼容性检查 ──
  if (fusibleHelper_->isRestrictedByNodeType(
          setType_[parentU], setType_[parentV], horizontal))
    return false;

  // ── 规则4: shape pivot 检查 (仅水平融合) ──
  if (horizontal && isRestrictedByShapePivot(parentU, parentV))
    return false;

  // ── 规则5: 依赖图检查 ──
  if (isRestrictedByDependency(parentU, parentV, horizontal))
    return false;

  // 全部通过 → Union
  join(parentU, parentV, horizontal);
  return true;
}
```

### BFS 融合主循环

**文件**: `bishengir/lib/Dialect/HFusion/Transforms/OpFusion/FusibleBlockAnalyzer.cpp:177-229`

```cpp
SmallVector<SetVector<Operation *>> FusibleBlockAnalyzer::fuseBlock() {
  // ── Step 1: 收集所有 producer→consumer 边 ──
  SmallVector<NodeNodePair> fusionCandidates;
  for (size_t nodeU = 0; nodeU < ops_.size(); ++nodeU) {
    Operation *op = ops_[nodeU];
    if (fusibleHelper_->isRestrictedByDynamicShape(op))
      continue;
    for (Operation *user : op->getUsers()) {
      int32_t nodeV = opIdx_[user];
      fusionCandidates.emplace_back(nodeU, nodeV);
    }
  }

  // ── Step 2: 按拓扑序排序 ──
  llvm::sort(fusionCandidates, [&](auto &a, auto &b) {
    if (topoRank_[a.second] != topoRank_[b.second])
      return topoRank_[a.second] < topoRank_[b.second];
    return topoRank_[a.first] < topoRank_[b.first];
  });

  // ── Step 3: 尝试融合每条边 ──
  for (auto [u, v] : fusionCandidates) {
    if (fusibleHelper_->isFusible(ops_[u], ops_[v]))
      verifyRulesAndJoin(u, v);  // Union-Find merge
  }

  // ── Step 4: 水平融合 ──
  // 尝试合并无依赖关系的并行 op 组
  // ...

  // ── Step 5: 按连通分量分组 ──
  SmallVector<SetVector<Operation *>> groups;
  DenseMap<int, int> setGroupIdx;
  // ... 分组逻辑

  return groups;
}
```

### checkGroupRequirements — 融合块约束

**文件**: `bishengir/lib/Dialect/HFusion/Transforms/OpFusion/FusibleBlockAnalyzer.cpp:149-173`

```cpp
bool FusibleBlockAnalyzer::checkGroupRequirements(
    const SetVector<Operation *> &group) {
  int importantCount = 0, matmulCount = 0;
  for (Operation *op : group) {
    if (FusibleHelper::isImportantPattern(op))
      importantCount++;
    if (FusibleHelper::getOpPattern(op) == OpPattern::kMatmul)
      matmulCount++;
  }

  // ShallowCV / MixCV 必须包含至少一个 matmul
  if (fusibleHelper_->getFusionKind() == FusionKind::ShallowCV ||
      fusibleHelper_->getFusionKind() == FusionKind::MixCV)
    if (matmulCount == 0) return false;

  // 必须有至少 2 个 important op 才值得融合
  if (importantCount <= 1) return false;

  return true;
}
```

---

## A5. 算子轮廓化

### FusibleBlock 结构

**文件**: `bishengir/include/bishengir/Dialect/HFusion/Transforms/OpFusion/FusibleBlock.h:28-60`

```cpp
class FusibleBlock {
  const FusibleHelper *fusibleHelper_;
  SetVector<Operation *> ops_;       // 融合块内的所有 op
  SmallVector<Value> ins_;           // 外部输入 (lazy init)
  SmallVector<Value> outs_;          // 外部输出 (lazy init)
  SetVector<Operation *> outsModification_; // 输出修正列表
  SetVector<Operation *> opWithAuxs_; // op + 辅助 op
};
```

### OpFusion 主循环

**文件**: `bishengir/lib/Dialect/HFusion/Transforms/OpFusion.cpp:56-83`

```cpp
inline std::optional<HFusionOpFusionOptions>
getOptionFromLabel(func::FuncOp func, const HFusionOpFusionOptions &options) {
  HFusionOpFusionOptions newOptions = options;
  // 从函数属性读取 FusionKind
  auto fusionKindAttr =
      func->getAttrOfType<FusionKindAttr>(FusionKindAttr::name);
  if (!fusionKindAttr) return std::nullopt;
  newOptions.fusionMode = fusionKindAttr.getFusionKind();
  return newOptions;
}

// 多 kernel 融合尝试顺序: 从深到浅
static const SmallVector<FusionKind> kMultiKernelOrder = {
    FusionKind::ShallowCV,    // 先试 Cube+Vector 浅融合
    FusionKind::ShallowVV,    // Vector+Vector
    FusionKind::MixCV,        // Cube+Vector 深融合
    FusionKind::MixC2,        // 双 Cube
    FusionKind::LastAxisPBR,  // PBR
    FusionKind::AnyPB         // 兜底
};
```

### 轮廓化数据流

```
func @entry(%input: tensor<128xf16>) {
  %0 = hfusion.matmul(%a, %b)           → FusibleBlock{matmul}
  %1 = hfusion.elemwise_add(%0, %bias) → FusibleBlock{matmul, add}
  %2 = hfusion.relu(%1)                → FusibleBlock{matmul, add, relu}
  return %2
}
                    │ OpFusion + FusibleBlockOutliner
                    ▼
func @entry(%input: tensor<128xf16>) {
  %0 = call @kernel_matmul_add_relu(%a, %b, %bias)
  return %0
}
func @kernel_matmul_add_relu(%A, %B, %bias) {
  %0 = hfusion.matmul(%A, %B)
  %1 = hfusion.elemwise_add(%0, %bias)
  %2 = hfusion.relu(%1)
  return %2
}
```

---

## A6. 自动调度器

### 调度器分发

**文件**: `bishengir/lib/Dialect/HFusion/Transforms/AutoSchedule/AutoScheduleBase.cpp:580-611`

```cpp
auto fusionKind = fusionKindAttr.getFusionKind();
std::unique_ptr<SchedulerBase> scheduler;

switch (fusionKind) {
  case FusionKind::PureElemwise:     // ┐
  case FusionKind::AnyPB:            // │
  case FusionKind::LastAxisPBR:      // ├─ AnyPBRScheduler
  case FusionKind::AnyPBR:           // ┘
    scheduler = std::make_unique<AnyPBRScheduler>(funcOp);
    break;

  case FusionKind::SingleCube:       // → SingleCubeScheduler
    scheduler = std::make_unique<SingleCubeScheduler>(funcOp);
    break;

  case FusionKind::ShallowCV:        // → ShallowCVScheduler
    scheduler = std::make_unique<ShallowCVScheduler>(funcOp);
    break;

  case FusionKind::ShallowVV:        // → 跳过，不做 tiling
    return success();

  case FusionKind::Unknown:
  default:
    return funcOp.emitError("Unknown kernel fusion kind");
}

return scheduler->runOnOperation(opBuilder);
```

### 调度器类层次

```
SchedulerBase (AutoScheduleBase.h)
├── AnyPBRScheduler      ← PureElemwise / AnyPB / LastAxisPBR / AnyPBR
├── SingleCubeScheduler  ← SingleCube (纯 matmul)
├── ShallowCVScheduler   ← ShallowCV (matmul + 少量 elemwise)
└── (ShallowVV → 无调度)
```

### SchedulerBase::applyScheduleImpl 执行链路

**文件**: `bishengir/lib/Dialect/HFusion/Transforms/AutoSchedule/AutoScheduleBase.cpp:613-626`

```cpp
LogicalResult SchedulerBase::applyScheduleImpl(OpBuilder &opBuilder) {
  PassManager pm(getContext());

  // 执行 transform dialect 调度脚本
  pm.addPass(hfusion::createAutoScheduleInterpreterPass(
      getToBeScheduledKernelName()));
  // 清理调度标记
  pm.addPass(hfusion::createEraseAutoSchedulePass(
      getToBeScheduledKernelName()));

  return pm.run(getModule());
}
```

---

## A7. HFusion 数据流链路

```
┌──────────────────────────────────────────────────────────────┐
│                  lower-hfusion-pipeline 完整数据流             │
└──────────────────────────────────────────────────────────────┘

阶段1: preProcess
  linalg::GenericOp ──→ hfusion::ElemwiseBinaryOp ──→ Normalize → 标准化 Op
  类型: linalg + arith   →     hfusion     →        hfusion (对齐后)

阶段2: preFlattenPass
  标准化 Op ──→ BubbleUpExtractSlice ──→ FoldUnitExtentDims ──→ ComposeMultiReduce

阶段3: flattenAndFold
  多维 Op ──→ FlattenOps ──→ 1D Op
  类型: tensor<NxMxf16> → tensor<?xf16>

阶段4: inferAndOutlineOp
  1D Op 组 ──→ InferFuncFusionKind() ──→ FusionKind 标记
  FusionKind ──→ FusibleBlockAnalyzer.fuseBlock() ──→ FusibleBlock[]
  FusibleBlock[] ──→ FusibleBlockOutliner ──→ func @kernel()

阶段6: autoSchedule
  func @kernel ──→ KernelInfoCollector ──→ KernelInfo 结构体组
  KernelInfo ──→ SchedulerBase (策略分发)
  调度输出: transform dialect script
  AutoScheduleInterpreter ──→ 生成 scf.forall / scf.for 循环

阶段7: postProcess
  调度后 Op ──→ NormalizeOps (二次) ──→ AddFFTSAddr ──→ HoistTensorEmpty
```

---

# Part B: HIVM 硬件映射管线

## B1. 双管线架构

### 入口调用

**文件**: `bishengir/lib/Tools/bishengir-compile/PassPipeline.cpp:82-85`

```cpp
// 先转换 HFusion → HIVM
hivm::buildConvertToHIVMPipeline(pm, convertToHIVMOptions);

// 再优化 HIVM
hivm::buildOptimizeHIVMPipeline(pm, options);
```

### 管线 A: ConvertToHIVM (4 Pass)

**文件**: `bishengir/lib/Dialect/HIVM/Pipelines/ConvertToHIVMPipeline.cpp:29-44`

```cpp
void buildConvertToHIVMPipeline(OpPassManager &pm,
                                const ConvertToHIVMPipelineOptions &options) {
  // Pass 1: HFusion Op → HIVM Op 映射
  ConvertHFusionToHIVMOptions hfs2hivmOptions;
  hfs2hivmOptions.mmMapMode = options.enableTritonKernelCompile
                                  ? hfusion::MmMapMode::MacroInstr
                                  : hfusion::MmMapMode::CoreOp;  // 默认
  pm.addPass(createHFusionToHIVMConversionPass(hfs2hivmOptions));

  // Pass 2: Triton Kernel 参数适配 (可选)
  if (options.enableTritonKernelCompile)
    pm.addPass(createTritonGlobalKernelArgsToHIVMOpPass());

  // Pass 3: 残余 Tensor op 转换
  pm.addPass(createTensorToHIVMConversionPass());

  // Pass 4: 收尾转换 (memref.copy → hivm.copy)
  pm.addPass(createConvertToHIVMOpPass());
}
```

### HFusion → HIVM 算子映射表

| HFusion Op | HIVM Op | 硬件单元 |
|-----------|---------|---------|
| `hfusion.elemwise_binary` | `hivm.v_add / v_mul / v_sub` | Vector |
| `hfusion.matmul` | `hivm.mmad` (CoreOp) 或 `hivm.macro_instr` | Cube |
| `hfusion.reduce` | `hivm.v_reduce` | Vector |
| `hfusion.broadcast` | `hivm.v_brc` | Vector |
| `hfusion.transpose` | `hivm.v_transpose` | Vector |
| `hfusion.load / store` | `hivm.load_scalar / store` | Scalar |
| `hfusion.concat` | `hivm.v_concat` | Vector |

### 管线 B: OptimizeHIVM (5 阶段)

**文件**: `bishengir/lib/Dialect/HIVM/Pipelines/HIVMPipelines.cpp:395-413`

```cpp
void buildOptimizeHIVMPipeline(OpPassManager &pm,
                               const HIVMPipelineOptions &options) {
  // 阶段0: 初始化
  pm.nest<func::FuncOp>().addPass(createInitEntryKernelPass());

  if (!options.disableHIVMTensorCompile) {
    // 阶段1: Buffer 化前优化 (Tensor SSA, ~30 Pass)
    hivmPreBufferizationOptimizationPipeline(pm, options);

    // 阶段2: Buffer 化 (Tensor → MemRef)
    bufferizationPipeline(pm, options);
  }

  // 阶段3: Buffer 化后优化 (MemRef 地址空间, ~25 Pass)
  hivmPostBufferizationOptimizationPipeline(pm, options);

  // 阶段4: 最终化 + Lowering
  pm.addPass(scope::createInlineScopePass(
      InlineScopeOptions{/*forceInline=*/true}));
  pm.addPass(createEnableHIVMCCompatiblePrintPass());
  pm.addPass(annotation::createAnnotationLoweringPass());
  pm.nest<func::FuncOp>().addPass(createInsertInitAndFinishForDebugPass());
  pm.nest<func::FuncOp>().addPass(createMarkDisableLoadPass());

  syncBlockLockPipeline(pm, SyncBlockLockPipelinePhase::Finalize);

  // 阶段5: HIVM → Standard MLIR 最终 Lowering
  pm.addPass(createConvertHIVMToStandardPass());
}
```

---

## B2. 内存规划算法

### Da Vinci 内存层级常量

**文件**: `bishengir/lib/Dialect/HIVM/Transforms/PlanMemory.cpp:49-57`

```cpp
namespace {
/// TODO: Obtain information from the same platform in the future
constexpr const int ubAlignSize = 32 * 8;       // 32 Bytes
constexpr const int ubSpaceSize = 192 * 1024;   // 192 KB
constexpr const int l1AlignSize = 32 * 8;       // 32 Bytes
constexpr const int l1SpaceSize = 512 * 1024;   // 512 KB
constexpr const int l0cAlignSize = 512 * 8;     // 512 Bytes
constexpr const int l0cSpaceSize = 128 * 1024;  // 128 KB
constexpr const int workSpaceAlignSize = 32 * 8; // 32 Bytes
}
```

### Buffer 复用规则

**文件**: `bishengir/lib/Dialect/HIVM/Transforms/PlanMemory.cpp:59-153`（关键函数）

```cpp
// 规则1: cast 复用 - 仅 1D contiguous
bool isReusableCastOp(hivm::VCastOp &castOp, Value output, Value input) {
  auto rank = dyn_cast<MemRefType>(output.getType()).getRank();
  if (rank > 1 || !isLastDimContiguous(output) || !isLastDimContiguous(input))
    return false;
  return true;
}

// 规则2: offset 复用 - 相同偏移
static bool isReusableByOffset(HIVMStructuredOp &hivmOp) {
  auto output = hivmOp.getDpsInits().front();
  auto outputOffset = getStaticOffset(output);
  if (!outputOffset) return false;
  for (Value in : hivmOp.getDpsInputs()) {
    auto inOffset = getStaticOffset(in);
    if (!inOffset) continue;
    if (*inOffset != *outputOffset) return false;
  }
  return true;
}

// 规则3: operand 复用 - 位宽兼容
bool isReusableOperands(Operation *op, HIVMStructuredOp &hivmOp) {
  auto output = hivmOp.getDpsInits().front();
  auto outputBitWidth =
      getElementTypeOrSelf(output.getType()).getIntOrFloatBitWidth();
  for (auto input : hivmOp.getDpsInputs()) {
    auto inputBitWidth =
        getElementTypeOrSelf(input.getType()).getIntOrFloatBitWidth();
    if (inputBitWidth == outputBitWidth) return true;
    if (inputBitWidth % outputBitWidth == 0)
      return isReusableNarrowWidth(op, output, input);
  }
  return false;
}

// 规则4: extra buffer 复用
inline bool isReusableExtraBuffer(Operation *op) {
  auto extraBufferOp = dyn_cast_if_present<ExtraBufferOpInterface>(op);
  if (!extraBufferOp || extraBufferOp.getExtraBuffers().empty())
    return true;
  if (extraBufferOp.getExtraBuffers().size() > 1)
    return false;
  return extraBufferOp.shouldAllocExtraBufferForScalarOrOTFBrc();
}
```

### 内存规划数据流

```
Buffer alloc ops
  ┌─────────────────────────────────────┐
  │ %buf1 = memref.alloc() : 128B       │
  │ %buf2 = memref.alloc() : 256B       │
  │ %buf3 = memref.alloc() : 64B        │
  └──────────┬──────────────────────────┘
             │ PlanMemory pass
             ▼
  1. 构建生命期区间: [start, end] for each buffer
     buf1: [t0, t5], buf2: [t2, t8], buf3: [t6, t10]
  
  2. 冲突检测 (Interval Graph):
     buf1 ∩ buf2 = [t2, t5] → 冲突!
     buf1 ∩ buf3 = ∅         → 可复用!
     buf2 ∩ buf3 = [t6, t8]  → 冲突!
  
  3. 分配地址:
     buf1: offset=0    (占用 0-127)
     buf2: offset=128  (占用 128-383, 对齐 32B)
     buf3: offset=0    (复用 buf1 空间, 生命期不重叠)
  
  4. 复用验证:
     isReusableOperands(op) && isReusableExtraBuffer(op)
  
  5. 输出:
     %ptr1 = hivm.pointer_cast %base, offset=0
     %ptr2 = hivm.pointer_cast %base, offset=128
     %ptr3 = hivm.pointer_cast %base, offset=0
```

---

## B3. 图同步求解器

### Solver 状态重置

**文件**: `bishengir/lib/Dialect/HIVM/Transforms/GraphSyncSolver/SyncSolver.cpp:51-72`

```cpp
void Solver::reset(bool resetEventIdRanOutOpts) {
  if (resetEventIdRanOutOpts) {
    reusePairs.clear();                       // 可复用同步对
    disabledMultiEventIdPairs.clear();
    backwardSyncEventsAfterMerge.clear();
    moveBackwardSyncPairsToOutmostLoop = false;
  }
  skipOcc.clear();
  syncedPairs.clear();                        // 已同步对
  processedOccPairs.clear();
  chosenConflictedPairs.clear();
  backwardSyncEvents.clear();
  replacedWithReusableSyncedPairs.clear();
  reusedPairs.clear();
  barrierAllPairs.clear();                    // barrier_all 对
  insertedBarrierAllBefore.clear();
  eventIdSolver.clear();                      // 事件ID 分配器
  resetUnitFlag();
}
```

### GraphSyncSolver 执行流程

```
                        func @kernel
                            │
                            ▼
            ┌───────────────────────────┐
            │   IRTranslator.Build()    │
            │   (InjectSync.cpp:69-71)  │
            │                           │
            │   HIVM ops → SyncNode     │
            │   分析 MemInfo (读写范围)  │
            │   构建依赖图 (GraphSolver) │
            └──────────┬────────────────┘
                       │ SyncIR (同步中间表示)
                       ▼
            ┌───────────────────────────┐
            │   SyncAnalyzer            │
            │   (InjectSync.cpp:80)     │
            │                           │
            │   检测:                    │
            │   - RAW 冲突              │
            │   - WAW 冲突              │
            │   - WAR 冲突              │
            │   按 PIPE 分: V/M/MTE1-3  │
            └──────────┬────────────────┘
                       │ conflicted pairs
                       ▼
            ┌───────────────────────────┐
            │   EventIdSolver           │
            │                           │
            │   图着色分配 event ID:    │
            │   - 不冲突对 → 复用 ID    │
            │   - 冲突对 → 新 ID       │
            │   - ID 溢出 → barrier_all │
            └──────────┬────────────────┘
                       │ event ID assignments
                       ▼
            ┌───────────────────────────┐
            │   SyncSolverCodeGen       │
            │                           │
            │   生成:                    │
            │   set_wait(event_id)      │
            │   set_flag(event_id)      │
            │   pipe_barrier(PIPE_ALL)  │
            └───────────────────────────┘
```

---

## B4. 自动同步注入

### InjectSync 两种模式

**文件**: `bishengir/lib/Dialect/HIVM/Transforms/InjectSync/InjectSync.cpp:47-83`

```cpp
// 模式1: BARRIERALL - 全屏障 (简单粗暴)
void InjectSyncAnalysis::InjectSyncAll() {
  MLIRContext *ctx = func_->getContext();
  IRRewriter rewriter(ctx);
  func_->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (op->getDialect()->getNamespace() ==
            HIVMDialect::getDialectNamespace() ||
        mlir::isa<func::ReturnOp>(op)) {
      rewriter.setInsertionPoint(op);
      auto pipeAll = PipeAttr::get(ctx, hivm::PIPE::PIPE_ALL);
      rewriter.create<hivm::PipeBarrierOp>(loc, pipeAll);
    }
  });
}

// 模式2: AutoInjectSync - 基于内存依赖分析
void InjectSyncAnalysis::AutoInjectSync(bool enableUnitFlag,
                                        bool assumeAliveLoops) {
  MemoryDependentAnalyzer memAnalyzer;
  SyncIRs syncIR;
  SyncOperations syncOperations;
  Buffer2MemInfoMap buffer2MemInfoMap;

  // Step 1: 翻译为同步 IR
  IRTranslator trans(syncIR, memAnalyzer, buffer2MemInfoMap, func_,
                     SyncAnalysisMode::NORMALSYNC);
  trans.Build();

  // Step 2: 单指令/无指令 → 无需同步
  if (syncIR.size() <= 1) return;

  // Step 3: 分析冲突并生成同步指令
  SyncAnalyzer syncAnalyzer(syncIR, memAnalyzer, syncOperations, func_,
                            enableUnitFlag, assumeAliveLoops);
  syncAnalyzer.Analyze();
}
```

### 两种同步策略对比

```
策略选择流程:
  ┌─ enableHIVMGraphSyncSolver=true ──▶ GraphSyncSolver (最优)
  │                                      ↓ 失败
  │                               ┌─ barrier_all ──▶ InjectSyncAll
  │                               └─ auto ─────────▶ AutoInjectSync
  │
  └─ enableHIVMGraphSyncSolver=false ──▶ InjectSync
                                           ├─ BARRIERALL → InjectSyncAll
                                           └─ default → AutoInjectSync
```

---

## B5. Lowering 到 Standard MLIR

### HIVMToStandard 入口

**文件**: `bishengir/lib/Conversion/HIVMToStandard/HIVMToStandard.cpp:47-61`

```cpp
namespace mlir {
#define GEN_PASS_DEF_CONVERTHIVMTOSTANDARD
#include "bishengir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

// 将 MemRef 布局转为 strided dynamic
static MemRefType makeStridedLayoutAndShapeDynamic(MemRefType type) {
  return MemRefType::Builder(type)
      .setLayout(StridedLayoutAttr::get(
          type.getContext(), ShapedType::kDynamic,
          SmallVector<int64_t>(type.getRank(), ShapedType::kDynamic)))
      .setShape(SmallVector<int64_t>(type.getRank(), ShapedType::kDynamic));
}
```

### HIVM Op → Standard 映射示例

```
hivm.v_add(%a, %b) → %c
        │ HIVMToStandard
        ▼
%c_alloc = memref.alloc() : memref<Nxf16>
scf.for %i = 0 to N {
  %ai = memref.load %a[%i] : memref<Nxf16>
  %bi = memref.load %b[%i] : memref<Nxf16>
  %ci = arith.addf %ai, %bi : f16
  memref.store %ci, %c_alloc[%i] : memref<Nxf16>
}

hivm.mmad(%A, %B) → %C
        │ HIVMToStandard
        ▼
(保留为硬件 intrinsic — 不 lowering 为 scf.for)
→ 由执行引擎直接解释为 Cube 单元指令

hivm.set_wait(%ev) / hivm.set_flag(%ev) / hivm.pipe_barrier(%pipe)
        │ HIVMToStandard
        ▼
(保留为硬件 intrinsic)
```

---

## B6. HIVM 数据流链路

```
┌──────────────────────────────────────────────────────────────────┐
│              optimize-hivm-pipeline 完整数据流                    │
└──────────────────────────────────────────────────────────────────┘

阶段0: Init
  func @kernel ──→ InitEntryKernel ──→ 标记 hacc.device + hivm.kernel

阶段1: PreBuf (Tensor SSA)
  HIVM Op (tensor) ──→ NormalizeMatmul ──→ NormalizeConvOps
  类型: tensor<128xf16> (虚拟值)
         │
         ├─ InsertLoadStoreForMixCV ──→ 插入自动 load/store
         ├─ TileCubeVectorLoop ──→ 混合 tiling
         ├─ MarkMultiBuffer ──→ 双缓冲标记
         ├─ CVPipelining ──→ Cube-Vector 流水线
         ├─ PlanMemory(GLOBAL_WORKSPACE) ──→ 内存偏移分配
         └─ hivmNormSyncPipeline ──→ 同步事件注入
             ├─ GraphSyncSolver (优先)
             └─ InjectSync (回退)

阶段2: Bufferization
  Tensor SSA ──→ OneShotBufferize ──→ MemRef
  类型: tensor<Nxf16> → memref<Nxf16>

阶段3: PostBuf (MemRef 地址空间)
  MemRef ──→ InferHIVMMemScope ──→ 内存层级标注
  类型: memref → memref<UB> / memref<L1> / memref<L0C> / memref<GM>
         │
         ├─ InferHIVMDataLayout ──→ ND/NZ 格式推断
         │   连续访问 → ND, 非连续 → NZ (Fractal)
         │
         ├─ AlignAllocSize / EnableStrideAlign ──→ 对齐 + stride
         │   UB: 32B 对齐, L1: 32B 对齐, L0C: 512B 对齐
         │
         ├─ PlanMemory(local) ──→ 局部 buffer 复用的偏移分配
         │
         ├─ HIVMLowerToLoops ──→ HIVM op → scf.for 循环
         │   ImplByScalarOpInterface.decomposeOperation()
         │
         ├─ GraphSyncSolver / InjectSync ──→ 同步指令
         │   set_wait / set_flag / pipe_barrier
         │
         └─ AllocExtraBuffer ──→ 额外 buffer (multi-buffer / scalar)

阶段4-5: Finalize + Lower
  同步指令 + DMA ──→ 保留为 intrinsic
  HIVM Op ──→ HIVMToStandard ──→ Standard MLIR (scf + memref + arith)
```

---

## 附录：完整循环链路

```
                     编译入口: bishengir-compile
                              │
    ┌─────────────────────────┼──────────────────────────┐
    │                         │                          │
    ▼                         ▼                          ▼
┌─────────┐          ┌─────────────────┐        ┌──────────────────┐
│ 高层 IR  │          │  lower-hfusion   │        │  optimize-hivm   │
│         │          │  -pipeline       │        │  -pipeline       │
│ Torch   │─────────▶│                  │───────▶│                  │──▶ Standard
│ Linalg  │  转换     │  7 阶段/30+ Pass │ 转换    │  5 阶段/60+ Pass │   MLIR
│ GPU     │          │                  │        │                  │
└─────────┘          └─────────────────┘        └──────────────────┘

数据类型演进:
  linalg::GenericOp                      (高层语义, 无 tiling)
       │ ArithToHFusion / LinalgToHFusion
       ▼
  hfusion::ElemwiseBinaryOp              (融合层, 有 FusionKind)
       │ Normalize + Flatten + OpFusion
       ▼
  func @kernel (%in: tensor<?xf16>)      (轮廓化 kernel)
       │ AutoSchedule
       ▼
  func @kernel (scf.forall / scf.for)    (tile 循环结构)
       │ HFusionToHIVM
       ▼
  hivm.v_add / hivm.mmad                  (硬件指令, tensor)
       │ OneShotBufferize
       ▼
  hivm.v_add / hivm.mmad (memref)        (硬件指令, memref)
       │ HIVMLowerToLoops
       ▼
  scf.for + hivm.set_wait/set_flag       (循环 + 同步)
       │ HIVMToStandard
       ▼
  scf.for + memref.load/store + arith    (Standard MLIR)
  + hivm.set_wait/set_flag (intrinsic)   (硬件同步保留)
```

---

> 文件: `/AscendNPU-IR/docs/HFusion_HIVM_完整详解.md`
> 所有代码片段均来自 `AscendNPU-IR` 源码，文件名和行号已标注。
