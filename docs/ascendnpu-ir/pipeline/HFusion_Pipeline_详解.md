# HFusion 编译管线详解

> 基于 AscendNPU-IR 源码分析，覆盖 `lower-hfusion-pipeline` 的 7 个阶段、9 种融合策略、4 个调度器的完整实现细节。

---

## 一、管线总览

`buildHFusionPipelines()` 在 `HFusionPipelines.cpp:279` 定义，入口 pass 名 `lower-hfusion-pipeline`。

```
┌─────────────────────────────────────────────────────────────────┐
│                     lower-hfusion-pipeline                      │
├──────────┬──────────┬──────────┬──────────┬──────────┬─────────┤
│ preProcess│preFlatten│flattenFold│inferOutline│autoSchedule│postProcess│
└──────────┴──────────┴──────────┴──────────┴──────────┴─────────┘
```

7 个阶段顺序不可调换，每个阶段有严格的 canonicalization 锚点。

---

## 二、阶段一：preProcess（预处理）

**文件**: `HFusionPipelines.cpp:79-115`

### 目标

将所有上游 IR（Linalg/Torch/Arith/Math/Tensor/GPU）统一转换为 HFusion Dialect，并做初始标准化。

### Pass 序列

| 顺序 | Pass | 作用 |
|------|------|------|
| 1 | `EraseSymbol` | 清理符号分析残留（可选，`enableSymbolAnalysis` 控制）|
| 2 | `ArithToHFusionConversion` | `arith.add/mul/sub` → `linalg.elemwise_binary` |
| 3 | `MathToHFusionConversion` | `math.sin/cos/exp` → `linalg.elemwise_unary` |
| 4 | `LinalgToHFusionConversion` | `linalg.generic` → `hfusion.*` named ops |
| 5 | `GPUToHFusionConversion` | Triton kernel → HFusion（可选，`enableTritonKernelCompile`）|
| 6 | `AdaptTritonKernel` | Triton 特殊适配 |
| 7 | `TensorToHFusionConversion` | `tensor.pad/extract_slice` → HFusion 等价 op |
| 8 | `CanonicalizeTensorReshape` | tensor reshape 规范化 |
| 9 | **canonicalization** | CSE → ExtendedCanonicalizer → NormalizeTensorOps |
| 10 | `ArithToHFusionConversion` | **二次转换**：FoldUnitExtentDims 后残余 arith op 转换 |
| 11 | `ConvertGenericToNamedOp` | `linalg.generic` → 命名算子 |
| 12 | `LegalizeBF16` | bf16 合法化 |
| 13 | `Decompose(NO_CONSTRAINT)` | **初次分解**：无约束阶段的所有聚合算子 |
| 14 | `NormalizeSliceOps` | slice op 对齐标准化 |
| 15 | `NormalizeOps` | ★ 核心标准化（9,479 行）|
| 16 | `LegalizeBool` | bool 类型合法化 |
| 17 | `SimplifyOps` | 代数化简 |
| 18 | `InlineBrc` | broadcast 内联化 |
| 19 | `NormalizeOps` | **二次标准化**：inline-brc 后重新标准化 |

### NormalizeOps 详解

**文件**: `Normalize.cpp` (9,479 行)

核心职能是让所有算子满足 Da Vinci 硬件的计算约束：

```
1. 类型标准化
   - fp64 → fp32 降级（硬件不支持 fp64）
   - bf16 格式校正
   - int8 → int32 提升（部分指令要求）

2. 形状标准化
   - 对齐到 16B（512-bit Cube 单元要求）
   - 末维 padding 到 32 对齐
   - broadcast 维度显式化

3. 算子标准化
   - 复合算子分解为基本算子（如 norm(x) → x - x_round * pi_approx + offset）
   - reduce 轴调整（优先最后一轴规约）
   - transpose 标准化为 2-排列

4. 特殊处理
   - overflow_mode 注入（saturate/trunc）
   - 精度锚点标记
   - gather/interleave 展开
```

---

## 三、阶段二：preFlattenPass（展平前预处理）

**文件**: `HFusionPipelines.cpp:117-141`

### 目标

在 IR 展平之前消除阻碍展平的 pattern，为后续融合创造最大自由度。

### Pass 序列

| 顺序 | Pass | 作用 |
|------|------|------|
| 1 | `BubbleUpExtractSlice` | extract_slice 上浮到 consumer 之前 |
| 2 | **canonicalization** | CSE + ExtendedCanonicalizer |
| 3 | `FoldUnitExtentDims` | 折叠尺寸为 1 的维度（`useRankReducingSlices=false`）|
| 4 | **canonicalization** | 折叠后清理 |
| 5 | `ArithToHFusionConversion` | 规约操作中 residual arith 再转换 |
| 6 | `ComposeMultiReduce(aggressive=true)` | 多个 reduce 合并为一个 multi-reduce |
| 7 | `PropagateSymbol` | 符号分析传播（可选）|
| 8 | `UnfoldSymbolicInt` | 符号整数展开（可选）|
| 9 | **CSE** | 去重 |
| 10 | `PropagateReshape` | reshape 传播穿越 ops |
| 11 | `SimplifyOps` | 再次简化 |
| 12 | **canonicalization** | 最终清理 |

---

## 四、阶段三：flattenAndFold（展平与折叠）

**文件**: `HFusionPipelines.cpp:153-175`

### 目标

将多维算子展平为一维，消除维度信息对融合的阻碍。

### FlattenOps 算法

**文件**: `FlattenOps.cpp:50-73`

```
算法: collapseOpIterationDims
输入: linalg op (N 维)
输出: 1 维 linalg op

for each linalg elemwise op:
  if numLoops > 1 AND isElementwise(op) AND NOT isBroadcast(op):
    reassociation = [0, 1, 2, ..., N-1]  # 合并所有维度
    result = collapseOpIterationDims(op, reassociation)
    replace op with result

跳过: broadcast op（维度信息不可折叠）
      reduce op（规约轴信息不可丢失）
```

### Pass 序列

| 顺序 | Pass | 作用 |
|------|------|------|
| 1 | `FoldTensorEmpty` | 折叠 `tensor.empty`（避免动态 shape 阻碍展平）|
| 2 | `FlattenOps(mode=Tidy, skipHost=true)` | ★ IR 展平 |
| 3 | `CanonicalizeTensorReshape` | 展平后 reshape 规范化 |
| 4 | **canonicalization** (AfterFlattenBeforeAutoSchedule) | CSE + 受限 Canonicalizer |
| 5 | `CacheIOForReturnArg` | 为返回参数添加 cache 标记 |
| 6 | `FoldTensorEmpty` | 再次折叠 empty |
| 7 | **canonicalization** (AfterFlattenBeforeAutoSchedule) | 最终清理 |

---

## 五、阶段四：inferAndOutline（推断与轮廓化）

**文件**: `HFusionPipelines.cpp:177-198`

### 目标

推断每个函数的融合策略，找到可融合算子块，将其轮廓化（outline）为独立 kernel 函数。

### 5.1 InferFuncFusionKind — 融合策略推断

**文件**: `InferFuncFusionKind.cpp:82-104`

```
算法:
  1. getImportantOpInformation(func)
     → 遍历函数内所有 op，统计 isImportantPattern(op) 为 true 的 op 数量
     → Important op 定义: OpPattern >= 100 (见 FusibleHelper.h:55-68)
       包括: kElementWise, kMatmul, kReduce, kBroadcast, kTranspose 等
  
  2. if opCount == 0: return FusionKind::AnyPB  // 空函数，默认策略
  
  3. if opCount == 1: return getSingleFusionKind(op)
     → 单算子场景直接推断:
        kMatmul → SingleCube
        kAllReduce → AnyPB
        kElementWise → PureElemwise
        ...
  
  4. for i = 1 to max(FusionKind):
       fusionKind = symbolizeFusionKind(i)
       if tryFusionKind(func, fusionKind):
         return fusionKind  // 返回第一个可用的策略
  
  5. tryFusionKind 内部逻辑:
     → 构造 FusibleHelper(fusionKind, bufferToOut=false)
     → 构造 FusibleBlockAnalyzer(funcBlock, fusibleHelper)
     → 调用 analyzer.isFusible()
     → 如果所有 op 可融合且满足该 FusionKind 约束 → true
```

**关键**: 遍历顺序从 `PureElemwise(1)` 到 `Unknown(10)`，越保守的策略越先尝试。

### 5.2 OpFusion — 算子融合

**文件**: `OpFusion.cpp`, `FusibleBlockAnalyzer.cpp`, `FusibleHelper.cpp`

```
核心数据结构:
  - FusibleBlock: 一组可融合的 op 集合 + 输入/输出 value
  - FusibleBlocks: vector<FusibleBlock>
  - FusibleHelper: 融合规则引擎
  - FusibleBlockAnalyzer: Union-Find + 验证规则
```

#### FusibleBlockAnalyzer 算法

**文件**: `FusibleBlockAnalyzer.cpp:48-84`

```
初始化:
  1. 将 Block 内所有 op 放入 ops_ 数组
  2. 建立 op → idx 映射
  3. 初始化 Union-Find 数据结构:
     disjointSet_[i] = -1  (每个 op 初始为独立集合)
     辅助数组: setType_, opMaxRank_, opReduceDim_, importantSize_, shapePivot_

规则验证 (verifyRulesAndJoin):
  调用条件: 两个 op 有数据依赖（producer→consumer）或水平关系
  
  检查项:
    1. isRestrictedByReduceRank(parentU, parentV)
       → 规约轴 rank 冲突检查
    2. isRestrictedByReduceDim(parentU, parentV)  
       → 规约维度冲突检查
    3. isRestrictedByNodeType(parentU, parentV, horizontal)
       → 节点类型兼容性检查:
         - PureElemwise: 只能与 elementwise/broadcast 融合
         - AnyPB: elementwise + broadcast 可融合
         - AnyPBR: 上述 + reduce（需规约轴一致）
         - SingleCube: 只能 matmul + elementwise
         - ShallowCV: matmul + 有限 elementwise
    4. isRestrictedByShapePivot(parentU, parentV)
       → shape pivot 冲突（水平融合时）
  
  全部通过 → Union(parentU, parentV)
```

#### 多 Kernel 融合顺序

**文件**: `OpFusion.cpp:68-70`

```cpp
static const SmallVector<FusionKind> kMultiKernelOrder = {
    FusionKind::ShallowCV,    // 先尝试 Cube+Vector 浅融合
    FusionKind::ShallowVV,    // Vector+Vector 浅融合
    FusionKind::MixCV,        // Cube+Vector 深度融合
    FusionKind::MixC2,        // 双 Cube 融合
    FusionKind::LastAxisPBR,  // 末轴 PBR
    FusionKind::AnyPB         // 兜底：任意 P+B
};
```

### 5.3 Outline — 轮廓化

**文件**: `FusibleBlockOutliner.cpp`

```
算法:
  1. FusibleBlockOutliner 接收上一步的 FusibleBlocks
  
  2. 多输出模式 (OutputMode::Multiple):
     → 移除重复的 alias 输出
     → 每个 FusibleBlock 轮廓化为一个独立 func.func
  
  3. 单输出模式 (OutputMode::Single):
     → 从每个 block 的输出 op 反向 BFS
     → 收集所有可达的 producer op
     → 构建单一 func.func
  
  4. 轮廓化操作:
     → 创建新 func.func，迁移 op
     → 参数化外部依赖的 value
     → return 融合块的输出
     → 在原位置插入 call @new_func
```

### Pass 序列

| 顺序 | Pass | 作用 |
|------|------|------|
| 1 | `FoldSymbolicDim` | 折叠符号维度 |
| 2 | `InferFuncFusionKind` | ★ 推断融合策略 |
| 3 | **canonicalization** | |
| 4 | `OpFusion` | ★ 执型算子融合 + 多 Kernel 轮廓化 |
| 5 | **canonicalization** (AfterFlattenBeforeAutoSchedule) | |
| 6 | `OutlineSingleOp` | 单算子轮廓化 |
| 7 | **canonicalization** | |
| 8 | `UnfoldSymbolicDim` | 展开符号维度 |
| 9 | `DropSymbols` | 丢弃符号标记 |
| 10 | `EliminateDuplicateFuncs` | 去重函数 |

---

## 六、阶段五：postProcessOutlinedKernel（轮廓后处理）

**文件**: `HFusionPipelines.cpp:143-151`

### 目标

对轮廓化后的 kernel 函数做硬件适配优化。

| 顺序 | Pass | 作用 |
|------|------|------|
| 1 | `DowngradeFP64CstOp` | fp64 常量 → fp32 |
| 2 | `TrickleConcatDown` | concat 下沉 |
| 3 | `BubblePadUp` | pad 上浮 |
| 4 | `LegalizeBool` | bool 再次合法化 |
| 5 | `FoldTensorEmpty` | empty 折叠 |
| 6 | `NormalizeLastDimUnalignedTensorOp` | 末维未对齐 tensor 标准化 |

---

## 七、阶段六：hfusionAutoSchedulePipeline（自动调度）

**文件**: `HFusionPipelines.cpp:216-254`

### 目标

为每个 kernel 函数生成 tile 循环结构，映射计算到 Da Vinci 的 Cube/Vector 单元。

### 调度器分发

**文件**: `AutoScheduleBase.cpp:580-611`

```cpp
switch (fusionKind) {
  case PureElemwise:   // 纯逐元素
  case AnyPB:          // 任意 P+B
  case LastAxisPBR:    // 末轴 P+B+R
  case AnyPBR:         // 任意 PBR
    scheduler = AnyPBRScheduler;    // ★ 最复杂的调度器，统一处理 PBR 家族
    break;
  
  case SingleCube:     // 单 Cube (matmul)
    scheduler = SingleCubeScheduler;
    break;
  
  case ShallowCV:      // 浅层 CV
    scheduler = ShallowCVScheduler;
    break;
  
  case ShallowVV:      // 浅层 VV → 直接跳过，不做 tiling
    return success();
}
```

### 7.1 AnyPBRScheduler — PBR 家族调度器

处理 `PureElemwise / AnyPB / LastAxisPBR / AnyPBR`

```
算法:
  1. KernelInfoCollector 收集 kernel 内所有 op 的信息:
     - OpInfo: 基本维度信息 (idx, anchor dimension, interchange)
     - MatmulInfo: A/B id, transpose, numParallel, numReduction
     - ReduceInfo: reduction dims, numLoops
     - BroadcastInfo: broadcast dims
     - TransposeInfo: permute dims, element bitwidth
     - ExtractSliceInfo: partial/full sliced dims
  
  2. DimensionAnalyzer 统一维度映射:
     - 以某个 anchor value 的维度为基准
     - 所有 op 的维度通过 BitVector 映射到 anchor 维度空间
  
  3. TilingUtils 决定 tile 大小:
     - Cube 单元: 16×16 (fp16) / 16×32 (int8)
     - Vector 单元: 32 元素 (512-bit)
     - 根据 op 类型和 ReduceInfo 决定规约轴 tile 大小
  
  4. 生成嵌套循环:
     scf.forall (%iv_parallel) in (%c0 to %dim) step (%tile) {
       // 并行循环 - 映射到多个 AI Core
       scf.for (%iv_reduce) in (0 to %reduce_dim) step (1) {
         // 规约循环 - 沿规约轴迭代
         %elemwise_result = linalg.elemwise_binary {mul} (%loaded_A, %loaded_B)
         %reduced = linalg.reduce {add} (%elemwise_result)
       }
     }
  
  5. applyScheduleImpl:
     → 生成 transform dialect 的 schedule 脚本
     → AutoScheduleInterpreter pass 执行脚本
     → EraseAutoSchedule pass 清理标记
```

### 7.2 SingleCubeScheduler — 单 Cube 调度器

处理 `SingleCube`（纯 matmul）

```
算法:
  1. 提取 MatmulInfo:
     - tensor A id, tensor B id
     - transposeA / transposeB
     - numParallel / numReduction
  
  2. Tiling 决策:
     - M tile: 128 (Cube 单元 M 方向)
     - N tile: 128 (Cube 单元 N 方向)  
     - K tile: 32  (规约方向，影响 L1 buffer 用量)
  
  3. 生成循环:
     scf.forall (%m_tile, %n_tile) in (...) {
       // 并行循环映射到多个 AI Core
       scf.for (%k_tile) in (0 to K step 32) {
         // 规约循环
         %loaded_A = load A[m_tile, k_tile]
         %loaded_B = load B[k_tile, n_tile]
         %matmul_tile = linalg.matmul(%loaded_A, %loaded_B)
         %accum = add(%accum, %matmul_tile)
       }
       store %accum
     }
  
  4. 优化:
     - 双缓冲 (double buffer): 计算和加载 overlap
     - 数据复用: A/B 的 L1 cache 策略
```

### 7.3 ShallowCVScheduler — 浅层 CV 调度器

处理 `ShallowCV`（matmul + 少数 elementwise）

```
算法:
  1. 先调用 SingleCubeScheduler 处理 matmul 部分
  
  2. 检查 matmul 后的 elementwise 链:
     for each direct consumer of matmul result:
       if consumer is ElementWise AND depth < SHALLOW_LIMIT (通常 ≤3):
         内联到 matmul 的循环体中
       else:
         stop  # 这就是"浅"的含义——只融合紧跟 matmul 的一小段
  
  3. 融合约束:
     - elementwise op 不能改变 shape
     - 不能引入新的 broadcast 维度
     - 数据依赖路径不能分叉太多
```

### 7.4 Tiling 优化管线

```
Tiling 后处理 (hfusionTilingOptimizationPipeline):
  1. ConstantizeTilingData  → tiling 参数转为常量
  2. canonicalization       → 清理
  3. PackTilingData         → tiling 数据打包为 struct
  4. ArithToAffineConversion → arith → affine（循环界简化）
  5. canonicalization       → 清理
  6. SCFForLoopCanonicalization → 循环规范化
  7. canonicalization       → 最终清理
```

### Pass 序列

| 顺序 | Pass | 作用 |
|------|------|------|
| 1 | `ReorderOpsByBFS` | BFS 重排 op 顺序（可选）|
| 2 | **canonicalization** | |
| 3 | `Decompose(AFTER_HFUSION_FLATTEN)` | ★ 展平后分解（transpose 分解为 2-排列）|
| 4 | `AutoSchedule` | ★★ 自动调度（上述 3 个调度器）|
| 5 | `DecomposeMulti` | 多路分解 |
| 6 | `ConvertGenericToNamedOp` | 调度可能产生 generic → 转命名 |
| 7 | `ReorderOpsByBFS` | 再次重排（可选）|
| 8 | `ConstantizeTilingData` | tiling 参数常量化 |
| 9 | **canonicalization** (AfterAutoSchedule) | |
| 10 | `PackTilingData` | 打包为 struct |
| 11 | `ArithToAffineConversion` | arith → affine |
| 12 | **canonicalization** | |
| 13 | `SCFForLoopCanonicalization` | 循环优化 |
| 14 | **canonicalization** | |
| 15 | `WrapHostFunc` | 包装为 host 函数（单 kernel 模式）|

---

## 八、阶段七：postProcess（后处理）

**文件**: `HFusionPipelines.cpp:256-277`

| 顺序 | Pass | 作用 |
|------|------|------|
| 1 | `InlineBrc` | broadcast 再次内联 |
| 2 | `NormalizeOps` | ★ 调度后标准化（tile reduction 可能产生不支持 op）|
| 3 | `AddFFTSAddr` | FFTS 地址添加（ShallowCV 特有）|
| 4 | `HoistTensorEmpty` | empty op 提升 |
| 5 | `Decompose(AFTER_HFUSION_FLATTEN)` | transpose 最终分解 |

---

## 九、Triton 特殊路径

**文件**: `HFusionPipelines.cpp:295-307`

当 `enableTritonKernelCompile = true` 时，跳过标准融合管线：

```
preProcess → canonicalization →
  CanonicalizeTensorReshape →
  PropagateReshape(forHIVM=true) →  # 面向 HIVM 的 reshape 传播
  FoldTensorEmpty →
  NormalizeLastDimUnalignedTensorOp →
canonicalization → postProcess
```

不执行 flatten/outline/schedule，直接交给 HIVM 管线处理。

---

## 十、关键数据结构速查

| 结构 | 定义位置 | 用途 |
|------|---------|------|
| `FusionKind` | `HFusionEnums.td:203-215` | 9 种融合策略枚举 |
| `OpPattern` | `FusibleHelper.h:44-68` | op 分类（Elementwise/Matmul/Reduce/...）|
| `TypePattern` | `FusibleHelper.h:70-75` | 块内类型模式 |
| `FusibleBlock` | `FusibleBlock.h` | 可融合 op 块 |
| `FusibleHelper` | `FusibleHelper.h:77` | 融合规则引擎 |
| `FusibleBlockAnalyzer` | `FusibleBlockAnalyzer.h` | Union-Find + 规则验证 |
| `KernelInfo` | `KernelInfo.h:28-39` | 调度器输入：op 维度信息 |
| `SchedulerBase` | `AutoScheduleBase.h` | 调度器基类 |

---

## 十一、一次完整编译的日志示例

```
[preProcess]
  ArithToHFusion: converted 324 ops
  LinalgToHFusion: converted 156 ops
  Normalize: 9479 patterns applied
  Decompose(NO_CONSTRAINT): decomposed 12 ops
  SimplifyOps: simplified 89 ops

[preFlattenPass]
  BubbleUpExtractSlice: 23 slices moved
  FoldUnitExtentDims: 67 dims folded
  ComposeMultiReduce: 4 multi-reduces composed

[flattenAndFold]
  FlattenOps(Tidy): 201 ops flattened to 1D
  FoldTensorEmpty: 34 empty ops folded

[inferAndOutlineOp]
  InferFuncFusionKind: func@add_mul_relu → PureElemwise
  InferFuncFusionKind: func@matmul_add → ShallowCV
  OpFusion: outlined 3 kernels, 47 ops fused

[autoSchedule]
  Scheduler: PureElemwise → AnyPBRScheduler
  Scheduler: ShallowCV → ShallowCVScheduler
  Tiling: 128×128 for matmul, 32 for elemwise

[postProcess]
  NormalizeOps: 23 patterns applied
  HoistTensorEmpty: 12 empties hoisted

Done: 3 kernels generated, 247 ops eliminated through fusion
```
