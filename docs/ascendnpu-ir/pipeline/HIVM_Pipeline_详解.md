# HIVM 编译管线详解

> 基于 AscendNPU-IR 源码分析，覆盖 `convert-to-hivm-pipeline` 和 `optimize-hivm-pipeline` 两条管线的完整实现细节。241 个文件，85,538 行代码。

---

## 零、HIVM 定位

HIVM（Huawei Intermediate Virtual Machine）是将优化后的 HFusion IR 映射到昇腾 Da Vinci 硬件的底层 IR。它不是通用虚拟机，而是昇腾 910 系列 NPU 的指令级抽象。

```
HFusion (融合后 kernel)  ──→  HIVM  ──→  Standard MLIR  ──→  LLVM IR / Ascend 二进制
                            ↑ 两条管线
                    1. convert-to-hivm (上层→HIVM)
                    2. optimize-hivm    (HIVM 内部优化)
```

两条管线在 `bishengir-compile` 工具中串联调用：

```cpp
// PassPipeline.cpp:82-85
hivm::buildConvertToHIVMPipeline(pm, convertToHIVMOptions);  // 先转换
hivm::buildOptimizeHIVMPipeline(pm, options);                // 再优化
```

---

## 一、管线 A：convert-to-hivm-pipeline（上层→HIVM 转换）

**文件**: `ConvertToHIVMPipeline.cpp:29-44`，共 4 个 Pass。

```
┌─────────────────────────────────────────┐
│      convert-to-hivm-pipeline           │
├─────────┬─────────┬──────────┬──────────┤
│HFusion→ │ Triton  │ Tensor→  │ Convert  │
│  HIVM   │  Args   │  HIVM    │ ToHIVM   │
└─────────┴─────────┴──────────┴──────────┘
```

### 1.1 HFusionToHIVMConversion

将 HFusion 算子直接映射为 HIVM 等价算子：

| HFusion Op | HIVM Op | 说明 |
|------------|---------|------|
| `hfusion.elemwise_binary` | `hivm.v_add/v_mul/v_sub` | 逐元素→向量指令 |
| `hfusion.matmul` | `hivm.mmad` (Cube) 或 `hivm.macro_instr` | 矩阵乘→Cube 单元 |
| `hfusion.reduce` | `hivm.v_reduce` | 规约→向量规约指令 |
| `hfusion.broadcast` | `hivm.v_brc` | broadcast→向量广播 |
| `hfusion.transpose` | `hivm.v_transpose` | 转置→向量转置 |
| `hfusion.load/store` | `hivm.load_scalar/store` | 加载/存储 |
| `hfusion.concat` | `hivm.v_concat` | 拼接→向量拼接 |

**关键选择**: `mmMapMode` 决定矩阵乘的映射方式：
- `CoreOp`（默认）：`hivm.mmad`，编译器完全控制 tiling
- `MacroInstr`（Triton）：`hivm.macro_instr`，保留高层语义给执行引擎

### 1.2 TritonGlobalKernelArgsToHIVMOp

Triton kernel 参数适配 HIVM 约定（可选）。

### 1.3 TensorToHIVMConversion

残余 `tensor.*` op 转换（`tensor.pad/extract_slice` 等）。

### 1.4 ConvertToHIVMOp

收尾转换，处理未被前面 Pass 覆盖的 `memref.copy` 等 op，统一转为 `hivm.copy`。

---

## 二、管线 B：optimize-hivm-pipeline（HIVM 优化）

**文件**: `HIVMPipelines.cpp:395-413`，`buildOptimizeHIVMPipeline()`。

这是整个代码库最长的单条管线，共 5 大阶段，约 60 个 Pass。

```
┌──────────────────────────────────────────────────────────────────┐
│                   optimize-hivm-pipeline                         │
├──────────┬──────────┬──────────┬──────────┬─────────────────────┤
│   Init   │ PreBuf   │Bufization│ PostBuf  │   Finalize + Lower  │
└──────────┴──────────┴──────────┴──────────┴─────────────────────┘
```

### canonicalizationHIVMPipeline — 贯穿全管线的锚点

**文件**: `HIVMPipelines.cpp:38-48`

每个阶段之间必调，标准清理序列：

```
ArithToAffine → CanonicalizeIterArg → ExtendedCanonicalizer → 
SCFForLoopCanonicalization → CSE → ExtendedCanonicalizer → 
HIVMOptSinglePoint → ExtendedCanonicalizer → DeadStoreElimination
```

---

## 三、阶段零：Init（初始化）

| Pass | 作用 |
|------|------|
| `InitEntryKernel` | 标记入口 kernel 函数属性（`hacc.device` + `hivm.kernel`）|

---

## 四、阶段一：hivmPreBufferizationOptimizationPipeline（Buffer化前优化）

**文件**: `HIVMPipelines.cpp:162-268`，约 30 个 Pass。

### 核心逻辑

在 Tensor SSA 层面做尽可能多的优化，因为 Buffer 化后 SSA 关系断裂。

### 4.1 重塑与标准化

```
PropagateReshape(forHIVM=true)  → reshape 传播（消除转换引入的 reshape）
RemoveRedundantLoopInit         → 去除冗余循环初始化
NormalizeMatmul                 → ★ 矩阵乘标准化（对齐 Cube 单元要求）
NormalizeConvOps                → ★ 卷积标准化（1907 行）
NormalizeBitwiseSelect          → bitwise select 标准化
InlineFixpipe                   → 内联 FixPipe（修正 Pipe 属性）
```

### 4.2 CV WorkSpace 管理（自动 Load/Store 插入）

**文件**: `HIVMPipelines.cpp:147-159`

```
阶段 A: InsertLoadStoreForMixCV
  → 混合 Cube+Vector kernel 的自动 load/store 插入
  → 新路径: InsertLoadStoreForMixCV + InsertLoadStoreForScalar

阶段 B: InsertWorkSpaceForMixCV
  → 为 MixCV 插入 workspace 分配

阶段 C: BindWorkSpaceArg
  → 将 workspace 绑定为 kernel 参数
```

### 4.3 Tiling 与循环优化

```
TileBatchMMIntoLoop       → batch matmul 展开为 scf.for 循环
TileCubeVectorLoop        → Cube/Vector 循环混合 tiling
  ┌─ tileMixCubeLoop      (default=1, 不 tile)
  └─ tileMixVectorLoop     (default=1)
```

### 4.4 InferFuncCoreType — Core 类型推断

```
分析每个 kernel 使用的指令类型：
  - 只有 Cube 指令 (mmad/macro_instr)  → AIC (AI Core)
  - 只有 Vector 指令                    → AIV (AI Vector)
  - 混合 Cube+Vector                    → MixCV (AI Core + Vector)
  
核心类型影响：
  - 内存层级选择（L1/L0C/UB）
  - 同步策略（同 Core 内 vs 跨 Core）
  - 物理 block 数量
```

### 4.5 Pipeline 与 Double Buffer

```
AutoBlockifyParallelLoop    → 自动 blockify 并行循环（Triton）
MarkMultiBuffer             → ★ 双缓冲标记
  enableAuto: 自动多缓冲
  workspaceMultiBufferNum: workspace 缓冲数
  
InlineOTFBroadcast          → 内联在线广播
CVPipelining(enableSkewMode) → ★ Cube-Vector 软件流水线化（2224 行）
  → 让 Cube 计算与 Vector 计算 overlap
  → skew mode: 错开执行时间窗口
```

### 4.6 内存规划

```
AutoInferBufferSize         → ★ 自动推断 buffer 大小
ConstantizeBufferSize       → buffer 大小常量化
SetBufferSize               → 设置最终 buffer 大小

PlanMemory(GLOBAL_WORKSPACE_PLAN) → ★★ 全局 workspace 内存规划（2885 行）
  → 详见第七节
```

### 4.7 同步注入（跨 Core）

**文件**: `HIVMPipelines.cpp:71-101`

```
MarkRealCoreType            → 标记 core-type 属性（gm/workspace/ub 读写）
  ↓
CrossCoreGSS / InjectBlockSync:
  - CrossCoreGSS（优先）: 图同步求解器（跨 Core 版本）
  - InjectBlockSync（回退）: block 同步注入
    blockAllSync: 全局 barrier 模式
    assumeAliveLoops: 假设循环存活
  ↓
MarkRealCoreType(remove=true) → 清理临时属性
```

### 4.8 收尾

```
SplitMixKernel             → 拆分混合 kernel（AIC/AIV 分离）
InlineScope                → 内联 scope
TileAndBindSubBlock        → 子块 tiling + 绑定
FoldTensorEmpty            → 折叠 tensor.empty
LoopInvariantCodeMotion    → 循环不变量外提
LoopInvariantSubsetHoisting → 循环不变量子集提升
CloneTensorEmpty           → 克隆 tensor.empty（避免 buffer 化冲突）
InlineOTFLoadStore         → 内联 OTF load/store
```

---

## 五、阶段二：bufferizationPipeline（Buffer 化）

**文件**: `HIVMPipelines.cpp:112-145`

### 核心：One-Shot Bufferization

```
输入: Tensor SSA (虚拟值、无地址)
输出: MemRef (真实内存、有地址)
```

```
1. SinkOpToConsumerInLoop           → 下沉 op 到 consumer 循环内（ubuf-saving）
2. OptimizeDpsOpWithYieldedInsertSlice → DPS op 优化（Triton）
3. CloneTensorEmpty                 → 克隆 empty（Triton）
4. OneShotBufferize                 → ★★ 核心 buffer 化
   配置:
   - bufferizeFunctionBoundaries = true  → 函数边界 buffer 化
   - allowReturnAllocsFromLoops = true   → 循环内 alloc 可返回
   - allowUnknownOps = true              → 未知 op 穿透
5. canonicalization                 → 清理
6. ConvertToHIVMOp                  → memref.copy → hivm.copy
7. DropEquivalentBufferResults ×2   → 去除等效 buffer 返回值
8. canonicalization                 → 最终清理
```

---

## 六、阶段三：hivmPostBufferizationOptimizationPipeline（Buffer 化后优化）

**文件**: `HIVMPipelines.cpp:295-393`

Buffer 化后可以访问真实内存地址，执行以下优化：

### 6.1 循环与块映射

```
LiftZeroRank                     → 0 秩操作提升
MapForToForall                   → scf.for → scf.forall（标记并行性）
MapForallToBlocks                → forall → 物理 block 映射
HIVMDecomposeOp(phase=1)          → 算子分解（初始化阶段）
```

### 6.2 同步块锁

**文件**: `HIVMPipelines.cpp:281-293`

```
SyncBlockHoisting                → 同步块提升到循环外
BindSyncBlockLockArg             → 绑定同步锁参数
InsertInferSyncBlockLockNumAndInitFunc → 推断锁数量+初始化函数
SyncBlockLockLowering            → 同步锁 lowering
```

### 6.3 内存布局与作用域

```
NonContiguousReshapeToCopy       → 非连续 reshape → hivm.copy
InferHIVMMemScope                → ★ 推断内存作用域（L1/L0C/UB/GM）
  → 根据 op 的 Core 类型和 buffer 用途自动分配层级
HIVMDecomposeOp(phase=2)          → copy_ub_to_ub 分解
HIVMAggregatedDecomposeOp ×5      → 聚合算子多阶段分解:
  BEFORE_HIVM_STRIDE_ALIGNMENT     → stride 对齐前的预分解
  AFTER_RECOGNIZE_DEINTERLEAVE     → deinterleave 识别后分解
  AFTER_RECOGNIZE_BROADCAST        → broadcast 识别后分解
  AFTER_HIVM_STRIDE_ALIGNMENT      → stride 对齐后分解 (vconcat)
  AFTER_INFER_HIVM_DATA_LAYOUT     → 数据布局推断后分解
  AFTER_HIVM_FLATTEN_OPS           → ops 展平后分解
  AFTER_LIFT_LOWEST_STRIDE         → stride 提升后分解
```

### 6.4 数据布局

```
InferHIVMDataLayout              → ★ 推断数据布局
  → 连续访问 → ND (Normal) 格式
  → 非连续访问 → NZ (Fractal) 格式
  → hivm.copy → hivm.nd2nz (格式转换)
  
RemoveHIVMDataLayoutAnnotation   → 清理布局标注
```

### 6.5 存储对齐

```
AlignAllocSize                   → 对齐 alloc 大小（UB: 32B, L1: 32B, L0C: 512B）
MarkStrideAlign                  → 标记 stride 对齐
FoldAllocReshape                 → 折叠 alloc reshape
EnableStrideAlign                → 启用 stride 对齐
```

### 6.6 Buffer 与内存

```
FlattenOps                       → ops 展平（buffer 化后版本）
ReduceRankSubview                → 降低 subview 秩
LiftLowestStride                 → 提升最低 stride
AllocExtraBuffer                 → 分配额外 buffer
InferHIVMMemScope                → 再次推断内存作用域
InlineLoadCopy                   → 内联 load copy
MarkMultiBuffer(localOnly=true)  → 局部 buffer 的多缓冲
PlanMemory(memoryDisplay)        → ★ 局部内存规划 + 可视化
```

### 6.7 Lowering 与同步

```
HIVMLowerToLoops                 → ★★ HIVM op → scf.for 循环 lowering
  → ImplByScalarOpInterface.decomposeOperation()
  → 每个 HIVM op 提供自己的 scalar 实现

HIVMDecomposeOp(phase=3)          → 标量分解（int32 ext op 等）

CreatePreload                    → 预加载 pass（enablePreload）

hivmNormSyncPipeline:            → ★ 同步管线
  ├─ GraphSyncSolver (优先)      → 图同步求解器
  └─ InjectSync (回退)           → 自动同步注入
```

---

## 七、记忆层级与同步系统详解

### 7.1 Da Vinci 内存层次

HIVM 直接映射昇腾 910 的物理内存层级：

| HIVM 作用域 | 硬件 | 大小 | 对齐 | 用途 |
|------------|------|------|------|------|
| `GM` | Global Memory (HBM) | 32GB | - | 持久数据 |
| `L1` | L1 Buffer | 512KB | 32B | Cube 单元缓存 |
| `L0C` | L0C Buffer | 128KB | 512B | Cube 计算缓冲区 |
| `UB` | Unified Buffer | 192KB | 32B | Vector 单元缓冲区 |
| `workspace` | Workspace | 动态 | 32B | 跨 Core 共享区 |

**代码常量** (`PlanMemory.cpp:50-57`):
```cpp
ubAlignSize = 32 * 8;       // 32B
ubSpaceSize = 192 * 1024;   // 192KB
l1AlignSize = 32 * 8;       // 32B
l1SpaceSize = 512 * 1024;   // 512KB
l0cAlignSize = 512 * 8;     // 512B
l0cSpaceSize = 128 * 1024;  // 128KB
workSpaceAlignSize = 32 * 8; // 32B
```

### 7.2 PlanMemory — 内存规划算法

**文件**: `PlanMemory.cpp` (2,884 行)

```
算法: Interval-based Memory Allocation

输入: kernel 内所有 buffer alloc op
输出: 每个 buffer 的偏移地址 (offset)

步骤:
1. 构建 buffer 生命期区间图
   → 遍历 scf.for/scf.forall 循环体
   → 对每个 alloc: 
     - start = 首次 use 的 op 序号
     - end   = 最后 use 的 op 序号

2. 按缓冲类型分层规划:
   GLOBAL_WORKSPACE_PLAN:
     → 全局复用: 分析跨循环的 buffer 生命期
     → 冲突检测: 如果两个 buffer 生命期重叠且大小冲突
     → 复用策略: 生命期不重叠 → 可共享地址空间

3. 对齐:
   → UB/L1 buffer: 对齐到 32B
   → L0C buffer: 对齐到 512B

4. Buffer 复用优化:
   → isReusableCastOp: 1D contiguous cast 可复用
   → isReusableNarrowWidth: 窄宽度操作可复用
   → isReusableVSelOp: v_select 条件分支可复用

5. 输出:
   → 每个 alloc 注入 offset 属性
   → 插入 pointer_cast (base + offset)
```

### 7.3 GraphSyncSolver — 图同步求解器

**文件**: `GraphSyncSolver/SyncSolver.cpp` (2,629 行) + 8 个子文件

```
算法: 基于依赖图的同步事件最小化

步骤:
1. IRTranslator 构建同步 IR (SyncIR)
   → 将 HIVM ops 翻译为 SyncNode (读写节点)
   → 分析每个 Node 的内存访问范围 (MemInfo)
   → 构建依赖图 (GraphSolver)

2. SyncAnalyzer 分析冲突:
   → RAW (Read After Write) 冲突
   → WAW (Write After Write) 冲突
   → WAR (Write After Read) 冲突
   → 按 PIPE 类型分组: PIPE_V / PIPE_M / PIPE_MTE1/2/3

3. EventIdSolver 分配同步事件 ID:
   → 硬件限制: 有限数量的 event ID (通常 32-64)
   → 图着色: 不冲突的同步对可复用同一 event ID
   → 溢出处理: event ID 不够时启用 barrier_all

4. SyncSolverCodeGen 生成同步指令:
   → set_wait(event_id)  ← 等待前序完成
   → set_flag(event_id)  → 标记当前完成
   → pipe_barrier(PIPE_ALL)  ← 全局屏障（回退方案）

5. 优化:
   → reusedPairs: 可复用的同步对
   → backwardSync: 反向同步事件（后→前）
   → barrierAllPairs: 合并为全局屏障
```

### 7.4 InjectSync — 自动同步注入（回退方案）

**文件**: `InjectSync/InjectSync.cpp` (126 行入口) + 7 个子文件

```
当 GraphSyncSolver 不可用时（enableHIVMGraphSyncSolver=false），
使用更简单的同步策略:

策略 1: BARRIERALL 模式 (enableHIVMInjectBarrierAllSync)
  → 在每个 HIVM op 前插入 pipe_barrier(PIPE_ALL)
  → 简单但低效，适合调试

策略 2: AutoInjectSync (默认)
  → MemoryDependentAnalyzer 分析内存依赖
  → 为每对冲突 op 插入 set_wait/set_flag
  → enableUnitFlag: 使用单元级 flag（细粒度）
```

### 7.5 HIVM 指令集（核心）

| 类别 | Op | 硬件单元 | 说明 |
|------|-----|---------|------|
| **标量** | `load_scalar` | Scalar | 加载标量 |
| | `set_atomic` | Scalar | 原子操作 |
| | `set_ctrl` | Scalar | 控制寄存器 |
| **向量** | `v_add/v_mul/v_sub` | Vector | 逐元素算术 |
| | `v_reduce` | Vector | 规约 (sum/max/min) |
| | `v_brc` | Vector | Broadcast |
| | `v_transpose` | Vector | 转置 |
| | `v_cast` | Vector | 类型转换 |
| | `v_concat` | Vector | 拼接 |
| | `v_select` | Vector | 条件选择 |
| | `gather_load` | Vector | 聚合加载 |
| | `scatter_store` | Vector | 分散存储 |
| **Cube** | `mmad` | Cube | 矩阵乘加 |
| | `macro_instr` | Cube(Triton) | 宏指令 |
| **DMA** | `copy` | DMA | 内存拷贝 |
| | `nd2nz` | DMA | ND→NZ 格式转换 |
| | `dcci` | DMA | L1 cache 失效 |
| **同步** | `set_wait` | Sync | 等待事件 |
| | `set_flag` | Sync | 标记事件 |
| | `pipe_barrier` | Sync | Pipeline 屏障 |
| **控制** | `get_block_idx` | Control | 获取 block ID |
| | `get_block_num` | Control | 获取 block 总数 |
| | `get_sys_cnt` | Control | 系统计数 |

---

## 八、阶段四：Finalize（最终化）

**文件**: `HIVMPipelines.cpp:404-413`

```
InlineScope(forceInline=true)     → 强制内联所有 scope.scope
EnableHIVMCCompatiblePrint       → C 兼容打印
AnnotationLowering               → annotation lowering
InsertInitAndFinishForDebug       → 调试用 init/finish 插入
MarkDisableLoad                  → 标记 disable_load（硬件特性）
SyncBlockLockPipeline(Finalize)   → 同步块锁最终化:
  MarkSyncBlockLockWithSubblock   → 子块标记
  InsertFreeLockVarBeforeReturn   → 释放锁变量
```

---

## 九、阶段五：ConvertHIVMToStandard（最终 Lowering）

**文件**: `HIVMToStandard.cpp` (1,987 行)

将 HIVM Dialect 完全 Lowering 为标准 MLIR（scf + memref + arith + func）：

```
hivm.v_add(%a, %b)  ──→  scf.for (%i) { 
                             %ai = memref.load %a[%i]
                             %bi = memref.load %b[%i]  
                             %ci = arith.addf %ai, %bi
                             memref.store %ci, %c[%i]
                           }

hivm.set_wait(%ev)   ──→  (保留为硬件 intrinsic，不下沉)
hivm.pipe_barrier     ──→  (保留为硬件 intrinsic)
```

**关键**: 同步指令和 DMA 指令保留为硬件 intrinsic，不做进一步 lowering，最终由硬件执行引擎直接解释。

---

## 十、完整编译流程示例

```
输入: HFusion kernel (matmul + add + relu)
  func.func @kernel(%A: tensor<128x128xf16>, %B: tensor<128x128xf16>) 
       → tensor<128x128xf16>

[Pipeline A: convert-to-hivm]
  HFusionToHIVM:
    hfusion.matmul → hivm.mmad         (Cube)
    hfusion.add    → hivm.v_add        (Vector)
    hfusion.relu   → hivm.v_max(%x, 0) (Vector)

[Pipeline B: optimize-hivm - PreBuf]
  InferFuncCoreType: MixCV (Cube + Vector)
  TileCubeVectorLoop: tileCube=2, tileVector=4
  CVPipelining: Cube 和 Vector 流水线化 (skew=2)
  PlanMemory: L1=128B, UB=64B, workspace=256B

[Pipeline B: Bufferization]
  Tensor SSA → MemRef (one-shot)

[Pipeline B: PostBuf]
  InferHIVMMemScope: mmad → L1, v_add → UB, v_max → UB
  InferHIVMDataLayout: contiguous → ND
  PlanMemory(local): UB offset=0, UB offset=64
  HIVMLowerToLoops: mmad/v_add/v_max → scf.for 循环
  GraphSyncSolver: 插入 2 个 set_wait/set_flag 对

[Finalize + Lower]
  HIVMToStandard: hivm.* → arith + memref + scf
  硬件 intrinsic 保留: set_wait, set_flag, pipe_barrier

输出: Standard MLIR + 硬件同步 intrinsic → LLVM IR
```

---

## 十一、关键配置选项

| 选项 | 默认 | 作用 |
|------|------|------|
| `enableTritonKernelCompile` | false | Triton kernel 模式 |
| `enableUbufSaving` | false | UB buffer 节省模式 |
| `enableHIVMGraphSyncSolver` | true | 图同步求解器开关 |
| `disableHIVMAutoInjectSync` | false | 禁用自动同步注入 |
| `enableHIVMInjectBarrierAllSync` | false | barrier_all 全局同步 |
| `enableHIVMCrossCoreGSS` | true | 跨 Core 图同步 |
| `enableAutoMultiBuffer` | true | 自动多缓冲 |
| `enablePreload` | false | 预加载模式 |
| `enableAutoBindSubBlock` | false | 自动子块绑定 |
| `tileMixCubeLoop / tileMixVectorLoop` | 1/1 | 混合循环 tiling 因子 |
