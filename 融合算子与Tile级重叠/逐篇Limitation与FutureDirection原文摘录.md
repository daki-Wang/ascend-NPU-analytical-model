# 逐篇 Limitation / Future Direction 原文摘录

来源：本地论文资料库，2026-11-03 核查；32 份 PDF、31 项研究。所有页码为 PDF 文件页序。仅做换行、断词、连字及排版粘连规范，不翻译、不改写；未发现明确段落时保留缺项状态。独立章节与正文中的范围/开销自述分别标注。

<a id="p01"></a>
### [P01] HyperParallel-MoE: Multi-Core Interleaved Scheduling for Fast MoE Training on Ascend NPUs

**原 PDF**：[Ascend-moe.pdf](<MoE_NPU%E8%AE%A1%E7%AE%97%E9%80%9A%E4%BF%A1%E8%9E%8D%E5%90%88/Ascend-moe.pdf>)

**章节状态**：未发现独立 Limitations 章节；仅收录结论中作者明确写出的 future work。

**Future direction（作者原文）** · `8 Conclusion` · PDF 第 15 页

> These results indicate that scheduling-layer optimization can expose AIC/AIV parallelism on existing hardware; full-model evaluation and broader operator coverage remain future work.

<a id="p02"></a>
### [P02] UniEP: Efficient and Deterministic MoE Expert Parallelism

**原 PDF**：[uniep.pdf](<MoE_NPU%E8%AE%A1%E7%AE%97%E9%80%9A%E4%BF%A1%E8%9E%8D%E5%90%88/uniep.pdf>)

**章节状态**：未发现独立 Limitations/Future Work 章节；保留作者在模型精度和实现性能讨论中明确承认的限制。

**Limitation / scope / overhead（作者正文自述）** · `Tuning Results and Overhead / Model Accuracy` · PDF 第 14 页

> We note that our analytical model tends to yield optimistic latency estimates, as cache contention at runtime can degrade performance in ways that are not yet well captured by the model.

**Limitation / scope / overhead（作者正文自述）** · `7.5 Discussion / Performance Impact of Triton` · PDF 第 20 页

> While Triton-based kernels may underperform CUDA kernels on the latest GPU architectures, UniEP design is language-agnostic and can be readily re-implemented in CUDA to close this gap, which has been proven in DeepGEMM [10] (https://github.com/deepseek-ai/DeepGEMM/pull/304)

<a id="p03"></a>
### [P03] DA-MoE / Decoding the Skew: Distribution-Aware Fused MoE Inference

**原 PDF**：[2607.23099v3.pdf](<MoE_NPU%E8%AE%A1%E7%AE%97%E9%80%9A%E4%BF%A1%E8%9E%8D%E5%90%88/2607.23099v3.pdf>)

**章节状态**：无独立 Limitations / Future Work 章节。以下为作者对 oracle 覆盖、在线开销与 HIP 支持边界的明确自述，另保留未来架构启示；架构启示不等同于本文承诺的后续实现。

**Future direction（作者原文）** · `VI Discussion & Architectural Implications` · PDF 第 11 页

> Overall, future inference accelerators need two forms of adaptivity: low-overhead device-side control flow, and tile-shape flexibility driven by the observed expert distribution.

**Limitation / scope / overhead（作者正文自述）** · `V-B / Table II — oracle coverage` · PDF 第 9 页

> At token count 512, both DA-MoE and NoDA sit approximately 20% above the oracle, indicating that the oracle selects a kernel outside the coverage of both the exemplar matcher and the static bucket choice.

**Limitation / scope / overhead（作者正文自述）** · `VI-A GPU-Side Branching Should Be a Hardware Primitive` · PDF 第 10 页

> This device-side branch is still not free as the full DAMoE selection path adds about 14μs per graph, including histogram construction, sorting, exemplar matching, and conditional dispatch. DeepSeek-V3 and Kimi K2 amortize this cost because their MoE workloads are large enough for better kernel selection to dominate the overhead.

**Limitation / scope / overhead（作者正文自述）** · `VI-A GPU-Side Branching Should Be a Hardware Primitive` · PDF 第 10–11 页

> Smaller MoE variants, such as Qwen3 models [14], may not expose enough latency savings, so distribution-aware dispatch is beneficial only when the kernel-selection gain exceeds the dispatch cost.

**Limitation / scope / overhead（作者正文自述）** · `VI-A GPU-Side Branching Should Be a Hardware Primitive` · PDF 第 11 页

> A similar issue appears on AMD accelerators and their programming interface, HIP. HIP Graphs supports graph capture and replay, but do not expose an equivalent of CUDA’s device-side conditionalSWITCHnode [19]. DA-MoE on HIP would therefore need CPU copy-back, removing the benefit of GPU-resident online dispatch. Thus, graph replay alone is insufficient for MoE serving because the runtime also requires a low-overhead GPU-resident control flow.

<a id="p04"></a>
### [P04] LENS: Latency Prediction for LLM Inference on NPU Systems

**原 PDF**：[2606.18042v2.pdf](<MoE_NPU%E8%AE%A1%E7%AE%97%E9%80%9A%E4%BF%A1%E8%9E%8D%E5%90%88/2606.18042v2.pdf>)

**章节状态**：未来工作章节完整摘录；原文将跨NPU扩展和生产系统级预测分别列为后续方向。

**Limitation / scope / overhead（作者正文自述）** · `6. Future Work` · PDF 第 10 页

> While LENS achieves efficient profiling with only two measurements per bucket, its predictions remain confined to the NPU on which profiling was conducted. NeuSight [22] and Habitat [38] address the analogous limitation in the GPU domain by decomposing inference into kernel-level units and applying analytical scaling factors derived from hardware characteristics such as compute throughput and memory bandwidth. LENS, however, operates on bucket-level E2E latency without kernel decomposition, rendering kernel-level scaling factors directly inapplicable.

**Future direction（作者原文）** · `6. Future Work` · PDF 第 10 页

> Whether bucket-level E2E latency itself exhibits a consistent scaling relationship across NPUs, and what set of hardware features most accurately predicts such a relationship, remains an open question. If such a relationship can be established, LENS could be extended to support prediction across heterogeneous NPU platforms without requiring profiling on each. We leave the empirical characterization of cross-NPU scaling behavior to future work.

**Limitation / scope / overhead（作者正文自述）** · `6. Future Work` · PDF 第 10 页

> LENS currently predicts latency under an offline serving setting. Production serving environments introduce system-level factors that affect E2E latency, such as queuing delay, inter-stage communication overhead in pipeline parallelism, and request routing across heterogeneous accelerators.

**Future direction（作者原文）** · `6. Future Work` · PDF 第 10 页

> Extending LENS to system-level performance prediction that accounts for these factors is a direction for future work.

<a id="p05"></a>
### [P05] AdaptCore: Adaptive Matrix Multiplication for Dynamic Shapes on Ascend NPUs

**原 PDF**：[Zhou 等 - 2026 - Adaptive Matrix Multiplication for Dynamic Shapes on Ascend NPUs.pdf](<MoE_NPU%E8%AE%A1%E7%AE%97%E9%80%9A%E4%BF%A1%E8%9E%8D%E5%90%88/Zhou%20%E7%AD%89%20-%202026%20-%20Adaptive%20Matrix%20Multiplication%20for%20Dynamic%20Shapes%20on%20Ascend%20NPUs.pdf>)

**章节状态**：独立章节完整摘录；不把本文实验范围自行改写为作者限制。

**Limitation / scope / overhead（作者正文自述）** · `7 Limitations and Future Work / Architectural Generality` · PDF 第 11 页

> While the methodology of AdaptCore (hardware-aware tiling taxonomy and optimization modeling) is theoretically generalizable, its specific templates and empirical estimates heavily rely on Ascend hardware. Porting AdaptCore to other Domain-Specific Architectures (DSAs) with different memory hierarchies would require redefining the taxonomy and analytical performance models.

**Limitation / scope / overhead（作者正文自述）** · `7 Limitations and Future Work / Operator Scope and Workload Shifts` · PDF 第 11 页

> Currently, AdaptCore focuses on matrix multiplication operators (e.g., MatMul, GroupMM, BatchMM). As AI workloads evolve towards ultra-low precision (FP8/INT4), highly unstructured sparsity, or complex operator fusion, execution bottlenecks may shift.

**Future direction（作者原文）** · `7 Limitations and Future Work / Operator Scope and Workload Shifts` · PDF 第 11 页

> Moving forward, we plan to extend our methodology to a broader spectrum of complex operators (e.g., FlashAttention, Convolution) by formalizing their hardware constraints.

**Future direction（作者原文）** · `7 Limitations and Future Work / Agent-Driven Operator Generation` · PDF 第 11 页

> In the future, we aim to integrate our architectural insights into an Ascend operator agent to assist efficient bottleneck localization, bug elimination, and performance tuning.

<a id="p06"></a>
### [P06] AscendOptimizer: Episodic Agent for Ascend NPU Operator Optimization

**原 PDF**：[operatorAgent.pdf](<MoE_NPU%E8%AE%A1%E7%AE%97%E9%80%9A%E4%BF%A1%E8%9E%8D%E5%90%88/operatorAgent.pdf>)

**章节状态**：Appendix C 的 Limitations 段落完整摘录。

**Limitation / scope / overhead（作者正文自述）** · `C Limitations and Broader Impacts / Limitations` · PDF 第 18 页

> The results apply to existing, functionally correct AscendC implementations on Ascend 910B2 with CANN 8.3. Performance still depends on reusable structure in the starting code, and broader transfer across chips, CANN versions, repositories, and non-transductive benchmark splits remains future work. We also account for hardware-evaluation budget, but finer-grained wall-clock and deployment-cost analysis can be expanded.

<a id="p07"></a>
### [P07] CANN Bench: Ascend-Oriented Operator-Generation Benchmark

**原 PDF**：[cannBenchmark.pdf](<MoE_NPU%E8%AE%A1%E7%AE%97%E9%80%9A%E4%BF%A1%E8%9E%8D%E5%90%88/cannBenchmark.pdf>)

**章节状态**：结论中明确列出现行限制与三类计划扩展；未将第3.4节安全策略的讨论额外改写为 limitation。

**Limitation / scope / overhead（作者正文自述）** · `4 Conclusion` · PDF 第 16 页

> We are explicit about the current limits of this release. Precision coverage is restricted to FP16, BF16, FP32, and INT8; FP8, HiF8, MXFP8, FP4, and MXFP4 are deliberate near-term extensions rather than shipped capabilities. Several reward-hacking surfaces remain intentionally conservative: multi-stream execution is restricted rather than fully ranked, because stream-level visibility across torch_npu versions is still fragile; evaluator-machine hardening is a controlled leaderboard policy rather than an automatic property of every local install; and opaque pre-compiled binary loading is presently addressed by policy and manual review rather than automated provenance checks. Input refresh and address-diversity controls should be treated as run-profile-dependent hardening knobs unless enabled by the official leaderboard configuration. The benchmark also begins on a single hardware target (Ascend 910B2) and a single kernel programming surface, so coverage of newer Ascend silicon and additional DSLs is a near-term task rather than an established result.

**Future direction（作者原文）** · `4 Conclusion` · PDF 第 16 页

> Looking forward, CANN Bench is intended to evolve with the Ascend ecosystem rather than freeze at release. Three extensions are scheduled: the online leaderboard portal will index and date every submission so that results stay directly citable; kernel-DSL coverage will extend beyond the initial Ascend-optimized path to Triton, PyPTO, TileLang, and other CANN-compatible paradigms under the same specification and scoring protocol; and precision coverage will expand to FP8, MXFP8, FP4, MXFP4, HiF8, and related low-precision formats as operators and baselines are validated. Versioned performance baselines, refreshable through a baseline-update utility planned for an upcoming release, ensure that reported speedups continue to track the current attainable reference rather than a stale snapshot as CANN and Ascend hardware progress.

<a id="p08"></a>
### [P08] NPUMeter: Automatic Operator Optimization for Ascend NPU with Accurate Analytical Performance Models

**原 PDF**：[Zhang 等 - 2026 - NPUMeter Automatic Operator Optimization for Ascend NPU with Accurate Analytical Performance Models.pdf](<MoE_NPU%E8%AE%A1%E7%AE%97%E9%80%9A%E4%BF%A1%E8%9E%8D%E5%90%88/Zhang%20%E7%AD%89%20-%202026%20-%20NPUMeter%20Automatic%20Operator%20Optimization%20for%20Ascend%20NPU%20with%20Accurate%20Analytical%20Performance%20Models.pdf>)

**章节状态**：全文未发现独立 Limitations/Future Work 章节；仅收录 Discussion 中作者明确说明的架构依赖和迁移工作。

**Limitation / scope / overhead（作者正文自述）** · `7 Discussion / Ascend-specific components` · PDF 第 21 页

> Some parts of NPUMeter are tied to the Ascend architecture and would require adaptation when porting to other NPUs. For example, the ND-to-Nz data layout conversion overhead is specific to Ascend NPUs and would not apply to architectures that natively support standard data layouts. In addition, architecture-dependent parameters such as memory bandwidth must be re-profiled for the target hardware.

**Limitation / scope / overhead（作者正文自述）** · `7 Discussion / Porting effort` · PDF 第 21 页

> Applying NPUMeter to a different NPU architecture requires analyzing the target accelerator’s pipeline organization, memory hierarchy, and execution model, and mapping them to the stages in our analytical model, or slightly making adjustments based on our generalizable insights. After that, architecture-specific parameters need to be obtained through profiling on the target platform.

<a id="p09"></a>
### [P09] TileSight: A First-Principles Tile-Centric Analytical GPU Performance Model from Cores to Clusters

**原 PDF**：[gpu-comp-com.pdf](<NPU_GPU%E6%80%A7%E8%83%BD%E5%BB%BA%E6%A8%A1%E4%B8%8E%E6%9E%B6%E6%9E%84/gpu-comp-com.pdf>)

**章节状态**：独立限制与未来工作章节已完整摘录；本条未把实验范围自行扩展成限制。

**Limitation / scope / overhead（作者正文自述）** · `7 Limitations and Future Work` · PDF 第 13 页

> TileSight targets regular, tile-structured programs whose runtime is dominated by resource utilization. It does not model data-dependent control flow, highly irregular memory access, instruction-level compiler decisions, undocumented warp/cooperative thread array (CTA) scheduling, or closed-source runtime behavior. The hardware abstraction focuses on throughput. Latency-bound cases such as small-batch decode attention, and multi-die effects such as B200 SM-to-HBM affinity, require finer-grained latency and topology parameters. TileSight also assumes tiles execute at a uniform rate across SMs; in reality SMs desynchronize, which makes L2 hit-rate prediction mildly optimistic for large-K GEMMs (§5.3). Finally, although TileSight has been validated as a TileLang cost model on selected GEMM workloads, broader compiler integration and non-GEMM schedule search remain future work.

<a id="p10"></a>
### [P10] TENET: A Framework for Modeling Tensor Dataflow Based on Relation-centric Notation

**原 PDF**：[ISCA_2021_TENET.pdf](<NPU_GPU%E6%80%A7%E8%83%BD%E5%BB%BA%E6%A8%A1%E4%B8%8E%E6%9E%B6%E6%9E%84/ISCA_2021_TENET.pdf>)

**章节状态**：未发现独立的 Limitations/Future Work 章节；正文明确未来工作仅涉及更高效的设计空间探索。

**Future direction（作者原文）** · `VI-B Design space exploration` · PDF 第 9 页

> To explore this design space, TENET takes less than one hour. We leave a more efficient design space exploration as future work.

<a id="p11"></a>
### [P11] ONNXim: A Fast, Cycle-Level Multi-Core NPU Simulator

**原 PDF**：[ONNXim_A_Fast_Cycle-Level_Multi-Core_NPU_Simulator.pdf](<NPU_GPU%E6%80%A7%E8%83%BD%E5%BB%BA%E6%A8%A1%E4%B8%8E%E6%9E%B6%E6%9E%84/ONNXim_A_Fast_Cycle-Level_Multi-Core_NPU_Simulator.pdf>)

**章节状态**：未发现独立的限制或未来工作章节；作者明确说明的验证范围限制只有 Gemmini 不支持的三类操作。

**Limitation / scope / overhead（作者正文自述）** · `III-C Validation With Core RTL Model` · PDF 第 3 页

> Since Gemmini does not support softmax, normalization, and skip connections, these operations were excluded from the validation.

<a id="p12"></a>
### [P12] Coherent Profiles: Enabling Efficient Reuse Distance Analysis of Multicore Scaling for Loop-based Parallel Programs

**原 PDF**：[Coherent_Profiles_Enabling_Efficient_Reuse_Distance_Analysis_of_Multicore_Scaling_for_Loop-based_Parallel_Programs.pdf](<NPU_GPU%E6%80%A7%E8%83%BD%E5%BB%BA%E6%A8%A1%E4%B8%8E%E6%9E%B6%E6%9E%84/Coherent_Profiles_Enabling_Efficient_Reuse_Distance_Analysis_of_Multicore_Scaling_for_Loop-based_Parallel_Programs.pdf>)

**章节状态**：未发现独立 Limitations/Future Work 章节；作者把可优化对象限定为 memory features。

**Limitation / scope / overhead（作者正文自述）** · `VI Related work` · PDF 第 11 页

> Our approach learns more per sample (a CRD profile), reducing the number of needed samples, but we can only optimize memory. ML learns very little per sample (an IPC value), but is very general and can optimize any architecture feature.

<a id="p13"></a>
### [P13] Analytical Modeling the Multi-Core Shared Cache Behavior With Considerations of Data-Sharing and Coherence

**原 PDF**：[Analytical_Modeling_the_Multi-Core_Shared_Cache_Behavior_With_Considerations_of_Data-Sharing_and_Coherence.pdf](<NPU_GPU%E6%80%A7%E8%83%BD%E5%BB%BA%E6%A8%A1%E4%B8%8E%E6%9E%B6%E6%9E%84/Analytical_Modeling_the_Multi-Core_Shared_Cache_Behavior_With_Considerations_of_Data-Sharing_and_Coherence.pdf>)

**章节状态**：未发现独立 Limitations/Future Work 章节；保留作者明确声明的 phase-transition 假设。

**Limitation / scope / overhead（作者正文自述）** · `III Model construction / Quantifying the insertion effect` · PDF 第 5 页

> Similar to StatCC, we assume that the shared cache accesses are uniform in the whole profiling interval, i.e., we do not consider the effect of program phase transitions.

<a id="p13v0"></a>
### [P13v0] Analytical Modeling the Multi-Core Shared Cache Behavior With Considerations of Data-Sharing and Coherence (early manuscript)

**原 PDF**：[multi-core_model.pdf](<NPU_GPU%E6%80%A7%E8%83%BD%E5%BB%BA%E6%A8%A1%E4%B8%8E%E6%9E%B6%E6%9E%84/multi-core_model.pdf>)

**章节状态**：未发现独立 Limitations/Future Work 章节；与期刊版分开记录，不能把两版实验误差合并。

**Limitation / scope / overhead（作者正文自述）** · `III Model construction / Quantifying the insertion effect` · PDF 第 4 页

> Similar to StatCC, we assume that the shared cache accesses are uniform in the whole profiling interval, i.e., we do not consider the effect of program phase transitions.

<a id="p14"></a>
### [P14] MDM: The GPU Memory Divergence Model

**原 PDF**：[Wang 等 - 2020 - MDM The GPU Memory Divergence Model.pdf](<NPU_GPU%E6%80%A7%E8%83%BD%E5%BB%BA%E6%A8%A1%E4%B8%8E%E6%9E%B6%E6%9E%84/Wang%20%E7%AD%89%20-%202020%20-%20MDM%20The%20GPU%20Memory%20Divergence%20Model.pdf>)

**章节状态**：本地版本未发现独立 Limitations / Future Work 章节，也未定位到作者明确自述 MDM 尚未覆盖范围或后续研究计划的段落；不将模型假设、实验范围或本次综合的判断改写为作者 limitation。

<a id="p15"></a>
### [P15] AMALI: An Analytical Model for Accurately Modeling LLM Inference on Modern GPUs

**原 PDF**：[Cao 等 - 2025 - AMALI An Analytical Model for Accurately Modeling LLM Inference on Modern GPUs.pdf](<NPU_GPU%E6%80%A7%E8%83%BD%E5%BB%BA%E6%A8%A1%E4%B8%8E%E6%9E%B6%E6%9E%84/Cao%20%E7%AD%89%20-%202025%20-%20AMALI%20An%20Analytical%20Model%20for%20Accurately%20Modeling%20LLM%20Inference%20on%20Modern%20GPUs.pdf>)

**章节状态**：第 3.2 节的四个 Limitation 标题是对 GCoM/已有模型的批评，不是 AMALI 自身限制，未收录；作者自述的 AMALI 后续方向仅发现 warp instruction number divergence 精度改进。

**Limitation / scope / overhead（作者正文自述）** · `6.5 Discussions` · PDF 第 13 页

> But there is still a room to further improve the accuracy by studying better strategy to model the warp instruction number divergence.

<a id="p16"></a>
### [P16] SYNPERF: A Hybrid Analytical-ML Framework for GPU Performance Prediction

**原 PDF**：[Zhang 等 - 2026 - SynPerf A Hybrid Analytical-ML Framework for GPU Performance Prediction.pdf](<NPU_GPU%E6%80%A7%E8%83%BD%E5%BB%BA%E6%A8%A1%E4%B8%8E%E6%9E%B6%E6%9E%84/Zhang%20%E7%AD%89%20-%202026%20-%20SynPerf%20A%20Hybrid%20Analytical-ML%20Framework%20for%20GPU%20Performance%20Prediction.pdf>)

**章节状态**：结论有明确 Future work 段；限制条目保留作者对 residual gap 的原始归因，不把前文假设改写成限制。

**Limitation / scope / overhead（作者正文自述）** · `VII Beyond Simulation / Model-guided optimization` · PDF 第 11 页

> We observe that a residual gap often persists. We hypothesize that this remaining gap may not be fully addressed through parameter tuning alone. It could originate from deeper issues, such as the kernel’s fundamental design or inherent limitations of Triton language [12], [58].

**Future direction（作者原文）** · `IX Conclusion` · PDF 第 12 页

> Future work will focus on two main areas. First, we will extend SYNPERF to complex distributed settings, incorporating support for multi-node clusters and advanced parallelism strategies such as Expert Parallelism (EP). Second, we plan to broaden our model-guided optimization method by developing automated tools that detect performance bottlenecks and enhance configuration logic for more production kernels.

<a id="p17"></a>
### [P17] SimAI: Unifying Architecture Design and Performance Tuning for Large-Scale Large Language Model Training with Scalability and Precision

**原 PDF**：[Wang 等 - SimAI Unifying Architecture Design and Performance Tuning for Large-Scale Large Language Model Trai.pdf](<NPU_GPU%E6%80%A7%E8%83%BD%E5%BB%BA%E6%A8%A1%E4%B8%8E%E6%9E%B6%E6%9E%84/Wang%20%E7%AD%89%20-%20SimAI%20Unifying%20Architecture%20Design%20and%20Performance%20Tuning%20for%20Large-Scale%20Large%20Language%20Model%20Trai.pdf>)

**章节状态**：无独立 Limitations 章节；正文自述均衡分布与 NIC/runtime 模拟范围，另有虚拟化、故障和通信特性扩展计划。

**Limitation / scope / overhead（作者正文自述）** · `3.4 Precise Simulation of Communication` · PDF 第 8 页

> In the simulation, we assume a balanced distribution, which has minimal impact on the simulation results.

**Future direction（作者原文）** · `7 Related work / Network simulators and scalability` · PDF 第 13 页

> In the future, SimAI will consider simulating network virtualization scenarios to evaluate various virtualization technologies (such as MasQ [15], etc.).

**Future direction（作者原文）** · `7 Related work / Fault simulation` · PDF 第 13 页

> We plan to incorporate simulations for supporting model faults in future work.

**Limitations and future work（作者原文）** · `9.1 NCCL environment variables` · PDF 第 17 页

> We can see besides those we think are unnecessary for simulation, only four environment variables are not yet supported, which are adaptive routing and Infiniband SHArP related features. We leave them as the future work of the communication simulator of SimAI.

**Limitation / scope / overhead（作者正文自述）** · `4.2 Communication Performance` · PDF 第 9 页

> This discrepancy arose because small message performance is more susceptible to inaccuracies, as we did not simulate the runtime software (e.g., libibverbs) and the NIC pipelines.

<a id="p18"></a>
### [P18] Analytical Resource Management for Fine-grained MoE Computation-Communication Overlap

**原 PDF**：[Analytical_Resource_Management_MoE_2609.07536.pdf](<%E6%80%A7%E8%83%BD%E6%A8%A1%E5%9E%8B%E4%B8%8E%E8%B5%84%E6%BA%90%E8%B0%83%E5%BA%A6/Analytical_Resource_Management_MoE_2609.07536.pdf>)

**章节状态**：无独立 Limitations 章节；结论中有明确 future work。

**Future direction（作者原文）** · `VI. Conclusion` · PDF 第 10 页

> Future work will evaluate the analytical model on newer GPU architectures and multi-node interconnects, and extend resource selection to decode and other computation– communication operators with fine-grained dependencies.

<a id="p19"></a>
### [P19] Weave: Fine-Grained Dynamic SM Scheduling in an MoE Megakernel for Compute-Communication Overlap

**原 PDF**：[Weave_Dynamic_SM_Scheduling_2609.21483v3.pdf](<%E6%80%A7%E8%83%BD%E6%A8%A1%E5%9E%8B%E4%B8%8E%E8%B5%84%E6%BA%90%E8%B0%83%E5%BA%A6/Weave_Dynamic_SM_Scheduling_2609.21483v3.pdf>)

**章节状态**：结论含明确 Limitations and future work 段，完整摘录。

**Limitations and future work（作者原文）** · `7 Conclusion — Limitations and future work` · PDF 第 11 页

> Due to hardware constraints, Weave is currently validated on a single 4 ×H100 NVLink node with EP=4. Extending to larger GPU counts and multi-node configurations is left for future work.

<a id="p20"></a>
### [P20] DynaNDE: Dynamic Near-Data Expert Scheduling for Batched MoE Inference

**原 PDF**：[DynaNDE_2609.00407.pdf](<%E6%80%A7%E8%83%BD%E6%A8%A1%E5%9E%8B%E4%B8%8E%E8%B5%84%E6%BA%90%E8%B0%83%E5%BA%A6/DynaNDE_2609.00407.pdf>)

**章节状态**：§6 Discussion 跨页段落完整摘录。

**Limitations and future work（作者原文）** · `6 Discussion` · PDF 第 8–9 页

> This work focuses on a single-NDP configuration. Extending DynaNDE to multiple NDP devices is a promising direction, where experts are partitioned across NDPs and expert computations are distributed accordingly. However, such an extension introduces additional full-system challenges beyond per-layer scheduling, including expert placement across memory devices, activation routing to the target NDP, inter-device load balancing, and coordination of communication under shared interconnect bandwidth. We leave these system-level extensions for future work.

<a id="p21"></a>
### [P21] Piper: Efficient Large-Scale MoE Training via Resource Modeling and Pipelined Hybrid Parallelism

**原 PDF**：[Piper_2605.05049.pdf](<%E6%80%A7%E8%83%BD%E6%A8%A1%E5%9E%8B%E4%B8%8E%E8%B5%84%E6%BA%90%E8%B0%83%E5%BA%A6/Piper_2605.05049.pdf>)

**章节状态**：本地版本未发现独立 Limitations / Future Work 章节，也未定位到作者明确自述尚未解决限制或具体未来研究计划的段落。不把结论中关于可扩展性的概述或对已有系统的批评当作本文 limitation。

<a id="p22"></a>
### [P22] MPMoE: Memory Efficient MoE for Pre-Trained Models With Adaptive Pipeline Parallelism

**原 PDF**：[MPMoE_TPDS_2024.pdf](<%E6%80%A7%E8%83%BD%E6%A8%A1%E5%9E%8B%E4%B8%8E%E8%B5%84%E6%BA%90%E8%B0%83%E5%BA%A6/MPMoE_TPDS_2024.pdf>)

**章节状态**：无独立 Limitations / Future Work 章节。以下为作者在评估中自述的模型预测、额外开销与扩展边界；未发现明确 future work 段。

**Limitation / scope / overhead（作者正文自述）** · `V-C Overall Speedup` · PDF 第 9 页

> On the Adira cluster, MPMoE-pm exhibits inferior performance compared to MPMoE-pb. This discrepancy can be attributed to the more obvious network fluctuations on the Adira cluster, which consequently degrade the prediction accuracy of the performance model. Conversely, MPMoE-pm demonstrates comparable performance on the V alor cluster.

**Limitation / scope / overhead（作者正文自述）** · `V-F Performance Breakdown and Overhead Analysis` · PDF 第 11 页

> Compared with native MoE, the overheads of MPMoE come from two aspects: 1) data partition cost when using pipelining, 2) the overheads of profiling for finding optimal granularity dynamically.

**Limitation / scope / overhead（作者正文自述）** · `V-G Scalability Analysis` · PDF 第 11 页

> However, network communication overhead inevitably reduces the practical scaling factor. The experiments demonstrate that MPMoE increases throughput by 5.76 × (72% of the ideal scaling) when scaling up to 8 nodes, while FasterMoE only achieves 5.4×.

<a id="p23"></a>
### [P23] Comet: Fine-grained Computation-communication Overlapping for Mixture-of-Experts

**原 PDF**：[COMET_MLSys_2025_2502.19811v3.pdf](<%E8%9E%8D%E5%90%88%E7%AE%97%E5%AD%90%E4%B8%8ETile%E7%BA%A7%E9%87%8D%E5%8F%A0/COMET_MLSys_2025_2502.19811v3.pdf>)

**章节状态**：未发现独立 Limitations / Future Work 章节或明确后续计划。以下仅为作者明确报告的 buffer 开销，不将本次综合归纳的 NPU 迁移问题写成作者原文。

**Limitation / scope / overhead（作者正文自述）** · `5.5 Overhead Analysis` · PDF 第 11 页

> Comet leverages NVSHMEM to allocate a shared memory buffer for communication on each device. The buffer size is dependent on the model configuration and equals to M N, where M is the input sequence length andN is the model hidden size. For datatype of BF16 or FP16, the allocated memory size is 2M N. The communication buffer is global for the execution of the entire model, which means that it is shared across layers and experts. We list the device memory consumption ofComet in Table 3, and it is negligible compared with the large device memory on current GPUs.

<a id="p24"></a>
### [P24] FlashMoE: Fast Distributed MoE in a Single Kernel

**原 PDF**：[FlashMoE_NeurIPS_2025_2506.04667v3.pdf](<%E8%9E%8D%E5%90%88%E7%AE%97%E5%AD%90%E4%B8%8ETile%E7%BA%A7%E9%87%8D%E5%8F%A0/FlashMoE_NeurIPS_2025_2506.04667v3.pdf>)

**章节状态**：§5 Limitations and Future Work 完整摘录，另附 §H 原文。

**Limitations and future work（作者原文）** · `5 Limitations and Future Work` · PDF 第 10 页

> Engineering complexity.Fully fused, persistent kernels demand deep GPU + distributed-systems expertise; future work may investigate compiler/DSL abstractions to lower this barrier. FP16 inefficiency.Our FP16 path is suboptimal (§H) due to insufficient tuning. We anticipate addressing this gap with autotuned GEMM operators like cuBLASDx [35] or CUTLASS builders. Training support.This work targets inference; enabling training will require fusing backward computation and gradient communication with new bookkeeping and task descriptors.

**Limitations and future work（作者原文）** · `H FP16 Memory Throughput` · PDF 第 24 页

> Notably, the FP16 implementation issues approximately 2× more shared memory instructions compared to its FP32 counterpart under identical workloads. We attribute this inefficiency to suboptimal shared memory layouts in FlashMoE when operating on half-precision data. While this bottleneck is addressable through improved layout strategies, we leave its resolution to future work.

<a id="p25"></a>
### [P25] Fine-grained Computation-Communication Overlap via Tile-level Signaling and Scheduling for Mixture-of-Experts

**原 PDF**：[Tile_level_signaling_ICPP_2026_2607.19539.pdf](<%E8%9E%8D%E5%90%88%E7%AE%97%E5%AD%90%E4%B8%8ETile%E7%BA%A7%E9%87%8D%E5%8F%A0/Tile_level_signaling_ICPP_2026_2607.19539.pdf>)

**章节状态**：没有独立 Limitations 章节；以下是结果中的负例与作者明确提出的 future work。

**Limitation / scope / overhead（作者正文自述）** · `4.2.1 End-to-End and MoE Layer Evaluation in MoE Models` · PDF 第 7 页

> For M-Trans-xl, our approach delivers a 2.64x speedup over FasterMoE, but slightly trails Tutel, MegatronCUTLASS, and Megatron-TE. This is likely because M-Trans-xl has the smallest hidden dimension (Dim.= 512) of the three models. Per-token all-to-all volume scales linearly with Dim., so a smaller Dim. leaves less communication to hide. At the same time, per-tile useful work is also reduced, so the fixed overhead of fine-grained overlap becomes a larger fraction of layer time.

**Future direction（作者原文）** · `4.3.4 Resource Contention` · PDF 第 8 页

> Production systems could benefit from an runtime-adaptive SM partition selector, which we leave to future work.

<a id="p26"></a>
### [P26] INC-MoE: Overlapping In-Network Collectives and Expert Computation for Expert-Parallel MoE Inference

**原 PDF**：[INC_MoE_2610.36339.pdf](<%E8%9E%8D%E5%90%88%E7%AE%97%E5%AD%90%E4%B8%8ETile%E7%BA%A7%E9%87%8D%E5%8F%A0/INC_MoE_2610.36339.pdf>)

**章节状态**：未发现独立 Limitations / Future Work 章节或明确 future-work 计划；以下为 §5.2 对 emulator 局限与真实硬件预期的自述，后者不是已完成硬件实验。

**Prototype limitation and hardware expectation（作者原文）** · `5.2 In-Network Collective Bandwidth` · PDF 第 14–15 页

> Unlike the ring, their equivalent bandwidth increases with the message size. This is an artifact of the prototype. Because the switch is emulated by an accelerator, every message incurs fixed costs that a hardware switch does not have. For theRS, the emulator serves theN rooted reductions of a message one after another, and each incurs a grid-wide synchronization between the emulator’s threads and the fill and drain of its chunk pipeline. Small messages amortize these costs over fewer chunks, which lowers the emulator’s efficiency. A hardware INC switch processes packets at line rate and can serve all roots concurrently, so we expect this dependence on message size, and most of the gap to the bound, to be largely reduced on switch hardware, as they are specific to the emulator rather than to the design.

<a id="p27"></a>
### [P27] TileLink: Generating Efficient Compute-Communication Overlapping Kernels using Tile-Centric Primitives

**原 PDF**：[TileLink_2503.20313.pdf](<%E8%9E%8D%E5%90%88%E7%AE%97%E5%AD%90%E4%B8%8ETile%E7%BA%A7%E9%87%8D%E5%8F%A0/TileLink_2503.20313.pdf>)

**章节状态**：§7.4 Discussion 两项扩展完整摘录；不把已有两节点 DP 评估误写为完全未测多节点。

**Future direction（作者原文）** · `7.4 Discussion — Support for model-level communication` · PDF 第 14 页

> Support for model-level communication: TileLink can be extended to support model-level parallelism (e.g., pipeline parallelism). To achieve this, we can integrate NVSHMEM functionalities intoTileLink’s tile_push_data primitive and follow the same compilation techniques asTileLink. We leave this for future work.

**Limitations and future work（作者原文）** · `7.4 Discussion — Support multiple backends` · PDF 第 14 页

> Support multiple backends: Currently, TileLink targets only NVIDIA GPUs. To support more hardware, we can extend the low-level compilers (e.g., TVM, which supports more hardware than Triton), while keeping the primitives and compilation techniques ofTileLink unchanged.

<a id="p28"></a>
### [P28] SwiftEP: Accelerating MoE Inference with Buffer Fusion and TMA Offloading

**原 PDF**：[SwiftEP_NSDI_2026.pdf](<%E4%B8%93%E5%AE%B6%E5%B9%B6%E8%A1%8C%E9%80%9A%E4%BF%A1%E4%B8%8E%E7%B3%BB%E7%BB%9F/SwiftEP_NSDI_2026.pdf>)

**章节状态**：§9 Discussion and Limitations 三个小节完整摘录；另收录 §10 关于研究范围的明确自述。

**Limitation / extension（作者原文）** · `9 Discussion and Limitations` · PDF 第 12–13 页

> Adaptation to MoE model training.In inter-server buffer fusion, a major bottleneck arises from the millisecond-scale latency of memory registration when tensor buffers are registered for RDMA after computation. To alleviate this overhead, we pre-register a fused buffer of the required size and modify the inference framework so that computation results are written directly into the pre-registered buffer. This design enables RDMA communication to start immediately and allows the same buffer to be reused across layers during inference. However, the training phase imposes a critical constraint: intermediate activations must be retained for backward propagation, which precludes straightforward buffer reuse. To address this, an alternating pre-registration strategy with multiple fused buffers is necessary. RNIC capability limits.WQE post coalescing introduces a subtle trade-off related to PCIe efficiency. In GPU-to-GPU communication, GPU threads post WQEs to the QP and ring the doorbell, triggering the RNIC DMA engine to parse each WQE and issue multiple PCIe read requests to fetch SGEs for transmission. Each PCIe read incurs link-layer header overhead [31]. Consequently, when a single WQE aggregates too many SGEs, the cumulative PCIe header overhead can outweigh the control-plane savings from WQE coalescing, leading to performance degradation. Empirically, this threshold typically lies in the hundreds of SGEs, such as on CX7 NICs [43]. Operating well within this range, SwiftEP achieves optimal performance without encountering this limitation. QP scalability.While QP parallel transmission increases the number of required QPs, this overhead remains manageable for MoE models. Expert parallelism is naturally bounded by the total number of experts, up to 384 in current open-source models [54]. Furthermore, since libraries like TCCL [32] and DeepEP optimize within-node communication, each RNIC only needs to communicate with at mostexperts/8remote peers. With 4 QPs per RNIC pair, even the largest MoE models utilize only 192 concurrent QPs, which is far below the CX7 NIC’s capacity of thousands [43], thereby avoiding QP context cache misses entirely.

**Limitation / scope / overhead（作者正文自述）** · `10 Related Work — Inter-node communication performance` · PDF 第 13 页

> In contrast, this work does not target inter-server communication or congestion control. SwiftEP is orthogonal to these efforts, focusing instead on intra-node communication, efficient memory management, and computation–communication overlap within GPU domains.

<a id="p29"></a>
### [P29] UEP: Portable Expert-Parallel Communication

**原 PDF**：[UEP_OSDI_2026.pdf](<%E4%B8%93%E5%AE%B6%E5%B9%B6%E8%A1%8C%E9%80%9A%E4%BF%A1%E4%B8%8E%E7%B3%BB%E7%BB%9F/UEP_OSDI_2026.pdf>)

**章节状态**：§6 Discussion and Future Work 全节摘录，包含作者对进一步优化、部署范围和故障语义的原文。

**Discussion and Future Work（作者原文）** · `6 Discussion and Future Work` · PDF 第 14–15 页

> Congestion control with CPU proxy.We observe that the number of outstanding requests can have a significant impact on various NICs, affecting tail latency [46, 66]. This becomes increasingly significant as the number of destinations increases, where having one straggler can significantly slow down dispatch and combine time. Software-based flow control is challenging to implement with IBGDA, as GPU threads typically are not flexible enough for implementing network-level policies. Instead, UEP delegatescontrol decisionsto the flexible CPU proxy, which could easily support request tracking and pacing. If the outstanding requests become high, the CPU proxy thread temporarily buffers the messages at the sender, so that the messages will not cause an incast at the receiver side. The CPU proxy also bears responsibility for multi-QP management: because each GPU may be connected to multiple NICs or multiple QPs, the proxy can throttle or shard the outgoing requests across NICs and QPs to avoid congestion [65], and adapt to NIC-specific characteristics (e.g., maximum outstanding write requests) without burdening the GPU kernel logic. This separation allows exploiting multiple NICs transparently and optimizing throughput while still keeping the GPU’s logic simple. Elastic EP with CPU proxy.NVSHMEM, including IBGDA, CPU-assisted IBGDA, and IBRC, does not expose error handling to GPU kernels and assumes RDMA operations always succeed. However, events such as GPU failure, scale up, and down can happen in practical scenarios, where NVSHMEM would need to restart the existing cluster. UEP could achieve elastic EP by using the flexible CPU proxy to handle failure and scaling events, transparently masking them from the GPU kernel. More efficient low-latency kernel.DeepEP’s low-latency kernel (without token deduplication or hierarchical reduce) could be further optimized. This would particularly benefit AWS EFA NICs that cannot process small message transfers efficiently (see Figure 15). UEP currently extends DeepEP LL kernels, which we plan to further improve as future work. We deem these optimizations as orthogonal to UEP’s contribution in a portable expert-parallel communication architecture by decoupling NICs and GPUs. Intended deployment mode for UEP.UEP is primarily designed for platforms where IBGDA is unavailable or impractical, such as non-NVIDIA GPUs or NICs without the required GPU-initiated RDMA support. However, UEP can also run on IBGDA-capable platforms, where the CPU proxy can provide additional flexibility for communication management. Failure semantics.In practice, the CPU and GPU on the same server often fate-share: CPU processes are already responsible for launching GPU kernels, and widely used systems such as NCCL also rely on host CPU threads to initiate and coordinate communication. Thus, UEP does not introduce a separate failure domain.

<a id="p30"></a>
### [P30] Lancet: Accelerating Mixture-of-Experts Training via Whole Graph Computation-Communication Overlapping

**原 PDF**：[Lancet_MLSys_2024.pdf](<%E4%B8%93%E5%AE%B6%E5%B9%B6%E8%A1%8C%E9%80%9A%E4%BF%A1%E4%B8%8E%E7%B3%BB%E7%BB%9F/Lancet_MLSys_2024.pdf>)

**章节状态**：§8 Discussion and Related Works 中与兼容性、未来通信编排、不规则计算有关的原文；不将正交工作的介绍作为限制。

**Limitations and future work（作者原文）** · `8 Discussion and Related Works — Compatibility with other large-scale training techniques` · PDF 第 11 页

> While Lancet is evaluated with data and expert parallelism, the techniques are in principle compatible with most other commonly used training optimizations. Weight gradient scheduling only utilizes operator dependency during backward propagation, thus unaffected by most distributed training sharding techniques. Some techniques introduce extra communication which may interfere with partition-based all-to-all overlapping. FSDP/ZeRO3 (Rajbhandari et al., 2020) inserts additional all-gather communication in the forward passes, which may require additional scheduling to avoid interference with overlapped all-to-all. Tensor parallelism (Shoeybi et al., 2019) requires all-reduce communication after self-attention; Ring-attention (sequence parallelism) (Liu et al., 2023) communicates the key-value blocks during the attention process. If different devices or communication channels are used for expert and tensor/sequence parallelism (e.g., inter-node vs. intra-node), the overlapped all-to-all communication can be arranged to execute concurrently with tensor/sequence parallelism traffic. Investigating the efficient orchestration and overlapping of communication arising from various sharding techniques, particularly the intricate patterns generated by automatic sharding (Zheng et al., 2022), remains future work.

**Limitation / extension（作者原文）** · `8 Discussion and Related Works — Optimizing irregular communication and expert computation` · PDF 第 11 页

> Optimizing irregular communication and expert computation Lancet’s partition produces irregular-shaped allto-alls and expert computation. While we use a simple NCCL based implementation (Fig. 10), better communication implementations targeting such dynamic workload may further improve the performance. Similarly, the shape irregularity in expert computation may cause extra computation due to padding. Block-sparse expert kernels (e.g., MegaBlocks (Gale et al., 2023)) can be further applied to accelerate the computation.

<a id="p31"></a>
### [P31] GCoM: A Detailed GPU Core Model for Accurate Analytical Modeling of Modern GPUs

**原 PDF**：[Lee 等 - 2022 - GCoM a detailed GPU core model for accurate analytical modeling of modern GPUs.pdf](<NPU_GPU%E6%80%A7%E8%83%BD%E5%BB%BA%E6%A8%A1%E4%B8%8E%E6%9E%B6%E6%9E%84/Lee%20%E7%AD%89%20-%202022%20-%20GCoM%20a%20detailed%20GPU%20core%20model%20for%20accurate%20analytical%20modeling%20of%20modern%20GPUs.pdf>)

**章节状态**：未发现以作者自身方法为主题的独立 Limitations 章节；第 2.3 节主要列出现有模型需要解决的问题，未当作 GCoM 自身限制。

**Future direction（作者原文）** · `5.2 Extending GCoM to Model Other GPUs` · PDF 第 11 页

> More accurate modeling of the instructions would require SM-wide synchronization and functional state modeling (e.g., track the value of the SM-wide VM_CNT counter), which we leave as future work.

**Limitation / scope / overhead（作者正文自述）** · `5.2 Extending GCoM to Model Other GPUs` · PDF 第 11 页

> In this work, we mainly focused on modeling modern NVIDIA GPUs with GCoM; however, one may seek to employ GCoM for modeling GPUs from other vendors which have different core microarchitectures.
