# Methodology

> A published copy of `docs/METHODOLOGY.md` from MachLume Studio's source, which is private: the file names in it refer to that source. Published with MachLume Studio 0.3.0.

How every measured number in MachLume Studio is produced, what can go wrong, and what was done about it.
The code is in `src/bench/` (Measure) and `src/engine/` (Verify).

## 1. Measure: microbenchmarks

All kernels are WGSL compute shaders in `kernels.ts`.

| # | Benchmark | What it measures | Work counted |
|---|---|---|---|
| 1 | **FMA peak** (f32, and f16 if `shader-f16` exists) | Register-only multiply-add chains: 8 independent `vec4` chains, unrolled 4× per loop iteration, 131,072 threads, 256 iterations | 2 FLOP per FMA, 256 FLOP per thread-iteration |
| 2 | **Tiled GEMM** 256³ to 2048³ (f32, f16) | Register-blocked matmul: 128×128 output tile per workgroup, 8×8 block per thread in 16 `vec4` accumulators, A/B tiles staged in workgroup memory | 2·n³ FLOP; best size = achievable matmul ceiling |
| 3 | **Stream read / copy** | `vec4<f32>` grid-stride sum (read) and copy over a 512 MiB (desktop) / 256 MiB (phone) working set | read: buffer bytes; copy: 2× (read + write) |
| 4 | **GEMV** f32, f16, and **4-bit** | y = W·x with 4096 columns, rows filling the same working set; 4-bit uses the MLC q4 layout (8 nibbles per u32, offset 7, one scale per 32), unpacked in the kernel | matrix bytes + x + y |
| 5 | **Dispatch overhead** and **readback round trip** | N ∈ {1, 64, 256, 1024} tiny kernels per submit. Each has a small uniform upload (`writeBuffer`), a new bind group and its own compute pass: the pattern WebLLM's TVM runtime uses, confirmed by the Verify call counts (one `writeBuffer` per dispatch). Separately: one kernel + copy + `mapAsync`, measured on the main thread and again inside a Web Worker, where the engine runs | slope (µs/dispatch) and round-trip times |

### Hygiene rules and how each is implemented

- **Dead-code elimination.** Every kernel writes a result that depends on all the data it read: FMA
  writes the sum of its chains, read writes per-thread partial sums, GEMV writes y. The FMA multiplier
  and addend come from a uniform buffer, so the compiler cannot constant-fold the chains.
- **Data is not trivially compressible.** Buffers are filled on the GPU with a hash of the index:
  floats in −0.5, 0.5) and random bits for packed 4-bit weights. Values are never zero or constant, and
  there is no large CPU-to-GPU upload.
- **Kernel self-checks.** Before timing, a 128×128 GEMM (f32 and f16) and 64-row GEMVs (f32, f16,
  4-bit) are read back and compared with a CPU reference. The run aborts if the relative error exceeds
  10⁻⁴ (f32 GEMM), 2·10⁻² (f16 GEMM), or 10⁻³ (GEMV). A fast but wrong kernel cannot produce a ceiling.
- **Timing source.**
  - When the adapter exposes `timestamp-query`, the device is created with it and each sample's compute
    pass is bracketed with `timestampWrites`. Chrome quantizes these timestamps to 100 µs.
  - Otherwise the sample is timed on the wall clock from `submit` to `onSubmittedWorkDone()`.
  - If timestamps come back zero or larger than the wall time, the harness switches to the wall clock
    for the rest of the run.
  - The source is recorded per result and shown in the table.
- **Sample size versus timer resolution.** Each sample is auto-sized by repeating the dispatch inside
  one pass until it lasts at least 20 ms with timestamps, or 50 ms on the wall clock (up to 4
  calibration runs, capped at 2000 dispatches). At 100 µs resolution the quantization error is then
  below 0.5%.
- **Warm-up.** The first (calibration) runs and one explicit warm-up sample are discarded.
- **Repetitions.** K = 7 by default (configurable 3–21). Reported: median, min, max, coefficient of
  variation (sample std / mean), and n.
- **Working set larger than any cache.** The working set is 512 MiB on desktop and 256 MiB on phones,
  split across several storage bindings if `maxStorageBufferBindingSize` is smaller. Phone system-level
  caches are at most tens of MB, but some desktop GPUs have 64–128 MB of last-level cache, so the
  desktop working set is only about 4× those. That limitation is stated here.
- **Device limits.** The device is requested with the adapter's own `maxBufferSize`,
  `maxStorageBufferBindingSize`, and `maxComputeWorkgroupStorageSize`; the WebGPU defaults would cap
  the bandwidth buffers. GEMM sizes that exceed a binding are marked "unavailable".
- **Thermal check.** The FMA f32 benchmark runs first and again last, with the same dispatch count. A
  drop of more than 10% shows a throttling warning, and both numbers are kept.
- **Time budget and cancel.** The default budget is 45 s. Optional items (GEMM 2048³ after a slow
  1024³, stream copy, the f16 and 4-bit GEMVs) are skipped once the budget is exceeded and are marked
  "skipped". Cancel checks between samples; a cancelled run is not saved.
- **Failures.**
  - `device.lost` is checked before and after each sample.
  - Buffers are created inside `out-of-memory` and `validation` error scopes, and pipelines inside a
    `validation` scope.
  - Each failure becomes a readable message (device lost, out of memory, shader rejected).
  - Uncaptured GPU errors are collected and shown.
- **Dispatch overhead without the sync pitfall.** Timing a single dispatch measures dispatch plus
  synchronization (arXiv 2608.08730). Instead, the per-dispatch cost is the **slope** of a least-squares
  fit of wall time against N. The **intercept** is the fixed submit-and-wait cost, which is reported
  separately as the readback round trip.
- **Fresh adapter per run.** Chrome 155 marks an adapter as "consumed" after one `requestDevice`. A
  second run on the same adapter failed during testing, so every run requests a new adapter.
- **Screen wake lock** is requested during a run, where supported.

## 2. Results on the development machine

| Item | Value |
|---|---|
| Hardware | Intel Core Ultra 5 225H with the integrated **Intel Arc 130T** (reported as `intel` / `xe-lpg`); 2× 8 GB DDR5-5600 SO-DIMM (dual channel, 128-bit) |
| Browser | Chrome 155.0.8059.85, headless, driven by [`scripts/bench-desktop.mjs` |
| Timing | `timestamp-query`, K = 7, 20 s cool-down between runs |
| Raw data | `docs/evidence/measure-arc130t-chrome155.json` |

| Ceiling | Run 1 / 2 / 3 (median of K = 7) | Within-run CV | Published reference | Measured / reference |
|---|---|---|---|---|
| FMA peak f32 | 3808 / 3808 / 3809 GFLOP/s | 0.3–1.1% | 7 Xe-cores × 128 FP32 lanes × 2 × 2.2 GHz ≈ 3942 GFLOP/s ¹ | 97% |
| FMA peak f16 | 6950 / 6845 / 6942 GFLOP/s | 0.7–1.2% | 2× f32 ≈ 7885 GFLOP/s ¹ | 87–88% |
| GEMM f32 (2048³) | 2457 / 2424 / 2404 GFLOP/s | 0.3–0.5% | – | 63–65% of measured FMA peak |
| GEMM f16 (2048³) | 2993 / 3038 / 2960 GFLOP/s | 2.0–2.5% | – | 43–44% of measured FMA f16 peak |
| Stream read | 62.9 / 63.1 / 62.7 GB/s | 1.5–2.7% | DDR5-5600 × 16 B = 89.6 GB/s ² | 70% |
| Stream copy | 56.3 / 55.1 / 55.7 GB/s | 1.2–1.9% | 89.6 GB/s | 62% |
| GEMV f32 | 69.8 / 70.3 / 71.6 GB/s | 2.2–2.9% | 89.6 GB/s | 78–80% |
| GEMV f16 | 68.6 / 70.2 / 70.0 GB/s | 0.9–3.7% | 89.6 GB/s | 77–78% |
| GEMV 4-bit | 71.7 / 73.4 / 73.6 GB/s (packed bytes) | 0.9–2.3% | 89.6 GB/s | 80–82% |
| Per-dispatch overhead | 3.47 / 2.91 / 3.85 µs ³ | (fit) | – | – |
| Readback round trip | 2900 / 3000 / 3000 µs | 8.8–15% | – | – |
| Thermal drop (FMA start→end) | 2.8% / 3.8% / 2.8% | – | – | below the 10% warning |

¹ The Xe-core count (7) and the 2.2 GHz maximum graphics clock for the Arc 130T in the 225H come from
third-party spec listings. Intel's own ARK page for the 225H did not load in my session, so treat
these references as unverified. Xe-LPG has 16 vector engines per Xe-core, each 8 FP32 lanes wide; fp16
runs at twice the FP32 rate. Any matrix engines are not reachable from standard WGSL and are not
measured.

² Memory type, speed, and module count were read from Windows (`Win32_PhysicalMemory`: DDR5, 5600 MT/s, 2 × 64-bit).

³ These three runs used the first version of the dispatch benchmark, which did not upload a uniform per dispatch.
After the Verify call counts showed that WebLLM issues one `writeBuffer` per dispatch, the benchmark was changed to
match. The current version measures **7.61 µs** per dispatch on the same laptop (the profile in
`verify-arc130t-chrome155.json`). The same profile measures the readback
round trip inside a Web Worker at 3.0 ms, identical to the main thread.

### What these numbers say, and the discrepancies

- **Compute.** FMA f32 reaches 97% of the theoretical peak, which suggests the kernel is
  issue-limited, as intended. f16 reaches 87%.
- **GEMM gap.** My GEMM kernel reaches 63–65% of FMA peak in f32 and only 43% in f16. The f16 gap is
  a property of **my** kernel: the same tiling with fp16 accumulators, no subgroup or matrix
  instructions. A tuned engine kernel could beat this "achievable" ceiling, so a prefill measurement
  above the GEMM roof would not be a contradiction. My first naive GEMM reached only 178 GFLOP/s (4.7%
  of peak). The 14× improvement came from register blocking, and it shows how much an "achievable
  ceiling" depends on the kernel that measures it.
- **Bandwidth.** GEMV reaches 78–82% of the DDR5 theoretical bandwidth, and streaming read only 70%.
  The read kernel is a grid-stride loop with one `vec4` load per iteration per thread. The GEMV kernels
  issue more independent loads per thread, which hides latency better on this iGPU. The model therefore
  uses the GEMV bandwidth for decode weight streaming and the read bandwidth for everything else
  (MODEL.md §5). The iGPU shares DRAM with the CPU, so background CPU activity lowers these numbers;
  that is the main source of the 1–4% CVs.
- **4-bit GEMV versus f16 GEMV.** 4-bit is as fast or faster per packed byte, so unpacking is hidden
  behind memory latency on this GPU. The first version of the 4-bit kernel reached only 49 GB/s because
  every thread reloaded 8 activation vectors per 16 bytes of weights (cache-bound). Reusing each
  activation load across 4 rows fixed it. MODEL.md therefore does not charge decode a separate
  dequantization term.
- **Sync latency.** The readback round trip is about 3 ms, with timer resolution 100 µs here. For a
  memory-bound 0.5B model a token takes about 4 ms of roofline time, so this overhead is of the same
  order. It is a property of the browser's WebGPU event loop, not of the GPU. The value varies more
  between runs (CV up to 15%) than the throughput numbers do.
- **Thermals.** The first in-app run (Electron/Chrome 152 pane) saw a 12% FMA drop by the end and was
  flagged. The three headless Chrome runs with cool-downs dropped 3–4%. Both outcomes are kept as
  measured.

## 3. Verify: trial protocol

The code is in `src/engine/protocol.ts`, the worker in
`src/engine/worker.ts`.

**Engine.** WebLLM 0.2.86 runs in a dedicated Web Worker. The Verify engine sits behind an `InferenceEngine`
interface, so it could be swapped. Weights are cached by the engine: the Cache API, or IndexedDB when the
browser refuses Cache Storage. This happened in a headless test profile, and the app now probes for it.
The Verify page lists cached models and deletes them on request.

**Before downloading:** the app shows the download size (read from the build's `tensor-cache.json`) and
the predicted GPU memory. It asks for confirmation when the connection is cellular or unknown. It refuses
builds that need `shader-f16` on devices without it, and builds whose largest tensor exceeds
`maxStorageBufferBindingSize`.

**Per prompt length (≈32, ≈128, ≈512 tokens):**

1. **Warm-up.** One generation of G tokens, discarded. It absorbs shader compilation; the first version
   of the protocol counted this generation and got a negative per-token wait time, which is why it now
   comes first.
2. **Prefill alone.** One generation of 1 token. Its WebGPU call counts and wait time are the cost of one
   prefill plus first-token sampling.
3. **Measured trials** (default 3). Each trial generates exactly G = 64 tokens with `ignore_eos`:
   - The decoding is greedy: temperature 0 and top-p 1.
   - Penalties are neutral, so the sampler does no extra work.
   - The chat is reset before each trial, so no KV-cache prefix is reused.
4. **Kernel profile.** A separate G-token generation and a separate 1-token generation with timestamp
   queries on every compute pass. These are never used as timing trials.

**What is reported:**
- **TTFT and decode speed:** from the engine's own counters (`usage.extra.time_to_first_token_s` and
  `decode_tokens_per_s`), as the median of the trials with min–max.
- **Model load time:** reported separately and never included in TTFT.
- **Prompt length:** the prompt is fixed text per bucket; the engine's `prompt_tokens`, which includes
  the chat template, is what is reported and used for the prediction.
- **Prediction:** the pure-roofline prediction is computed from the active profile's ceilings at that
  exact token count and frozen into the run record, so later model changes cannot rewrite history.

**Call instrumentation (worker).** Before the engine loads, the worker wraps
`GPUComputePassEncoder.dispatchWorkgroups`, `GPUQueue.submit`, `GPUQueue.writeBuffer`,
`GPUBuffer.mapAsync` and `GPUQueue.onSubmittedWorkDone`. The wrappers count calls and sum the time the
engine spends blocked on the two promise APIs. Per decoded token: (trial − prefill-only) / (G − 1),
averaged over trials. That splits each token into **waiting for the GPU** and **CPU work**.

**Kernel profiler (worker).** WebLLM's TVM runtime names every compute pipeline after its kernel (the
entry point is the TVM function name).
- **How it works.** During a profiled generation the worker adds `timestamp-query` to the engine's device
  and gives every compute pass `timestampWrites`. It resolves the queries when each command buffer is
  finished, then sums durations per kernel name.
- **Per decoded token:** (G-token profile − 1-token profile) / (G − 1).
- **Categories:** matmul, attention, sampling, norm, RoPE / KV append, elementwise, other. They are
  assigned by name pattern in `kernels.ts`.
- **Precision.** Chrome quantizes timestamps to 100 µs, so a single short kernel reads as 0 or 100 µs.
  Summed over thousands of passes, the totals are an unbiased estimate (assumption A-17).
- **Cost.** The profiler adds timestamp writes and one readback per command buffer, which is why profiled
  generations are never timing trials.

**Thermal evidence.** A GEMV bandwidth probe (128 MiB, K = 5) runs before the model loads and after it
is unloaded. The trial-to-trial drift in decode speed is also recorded.

**Tab killed mid-run.** An "in flight" marker is written before loading and cleared after. If it is still
there on the next page load, the Verify page says the last run never finished and that the cause is
probably memory.

## 4. Error analysis

For each run (`src/model/errors.ts`):

- **Error definition:** (predicted − measured) / measured, for TTFT seconds and for decode tokens/s.
- **Calibrated comparison:** shown once 3 runs exist on the profile. It comes from a two-parameter fit
  (α, τ) graded leave-one-run-out (MODEL.md §7).
- **Time split per decoded token:** measured time = GPU kernel time (profiled) + waiting while the GPU
  is idle + CPU time. This is set against the model's roofline time plus its assumed overhead.

Every candidate cause is a hypothesis with a verdict (supported / not supported / not enough evidence)
and the evidence that decided it:

| Hypothesis | Evidence used | Supported when |
|---|---|---|
| Engine matrix-vector kernels slower than my GEMV benchmark | profiled matmul time vs weight bytes | effective bandwidth < 80% of the ceiling |
| Decode attention far below the bandwidth roofline | profiled attention time vs KV bytes read | > 2× the model's attention time and > 5% of the token |
| Fixed cost of many small kernels | profiled norm / RoPE / elementwise time | > 10% of the token |
| Sampling work (softmax + full-vocabulary argsort) | profiled sampling kernels | > 15% of the token |
| Round-trip latency with the GPU idle | wait time − GPU busy time | > 15% of the token |
| CPU work per token | token time − wait time | > 20% of the token |
| Prefill matmuls below the matmul ceiling | profiled prefill matmul time vs FLOPs | < 50% of the ceiling, prompt compute-bound |
| Dequantization cost | 4-bit vs f16 GEMV benchmark | 4-bit < 90% of f16 |
| Cost growing with context | runs of the same model at different lengths | growth > 2× the model's KV-read growth |
| Thermal throttling | bandwidth probe before/after, trial drift | probe drop > 10% or drift < −15% |
| Memory pressure | not observable from a web page | always "not enough evidence" |

### Results on the development laptop (Intel Arc 130T, Chrome 155, headless)

Raw data: `docs/evidence/verify-arc130t-chrome155.json`.
These are 9 runs: 3 models × 3 prompt lengths, G = 64, 3 measured trials each, weights already cached.
"Pred" is the pure roofline prediction made before the trials. Per-token columns are measured.

| Build | Prompt tokens | TTFT meas / pred | Decode meas / pred (tok/s) | Per token | GPU kernels | Matmul | Attention | Sampling | GPU-wait | Dispatches |
|---|---|---|---|---|---|---|---|---|---|---|
| SmolLM2-135M q0f16 | 53 | 59 / 10.6 ms | 61.5 / 107.8 | 16.3 ms | 9.25 ms | 6.98 | 0.90 | 0.41 | 14.1 ms | 324 |
| SmolLM2-135M q0f16 | 129 | 114 / 16.6 ms | 58.9 / 107.5 | 17.0 ms | 9.37 ms | 6.71 | 1.33 | 0.38 | 14.7 ms | 324 |
| SmolLM2-135M q0f16 | 436 | 343 / 42.6 ms | 50.2 / 106.2 | 19.9 ms | 11.19 ms | 6.62 | 3.24 | 0.37 | 17.1 ms | 324 |
| Qwen2.5-0.5B q4f16_1 | 41 | 115 / 16.8 ms | 57.0 / 113.7 | 17.5 ms | 10.64 ms | 7.85 | 1.08 | 0.78 | 15.1 ms | 268 |
| Qwen2.5-0.5B q4f16_1 | 119 | 219 / 36.7 ms | 49.8 / 113.5 | 20.1 ms | 11.52 ms | 7.88 | 1.93 | 0.77 | 17.6 ms | 268 |
| Qwen2.5-0.5B q4f16_1 | 427 | 684 / 117.8 ms | 38.9 / 112.8 | 25.7 ms | 15.21 ms | 8.14 | 5.35 | 0.77 | 21.8 ms | 268 |
| SmolLM2-360M q4f16_1 | 53 | 116 / 18.2 ms | 55.8 / 119.1 | 17.9 ms | 10.13 ms | 6.58 | 1.76 | 0.41 | 15.5 ms | 344 |
| SmolLM2-360M q4f16_1 | 129 | 257 / 35.9 ms | 50.5 / 118.4 | 19.8 ms | 10.95 ms | 6.55 | 2.87 | 0.39 | 17.7 ms | 344 |
| SmolLM2-360M q4f16_1 | 436 | 686 / 108.6 ms | 35.3 / 115.7 | 28.3 ms | 15.58 ms | 6.49 | 7.50 | 0.38 | 23.3 ms | 344 |

Matmul, attention, and sampling columns are ms per decoded token. Every build made 6 queue submits per
token and about 15 more `writeBuffer` calls than dispatches.

**What this says:**

1. **The pure model is far too optimistic on this laptop.**
   - Decode is predicted 1.8–3.3× too fast and TTFT 5.6–8× too fast.
   - The pure model's dispatch assumption (L × 10 + 8 = 328 for SmolLM2-360M) was close to the counted
     344. The overhead *structure* is roughly right; the costs inside it are not.
2. **Decode is not bandwidth-bound in this engine on this GPU.** All three models decode at 55–62 tok/s at
   short prompts, though their weights differ in size (204–278 MB). Per token:
   - **GPU kernels: about 9–11 ms.** The roofline time is 3–4 ms.
   - **Matmuls: 6.5–8 ms.** That is 31–41 GB/s effective, roughly half of the 70 GB/s my GEMV kernel
     reaches.
   - **Waiting with the GPU idle: about 5 ms.**
   - **CPU: about 2–2.5 ms.**
3. **Attention grows with context far faster than its memory traffic.**
   - For SmolLM2-360M, attention rises from 1.8 ms to 7.5 ms per token between 53 and 436 prompt tokens,
     about 15 µs per extra token of context.
   - Reading the KV cache at the measured bandwidth would cost about 0.6 µs per token.
   - The paged-KV decode kernel is latency-bound at batch 1: it has only a few KV heads to parallelize over.
4. **Sampling is not the problem.** The full-vocabulary argsort that the model ignores costs 0.4 ms per
   token (0.8 ms with Qwen's 151,936-entry vocabulary): about 2–4% of the token.
5. **Prefill runs far below the matmul ceiling.** TTFT grows about 1.5 ms per prompt token for
   SmolLM2-360M, and TTFT is almost entirely GPU kernel time (profiled prefill GPU time is 88–99% of
   TTFT), so prefill is kernel-bound, not overhead-bound. The profiled prefill matmuls reach 216–648
   GFLOP/s: 7–21% of the 3.1 TFLOP/s my GEMM kernel reaches, rising with prompt length.
6. **Calibration helps, but two parameters are not enough.**
   - **Fit:** α = 5.95, with τ clamped to 0.
   - **Leave-one-out median error:** 35% for TTFT and 22% for decode, against −80% to −90% and +75% to
     +227% for the pure model.
   - **What remains:** structured residuals. One efficiency factor cannot describe both the fixed
     per-token cost and the context-dependent attention cost.
7. **No throttling.** The bandwidth probe read 63–69 GB/s before each model's runs and 70–72 GB/s after:
   no drop. No trial-to-trial drift exceeded the threshold.

## 5. The CPU backend

A PC whose window has no usable WebGPU adapter is measured on its CPU instead. A PC with WebGPU can
measure its CPU too, from the Measure screen, natively or through WebAssembly.

### 5.1 Measurement

The kernels are hand-written WebAssembly text (`src/cpu/kernels.wat`),
compiled to a 1.6 KB binary by `scripts/build-cpu-wasm.mjs`. CI fails if
the committed binary does not match the source. The inner loops copy what llama.cpp's WebAssembly build
(ggml-cpu, `-msimd128`, no relaxed SIMD) runs for the same work:

| Kernel | Mirrors | Used for |
|---|---|---|
| `fma_f32`: 8 independent f32x4 chains, mul then add | the classic peak | SIMD peak (reference only) |
| `gemv_f32`: 4 accumulators, 16 floats per step | `ggml_vec_dot_f32` | f32 dot rate (attention and elementwise work) |
| `gemv_q8`: i8 → i16 extend, `i32x4.dot_i16x8_s`, fp16 scales through a 65 536-entry table | `ggml_vec_dot_q8_0_q8_0` | 8-bit weights |
| `gemv_q4`: nibble unpack, subtract 8, then as q8 | `ggml_vec_dot_q4_0_q8_0` | 4-bit weights |
| `sum_f32`: 4 accumulators of v128 loads | | stream read |

Each matrix kernel runs at two sizes:
- **In cache** (16 rows of 512–1024 columns, 9–34 KB): the rate a matmul reaches when its weights are
  reused across prompt tokens. This is the prefill compute roof, `dotQ4` / `dotQ8` / `gemmF32`.
- **From memory** (rows filling 512 MiB on desktops, 128 MiB on phones, split across threads): the
  decode roof, `bwGemvQ4` / `bwGemvQ8` / `bwGemvF32`, in bytes of packed weights per second.

**Hygiene, as on the GPU:**
- Every matrix kernel is checked against a JavaScript reference on the same data before it is timed.
  The unit tests run the same checks in Node.
- Data is pseudo-random with valid fp16 scales and no denormals.
- Every kernel returns a value that depends on all of its work.
- The first call (compilation tiers, page faults) and one warm-up are discarded.
- Samples are sized to at least 50 ms, because `performance.now()` is coarsened to 100 µs (5 µs when
  isolated).
- K samples are taken, with median, spread and CV.
- The SIMD peak is measured again at the end to catch throttling.
- A sample taken while the tab was hidden aborts the run, because browsers throttle background tabs.
  The same guard now protects the GPU suite and Verify.

**Threads.** The engine (wllama) uses ⌊logical cores / 2⌋ threads when the page is cross-origin
isolated (WebAssembly threads need `SharedArrayBuffer`), otherwise one. The benchmarks run on exactly that
many Web Workers:
- Each worker owns its own memory and instance, so no shared memory is needed.
- Every sample starts on all workers at once.
- Throughput = total work / the slowest worker's time, which is conservative.

The window sends COOP `same-origin` and COEP `require-corp` so the engine can use threads.

### 5.2 Verify on the CPU

The engine is llama.cpp compiled to WebAssembly (wllama 3.8.1), with `n_gpu_layers = 0` so it never
touches WebGPU. Model files are GGUF builds from the official Qwen and ggml-org repositories, or else
from bartowski's conversions made with llama.cpp's own quantizer. The facts of each file are read from
its header (tensor types and bytes per tensor) by
`scripts/fetch-gguf-facts.mjs`. The build fails if a file's
architecture differs from the official config.

The protocol is the GPU one (§3) with three differences:
- **No prefill-only generation.** It exists to separate WebGPU call counts.
- **No call counts or kernel profile.**
- **A 4-bit GEMV probe** before and after the run, instead of the GPU probe.

Timings come from llama.cpp's own timers:
- **TTFT** = `prompt_ms`: prompt evaluation until the first token is sampled.
- **Decode speed** = (`predicted_n` − 1) / `predicted_ms`. That is the same definition as WebLLM's
  `decode_tokens_per_s`.

`cache_prompt` is off, so no trial reuses an earlier trial's KV cache; a trial that reports reused tokens
is rejected. Weights are cached in the origin-private file system. Where that storage is unusable (some
private modes, the Windows WebKit build), the file is downloaded into memory instead and not cached.

**Two builds of the engine.** The default build needs JSPI and 64-bit WebAssembly memory. Browsers without
them, mainly Safari, get the compatibility build: Asyncify and 32-bit memory. Both ship with the app
byte for byte as published; the build checks that.

### 5.3 Results on the development laptop

Raw data: `measure-cpu-chrome.json`,
`measure-cpu-webkit.json`,
`verify-cpu-chrome.json`.

| Ceiling (3 runs) | Chrome 155, 7 threads | WebKit 27.2, 4 threads |
|---|---|---|
| SIMD peak f32 | 306–309 GFLOP/s | 175–177 GFLOP/s |
| In-cache dot f32 / q8 / q4 | 212–241 / 147–173 / 119–142 GFLOP/s | 139–143 / 129–138 / 96–100 GFLOP/s |
| Stream read | 53–60 GB/s | 49–50 GB/s |
| GEMV f32 / q8 / q4 from memory | 51–57 / 47 / 35–36 GB/s | 42–45 / 41–42 / 22–23 GB/s |
| Run time, thermal drop | 8.8–9.2 s, under 2% | 8.8–9.2 s, under 3% |

**4-bit decode is arithmetic-bound on this CPU.** The 4-bit GEMV from memory streams 35 GB/s, below the
8-bit GEMV's 47 GB/s. At 3.56 FLOP/byte, 35 GB/s is about 125 GFLOP/s, inside the range of the in-cache
4-bit dot rate (119–142). The nibble unpacking, not DRAM, limits it.

| CPU Verify (llama.cpp, 7 threads) | SmolLM2-135M Q4_0 | SmolLM2-360M Q4_0 | Qwen2.5-0.5B Q4_0 |
|---|---|---|---|
| Decode, measured vs pure prediction (P ≈ 53 / 129 / 436) | 81.5 / 70.3 / 53.0 vs 408 / 403 / 382 tok/s | 33.5 / 23.9 / 18.8 vs 159 / 157 / 152 | 28.8 / 25.7 / 18.9 vs 110 / 110 / 109 |
| TTFT, measured vs pure prediction | 434 / 1016 / 3352 vs 90 / 219 / 760 ms | 1127 / 2904 / 10683 vs 263 / 642 / 2207 | 1206 / 3203 / 11979 vs 289 / 730 / 2492 |

The pure CPU model is optimistic everywhere: decode by +284% to +708%, TTFT by 76–79%. The calibrated
model (α = 4.50, τ = 3.2 ms, 9 runs) predicts each run from the other eight with a median error of 6% for
TTFT and 24% for decode.

**What explains the gap:**
1. **The engine reaches about a fifth of my kernels' rate.**
   - Prefill matmuls run at 26–31 GFLOP/s, 20–24% of the in-cache 4-bit dot rate, for every model and
     prompt length.
   - That constant factor is why one calibration parameter fits TTFT to 6%.
   - At the shortest prompts, decode reaches 17–22% of the dot rate (21–28% of the GEMV ceiling), and less
     as the context grows (finding 4).
2. **Thread synchronisation is not it.** A separate run of SmolLM2-135M at 1, 2, 4, 7 and 12 threads
   decoded at 17.5, 33, 53, 72 and 44 tok/s.
   - Scaling is near-linear up to the engine's default of 7, then drops at 12 (this CPU has 4 performance,
     8 efficiency and 2 low-power cores; why 12 is slower was not measured).
   - The per-thread rate is about 4.7 GFLOP/s, against about 17 for my kernel.
3. **64-bit memory explains only a little.** Run on one thread, the engine's default build (64-bit
   memory, so every access is bounds-checked) decoded at 17.7 tok/s; its compatibility build (32-bit
   memory) at 20.8. That is about 15%. On 7 threads the default build is faster (78 against 59 tok/s),
   so the app keeps wllama's own choice.
4. **Attention grows far faster than its memory traffic.** Time per token grows 17–61 µs per extra context
   token against 0.4–0.8 µs predicted. llama.cpp keeps the KV cache in fp16 and, on WebAssembly, converts
   values to f32 through a lookup table; that is the likely cause, not measured separately here.

The app reports finding 1 and the context growth per run, with the numbers. It does not claim a cause it
did not measure.

## 6. Screen tests

The Playwright tests (`test/e2e/`) run the production build with the app's own policy
headers in Chromium, the engine inside the app's WebView2 window, at a wide and at the narrowest window.
The native side is replaced by recorded hardware data. They check:
- every screen loads without console errors;
- no screen scrolls sideways at the narrowest window;
- the window is cross-origin isolated;
- a real CPU measurement completes with every kernel check passing, and feeds Predict and Recommend;
- a corrupted saved profile cannot take the whole app down.

**What the tests found:**
- **Vite inlined the 1.6 KB kernel binary as a `data:` URL,** which `connect-src` blocks. WebAssembly is
  never inlined now.
- **The policy needs inline style attributes** (`style-src-attr 'unsafe-inline'`) and nothing else inline.

