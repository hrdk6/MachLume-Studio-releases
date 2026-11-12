# The performance model

> A published copy of `docs/MODEL.md` from MachLume Studio's source, which is private: the file names in it refer to that source. Published with MachLume Studio 0.3.0.

This is the model behind every prediction in MachLume Studio. It is deliberately small: about 300 lines
of pure functions in `src/model/`, with no fitted constants in the pure prediction.
Equation tags such as (A1) and (D3) appear in the code's doc comments next to the function that
implements them. Sections 1–9 describe the GPU engine (WebLLM); section 10 gives the changes for the
CPU engine (llama.cpp). Section 8 works one prediction by hand, and
`test/unit/model-doc-example.test.ts` checks that the
code reproduces it.

## 0. Notation

| Symbol | Meaning | Example (Qwen2.5-0.5B) |
|---|---|---|
| L | transformer layers | 24 |
| d | model (hidden) width | 896 |
| h | query heads | 14 |
| h_kv | key/value heads (GQA when h_kv < h) | 2 |
| d_h | head dimension | 64 |
| d_ff | FFN inner width (per expert for MoE) | 4864 |
| V | vocabulary size | 151,936 |
| E, k | MoE: experts, experts used per token | – |
| P | prompt tokens (prefill length) | 128 |
| G | generated tokens | 64 |
| c | tokens in the KV cache that the current token attends to (including itself) | – |
| b | batch size (sequences decoded together) | 1 |
| b_w | bytes per stored weight, including group scales (Q1) | 0.5625 |
| a | bytes per activation / KV element (the scheme's model dtype) | 2 |
| C | achievable matmul throughput (measured tiled GEMM), FLOP/s | – |
| Π | FMA peak (measured register-only kernel), FLOP/s | – |
| β | achievable bandwidth for the op's access pattern (measured), byte/s | – |
| t_disp, t_sync | measured per-dispatch overhead and GPU→CPU round trip | – |

Counting conventions: one multiply-add is **2 FLOPs**; logits are fp32 (4 B); 1 GB = 10⁹ bytes.

## 1. Parameters

Per layer, with q = h·d_h and kv = h_kv·d_h:

- (A1) attention matrices: N_attn = d·q + 2·d·kv + q·d (W_Q, W_K, W_V, W_O)
- (A2) Q/K/V biases (Qwen2.5 only): q + 2·kv
- (A3) one gated FFN (gate, up, down): 3·d·d_ff
- (A4) FFN weights in memory: dense = (A3); MoE = E·(A3) + d·E (router)
- (A5) FFN weights read per token: dense = (A3); MoE = k·(A3) + d·E
- (A6) norm vectors: L·(n_norm·d + [2·d_h if q/k-norm]) + d. Here n_norm = 2 for pre-norm blocks and 4 for Gemma 2.
- (A7) embedding V·d, plus another V·d for the LM head unless the embeddings are tied
- (A8) total = (A7) + L·(A1 + A4) + L·(A2) + (A6). "Non-embedding" removes (A7).

**Validation.** The counts reproduce the WebLLM builds' in-memory parameter bytes
(`tensor-cache.json` → `ParamBytes`) within 0.1% for all 36 engine builds in the data file, and exactly
for most of them. For example, Qwen2.5-0.5B q4f16_1 gives 277,996,288 B in both the code and the
engine. They also match the Qwen, Phi-3.5, and TinyLlama model cards within the cards' rounding.
For Qwen3-30B-A3B the model gives 30.53B total and 3.35B active, against the card's "30.5B in total
and 3.3B activated".

## 2. Quantization

The layout comes from MLC's source (`quantization.py`, `group_quantization.py`). `q4f16_1` and
`q4f32_1` use group quantization with group size 32 and int4 symmetric values with an implicit offset
of 7, so no zero-point is stored. Eight values are packed per `uint32`, with one scale per group
stored in the model dtype. The embedding and the final FC layer are quantized as well. Norm weights
and biases stay in the model dtype.

- (Q1) b_w = bits/8 + scale_bytes/32. That gives 0.5625 B (q4f16_1), 0.625 B (q4f32_1), 2 B (q0f16),
  and 4 B (q0f32).
- (A9) weight bytes = (quantizable weights)·b_w + (norms + biases)·a
- (Q2) Dequantization costs about 4 ALU instructions per weight each time a weight is unpacked: shift
  and mask, convert, subtract the offset, multiply by the scale. **This is an assumption.** It is
  charged against Π/2 (one FMA-instruction slot each).
- (Q3) In prefill, a weight is unpacked once per 64-row output tile, so ⌈P/64⌉ times in total. **This
  is also an assumption.** In decode the term is **not** used: the measured 4-bit GEMV bandwidth already
  includes unpacking, because the benchmark kernel unpacks exactly the MLC layout.

WebLLM ships no int8 builds for these models, so the app offers none.

## 3. Decode: one token, KV length c

| Op class | FLOPs | DRAM bytes |
|---|---|---|
| (D1) Q/K/V/O | 2·b·L·N_attn | L·(N_attn·b_w + bias·a) + b·L·(2d + 2q + 2kv)·a |
| (D2) FFN | 2·b·L·N_ffn,active | L·N_ffn,read·b_w + b·L·(2d + 3·d_ff)·a |
| (D3) attention | 4·b·L·h·d_h·c | 2·b·L·kv·c·a (read K and V) + b·L·(2q + 2kv)·a |
| (D4) LM head | 2·b·V·d | V·d·b_w + b·(4V + d·a) |
| (D5) other | norms, RoPE, residuals, activation, attention softmax (lumped), plus the sampler's softmax over V: 5·b·V | elementwise reads and writes, plus 2·b·V·4 |

N_ffn,read is the expert weights touched by the batch. (O1) A batch of b tokens picking k of E experts
touches E·(1 − (1 − k/E)^b) experts on average, which is k for b = 1.

**Why decode is memory-bound.** For the weight-streaming ops (D1, D2, D4), FLOPs/bytes ≈ 2/b_w:

- about 1 FLOP/byte at fp16
- about 3.6 FLOP/byte at 4-bit with fp16 scales

Measured ridge points on GPUs are tens of FLOP/byte (this laptop: 2993/63 ≈ 47), so these ops sit far
left of the ridge. Attention over the KV cache has intensity 4·h·d_h·c / (2·h_kv·d_h·c·a) =
2·(h/h_kv)/a. For multi-head attention with fp16 that is 1. GQA multiplies it by h/h_kv: about 6 for
Qwen2.5-1.5B, whose 12 query heads share 2 KV heads. That is still memory-bound. The unit test
asserts this relation.

## 4. Prefill: P tokens after an offset s (s = 0 unless the prompt is chunked)

| Op class | FLOPs | DRAM bytes |
|---|---|---|
| (P1) Q/K/V/O | 2·P·L·N_attn | L·(N_attn·b_w + bias·a) + P·L·(2d + 2q + 2kv)·a |
| (P2) FFN | 2·P·L·N_ffn,active | L·N_ffn,read(P)·b_w + P·L·(2d + 3·d_ff)·a |
| (P3) causal attention | 2·L·h·d_h·[(s+P)(s+P+1) − s(s+1)] | L·P·(2q + 4kv)·a + 2·L·kv·s·a |
| (P4) LM head (last token only) | 2·V·d | V·d·b_w + 4V |
| (P5) other | P × per-token elementwise work + softmax | elementwise traffic |

Conventions and assumptions:

- **Causal counting.** Token i attends to i positions, so QKᵀ plus AV cost Σᵢ 4·h·d_h·i = 2·h·d_h·P(P+1)
  per layer. This counts only the computed half. A kernel that computes the full P×P and masks it
  would do about twice the work.
- **Fused attention.** Scores never touch DRAM, as in FlashAttention.
- **Weights are read once per forward pass.** This is ideal reuse; a real GEMM re-reads tiles from cache.
- **LM head for the last position only.** I checked this in MLC's model code: `prefill()` calls
  `index_last_token` before `lm_head`.
- **Chunking.** WebLLM's prebuilt WebGPU libraries are compiled with a 1024-token prefill chunk
  (`_cs1k` in the library name). A longer prompt is split into ⌈P/1024⌉ forward passes. Each pass
  re-reads the weights, and later chunks attend to the earlier ones (offset s).

Intensity of (P1) and (P2) ≈ 2P/b_w, so it grows linearly with P. A prompt becomes compute-bound when
2P/b_w passes the ridge. On this laptop with 4-bit weights that happens at P ≈ 47·0.5625/2 ≈ 13 tokens.

## 5. Roofline timing

- (R1) t_op = max( F/C + D/(Π/2), B/β ). F is FLOPs, D is dequant instructions, B is bytes. The op is
  *compute-bound* if the first term wins and *memory-bound* otherwise.
- (R2) t_phase = Σ_op t_op + n_disp·t_disp + n_sync·t_sync
- (R3) decode: t_token at the mean context c̄ = P + G/2. The G−1 decode steps attend to P+1 … P+G−1
  positions, cost is linear in c, so the mean context gives the mean time. Then tokens/s = 1/t_token.
- (R4) TTFT = t_prefill(P), including sampling of the first token.

**Which measured ceiling applies.** Every ceiling in this table is measured on the device (see
METHODOLOGY.md):

| Phase | Op | β (bandwidth) | C (compute) |
|---|---|---|---|
| decode | Q/K/V/O, FFN, LM head | GEMV bandwidth of the weight format (4-bit, f16, or f32 kernel) | GEMM, precision of the scheme |
| decode | attention, other | stream read | GEMM |
| prefill | all | stream read | GEMM, precision of the scheme (f16 if the device has it) |

**Overhead term.**

- n_disp = L·10 + 8 per forward pass. This is an assumption (ASSUMPTIONS A-10); Verify replaces it with
  a **counted** number. Prefill multiplies it by the number of chunks.
- n_sync = 1 per forward, because the sampled token must come back to the CPU. t_sync is the round trip
  measured inside a Web Worker when available, because the engine runs in one; otherwise the main-thread
  value is used. On the laptop the two are the same, 3.0 ms.
- The overhead is **added**, not overlapped. WebLLM's TVM runtime encodes every kernel of a forward
  into one command encoder, submits once, and then waits: encoding, GPU work, and the round trip run
  one after another.

## 6. Memory

- (K1) KV per token = 2·L·h_kv·d_h·a
- (K2) paged KV, with the 16-token pages MLC uses: ⌈ctx/16⌉·16·(K1)·b. The waste is less than one page
  per sequence.
- (K3) naive static KV: max_ctx·(K1)·b, reserved up front. The waste is (max_ctx − ctx)·(K1)·b, which
  is the internal fragmentation that PagedAttention removes.
- (K4) scratch is about C_chunk·(2d + q + 2kv + 3·d_ff)·a + 4V. **This is an estimate.**

The model also checks the largest single tensor (usually the V×d embedding) against
`maxStorageBufferBindingSize`, and the total against the user's RAM budget.

## 7. Calibrated prediction: two fitted parameters

- (C1) t_cal = α·t_roof + τ. Here t_roof is Σ_op t_op **without** the overhead term, α is an efficiency
  factor (the engine reaches 1/α of the measured ceilings), and τ is a fixed per-forward overhead.
- **Fit.** Weighted least squares over every verified run's TTFT and per-token decode time on this
  device, with weights 1/t_meas², so the fit minimizes relative error. The constraints α > 0 and τ ≥ 0
  are enforced by refitting one parameter if needed.
- **Shown only from 3 verified runs.** Each run is graded **leave-one-run-out**, so a run is never
  scored by a fit that saw it. The app labels "pure" and "calibrated" everywhere.

## 8. Worked example, by hand

Qwen2.5-0.5B-Instruct, q4f16_1, P = 128, G = 64. The ceilings are this laptop's measurements, rounded:

| Ceiling | Value |
|---|---|
| C (GEMM f16) | 3000 GFLOP/s |
| Π (FMA f16) | 6950 GFLOP/s |
| β_GEMV,4-bit | 72 GB/s |
| β_read | 63 GB/s |
| t_disp | 3.5 µs |
| t_sync | 2900 µs |

**Decode FFN (D2).**

- N_ffn = 3·896·4864 = 13,074,432 per layer; L·N_ffn = 313,786,368 weights.
- F = 2·313,786,368 = **627,572,736 FLOP**.
- B = 313,786,368·0.5625 + 24·(2·896 + 3·4864)·2 = 176,504,832 + 786,432 = **177,291,264 B**.
- Intensity 3.54 FLOP/B, which is memory-bound because 3.54 < 47.
- t = 177,291,264 / 72·10⁹ = **2.462 ms**. The compute side would take only 0.209 ms.

**Decode total.**

- The other ops are worked the same way: Q/K/V/O 0.347 ms, LM head 1.072 ms, attention at c̄ = 160
  0.033 ms, other 0.037 ms (half of it is the sampler's softmax over 151,936 logits).
  Roofline = **3.952 ms**.
- Overhead: n_disp = 24·10 + 8 = 248, so 248·3.5 µs + 2900 µs = **3.768 ms**.
- t_token = 7.720 ms, so **129.5 tok/s** predicted. On this device, launch and round-trip overhead is
  as large as the whole memory-bound roofline time. That is the hypothesis Verify tests.

**Prefill FFN (P2).**

- F = 2·128·313,786,368 = **80,329,310,208 FLOP**.
- Dequant D = 313,786,368·⌈128/64⌉·4 = 2.51·10⁹ instructions.
- t_compute = 80.33·10⁹/3000·10⁹ + 2.51·10⁹/3475·10⁹ = 26.78 + 0.72 = **27.50 ms**.
- t_memory = 4.40 ms, so the op is **compute-bound**.
- TTFT = roofline 35.16 ms + overhead 3.77 ms = **38.93 ms**.

**Memory at 192 tokens.**

- Weights 277,996,288 B. This is exactly the engine's `ParamBytes`.
- Paged KV = ⌈192/16⌉·16·12,288 = 2,359,296 B. A static reservation for 4096 tokens would be 50,331,648 B.

## 9. What the model ignores

These are deliberate, and each is a candidate explanation when predictions miss:

- **Sampling.** WebLLM 0.2.86 does not shortcut greedy decoding. Even at temperature 0, every token
  runs a softmax over V, a **full-vocabulary argsort**, and a top-p kernel on the GPU, then copies the
  token to the CPU (read from `index.js`, `sampleTokenFromLogits`). The softmax and the round trip are
  modelled. The argsort is not, because its pass count depends on TVM's sort implementation. Its
  launches do show up in the counted dispatches.
- **Cache effects.** Weights are assumed to stream from DRAM every token. A small model whose weights
  fit in a big last-level cache would beat the prediction.
- **Kernel quality.** The engine's kernels may reach more or less of a ceiling than my benchmark kernels.
- **Activation and KV traffic beyond the listed terms.** For example, an engine that materializes attention scores.
- **CPU work.** Tokenization and sampling logic other than the measured round trip.
- **Concurrency.** It overlaps nothing: no CPU-GPU pipelining. That is true for this engine, but it
  would be pessimistic for one that pipelines.
- **Thermal and power drift during a run.** Verify measures bandwidth before and after to detect it.
- **Speculative decoding, batching beyond b, and prefix caching.**

## 10. The CPU engine (llama.cpp through wllama)

The same equations apply, with the CPU engine's formats, ceilings and memory layout. Code:
`cpuCeilingsFor` in `predict.ts`, `weightTraffic` in `ops.ts`.

**Formats (Q-CPU).** llama.cpp's block formats (`ggml-common.h`):
- **Q4_0:** an fp16 scale plus 32 4-bit values in 18 bytes, so b_w = 0.5625.
- **Q8_0:** an fp16 scale plus 32 int8 values in 34 bytes, so b_w = 1.0625.

Activations are quantized to Q8_0 and multiplied directly with the weight blocks, so there is no
separate dequantization pass (D = 0). The measured dot rates already include the unpacking.

**Exact bytes per operation.** Real files mix types:
- Qwen's official Q4_0 file stores its separate LM head in Q8_0.
- Llama 3.2's tied embedding is Q6_K.
- bartowski's SmolLM2 files keep some layers in Q4_1.

So for GGUF builds the weight bytes are not params × b_w. They are the sums, read from the file's tensor
table, of the attention matrices, the FFN matrices, the LM head (or the tied embedding) and one embedding
row per token. A unit test checks that these add up to the file's tensor bytes within 1%. Activation and
KV terms are as in §3–4, with a = 2 bytes (llama.cpp's KV cache is fp16).

**Ceiling selection (C-CPU):**

| Op | Prefill compute C | Decode bandwidth β |
|---|---|---|
| Q/K/V/O, FFN | in-cache dot rate of the layer format (`dotQ4` or `dotQ8`) | DRAM GEMV of that format (`bwGemvQ4` or `bwGemvQ8`) |
| LM head | 4-bit types: `dotQ4`; other types (Q6_K, Q8_0): `dotQ8` | `bwGemvQ4` or `bwGemvQ8` likewise |
| attention, other | in-cache f32 dot rate (`gemmF32`) | stream read |

Decode ops take max(F / C, B / β) like every other op. On the development laptop's CPU the two sides of
4-bit decode nearly meet, so neither alone predicts it.

**No overhead term.** There are no GPU dispatches or readbacks: t_dispatch = t_sync = 0. Per-operation
thread synchronisation inside llama.cpp is not modelled (ASSUMPTIONS.md A-22). Verify measures what that
leaves out.

**Prefill chunking.** llama.cpp evaluates a prompt in micro-batches of n_ubatch = 512 tokens, so P > 512
re-reads the weights once per micro-batch, as in §4 with chunk 512.

**Memory.** llama.cpp allocates the KV cache for its whole context window up front: total = weights +
KV_static(n_ctx) + scratch, with n_ctx = 2048 for the catalog's CPU builds.

**Threads.** The ceilings are measured on the same number of threads the engine uses (⌊cores / 2⌋ when
the page is cross-origin isolated, else 1), so threads are inside C and β rather than a separate term.

## 11. Native llama.cpp in Ollama (desktop app)

The desktop app predicts runs of Ollama's llama.cpp with the same equations. Three inputs differ from §10,
and none of them comes from the run being predicted. Code: `src/desktop/predict.ts`,
the kernels in `src-tauri/src/bench`.

**Native ceilings.** The WebAssembly ceilings describe the browser's build of llama.cpp, which reaches about
a fifth of what native code does. So the desktop app measures llama.cpp's own x86 loops instead, ported to
Rust from `ggml-cpu/arch/x86` (AVX2, with AVX-VNNI where the CPU has it: Ollama's "alderlake" and
"haswell" builds of ggml). Before timing, each kernel is checked against a scalar reference.

| Ceiling | Kernel | Use |
|---|---|---|
| `gemmQ4x8` | `ggml_gemm_q4_0_8x8_q8_0`: 4-bit weights interleaved 8 rows at a time, 16 prompt tokens per tile, 512-token micro-batch | prefill of 4-bit layer weights |
| `bwGemvQ4x8` | `ggml_gemv_q4_0_8x8_q8_0` over a buffer far larger than any cache | decode of 4-bit layer weights |
| `dotQ8`, `bwGemvQ8` | `ggml_vec_dot_q8_0_q8_0` (8-bit weights are not repacked on x86) | 8-bit weights, and the LM head (often Q6_K) |
| `dotQ4`, `bwGemvQ4` | `ggml_vec_dot_q4_0_q8_0` | 4-bit weights when repacking is off |
| `attnF16` | the per-position loop of llama.cpp's CPU flash attention for one query (fp16 dot product, online softmax with `expf`, fp16 accumulation of the value), from cache | decode attention, timed by (R1s) in §12 |
| `dotF16` | `ggml_vec_dot_f16` | prefill attention (llama.cpp's tiled path; the dot rate stands in) |
| `gemmF32`, `peakF32`, `bwRead` | f32 dot, FMA chains, streaming read | other ops; the roofline's reference lines |

ggml repacks Q4_0 (and Q4_K) layer weights into this interleaved layout on every AVX2 CPU unless told
not to; Ollama's log confirms it (`CPU_REPACK model buffer size = 3744.00 MiB` for Llama 3 8B).

**Threads.** llama.cpp hands out matmul chunks dynamically (`ggml_threadpool_chunk_add`), so every thread
stays busy until an operation ends, on fast and slow cores alike. The benchmark therefore runs all
threads together for a fixed window and sums their rates. Timing the slowest of equal static shares
would understate a hybrid CPU: P-cores would sit idle waiting for E-cores. Like llama.cpp
(`ggml_thread_apply_priority`), the benchmark threads also ask Windows not to throttle them. Without
that, a run with the app's window behind others measured 43 GB/s where Ollama was getting 55. The
default thread count is
llama.cpp's own: half the logical processors when there are more than four (`common.cpp`), which is
what Ollama runs with unless `num_thread` is set. Windows decides which cores run which thread. On the
development PC the 4-bit GEMM at 7 of 14 threads measured between 480 and 800 GFLOP/s from one run to
the next. Windows of 250 ms keep the spread inside a run to a few percent (CV); the spread between runs
stays, and it bounds how closely prefill can be predicted.

**Engine configuration.** llama.cpp writes its setup to Ollama's server log when it loads a model. The
app reads these fields from the newest load: thread count, ggml CPU build, enabled instruction sets,
repacked bytes, layers on the GPU, context, micro-batch and flash attention. A log block counts only if
its GGUF blob matches the profiled model's (`sha256` in both). Without one, llama.cpp's defaults apply
and the prediction says so.

**Exact model facts.** Ollama's `/api/show` (verbose) returns the tensor table. From it come:
- the bytes per op class, as in §10;
- the weight counts per op class, so the FLOPs need no assumption about layer shapes;
- the query and key widths summed over the layers, so attention work and KV bytes are exact even when
  layers differ.

The last two matter for models whose layers differ. Gemma 4 12B's global-attention layers are wider than
its sliding-window ones. Its metadata reports only the global head size, which would put the attention
projections at 6.0 B parameters; the tensor table holds 2.4 B.

**Weight types.** Q4_0 and Q8_0 use their own kernels. Q4_1, Q4_K, IQ4_NL, IQ4_XS and MXFP4 are timed
with the 4-bit kernels; Q5 and Q6 with the 8-bit ones. Every matrix's bytes stay exact, so memory-bound
decode is unaffected, but the K-quant kernels unpack more per block, and the prediction states this
approximation. Runs with layers on the GPU are not predicted yet.

**Results on the development PC** (Llama 3 8B Q4_0, 64 generated tokens, ceilings measured in the app:
`gemmQ4x8` 796 GFLOP/s, `bwGemvQ4x8` 56.8 GB/s, evidence in docs/evidence/desktop-ollama-llama3-p*.json):

| Prompt | Prefill predicted | measured | error | Decode predicted | measured | error |
|---|---|---|---|---|---|---|
| 44 tokens (cold start) | 0.785 s | 0.961 s | −18% | 12.7 tok/s | 13.1 tok/s | −3.2% |
| 119 tokens | 2.12 s | 2.37 s | −11% | 12.7 tok/s | 13.0 tok/s | −2.4% |
| 427 tokens | 7.73 s | 8.21 s | −5.9% | 12.6 tok/s | 12.7 tok/s | −0.8% |

Decode, which is memory-bound, is within 3.2% at every length. The prefill shortfall shrinks as the
prompt grows. A straight line through the three points, measured ≈ 1.04 × predicted + 0.14 s, suggests a
fixed cost per prefill that the roofline leaves out:
- thread synchronisation across the few hundred operations of a forward pass;
- ggml's slower 4-row path for prompt rows that do not fill a 16-row tile.

The 44-token run also had the PC paging (0.2 GB of RAM free, 5,260 hard faults/s). The calibration in §12,
with many more runs, revises this reading: the 44-token run was a cold start, and warm runs show a scale,
not a fixed cost.

## 12. Calibration for Ollama runs, and what it found

Code: `src/desktop/calibration.ts`; shown in the profiler and on Predict
for native profiles.

**Fit.** Every saved session's pure prediction is recomputed with the current native profile, so all runs
are measured against one set of ceilings. Then, per phase:

- (C1) t_cal = α·t_roof + τ, by weighted least squares on relative error, as in §7.
- **Fixed term τ.** It is fitted only when the runs' roofline times span at least 2×. On runs of nearly
  equal length, which is every decode token, a constant cannot be told apart from a scale, so they get
  α alone.
- **Grading.** Every run is graded leave-one-out, and the app says plainly whether calibration beat the
  pure prediction on unseen runs.
- **Data quality.** Two kinds of runs are kept out of the prefill fit:
  - a prefill more than 2× faster than the roofline allows (Ollama answered from its prompt cache);
  - the first prompt after Ollama loads the model (a cold start), which is not the steady state the
    roofline describes. Its penalty is reported against the warm fit.
- **Reused prompt tokens.** `prompt_eval_count` includes the tokens Ollama reused from its cache, and
  `prompt_eval_duration` times only the others. Ollama 0.40.15 reports the reused ones as
  `prompt_eval_cached_count`: on warm Llama 3 runs on 2026-11-08, 10 and 12 of about 40 (the chat
  template's opening and the run tag's first characters), which would have read as a third faster
  prefill. So prefill is predicted for the evaluated tokens, with the reused ones as context
  (`predictPrefill`'s `past`), and timed for them. The runs below, from Ollama 0.40.10, report no such
  count and show no such reuse: an unclamped fit of (C1) to their warm prefills gives τ = −0.02 s, about
  one token.

**Decode attention (R1s).** llama.cpp's decode attention handles each cached position in a dependent
chain: read the key row, take a dot product and an `expf`, then read and accumulate the value row. Memory
time cannot hide under compute time there, so for this op

(R1s) t = F / C_attn + B / β

with C_attn the in-cache rate of that loop (`attnF16`) and B the KV bytes, counted once per KV head. With
grouped-query attention, each row read from memory serves every query head of its group. Two
measurements support the sum:
- streaming the KV cache from memory instead of cache took 16.2 instead of 7.3 ps per FLOP, close to the
  two times added;
- a bracketed experiment, below.

**Results on the development PC.** Llama 3 8B Q4_0; 16 recorded runs (12 warm), 42–1,652 prompt tokens;
evidence in docs/evidence/desktop-ollama-llama3-*.json; profile docs/evidence/desktop-native-cpu-m6.json.
The calibrated column is graded leave-one-out.

| Phase | Fit | Pure, median error | Calibrated, median error |
|---|---|---|---|
| Prefill (12 warm runs) | 1.06 × roofline, τ = 0 | 4.8% | 2.4% |
| Decode (16 runs) | 1.11 × roofline | 9.6% | 3.8% |

- **Prefill falls short of the roofline by a scale, not a fixed cost.** The runs span a 40× range of
  lengths, so a τ was free to appear; it fitted to zero. llama.cpp's prefill reaches about 94% of the
  measured GEMM ceiling (91% with the milestone 5 profile). The 0.14 s "fixed cost" seen in §11 came
  from mixing in cold starts.
- **Cold starts.** The first prompt after a load took a median 14% longer than the warm calibrated
  prediction (4 runs). Weights still arriving from the file cache and memory pressure both fit this; the
  session readings show which applied.
- **Decode depends on the PC's state as much as on the model.** The same PC's streaming bandwidth
  measured 63–79 GB/s over one evening, unthrottled: power management, other programs and memory
  pressure, with only 15 GB of RAM for an 8B model. A profile measured in a faster state than the runs predicts decode
  9.6% too fast, and calibration corrects that (3.8%). With the milestone 5 profile, measured near most
  runs, pure decode is already within 3.2%, and calibration makes it worse (5.2%). The app reports
  whichever is true for the user's runs.
- **Attention at long context.** Three runs back to back (120, 1,652 and 120 prompt tokens) share one
  state of the PC, so the ratio of their decode speeds tests the model, whatever the ceilings' absolute
  level:
  - measured: 1,652-token decode at 0.886 of the short runs';
  - timing attention as an f16 dot product predicted 0.959;
  - with `attnF16` (89 GFLOP/s, against 257 for the plain dot) and (R1s): 0.863.

  This rests on one bracketed comparison; more contexts and models would firm it up.
