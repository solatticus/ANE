# ANE — Direct Apple Neural Engine Access (macOS + iOS)

Fork of [maderix/ANE](https://github.com/maderix/ANE). Training and inference on Apple's Neural Engine via reverse-engineered private APIs. No CoreML, no Metal, no GPU — pure ANE compute.

This fork adds: **iOS support (confirmed on iPhone 16 Pro)**, the **Annie pipeline (Qwen2.5-3B LoRA)**, and **Ghidra-based reverse engineering** of the private framework internals.

## iOS — Confirmed Working

Direct ANE hardware access on iOS, bypassing CoreML entirely. Compile, load, and eval of MIL programs on the A18 Pro Neural Engine — confirmed March 15, 2026 on iPhone 16 Pro running iOS 18.3.

|  | macOS | iOS |
|---|---|---|
| **Private classes** | `_ANEInMemoryModel`, `_ANERequest`, etc. | Same — confirmed in dyld cache + on-device |
| **MIL compiler** | `_ANECCompile`, `_ANECCompileJIT` | Same |
| **IOSurface I/O** | `[1,C,1,S]` fp16 | Same |
| **Layer types** | MatMul, Softmax, Conv, etc. | Same — all present |
| **Bridge change** | `dlopen(framework_path)` | `dlopen(NULL)` fallback for shared cache |
| **Smoke test** | conv [16,16,1,16] identity | **PASS** — hardware eval confirmed |

### How We Got Here

1. **ipsw dyld cache extraction** — Pulled the iPhone 15 Pro shared cache (build 23D8133), confirmed all 4 private classes present with identical selectors
2. **Ghidra 12.0.4 decompilation** — Decompiled `initWithNetworkText:weights:optionsPlist:isMILModel:` from the macOS framework. Key findings:
   - `weights` parameter has an early null check — must pass `@{}`, not `nil`, even for weightless programs
   - `modelWithMILText:weights:optionsPlist:` is a thin wrapper that calls `alloc` + `initWithNetworkText:...:isMILModel:YES`
   - Weight dict values are iterated (sorted keys), `allValues`/`firstObject` extracted, hashed for `hexStringIdentifier`
   - The ANE compiler reads `model.mil` + `weights/` from `$TMPDIR/<hexStringIdentifier>/` during `compileWithQoS:`
3. **Empirical testing** — ANE hardware requires minimum **16 channels AND 16 spatial** for eval. Smaller tensors compile and load but fail at eval with `status=0x1d "Program Inference error"`. Confirmed on both M4 (macOS) and A18 Pro (iOS).

CoreML's E5RT runtime uses the same `_ANERequest` code path internally — these private APIs are the production inference path on both platforms.

### Constraints

- **App Store**: Private API usage = guaranteed rejection. Sideload via Xcode developer signing only.
- **Memory**: iOS terminates apps over ~1.5GB. Model weights must fit within budget.
- **Thermal**: iPhone thermal envelope is tighter than Mac. Sustained workloads may throttle.

## What This Is

A from-scratch implementation of transformer training (forward + backward pass) running on the ANE in Apple Silicon. The ANE is a 15.8 TFLOPS FP16 (M4) inference accelerator that Apple does not expose for training. This project reverse-engineers the `_ANEClient` / `_ANECompiler` private APIs and the MIL (Model Intermediate Language) format to run custom compute graphs — including backpropagation — directly on ANE hardware.

**Training results:**

| Model | Params | ms/step | Pipeline |
|-------|--------|---------|----------|
| Stories110M (12L, dim=768, MHA 12/12) | 109M | **91 ms** | Dynamic (no recompile) |
| Qwen3-0.6B (28L, dim=1024, GQA 16/8) | 596M | **412 ms** | Dynamic (no recompile) |
| Qwen2.5-3B (36L, dim=2048, GQA 16/2) | 3B | **~135 s** | Annie (LoRA rank 8) |

**INT8 W8A8 quantization — 1.88x throughput (M4, H16G):**

| Config | FP16 | INT8 W8A8 | Speedup |
|--------|------|-----------|---------|
| 128x conv 512ch 64x64 | 18.6 TOPS, 14.8ms | 35.1 TOPS, 7.8ms | **1.88x** |
| 64x conv 512ch 64x64 | 18.4 TOPS, 7.5ms | 34.1 TOPS, 4.0ms | **1.85x** |

## Annie — Qwen2.5-3B LoRA Training on ANE

First 3B parameter model trained on Apple Neural Engine:

- **Model**: Qwen2.5-3B (36 layers, 2048-dim, GQA 16/2 heads, 151K vocab)
- **Method**: LoRA rank 8 on Wq/Wv — 1.84M trainable params (0.06% of 3B)
- **Kernels**: 4 dynamic kernels shared across all 36 layers
- **Speed**: ~135s/step, 324 ANE evals/step (144 forward + 180 backward)
- **Hardware**: M4 Pro, 24GB unified memory, 16-core ANE
- **Stability**: Zero crashes, zero ANE errors across extended training runs
- **Loss scaling**: 256x FP16 loss scaling — ANE backward matmul products flush to zero without it

## Architecture

The dynamic pipeline uses shared ANE kernels with weights packed into spatial dimensions (no recompilation when weights change):

**MHA models (Stories110M) — 6 kernels per layer:**

| Kernel | Function |
|--------|----------|
| `sdpaFwd` | QKV projection + SDPA + output projection |
| `ffnFused` | SwiGLU FFN (W1, W3, SiLU, W2) |
| `ffnBwdW2t` / `ffnBwdW13t` | FFN backward (split for memory) |
| `sdpaBwd1` / `sdpaBwd2` | SDPA backward |

**GQA models (Qwen3-0.6B) — 10 kernels per layer.**
**Annie (Qwen2.5-3B) — 4 shared kernels** (`qkvProj`, `dimToHidden`, `hiddenToDim`, `dimToDim`).

Key optimizations:
- **Channel-first CPU layout** — matches ANE IOSurface `[1,C,1,S]` format, eliminates all transpose overhead
- **vDSP vectorized RMSNorm** — 10x faster than naive (6.7ms -> 0.7ms)
- **GCD async cblas overlap** — dW gradient sgemms run in parallel with ANE evals
- **FP16 loss scaling** — 256x scaling prevents gradient underflow in ANE backward matmuls
- **exec() restart** — bypasses ~119 ANE compile limit per process

## File Structure

```
ane_int8_bench.m                    # INT8 W8A8 vs FP16 throughput benchmark
api_exploration.m                   # Initial ANE API discovery
inmem_basic.m                       # In-memory MIL compilation proof-of-concept
inmem_bench.m                       # ANE dispatch latency benchmarks
inmem_peak.m                        # Peak TFLOPS measurement
sram_bench.m / sram_probe.m        # ANE SRAM bandwidth probing
gpu_ane_share.m                     # GPU<>ANE zero-copy IOSurface demo
gpu_prefill_ane_decode.m            # GPU prefill -> ANE decode pipeline

bridge/
  ane_bridge.h                      # C-callable ANE API (compile, eval, I/O, int8)
  ane_bridge.m                      # ObjC implementation (macOS + iOS)
  Makefile

ios/                                # iOS port
  bridge/
    ane_bridge.h                    # iOS-adapted C API header (Ghidra-annotated)
    ane_bridge.m                    # iOS bridge (dlopen fallback for shared cache)
    ANEEngine.swift                 # Swift wrapper for ANE C API
    ANESmokeTest.swift              # SwiftUI smoke test (conv [16,16,1,16] identity)
  Oscar-Bridging-Header.h          # Exposes C API to Swift

training/
  ane_runtime.h                     # ANE private API wrapper (compile, eval, IOSurface)
  ane_mil_gen.h                     # MIL program generation (conv, matmul, QKV, FFN)
  model.h                          # Model weight loading and kernel compilation
  train_large.m                     # Static pipeline (Stories110M)
  dashboard.py                      # Live training dashboard (multi-model, W&B)
  training_dynamic/                 # Dynamic weight pipeline
    train.m                         # Dynamic training loop
    config.h / mil_dynamic.h / io.h
  annie/                            # Qwen2.5-3B LoRA fine-tuning
    config.h                        # Qwen2.5-3B architecture (36L, 2048-dim, GQA 16/2)
    mil_dynamic.h                   # 4 shared dynamic kernels
    forward.h / backward.h         # Forward + backward pass (FP16 loss scaling)
    cpu_ops.h                       # RMSNorm, cross-entropy, Adam
    io.h / lora.h                   # IOSurface I/O, LoRA rank-8
    train_lora.m                    # Main training loop
    convert_weights.py              # HuggingFace safetensors -> ANE binary
    tokenize_data.py                # Conversation JSONL -> tokenized binary
```

## Building

Requires macOS 15+ on Apple Silicon.

```bash
# Bridge library (C-callable ANE API)
cd bridge && make

# Dynamic pipeline (recommended)
cd training/training_dynamic
make MODEL=stories110m
./train --scratch

# Annie / Qwen2.5-3B LoRA
cd training/annie && make train_lora
python convert_weights.py --model Qwen/Qwen2.5-3B --output qwen3b_weights.bin
python tokenize_data.py --input conversations.jsonl --output annie_train_data.bin
./train_lora --steps 1000 --lr 1e-4 --accum 10

# INT8 benchmark
xcrun clang -O2 -fobjc-arc -framework Foundation -framework IOSurface -ldl \
  -o ane_int8_bench ane_int8_bench.m
./ane_int8_bench
```

No external dependencies for C code. Uses only system frameworks + private ANE APIs resolved at runtime via `objc_msgSend`.

## How It Works

1. **MIL generation** — Objective-C code constructs MIL program text at runtime (convolutions for linear layers, matmul for attention, softmax, element-wise ops)
2. **In-memory compilation** — `_ANEInMemoryModelDescriptor` compiles MIL text + weight blobs directly to ANE programs, no disk mlmodelc needed
3. **IOSurface I/O** — Input/output tensors via IOSurface shared memory in `[1, channels, 1, spatial]` format
4. **Dynamic weights** — Activations and weights packed into a single spatial input dimension, sliced apart inside the MIL kernel. Weights change without recompilation
5. **Gradient flow** — Forward taps expose intermediates for backward; backward kernels compute dx on ANE; dW computed on CPU via cblas
6. **INT8 quantization** — `constexpr_affine_dequantize` for int8 weights, `quantize`/`dequantize` for int8 activation caching in L2 SRAM

## Limitations

- **SDPA causal masking** — ANE hardware ignores `attn_mask` in SDPA ops; causal attention decomposed into separate Q@K^T -> mask+softmax -> scores@V
- **~119 compile limit** — ANE compiler leaks resources; worked around via `exec()` restart
- **FP16 gradient underflow** — backward matmuls underflow in fp16; fixed with 256x loss scaling
- **Minimum tensor size** — ANE eval requires >= 16 channels and >= 16 spatial. Smaller tensors compile but fail at eval with 0x1d
- **Single-input constraint** — multi-input ANE requests cause 0x1d error; inputs packed into spatial dimension instead

## Upstream

This is a fork of [maderix/ANE](https://github.com/maderix/ANE). Upstream contributions (INT8, multi-model dashboard) are merged periodically. Our additions (iOS port, Annie pipeline, Ghidra RE) live here. See upstream for the original author's articles:

- [Part 1: Reverse Engineering](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine)
- [Part 2: Benchmarks](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615)
- [Part 3: Training](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-c8b)

## Disclaimer

This project uses Apple's private, undocumented APIs (`_ANEClient`, `_ANECompiler`, `_ANEInMemoryModelDescriptor`). These APIs are not covered by any public stability guarantee and may change or break with any OS update. This is independent research into Apple Neural Engine architecture, using APIs discovered through runtime introspection and binary analysis for research and educational purposes under fair use and interoperability provisions (see *Sega v. Accolade*, 1992; DMCA 1201(f)). No Apple proprietary code or binaries are included in this repository. This project is not affiliated with or endorsed by Apple Inc. Use at your own risk.

## License

MIT — see [LICENSE](LICENSE)
