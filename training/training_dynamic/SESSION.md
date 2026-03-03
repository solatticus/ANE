# ANE LoRA on Dynamic Weight Pipeline — Session Dump

## Date: 2026-03-03
## Machines: Blue (authoring) → Mr-Build/Iz (compile + run)
## Files Changed
- `training/training_dynamic/lora.h` (NEW — 163 lines)
- `training/training_dynamic/train_lora.m` (NEW — 732 lines)
- `training/training_dynamic/Makefile` (MODIFIED — added train_lora target)

---

## Background: The Three Approaches

We went through three iterations trying to eliminate ANE recompilation overhead for LoRA training on Stories110M (109M params, 12 transformer layers):

### Approach 1: Multi-input IOSurface (`stories_flex.h`)
- Tried passing weights as separate IOSurfaces alongside activations
- ANE rejected it with `status=0x1d` — the hardware doesn't support multiple dynamic inputs
- **Dead end**

### Approach 2: Partial Recompile (`train_lora_flex.m`)
- Static pipeline where weights are baked into MIL as constants
- Recompiles only changed kernels (Wq/Wv layers) when LoRA merges new weights
- **Works**: 2x speedup over full recompile, ~210ms/step
- **Problem**: Still recompiles 24 kernels per batch, exec() restart every 70 steps

### Approach 3: Dynamic Weight Pipeline (`training_dynamic/train.m`)
- **Key insight**: Pack weights into the spatial dimension of a single IOSurface
- Layout: `[1, DIM, 1, SEQ + weight_cols]` — activations in `sp[0:SEQ]`, weights as extra columns
- MIL graph uses `slice_by_size` to extract weight region, `reshape` + `matmul` to multiply
- ANE doesn't know the "extra columns" are weights — just sees a wider tensor
- Compiles 9 kernels ONCE at startup. Zero recompilation. ~100ms/step for full fine-tuning.
- **This was the breakthrough**

---

## What We Built: Approach 4 — LoRA on Dynamic Pipeline

**Goal**: Graft LoRA onto the dynamic pipeline. Best of both worlds: zero recompilation + LoRA's parameter efficiency.

### Architecture

**Forward pass per layer:**
1. `lora_merge()` on CPU: `W_eff = W_frozen + (alpha/rank) * B @ A` for Wq, Wv
2. `transpose_weight()` to update transposed buffers (ANE kernels use transposed weights)
3. Pack merged Wqt/Wvt + frozen Wkt/Wot into SDPA IOSurface → `ane_eval()`
4. Pack frozen W1t/W3t into FFN W13 IOSurface → `ane_eval()`
5. Pack frozen W2t into FFN W2 IOSurface → `ane_eval()`

**Backward pass per layer — skip 5 of 7 dW GEMMs:**
- **SKIP**: dW1, dW2, dW3 (FFN frozen), dWo (frozen), dWk (frozen)
- **KEEP**: dWq, dWv (LoRA targets)
- All ANE backward kernels still run for activation gradients (dx needed for backprop)
- Only the CPU-side `cblas_sgemm` dW dispatches are skipped for frozen weights

**Post-batch:**
- `lora_extract_grads()` projects full dWq/dWv → low-rank dA/dB
- Adam updates only LoRA A/B matrices (48 small matrices, 147K params)
- Zero grads for Wq/Wv only

**Also frozen (no grad accumulation):**
- Embedding layer — skip `embed_backward()` and dEmbed dispatch entirely
- RMS norm weights — `rmsnorm_bwd()` still runs for dx flow, but dw goes to dummy buffer
- Final RMS norm + classifier — dx computed, dw discarded

### Files Created

**`training/training_dynamic/lora.h`** (163 lines)
- Exact copy of `training/lora.h` with one include change: `stories_config.h` → `config.h`
- Contains: `LoRAAdapter`, `LayerLoRA`, `lora_merge()`, `lora_extract_grads()`, `lora_save()`, `lora_load()`
- Needed because `stories_config.h` conflicts with the dynamic pipeline's `config.h` (same structs, different files)

**`training/training_dynamic/train_lora.m`** (732 lines)
- Adapted from `train.m` (876 lines)
- Removed: `LayerAdam la[NLAYERS]`, `AdamState arms_final`, `AdamState aembed`, full Adam state (~1.7GB)
- Added: `frozen_Wq[NLAYERS]`, `frozen_Wv[NLAYERS]`, `LayerLoRA lora[NLAYERS]`
- Forward: identical to train.m but Wqt/Wvt come from LoRA-merged weights
- Backward: removed 5 dW GEMM dispatch blocks (FFN dW2/dW1/dW3, dWo, dWk) and their malloc+memcpy captures
- Post-batch: LoRA extract → Adam on 48 small matrices instead of all 109M params
- Checkpoint: `lora_save()`/`lora_load()` (~2.9MB) instead of full checkpoint (~438MB)
- No exec() restart, no recompilation

**`Makefile`** — added `train_lora` target with same CC/CFLAGS/LDFLAGS as `train`

---

## Results

### 20-step validation run
```
Step 0 loss: 3.7203 (matches all other trainers with same pretrained weights)
Compile: 330ms one-time, 9 kernels
|B| growing: 0.000000 → 0.052338 (gradient flow confirmed)
127.5ms/step average
```

### 100-step benchmark run
```
=== ANE Dynamic LoRA Training: Stories110M ===
LoRA: rank=4 alpha=4.0 trainable=147,456
Compiled 9 kernels in 333ms (one-time)

step 0    loss=3.7203  |A|=13.7915 |B|=0.000000  164.4ms/step
step 40   loss=3.4296  |A|=13.8015 |B|=0.322337  119.7ms/step
step 80   loss=3.4178  |A|=13.8369 |B|=0.729982  126.4ms/step
step 100  [checkpoint saved]

=== LoRA Dynamic Efficiency Report ===
Total steps:  100
Compile:      333ms (one-time, 2.6%)
Train time:   12560ms (125.6ms/step)
Wall time:    12.6s
Compile count: 9 (should be 9) ✓
dW GEMMs/layer: 2 (Wq, Wv) — skipped 5 frozen
```

### Performance Comparison (all approaches)

| Metric | train_lora (static) | train_lora_flex (partial) | train.m (dynamic full) | **train_lora.m (dynamic LoRA)** |
|--------|--------------------|--------------------------|-----------------------|-------------------------------|
| Compile | 3400ms/batch | 1327ms/batch | 333ms once | **333ms once** |
| ms/step | ~400ms | ~210ms | ~100ms | **125.6ms** |
| exec() restart | every 10 steps | every 70 steps | never | **never** |
| Trainable | 147K | 147K | 109M | **147K** |
| Adam memory | 2.3MB | 2.3MB | 1.7GB | **2.3MB** |
| Checkpoint | 2.9MB | 2.9MB | 438MB | **2.9MB** |
| dW GEMMs/layer | 7 (2 used) | 7 (2 used) | 7 | **2** |
| Speedup vs static | 1x | 1.9x | 4x | **3.2x** |

The 125ms/step vs projected 80-90ms is due to LoRA merge + retranspose overhead (~30ms for 12 layers × 2 weights). The ANE kernels themselves are fast (fwd ~23ms, bwd ~32ms).

---

## The Broader Context: What This Means

### What we actually did vs the hype
An article circulated (OpenClaw / Zero-Human Company / "Mr. Grok") claiming to do real-time ANE fine-tuning with fictional Python pseudocode (`model.to("ane")`, `ANEBackprop()` context manager — none of which exists). Our pipeline is the real thing:

- Actual MIL program compilation targeting ANE via private `_ANEInMemoryModel` APIs
- IOSurface-based weight packing in spatial dimension (the key insight that makes dynamic weights work)
- Real backpropagation kernels (SDPA backward, FFN backward) running on the Neural Engine
- CPU-side cblas for dW GEMMs + LoRA merge, Accelerate framework for Adam
- Working, benchmarked, checkpoint-resumable

### Key technical insight worth documenting
The ANE rejects multiple dynamic IOSurface inputs (status 0x1d). The workaround is packing weights as extra spatial columns in a single IOSurface: `[1, IC, 1, SEQ + weight_cols]`. The MIL graph uses `slice_by_size` to extract regions. The ANE just sees a wider tensor — it doesn't know some columns are weights. This is what enables zero-recompile training.

---

## Next Steps: Integration into Oscar/Cortex/Charlotte

### The question on the table
How do we integrate this LoRA training capability into the production stack?

### What needs investigation (Lex containers were down)
1. **Oscar** (10.10.0.2:9090) — The LLM inference server. Does it serve Qwen? Can it hot-swap LoRA adapters? Does it have an API for weight injection?
2. **Cortex** — The orchestration/routing layer that Fortress proxies to. How does it manage model state? Does it have hooks for fine-tuning triggers?
3. **Charlotte** — Agent system. Could it orchestrate training jobs? Request fine-tuning based on conversation patterns?

### Possible integration patterns
- **On-demand LoRA training**: Charlotte detects a use case (user correction, style preference) → triggers train_lora.m on Mr-Build → produces 2.9MB checkpoint → hot-loads into Oscar's inference
- **Continuous adaptation**: Rolling buffer of recent interactions → periodic LoRA fine-tune → merge into serving weights
- **Multi-adapter**: Different LoRA checkpoints for different users/contexts, swap at inference time

### To research when Lex is back up
- Oscar's model loading API and whether it supports adapter merging
- Cortex's architecture and how it routes between models
- Whether Qwen (the production model) can use the same pipeline (different dimensions/layers than Stories110M)
- Charlotte's task system for orchestrating training runs

---

## Code Location
All source on Mr-Build (Iz): `/Users/iz/src/ANE/training/training_dynamic/`
- `train_lora.m` — the LoRA dynamic trainer
- `lora.h` — LoRA adapter (local copy with config.h include)
- `train.m` — full fine-tuning dynamic trainer (unchanged)
- `config.h`, `io.h`, `mil_dynamic.h`, `cpu_ops.h` — shared infrastructure (unchanged)
