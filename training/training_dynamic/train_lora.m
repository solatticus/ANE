// train_lora.m — LoRA fine-tuning on dynamic weight ANE pipeline
// Compile kernels ONCE at startup. LoRA-merge on CPU, write merged weights to IOSurface.
// Skip frozen weight dW GEMMs (5 of 7). Zero recompilation. ~80-90ms/step.
#include "mil_dynamic.h"
#include "cpu_ops.h"
#include "lora.h"

#define LORA_CKPT_PATH "ane_lora_dyn_ckpt.bin"
#define MODEL_PATH "../../../assets/models/stories110M.bin"
#define DATA_PATH "../tinystories_data00.bin"

// Dynamic kernel set per layer (same as train.m)
typedef struct {
    Kern *sdpaFwd;     // QKV matmul + SDPA + Wo matmul (dynamic weights via IOSurface)
    Kern *ffnW13;      // W1,W3 matmul (dynamic)
    Kern *ffnW2;       // W2 matmul (dynamic)
    Kern *ffnBwdW2t;   // dffn @ W2^T (dynamic)
    Kern *ffnBwdW13t;  // dh1@W1^T + dh3@W3^T (dynamic)
    Kern *wotBwd;      // dx2 @ Wo^T (dynamic)
    Kern *sdpaBwd1;    // Q,K,V,da → dV,probs,dp (weight-free, has mask const)
    Kern *sdpaBwd2;    // probs,dp,Q,K → dQ,dK (weight-free)
    Kern *qkvBwd;      // dq@Wq^T + dk@Wk^T + dv@Wv^T (dynamic)
} DynLayerKernels;

// ===== Weight loading from llama2.c format =====
static bool load_pretrained(LayerWeights *lw, float *rms_final, float *embed, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { printf("Cannot open %s\n", path); return false; }
    Llama2Config cfg;
    fread(&cfg, sizeof(cfg), 1, f);
    printf("  Model: dim=%d hidden=%d layers=%d heads=%d vocab=%d seq=%d\n",
           cfg.dim, cfg.hidden_dim, cfg.n_layers, cfg.n_heads, abs(cfg.vocab_size), cfg.seq_len);
    if (cfg.dim != DIM || cfg.hidden_dim != HIDDEN || cfg.n_layers != NLAYERS) {
        printf("  ERROR: Config mismatch!\n"); fclose(f); return false;
    }
    int V = abs(cfg.vocab_size);
    fread(embed, 4, V * DIM, f);
    for (int L = 0; L < NLAYERS; L++) fread(lw[L].rms_att, 4, DIM, f);
    for (int L = 0; L < NLAYERS; L++) fread(lw[L].Wq, 4, WQ_SZ, f);
    for (int L = 0; L < NLAYERS; L++) fread(lw[L].Wk, 4, WQ_SZ, f);
    for (int L = 0; L < NLAYERS; L++) fread(lw[L].Wv, 4, WQ_SZ, f);
    for (int L = 0; L < NLAYERS; L++) fread(lw[L].Wo, 4, WO_SZ, f);
    for (int L = 0; L < NLAYERS; L++) fread(lw[L].rms_ffn, 4, DIM, f);
    for (int L = 0; L < NLAYERS; L++) fread(lw[L].W1, 4, W1_SZ, f);
    for (int L = 0; L < NLAYERS; L++) fread(lw[L].W2, 4, W2_SZ, f);
    for (int L = 0; L < NLAYERS; L++) fread(lw[L].W3, 4, W3_SZ, f);
    fread(rms_final, 4, DIM, f);
    fclose(f);
    printf("  Loaded pretrained weights\n");
    return true;
}

// Transpose W[rows,cols] → W^T[cols,rows]
static void transpose_weight(float *dst, const float *src, int rows, int cols) {
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            dst[c * rows + r] = src[r * cols + c];
}

// ===== Compile all dynamic kernels (ONCE) =====
static bool compile_dynamic_kernels(DynLayerKernels *dk) {
    NSDictionary *mask_w = @{@"@model_path/weights/mask.bin": @{@"offset":@0, @"data":get_mask_blob()}};

    printf("  Compiling sdpaFwd...\n");
    dk->sdpaFwd = compile_kern_mil_w(gen_sdpa_fwd_dynamic(), mask_w,
        DIM*(SEQ+4*DIM)*4, 6*DIM*SEQ*4);
    if (!dk->sdpaFwd) return false;

    printf("  Compiling ffnW13...\n");
    dk->ffnW13 = compile_kern_mil_w(gen_ffn_w13_dynamic(), @{},
        DIM*(SEQ+2*HIDDEN)*4, 3*HIDDEN*SEQ*4);
    if (!dk->ffnW13) return false;

    printf("  Compiling ffnW2...\n");
    dk->ffnW2 = compile_kern_mil_w(gen_ffn_w2_dynamic(), @{},
        HIDDEN*(SEQ+DIM)*4, DIM*SEQ*4);
    if (!dk->ffnW2) return false;

    printf("  Compiling ffnBwdW2t...\n");
    dk->ffnBwdW2t = compile_kern_mil_w(gen_ffn_bwd_w2t_dynamic(), @{},
        DIM*(SEQ+HIDDEN)*4, HIDDEN*SEQ*4);
    if (!dk->ffnBwdW2t) return false;

    printf("  Compiling ffnBwdW13t...\n");
    dk->ffnBwdW13t = compile_kern_mil_w(gen_ffn_bwd_w13t_dynamic(), @{},
        HIDDEN*(2*SEQ+2*DIM)*4, DIM*SEQ*4);
    if (!dk->ffnBwdW13t) return false;

    printf("  Compiling wotBwd...\n");
    dk->wotBwd = compile_kern_mil_w(gen_wot_dynamic(), @{},
        DIM*(SEQ+DIM)*4, DIM*SEQ*4);
    if (!dk->wotBwd) return false;

    printf("  Compiling sdpaBwd1...\n");
    dk->sdpaBwd1 = compile_kern_mil_w(gen_sdpa_bwd1_noweight(), mask_w,
        4*DIM*SEQ*2, (DIM+2*SCORE_CH)*SEQ*2);
    if (!dk->sdpaBwd1) return false;

    printf("  Compiling sdpaBwd2...\n");
    dk->sdpaBwd2 = compile_kern_mil_w(gen_sdpa_bwd2(), @{},
        (2*SCORE_CH+2*DIM)*SEQ*2, 2*DIM*SEQ*2);
    if (!dk->sdpaBwd2) return false;

    printf("  Compiling qkvBwd...\n");
    dk->qkvBwd = compile_kern_mil_w(gen_qkvb_dynamic(), @{},
        DIM*(3*SEQ+3*DIM)*4, DIM*SEQ*4);
    if (!dk->qkvBwd) return false;

    return true;
}

// ===== Write dynamic weights into IOSurface =====
static void write_sdpa_fwd_input(DynLayerKernels *dk, const float *xnorm,
                                  const float *Wq, const float *Wk, const float *Wv, const float *Wo) {
    IOSurfaceLock(dk->sdpaFwd->ioIn, 0, NULL);
    float *buf = (float*)IOSurfaceGetBaseAddress(dk->sdpaFwd->ioIn);
    int sp = SEQ + 4*DIM;
    for (int d = 0; d < DIM; d++) {
        memcpy(buf + d*sp, xnorm + d*SEQ, SEQ*4);
        memcpy(buf + d*sp + SEQ,       Wq + d*DIM, DIM*4);
        memcpy(buf + d*sp + SEQ+DIM,   Wk + d*DIM, DIM*4);
        memcpy(buf + d*sp + SEQ+2*DIM, Wv + d*DIM, DIM*4);
        memcpy(buf + d*sp + SEQ+3*DIM, Wo + d*DIM, DIM*4);
    }
    IOSurfaceUnlock(dk->sdpaFwd->ioIn, 0, NULL);
}

static void write_ffn_w13_input(DynLayerKernels *dk, const float *xnorm,
                                const float *W1, const float *W3) {
    IOSurfaceLock(dk->ffnW13->ioIn, 0, NULL);
    float *buf = (float*)IOSurfaceGetBaseAddress(dk->ffnW13->ioIn);
    int sp = SEQ + 2*HIDDEN;
    for (int d = 0; d < DIM; d++) {
        memcpy(buf + d*sp, xnorm + d*SEQ, SEQ*4);
        memcpy(buf + d*sp + SEQ,        W1 + d*HIDDEN, HIDDEN*4);
        memcpy(buf + d*sp + SEQ+HIDDEN,  W3 + d*HIDDEN, HIDDEN*4);
    }
    IOSurfaceUnlock(dk->ffnW13->ioIn, 0, NULL);
}

static void write_ffn_w2_input(DynLayerKernels *dk, const float *gate, const float *W2) {
    IOSurfaceLock(dk->ffnW2->ioIn, 0, NULL);
    float *buf = (float*)IOSurfaceGetBaseAddress(dk->ffnW2->ioIn);
    int sp = SEQ + DIM;
    for (int d = 0; d < HIDDEN; d++) {
        memcpy(buf + d*sp, gate + d*SEQ, SEQ*4);
        memcpy(buf + d*sp + SEQ, W2 + d*DIM, DIM*4);
    }
    IOSurfaceUnlock(dk->ffnW2->ioIn, 0, NULL);
}

int main(int argc, char *argv[]) {
    @autoreleasepool {
        setbuf(stdout, NULL);
        ane_init();
        mach_timebase_info(&g_tb);

        int total_steps = 10000;
        float max_lr = 1e-3f;    // LoRA typically uses higher LR
        float adam_b1=0.9f, adam_b2=0.999f, adam_eps=1e-8f;
        int adam_t = 0, start_step = 0;
        int accum_steps = 10;
        int warmup_steps = 50;
        float grad_clip = 1.0f;
        float min_lr_frac = 0.1f;

        bool do_resume = false;
        for (int i=1; i<argc; i++) {
            if (strcmp(argv[i], "--resume") == 0) do_resume = true;
            else if (strcmp(argv[i], "--steps") == 0 && i+1<argc) total_steps = atoi(argv[++i]);
            else if (strcmp(argv[i], "--lr") == 0 && i+1<argc) max_lr = atof(argv[++i]);
            else if (strcmp(argv[i], "--accum") == 0 && i+1<argc) accum_steps = atoi(argv[++i]);
            else if (strcmp(argv[i], "--warmup") == 0 && i+1<argc) warmup_steps = atoi(argv[++i]);
            else if (strcmp(argv[i], "--clip") == 0 && i+1<argc) grad_clip = atof(argv[++i]);
        }
        float lr = max_lr;

        // Allocate per-layer state
        LayerWeights lw[NLAYERS];
        LayerActs acts[NLAYERS];
        LayerGrads grads[NLAYERS];  // Only Wq/Wv used, but struct is convenient
        for (int L=0; L<NLAYERS; L++) {
            lw[L] = layer_weights_alloc();
            acts[L] = layer_acts_alloc();
            grads[L] = layer_grads_alloc();
        }
        float *rms_final = (float*)malloc(DIM*4);
        float *embed = (float*)malloc(VOCAB*DIM*4);

        // LoRA state: frozen weight copies + adapters
        float *frozen_Wq[NLAYERS], *frozen_Wv[NLAYERS];
        LayerLoRA lora[NLAYERS];

        double cum_compile=0, cum_train=0, cum_wall=0;
        int cum_steps=0, cum_batches=0;
        float resume_loss = 0;
        bool resuming = false;

        printf("=== ANE Dynamic LoRA Training: Stories110M ===\n");
        printf("dim=%d hidden=%d heads=%d seq=%d vocab=%d layers=%d\n", DIM, HIDDEN, HEADS, SEQ, VOCAB, NLAYERS);
        printf("LoRA: rank=%d alpha=%.1f trainable_per_layer=%d total=%d\n",
               LORA_RANK, LORA_ALPHA, 2*(DIM*LORA_RANK + DIM*LORA_RANK), 2*2*DIM*LORA_RANK*NLAYERS);
        printf("Accum %d steps, LR=%g\n", accum_steps, max_lr);

        // Load pretrained weights (always needed — LoRA adapts frozen base)
        if (!load_pretrained(lw, rms_final, embed, MODEL_PATH)) {
            printf("FATAL: Cannot load pretrained model\n");
            return 1;
        }

        // Save frozen copies + init LoRA adapters
        srand48(42);
        for (int L=0; L<NLAYERS; L++) {
            frozen_Wq[L] = (float*)malloc(WQ_SZ*4);
            frozen_Wv[L] = (float*)malloc(WQ_SZ*4);
            memcpy(frozen_Wq[L], lw[L].Wq, WQ_SZ*4);
            memcpy(frozen_Wv[L], lw[L].Wv, WQ_SZ*4);
            lora_adapter_init(&lora[L].wq, DIM, LORA_RANK);
            lora_adapter_init(&lora[L].wv, DIM, LORA_RANK);
        }

        // Resume from LoRA checkpoint
        if (do_resume) {
            resuming = lora_load(LORA_CKPT_PATH, &start_step, &total_steps, &lr, &resume_loss,
                &cum_compile, &cum_train, &cum_wall, &cum_steps, &cum_batches, &adam_t,
                lora, NLAYERS, DIM, LORA_RANK);
            if (resuming) {
                printf("[RESUMED step %d, loss=%.4f]\n", start_step, resume_loss);
                // Re-merge Wq/Wv from frozen + loaded LoRA
                for (int L=0; L<NLAYERS; L++) {
                    lora_merge(lw[L].Wq, frozen_Wq[L], &lora[L].wq, DIM, LORA_RANK, LORA_ALPHA);
                    lora_merge(lw[L].Wv, frozen_Wv[L], &lora[L].wv, DIM, LORA_RANK, LORA_ALPHA);
                }
            }
        }

        // Precompute transposed weights for forward pass kernels
        float *Wqt_buf[NLAYERS], *Wkt_buf[NLAYERS], *Wvt_buf[NLAYERS], *Wot_buf[NLAYERS];
        float *W1t_buf[NLAYERS], *W2t_buf[NLAYERS], *W3t_buf[NLAYERS];
        for (int L=0; L<NLAYERS; L++) {
            Wqt_buf[L]=(float*)malloc(WQ_SZ*4); Wkt_buf[L]=(float*)malloc(WQ_SZ*4);
            Wvt_buf[L]=(float*)malloc(WQ_SZ*4); Wot_buf[L]=(float*)malloc(WO_SZ*4);
            W1t_buf[L]=(float*)malloc(W1_SZ*4); W2t_buf[L]=(float*)malloc(W2_SZ*4);
            W3t_buf[L]=(float*)malloc(W3_SZ*4);
            transpose_weight(Wqt_buf[L], lw[L].Wq, DIM, DIM);
            transpose_weight(Wkt_buf[L], lw[L].Wk, DIM, DIM);
            transpose_weight(Wvt_buf[L], lw[L].Wv, DIM, DIM);
            transpose_weight(Wot_buf[L], lw[L].Wo, DIM, DIM);
            transpose_weight(W1t_buf[L], lw[L].W1, HIDDEN, DIM);
            transpose_weight(W2t_buf[L], lw[L].W2, DIM, HIDDEN);
            transpose_weight(W3t_buf[L], lw[L].W3, HIDDEN, DIM);
        }

        // mmap token data
        int data_fd = open(DATA_PATH, O_RDONLY);
        if (data_fd < 0) { printf("Cannot open %s\n", DATA_PATH); return 1; }
        struct stat st; fstat(data_fd, &st);
        size_t data_len = st.st_size;
        uint16_t *token_data = (uint16_t*)mmap(NULL, data_len, PROT_READ, MAP_PRIVATE, data_fd, 0);
        if (token_data == MAP_FAILED) { printf("mmap failed\n"); return 1; }
        size_t n_tokens = data_len / 2;
        printf("Token data: %zu tokens (%.1f MB)\n", n_tokens, data_len/1e6);

        // Vocab compaction
        VocabMap vm = vocab_map_build(token_data, n_tokens, VOCAB);
        int CV = vm.compact_vocab;
        printf("Vocab compaction: %d → %d active tokens (%.1fx reduction)\n", VOCAB, CV, (float)VOCAB/CV);
        float *cembed = vocab_compact_embed(embed, &vm, DIM);

        // ===== Compile all kernels ONCE =====
        printf("Compiling %d dynamic kernels (one-time)...\n", 9);
        uint64_t tc = mach_absolute_time();
        DynLayerKernels dk;
        if (!compile_dynamic_kernels(&dk)) {
            printf("Compilation failed!\n"); return 1;
        }
        double compile_ms = tb_ms(mach_absolute_time() - tc);
        printf("Compiled 9 kernels in %.0fms (shared across all %d layers)\n\n", compile_ms, NLAYERS);

        // Gradient + work buffers
        float *dy = (float*)malloc(SEQ*DIM*4);
        float *dffn = (float*)malloc(SEQ*DIM*4);
        float *dx_ffn = (float*)malloc(SEQ*DIM*4);
        float *dx2 = (float*)malloc(SEQ*DIM*4);
        float *dx_attn = (float*)malloc(SEQ*DIM*4);
        float *dq = (float*)malloc(SEQ*DIM*4);
        float *dk_buf = (float*)malloc(SEQ*DIM*4);
        float *dv = (float*)malloc(SEQ*DIM*4);
        float *x_cur = (float*)malloc(SEQ*DIM*4);
        float *x_final = (float*)malloc(SEQ*DIM*4);
        float *xnorm_buf = (float*)malloc(SEQ*DIM*4);
        float *logits = (float*)malloc(SEQ*CV*4);
        float *dlogits = (float*)malloc(SEQ*CV*4);
        float *gate_buf = (float*)malloc(SEQ*HIDDEN*4);
        float *dh1 = (float*)malloc(SEQ*HIDDEN*4);
        float *dh3 = (float*)malloc(SEQ*HIDDEN*4);
        float *dsilu = (float*)malloc(SEQ*HIDDEN*4);
        float *silu_tmp = (float*)malloc(SEQ*HIDDEN*4);
        float *silu_tmp2 = (float*)malloc(SEQ*HIDDEN*4);

        // dW dispatch queue — only Wq/Wv GEMMs now
        dispatch_queue_t dw_q = dispatch_queue_create("dw_cblas", DISPATCH_QUEUE_SERIAL);
        dispatch_group_t dw_grp = dispatch_group_create();

        float last_loss = 999.0f;
        double total_train_ms = 0;
        int total_steps_done = 0;
        uint64_t t_wall_start = mach_absolute_time();
        srand48(42 + start_step);

        for (int step = start_step; step < total_steps; step++) {
            uint64_t t0, t1, t_step = mach_absolute_time();

            // LoRA merge before forward: W_eff = W_frozen + scale * B @ A
            for (int L=0; L<NLAYERS; L++) {
                lora_merge(lw[L].Wq, frozen_Wq[L], &lora[L].wq, DIM, LORA_RANK, LORA_ALPHA);
                lora_merge(lw[L].Wv, frozen_Wv[L], &lora[L].wv, DIM, LORA_RANK, LORA_ALPHA);
                transpose_weight(Wqt_buf[L], lw[L].Wq, DIM, DIM);
                transpose_weight(Wvt_buf[L], lw[L].Wv, DIM, DIM);
            }

            // Sample data
            size_t max_pos = n_tokens - SEQ - 1;
            size_t pos = (size_t)(drand48() * max_pos);
            uint16_t *input_tokens = token_data + pos;
            uint16_t *target_tokens_raw = token_data + pos + 1;
            uint16_t ctargets[SEQ];
            for (int t = 0; t < SEQ; t++) ctargets[t] = (uint16_t)vm.full_to_compact[target_tokens_raw[t]];

            // Embedding lookup (frozen)
            embed_lookup(x_cur, embed, input_tokens, DIM, SEQ);

            double t_rms=0, t_ane_fwd=0, t_io_fwd=0, t_cblas_wait=0;
            double t_ane_bwd=0, t_io_bwd=0, t_silu=0, t_rms_bwd=0, t_cls=0, t_merge=0;

            // ===== FORWARD (12 layers) =====
            for (int L=0; L<NLAYERS; L++) {
                LayerActs *ac = &acts[L];
                memcpy(ac->layer_in, x_cur, SEQ*DIM*4);

                // RMSNorm1 (CPU)
                t0 = mach_absolute_time();
                rmsnorm(xnorm_buf, x_cur, lw[L].rms_att, DIM, SEQ);
                memcpy(ac->xnorm, xnorm_buf, SEQ*DIM*4);
                t_rms += tb_ms(mach_absolute_time() - t0);

                // Wait for any pending dW cblas
                t0 = mach_absolute_time();
                dispatch_group_wait(dw_grp, DISPATCH_TIME_FOREVER);
                t_cblas_wait += tb_ms(mach_absolute_time() - t0);

                // SDPA forward (ANE): xnorm + Wqt,Wkt,Wvt,Wot → o_out,Q,K,V,attn_out
                t0 = mach_absolute_time();
                write_sdpa_fwd_input(&dk, xnorm_buf, Wqt_buf[L], Wkt_buf[L], Wvt_buf[L], Wot_buf[L]);
                t_io_fwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                ane_eval(dk.sdpaFwd);
                t_ane_fwd += tb_ms(mach_absolute_time() - t0);

                // Read output: [1, 6*DIM, 1, SEQ] fp32
                t0 = mach_absolute_time();
                IOSurfaceLock(dk.sdpaFwd->ioOut, kIOSurfaceLockReadOnly, NULL);
                float *fwd_out = (float*)IOSurfaceGetBaseAddress(dk.sdpaFwd->ioOut);
                memcpy(ac->o_out,    fwd_out + 0*DIM*SEQ, DIM*SEQ*4);
                memcpy(ac->Q,       fwd_out + 1*DIM*SEQ, DIM*SEQ*4);
                memcpy(ac->K,       fwd_out + 2*DIM*SEQ, DIM*SEQ*4);
                memcpy(ac->V,       fwd_out + 3*DIM*SEQ, DIM*SEQ*4);
                memcpy(ac->attn_out, fwd_out + 4*DIM*SEQ, DIM*SEQ*4);
                IOSurfaceUnlock(dk.sdpaFwd->ioOut, kIOSurfaceLockReadOnly, NULL);
                t_io_fwd += tb_ms(mach_absolute_time() - t0);

                // Residual: x2 = x_cur + o_out
                vDSP_vadd(x_cur, 1, ac->o_out, 1, ac->x2, 1, (vDSP_Length)(SEQ*DIM));

                // RMSNorm2 (CPU)
                t0 = mach_absolute_time();
                rmsnorm(xnorm_buf, ac->x2, lw[L].rms_ffn, DIM, SEQ);
                memcpy(ac->x2norm, xnorm_buf, SEQ*DIM*4);
                t_rms += tb_ms(mach_absolute_time() - t0);

                // FFN W1+W3 (ANE)
                t0 = mach_absolute_time();
                write_ffn_w13_input(&dk, xnorm_buf, W1t_buf[L], W3t_buf[L]);
                t_io_fwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                ane_eval(dk.ffnW13);
                t_ane_fwd += tb_ms(mach_absolute_time() - t0);

                t0 = mach_absolute_time();
                IOSurfaceLock(dk.ffnW13->ioOut, kIOSurfaceLockReadOnly, NULL);
                float *ffn13_out = (float*)IOSurfaceGetBaseAddress(dk.ffnW13->ioOut);
                memcpy(ac->h1,       ffn13_out,                   HIDDEN*SEQ*4);
                memcpy(ac->h3,       ffn13_out + HIDDEN*SEQ,      HIDDEN*SEQ*4);
                memcpy(gate_buf,     ffn13_out + 2*HIDDEN*SEQ,    HIDDEN*SEQ*4);
                memcpy(ac->silu_out, gate_buf,                    HIDDEN*SEQ*4);
                IOSurfaceUnlock(dk.ffnW13->ioOut, kIOSurfaceLockReadOnly, NULL);
                t_io_fwd += tb_ms(mach_absolute_time() - t0);

                // FFN W2 (ANE)
                t0 = mach_absolute_time();
                write_ffn_w2_input(&dk, gate_buf, W2t_buf[L]);
                t_io_fwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                ane_eval(dk.ffnW2);
                t_ane_fwd += tb_ms(mach_absolute_time() - t0);

                t0 = mach_absolute_time();
                IOSurfaceLock(dk.ffnW2->ioOut, kIOSurfaceLockReadOnly, NULL);
                memcpy(ac->ffn_out, (float*)IOSurfaceGetBaseAddress(dk.ffnW2->ioOut), DIM*SEQ*4);
                IOSurfaceUnlock(dk.ffnW2->ioOut, kIOSurfaceLockReadOnly, NULL);
                t_io_fwd += tb_ms(mach_absolute_time() - t0);

                // Residual: x_cur = x2 + ffn_out
                vDSP_vadd(ac->x2, 1, ac->ffn_out, 1, x_cur, 1, (vDSP_Length)(SEQ*DIM));
            }

            // Final RMSNorm + classifier + loss (CPU, all frozen)
            t0 = mach_absolute_time();
            rmsnorm(x_final, x_cur, rms_final, DIM, SEQ);
            t_rms += tb_ms(mach_absolute_time() - t0);
            t0 = mach_absolute_time();
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        CV, SEQ, DIM, 1.0f, cembed, DIM, x_final, SEQ, 0.0f, logits, SEQ);
            float loss = cross_entropy_loss(dlogits, logits, ctargets, CV, SEQ);
            t_cls += tb_ms(mach_absolute_time() - t0);
            last_loss = loss;

            // ===== BACKWARD =====
            // Classifier backward (frozen — compute dx only, skip dEmbed)
            t0 = mach_absolute_time();
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        DIM, SEQ, CV, 1.0f, cembed, DIM, dlogits, SEQ, 0.0f, dy, SEQ);
            t_cls += tb_ms(mach_absolute_time() - t0);

            // Skip dEmbed — embedding is frozen

            // Final RMSNorm backward (frozen — compute dx only, skip dw)
            float *dx_rms_final = (float*)calloc(SEQ*DIM, 4);
            float drms_final_dummy[DIM];
            memset(drms_final_dummy, 0, DIM*4);
            rmsnorm_bwd(dx_rms_final, drms_final_dummy, dy, x_cur, rms_final, DIM, SEQ);
            memcpy(dy, dx_rms_final, SEQ*DIM*4);
            free(dx_rms_final);

            // ===== BACKWARD (12 layers, reverse) =====
            for (int L=NLAYERS-1; L>=0; L--) {
                LayerActs *ac = &acts[L];
                LayerGrads *gr = &grads[L];
                memcpy(dffn, dy, SEQ*DIM*4);

                // FFN backward: dffn @ W2^T → dsilu_raw (ANE)
                t0 = mach_absolute_time();
                io_write_dyn(dk.ffnBwdW2t->ioIn, dffn, DIM, SEQ, lw[L].W2, HIDDEN);
                t_io_bwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                ane_eval(dk.ffnBwdW2t);
                t_ane_bwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                io_read_dyn(dk.ffnBwdW2t->ioOut, dsilu, HIDDEN, SEQ);
                t_io_bwd += tb_ms(mach_absolute_time() - t0);

                // SiLU derivative (vectorized)
                t0 = mach_absolute_time();
                {
                    int n = HIDDEN*SEQ;
                    float minus1 = -1.0f, one = 1.0f;
                    vDSP_vsmul(ac->h1, 1, &minus1, silu_tmp, 1, (vDSP_Length)n);
                    vvexpf(silu_tmp, silu_tmp, &n);
                    vDSP_vsadd(silu_tmp, 1, &one, silu_tmp, 1, (vDSP_Length)n);
                    vvrecf(silu_tmp, silu_tmp, &n);  // silu_tmp = sig
                    vDSP_vmul(ac->h1, 1, silu_tmp, 1, dh3, 1, (vDSP_Length)n);
                    vDSP_vmul(dsilu, 1, dh3, 1, dh3, 1, (vDSP_Length)n);
                    vDSP_vsadd(silu_tmp, 1, &minus1, silu_tmp2, 1, (vDSP_Length)n);
                    vDSP_vneg(silu_tmp2, 1, silu_tmp2, 1, (vDSP_Length)n);
                    vDSP_vmul(ac->h1, 1, silu_tmp2, 1, silu_tmp2, 1, (vDSP_Length)n);
                    vDSP_vsadd(silu_tmp2, 1, &one, silu_tmp2, 1, (vDSP_Length)n);
                    vDSP_vmul(silu_tmp, 1, silu_tmp2, 1, silu_tmp2, 1, (vDSP_Length)n);
                    vDSP_vmul(dsilu, 1, ac->h3, 1, dh1, 1, (vDSP_Length)n);
                    vDSP_vmul(dh1, 1, silu_tmp2, 1, dh1, 1, (vDSP_Length)n);
                }
                t_silu += tb_ms(mach_absolute_time() - t0);

                // dh1@W1^T + dh3@W3^T → dx_ffn (ANE)
                t0 = mach_absolute_time();
                {
                    IOSurfaceLock(dk.ffnBwdW13t->ioIn, 0, NULL);
                    float *buf = (float*)IOSurfaceGetBaseAddress(dk.ffnBwdW13t->ioIn);
                    int sp = 2*SEQ + 2*DIM;
                    for (int d = 0; d < HIDDEN; d++) {
                        memcpy(buf + d*sp,            dh1 + d*SEQ, SEQ*4);
                        memcpy(buf + d*sp + SEQ,      dh3 + d*SEQ, SEQ*4);
                        memcpy(buf + d*sp + 2*SEQ,        lw[L].W1 + d*DIM, DIM*4);
                        memcpy(buf + d*sp + 2*SEQ + DIM,  lw[L].W3 + d*DIM, DIM*4);
                    }
                    IOSurfaceUnlock(dk.ffnBwdW13t->ioIn, 0, NULL);
                }
                t_io_bwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                ane_eval(dk.ffnBwdW13t);
                t_ane_bwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                io_read_dyn(dk.ffnBwdW13t->ioOut, dx_ffn, DIM, SEQ);
                t_io_bwd += tb_ms(mach_absolute_time() - t0);

                // SKIP: dW2, dW1, dW3 GEMMs — frozen weights

                // RMSNorm2 backward (frozen — compute dx only, skip dw)
                t0 = mach_absolute_time();
                memset(dx2, 0, SEQ*DIM*4);
                float drms_ffn_dummy[DIM];
                memset(drms_ffn_dummy, 0, DIM*4);
                rmsnorm_bwd(dx2, drms_ffn_dummy, dx_ffn, ac->x2, lw[L].rms_ffn, DIM, SEQ);
                for(int i=0;i<SEQ*DIM;i++) dx2[i] += dy[i];
                t_rms_bwd += tb_ms(mach_absolute_time() - t0);

                // Wo^T backward (ANE): dx2 @ Wo^T → da
                t0 = mach_absolute_time();
                io_write_dyn(dk.wotBwd->ioIn, dx2, DIM, SEQ, lw[L].Wo, DIM);
                t_io_bwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                ane_eval(dk.wotBwd);
                t_ane_bwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                float *da_buf = (float*)malloc(SEQ*DIM*4);
                io_read_dyn(dk.wotBwd->ioOut, da_buf, DIM, SEQ);
                t_io_bwd += tb_ms(mach_absolute_time() - t0);

                // SKIP: dWo GEMM — frozen

                // SDPA backward part 1 (ANE, fp16): Q,K,V,da → dV,probs,dp
                t0 = mach_absolute_time();
                io_write_fp16_at(dk.sdpaBwd1->ioIn, 0,     ac->Q,  DIM, SEQ);
                io_write_fp16_at(dk.sdpaBwd1->ioIn, DIM,   ac->K,  DIM, SEQ);
                io_write_fp16_at(dk.sdpaBwd1->ioIn, 2*DIM, ac->V,  DIM, SEQ);
                io_write_fp16_at(dk.sdpaBwd1->ioIn, 3*DIM, da_buf, DIM, SEQ);
                free(da_buf);
                t_io_bwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                ane_eval(dk.sdpaBwd1);
                t_ane_bwd += tb_ms(mach_absolute_time() - t0);

                // SDPA backward part 2: probs,dp,Q,K → dQ,dK
                t0 = mach_absolute_time();
                io_copy(dk.sdpaBwd2->ioIn, 0, dk.sdpaBwd1->ioOut, DIM, 2*SCORE_CH, SEQ);
                io_write_fp16_at(dk.sdpaBwd2->ioIn, 2*SCORE_CH,     ac->Q, DIM, SEQ);
                io_write_fp16_at(dk.sdpaBwd2->ioIn, 2*SCORE_CH+DIM, ac->K, DIM, SEQ);
                t_io_bwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                ane_eval(dk.sdpaBwd2);
                t_ane_bwd += tb_ms(mach_absolute_time() - t0);

                t0 = mach_absolute_time();
                io_read_fp16(dk.sdpaBwd2->ioOut, dq, 0,   DIM, SEQ);
                io_read_fp16(dk.sdpaBwd2->ioOut, dk_buf, DIM, DIM, SEQ);
                io_read_fp16(dk.sdpaBwd1->ioOut, dv, 0, DIM, SEQ);
                t_io_bwd += tb_ms(mach_absolute_time() - t0);

                // dWq and dWv async (KEEP — LoRA targets)
                // SKIP: dWk — frozen
                t0 = mach_absolute_time();
                float *capt_dq = (float*)malloc(SEQ*DIM*4); memcpy(capt_dq, dq, SEQ*DIM*4);
                float *capt_dv = (float*)malloc(SEQ*DIM*4); memcpy(capt_dv, dv, SEQ*DIM*4);
                float *capt_xn = (float*)malloc(SEQ*DIM*4); memcpy(capt_xn, ac->xnorm, SEQ*DIM*4);
                t0 = mach_absolute_time();
                dispatch_group_async(dw_grp, dw_q, ^{
                    // dWq += dq @ xnorm^T
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, DIM, DIM, SEQ,
                                1.0f, capt_dq, SEQ, capt_xn, SEQ, 1.0f, gr->Wq, DIM);
                    // dWv += dv @ xnorm^T
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, DIM, DIM, SEQ,
                                1.0f, capt_dv, SEQ, capt_xn, SEQ, 1.0f, gr->Wv, DIM);
                    free(capt_dq); free(capt_dv); free(capt_xn);
                });

                // QKV backward (ANE): dq,dk,dv @ Wq^T,Wk^T,Wv^T → dx_attn
                t0 = mach_absolute_time();
                {
                    IOSurfaceLock(dk.qkvBwd->ioIn, 0, NULL);
                    float *buf = (float*)IOSurfaceGetBaseAddress(dk.qkvBwd->ioIn);
                    int sp = 3*SEQ + 3*DIM;
                    for (int d = 0; d < DIM; d++) {
                        memcpy(buf + d*sp,             dq     + d*SEQ, SEQ*4);
                        memcpy(buf + d*sp + SEQ,       dk_buf + d*SEQ, SEQ*4);
                        memcpy(buf + d*sp + 2*SEQ,     dv     + d*SEQ, SEQ*4);
                        memcpy(buf + d*sp + 3*SEQ,         lw[L].Wq + d*DIM, DIM*4);
                        memcpy(buf + d*sp + 3*SEQ+DIM,     lw[L].Wk + d*DIM, DIM*4);
                        memcpy(buf + d*sp + 3*SEQ+2*DIM,   lw[L].Wv + d*DIM, DIM*4);
                    }
                    IOSurfaceUnlock(dk.qkvBwd->ioIn, 0, NULL);
                }
                t_io_bwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                ane_eval(dk.qkvBwd);
                t_ane_bwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                io_read_dyn(dk.qkvBwd->ioOut, dx_attn, DIM, SEQ);
                t_io_bwd += tb_ms(mach_absolute_time() - t0);

                // RMSNorm1 backward (frozen — compute dx only, skip dw)
                t0 = mach_absolute_time();
                float *dx_rms1 = (float*)calloc(SEQ*DIM, 4);
                float drms_att_dummy[DIM];
                memset(drms_att_dummy, 0, DIM*4);
                rmsnorm_bwd(dx_rms1, drms_att_dummy, dx_attn, ac->layer_in, lw[L].rms_att, DIM, SEQ);
                for(int i=0;i<SEQ*DIM;i++) dy[i] = dx_rms1[i] + dx2[i];
                free(dx_rms1);
                t_rms_bwd += tb_ms(mach_absolute_time() - t0);
            }

            // Skip embedding backward — frozen
            dispatch_group_wait(dw_grp, DISPATCH_TIME_FOREVER);

            double step_ms = tb_ms(mach_absolute_time() - t_step);
            total_train_ms += step_ms;
            total_steps_done++;

            if (step % 10 == 0 || step == start_step) {
                // LoRA norm diagnostics
                float a_norm_sq = 0, b_norm_sq = 0;
                for (int L=0; L<NLAYERS; L++) {
                    float s;
                    size_t a_sz = (size_t)LORA_RANK*DIM, b_sz = (size_t)DIM*LORA_RANK;
                    vDSP_dotpr(lora[L].wq.A,1,lora[L].wq.A,1,&s,(vDSP_Length)a_sz); a_norm_sq+=s;
                    vDSP_dotpr(lora[L].wv.A,1,lora[L].wv.A,1,&s,(vDSP_Length)a_sz); a_norm_sq+=s;
                    vDSP_dotpr(lora[L].wq.B,1,lora[L].wq.B,1,&s,(vDSP_Length)b_sz); b_norm_sq+=s;
                    vDSP_dotpr(lora[L].wv.B,1,lora[L].wv.B,1,&s,(vDSP_Length)b_sz); b_norm_sq+=s;
                }
                printf("  timing: ane_fwd=%.1f io_fwd=%.1f rms=%.1f ane_bwd=%.1f io_bwd=%.1f silu=%.1f rms_bwd=%.1f cls=%.1f cblas_wait=%.1f\n",
                       t_ane_fwd, t_io_fwd, t_rms, t_ane_bwd, t_io_bwd, t_silu, t_rms_bwd, t_cls, t_cblas_wait);
                printf("step %-4d loss=%.4f  lr=%.2e  |A|=%.4f |B|=%.6f  %.1fms/step\n",
                       step, loss, lr, sqrtf(a_norm_sq), sqrtf(b_norm_sq), step_ms);
                // JSON telemetry
                fprintf(stderr, "{\"type\":\"lora_dyn\",\"step\":%d,\"loss\":%.6f,\"lr\":%.2e,\"a_norm\":%.6f,\"b_norm\":%.6f,\"ms\":%.1f,\"compile_count\":%d}\n",
                        step, loss, lr, sqrtf(a_norm_sq), sqrtf(b_norm_sq), step_ms, g_compile_count);
            }

            // LoRA Adam update every accum_steps
            if ((step+1) % accum_steps == 0 || step == total_steps-1) {
                dispatch_group_wait(dw_grp, DISPATCH_TIME_FOREVER);
                float gsc = 1.0f / accum_steps;
                adam_t++;

                // Scale only Wq/Wv grads
                for (int L=0; L<NLAYERS; L++) {
                    for(size_t i=0;i<WQ_SZ;i++) { grads[L].Wq[i] *= gsc; grads[L].Wv[i] *= gsc; }
                }

                // Gradient norm (Wq/Wv only)
                float grad_norm_sq = 0;
                for (int L=0; L<NLAYERS; L++) {
                    float s;
                    vDSP_dotpr(grads[L].Wq,1,grads[L].Wq,1,&s,(vDSP_Length)WQ_SZ); grad_norm_sq+=s;
                    vDSP_dotpr(grads[L].Wv,1,grads[L].Wv,1,&s,(vDSP_Length)WQ_SZ); grad_norm_sq+=s;
                }
                float grad_norm = sqrtf(grad_norm_sq);
                if ((step+1) % 10 == 0) printf("  grad_norm=%.4f\n", grad_norm);

                // Gradient clipping
                if (grad_clip > 0 && grad_norm > grad_clip) {
                    float clip_scale = grad_clip / grad_norm;
                    for (int L=0; L<NLAYERS; L++) {
                        vDSP_vsmul(grads[L].Wq,1,&clip_scale,grads[L].Wq,1,(vDSP_Length)WQ_SZ);
                        vDSP_vsmul(grads[L].Wv,1,&clip_scale,grads[L].Wv,1,(vDSP_Length)WQ_SZ);
                    }
                }

                // Cosine LR schedule with warmup
                if (step < warmup_steps) {
                    lr = max_lr * ((float)(step + 1)) / warmup_steps;
                } else {
                    float decay_ratio = (float)(step - warmup_steps) / (float)(total_steps - warmup_steps);
                    float min_lr = max_lr * min_lr_frac;
                    lr = min_lr + 0.5f * (1.0f + cosf(M_PI * decay_ratio)) * (max_lr - min_lr);
                }

                // Extract LoRA gradients from full dWq/dWv → update LoRA A/B
                for (int L=0; L<NLAYERS; L++) {
                    lora_extract_grads(&lora[L].wq, grads[L].Wq, DIM, LORA_RANK, LORA_ALPHA);
                    lora_extract_grads(&lora[L].wv, grads[L].Wv, DIM, LORA_RANK, LORA_ALPHA);
                    adam_update(lora[L].wq.A, lora[L].wq.dA, &lora[L].wq.sA, adam_t, lr, adam_b1, adam_b2, adam_eps);
                    adam_update(lora[L].wq.B, lora[L].wq.dB, &lora[L].wq.sB, adam_t, lr, adam_b1, adam_b2, adam_eps);
                    adam_update(lora[L].wv.A, lora[L].wv.dA, &lora[L].wv.sA, adam_t, lr, adam_b1, adam_b2, adam_eps);
                    adam_update(lora[L].wv.B, lora[L].wv.dB, &lora[L].wv.sB, adam_t, lr, adam_b1, adam_b2, adam_eps);
                }

                // Zero only Wq/Wv grads
                for (int L=0; L<NLAYERS; L++) {
                    memset(grads[L].Wq, 0, WQ_SZ*4);
                    memset(grads[L].Wv, 0, WQ_SZ*4);
                }

                // Checkpoint
                if ((step+1) % 100 == 0) {
                    double wall = tb_ms(mach_absolute_time() - t_wall_start);
                    lora_save(LORA_CKPT_PATH, step+1, total_steps, lr, last_loss,
                        compile_ms+cum_compile, total_train_ms+cum_train, wall+cum_wall,
                        total_steps_done+cum_steps, (total_steps_done+cum_steps)/accum_steps+cum_batches,
                        adam_t, lora, NLAYERS, DIM, LORA_RANK, LORA_ALPHA);
                    printf("  [checkpoint saved: step %d]\n", step+1);
                }
            }
        }

        // Report
        double wall = tb_ms(mach_absolute_time() - t_wall_start);
        printf("\n=== LoRA Dynamic Efficiency Report ===\n");
        printf("Total steps:  %d\n", total_steps_done);
        printf("Compile:      %.0fms (one-time, %.1f%%)\n", compile_ms, 100*compile_ms/(wall+cum_wall));
        printf("Train time:   %.0fms (%.1fms/step)\n", total_train_ms, total_train_ms/total_steps_done);
        printf("Wall time:    %.1fs\n", (wall+cum_wall)/1000);
        printf("Compile count: %d (should be 9)\n", g_compile_count);
        printf("LoRA params:  %d (rank=%d, alpha=%.1f)\n", 2*2*DIM*LORA_RANK*NLAYERS, LORA_RANK, LORA_ALPHA);
        printf("dW GEMMs/layer: 2 (Wq, Wv) — skipped 5 frozen\n");

        // Cleanup
        for (int L=0; L<NLAYERS; L++) {
            layer_weights_free(&lw[L]);
            layer_acts_free(&acts[L]); layer_grads_free(&grads[L]);
            free(Wqt_buf[L]); free(Wkt_buf[L]); free(Wvt_buf[L]); free(Wot_buf[L]);
            free(W1t_buf[L]); free(W2t_buf[L]); free(W3t_buf[L]);
            free(frozen_Wq[L]); free(frozen_Wv[L]);
            lora_adapter_free(&lora[L].wq);
            lora_adapter_free(&lora[L].wv);
        }
        free_kern(dk.sdpaFwd); free_kern(dk.ffnW13); free_kern(dk.ffnW2);
        free_kern(dk.ffnBwdW2t); free_kern(dk.ffnBwdW13t); free_kern(dk.wotBwd);
        free_kern(dk.sdpaBwd1); free_kern(dk.sdpaBwd2); free_kern(dk.qkvBwd);
        free(cembed);
        munmap(token_data, data_len); close(data_fd);
    }
    return 0;
}
