// train_lora_flex.m — Partial-recompile LoRA fine-tuning for ANE
// Key insight: only 2 of 6 kernels per layer have LoRA-changing weights (fwdAttn, qkvBwd).
// The other 4 (fwdFFN, ffnBwd, sdpaBwd1, sdpaBwd2) have frozen weights — compile once, reuse forever.
// This cuts per-batch compiles from 60 to 24, giving ~2.5x wall-time improvement.
#include "stories_io.h"
#include "stories_mil.h"
#include "stories_cpu_ops.h"
#include "lora.h"

#define LORA_CKPT_PATH "ane_lora_ckpt.bin"
#define MODEL_PATH "../../assets/models/stories110M.bin"
#define DATA_PATH "tinystories_data00.bin"

// Partial recompile: only 2 kernels per layer change each batch
#define LORA_RECOMPILE_KERNELS (2 * NLAYERS)   // fwdAttn + qkvBwd = 24
#define MAX_COMPILES_FLEX 250                    // Higher budget since fewer compiles per batch

// ===== Weight loading from llama2.c format =====
static bool load_pretrained(LayerWeights *lw, float *rms_final, float *embed, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { printf("Cannot open %s\n", path); return false; }
    Llama2Config cfg;
    fread(&cfg, sizeof(cfg), 1, f);
    printf("  Model config: dim=%d hidden=%d layers=%d heads=%d vocab=%d seq=%d\n",
           cfg.dim, cfg.hidden_dim, cfg.n_layers, cfg.n_heads, abs(cfg.vocab_size), cfg.seq_len);
    if (cfg.dim != DIM || cfg.hidden_dim != HIDDEN || cfg.n_layers != NLAYERS) {
        printf("  ERROR: Config mismatch! Expected dim=%d hidden=%d layers=%d\n", DIM, HIDDEN, NLAYERS);
        fclose(f); return false;
    }
    int V = abs(cfg.vocab_size);
    bool shared = cfg.vocab_size > 0;
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
    printf("  Loaded pretrained weights (%s)\n", shared ? "shared embed/cls" : "separate cls");
    return true;
}

// ===== Compile only the LoRA-changing kernels (fwdAttn + qkvBwd) =====
static bool compile_lora_kernels(LayerKernels *lk, LayerWeights *w) {
    lk->fwdAttn = compile_kern_mil_w(gen_sdpa_fwd_taps(), (@{
        @"@model_path/weights/rms1.bin": @{@"offset":@0, @"data":build_blob(w->rms_att,1,DIM)},
        @"@model_path/weights/wq.bin":   @{@"offset":@0, @"data":build_blob(w->Wq,DIM,DIM)},
        @"@model_path/weights/wk.bin":   @{@"offset":@0, @"data":build_blob(w->Wk,DIM,DIM)},
        @"@model_path/weights/wv.bin":   @{@"offset":@0, @"data":build_blob(w->Wv,DIM,DIM)},
        @"@model_path/weights/wo.bin":   @{@"offset":@0, @"data":build_blob(w->Wo,DIM,DIM)},
        @"@model_path/weights/mask.bin": @{@"offset":@0, @"data":get_mask_blob()},
    }), DIM*SEQ*2, 6*DIM*SEQ*2);

    lk->qkvBwd = compile_kern_mil_w(gen_qkvb(), (@{
        @"@model_path/weights/wqt.bin": @{@"offset":@0, @"data":build_blob_t(w->Wq,DIM,DIM)},
        @"@model_path/weights/wkt.bin": @{@"offset":@0, @"data":build_blob_t(w->Wk,DIM,DIM)},
        @"@model_path/weights/wvt.bin": @{@"offset":@0, @"data":build_blob_t(w->Wv,DIM,DIM)},
    }), 3*DIM*SEQ*2, DIM*SEQ*2);

    return lk->fwdAttn && lk->qkvBwd;
}

// ===== Compile the static (frozen) kernels — called once at startup =====
static bool compile_static_kernels(LayerKernels *lk, LayerWeights *w) {
    lk->fwdFFN = compile_kern_mil_w(gen_ffn_fwd_taps(), (@{
        @"@model_path/weights/rms2.bin": @{@"offset":@0, @"data":build_blob(w->rms_ffn,1,DIM)},
        @"@model_path/weights/w1.bin":   @{@"offset":@0, @"data":build_blob(w->W1,HIDDEN,DIM)},
        @"@model_path/weights/w3.bin":   @{@"offset":@0, @"data":build_blob(w->W3,HIDDEN,DIM)},
        @"@model_path/weights/w2.bin":   @{@"offset":@0, @"data":build_blob(w->W2,DIM,HIDDEN)},
    }), DIM*SEQ*2, (2*DIM+3*HIDDEN)*SEQ*2);

    lk->ffnBwd = compile_kern_mil_w(gen_ffn_bwd(), (@{
        @"@model_path/weights/w2t.bin": @{@"offset":@0, @"data":build_blob_t(w->W2,DIM,HIDDEN)},
        @"@model_path/weights/w1t.bin": @{@"offset":@0, @"data":build_blob_t(w->W1,HIDDEN,DIM)},
        @"@model_path/weights/w3t.bin": @{@"offset":@0, @"data":build_blob_t(w->W3,HIDDEN,DIM)},
    }), (DIM+2*HIDDEN)*SEQ*2, (DIM+2*HIDDEN)*SEQ*2);

    lk->sdpaBwd1 = compile_kern_mil_w(gen_sdpa_bwd1(), (@{
        @"@model_path/weights/mask.bin": @{@"offset":@0, @"data":get_mask_blob()},
        @"@model_path/weights/wot.bin":  @{@"offset":@0, @"data":build_blob_t(w->Wo,DIM,DIM)},
    }), 4*DIM*SEQ*2, (DIM+2*SCORE_CH)*SEQ*2);

    lk->sdpaBwd2 = compile_kern_mil_w(gen_sdpa_bwd2(), @{},
        (2*SCORE_CH+2*DIM)*SEQ*2, 2*DIM*SEQ*2);

    return lk->fwdFFN && lk->ffnBwd && lk->sdpaBwd1 && lk->sdpaBwd2;
}

static void free_lora_kernels(LayerKernels *lk) {
    free_kern(lk->fwdAttn); free_kern(lk->qkvBwd);
    lk->fwdAttn = lk->qkvBwd = NULL;
}

static void free_static_kernels(LayerKernels *lk) {
    free_kern(lk->fwdFFN); free_kern(lk->ffnBwd);
    free_kern(lk->sdpaBwd1); free_kern(lk->sdpaBwd2);
    lk->fwdFFN = lk->ffnBwd = lk->sdpaBwd1 = lk->sdpaBwd2 = NULL;
}

// ===== Main =====
int main(int argc, char *argv[]) {
    @autoreleasepool {
        setbuf(stdout, NULL);
        ane_init();
        mach_timebase_info(&g_tb);

        int total_steps = 1000;
        float lr = 1e-4f;
        float adam_b1=0.9f, adam_b2=0.999f, adam_eps=1e-8f;
        int adam_t = 0, start_step = 0;

        bool do_resume = false;
        int user_steps = -1;
        float user_lr = -1;
        for (int i=1; i<argc; i++) {
            if (strcmp(argv[i], "--resume") == 0) do_resume = true;
            else if (strcmp(argv[i], "--steps") == 0 && i+1<argc) user_steps = atoi(argv[++i]);
            else if (strcmp(argv[i], "--lr") == 0 && i+1<argc) user_lr = atof(argv[++i]);
        }
        if (user_steps > 0) total_steps = user_steps;
        if (user_lr > 0) lr = user_lr;

        // ── Allocate per-layer state ──
        LayerWeights lw[NLAYERS];
        LayerActs acts[NLAYERS];
        LayerGrads grads[NLAYERS];
        LayerKernels kern[NLAYERS];
        for (int L=0; L<NLAYERS; L++) {
            lw[L] = layer_weights_alloc();
            acts[L] = layer_acts_alloc();
            grads[L] = layer_grads_alloc();
            memset(&kern[L], 0, sizeof(LayerKernels));
        }

        float *rms_final = (float*)malloc(DIM*4);
        float *embed = (float*)malloc(VOCAB*DIM*4);
        float *grms_final = (float*)calloc(DIM, 4);
        float *gembed = (float*)calloc(VOCAB*DIM, 4);

        // Frozen copies of LoRA target weights
        float *frozen_Wq[NLAYERS], *frozen_Wv[NLAYERS];
        for (int L=0; L<NLAYERS; L++) {
            frozen_Wq[L] = (float*)malloc(WQ_SZ*4);
            frozen_Wv[L] = (float*)malloc(WQ_SZ*4);
        }

        // LoRA adapters
        LayerLoRA lora[NLAYERS];

        double cum_compile=0, cum_train=0, cum_wall=0;
        int cum_steps=0, cum_batches=0;

        // ── Load pretrained model ──
        printf("=== ANE LoRA Flex (Partial Recompile) Fine-Tuning: Stories110M ===\n");
        printf("dim=%d hidden=%d heads=%d seq=%d vocab=%d layers=%d\n",
               DIM, HIDDEN, HEADS, SEQ, VOCAB, NLAYERS);
        printf("LoRA rank=%d alpha=%.1f targets=Wq,Wv\n", LORA_RANK, LORA_ALPHA);

        if (!load_pretrained(lw, rms_final, embed, MODEL_PATH)) {
            printf("ERROR: Pretrained model required for LoRA. Cannot find %s\n", MODEL_PATH);
            return 1;
        }

        // Save frozen copies of Wq, Wv
        for (int L=0; L<NLAYERS; L++) {
            memcpy(frozen_Wq[L], lw[L].Wq, WQ_SZ*4);
            memcpy(frozen_Wv[L], lw[L].Wv, WQ_SZ*4);
        }

        // Init or load LoRA
        float resume_loss = 0;
        bool resuming = false;
        srand48(42);
        lora_init(lora, NLAYERS, DIM, LORA_RANK);

        if (do_resume) {
            resuming = lora_load(LORA_CKPT_PATH, &start_step, &total_steps, &lr, &resume_loss,
                &cum_compile, &cum_train, &cum_wall, &cum_steps, &cum_batches, &adam_t,
                lora, NLAYERS, DIM, LORA_RANK);
            if (resuming) {
                if (user_steps > 0) total_steps = user_steps;
                if (user_lr > 0) lr = user_lr;
                printf("[RESUMED LoRA step %d/%d, loss=%.4f]\n", start_step, total_steps, resume_loss);
            }
        }

        size_t lora_params = (size_t)NLAYERS * 2 * 2 * LORA_RANK * DIM;
        size_t frozen_params = (size_t)NLAYERS * LAYER_PARAMS + DIM + (size_t)VOCAB * DIM;
        printf("Trainable: %zu params (%.1f KB) | Frozen: %zu params (%.1f MB)\n",
               lora_params, lora_params*4/1024.0, frozen_params, frozen_params*4/1e6);
        printf("Mode: partial recompile (24/batch vs 60/batch) | Adam LR=%.1e\n", lr);
        printf("Compile budget: %d (vs %d original)\n\n", MAX_COMPILES_FLEX, MAX_COMPILES);

        // ── mmap token data ──
        int data_fd = open(DATA_PATH, O_RDONLY);
        if (data_fd < 0) { printf("Cannot open %s\n", DATA_PATH); return 1; }
        struct stat st; fstat(data_fd, &st);
        size_t data_len = st.st_size;
        uint16_t *token_data = (uint16_t*)mmap(NULL, data_len, PROT_READ, MAP_PRIVATE, data_fd, 0);
        if (token_data == MAP_FAILED) { printf("mmap failed\n"); return 1; }
        size_t n_tokens = data_len / 2;
        printf("Token data: %zu tokens (%.1f MB)\n", n_tokens, data_len/1e6);

        // ── Work buffers ──
        float *dy = (float*)malloc(SEQ*DIM*4);
        float *dffn = (float*)malloc(SEQ*DIM*4);
        float *dh1 = (float*)malloc(SEQ*HIDDEN*4);
        float *dh3 = (float*)malloc(SEQ*HIDDEN*4);
        float *dx_ffn = (float*)malloc(SEQ*DIM*4);
        float *dx2 = (float*)malloc(SEQ*DIM*4);
        float *do_out_buf = (float*)malloc(SEQ*DIM*4);
        float *dq = (float*)malloc(SEQ*DIM*4);
        float *dk = (float*)malloc(SEQ*DIM*4);
        float *dv = (float*)malloc(SEQ*DIM*4);
        float *dx_attn = (float*)malloc(SEQ*DIM*4);
        float *x_cur = (float*)malloc(SEQ*DIM*4);
        float *x_final = (float*)malloc(SEQ*DIM*4);
        float *logits = (float*)malloc(SEQ*VOCAB*4);
        float *dlogits = (float*)malloc(SEQ*VOCAB*4);

        // ===== ONE-TIME: Compile static kernels (frozen weights) =====
        printf("Compiling static kernels (one-time, 48 kernels)...\n");
        uint64_t tc_static = mach_absolute_time();
        for (int L=0; L<NLAYERS; L++) {
            printf("  Static layer %d/%d...\r", L+1, NLAYERS);
            fflush(stdout);
            if (!compile_static_kernels(&kern[L], &lw[L])) {
                printf("\nStatic compile failed at layer %d\n", L);
                return 1;
            }
        }
        double static_compile_ms = tb_ms(mach_absolute_time() - tc_static);
        int static_compiles = g_compile_count;
        printf("  Compiled %d static kernels in %.0fms (never recompiled)      \n",
               static_compiles, static_compile_ms);

        dispatch_queue_t dw_q = dispatch_queue_create("dw_cblas", DISPATCH_QUEUE_SERIAL);
        dispatch_group_t dw_grp = dispatch_group_create();

        float last_loss = 999.0f;
        double total_compile_ms=0, total_train_ms=0;
        int total_steps_done=0, total_batches=0;
        uint64_t t_wall_start = mach_absolute_time();

        srand48(42 + start_step);

        // ===== Training loop =====
        int step = start_step;
        while (step < total_steps) {
            // Check compile budget — only count dynamic compiles against remaining budget
            if (g_compile_count + LORA_RECOMPILE_KERNELS > MAX_COMPILES_FLEX) {
                for (int L=0; L<NLAYERS; L++) { free_lora_kernels(&kern[L]); free_static_kernels(&kern[L]); }
                double wall = tb_ms(mach_absolute_time() - t_wall_start);
                lora_save(LORA_CKPT_PATH, step, total_steps, lr, last_loss,
                    total_compile_ms+static_compile_ms+cum_compile, total_train_ms+cum_train, wall+cum_wall,
                    total_steps_done+cum_steps, total_batches+cum_batches, adam_t,
                    lora, NLAYERS, DIM, LORA_RANK, LORA_ALPHA);
                printf("[exec() restart step %d, %d compiles, loss=%.4f]\n", step, g_compile_count, last_loss);
                fflush(stdout);
                execl(argv[0], argv[0], "--resume", NULL);
                perror("execl"); return 1;
            }

            // ── Merge LoRA into working weights ──
            for (int L=0; L<NLAYERS; L++) {
                lora_merge(lw[L].Wq, frozen_Wq[L], &lora[L].wq, DIM, LORA_RANK, LORA_ALPHA);
                lora_merge(lw[L].Wv, frozen_Wv[L], &lora[L].wv, DIM, LORA_RANK, LORA_ALPHA);
            }

            // ── Recompile ONLY LoRA-changing kernels (fwdAttn + qkvBwd) ──
            uint64_t tc = mach_absolute_time();
            for (int L=0; L<NLAYERS; L++) free_lora_kernels(&kern[L]);

            bool compile_ok = true;
            for (int L=0; L<NLAYERS; L++) {
                printf("  Compiling LoRA layer %d/%d... (%d compiles)\r", L+1, NLAYERS, g_compile_count);
                fflush(stdout);
                if (!compile_lora_kernels(&kern[L], &lw[L])) {
                    printf("\nLoRA compile failed at layer %d, restart\n", L);
                    compile_ok = false; break;
                }
            }
            if (!compile_ok) { g_compile_count = MAX_COMPILES_FLEX; continue; }

            double cms = tb_ms(mach_absolute_time() - tc);
            total_compile_ms += cms;
            printf("  Compiled %d LoRA kernels in %.0fms                    \n", LORA_RECOMPILE_KERNELS, cms);

            // Zero gradient accumulators
            for (int L=0; L<NLAYERS; L++) layer_grads_zero(&grads[L]);
            memset(grms_final, 0, DIM*4);
            memset(gembed, 0, (size_t)VOCAB*DIM*4);

            int steps_batch = 0;
            float batch_loss_sum = 0;
            uint64_t tt = mach_absolute_time();

            // ── Accumulation steps (forward + backward) ──
            for (int a=0; a<ACCUM_STEPS && step<total_steps; a++, step++) {
                size_t max_pos = n_tokens - SEQ - 1;
                size_t pos = (size_t)(drand48() * max_pos);
                uint16_t *input_tokens = token_data + pos;
                uint16_t *target_tokens = token_data + pos + 1;

                embed_lookup(x_cur, embed, input_tokens, DIM, SEQ);

                // ===== FORWARD =====
                for (int L=0; L<NLAYERS; L++) {
                    LayerActs *ac = &acts[L];
                    memcpy(ac->layer_in, x_cur, SEQ*DIM*4);

                    dispatch_group_wait(dw_grp, DISPATCH_TIME_FOREVER);
                    io_write_fp16(kern[L].fwdAttn->ioIn, x_cur, DIM, SEQ);
                    ane_eval(kern[L].fwdAttn);
                    io_read_fp16(kern[L].fwdAttn->ioOut, ac->o_out,    0,     DIM, SEQ);
                    io_read_fp16(kern[L].fwdAttn->ioOut, ac->attn_out, 4*DIM, DIM, SEQ);
                    io_read_fp16(kern[L].fwdAttn->ioOut, ac->xnorm,    5*DIM, DIM, SEQ);

                    vDSP_vadd(x_cur, 1, ac->o_out, 1, ac->x2, 1, (vDSP_Length)(SEQ*DIM));

                    io_write_fp16(kern[L].fwdFFN->ioIn, ac->x2, DIM, SEQ);
                    ane_eval(kern[L].fwdFFN);
                    io_read_fp16(kern[L].fwdFFN->ioOut, ac->ffn_out,  0,            DIM,    SEQ);
                    io_read_fp16(kern[L].fwdFFN->ioOut, ac->h1,       DIM,          HIDDEN, SEQ);
                    io_read_fp16(kern[L].fwdFFN->ioOut, ac->h3,       DIM+HIDDEN,   HIDDEN, SEQ);
                    io_read_fp16(kern[L].fwdFFN->ioOut, ac->silu_out, DIM+2*HIDDEN, HIDDEN, SEQ);
                    io_read_fp16(kern[L].fwdFFN->ioOut, ac->x2norm,   DIM+3*HIDDEN, DIM,    SEQ);

                    vDSP_vadd(ac->x2, 1, ac->ffn_out, 1, x_cur, 1, (vDSP_Length)(SEQ*DIM));
                }

                rmsnorm(x_final, x_cur, rms_final, DIM, SEQ);
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            VOCAB, SEQ, DIM, 1.0f,
                            embed, DIM, x_final, SEQ, 0.0f, logits, SEQ);
                float loss = cross_entropy_loss(dlogits, logits, target_tokens, VOCAB, SEQ);
                last_loss = loss;

                // ===== BACKWARD =====
                cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                            DIM, SEQ, VOCAB, 1.0f,
                            embed, DIM, dlogits, SEQ, 0.0f, dy, SEQ);

                dispatch_group_async(dw_grp, dw_q, ^{
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                                VOCAB, DIM, SEQ, 1.0f,
                                dlogits, SEQ, x_final, SEQ, 1.0f, gembed, DIM);
                });

                float *dx_rms_final = (float*)calloc(SEQ*DIM, 4);
                rmsnorm_bwd(dx_rms_final, grms_final, dy, x_cur, rms_final, DIM, SEQ);
                memcpy(dy, dx_rms_final, SEQ*DIM*4);
                free(dx_rms_final);

                for (int L=NLAYERS-1; L>=0; L--) {
                    LayerActs *ac = &acts[L];
                    LayerGrads *gr = &grads[L];

                    memcpy(dffn, dy, SEQ*DIM*4);

                    io_write_fp16_at(kern[L].ffnBwd->ioIn, 0, dffn, DIM, SEQ);
                    io_copy(kern[L].ffnBwd->ioIn, DIM, kern[L].fwdFFN->ioOut, DIM, 2*HIDDEN, SEQ);
                    ane_eval(kern[L].ffnBwd);
                    io_read_fp16(kern[L].ffnBwd->ioOut, dx_ffn, 0,          DIM,    SEQ);
                    io_read_fp16(kern[L].ffnBwd->ioOut, dh1,    DIM,        HIDDEN, SEQ);
                    io_read_fp16(kern[L].ffnBwd->ioOut, dh3,    DIM+HIDDEN, HIDDEN, SEQ);

                    float *capt_dffn = (float*)malloc(SEQ*DIM*4); memcpy(capt_dffn, dffn, SEQ*DIM*4);
                    float *capt_silu = (float*)malloc(SEQ*HIDDEN*4); memcpy(capt_silu, ac->silu_out, SEQ*HIDDEN*4);
                    float *capt_dh1 = (float*)malloc(SEQ*HIDDEN*4); memcpy(capt_dh1, dh1, SEQ*HIDDEN*4);
                    float *capt_dh3 = (float*)malloc(SEQ*HIDDEN*4); memcpy(capt_dh3, dh3, SEQ*HIDDEN*4);
                    float *capt_x2n = (float*)malloc(SEQ*DIM*4); memcpy(capt_x2n, ac->x2norm, SEQ*DIM*4);
                    dispatch_group_async(dw_grp, dw_q, ^{
                        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, DIM, HIDDEN, SEQ,
                                    1.0f, capt_dffn, SEQ, capt_silu, SEQ, 1.0f, gr->W2, HIDDEN);
                        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, HIDDEN, DIM, SEQ,
                                    1.0f, capt_dh1, SEQ, capt_x2n, SEQ, 1.0f, gr->W1, DIM);
                        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, HIDDEN, DIM, SEQ,
                                    1.0f, capt_dh3, SEQ, capt_x2n, SEQ, 1.0f, gr->W3, DIM);
                        free(capt_dffn); free(capt_silu); free(capt_dh1); free(capt_dh3); free(capt_x2n);
                    });

                    memset(dx2, 0, SEQ*DIM*4);
                    rmsnorm_bwd(dx2, gr->rms_ffn, dx_ffn, ac->x2, lw[L].rms_ffn, DIM, SEQ);
                    for(int i=0;i<SEQ*DIM;i++) dx2[i] += dy[i];

                    memcpy(do_out_buf, dx2, SEQ*DIM*4);
                    float *capt_do = (float*)malloc(SEQ*DIM*4); memcpy(capt_do, do_out_buf, SEQ*DIM*4);
                    float *capt_attn = (float*)malloc(SEQ*DIM*4); memcpy(capt_attn, ac->attn_out, SEQ*DIM*4);
                    dispatch_group_async(dw_grp, dw_q, ^{
                        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, DIM, DIM, SEQ,
                                    1.0f, capt_do, SEQ, capt_attn, SEQ, 1.0f, gr->Wo, DIM);
                        free(capt_do); free(capt_attn);
                    });

                    io_copy(kern[L].sdpaBwd1->ioIn, 0, kern[L].fwdAttn->ioOut, DIM, 3*DIM, SEQ);
                    io_write_fp16_at(kern[L].sdpaBwd1->ioIn, 3*DIM, dx2, DIM, SEQ);
                    ane_eval(kern[L].sdpaBwd1);
                    io_copy(kern[L].sdpaBwd2->ioIn, 0, kern[L].sdpaBwd1->ioOut, DIM, 2*SCORE_CH, SEQ);
                    io_copy(kern[L].sdpaBwd2->ioIn, 2*SCORE_CH, kern[L].fwdAttn->ioOut, DIM, 2*DIM, SEQ);
                    ane_eval(kern[L].sdpaBwd2);

                    io_read_fp16(kern[L].sdpaBwd2->ioOut, dq, 0,   DIM, SEQ);
                    io_read_fp16(kern[L].sdpaBwd2->ioOut, dk, DIM,  DIM, SEQ);
                    io_read_fp16(kern[L].sdpaBwd1->ioOut, dv, 0, DIM, SEQ);

                    float *capt_dq = (float*)malloc(SEQ*DIM*4); memcpy(capt_dq, dq, SEQ*DIM*4);
                    float *capt_dk = (float*)malloc(SEQ*DIM*4); memcpy(capt_dk, dk, SEQ*DIM*4);
                    float *capt_dv = (float*)malloc(SEQ*DIM*4); memcpy(capt_dv, dv, SEQ*DIM*4);
                    float *capt_xn = (float*)malloc(SEQ*DIM*4); memcpy(capt_xn, ac->xnorm, SEQ*DIM*4);
                    dispatch_group_async(dw_grp, dw_q, ^{
                        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, DIM, DIM, SEQ,
                                    1.0f, capt_dq, SEQ, capt_xn, SEQ, 1.0f, gr->Wq, DIM);
                        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, DIM, DIM, SEQ,
                                    1.0f, capt_dk, SEQ, capt_xn, SEQ, 1.0f, gr->Wk, DIM);
                        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, DIM, DIM, SEQ,
                                    1.0f, capt_dv, SEQ, capt_xn, SEQ, 1.0f, gr->Wv, DIM);
                        free(capt_dq); free(capt_dk); free(capt_dv); free(capt_xn);
                    });

                    io_copy(kern[L].qkvBwd->ioIn, 0, kern[L].sdpaBwd2->ioOut, 0, 2*DIM, SEQ);
                    io_copy(kern[L].qkvBwd->ioIn, 2*DIM, kern[L].sdpaBwd1->ioOut, 0, DIM, SEQ);
                    ane_eval(kern[L].qkvBwd);
                    io_read_fp16(kern[L].qkvBwd->ioOut, dx_attn, 0, DIM, SEQ);

                    float *dx_rms1 = (float*)calloc(SEQ*DIM, 4);
                    rmsnorm_bwd(dx_rms1, gr->rms_att, dx_attn, ac->layer_in, lw[L].rms_att, DIM, SEQ);
                    for(int i=0;i<SEQ*DIM;i++) dy[i] = dx_rms1[i] + dx2[i];
                    free(dx_rms1);
                }

                dispatch_group_wait(dw_grp, DISPATCH_TIME_FOREVER);
                embed_backward(gembed, dy, input_tokens, DIM, SEQ);

                steps_batch++;
                batch_loss_sum += loss;
                if (step % 10 == 0 || step == start_step)
                    printf("step %-4d loss=%.4f\n", step, loss);
            }
            double tms = tb_ms(mach_absolute_time() - tt);
            total_train_ms += tms;
            total_steps_done += steps_batch;
            total_batches++;
            float avg_batch_loss = batch_loss_sum / steps_batch;

            dispatch_group_wait(dw_grp, DISPATCH_TIME_FOREVER);

            // Scale gradients
            float gsc = 1.0f / steps_batch;
            for (int L=0; L<NLAYERS; L++) {
                for(size_t i=0;i<WQ_SZ;i++) { grads[L].Wq[i]*=gsc; grads[L].Wv[i]*=gsc; }
            }

            // ── Extract LoRA gradients and Adam update ──
            adam_t++;
            for (int L=0; L<NLAYERS; L++) {
                lora_extract_grads(&lora[L].wq, grads[L].Wq, DIM, LORA_RANK, LORA_ALPHA);
                lora_extract_grads(&lora[L].wv, grads[L].Wv, DIM, LORA_RANK, LORA_ALPHA);

                adam_update(lora[L].wq.A, lora[L].wq.dA, &lora[L].wq.sA, adam_t, lr, adam_b1, adam_b2, adam_eps);
                adam_update(lora[L].wq.B, lora[L].wq.dB, &lora[L].wq.sB, adam_t, lr, adam_b1, adam_b2, adam_eps);
                adam_update(lora[L].wv.A, lora[L].wv.dA, &lora[L].wv.sA, adam_t, lr, adam_b1, adam_b2, adam_eps);
                adam_update(lora[L].wv.B, lora[L].wv.dB, &lora[L].wv.sB, adam_t, lr, adam_b1, adam_b2, adam_eps);
            }

            // LoRA parameter norms
            double norm_A=0, norm_B=0;
            for (int L=0; L<NLAYERS; L++) {
                for (size_t i=0; i<(size_t)LORA_RANK*DIM; i++) {
                    norm_A += lora[L].wq.A[i]*lora[L].wq.A[i] + lora[L].wv.A[i]*lora[L].wv.A[i];
                }
                for (size_t i=0; i<(size_t)DIM*LORA_RANK; i++) {
                    norm_B += lora[L].wq.B[i]*lora[L].wq.B[i] + lora[L].wv.B[i]*lora[L].wv.B[i];
                }
            }
            printf("  [batch %d: avg_loss=%.4f compile=%.0fms train=%.1fms (%.1fms/step) compiles=%d]\n",
                   steps_batch, avg_batch_loss, cms, tms, tms/steps_batch, g_compile_count);
            printf("    |A|=%.6f |B|=%.6f\n", sqrt(norm_A), sqrt(norm_B));
            fprintf(stderr, "{\"type\":\"lora_flex_batch\",\"batch\":%d,\"avg_loss\":%.6f,\"norm_A\":%.6f,\"norm_B\":%.6f,"
                "\"compile_ms\":%.1f,\"train_ms\":%.1f,\"ms_per_step\":%.1f}\n",
                total_batches, avg_batch_loss, sqrt(norm_A), sqrt(norm_B), cms, tms, tms/steps_batch);
        }

        // ── Final save ──
        double wall = tb_ms(mach_absolute_time() - t_wall_start);
        lora_save(LORA_CKPT_PATH, step, total_steps, lr, last_loss,
            total_compile_ms+static_compile_ms+cum_compile, total_train_ms+cum_train, wall+cum_wall,
            total_steps_done+cum_steps, total_batches+cum_batches, adam_t,
            lora, NLAYERS, DIM, LORA_RANK, LORA_ALPHA);
        printf("\nLoRA checkpoint saved: %s\n", LORA_CKPT_PATH);

        // ── Efficiency report ──
        total_compile_ms += static_compile_ms + cum_compile;
        total_train_ms += cum_train;
        wall += cum_wall; total_steps_done += cum_steps; total_batches += cum_batches;
        printf("\n=== LoRA Flex (Partial Recompile) Training Report ===\n");
        printf("Total steps:     %d\n", total_steps_done);
        printf("Wall time:       %.0f ms (%.1f s)\n", wall, wall/1000);
        printf("Compile time:    %.0f ms (%.1f%%) [static=%.0fms + dynamic=%.0fms]\n",
               total_compile_ms, 100*total_compile_ms/(wall+static_compile_ms),
               static_compile_ms, total_compile_ms-static_compile_ms);
        printf("Train time:      %.0f ms (%.1f%%)\n", total_train_ms, 100*total_train_ms/wall);
        printf("Avg train:       %.1f ms/step\n", total_train_ms/fmax(1, total_steps_done));
        printf("Compiles/batch:  %d (vs %d original = %.0f%% reduction)\n",
               LORA_RECOMPILE_KERNELS, TOTAL_WEIGHT_KERNELS,
               100.0*(1.0 - (double)LORA_RECOMPILE_KERNELS/TOTAL_WEIGHT_KERNELS));
        printf("LoRA params:     %zu (%.1f KB)\n", lora_params, lora_params*4/1024.0);

        // ── Cleanup ──
        for (int L=0; L<NLAYERS; L++) {
            free_lora_kernels(&kern[L]);
            free_static_kernels(&kern[L]);
            layer_weights_free(&lw[L]);
            layer_acts_free(&acts[L]);
            layer_grads_free(&grads[L]);
            free(frozen_Wq[L]); free(frozen_Wv[L]);
        }
        lora_free(lora, NLAYERS);
        munmap(token_data, data_len);
        close(data_fd);
        free(rms_final); free(embed); free(grms_final); free(gembed);
        free(dy); free(dffn); free(dh1); free(dh3); free(dx_ffn); free(dx2);
        free(do_out_buf); free(dq); free(dk); free(dv); free(dx_attn);
        free(x_cur); free(x_final); free(logits); free(dlogits);
    }
    return 0;
}
