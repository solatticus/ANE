// train_lora.m — Annie LoRA fine-tuning on Qwen2.5-3B via ANE
// Compile 4 unique dynamic kernels at startup. LoRA on Wq/Wv only.
// 4 ANE evals/layer forward + 5 ANE evals/layer backward = 324 evals/step.
#include "forward.h"
#include "backward.h"
#include "lora.h"

#define CKPT_PATH "annie_lora_ckpt.bin"
#define MODEL_PATH "qwen3b_weights.bin"
#define DATA_PATH "annie_train_data.bin"

// ===== Weight loading from Annie binary format =====
static bool load_pretrained(AnnieLayerWeights *lw, float *rms_final, float *embed,
                             const char *path, const AnnieConfig *cfg) {
    FILE *f = fopen(path, "rb");
    if (!f) { printf("Cannot open %s\n", path); return false; }

    AnnieWeightHdr hdr;
    fread(&hdr, sizeof(hdr), 1, f);
    if (hdr.magic != ANNIE_WEIGHT_MAGIC || hdr.version != ANNIE_WEIGHT_VERSION) {
        printf("  ERROR: Invalid weight file magic/version\n"); fclose(f); return false;
    }
    if (hdr.config.dim != cfg->dim || hdr.config.n_layers != cfg->n_layers) {
        printf("  ERROR: Config mismatch (dim=%d/%d layers=%d/%d)\n",
               hdr.config.dim, cfg->dim, hdr.config.n_layers, cfg->n_layers);
        fclose(f); return false;
    }

    int D = cfg->dim, H = cfg->hidden_dim, NL = cfg->n_layers;
    int kv = ac_kv_dim(cfg), V = cfg->vocab_size;

    printf("  Model: dim=%d hidden=%d layers=%d heads=%d/%d vocab=%d\n",
           D, H, NL, cfg->n_heads, cfg->n_kv_heads, V);

    // Embedding table
    fread(embed, 4, (size_t)V * D, f);

    // Per-layer weights
    for (int L = 0; L < NL; L++) {
        fread(lw[L].rms_att, 4, D, f);
        fread(lw[L].Wq, 4, ac_wq_sz(cfg), f);
        if (cfg->qkv_bias) fread(lw[L].bq, 4, D, f);
        fread(lw[L].Wk, 4, ac_wk_sz(cfg), f);
        if (cfg->qkv_bias) fread(lw[L].bk, 4, kv, f);
        fread(lw[L].Wv, 4, ac_wv_sz(cfg), f);
        if (cfg->qkv_bias) fread(lw[L].bv, 4, kv, f);
        fread(lw[L].Wo, 4, ac_wo_sz(cfg), f);
        if (cfg->qkv_bias) fread(lw[L].bo, 4, D, f);
        fread(lw[L].rms_ffn, 4, D, f);
        fread(lw[L].W1, 4, ac_w1_sz(cfg), f);
        fread(lw[L].W2, 4, ac_w2_sz(cfg), f);
        fread(lw[L].W3, 4, ac_w3_sz(cfg), f);
    }

    // Final RMSNorm
    fread(rms_final, 4, D, f);
    fclose(f);
    printf("  Loaded pretrained weights (%.1f GB)\n",
           (double)((size_t)V * D + (size_t)NL * (ac_wq_sz(cfg) + ac_wk_sz(cfg) + ac_wv_sz(cfg)
           + ac_wo_sz(cfg) + ac_w1_sz(cfg) + ac_w2_sz(cfg) + ac_w3_sz(cfg))) * 4 / 1e9);
    return true;
}

int main(int argc, char *argv[]) {
    @autoreleasepool {
        setbuf(stdout, NULL);
        ane_init();
        mach_timebase_info(&g_tb);

        AnnieConfig cfg = annie_config_qwen3b();
        int D = cfg.dim, H = cfg.hidden_dim, S = cfg.max_seq_len, NL = cfg.n_layers;
        int kv = ac_kv_dim(&cfg), V = cfg.vocab_size;

        int total_steps = 10000;
        float max_lr = 1e-4f;
        float adam_b1 = 0.9f, adam_b2 = 0.999f, adam_eps = 1e-8f;
        int adam_t = 0, start_step = 0;
        int accum_steps = 10;
        int warmup_steps = 50;
        float grad_clip = 1.0f;
        float loss_scale = 256.0f; // fp16 loss scaling for ANE backward
        float min_lr_frac = 0.1f;

        bool do_resume = false;
        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "--resume") == 0) do_resume = true;
            else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) total_steps = atoi(argv[++i]);
            else if (strcmp(argv[i], "--lr") == 0 && i + 1 < argc) max_lr = atof(argv[++i]);
            else if (strcmp(argv[i], "--accum") == 0 && i + 1 < argc) accum_steps = atoi(argv[++i]);
            else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) warmup_steps = atoi(argv[++i]);
            else if (strcmp(argv[i], "--clip") == 0 && i + 1 < argc) grad_clip = atof(argv[++i]);
        }
        float lr = max_lr;

        printf("=== Annie ANE LoRA Training: Qwen2.5-3B ===\n");
        printf("dim=%d hidden=%d heads=%d/%d head_dim=%d seq=%d vocab=%d layers=%d\n",
               D, H, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, S, V, NL);
        printf("LoRA: rank=%d alpha=%.1f\n", cfg.lora_rank, cfg.lora_alpha);
        int lora_params_wq = 2 * D * cfg.lora_rank;         // A[r,D]+B[D,r] per layer
        int lora_params_wv = (kv + D) * cfg.lora_rank;      // A[r,D]+B[kv,r] per layer
        int lora_total = (lora_params_wq + lora_params_wv) * NL;
        printf("  Trainable: %d per layer, %d total (%.2f%% of 3B)\n",
               lora_params_wq + lora_params_wv, lora_total, 100.0f * lora_total / 3.09e9f);
        printf("Accum %d steps, LR=%g\n", accum_steps, max_lr);

        // ── Allocate per-layer state ──
        AnnieLayerWeights *lw = (AnnieLayerWeights*)malloc(NL * sizeof(AnnieLayerWeights));
        AnnieLayerActs *acts = (AnnieLayerActs*)malloc(NL * sizeof(AnnieLayerActs));
        AnnieLayerGrads *grads = (AnnieLayerGrads*)malloc(NL * sizeof(AnnieLayerGrads));
        for (int L = 0; L < NL; L++) {
            lw[L] = annie_weights_alloc(&cfg);
            acts[L] = annie_acts_alloc(&cfg);
            grads[L] = annie_grads_alloc(&cfg);
        }
        float *rms_final = (float*)malloc(D * 4);
        float *embed = (float*)malloc((size_t)V * D * 4);

        // LoRA state
        float **frozen_Wq = (float**)malloc(NL * sizeof(float*));
        float **frozen_Wv = (float**)malloc(NL * sizeof(float*));
        AnnieLayerLoRA *lora = (AnnieLayerLoRA*)malloc(NL * sizeof(AnnieLayerLoRA));

        double cum_compile = 0, cum_train = 0, cum_wall = 0;
        int cum_steps = 0, cum_batches = 0;
        float resume_loss = 0;

        // ── Load pretrained weights ──
        if (!load_pretrained(lw, rms_final, embed, MODEL_PATH, &cfg)) {
            printf("FATAL: Cannot load pretrained model\n");
            return 1;
        }

        // Save frozen copies + init LoRA adapters
        srand48(42);
        for (int L = 0; L < NL; L++) {
            frozen_Wq[L] = (float*)malloc(ac_wq_sz(&cfg) * 4);
            frozen_Wv[L] = (float*)malloc(ac_wv_sz(&cfg) * 4);
            memcpy(frozen_Wq[L], lw[L].Wq, ac_wq_sz(&cfg) * 4);
            memcpy(frozen_Wv[L], lw[L].Wv, ac_wv_sz(&cfg) * 4);
        }
        annie_lora_init(lora, &cfg);

        // Resume from checkpoint
        if (do_resume) {
            bool ok = annie_lora_load(CKPT_PATH, &start_step, &total_steps, &lr, &resume_loss,
                &cum_compile, &cum_train, &cum_wall, &cum_steps, &cum_batches, &adam_t,
                lora, &cfg);
            if (ok) {
                printf("[RESUMED step %d, loss=%.4f]\n", start_step, resume_loss);
                for (int L = 0; L < NL; L++) {
                    lora_merge(lw[L].Wq, frozen_Wq[L], &lora[L].wq, cfg.lora_alpha);
                    lora_merge(lw[L].Wv, frozen_Wv[L], &lora[L].wv, cfg.lora_alpha);
                }
            }
        }

        // Precompute transposed weights for ANE dynamic kernels
        // Forward: Wq^T, Wk^T, Wv^T (for qkvProj); W1^T, W3^T (for dimToHidden); W2^T (for hiddenToDim)
        float **Wqt = (float**)malloc(NL * sizeof(float*));
        float **Wkt = (float**)malloc(NL * sizeof(float*));
        float **Wvt = (float**)malloc(NL * sizeof(float*));
        float **W1t = (float**)malloc(NL * sizeof(float*));
        float **W2t = (float**)malloc(NL * sizeof(float*));
        float **W3t = (float**)malloc(NL * sizeof(float*));
        for (int L = 0; L < NL; L++) {
            Wqt[L] = (float*)malloc(ac_wq_sz(&cfg) * 4);
            Wkt[L] = (float*)malloc(ac_wk_sz(&cfg) * 4);
            Wvt[L] = (float*)malloc(ac_wv_sz(&cfg) * 4);
            W1t[L] = (float*)malloc(ac_w1_sz(&cfg) * 4);
            W2t[L] = (float*)malloc(ac_w2_sz(&cfg) * 4);
            W3t[L] = (float*)malloc(ac_w3_sz(&cfg) * 4);
            transpose_weight(Wqt[L], lw[L].Wq, D, D);
            transpose_weight(Wkt[L], lw[L].Wk, kv, D);
            transpose_weight(Wvt[L], lw[L].Wv, kv, D);
            transpose_weight(W1t[L], lw[L].W1, H, D);
            transpose_weight(W2t[L], lw[L].W2, D, H);
            transpose_weight(W3t[L], lw[L].W3, H, D);
        }

        // ── Load token data ──
        int data_fd = open(DATA_PATH, O_RDONLY);
        if (data_fd < 0) { printf("Cannot open %s\n", DATA_PATH); return 1; }
        struct stat st; fstat(data_fd, &st);
        size_t data_len = st.st_size;
        uint32_t *token_data = (uint32_t*)mmap(NULL, data_len, PROT_READ, MAP_PRIVATE, data_fd, 0);
        if (token_data == MAP_FAILED) { printf("mmap failed\n"); return 1; }
        size_t n_tokens = data_len / 4;  // uint32 tokens
        printf("Token data: %zu tokens (%.1f MB)\n", n_tokens, data_len / 1e6);

        // ── Compile ANE kernels (one-time) ──
        printf("Compiling 4 dynamic kernels (one-time)...\n");
        uint64_t tc = mach_absolute_time();
        AnnieKernels ak;
        if (!annie_compile_kernels(&ak, &cfg)) {
            printf("Compilation failed!\n"); return 1;
        }
        double compile_ms = tb_ms(mach_absolute_time() - tc);
        printf("Compiled 4 kernels in %.0fms (shared across all %d layers)\n\n", compile_ms, NL);

        // ── Work buffers ──
        float *dy       = (float*)malloc(S * D * 4);
        float *dffn     = (float*)malloc(S * D * 4);
        float *dx_ffn   = (float*)malloc(S * D * 4);
        float *dx2      = (float*)malloc(S * D * 4);
        float *da_buf   = (float*)malloc(S * D * 4);
        float *dq       = (float*)malloc(S * D * 4);
        float *dk_buf   = (float*)malloc(S * kv * 4);
        float *dv       = (float*)malloc(S * kv * 4);
        float *dx_attn  = (float*)malloc(S * D * 4);
        float *x_cur    = (float*)malloc(S * D * 4);
        float *x_final  = (float*)malloc(S * D * 4);
        float *xnorm_buf= (float*)malloc(S * D * 4);
        float *gate_buf = (float*)malloc(S * H * 4);
        float *dh1      = (float*)malloc(S * H * 4);
        float *dh3      = (float*)malloc(S * H * 4);
        float *dsilu    = (float*)malloc(S * H * 4);
        float *silu_tmp = (float*)malloc(S * H * 4);
        float *silu_tmp2= (float*)malloc(S * H * 4);

        // Classifier: use embed as classifier (tied embeddings)
        // For 151K vocab, tiled on CPU
        float *logits  = (float*)malloc((size_t)S * V * 4);
        float *dlogits = (float*)malloc((size_t)S * V * 4);

        dispatch_queue_t dw_q = dispatch_queue_create("dw_cblas", DISPATCH_QUEUE_SERIAL);
        dispatch_group_t dw_grp = dispatch_group_create();

        float last_loss = 999.0f;
        double total_train_ms = 0;
        int total_steps_done = 0;
        uint64_t t_wall_start = mach_absolute_time();
        srand48(42 + start_step);

        // ═══════════════════════════════════════════════
        // ═══         MAIN TRAINING LOOP             ═══
        // ═══════════════════════════════════════════════
        for (int step = start_step; step < total_steps; step++) {
            uint64_t t_step = mach_absolute_time();

            // LoRA merge: W_eff = W_frozen + scale * B @ A
            for (int L = 0; L < NL; L++) {
                lora_merge(lw[L].Wq, frozen_Wq[L], &lora[L].wq, cfg.lora_alpha);
                lora_merge(lw[L].Wv, frozen_Wv[L], &lora[L].wv, cfg.lora_alpha);
                // Update transposed weights for ANE
                transpose_weight(Wqt[L], lw[L].Wq, D, D);
                transpose_weight(Wvt[L], lw[L].Wv, kv, D);
            }

            // Sample data
            size_t max_pos = n_tokens - S - 1;
            size_t pos = (size_t)(drand48() * max_pos);
            uint32_t *input_tokens  = token_data + pos;
            uint32_t *target_tokens = token_data + pos + 1;

            // Embedding lookup (frozen)
            embed_lookup(x_cur, embed, input_tokens, D, S);

            // Wait for pending dW cblas from previous step
            dispatch_group_wait(dw_grp, DISPATCH_TIME_FOREVER);

            // ═══ FORWARD (36 layers) ═══
            for (int L = 0; L < NL; L++) {
                annie_forward_layer(x_cur, &acts[L], &lw[L],
                    Wqt[L], Wkt[L], Wvt[L], W1t[L], W2t[L], W3t[L],
                    xnorm_buf, gate_buf, &ak, &cfg);
            }

            // Final RMSNorm + classifier + loss (CPU)
            rmsnorm(x_final, x_cur, rms_final, D, S, cfg.rms_norm_eps);

            // Classifier: embed[V, D] @ x_final[D, S] → logits[V, S]
            // Tied embeddings — use embed table as classifier weight
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        V, S, D, 1.0f, embed, D, x_final, S, 0.0f, logits, S);
            float loss = cross_entropy_loss(dlogits, logits, target_tokens, V, S);
            last_loss = loss;

            // Loss scaling: scale dlogits to prevent fp16 underflow in ANE backward kernels
            // All gradients flow scaled; divided out by loss_scale before Adam update
            vDSP_vsmul(dlogits, 1, &loss_scale, dlogits, 1, (vDSP_Length)(S * V));

            // ═══ BACKWARD ═══

            // Classifier backward (frozen — compute dx only)
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        D, S, V, 1.0f, embed, D, dlogits, S, 0.0f, dy, S);

            // Final RMSNorm backward (frozen)
            float *dx_rms_final = (float*)calloc(S * D, 4);
            float drms_final_dummy[D];
            memset(drms_final_dummy, 0, D * 4);
            rmsnorm_bwd(dx_rms_final, drms_final_dummy, dy, x_cur, rms_final, D, S, cfg.rms_norm_eps);
            memcpy(dy, dx_rms_final, S * D * 4);
            free(dx_rms_final);

            // ═══ BACKWARD (36 layers, reverse) ═══
            for (int L = NL - 1; L >= 0; L--) {
                annie_backward_layer(dy, &acts[L], &lw[L], &grads[L], &ak, &cfg,
                    dffn, dx_ffn, dx2, da_buf, dq, dk_buf, dv,
                    dh1, dh3, dsilu, silu_tmp, silu_tmp2, dx_attn,
                    dw_grp, dw_q);
            }

            dispatch_group_wait(dw_grp, DISPATCH_TIME_FOREVER);

            double step_ms = tb_ms(mach_absolute_time() - t_step);
            total_train_ms += step_ms;
            total_steps_done++;

            // ── Logging ──
            if (step % 10 == 0 || step == start_step) {
                float a_norm_sq = 0, b_norm_sq = 0;
                for (int L = 0; L < NL; L++) {
                    float s;
                    size_t qa = (size_t)cfg.lora_rank * D, qb = (size_t)D * cfg.lora_rank;
                    size_t va = (size_t)cfg.lora_rank * D, vb = (size_t)kv * cfg.lora_rank;
                    vDSP_dotpr(lora[L].wq.A, 1, lora[L].wq.A, 1, &s, (vDSP_Length)qa); a_norm_sq += s;
                    vDSP_dotpr(lora[L].wv.A, 1, lora[L].wv.A, 1, &s, (vDSP_Length)va); a_norm_sq += s;
                    vDSP_dotpr(lora[L].wq.B, 1, lora[L].wq.B, 1, &s, (vDSP_Length)qb); b_norm_sq += s;
                    vDSP_dotpr(lora[L].wv.B, 1, lora[L].wv.B, 1, &s, (vDSP_Length)vb); b_norm_sq += s;
                }
                printf("step %-4d loss=%.4f  lr=%.2e  |A|=%.4f |B|=%.6f  %.1fms/step\n",
                       step, loss, lr, sqrtf(a_norm_sq), sqrtf(b_norm_sq), step_ms);
                fprintf(stderr, "{\"type\":\"annie_lora\",\"step\":%d,\"loss\":%.6f,\"lr\":%.2e,"
                        "\"a_norm\":%.6f,\"b_norm\":%.6f,\"ms\":%.1f,\"compile_count\":%d}\n",
                        step, loss, lr, sqrtf(a_norm_sq), sqrtf(b_norm_sq), step_ms, g_compile_count);
            }

            // ── LoRA Adam update every accum_steps ──
            if ((step + 1) % accum_steps == 0 || step == total_steps - 1) {
                dispatch_group_wait(dw_grp, DISPATCH_TIME_FOREVER);
                float gsc = 1.0f / (accum_steps * loss_scale);
                adam_t++;

                // Scale accumulated gradients
                for (int L = 0; L < NL; L++) {
                    vDSP_vsmul(grads[L].Wq, 1, &gsc, grads[L].Wq, 1, (vDSP_Length)ac_wq_sz(&cfg));
                    vDSP_vsmul(grads[L].Wv, 1, &gsc, grads[L].Wv, 1, (vDSP_Length)ac_wv_sz(&cfg));
                }

                // Gradient norm
                float grad_norm_sq = 0;
                for (int L = 0; L < NL; L++) {
                    float s;
                    vDSP_dotpr(grads[L].Wq, 1, grads[L].Wq, 1, &s, (vDSP_Length)ac_wq_sz(&cfg));
                    grad_norm_sq += s;
                    vDSP_dotpr(grads[L].Wv, 1, grads[L].Wv, 1, &s, (vDSP_Length)ac_wv_sz(&cfg));
                    grad_norm_sq += s;
                }
                float grad_norm = sqrtf(grad_norm_sq);
                if ((step + 1) % 10 == 0) printf("  grad_norm=%.4f\n", grad_norm);

                // Gradient clipping
                if (grad_clip > 0 && grad_norm > grad_clip) {
                    float clip_scale = grad_clip / grad_norm;
                    for (int L = 0; L < NL; L++) {
                        vDSP_vsmul(grads[L].Wq, 1, &clip_scale, grads[L].Wq, 1, (vDSP_Length)ac_wq_sz(&cfg));
                        vDSP_vsmul(grads[L].Wv, 1, &clip_scale, grads[L].Wv, 1, (vDSP_Length)ac_wv_sz(&cfg));
                    }
                }

                // Cosine LR schedule with warmup
                if (step < warmup_steps) {
                    lr = max_lr * ((float)(step + 1)) / warmup_steps;
                } else {
                    float decay = (float)(step - warmup_steps) / (float)(total_steps - warmup_steps);
                    float min_lr = max_lr * min_lr_frac;
                    lr = min_lr + 0.5f * (1.0f + cosf(M_PI * decay)) * (max_lr - min_lr);
                }

                // Extract LoRA gradients + Adam update
                for (int L = 0; L < NL; L++) {
                    lora_extract_grads(&lora[L].wq, grads[L].Wq, cfg.lora_alpha);
                    lora_extract_grads(&lora[L].wv, grads[L].Wv, cfg.lora_alpha);
                    adam_update(lora[L].wq.A, lora[L].wq.dA, &lora[L].wq.sA, adam_t, lr, adam_b1, adam_b2, adam_eps);
                    adam_update(lora[L].wq.B, lora[L].wq.dB, &lora[L].wq.sB, adam_t, lr, adam_b1, adam_b2, adam_eps);
                    adam_update(lora[L].wv.A, lora[L].wv.dA, &lora[L].wv.sA, adam_t, lr, adam_b1, adam_b2, adam_eps);
                    adam_update(lora[L].wv.B, lora[L].wv.dB, &lora[L].wv.sB, adam_t, lr, adam_b1, adam_b2, adam_eps);
                }

                // Zero gradients
                for (int L = 0; L < NL; L++)
                    annie_grads_zero(&grads[L], &cfg);

                // Checkpoint
                if ((step + 1) % 100 == 0) {
                    double wall = tb_ms(mach_absolute_time() - t_wall_start);
                    annie_lora_save(CKPT_PATH, step + 1, total_steps, lr, last_loss,
                        compile_ms + cum_compile, total_train_ms + cum_train, wall + cum_wall,
                        total_steps_done + cum_steps,
                        (total_steps_done + cum_steps) / accum_steps + cum_batches,
                        adam_t, lora, &cfg);
                    printf("  [checkpoint saved: step %d]\n", step + 1);
                }
            }
        }

        // ── Report ──
        double wall = tb_ms(mach_absolute_time() - t_wall_start);
        printf("\n=== Annie LoRA Training Report ===\n");
        printf("Total steps:    %d\n", total_steps_done);
        printf("Compile:        %.0fms (one-time)\n", compile_ms);
        printf("Train time:     %.0fms (%.1fms/step)\n", total_train_ms, total_train_ms / total_steps_done);
        printf("Wall time:      %.1fs\n", (wall + cum_wall) / 1000);
        printf("Compile count:  %d (should be 4)\n", g_compile_count);
        printf("ANE evals/step: %d (%d forward + %d backward)\n",
               9 * NL, 4 * NL, 5 * NL);

        // ── Cleanup ──
        for (int L = 0; L < NL; L++) {
            annie_weights_free(&lw[L]);
            annie_acts_free(&acts[L]);
            annie_grads_free(&grads[L]);
            free(Wqt[L]); free(Wkt[L]); free(Wvt[L]);
            free(W1t[L]); free(W2t[L]); free(W3t[L]);
            free(frozen_Wq[L]); free(frozen_Wv[L]);
        }
        free(lw); free(acts); free(grads);
        free(Wqt); free(Wkt); free(Wvt); free(W1t); free(W2t); free(W3t);
        free(frozen_Wq); free(frozen_Wv);
        annie_lora_free(lora, NL);
        free(lora);
        annie_free_kernels(&ak);
        free(rms_final); free(embed);
        free(dy); free(dffn); free(dx_ffn); free(dx2); free(da_buf);
        free(dq); free(dk_buf); free(dv); free(dx_attn);
        free(x_cur); free(x_final); free(xnorm_buf); free(gate_buf);
        free(dh1); free(dh3); free(dsilu); free(silu_tmp); free(silu_tmp2);
        free(logits); free(dlogits);
        munmap(token_data, data_len); close(data_fd);
    }
    return 0;
}
