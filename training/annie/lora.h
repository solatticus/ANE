// lora.h — LoRA (Low-Rank Adaptation) for Annie
// Parameterized for GQA: LoRA on Wq (dim×dim) and Wv (dim×kv_dim).
// CPU-side merge: W_eff = W_frozen + (alpha/rank) * B @ A
#pragma once
#include "config.h"

typedef struct {
    float *A;        // [rank × in_dim]  — Xavier init
    float *B;        // [out_dim × rank] — zero init
    float *dA, *dB;  // gradient buffers
    AdamState sA, sB;
    int in_dim, out_dim, rank;
} LoRAAdapter;

typedef struct {
    LoRAAdapter wq;  // Query: in_dim=dim, out_dim=dim
    LoRAAdapter wv;  // Value: in_dim=dim, out_dim=kv_dim
} AnnieLayerLoRA;

// ── Adapter lifecycle ──

static void lora_adapter_init(LoRAAdapter *adp, int out_dim, int in_dim, int rank) {
    size_t a_sz = (size_t)rank * in_dim;
    size_t b_sz = (size_t)out_dim * rank;
    adp->in_dim = in_dim;
    adp->out_dim = out_dim;
    adp->rank = rank;
    adp->A  = (float*)malloc(a_sz * 4);
    adp->B  = (float*)calloc(b_sz, 4);
    adp->dA = (float*)malloc(a_sz * 4);
    adp->dB = (float*)malloc(b_sz * 4);
    adp->sA = adam_alloc(a_sz);
    adp->sB = adam_alloc(b_sz);
    float limit = sqrtf(6.0f / (rank + in_dim));
    for (size_t i = 0; i < a_sz; i++)
        adp->A[i] = limit * (2.0f * drand48() - 1.0f);
}

static void lora_adapter_free(LoRAAdapter *adp) {
    free(adp->A); free(adp->B);
    free(adp->dA); free(adp->dB);
    adam_free(&adp->sA); adam_free(&adp->sB);
}

static void annie_lora_init(AnnieLayerLoRA *lora, const AnnieConfig *cfg) {
    int kv = ac_kv_dim(cfg);
    for (int L = 0; L < cfg->n_layers; L++) {
        // Wq: [dim, dim] → LoRA A[rank, dim], B[dim, rank]
        lora_adapter_init(&lora[L].wq, cfg->dim, cfg->dim, cfg->lora_rank);
        // Wv: [kv_dim, dim] → LoRA A[rank, dim], B[kv_dim, rank]
        lora_adapter_init(&lora[L].wv, kv, cfg->dim, cfg->lora_rank);
    }
}

static void annie_lora_free(AnnieLayerLoRA *lora, int nlayers) {
    for (int L = 0; L < nlayers; L++) {
        lora_adapter_free(&lora[L].wq);
        lora_adapter_free(&lora[L].wv);
    }
}

// ── Core operations ──

// Merge: W_eff = W_frozen + (alpha/rank) * B @ A
// W: [out_dim, in_dim], A: [rank, in_dim], B: [out_dim, rank]
static void lora_merge(float *W_eff, const float *W_frozen, const LoRAAdapter *adp,
                       float alpha) {
    int out_dim = adp->out_dim, in_dim = adp->in_dim, rank = adp->rank;
    memcpy(W_eff, W_frozen, (size_t)out_dim * in_dim * 4);
    float scale = alpha / rank;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                out_dim, in_dim, rank, scale,
                adp->B, rank, adp->A, in_dim,
                1.0f, W_eff, in_dim);
}

// Extract LoRA gradients from full dW: dA = s·B^T·dW, dB = s·dW·A^T
static void lora_extract_grads(LoRAAdapter *adp, const float *dW, float alpha) {
    int out_dim = adp->out_dim, in_dim = adp->in_dim, rank = adp->rank;
    float scale = alpha / rank;
    // dA = scale * B^T @ dW = [rank, out_dim] @ [out_dim, in_dim] → [rank, in_dim]
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                rank, in_dim, out_dim, scale,
                adp->B, rank, dW, in_dim,
                0.0f, adp->dA, in_dim);
    // dB = scale * dW @ A^T = [out_dim, in_dim] @ [in_dim, rank] → [out_dim, rank]
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                out_dim, rank, in_dim, scale,
                dW, in_dim, adp->A, in_dim,
                0.0f, adp->dB, rank);
}

// ── Checkpoint I/O ──

static void annie_lora_save(const char *path, int step, int total_steps, float lr, float loss,
                            double cc, double ct, double cw, int cs, int cb, int adam_t,
                            const AnnieLayerLoRA *lora, const AnnieConfig *cfg) {
    FILE *f = fopen(path, "wb");
    AnnieCkptHdr h = {0};
    h.magic = ANNIE_CKPT_MAGIC; h.version = ANNIE_CKPT_VERSION;
    h.step = step; h.total_steps = total_steps;
    h.n_layers = cfg->n_layers; h.dim = cfg->dim; h.rank = cfg->lora_rank;
    h.alpha = cfg->lora_alpha; h.lr = lr; h.loss = loss; h.adam_t = adam_t;
    h.cum_compile = cc; h.cum_train = ct; h.cum_wall = cw;
    h.cum_steps = cs; h.cum_batches = cb;
    fwrite(&h, sizeof(h), 1, f);

    for (int L = 0; L < cfg->n_layers; L++) {
        const LoRAAdapter *q = &lora[L].wq, *v = &lora[L].wv;
        size_t q_a = (size_t)q->rank * q->in_dim, q_b = (size_t)q->out_dim * q->rank;
        size_t v_a = (size_t)v->rank * v->in_dim, v_b = (size_t)v->out_dim * v->rank;
        fwrite(q->A, 4, q_a, f); fwrite(q->B, 4, q_b, f);
        fwrite(q->sA.m, 4, q_a, f); fwrite(q->sA.v, 4, q_a, f);
        fwrite(q->sB.m, 4, q_b, f); fwrite(q->sB.v, 4, q_b, f);
        fwrite(v->A, 4, v_a, f); fwrite(v->B, 4, v_b, f);
        fwrite(v->sA.m, 4, v_a, f); fwrite(v->sA.v, 4, v_a, f);
        fwrite(v->sB.m, 4, v_b, f); fwrite(v->sB.v, 4, v_b, f);
    }
    fclose(f);
}

static bool annie_lora_load(const char *path, int *step, int *total_steps, float *lr, float *loss,
                            double *cc, double *ct, double *cw, int *cs, int *cb, int *adam_t,
                            AnnieLayerLoRA *lora, const AnnieConfig *cfg) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    AnnieCkptHdr h;
    fread(&h, sizeof(h), 1, f);
    if (h.magic != ANNIE_CKPT_MAGIC || h.version != ANNIE_CKPT_VERSION) { fclose(f); return false; }
    if (h.n_layers != cfg->n_layers || h.dim != cfg->dim || h.rank != cfg->lora_rank) {
        printf("LoRA ckpt mismatch: layers=%d/%d dim=%d/%d rank=%d/%d\n",
               h.n_layers, cfg->n_layers, h.dim, cfg->dim, h.rank, cfg->lora_rank);
        fclose(f); return false;
    }
    *step = h.step; *total_steps = h.total_steps;
    *lr = h.lr; *loss = h.loss; *adam_t = h.adam_t;
    *cc = h.cum_compile; *ct = h.cum_train; *cw = h.cum_wall;
    *cs = h.cum_steps; *cb = h.cum_batches;

    for (int L = 0; L < cfg->n_layers; L++) {
        LoRAAdapter *q = &lora[L].wq, *v = &lora[L].wv;
        size_t q_a = (size_t)q->rank * q->in_dim, q_b = (size_t)q->out_dim * q->rank;
        size_t v_a = (size_t)v->rank * v->in_dim, v_b = (size_t)v->out_dim * v->rank;
        fread(q->A, 4, q_a, f); fread(q->B, 4, q_b, f);
        fread(q->sA.m, 4, q_a, f); fread(q->sA.v, 4, q_a, f);
        fread(q->sB.m, 4, q_b, f); fread(q->sB.v, 4, q_b, f);
        fread(v->A, 4, v_a, f); fread(v->B, 4, v_b, f);
        fread(v->sA.m, 4, v_a, f); fread(v->sA.v, 4, v_a, f);
        fread(v->sB.m, 4, v_b, f); fread(v->sB.v, 4, v_b, f);
    }
    fclose(f);
    return true;
}
