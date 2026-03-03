// lora.h — LoRA (Low-Rank Adaptation) for ANE training
// Adds trainable low-rank matrices to Wq and Wv projections.
// CPU-side merge: W_eff = W_frozen + (alpha/rank) * B @ A
// No changes to MIL generators or ANE compile/eval pipeline.
#pragma once
#include "config.h"

#ifndef LORA_RANK
#define LORA_RANK 4
#endif
#ifndef LORA_ALPHA
#define LORA_ALPHA 4.0f
#endif

#define LORA_CKPT_MAGIC   0x41524F4C  // "LORA"
#define LORA_CKPT_VERSION 1

typedef struct {
    float *A;        // [rank × dim]   — initialized Xavier
    float *B;        // [dim × rank]   — initialized zero
    float *dA;       // [rank × dim]   gradient buffer
    float *dB;       // [dim × rank]   gradient buffer
    AdamState sA;    // Adam state for A
    AdamState sB;    // Adam state for B
} LoRAAdapter;

typedef struct {
    LoRAAdapter wq;  // Query projection adapter
    LoRAAdapter wv;  // Value projection adapter
} LayerLoRA;

typedef struct {
    int magic, version;
    int step, total_steps;
    int n_layers, dim, rank;
    float alpha, lr, loss;
    int adam_t;
    double cum_compile, cum_train, cum_wall;
    int cum_steps, cum_batches;
    int pad[2];
} LoRACkptHdr;

// ── Adapter lifecycle ──

static void lora_adapter_init(LoRAAdapter *adp, int dim, int rank) {
    size_t a_sz = (size_t)rank * dim;
    size_t b_sz = (size_t)dim * rank;
    adp->A  = (float*)malloc(a_sz * 4);
    adp->B  = (float*)calloc(b_sz, 4);
    adp->dA = (float*)malloc(a_sz * 4);
    adp->dB = (float*)malloc(b_sz * 4);
    adp->sA = adam_alloc(a_sz);
    adp->sB = adam_alloc(b_sz);
    float limit = sqrtf(6.0f / (rank + dim));
    for (size_t i = 0; i < a_sz; i++)
        adp->A[i] = limit * (2.0f * drand48() - 1.0f);
}

static void lora_adapter_free(LoRAAdapter *adp) {
    free(adp->A); free(adp->B);
    free(adp->dA); free(adp->dB);
    adam_free(&adp->sA); adam_free(&adp->sB);
}

static void lora_init(LayerLoRA *lora, int nlayers, int dim, int rank) {
    for (int L = 0; L < nlayers; L++) {
        lora_adapter_init(&lora[L].wq, dim, rank);
        lora_adapter_init(&lora[L].wv, dim, rank);
    }
}

static void lora_free(LayerLoRA *lora, int nlayers) {
    for (int L = 0; L < nlayers; L++) {
        lora_adapter_free(&lora[L].wq);
        lora_adapter_free(&lora[L].wv);
    }
}

// ── Core operations ──

// Merge: W_eff = W_frozen + (alpha/rank) * B @ A
// All matrices are [dim × dim] (Wq, Wv), A is [rank × dim], B is [dim × rank]
static void lora_merge(float *W_eff, const float *W_frozen, const LoRAAdapter *adp,
                       int dim, int rank, float alpha) {
    memcpy(W_eff, W_frozen, (size_t)dim * dim * 4);
    float scale = alpha / rank;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                dim, dim, rank, scale,
                adp->B, rank, adp->A, dim,
                1.0f, W_eff, dim);
}

// Extract LoRA gradients from full dW: dA = s·B^T·dW, dB = s·dW·A^T
static void lora_extract_grads(LoRAAdapter *adp, const float *dW,
                               int dim, int rank, float alpha) {
    float scale = alpha / rank;
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                rank, dim, dim, scale,
                adp->B, rank, dW, dim,
                0.0f, adp->dA, dim);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                dim, rank, dim, scale,
                dW, dim, adp->A, dim,
                0.0f, adp->dB, rank);
}

// ── Checkpoint ──

static void lora_save(const char *path, int step, int total_steps, float lr, float loss,
                      double cc, double ct, double cw, int cs, int cb, int adam_t,
                      const LayerLoRA *lora, int nlayers, int dim, int rank, float alpha) {
    FILE *f = fopen(path, "wb");
    LoRACkptHdr h = {0};
    h.magic = LORA_CKPT_MAGIC; h.version = LORA_CKPT_VERSION;
    h.step = step; h.total_steps = total_steps;
    h.n_layers = nlayers; h.dim = dim; h.rank = rank;
    h.alpha = alpha; h.lr = lr; h.loss = loss; h.adam_t = adam_t;
    h.cum_compile = cc; h.cum_train = ct; h.cum_wall = cw;
    h.cum_steps = cs; h.cum_batches = cb;
    fwrite(&h, sizeof(h), 1, f);
    size_t a_sz = (size_t)rank * dim, b_sz = (size_t)dim * rank;
    for (int L = 0; L < nlayers; L++) {
        const LoRAAdapter *q = &lora[L].wq, *v = &lora[L].wv;
        fwrite(q->A, 4, a_sz, f); fwrite(q->B, 4, b_sz, f);
        fwrite(q->sA.m, 4, a_sz, f); fwrite(q->sA.v, 4, a_sz, f);
        fwrite(q->sB.m, 4, b_sz, f); fwrite(q->sB.v, 4, b_sz, f);
        fwrite(v->A, 4, a_sz, f); fwrite(v->B, 4, b_sz, f);
        fwrite(v->sA.m, 4, a_sz, f); fwrite(v->sA.v, 4, a_sz, f);
        fwrite(v->sB.m, 4, b_sz, f); fwrite(v->sB.v, 4, b_sz, f);
    }
    fclose(f);
}

static bool lora_load(const char *path, int *step, int *total_steps, float *lr, float *loss,
                      double *cc, double *ct, double *cw, int *cs, int *cb, int *adam_t,
                      LayerLoRA *lora, int nlayers, int dim, int rank) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    LoRACkptHdr h;
    fread(&h, sizeof(h), 1, f);
    if (h.magic != LORA_CKPT_MAGIC || h.version != LORA_CKPT_VERSION) { fclose(f); return false; }
    if (h.n_layers != nlayers || h.dim != dim || h.rank != rank) {
        printf("LoRA ckpt mismatch: layers=%d/%d dim=%d/%d rank=%d/%d\n",
               h.n_layers, nlayers, h.dim, dim, h.rank, rank);
        fclose(f); return false;
    }
    *step = h.step; *total_steps = h.total_steps;
    *lr = h.lr; *loss = h.loss; *adam_t = h.adam_t;
    *cc = h.cum_compile; *ct = h.cum_train; *cw = h.cum_wall;
    *cs = h.cum_steps; *cb = h.cum_batches;
    size_t a_sz = (size_t)rank * dim, b_sz = (size_t)dim * rank;
    for (int L = 0; L < nlayers; L++) {
        LoRAAdapter *q = &lora[L].wq, *v = &lora[L].wv;
        fread(q->A, 4, a_sz, f); fread(q->B, 4, b_sz, f);
        fread(q->sA.m, 4, a_sz, f); fread(q->sA.v, 4, a_sz, f);
        fread(q->sB.m, 4, b_sz, f); fread(q->sB.v, 4, b_sz, f);
        fread(v->A, 4, a_sz, f); fread(v->B, 4, b_sz, f);
        fread(v->sA.m, 4, a_sz, f); fread(v->sA.v, 4, a_sz, f);
        fread(v->sB.m, 4, b_sz, f); fread(v->sB.v, 4, b_sz, f);
    }
    fclose(f);
    return true;
}
