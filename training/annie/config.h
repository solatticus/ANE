// config.h — Parameterized model config for Annie (Qwen2.5-3B on ANE)
// Replaces hardcoded #define DIM/HIDDEN/etc with runtime AnnieConfig struct.
#pragma once
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#import <mach/mach_time.h>
#import <Accelerate/Accelerate.h>
#include <math.h>
#include <unistd.h>
#include <dispatch/dispatch.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <arm_neon.h>

// ── Model configuration ──

typedef struct {
    int dim, hidden_dim, n_layers;
    int n_heads, n_kv_heads, head_dim;
    int vocab_size, max_seq_len;
    float rope_theta, rms_norm_eps;
    float lora_alpha;
    int lora_rank;
    bool tie_embeddings, qkv_bias;
} AnnieConfig;

// Derived dimensions
static inline int ac_kv_dim(const AnnieConfig *c) { return c->n_kv_heads * c->head_dim; }
static inline int ac_gqa_ratio(const AnnieConfig *c) { return c->n_heads / c->n_kv_heads; }

// Weight sizes per layer
static inline int ac_wq_sz(const AnnieConfig *c) { return c->dim * c->dim; }
static inline int ac_wk_sz(const AnnieConfig *c) { return c->dim * ac_kv_dim(c); }
static inline int ac_wv_sz(const AnnieConfig *c) { return c->dim * ac_kv_dim(c); }
static inline int ac_wo_sz(const AnnieConfig *c) { return c->dim * c->dim; }
static inline int ac_w1_sz(const AnnieConfig *c) { return c->hidden_dim * c->dim; }
static inline int ac_w2_sz(const AnnieConfig *c) { return c->dim * c->hidden_dim; }
static inline int ac_w3_sz(const AnnieConfig *c) { return c->hidden_dim * c->dim; }

// Qwen2.5-3B preset
static AnnieConfig annie_config_qwen3b(void) {
    return (AnnieConfig){
        .dim = 2048, .hidden_dim = 11008, .n_layers = 36,
        .n_heads = 16, .n_kv_heads = 2, .head_dim = 128,
        .vocab_size = 151936, .max_seq_len = 256,
        .rope_theta = 1000000.0f, .rms_norm_eps = 1e-6f,
        .lora_alpha = 8.0f, .lora_rank = 8,
        .tie_embeddings = true, .qkv_bias = true,
    };
}

// ── Per-layer structs ──

typedef struct {
    float *Wq, *Wk, *Wv, *Wo;           // Projection weights
    float *bq, *bk, *bv, *bo;           // QKV+O bias (Qwen2.5)
    float *W1, *W2, *W3;                // FFN weights
    float *rms_att, *rms_ffn;           // RMSNorm weights
} AnnieLayerWeights;

typedef struct { float *m, *v; size_t n; } AdamState;

typedef struct {
    float *layer_in, *xnorm;
    float *Q, *K, *V;                   // Q: [dim,S], K/V: [kv_dim,S]
    float *attn_out, *o_out;
    float *x2, *x2norm;
    float *h1, *h3, *silu_out, *ffn_out;
} AnnieLayerActs;

typedef struct {
    float *Wq, *Wv;                     // Only LoRA targets get gradients
} AnnieLayerGrads;

typedef struct { void *model; IOSurfaceRef ioIn, ioOut; void *request; void *tmpDir; } Kern;

// ── Binary weight file header ──

#define ANNIE_WEIGHT_MAGIC 0x414E4E49   // "ANNI"
#define ANNIE_WEIGHT_VERSION 1

typedef struct {
    int magic, version;
    AnnieConfig config;
    int pad[4];
} AnnieWeightHdr;

// ── Checkpoint header ──

#define ANNIE_CKPT_MAGIC  0x41434B50    // "ACKP"
#define ANNIE_CKPT_VERSION 1

typedef struct {
    int magic, version, step, total_steps;
    int n_layers, dim, rank;
    float alpha, lr, loss;
    int adam_t;
    double cum_compile, cum_train, cum_wall;
    int cum_steps, cum_batches;
    int pad[2];
} AnnieCkptHdr;

// ── Globals ──

static Class g_D, g_I, g_AR, g_AIO;
static mach_timebase_info_data_t g_tb;
static int g_compile_count = 0;

static void ane_init(void) {
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    g_D  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    g_I  = NSClassFromString(@"_ANEInMemoryModel");
    g_AR = NSClassFromString(@"_ANERequest");
    g_AIO= NSClassFromString(@"_ANEIOSurfaceObject");
}

static double tb_ms(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }

// ── Alloc helpers ──

static AdamState adam_alloc(size_t n) {
    AdamState s; s.m = (float*)calloc(n, 4); s.v = (float*)calloc(n, 4); s.n = n; return s;
}
static void adam_free(AdamState *s) { free(s->m); free(s->v); }

static AnnieLayerWeights annie_weights_alloc(const AnnieConfig *c) {
    int kv = ac_kv_dim(c);
    AnnieLayerWeights w;
    w.Wq = (float*)malloc(ac_wq_sz(c) * 4); w.Wk = (float*)malloc(ac_wk_sz(c) * 4);
    w.Wv = (float*)malloc(ac_wv_sz(c) * 4); w.Wo = (float*)malloc(ac_wo_sz(c) * 4);
    w.W1 = (float*)malloc(ac_w1_sz(c) * 4); w.W2 = (float*)malloc(ac_w2_sz(c) * 4);
    w.W3 = (float*)malloc(ac_w3_sz(c) * 4);
    w.rms_att = (float*)malloc(c->dim * 4); w.rms_ffn = (float*)malloc(c->dim * 4);
    if (c->qkv_bias) {
        w.bq = (float*)calloc(c->dim, 4);   w.bk = (float*)calloc(kv, 4);
        w.bv = (float*)calloc(kv, 4);       w.bo = (float*)calloc(c->dim, 4);
    } else {
        w.bq = w.bk = w.bv = w.bo = NULL;
    }
    return w;
}

static void annie_weights_free(AnnieLayerWeights *w) {
    free(w->Wq); free(w->Wk); free(w->Wv); free(w->Wo);
    free(w->W1); free(w->W2); free(w->W3);
    free(w->rms_att); free(w->rms_ffn);
    free(w->bq); free(w->bk); free(w->bv); free(w->bo);
}

static AnnieLayerActs annie_acts_alloc(const AnnieConfig *c) {
    int S = c->max_seq_len, D = c->dim, H = c->hidden_dim, kv = ac_kv_dim(c);
    AnnieLayerActs a;
    a.layer_in = (float*)malloc(S * D * 4);
    a.xnorm    = (float*)malloc(S * D * 4);
    a.Q        = (float*)malloc(S * D * 4);
    a.K        = (float*)malloc(S * kv * 4);
    a.V        = (float*)malloc(S * kv * 4);
    a.attn_out = (float*)malloc(S * D * 4);
    a.o_out    = (float*)malloc(S * D * 4);
    a.x2       = (float*)malloc(S * D * 4);
    a.x2norm   = (float*)malloc(S * D * 4);
    a.h1       = (float*)malloc(S * H * 4);
    a.h3       = (float*)malloc(S * H * 4);
    a.silu_out = (float*)malloc(S * H * 4);
    a.ffn_out  = (float*)malloc(S * D * 4);
    return a;
}

static void annie_acts_free(AnnieLayerActs *a) {
    free(a->layer_in); free(a->xnorm);
    free(a->Q); free(a->K); free(a->V);
    free(a->attn_out); free(a->o_out);
    free(a->x2); free(a->x2norm);
    free(a->h1); free(a->h3); free(a->silu_out); free(a->ffn_out);
}

static AnnieLayerGrads annie_grads_alloc(const AnnieConfig *c) {
    AnnieLayerGrads g;
    g.Wq = (float*)calloc(ac_wq_sz(c), 4);
    g.Wv = (float*)calloc(ac_wv_sz(c), 4);
    return g;
}

static void annie_grads_zero(AnnieLayerGrads *g, const AnnieConfig *c) {
    memset(g->Wq, 0, ac_wq_sz(c) * 4);
    memset(g->Wv, 0, ac_wv_sz(c) * 4);
}

static void annie_grads_free(AnnieLayerGrads *g) {
    free(g->Wq); free(g->Wv);
}
