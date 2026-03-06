// cpu_ops.h — CPU operations: RMSNorm, GQA attention, RoPE, cross-entropy, Adam, embedding
// All parameterized through AnnieConfig — no hardcoded dimensions.
#pragma once
#include "config.h"

// ── RMSNorm ──

static void rmsnorm(float *out, const float *x, const float *w, int d, int S, float eps) {
    float *ss = (float*)calloc(S, 4);
    float *tmp = (float*)malloc(S * 4);
    for (int i = 0; i < d; i++) {
        vDSP_vmul(x + i * S, 1, x + i * S, 1, tmp, 1, (vDSP_Length)S);
        vDSP_vadd(tmp, 1, ss, 1, ss, 1, (vDSP_Length)S);
    }
    float invd = 1.0f / d;
    vDSP_vsmsa(ss, 1, &invd, &eps, ss, 1, (vDSP_Length)S);
    int n = S; vvrsqrtf(ss, ss, &n);
    for (int i = 0; i < d; i++) {
        vDSP_vmul(x + i * S, 1, ss, 1, out + i * S, 1, (vDSP_Length)S);
        vDSP_vsmul(out + i * S, 1, &w[i], out + i * S, 1, (vDSP_Length)S);
    }
    free(ss); free(tmp);
}

static void rmsnorm_bwd(float *dx, float *dw, const float *dy, const float *x,
                        const float *w, int d, int S, float eps) {
    float *tmp = (float*)malloc(S * 4);
    float *ss = (float*)calloc(S, 4);
    for (int i = 0; i < d; i++) {
        vDSP_vmul(x + i * S, 1, x + i * S, 1, tmp, 1, (vDSP_Length)S);
        vDSP_vadd(tmp, 1, ss, 1, ss, 1, (vDSP_Length)S);
    }
    float invd = 1.0f / d;
    vDSP_vsmsa(ss, 1, &invd, &eps, ss, 1, (vDSP_Length)S);
    float *rrms = (float*)malloc(S * 4);
    int n = S; vvrsqrtf(rrms, ss, &n);
    float *dot = (float*)calloc(S, 4);
    for (int i = 0; i < d; i++) {
        vDSP_vmul(dy + i * S, 1, x + i * S, 1, tmp, 1, (vDSP_Length)S);
        vDSP_vsma(tmp, 1, &w[i], dot, 1, dot, 1, (vDSP_Length)S);
    }
    vDSP_vmul(rrms, 1, rrms, 1, ss, 1, (vDSP_Length)S);
    vDSP_vsmul(ss, 1, &invd, ss, 1, (vDSP_Length)S);
    vDSP_vmul(dot, 1, ss, 1, dot, 1, (vDSP_Length)S);
    for (int i = 0; i < d; i++) {
        vDSP_vmul(x + i * S, 1, dot, 1, tmp, 1, (vDSP_Length)S);
        vDSP_vsub(tmp, 1, dy + i * S, 1, tmp, 1, (vDSP_Length)S);
        vDSP_vmul(tmp, 1, rrms, 1, tmp, 1, (vDSP_Length)S);
        vDSP_vsmul(tmp, 1, &w[i], dx + i * S, 1, (vDSP_Length)S);
        vDSP_vmul(dy + i * S, 1, x + i * S, 1, tmp, 1, (vDSP_Length)S);
        vDSP_vmul(tmp, 1, rrms, 1, tmp, 1, (vDSP_Length)S);
        float s; vDSP_sve(tmp, 1, &s, (vDSP_Length)S);
        dw[i] += s;
    }
    free(ss); free(rrms); free(dot); free(tmp);
}

// ── RoPE (parameterized theta and head_dim) ──

static void rope_apply(float *Q, float *K, int n_heads, int n_kv_heads,
                       int head_dim, int S, float theta) {
    // Q layout: [dim, S] where dim = n_heads * head_dim
    // K layout: [kv_dim, S] where kv_dim = n_kv_heads * head_dim
    // RoPE applies per-head: pairs (2i, 2i+1) within each head_dim block
    for (int t = 0; t < S; t++) {
        // Apply to Q heads
        for (int h = 0; h < n_heads; h++) {
            for (int i = 0; i < head_dim / 2; i++) {
                float freq = 1.0f / powf(theta, (float)(2 * i) / head_dim);
                float angle = (float)t * freq;
                float cos_a = cosf(angle), sin_a = sinf(angle);
                int idx0 = (h * head_dim + 2 * i) * S + t;
                int idx1 = (h * head_dim + 2 * i + 1) * S + t;
                float q0 = Q[idx0], q1 = Q[idx1];
                Q[idx0] = q0 * cos_a - q1 * sin_a;
                Q[idx1] = q0 * sin_a + q1 * cos_a;
            }
        }
        // Apply to K heads
        for (int h = 0; h < n_kv_heads; h++) {
            for (int i = 0; i < head_dim / 2; i++) {
                float freq = 1.0f / powf(theta, (float)(2 * i) / head_dim);
                float angle = (float)t * freq;
                float cos_a = cosf(angle), sin_a = sinf(angle);
                int idx0 = (h * head_dim + 2 * i) * S + t;
                int idx1 = (h * head_dim + 2 * i + 1) * S + t;
                float k0 = K[idx0], k1 = K[idx1];
                K[idx0] = k0 * cos_a - k1 * sin_a;
                K[idx1] = k0 * sin_a + k1 * cos_a;
            }
        }
    }
}

static void rope_backward(float *dQ, float *dK, int n_heads, int n_kv_heads,
                          int head_dim, int S, float theta) {
    // RoPE backward is the inverse rotation (negate sin)
    for (int t = 0; t < S; t++) {
        for (int h = 0; h < n_heads; h++) {
            for (int i = 0; i < head_dim / 2; i++) {
                float freq = 1.0f / powf(theta, (float)(2 * i) / head_dim);
                float angle = (float)t * freq;
                float cos_a = cosf(angle), sin_a = -sinf(angle);
                int idx0 = (h * head_dim + 2 * i) * S + t;
                int idx1 = (h * head_dim + 2 * i + 1) * S + t;
                float q0 = dQ[idx0], q1 = dQ[idx1];
                dQ[idx0] = q0 * cos_a - q1 * sin_a;
                dQ[idx1] = q0 * sin_a + q1 * cos_a;
            }
        }
        for (int h = 0; h < n_kv_heads; h++) {
            for (int i = 0; i < head_dim / 2; i++) {
                float freq = 1.0f / powf(theta, (float)(2 * i) / head_dim);
                float angle = (float)t * freq;
                float cos_a = cosf(angle), sin_a = -sinf(angle);
                int idx0 = (h * head_dim + 2 * i) * S + t;
                int idx1 = (h * head_dim + 2 * i + 1) * S + t;
                float k0 = dK[idx0], k1 = dK[idx1];
                dK[idx0] = k0 * cos_a - k1 * sin_a;
                dK[idx1] = k0 * sin_a + k1 * cos_a;
            }
        }
    }
}

// ── Add bias (broadcast over sequence dim) ──

static void add_bias(float *x, const float *bias, int channels, int S) {
    if (!bias) return;
    for (int c = 0; c < channels; c++) {
        float b = bias[c];
        vDSP_vsadd(x + c * S, 1, &b, x + c * S, 1, (vDSP_Length)S);
    }
}

// ── GQA Attention (CPU, causal) ──
// Q: [n_heads * head_dim, S], K: [n_kv_heads * head_dim, S], V: same as K
// Output: attn_out [n_heads * head_dim, S]
// For each query head h, use KV head h / gqa_ratio

static void gqa_attention(float *attn_out, const float *Q, const float *K, const float *V,
                          int n_heads, int n_kv_heads, int head_dim, int S) {
    int gqa_ratio = n_heads / n_kv_heads;
    float scale = 1.0f / sqrtf((float)head_dim);
    float *scores = (float*)malloc(S * S * 4);

    for (int h = 0; h < n_heads; h++) {
        int kv_h = h / gqa_ratio;
        const float *q_head = Q + h * head_dim * S;       // [head_dim, S]
        const float *k_head = K + kv_h * head_dim * S;    // [head_dim, S]
        const float *v_head = V + kv_h * head_dim * S;    // [head_dim, S]
        float *out_head = attn_out + h * head_dim * S;     // [head_dim, S]

        // scores[t1, t2] = scale * sum_d(Q[d,t1] * K[d,t2])
        // = scale * Q^T @ K where Q/K are [head_dim, S]
        // scores = scale * Q_head^T[S, head_dim] @ K_head[head_dim, S] = [S, S]
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    S, S, head_dim, scale,
                    q_head, S, k_head, S, 0.0f, scores, S);

        // Causal mask + softmax per row
        for (int t1 = 0; t1 < S; t1++) {
            float *row = scores + t1 * S;
            // Mask future positions
            for (int t2 = t1 + 1; t2 < S; t2++) row[t2] = -1e9f;
            // Softmax
            float maxv; vDSP_maxv(row, 1, &maxv, (vDSP_Length)S);
            float neg_max = -maxv;
            vDSP_vsadd(row, 1, &neg_max, row, 1, (vDSP_Length)S);
            int n = S; vvexpf(row, row, &n);
            float sum; vDSP_sve(row, 1, &sum, (vDSP_Length)S);
            float inv = 1.0f / sum;
            vDSP_vsmul(row, 1, &inv, row, 1, (vDSP_Length)S);
        }

        // out = V @ scores^T = V[head_dim, S] @ scores^T[S, S] = [head_dim, S]
        // Actually: out[d, t1] = sum_t2(V[d, t2] * scores[t1, t2])
        // = V[head_dim, S] @ scores^T[S, S]
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    head_dim, S, S, 1.0f,
                    v_head, S, scores, S, 0.0f, out_head, S);
    }
    free(scores);
}

// ── GQA Attention Backward (CPU) ──
// Computes dQ, dK, dV from d_attn_out
// d_attn_out: [dim, S], Q: [dim, S], K: [kv_dim, S], V: [kv_dim, S]

static void gqa_attention_backward(float *dQ, float *dK, float *dV,
                                   const float *d_attn_out, const float *Q,
                                   const float *K, const float *V,
                                   int n_heads, int n_kv_heads, int head_dim, int S) {
    int gqa_ratio = n_heads / n_kv_heads;
    float scale = 1.0f / sqrtf((float)head_dim);
    float *scores = (float*)malloc(S * S * 4);
    float *dscores = (float*)malloc(S * S * 4);

    // Zero dK, dV (they accumulate from multiple query heads)
    int kv_dim = n_kv_heads * head_dim;
    memset(dK, 0, kv_dim * S * 4);
    memset(dV, 0, kv_dim * S * 4);

    for (int h = 0; h < n_heads; h++) {
        int kv_h = h / gqa_ratio;
        const float *q_head = Q + h * head_dim * S;
        const float *k_head = K + kv_h * head_dim * S;
        const float *v_head = V + kv_h * head_dim * S;
        const float *dout_head = d_attn_out + h * head_dim * S;
        float *dq_head = dQ + h * head_dim * S;
        float *dk_head = dK + kv_h * head_dim * S;
        float *dv_head = dV + kv_h * head_dim * S;

        // Recompute forward: scores = scale * Q^T @ K
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    S, S, head_dim, scale,
                    q_head, S, k_head, S, 0.0f, scores, S);
        // Causal mask + softmax
        for (int t1 = 0; t1 < S; t1++) {
            float *row = scores + t1 * S;
            for (int t2 = t1 + 1; t2 < S; t2++) row[t2] = -1e9f;
            float maxv; vDSP_maxv(row, 1, &maxv, (vDSP_Length)S);
            float neg_max = -maxv;
            vDSP_vsadd(row, 1, &neg_max, row, 1, (vDSP_Length)S);
            int n = S; vvexpf(row, row, &n);
            float sum; vDSP_sve(row, 1, &sum, (vDSP_Length)S);
            float inv = 1.0f / sum;
            vDSP_vsmul(row, 1, &inv, row, 1, (vDSP_Length)S);
        }
        // Now scores contains softmax probabilities [S, S]

        // dV += scores^T @ dout^T ... actually:
        // Forward: out[d,t1] = sum_t2(V[d,t2] * scores[t1,t2])
        // dV[d,t2] += sum_t1(dout[d,t1] * scores[t1,t2])
        //           = dout[d,:] @ scores[:,t2] per d
        // dV += dout[hd,S] @ scores^T[S,S]  — but accumulated across query heads
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    head_dim, S, S, 1.0f,
                    dout_head, S, scores, S, 1.0f, dv_head, S);

        // dp[t1,t2] = sum_d(dout[d,t1] * V[d,t2]) = dout^T @ V
        // dp = dout^T[S,hd] @ V[hd,S] = [S,S]
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    S, S, head_dim, 1.0f,
                    dout_head, S, v_head, S, 0.0f, dscores, S);

        // Softmax backward: ds[t1,t2] = probs[t1,t2] * (dp[t1,t2] - sum_j(probs[t1,j]*dp[t1,j]))
        for (int t1 = 0; t1 < S; t1++) {
            float *p_row = scores + t1 * S;
            float *dp_row = dscores + t1 * S;
            float dot; vDSP_dotpr(p_row, 1, dp_row, 1, &dot, (vDSP_Length)S);
            for (int t2 = 0; t2 < S; t2++)
                dp_row[t2] = p_row[t2] * (dp_row[t2] - dot);
        }
        // Scale
        vDSP_vsmul(dscores, 1, &scale, dscores, 1, (vDSP_Length)(S * S));

        // dQ[d,t1] = sum_t2(ds[t1,t2] * K[d,t2]) = K @ ds^T
        // dQ += K[hd,S] @ ds^T[S,S]
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    head_dim, S, S, 1.0f,
                    k_head, S, dscores, S, 0.0f, dq_head, S);

        // dK[d,t2] += sum_t1(ds[t1,t2] * Q[d,t1]) = Q @ ds
        // dK += Q[hd,S] @ ds[S,S]
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    head_dim, S, S, 1.0f,
                    q_head, S, dscores, S, 1.0f, dk_head, S);
    }
    free(scores); free(dscores);
}

// ── Adam optimizer ──

static void adam_update(float *w, const float *g, AdamState *s,
                        int t, float lr, float b1, float b2, float eps) {
    float bc1 = 1.0f - powf(b1, t), bc2 = 1.0f - powf(b2, t);
    for (size_t i = 0; i < s->n; i++) {
        s->m[i] = b1 * s->m[i] + (1 - b1) * g[i];
        s->v[i] = b2 * s->v[i] + (1 - b2) * g[i] * g[i];
        float mh = s->m[i] / bc1, vh = s->v[i] / bc2;
        w[i] -= lr * mh / (sqrtf(vh) + eps);
    }
}

// ── Cross-entropy loss (column-major logits[V, S]) ──

static float cross_entropy_loss(float *dlogits, const float *logits,
                                const uint32_t *targets, int V, int S) {
    float *col = (float*)malloc(V * 4);
    float total_loss = 0;
    float invS = 1.0f / S;
    for (int t = 0; t < S; t++) {
        cblas_scopy(V, logits + t, S, col, 1);
        float maxv; vDSP_maxv(col, 1, &maxv, (vDSP_Length)V);
        float neg_max = -maxv;
        vDSP_vsadd(col, 1, &neg_max, col, 1, (vDSP_Length)V);
        int n = V; vvexpf(col, col, &n);
        float sum; vDSP_sve(col, 1, &sum, (vDSP_Length)V);
        float inv_sum = 1.0f / sum;
        vDSP_vsmul(col, 1, &inv_sum, col, 1, (vDSP_Length)V);
        int tgt = targets[t];
        total_loss -= logf(col[tgt] + 1e-10f);
        col[tgt] -= 1.0f;
        vDSP_vsmul(col, 1, &invS, col, 1, (vDSP_Length)V);
        cblas_scopy(V, col, 1, dlogits + t, S);
    }
    free(col);
    return total_loss / S;
}

// ── Embedding (uint32 tokens for 151K vocab) ──

static void embed_lookup(float *x, const float *embed, const uint32_t *tokens,
                         int dim, int seq) {
    for (int t = 0; t < seq; t++) {
        int tok = tokens[t];
        for (int d = 0; d < dim; d++)
            x[d * seq + t] = embed[tok * dim + d];
    }
}

static void embed_backward(float *d_embed, const float *dx, const uint32_t *tokens,
                            int dim, int seq) {
    for (int t = 0; t < seq; t++) {
        int tok = tokens[t];
        for (int d = 0; d < dim; d++)
            d_embed[tok * dim + d] += dx[d * seq + t];
    }
}

// ── Weight transpose helper ──

static void transpose_weight(float *dst, const float *src, int rows, int cols) {
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            dst[c * rows + r] = src[r * cols + c];
}
