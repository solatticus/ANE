// backward.h — Backward pass orchestration for Annie
// Per-layer (reverse order):
//   FFN backward: W2^T (ANE), SiLU derivative (CPU), W1^T+W3^T (ANE)
//   RMSNorm2 bwd (CPU)
//   Attn backward: Wo^T (ANE), SDPA bwd (CPU), bias bwd (CPU), RoPE bwd (CPU)
//   dWq/dWv accumulation (async CPU cblas for LoRA)
//   QKV backward: Wq^T (ANE), Wk^T+Wv^T (CPU — small KV dim)
//   RMSNorm1 bwd (CPU)
#pragma once
#include "mil_dynamic.h"
#include "cpu_ops.h"

// Backward pass for one transformer layer.
// dy: [dim, S] input gradient (modified in place — becomes dx for next layer down)
// Accumulates dWq, dWv into grads (for LoRA extraction later).
static void annie_backward_layer(
    float *dy,                           // [dim, S] in/out gradient
    const AnnieLayerActs *ac,
    const AnnieLayerWeights *lw,
    AnnieLayerGrads *gr,
    AnnieKernels *ak,
    const AnnieConfig *cfg,
    // Work buffers (caller-allocated, reused across layers)
    float *dffn,       // [dim, S]
    float *dx_ffn,     // [dim, S]
    float *dx2,        // [dim, S]
    float *da_buf,     // [dim, S]
    float *dq,         // [dim, S]
    float *dk_buf,     // [kv_dim, S]
    float *dv,         // [kv_dim, S]
    float *dh1,        // [hidden, S]
    float *dh3,        // [hidden, S]
    float *dsilu,      // [hidden, S]
    float *silu_tmp,   // [hidden, S]
    float *silu_tmp2,  // [hidden, S]
    float *dx_attn,    // [dim, S]
    dispatch_group_t dw_grp,
    dispatch_queue_t dw_q)
{
    int D = cfg->dim, H = cfg->hidden_dim, S = cfg->max_seq_len;
    int kv = ac_kv_dim(cfg);

    memcpy(dffn, dy, S * D * 4);

    // ══════ FFN BACKWARD ══════

    // 1. dffn @ W2^T → dsilu_raw (ANE, dimToHidden kernel)
    //    W2 is [dim, hidden] stored row-major.
    //    Kernel: act=dffn[dim,S], W=W2[dim,hidden] → output [hidden,S]
    //    Effectively: W2^T[hidden,dim] @ dffn[dim,S] = [hidden,S]
    io_write_dyn(ak->dimToHidden->ioIn, dffn, D, S, lw->W2, H);
    ane_eval(ak->dimToHidden);
    io_read_dyn(ak->dimToHidden->ioOut, dsilu, H, S);

    // 2. SiLU derivative (CPU)
    {
        int n = H * S;
        float minus1 = -1.0f, one = 1.0f;
        // sig = sigmoid(h1)
        vDSP_vsmul(ac->h1, 1, &minus1, silu_tmp, 1, (vDSP_Length)n);
        vvexpf(silu_tmp, silu_tmp, &n);
        vDSP_vsadd(silu_tmp, 1, &one, silu_tmp, 1, (vDSP_Length)n);
        vvrecf(silu_tmp, silu_tmp, &n);  // silu_tmp = sig

        // dh3 = dsilu * silu(h1) = dsilu * h1 * sig
        vDSP_vmul(ac->h1, 1, silu_tmp, 1, dh3, 1, (vDSP_Length)n);
        vDSP_vmul(dsilu, 1, dh3, 1, dh3, 1, (vDSP_Length)n);

        // dh1 = dsilu * h3 * silu'(h1) where silu'(x) = sig(x) * (1 + x*(1-sig(x)))
        // silu_tmp2 = sig * (1 + h1 * (1 - sig))
        vDSP_vsadd(silu_tmp, 1, &minus1, silu_tmp2, 1, (vDSP_Length)n);
        vDSP_vneg(silu_tmp2, 1, silu_tmp2, 1, (vDSP_Length)n);        // 1-sig
        vDSP_vmul(ac->h1, 1, silu_tmp2, 1, silu_tmp2, 1, (vDSP_Length)n); // h1*(1-sig)
        vDSP_vsadd(silu_tmp2, 1, &one, silu_tmp2, 1, (vDSP_Length)n);     // 1+h1*(1-sig)
        vDSP_vmul(silu_tmp, 1, silu_tmp2, 1, silu_tmp2, 1, (vDSP_Length)n); // sig*(1+h1*(1-sig))
        vDSP_vmul(dsilu, 1, ac->h3, 1, dh1, 1, (vDSP_Length)n);           // dsilu*h3
        vDSP_vmul(dh1, 1, silu_tmp2, 1, dh1, 1, (vDSP_Length)n);          // dsilu*h3*silu'
    }

    // 3. dh1 @ W1^T → dx1 (ANE, hiddenToDim kernel)
    //    W1 is [hidden, dim]. Kernel: act=dh1[hidden,S], W=W1[hidden,dim] → [dim,S]
    io_write_dyn(ak->hiddenToDim->ioIn, dh1, H, S, lw->W1, D);
    ane_eval(ak->hiddenToDim);
    io_read_dyn(ak->hiddenToDim->ioOut, dx_ffn, D, S);

    // 4. dh3 @ W3^T → dx3 (ANE, same hiddenToDim kernel)
    io_write_dyn(ak->hiddenToDim->ioIn, dh3, H, S, lw->W3, D);
    ane_eval(ak->hiddenToDim);
    float *dx3_buf = (float*)malloc(D * S * 4);
    io_read_dyn(ak->hiddenToDim->ioOut, dx3_buf, D, S);

    // dx_ffn += dx3
    vDSP_vadd(dx_ffn, 1, dx3_buf, 1, dx_ffn, 1, (vDSP_Length)(D * S));
    free(dx3_buf);

    // ══════ RMSNorm2 BACKWARD (frozen — compute dx only) ══════

    memset(dx2, 0, S * D * 4);
    float drms_ffn_dummy[D];
    memset(drms_ffn_dummy, 0, D * 4);
    rmsnorm_bwd(dx2, drms_ffn_dummy, dx_ffn, ac->x2, lw->rms_ffn, D, S, cfg->rms_norm_eps);
    // Add skip connection gradient
    for (int i = 0; i < S * D; i++) dx2[i] += dy[i];

    // ══════ ATTENTION BACKWARD ══════

    // 5. Wo^T backward (ANE, dimToDim): dx2 @ Wo → da
    //    Kernel: act=dx2[dim,S], W=Wo[dim,dim] → [dim,S]
    io_write_dyn(ak->dimToDim->ioIn, dx2, D, S, lw->Wo, D);
    ane_eval(ak->dimToDim);
    io_read_dyn(ak->dimToDim->ioOut, da_buf, D, S);

    // 6. GQA attention backward (CPU) — da → dQ, dK, dV
    //    Must undo RoPE first: apply inverse rotation to saved Q, K
    //    Actually we need pre-RoPE Q, K for the backward... but we saved post-RoPE.
    //    Solution: recompute pre-RoPE Q, K from xnorm. OR: undo RoPE on da.
    //    Standard approach: backward through attention gives dQ_rope, dK_rope,
    //    then backward through RoPE gives dQ_proj, dK_proj.
    gqa_attention_backward(dq, dk_buf, dv, da_buf,
                           ac->Q, ac->K, ac->V,
                           cfg->n_heads, cfg->n_kv_heads, cfg->head_dim, S);

    // 7. RoPE backward (CPU): undo rotation on dQ, dK
    rope_backward(dq, dk_buf, cfg->n_heads, cfg->n_kv_heads, cfg->head_dim, S, cfg->rope_theta);

    // 8. Bias backward: skip (bias grads not needed — frozen)

    // 9. dWq and dWv accumulation (async CPU cblas — LoRA targets)
    float *capt_dq = (float*)malloc(S * D * 4);  memcpy(capt_dq, dq, S * D * 4);
    float *capt_dv = (float*)malloc(S * kv * 4); memcpy(capt_dv, dv, S * kv * 4);
    float *capt_xn = (float*)malloc(S * D * 4);  memcpy(capt_xn, ac->xnorm, S * D * 4);
    float *gr_wq = gr->Wq, *gr_wv = gr->Wv;
    dispatch_group_async(dw_grp, dw_q, ^{
        // dWq += dq @ xnorm^T: [dim,S] @ [S,dim] → [dim,dim]
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, D, D, S,
                    1.0f, capt_dq, S, capt_xn, S, 1.0f, gr_wq, D);
        // dWv += dv @ xnorm^T: [kv_dim,S] @ [S,dim] → [kv_dim,dim]
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, kv, D, S,
                    1.0f, capt_dv, S, capt_xn, S, 1.0f, gr_wv, D);
        free(capt_dq); free(capt_dv); free(capt_xn);
    });

    // 10. QKV backward: dq@Wq^T + dk@Wk^T + dv@Wv^T → dx_attn
    //     Wq^T is large (dim→dim) — use ANE dimToDim kernel
    io_write_dyn(ak->dimToDim->ioIn, dq, D, S, lw->Wq, D);
    ane_eval(ak->dimToDim);
    io_read_dyn(ak->dimToDim->ioOut, dx_attn, D, S);

    //     Wk^T and Wv^T are small (kv_dim→dim) — CPU cblas
    //     dk[kv,S] @ Wk[kv,dim] → dx_k[dim,S] (computed as Wk^T @ dk)
    //     Using cblas: C[dim,S] += Wk^T[dim,kv] @ dk[kv,S]
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                D, S, kv, 1.0f, lw->Wk, D, dk_buf, S, 1.0f, dx_attn, S);
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                D, S, kv, 1.0f, lw->Wv, D, dv, S, 1.0f, dx_attn, S);

    // ══════ RMSNorm1 BACKWARD (frozen — compute dx only) ══════

    float *dx_rms1 = (float*)calloc(S * D, 4);
    float drms_att_dummy[D];
    memset(drms_att_dummy, 0, D * 4);
    rmsnorm_bwd(dx_rms1, drms_att_dummy, dx_attn, ac->layer_in, lw->rms_att, D, S, cfg->rms_norm_eps);

    // Combine: dy = dx_rms1 + dx2 (skip connection)
    for (int i = 0; i < S * D; i++) dy[i] = dx_rms1[i] + dx2[i];
    free(dx_rms1);
}
