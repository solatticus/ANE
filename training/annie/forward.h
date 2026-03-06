// forward.h — Forward pass orchestration for Annie
// Per-layer: RMSNorm → QKV proj (ANE) → bias → RoPE → GQA attn (CPU) →
//            Wo proj (CPU) → bias → residual → RMSNorm → W1/W3 (ANE) →
//            SiLU gate (CPU) → W2 (ANE) → residual
#pragma once
#include "mil_dynamic.h"
#include "cpu_ops.h"

// Forward pass for one transformer layer.
// x_cur: [dim, S] in/out (modified in place — residual added)
// Transposed weight buffers (Wqt, Wkt, etc.) are precomputed for ANE.
static void annie_forward_layer(
    float *x_cur,
    AnnieLayerActs *ac,
    const AnnieLayerWeights *lw,
    const float *Wqt, const float *Wkt, const float *Wvt,
    const float *W1t, const float *W2t, const float *W3t,
    float *xnorm_buf, float *gate_buf,
    AnnieKernels *ak,
    const AnnieConfig *cfg)
{
    int D = cfg->dim, H = cfg->hidden_dim, S = cfg->max_seq_len;
    int kv = ac_kv_dim(cfg);

    // Save input for residual backward
    memcpy(ac->layer_in, x_cur, S * D * 4);

    // ── Attention block ──

    // 1. RMSNorm (CPU)
    rmsnorm(xnorm_buf, x_cur, lw->rms_att, D, S, cfg->rms_norm_eps);
    memcpy(ac->xnorm, xnorm_buf, S * D * 4);

    // 2. QKV projection (ANE)
    write_qkv_proj_input(ak, xnorm_buf, Wqt, Wkt, Wvt, D, kv, S);
    ane_eval(ak->qkvProj);
    read_qkv_proj_output(ak, ac->Q, ac->K, ac->V, D, kv, S);

    // 3. Add QKV bias (CPU)
    if (cfg->qkv_bias) {
        add_bias(ac->Q, lw->bq, D, S);
        add_bias(ac->K, lw->bk, kv, S);
        add_bias(ac->V, lw->bv, kv, S);
    }

    // 4. RoPE (CPU)
    rope_apply(ac->Q, ac->K, cfg->n_heads, cfg->n_kv_heads, cfg->head_dim, S, cfg->rope_theta);

    // 5. GQA attention (CPU)
    gqa_attention(ac->attn_out, ac->Q, ac->K, ac->V,
                  cfg->n_heads, cfg->n_kv_heads, cfg->head_dim, S);

    // 6. Wo projection (CPU cblas) — attn_out[dim,S] = Wo[dim,dim] @ attn_out[dim,S]
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                D, S, D, 1.0f, lw->Wo, D, ac->attn_out, S, 0.0f, ac->o_out, S);

    // 7. Add O bias (CPU)
    if (cfg->qkv_bias) {
        add_bias(ac->o_out, lw->bo, D, S);
    }

    // 8. Residual: x2 = x_cur + o_out
    vDSP_vadd(x_cur, 1, ac->o_out, 1, ac->x2, 1, (vDSP_Length)(S * D));

    // ── FFN block ──

    // 9. RMSNorm (CPU)
    rmsnorm(xnorm_buf, ac->x2, lw->rms_ffn, D, S, cfg->rms_norm_eps);
    memcpy(ac->x2norm, xnorm_buf, S * D * 4);

    // 10. W1 projection (ANE): xnorm → h1 [hidden, S]
    io_write_dyn(ak->dimToHidden->ioIn, xnorm_buf, D, S, W1t, H);
    ane_eval(ak->dimToHidden);
    io_read_dyn(ak->dimToHidden->ioOut, ac->h1, H, S);

    // 11. W3 projection (ANE, reuse same kernel): xnorm → h3 [hidden, S]
    io_write_dyn(ak->dimToHidden->ioIn, xnorm_buf, D, S, W3t, H);
    ane_eval(ak->dimToHidden);
    io_read_dyn(ak->dimToHidden->ioOut, ac->h3, H, S);

    // 12. SiLU gate (CPU): silu_out = SiLU(h1) * h3
    {
        int n = H * S;
        float minus1 = -1.0f, one = 1.0f;
        // sig = sigmoid(h1)
        vDSP_vsmul(ac->h1, 1, &minus1, gate_buf, 1, (vDSP_Length)n);
        vvexpf(gate_buf, gate_buf, &n);
        vDSP_vsadd(gate_buf, 1, &one, gate_buf, 1, (vDSP_Length)n);
        vvrecf(gate_buf, gate_buf, &n);
        // silu = h1 * sig
        vDSP_vmul(ac->h1, 1, gate_buf, 1, ac->silu_out, 1, (vDSP_Length)n);
        // gate = silu * h3
        vDSP_vmul(ac->silu_out, 1, ac->h3, 1, gate_buf, 1, (vDSP_Length)n);
    }

    // 13. W2 projection (ANE): gate → ffn_out [dim, S]
    io_write_dyn(ak->hiddenToDim->ioIn, gate_buf, H, S, W2t, D);
    ane_eval(ak->hiddenToDim);
    io_read_dyn(ak->hiddenToDim->ioOut, ac->ffn_out, D, S);

    // 14. Residual: x_cur = x2 + ffn_out
    vDSP_vadd(ac->x2, 1, ac->ffn_out, 1, x_cur, 1, (vDSP_Length)(S * D));
}
