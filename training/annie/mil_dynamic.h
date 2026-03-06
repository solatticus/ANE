// mil_dynamic.h — MIL generators for Annie (Qwen2.5-3B on ANE)
// Parameterized for GQA (16Q/2KV heads), different weight dims.
// 4 unique kernel shapes compiled once, reused across all 36 layers:
//   1. qkvProj:     dim → dim + 2*kv_dim (Q/K/V projections)
//   2. dimToHidden:  dim → hidden_dim    (W1, W3 forward, W2^T backward)
//   3. hiddenToDim: hidden_dim → dim     (W2 forward, W1^T/W3^T backward)
//   4. dimToDim:    dim → dim            (Wo^T backward, Wq^T backward)
#pragma once
#include "io.h"

#define MIL_HDR \
    @"program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, " \
    "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, " \
    "{\"coremltools-version\", \"9.0\"}})]\n{\n"

// ===== Generic dynamic matmul: y = x @ W =====
// Input:  [1, IC, 1, SEQ+OC] fp32 — act[0:SEQ] + W[SEQ:SEQ+OC]
// Output: [1, OC, 1, SEQ] fp32
// Effectively computes: W^T[OC,IC] @ act[IC,SEQ] → [OC,SEQ]
static NSString *gen_dyn_matmul_mil(int ic, int oc, int seq) {
    NSMutableString *m = [NSMutableString string];
    int sp = seq + oc;
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n", ic, sp];
    [m appendString:@"        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> xh = cast(dtype=to16,x=x)[name=string(\"cin\")];\n", ic, sp];

    // Slice activation [1,ic,1,seq]
    [m appendString:@"        tensor<int32, [4]> ba = const()[name=string(\"ba\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
    [m appendFormat:@"        tensor<int32, [4]> sa = const()[name=string(\"sa\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", ic, seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> act = slice_by_size(x=xh,begin=ba,size=sa)[name=string(\"act\")];\n", ic, seq];

    // Slice weight [1,ic,1,oc]
    [m appendFormat:@"        tensor<int32, [4]> bw = const()[name=string(\"bw\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", seq];
    [m appendFormat:@"        tensor<int32, [4]> sw = const()[name=string(\"sw\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", ic, oc];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> wt = slice_by_size(x=xh,begin=bw,size=sw)[name=string(\"wt\")];\n", ic, oc];

    // Reshape for matmul: [1,ic,1,seq] → [1,1,seq,ic]
    [m appendFormat:@"        tensor<int32, [4]> ra = const()[name=string(\"ra\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", ic, seq];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> a2 = reshape(shape=ra,x=act)[name=string(\"a2\")];\n", ic, seq];
    [m appendString:@"        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> at = transpose(perm=pm,x=a2)[name=string(\"at\")];\n", seq, ic];

    // Reshape weight: [1,ic,1,oc] → [1,1,ic,oc]
    [m appendFormat:@"        tensor<int32, [4]> rw = const()[name=string(\"rw\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", ic, oc];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> W = reshape(shape=rw,x=wt)[name=string(\"W\")];\n", ic, oc];

    // matmul: [1,1,seq,ic] @ [1,1,ic,oc] → [1,1,seq,oc]
    [m appendString:@"        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> ym = matmul(transpose_x=bF,transpose_y=bF,x=at,y=W)[name=string(\"ym\")];\n", seq, oc];

    // Transpose back + reshape: [1,1,seq,oc] → [1,oc,1,seq]
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> yt = transpose(perm=pm,x=ym)[name=string(\"yt\")];\n", oc, seq];
    [m appendFormat:@"        tensor<int32, [4]> ro = const()[name=string(\"ro\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", oc, seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> yr = reshape(shape=ro,x=yt)[name=string(\"yr\")];\n", oc, seq];
    [m appendString:@"        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];\n"];
    [m appendFormat:@"        tensor<fp32, [1,%d,1,%d]> y = cast(dtype=to32,x=yr)[name=string(\"cout\")];\n", oc, seq];
    [m appendString:@"    } -> (y);\n}\n"];
    return m;
}

// ===== QKV projection (GQA-aware) =====
// Three matmuls with shared activation but different output dims.
// Input: [1, dim, 1, SEQ + dim + kv_dim + kv_dim] fp32
//   sp[0:SEQ]                              = xnorm [dim, SEQ]
//   sp[SEQ:SEQ+dim]                        = Wq^T  [dim, dim]
//   sp[SEQ+dim:SEQ+dim+kv]                 = Wk^T  [dim, kv_dim]
//   sp[SEQ+dim+kv:SEQ+dim+2*kv]            = Wv^T  [dim, kv_dim]
// Output: [1, dim+2*kv_dim, 1, SEQ] fp32 = concat(Q, K, V)
static NSString *gen_qkv_proj_dynamic(int dim, int kv_dim, int seq) {
    int sp_in = seq + dim + 2 * kv_dim;
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n", dim, sp_in];
    [m appendString:@"        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> xh = cast(dtype=to16,x=x)[name=string(\"cin\")];\n", dim, sp_in];

    // Slice xnorm [1,dim,1,seq]
    [m appendString:@"        tensor<int32, [4]> bx = const()[name=string(\"bx\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
    [m appendFormat:@"        tensor<int32, [4]> sx = const()[name=string(\"sx\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", dim, seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> xn = slice_by_size(x=xh,begin=bx,size=sx)[name=string(\"xn\")];\n", dim, seq];

    // Slice Wq^T [1,dim,1,dim]
    [m appendFormat:@"        tensor<int32, [4]> bq = const()[name=string(\"bq\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", seq];
    [m appendFormat:@"        tensor<int32, [4]> sq = const()[name=string(\"sq\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", dim, dim];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> Wq = slice_by_size(x=xh,begin=bq,size=sq)[name=string(\"Wq\")];\n", dim, dim];

    // Slice Wk^T [1,dim,1,kv_dim]
    [m appendFormat:@"        tensor<int32, [4]> bk = const()[name=string(\"bk\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", seq + dim];
    [m appendFormat:@"        tensor<int32, [4]> sk = const()[name=string(\"sk\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", dim, kv_dim];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> Wk = slice_by_size(x=xh,begin=bk,size=sk)[name=string(\"Wk\")];\n", dim, kv_dim];

    // Slice Wv^T [1,dim,1,kv_dim]
    [m appendFormat:@"        tensor<int32, [4]> bv = const()[name=string(\"bv\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", seq + dim + kv_dim];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> Wv = slice_by_size(x=xh,begin=bv,size=sk)[name=string(\"Wv\")];\n", dim, kv_dim];

    // Reshape xnorm for matmul: [1,dim,1,seq] → [1,1,seq,dim]
    [m appendFormat:@"        tensor<int32, [4]> rd = const()[name=string(\"rd\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", dim, seq];
    [m appendString:@"        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> xn2 = reshape(shape=rd,x=xn)[name=string(\"xn2\")];\n", dim, seq];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> xnt = transpose(perm=pm,x=xn2)[name=string(\"xnt\")];\n", seq, dim];

    // Reshape weights for matmul
    [m appendFormat:@"        tensor<int32, [4]> rwq = const()[name=string(\"rwq\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", dim, dim];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> Wq2 = reshape(shape=rwq,x=Wq)[name=string(\"Wq2\")];\n", dim, dim];
    [m appendFormat:@"        tensor<int32, [4]> rwk = const()[name=string(\"rwk\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", dim, kv_dim];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> Wk2 = reshape(shape=rwk,x=Wk)[name=string(\"Wk2\")];\n", dim, kv_dim];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> Wv2 = reshape(shape=rwk,x=Wv)[name=string(\"Wv2\")];\n", dim, kv_dim];

    // QKV matmul: [1,1,seq,dim] @ [1,1,dim,X] → [1,1,seq,X]
    [m appendString:@"        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> qm = matmul(transpose_x=bF,transpose_y=bF,x=xnt,y=Wq2)[name=string(\"qm\")];\n", seq, dim];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> km = matmul(transpose_x=bF,transpose_y=bF,x=xnt,y=Wk2)[name=string(\"km\")];\n", seq, kv_dim];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> vm = matmul(transpose_x=bF,transpose_y=bF,x=xnt,y=Wv2)[name=string(\"vm\")];\n", seq, kv_dim];

    // Transpose back: [1,1,seq,X] → [1,X,1,seq]
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> qt = transpose(perm=pm,x=qm)[name=string(\"qt\")];\n", dim, seq];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> kt = transpose(perm=pm,x=km)[name=string(\"kt\")];\n", kv_dim, seq];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> vt = transpose(perm=pm,x=vm)[name=string(\"vt\")];\n", kv_dim, seq];

    [m appendFormat:@"        tensor<int32, [4]> rq = const()[name=string(\"rq\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", dim, seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> qf = reshape(shape=rq,x=qt)[name=string(\"qf\")];\n", dim, seq];
    [m appendFormat:@"        tensor<int32, [4]> rk = const()[name=string(\"rk\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", kv_dim, seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> kf = reshape(shape=rk,x=kt)[name=string(\"kf\")];\n", kv_dim, seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> vf = reshape(shape=rk,x=vt)[name=string(\"vf\")];\n", kv_dim, seq];

    // Concat output: (Q, K, V) along channel axis
    int out_ch = dim + 2 * kv_dim;
    [m appendString:@"        int32 cax = const()[name=string(\"cax\"), val=int32(1)];\n"];
    [m appendString:@"        bool cid = const()[name=string(\"cid\"), val=bool(false)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> out = concat(axis=cax,interleave=cid,values=(qf,kf,vf))[name=string(\"cat\")];\n", out_ch, seq];
    [m appendString:@"        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];\n"];
    [m appendFormat:@"        tensor<fp32, [1,%d,1,%d]> out32 = cast(dtype=to32,x=out)[name=string(\"cout\")];\n", out_ch, seq];
    [m appendString:@"    } -> (out32);\n}\n"];
    return m;
}

// ===== Causal mask blob (parameterized for sequence length) =====
static NSData *g_mask_blob = nil;
static int g_mask_seq = 0;

static NSData *get_mask_blob(int seq) {
    if (!g_mask_blob || g_mask_seq != seq) {
        _Float16 *mask = (_Float16*)calloc(seq * seq, sizeof(_Float16));
        for (int t = 0; t < seq; t++)
            for (int t2 = 0; t2 < seq; t2++)
                mask[t * seq + t2] = (t2 <= t) ? (_Float16)0.0f : (_Float16)(-65504.0f);
        g_mask_blob = build_blob_fp16(mask, seq * seq);
        g_mask_seq = seq;
        free(mask);
    }
    return g_mask_blob;
}

// ===== Kernel set for Annie training =====
typedef struct {
    Kern *qkvProj;       // QKV projection (dim → dim + 2*kv_dim)
    Kern *dimToHidden;   // dim → hidden_dim (W1, W3 fwd; W2^T bwd)
    Kern *hiddenToDim;   // hidden_dim → dim (W2 fwd; W1^T, W3^T bwd)
    Kern *dimToDim;      // dim → dim (Wo^T bwd, Wq^T bwd)
} AnnieKernels;

static bool annie_compile_kernels(AnnieKernels *ak, const AnnieConfig *cfg) {
    int D = cfg->dim, H = cfg->hidden_dim, S = cfg->max_seq_len;
    int kv = ac_kv_dim(cfg);

    printf("  Compiling qkvProj (dim=%d → %d+2*%d)...\n", D, D, kv);
    int qkv_sp = S + D + 2 * kv;
    int qkv_out = D + 2 * kv;
    ak->qkvProj = compile_kern_mil_w(gen_qkv_proj_dynamic(D, kv, S), @{},
        D * qkv_sp * 4, qkv_out * S * 4);
    if (!ak->qkvProj) return false;

    printf("  Compiling dimToHidden (%d → %d)...\n", D, H);
    ak->dimToHidden = compile_kern_mil_w(gen_dyn_matmul_mil(D, H, S), @{},
        D * (S + H) * 4, H * S * 4);
    if (!ak->dimToHidden) return false;

    printf("  Compiling hiddenToDim (%d → %d)...\n", H, D);
    ak->hiddenToDim = compile_kern_mil_w(gen_dyn_matmul_mil(H, D, S), @{},
        H * (S + D) * 4, D * S * 4);
    if (!ak->hiddenToDim) return false;

    printf("  Compiling dimToDim (%d → %d)...\n", D, D);
    ak->dimToDim = compile_kern_mil_w(gen_dyn_matmul_mil(D, D, S), @{},
        D * (S + D) * 4, D * S * 4);
    if (!ak->dimToDim) return false;

    return true;
}

static void annie_free_kernels(AnnieKernels *ak) {
    free_kern(ak->qkvProj);
    free_kern(ak->dimToHidden);
    free_kern(ak->hiddenToDim);
    free_kern(ak->dimToDim);
}

// ===== IOSurface packing helpers for QKV projection =====

static void write_qkv_proj_input(AnnieKernels *ak, const float *xnorm,
                                  const float *Wqt, const float *Wkt, const float *Wvt,
                                  int dim, int kv_dim, int seq) {
    int sp = seq + dim + 2 * kv_dim;
    IOSurfaceLock(ak->qkvProj->ioIn, 0, NULL);
    float *buf = (float*)IOSurfaceGetBaseAddress(ak->qkvProj->ioIn);
    for (int d = 0; d < dim; d++) {
        memcpy(buf + d * sp,                           xnorm + d * seq, seq * 4);
        memcpy(buf + d * sp + seq,                     Wqt + d * dim,   dim * 4);
        memcpy(buf + d * sp + seq + dim,               Wkt + d * kv_dim, kv_dim * 4);
        memcpy(buf + d * sp + seq + dim + kv_dim,      Wvt + d * kv_dim, kv_dim * 4);
    }
    IOSurfaceUnlock(ak->qkvProj->ioIn, 0, NULL);
}

static void read_qkv_proj_output(AnnieKernels *ak, float *Q, float *K, float *V,
                                  int dim, int kv_dim, int seq) {
    IOSurfaceLock(ak->qkvProj->ioOut, kIOSurfaceLockReadOnly, NULL);
    float *out = (float*)IOSurfaceGetBaseAddress(ak->qkvProj->ioOut);
    memcpy(Q, out,                      dim * seq * 4);
    memcpy(K, out + dim * seq,          kv_dim * seq * 4);
    memcpy(V, out + (dim + kv_dim) * seq, kv_dim * seq * 4);
    IOSurfaceUnlock(ak->qkvProj->ioOut, kIOSurfaceLockReadOnly, NULL);
}
