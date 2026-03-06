// test_latency_estimate.m — Per-token and prefill latency for Annie candidate models
// Measures dynamic matmul ANE time at S=1 (decode) and S=16/32/64 (prefill)
// Build: xcrun clang -O2 -Wall -fobjc-arc -o test_latency_estimate test_latency_estimate.m \
//        -framework Foundation -framework IOSurface -framework Accelerate -ldl
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#import <mach/mach_time.h>
#include <stdio.h>
#include <stdlib.h>

static Class g_Desc, g_InMem, g_Req, g_IO;
static mach_timebase_info_data_t g_tb;

static void ane_init(void) {
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    g_Desc  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    g_InMem = NSClassFromString(@"_ANEInMemoryModel");
    g_Req   = NSClassFromString(@"_ANERequest");
    g_IO    = NSClassFromString(@"_ANEIOSurfaceObject");
    mach_timebase_info(&g_tb);
}

static double tb_ms(uint64_t t) {
    return (double)t * g_tb.numer / g_tb.denom / 1e6;
}

static IOSurfaceRef make_surface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(bytes), (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1, (id)kIOSurfaceBytesPerRow: @(bytes),
        (id)kIOSurfaceAllocSize: @(bytes), (id)kIOSurfacePixelFormat: @0
    });
}

typedef struct {
    id model;
    IOSurfaceRef ioIn, ioOut;
    id request;
    NSString *tmpDir;
} Kern;

static Kern *compile_dynamic_matmul(int in_ch, int out_ch, int spatial) {
    @autoreleasepool {
        int sp_total = spatial + out_ch;
        NSMutableString *mil = [NSMutableString string];
        [mil appendString:@"program(1.3)\n"
            "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
            "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
            "{\"coremltools-version\", \"9.0\"}})]\n{\n"];
        [mil appendFormat:@"    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n", in_ch, sp_total];
        [mil appendString:@"        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n"];
        [mil appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> xh = cast(dtype = to16, x = x)[name = string(\"cin\")];\n", in_ch, sp_total];
        [mil appendString:@"        tensor<int32, [4]> ba = const()[name = string(\"ba\"), val = tensor<int32, [4]>([0,0,0,0])];\n"];
        [mil appendFormat:@"        tensor<int32, [4]> sa = const()[name = string(\"sa\"), val = tensor<int32, [4]>([1,%d,1,%d])];\n", in_ch, spatial];
        [mil appendFormat:@"        tensor<fp16, [1,%d,1,%d]> act = slice_by_size(x=xh,begin=ba,size=sa)[name=string(\"act\")];\n", in_ch, spatial];
        [mil appendFormat:@"        tensor<int32, [4]> bw = const()[name = string(\"bw\"), val = tensor<int32, [4]>([0,0,0,%d])];\n", spatial];
        [mil appendFormat:@"        tensor<int32, [4]> sw = const()[name = string(\"sw\"), val = tensor<int32, [4]>([1,%d,1,%d])];\n", in_ch, out_ch];
        [mil appendFormat:@"        tensor<fp16, [1,%d,1,%d]> wt = slice_by_size(x=xh,begin=bw,size=sw)[name=string(\"wt\")];\n", in_ch, out_ch];
        [mil appendFormat:@"        tensor<int32, [4]> ra = const()[name = string(\"ra\"), val = tensor<int32, [4]>([1,1,%d,%d])];\n", in_ch, spatial];
        [mil appendFormat:@"        tensor<fp16, [1,1,%d,%d]> a2 = reshape(shape=ra,x=act)[name=string(\"a2\")];\n", in_ch, spatial];
        [mil appendString:@"        tensor<int32, [4]> pm = const()[name = string(\"pm\"), val = tensor<int32, [4]>([0,1,3,2])];\n"];
        [mil appendFormat:@"        tensor<fp16, [1,1,%d,%d]> a3 = transpose(perm=pm,x=a2)[name=string(\"a3\")];\n", spatial, in_ch];
        [mil appendFormat:@"        tensor<int32, [4]> rw = const()[name = string(\"rw\"), val = tensor<int32, [4]>([1,1,%d,%d])];\n", in_ch, out_ch];
        [mil appendFormat:@"        tensor<fp16, [1,1,%d,%d]> W = reshape(shape=rw,x=wt)[name=string(\"W\")];\n", in_ch, out_ch];
        [mil appendString:@"        bool bF = const()[name = string(\"bF\"), val = bool(false)];\n"];
        [mil appendFormat:@"        tensor<fp16, [1,1,%d,%d]> yh = matmul(transpose_x=bF,transpose_y=bF,x=a3,y=W)[name=string(\"mm\")];\n", spatial, out_ch];
        [mil appendFormat:@"        tensor<fp16, [1,1,%d,%d]> yt = transpose(perm=pm,x=yh)[name=string(\"yt\")];\n", out_ch, spatial];
        [mil appendFormat:@"        tensor<int32, [4]> ro = const()[name = string(\"ro\"), val = tensor<int32, [4]>([1,%d,1,%d])];\n", out_ch, spatial];
        [mil appendFormat:@"        tensor<fp16, [1,%d,1,%d]> yr = reshape(shape=ro,x=yt)[name=string(\"yr\")];\n", out_ch, spatial];
        [mil appendString:@"        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n"];
        [mil appendFormat:@"        tensor<fp32, [1,%d,1,%d]> y = cast(dtype = to32, x = yr)[name = string(\"out\")];\n", out_ch, spatial];
        [mil appendString:@"    } -> (y);\n}\n"];

        NSData *md = [mil dataUsingEncoding:NSUTF8StringEncoding];
        id desc = ((id(*)(Class, SEL, id, id, id))objc_msgSend)(
            g_Desc, @selector(modelWithMILText:weights:optionsPlist:), md, @{}, nil);
        if (!desc) return NULL;

        id mdl = ((id(*)(Class, SEL, id))objc_msgSend)(
            g_InMem, @selector(inMemoryModelWithDescriptor:), desc);

        id hx = ((id(*)(id, SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
        NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
        NSFileManager *fm = [NSFileManager defaultManager];
        [fm createDirectoryAtPath:td withIntermediateDirectories:YES attributes:nil error:nil];
        [md writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];

        NSError *e = nil;
        if (!((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
                mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e)) {
            [fm removeItemAtPath:td error:nil];
            return NULL;
        }
        if (!((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
                mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e)) {
            [fm removeItemAtPath:td error:nil];
            return NULL;
        }

        size_t inBytes = (size_t)in_ch * sp_total * 4;
        size_t outBytes = (size_t)out_ch * spatial * 4;
        IOSurfaceRef ioIn = make_surface(inBytes);
        IOSurfaceRef ioOut = make_surface(outBytes);

        id wI = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(g_IO, @selector(objectWithIOSurface:), ioIn);
        id wO = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(g_IO, @selector(objectWithIOSurface:), ioOut);
        id req = ((id(*)(Class, SEL, id, id, id, id, id, id, id))objc_msgSend)(g_Req,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[wI], @[@0], @[wO], @[@0], nil, nil, @0);

        Kern *k = (Kern *)calloc(1, sizeof(Kern));
        k->model = mdl;
        k->ioIn = ioIn;
        k->ioOut = ioOut;
        k->request = req;
        k->tmpDir = td;
        return k;
    }
}

static void free_kern(Kern *k) {
    if (!k) return;
    @autoreleasepool {
        NSError *e = nil;
        ((BOOL(*)(id, SEL, unsigned int, NSError **))objc_msgSend)(
            k->model, @selector(unloadWithQoS:error:), 21, &e);
        CFRelease(k->ioIn);
        CFRelease(k->ioOut);
        [[NSFileManager defaultManager] removeItemAtPath:k->tmpDir error:nil];
    }
    free(k);
}

static double bench_kernel(Kern *k, int warmup, int iters) {
    NSError *e = nil;
    // Fill with random data
    IOSurfaceLock(k->ioIn, 0, NULL);
    float *p = (float *)IOSurfaceGetBaseAddress(k->ioIn);
    int total = (int)(IOSurfaceGetAllocSize(k->ioIn) / 4);
    for (int i = 0; i < total; i++) p[i] = ((float)arc4random() / UINT32_MAX - 0.5f) * 0.01f;
    IOSurfaceUnlock(k->ioIn, 0, NULL);

    for (int i = 0; i < warmup; i++)
        ((BOOL(*)(id, SEL, unsigned int, id, id, NSError **))objc_msgSend)(
            k->model, @selector(evaluateWithQoS:options:request:error:),
            21, @{}, k->request, &e);

    uint64_t t0 = mach_absolute_time();
    for (int i = 0; i < iters; i++)
        ((BOOL(*)(id, SEL, unsigned int, id, id, NSError **))objc_msgSend)(
            k->model, @selector(evaluateWithQoS:options:request:error:),
            21, @{}, k->request, &e);
    return tb_ms(mach_absolute_time() - t0) / iters;
}

typedef struct {
    const char *name;
    int dim, hidden, n_heads, n_kv_heads, head_dim, n_layers, vocab;
} ModelSpec;

typedef struct {
    int ic, oc;
    const char *name;
} KernelShape;

int main(int argc, char **argv) {
    @autoreleasepool {
        ane_init();

        ModelSpec models[] = {
            {"SmolLM2-135M",   576,  1536,  9, 3, 64, 30, 49152},
            {"SmolLM2-360M",   960,  2560, 15, 5, 64, 32, 49152},
            {"Qwen2.5-0.5B",   896,  4864, 14, 2, 64, 24, 151936},
            {"Qwen2.5-1.5B",  1536,  8960, 12, 2, 128, 28, 151936},
            {"Qwen2.5-3B",    2048, 11008, 16, 2, 128, 36, 151936},
        };
        int nmodels = sizeof(models) / sizeof(models[0]);

        int seq_lens[] = {1, 4, 16, 32, 64};
        int nseqs = sizeof(seq_lens) / sizeof(seq_lens[0]);

        printf("=============================================================\n");
        printf("Annie Latency Estimate — M4 Pro ANE (dynamic matmul)\n");
        printf("=============================================================\n");
        printf("Measures ANE linear projection time only (no attention, RMSnorm, etc.)\n");
        printf("Real latency will be higher due to CPU ops, but this is the dominant cost.\n\n");

        for (int m = 0; m < nmodels; m++) {
            ModelSpec *spec = &models[m];
            int kv_dim = spec->n_kv_heads * spec->head_dim;

            printf("--- %s (dim=%d, hidden=%d, kv_dim=%d, layers=%d) ---\n",
                   spec->name, spec->dim, spec->hidden, kv_dim, spec->n_layers);

            // Unique kernel shapes for this model
            KernelShape shapes[] = {
                {spec->dim,    spec->dim,    "Wq"},
                {spec->dim,    kv_dim,       "Wk"},
                {spec->dim,    kv_dim,       "Wv"},
                {spec->dim,    spec->dim,    "Wo"},
                {spec->dim,    spec->hidden, "W1"},
                {spec->hidden, spec->dim,    "W2"},
                {spec->dim,    spec->hidden, "W3"},
            };
            int nshapes = 7;

            printf("%-6s", "S");
            for (int s = 0; s < nshapes; s++)
                printf("  %-8s", shapes[s].name);
            printf("  %-10s %-10s %-10s\n", "layer_ms", "all_ms", "tok/s");

            for (int si = 0; si < nseqs; si++) {
                int S = seq_lens[si];
                printf("%-6d", S);
                double layer_ms = 0;
                int iters = S <= 4 ? 100 : 30;
                int warmup = 5;

                for (int s = 0; s < nshapes; s++) {
                    Kern *k = compile_dynamic_matmul(shapes[s].ic, shapes[s].oc, S);
                    if (!k) {
                        printf("  FAIL    ");
                        continue;
                    }
                    double ms = bench_kernel(k, warmup, iters);
                    printf("  %-8.3f", ms);
                    layer_ms += ms;
                    free_kern(k);
                }

                double all_ms = layer_ms * spec->n_layers;
                double tok_per_s = S == 1 ? 1000.0 / all_ms : 0;
                printf("  %-10.2f %-10.1f", layer_ms, all_ms);
                if (S == 1)
                    printf(" %-10.1f", tok_per_s);
                printf("\n");
            }

            // Estimate realistic generation scenario
            // "Hey Oscar, what's up?" ≈ 8 tokens prompt, generate 30 tokens response
            printf("\n  Scenario: 8-token prompt → 30-token response\n");
            // Prefill: use S=8 estimate (interpolate from S=4 and S=16)
            // Decode: use S=1 numbers
            // We'll just compile S=1 and S=8 and measure directly
            {
                double prefill_layer = 0, decode_layer = 0;
                for (int s = 0; s < nshapes; s++) {
                    Kern *k1 = compile_dynamic_matmul(shapes[s].ic, shapes[s].oc, 1);
                    Kern *k8 = compile_dynamic_matmul(shapes[s].ic, shapes[s].oc, 8);
                    if (k1) { decode_layer += bench_kernel(k1, 5, 100); free_kern(k1); }
                    if (k8) { prefill_layer += bench_kernel(k8, 5, 50); free_kern(k8); }
                }
                double prefill_ms = prefill_layer * spec->n_layers;
                double decode_per_tok = decode_layer * spec->n_layers;
                double total = prefill_ms + 30 * decode_per_tok;
                printf("  Prefill (8 tok):  %.1fms\n", prefill_ms);
                printf("  Decode (per tok): %.1fms\n", decode_per_tok);
                printf("  Total (8+30):     %.0fms  (%.1f tok/s decode)\n",
                       total, 1000.0 / decode_per_tok);
            }
            printf("\n");
        }
    }
    return 0;
}
