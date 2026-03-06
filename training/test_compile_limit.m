// test_compile_limit.m — ANE kernel compile limit investigation for M4 Pro
// Usage: ./test_compile_limit <test> [args...]
//   accumulate <ic> <oc> <sp> <max>   — compile N kernels without unloading
//   unload <ic> <oc> <sp> <batch> <rounds> — compile batch, unload, repeat
//   cycle <ic> <oc> <sp> <iters>      — compile 1, unload 1, repeat
//   dynamic <ic> <oc> <sp> <max>      — dynamic matmul kernels (no baked weights)
//   identical <ic> <oc> <sp> <max>    — identical kernels (cache test)
//   transformer                       — 3B model shape feasibility
//
// Build: xcrun clang -O2 -Wall -fobjc-arc -o test_compile_limit test_compile_limit.m \
//        -framework Foundation -framework IOSurface -framework Accelerate -ldl
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#import <mach/mach_time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

static Kern *compile_baked_conv(int in_ch, int out_ch, int spatial, const float *weights) {
    @autoreleasepool {
        NSUInteger wsize = (NSUInteger)out_ch * in_ch * 2;
        NSUInteger total = 128 + wsize;
        uint8_t *buf = (uint8_t *)calloc(total, 1);
        buf[0] = 1; buf[4] = 2;
        buf[64] = 0xEF; buf[65] = 0xBE; buf[66] = 0xAD; buf[67] = 0xDE; buf[68] = 1;
        *(uint32_t *)(buf + 72) = (uint32_t)wsize;
        *(uint32_t *)(buf + 80) = 128;
        _Float16 *fp16 = (_Float16 *)(buf + 128);
        for (NSUInteger i = 0; i < (NSUInteger)out_ch * in_ch; i++)
            fp16[i] = (_Float16)weights[i];
        NSData *wblob = [NSData dataWithBytesNoCopy:buf length:total freeWhenDone:YES];

        NSString *mil = [NSString stringWithFormat:
            @"program(1.3)\n"
            "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
            "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
            "{\"coremltools-version\", \"9.0\"}})]\n"
            "{\n"
            "    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n"
            "        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n"
            "        tensor<fp16, [1, %d, 1, %d]> xh = cast(dtype = to16, x = x)[name = string(\"cin\")];\n"
            "        tensor<fp16, [%d, %d, 1, 1]> W = const()[name = string(\"W\"), "
            "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n"
            "        string vp = const()[name = string(\"vp\"), val = string(\"valid\")];\n"
            "        tensor<int32, [2]> st = const()[name = string(\"st\"), val = tensor<int32, [2]>([1, 1])];\n"
            "        tensor<int32, [4]> pd = const()[name = string(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
            "        tensor<int32, [2]> dl = const()[name = string(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n"
            "        int32 g = const()[name = string(\"g\"), val = int32(1)];\n"
            "        tensor<fp16, [1, %d, 1, %d]> yh = conv(dilations = dl, groups = g, "
            "pad = pd, pad_type = vp, strides = st, weight = W, x = xh)[name = string(\"conv\")];\n"
            "        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n"
            "        tensor<fp32, [1, %d, 1, %d]> y = cast(dtype = to32, x = yh)[name = string(\"out\")];\n"
            "    } -> (y);\n"
            "}\n",
            in_ch, spatial, in_ch, spatial,
            out_ch, in_ch, out_ch, in_ch,
            out_ch, spatial, out_ch, spatial];

        NSDictionary *wdict = @{
            @"@model_path/weights/weight.bin": @{@"offset": @0, @"data": wblob}
        };
        NSData *md = [mil dataUsingEncoding:NSUTF8StringEncoding];

        id desc = ((id(*)(Class, SEL, id, id, id))objc_msgSend)(
            g_Desc, @selector(modelWithMILText:weights:optionsPlist:), md, wdict, nil);
        if (!desc) return NULL;

        id mdl = ((id(*)(Class, SEL, id))objc_msgSend)(
            g_InMem, @selector(inMemoryModelWithDescriptor:), desc);

        id hx = ((id(*)(id, SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
        NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
        NSFileManager *fm = [NSFileManager defaultManager];
        [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
            withIntermediateDirectories:YES attributes:nil error:nil];
        [md writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        [wblob writeToFile:[td stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];

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

        size_t inBytes = (size_t)in_ch * spatial * 4;
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
        if (!desc) { fprintf(stderr, "  [dynamic] desc=NULL\n"); return NULL; }

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

static float *rand_weights(int n) {
    float *w = (float *)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++)
        w[i] = ((float)arc4random() / UINT32_MAX - 0.5f) * 0.02f;
    return w;
}

// ---- TEST: accumulate ----
static void cmd_accumulate(int ic, int oc, int sp, int max_n) {
    printf("ACCUMULATE: %dx%d S=%d max=%d\n", ic, oc, sp, max_n);
    float *w = rand_weights(ic * oc);
    Kern **kernels = (Kern **)calloc(max_n, sizeof(Kern *));
    int compiled = 0;

    uint64_t t0 = mach_absolute_time();
    for (int i = 0; i < max_n; i++) {
        w[i % (ic * oc)] += 0.001f;
        kernels[i] = compile_baked_conv(ic, oc, sp, w);
        if (!kernels[i]) {
            printf("  FAILED at #%d\n", i);
            break;
        }
        compiled++;
        if (compiled % 25 == 0 || compiled <= 5)
            printf("  %d compiled (%.1fms avg)\n", compiled,
                   tb_ms(mach_absolute_time() - t0) / compiled);
    }
    printf("RESULT: %d kernels in %.1fms (%.1fms/kernel)\n",
           compiled, tb_ms(mach_absolute_time() - t0),
           compiled > 0 ? tb_ms(mach_absolute_time() - t0) / compiled : 0);

    for (int i = 0; i < compiled; i++) free_kern(kernels[i]);
    free(kernels);
    free(w);
}

// ---- TEST: unload-recompile ----
static void cmd_unload(int ic, int oc, int sp, int batch, int rounds) {
    printf("UNLOAD-RECOMPILE: %dx%d S=%d batch=%d rounds=%d\n", ic, oc, sp, batch, rounds);
    float *w = rand_weights(ic * oc);
    Kern **kernels = (Kern **)calloc(batch, sizeof(Kern *));
    int total = 0;

    for (int r = 0; r < rounds; r++) {
        uint64_t t0 = mach_absolute_time();
        int ok = 0;
        for (int i = 0; i < batch; i++) {
            w[i % (ic * oc)] += 0.001f;
            kernels[i] = compile_baked_conv(ic, oc, sp, w);
            if (!kernels[i]) {
                printf("  Round %d: FAILED at #%d (total so far: %d)\n", r, i, total + ok);
                for (int j = 0; j < i; j++) free_kern(kernels[j]);
                goto done;
            }
            ok++;
        }
        total += ok;
        for (int i = 0; i < ok; i++) { free_kern(kernels[i]); kernels[i] = NULL; }
        printf("  Round %d: %d compiled+unloaded (%.1fms, cumulative=%d)\n",
               r, ok, tb_ms(mach_absolute_time() - t0), total);
    }
done:
    printf("RESULT: %d total compiles across %d rounds\n", total, rounds);
    free(kernels);
    free(w);
}

// ---- TEST: cycle ----
static void cmd_cycle(int ic, int oc, int sp, int iters) {
    printf("CYCLE: %dx%d S=%d iters=%d\n", ic, oc, sp, iters);
    float *w = rand_weights(ic * oc);
    int ok = 0;

    uint64_t t0 = mach_absolute_time();
    for (int i = 0; i < iters; i++) {
        w[i % (ic * oc)] += 0.001f;
        Kern *k = compile_baked_conv(ic, oc, sp, w);
        if (!k) {
            printf("  FAILED at iteration %d\n", i);
            break;
        }
        free_kern(k);
        ok++;
        if (ok % 100 == 0)
            printf("  %d cycles (%.1fms avg)\n", ok, tb_ms(mach_absolute_time() - t0) / ok);
    }
    printf("RESULT: %d/%d cycles in %.1fms (%.1fms/cycle)\n",
           ok, iters, tb_ms(mach_absolute_time() - t0),
           ok > 0 ? tb_ms(mach_absolute_time() - t0) / ok : 0);
    free(w);
}

// ---- TEST: dynamic ----
static void cmd_dynamic(int ic, int oc, int sp, int max_n) {
    printf("DYNAMIC: %dx%d S=%d max=%d\n", ic, oc, sp, max_n);
    Kern **kernels = (Kern **)calloc(max_n, sizeof(Kern *));
    int compiled = 0;

    uint64_t t0 = mach_absolute_time();
    for (int i = 0; i < max_n; i++) {
        kernels[i] = compile_dynamic_matmul(ic, oc, sp + i);  // unique spatial per kernel
        if (!kernels[i]) {
            printf("  FAILED at #%d\n", i);
            break;
        }
        compiled++;
        if (compiled % 25 == 0 || compiled <= 5)
            printf("  %d compiled (%.1fms avg)\n", compiled,
                   tb_ms(mach_absolute_time() - t0) / compiled);
    }
    printf("RESULT: %d dynamic kernels in %.1fms (%.1fms/kernel)\n",
           compiled, tb_ms(mach_absolute_time() - t0),
           compiled > 0 ? tb_ms(mach_absolute_time() - t0) / compiled : 0);

    for (int i = 0; i < compiled; i++) free_kern(kernels[i]);
    free(kernels);
}

// ---- TEST: identical ----
static void cmd_identical(int ic, int oc, int sp, int max_n) {
    printf("IDENTICAL: %dx%d S=%d max=%d\n", ic, oc, sp, max_n);
    float *w = rand_weights(ic * oc);
    Kern **kernels = (Kern **)calloc(max_n, sizeof(Kern *));
    int compiled = 0;

    uint64_t t0 = mach_absolute_time();
    for (int i = 0; i < max_n; i++) {
        kernels[i] = compile_baked_conv(ic, oc, sp, w);  // same weights every time
        if (!kernels[i]) {
            printf("  FAILED at #%d\n", i);
            break;
        }
        compiled++;
        if (compiled % 25 == 0 || compiled <= 5)
            printf("  %d compiled (%.1fms avg)\n", compiled,
                   tb_ms(mach_absolute_time() - t0) / compiled);
    }
    printf("RESULT: %d identical kernels in %.1fms\n",
           compiled, tb_ms(mach_absolute_time() - t0));

    for (int i = 0; i < compiled; i++) free_kern(kernels[i]);
    free(kernels);
    free(w);
}

// ---- TEST: transformer (3B shapes) ----
static void cmd_transformer(void) {
    printf("TRANSFORMER BUDGET: Qwen2.5-3B shapes (dynamic matmul)\n\n");

    struct { int ic, oc, sp; const char *name; } shapes[] = {
        {2048, 2048, 256, "Wq [2048->2048]"},
        {2048,  256, 256, "Wk [2048->256]"},
        {2048,  256, 256, "Wv [2048->256]"},
        {2048, 2048, 256, "Wo [2048->2048]"},
        {2048, 11008, 256, "W1 [2048->11008]"},
        {11008, 2048, 256, "W2 [11008->2048]"},
        {2048, 11008, 256, "W3 [2048->11008]"},
    };
    int n = sizeof(shapes) / sizeof(shapes[0]);

    Kern **kernels = (Kern **)calloc(n, sizeof(Kern *));
    int ok = 0;
    for (int s = 0; s < n; s++) {
        printf("  %s ... ", shapes[s].name);
        fflush(stdout);
        uint64_t t0 = mach_absolute_time();
        kernels[s] = compile_dynamic_matmul(shapes[s].ic, shapes[s].oc, shapes[s].sp);
        if (kernels[s]) {
            double ms = tb_ms(mach_absolute_time() - t0);
            size_t in_mb = (size_t)shapes[s].ic * (shapes[s].sp + shapes[s].oc) * 4;
            size_t out_mb = (size_t)shapes[s].oc * shapes[s].sp * 4;
            printf("OK (%.0fms, in=%.1fMB out=%.1fMB)\n", ms, in_mb/1e6, out_mb/1e6);
            ok++;
        } else {
            printf("FAIL\n");
        }
    }

    // Eval test for successful kernels
    if (ok > 0) {
        printf("\n  Eval benchmark (10 iters each):\n");
        for (int s = 0; s < n; s++) {
            if (!kernels[s]) continue;
            // Fill with random data
            IOSurfaceLock(kernels[s]->ioIn, 0, NULL);
            float *p = (float *)IOSurfaceGetBaseAddress(kernels[s]->ioIn);
            int total = (int)(IOSurfaceGetAllocSize(kernels[s]->ioIn) / 4);
            for (int i = 0; i < total; i++) p[i] = ((float)arc4random() / UINT32_MAX - 0.5f) * 0.01f;
            IOSurfaceUnlock(kernels[s]->ioIn, 0, NULL);

            NSError *e = nil;
            // Warmup
            for (int w = 0; w < 3; w++)
                ((BOOL(*)(id, SEL, unsigned int, id, id, NSError **))objc_msgSend)(
                    kernels[s]->model, @selector(evaluateWithQoS:options:request:error:),
                    21, @{}, kernels[s]->request, &e);

            uint64_t t0 = mach_absolute_time();
            for (int i = 0; i < 10; i++)
                ((BOOL(*)(id, SEL, unsigned int, id, id, NSError **))objc_msgSend)(
                    kernels[s]->model, @selector(evaluateWithQoS:options:request:error:),
                    21, @{}, kernels[s]->request, &e);
            double ms = tb_ms(mach_absolute_time() - t0) / 10.0;
            double flops = 2.0 * shapes[s].ic * shapes[s].oc * shapes[s].sp;
            printf("  %s: %.2fms/eval  %.1f GFLOP/s\n", shapes[s].name, ms, flops / (ms * 1e6));
        }
    }

    printf("\nRESULT: %d/%d 3B shapes compiled\n", ok, n);
    if (ok == n)
        printf("  Dynamic pipeline needs only %d compiles for ANY number of layers!\n", n);

    for (int s = 0; s < n; s++) free_kern(kernels[s]);
    free(kernels);
}

// ---- TEST: sizes (compile limit vs kernel size) ----
static void cmd_sizes(void) {
    printf("SIZE SENSITIVITY: accumulate kernels of various sizes\n\n");
    printf("%-14s %-12s %-12s %-12s\n", "Shape", "Weight(KB)", "IOSurf(KB)", "MaxCompiles");

    struct { int ic, oc, sp; } sizes[] = {
        {  64,   64, 16},
        { 128,  128, 16},
        { 256,  256, 16},
        { 512,  512, 16},
        { 768,  768, 16},
        { 768, 2048, 16},
    };
    int n = sizeof(sizes) / sizeof(sizes[0]);

    for (int s = 0; s < n; s++) {
        int ic = sizes[s].ic, oc = sizes[s].oc, sp = sizes[s].sp;
        float *w = rand_weights(ic * oc);
        Kern **kernels = (Kern **)calloc(300, sizeof(Kern *));
        int compiled = 0;

        for (int i = 0; i < 300; i++) {
            w[i % (ic * oc)] += 0.001f;
            kernels[i] = compile_baked_conv(ic, oc, sp, w);
            if (!kernels[i]) break;
            compiled++;
        }

        size_t w_kb = (size_t)ic * oc * 2 / 1024;
        size_t io_kb = ((size_t)ic * sp * 4 + (size_t)oc * sp * 4) / 1024;
        printf("%-4dx%-8d %-12zu %-12zu %-12d\n", ic, oc, w_kb, io_kb, compiled);

        for (int i = 0; i < compiled; i++) free_kern(kernels[i]);
        free(kernels);
        free(w);
    }
}

int main(int argc, char **argv) {
    @autoreleasepool {
        ane_init();

        if (argc < 2) {
            printf("Usage: %s <test> [args...]\n", argv[0]);
            printf("  accumulate <ic> <oc> <sp> <max>\n");
            printf("  unload <ic> <oc> <sp> <batch> <rounds>\n");
            printf("  cycle <ic> <oc> <sp> <iters>\n");
            printf("  dynamic <ic> <oc> <sp> <max>\n");
            printf("  identical <ic> <oc> <sp> <max>\n");
            printf("  transformer\n");
            printf("  sizes\n");
            printf("  all  — run all tests (each in fresh process via exec)\n");
            return 1;
        }

        const char *test = argv[1];

        if (strcmp(test, "accumulate") == 0 && argc >= 6) {
            cmd_accumulate(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]));
        } else if (strcmp(test, "unload") == 0 && argc >= 7) {
            cmd_unload(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), atoi(argv[6]));
        } else if (strcmp(test, "cycle") == 0 && argc >= 6) {
            cmd_cycle(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]));
        } else if (strcmp(test, "dynamic") == 0 && argc >= 6) {
            cmd_dynamic(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]));
        } else if (strcmp(test, "identical") == 0 && argc >= 6) {
            cmd_identical(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]));
        } else if (strcmp(test, "transformer") == 0) {
            cmd_transformer();
        } else if (strcmp(test, "sizes") == 0) {
            cmd_sizes();
        } else if (strcmp(test, "all") == 0) {
            // Run each test in a fresh process
            const char *self = argv[0];
            const char *tests[] = {
                "accumulate 64 64 16 300",
                "accumulate 768 768 16 300",
                "unload 64 64 16 100 5",
                "cycle 64 64 16 500",
                "dynamic 64 64 16 300",
                "identical 64 64 16 300",
                "sizes",
                "transformer",
                NULL
            };
            for (int i = 0; tests[i]; i++) {
                printf("\n========================================\n");
                char cmd[512];
                snprintf(cmd, sizeof(cmd), "%s %s", self, tests[i]);
                printf(">>> %s\n", cmd);
                printf("========================================\n");
                fflush(stdout);
                int ret = system(cmd);
                if (ret != 0) printf("  (exit code %d)\n", ret);
            }
        } else {
            fprintf(stderr, "Unknown test: %s\n", test);
            return 1;
        }
    }
    return 0;
}
