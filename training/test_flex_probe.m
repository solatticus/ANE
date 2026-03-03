// test_flex_probe.m — Does ANE accept weights as function parameters?
#include "stories_io.h"
#include "stories_mil.h"
#include "stories_flex.h"

// Use actual model dimensions
#define T_DIM DIM
#define T_SEQ SEQ

static void print_sample(const char *label, float *data, int n) {
    printf("  %s: ", label);
    for (int i=0; i<(n<8?n:8); i++) printf("%.4f ", data[i]);
    if (n>8) printf("...");
    printf("\n");
}

static bool ane_eval_check(Kern *k) {
    id mdl = (__bridge id)k->model; id req = (__bridge id)k->request; NSError *e = nil;
    BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(mdl,
        @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
    if (!ok || e) printf("  eval error: %s\n", e ? [[e description] UTF8String] : "returned NO");
    return ok && !e;
}

static bool flex_eval_check(FlexKern *k) {
    id mdl = (__bridge id)k->model; id req = (__bridge id)k->request; NSError *e = nil;
    BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(mdl,
        @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
    if (!ok || e) printf("  eval error: %s\n", e ? [[e description] UTF8String] : "returned NO");
    return ok && !e;
}

int main(void) {
    @autoreleasepool {
        setbuf(stdout, NULL);
        ane_init();
        mach_timebase_info(&g_tb);

        printf("=== Flex IOSurface Probe (dim=%d, seq=%d) ===\n\n", T_DIM, T_SEQ);

        // Build diagonal weight: W[i][j] = (i==j) ? 0.5 : 0
        float *W = (float*)calloc(T_DIM*T_DIM, 4);
        for (int i=0; i<T_DIM; i++) W[i*T_DIM+i] = 0.5f;

        // Input: x[c*SEQ + t] = (c+1)*0.01 for all t
        float *x_in = (float*)malloc(T_DIM*T_SEQ*4);
        for (int c=0; c<T_DIM; c++)
            for (int t=0; t<T_SEQ; t++)
                x_in[c*T_SEQ + t] = (float)(c+1) * 0.01f;

        float *y_out = (float*)malloc(T_DIM*T_SEQ*4);
        float *expected = (float*)malloc(T_DIM*T_SEQ*4);
        for (int i=0; i<T_DIM*T_SEQ; i++) expected[i] = x_in[i] * 0.5f;

        // ── Test 1: Weight as const (same path as train_lora) ──
        printf("Test 1: Weight baked as const (baseline)\n");
        {
            NSMutableString *mil = [NSMutableString string];
            [mil appendString:MIL_HDR];
            [mil appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", T_DIM, T_SEQ];
            [mil appendFormat:@"        tensor<fp16, [%d,%d,1,1]> W = const()[name=string(\"W\"), "
                "val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/w.bin\"), offset=uint64(64)))];\n",
                T_DIM, T_DIM, T_DIM, T_DIM];
            [mil appendString:@CONV_CONST];
            [mil appendFormat:@"        tensor<fp16, [1,%d,1,%d]> y = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W,x=x)[name=string(\"y\")];\n", T_DIM, T_SEQ];
            [mil appendString:@"    } -> (y);\n}\n"];

            Kern *k = compile_kern_mil_w(mil, (@{
                @"@model_path/weights/w.bin": @{@"offset":@0, @"data":build_blob(W, T_DIM, T_DIM)},
            }), T_DIM*T_SEQ*2, T_DIM*T_SEQ*2);

            if (!k) { printf("  FAILED to compile\n"); return 1; }
            printf("  compiled OK\n");

            io_write_fp16(k->ioIn, x_in, T_DIM, T_SEQ);
            bool ok = ane_eval_check(k);
            printf("  eval: %s\n", ok ? "OK" : "FAIL");
            io_read_fp16(k->ioOut, y_out, 0, T_DIM, T_SEQ);

            print_sample("x_in    ", x_in, 8);
            print_sample("y_out   ", y_out, 8);
            print_sample("expected", expected, 8);

            float max_err = 0, sum = 0;
            for (int i=0; i<T_DIM*T_SEQ; i++) {
                float err = fabsf(y_out[i] - expected[i]);
                if (err > max_err) max_err = err;
                sum += fabsf(y_out[i]);
            }
            printf("  sum(|y|)=%.4f max_err=%.6f %s\n\n", sum, max_err, max_err < 0.01f ? "PASS" : "FAIL");
            free_kern(k);
        }

        // ── Test 2: Weight as function parameter (flex, 2 inputs) ──
        printf("Test 2: Weight as function parameter (flex)\n");
        {
            NSMutableString *mil = [NSMutableString string];
            [mil appendString:MIL_HDR];
            [mil appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x,\n", T_DIM, T_SEQ];
            [mil appendFormat:@"                     tensor<fp16, [%d, %d, 1, 1]> W) {\n", T_DIM, T_DIM];
            [mil appendString:@CONV_CONST];
            [mil appendFormat:@"        tensor<fp16, [1,%d,1,%d]> y = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W,x=x)[name=string(\"y\")];\n", T_DIM, T_SEQ];
            [mil appendString:@"    } -> (y);\n}\n"];

            int in_sizes[] = {T_DIM*T_SEQ*2, T_DIM*T_DIM*2};
            FlexKern *fk = compile_flex_kern(mil, in_sizes, 2, T_DIM*T_SEQ*2);

            if (!fk) { printf("  FAILED to compile flex kern\n"); return 1; }
            printf("  compiled OK (2 inputs)\n");

            io_write_fp16(fk->inputs[1], W, T_DIM, T_DIM);
            io_write_fp16(fk->inputs[0], x_in, T_DIM, T_SEQ);
            bool ok = flex_eval_check(fk);
            printf("  eval: %s\n", ok ? "OK" : "FAIL");
            io_read_fp16(fk->ioOut, y_out, 0, T_DIM, T_SEQ);

            print_sample("x_in    ", x_in, 8);
            print_sample("y_out   ", y_out, 8);
            print_sample("expected", expected, 8);

            float max_err = 0, sum = 0;
            for (int i=0; i<T_DIM*T_SEQ; i++) {
                float err = fabsf(y_out[i] - expected[i]);
                if (err > max_err) max_err = err;
                sum += fabsf(y_out[i]);
            }
            printf("  sum(|y|)=%.4f max_err=%.6f %s\n\n", sum, max_err, max_err < 0.01f ? "PASS" : "FAIL");
            free_flex_kern(fk);
        }

        free(W); free(x_in); free(y_out); free(expected);
        printf("=== Probe complete ===\n");
    }
    return 0;
}
