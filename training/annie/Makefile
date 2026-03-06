CC = xcrun clang
CFLAGS = -O2 -framework Foundation -framework IOSurface -framework Accelerate \
         -isysroot $(shell xcrun --show-sdk-path) -fobjc-arc \
         -DACCELERATE_NEW_LAPACK

HEADERS = config.h io.h cpu_ops.h mil_dynamic.h lora.h forward.h backward.h

train_lora: train_lora.m $(HEADERS)
	$(CC) $(CFLAGS) -o train_lora train_lora.m

clean:
	rm -f train_lora

.PHONY: clean
