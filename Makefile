CC ?= cc
ARCH := $(shell uname -m)
CFLAGS ?= -O3 -std=c11 -Wall -Wextra -fno-math-errno -fno-vectorize -fno-slp-vectorize
LDFLAGS ?= -lm
ifeq ($(ARCH),arm64)
CFLAGS += -DUSE_NEON
endif
ifeq ($(ARCH),aarch64)
CFLAGS += -DUSE_NEON
endif

all: build/bench build/test_kernels

build:
	mkdir -p build results

build/bench: benchmarks/bench.c benchmarks/kernels.c benchmarks/kernels.h | build
	$(CC) $(CFLAGS) benchmarks/bench.c benchmarks/kernels.c -o $@ $(LDFLAGS)

build/test_kernels: tests/test_kernels.c benchmarks/kernels.c benchmarks/kernels.h | build
	$(CC) $(CFLAGS) tests/test_kernels.c benchmarks/kernels.c -o $@ $(LDFLAGS)

test: build/test_kernels
	./build/test_kernels

bench: build/bench
	./build/bench results/bench.json

clean:
	rm -rf build results/*.json

.PHONY: all test bench clean
