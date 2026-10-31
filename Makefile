CC ?= cc
ARCH := $(shell uname -m)
CFLAGS ?= -O3 -std=c11 -Wall -Wextra -fno-math-errno
CC_IS_CLANG := $(shell $(CC) --version 2>/dev/null | grep -qi clang && echo 1)
ifeq ($(CC_IS_CLANG),1)
CFLAGS += -fno-vectorize -fno-slp-vectorize
else
CFLAGS += -fno-tree-vectorize -fno-tree-slp-vectorize
endif
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
