#define _POSIX_C_SOURCE 200809L
#include "kernels.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static unsigned state = 7;
static float rnd(void) {
    state = state * 1664525u + 1013904223u;
    return ((state >> 8) & 0xFFFFFF) / (float)0xFFFFFF * 2.0f - 1.0f;
}

typedef void (*rms_fn)(const float *, const float *, float *, int, float);
typedef void (*rope_fn)(float *, const float *, const float *, int);
typedef void (*gemv_fn)(const uint8_t *, const float *, const float *, float *, int, int, int);

static double bench_rms(rms_fn fn, const float *x, const float *r, float *out, int n, int iters) {
    double best = 1e30;
    for (int it = 0; it < iters; it++) {
        double start = now_ms();
        fn(x, r, out, n, 1e-5f);
        double elapsed = now_ms() - start;
        if (elapsed < best) best = elapsed;
    }
    return best;
}

static double bench_rope(rope_fn fn, float *x, const float *c, const float *s, int n, int iters) {
    double best = 1e30;
    for (int it = 0; it < iters; it++) {
        double start = now_ms();
        fn(x, c, s, n);
        double elapsed = now_ms() - start;
        if (elapsed < best) best = elapsed;
    }
    return best;
}

static double bench_gemv(gemv_fn fn, const uint8_t *w, const float *sc, const float *v, float *y,
                         int rows, int cols, int group, int iters) {
    double best = 1e30;
    for (int it = 0; it < iters; it++) {
        double start = now_ms();
        fn(w, sc, v, y, rows, cols, group);
        double elapsed = now_ms() - start;
        if (elapsed < best) best = elapsed;
    }
    return best;
}

static double measure_copy_bandwidth(size_t bytes) {
    float *a = malloc(bytes);
    float *b = malloc(bytes);
    memset(a, 1, bytes);
    memset(b, 0, bytes);
    double best = 1e30;
    volatile float sink = 0.0f;
    typedef void *(*copy_fn_t)(void *, const void *, size_t);
    static copy_fn_t volatile copy_fn = memcpy;
    for (int i = 0; i < 20; i++) {
        double start = now_ms();
        copy_fn(b, a, bytes);
        sink += b[0] + b[bytes / sizeof(float) - 1];
        double elapsed = now_ms() - start;
        if (elapsed < best) best = elapsed;
    }
    (void)sink;
    free(a); free(b);
    return (2.0 * bytes) / (best * 1e6);
}

typedef struct { const char *name; const char *variant; double ms; double flops; double bytes; } record;

static void print_record(FILE *out, record r) {
    double gflops = r.flops / (r.ms * 1e6);
    double gbps = r.bytes / (r.ms * 1e6);
    fprintf(out,
            "    {\"kernel\": \"%s\", \"variant\": \"%s\", \"ms\": %.6f, "
            "\"flops\": %.0f, \"bytes\": %.0f, \"gflops\": %.3f, \"gbps\": %.3f}",
            r.name, r.variant, r.ms, r.flops, r.bytes, gflops, gbps);
}

int main(int argc, char **argv) {
    const int n = 1 << 22;
    float *x = malloc(sizeof(float) * n);
    float *r = malloc(sizeof(float) * n);
    float *out = malloc(sizeof(float) * n);
    float *cos_t = malloc(sizeof(float) * (n / 2));
    float *sin_t = malloc(sizeof(float) * (n / 2));
    for (int i = 0; i < n; i++) { x[i] = rnd(); r[i] = rnd(); }
    for (int i = 0; i < n / 2; i++) { cos_t[i] = cosf(i * 0.001f); sin_t[i] = sinf(i * 0.001f); }

    const int rows = 4096, cols = 4096, group = 32;
    uint8_t *weights = malloc((size_t)rows * cols / 2);
    float *scales = malloc(sizeof(float) * rows * (cols / group));
    float *vec = malloc(sizeof(float) * cols);
    float *y = malloc(sizeof(float) * rows);
    for (size_t i = 0; i < (size_t)rows * cols / 2; i++) weights[i] = (uint8_t)(state >> 8);
    for (int i = 0; i < rows * (cols / group); i++) scales[i] = 0.001f + fabsf(rnd()) * 0.02f;
    for (int i = 0; i < cols; i++) vec[i] = rnd();

    double bw = measure_copy_bandwidth(64u * 1024u * 1024u);
    record records[12];
    int count = 0;
    records[count++] = (record){"rmsnorm_residual", "scalar", bench_rms(rmsnorm_scalar, x, r, out, n, 80), 5.0 * n, 20.0 * n};
    records[count++] = (record){"rmsnorm_residual", "unrolled", bench_rms(rmsnorm_unrolled, x, r, out, n, 80), 5.0 * n, 20.0 * n};
    records[count++] = (record){"rmsnorm_residual", "neon", bench_rms(rmsnorm_neon, x, r, out, n, 80), 5.0 * n, 20.0 * n};
    records[count++] = (record){"rope", "scalar", bench_rope(rope_scalar, x, cos_t, sin_t, n, 80), 4.0 * n, 12.0 * n};
    records[count++] = (record){"rope", "unrolled", bench_rope(rope_unrolled, x, cos_t, sin_t, n, 80), 4.0 * n, 12.0 * n};
    records[count++] = (record){"rope", "neon", bench_rope(rope_neon, x, cos_t, sin_t, n, 80), 4.0 * n, 12.0 * n};
    records[count++] = (record){"int4_gemv", "scalar", bench_gemv(int4_gemv_scalar, weights, scales, vec, y, rows, cols, group, 20), 2.0 * rows * cols, (double)rows * cols / 2 + (double)rows * (cols / group) * 4 + cols * 4 + rows * 4};
    records[count++] = (record){"int4_gemv", "unrolled", bench_gemv(int4_gemv_unrolled, weights, scales, vec, y, rows, cols, group, 20), 2.0 * rows * cols, (double)rows * cols / 2 + (double)rows * (cols / group) * 4 + cols * 4 + rows * 4};
    records[count++] = (record){"int4_gemv", "neon", bench_gemv(int4_gemv_neon, weights, scales, vec, y, rows, cols, group, 20), 2.0 * rows * cols, (double)rows * cols / 2 + (double)rows * (cols / group) * 4 + cols * 4 + rows * 4};

    FILE *out_file = stdout;
    if (argc > 1) out_file = fopen(argv[1], "w");
    if (!out_file) return 1;
    fprintf(out_file, "{\n  \"copy_bandwidth_gbps\": %.3f,\n  \"records\": [\n", bw);
    for (int i = 0; i < count; i++) {
        print_record(out_file, records[i]);
        fprintf(out_file, "%s\n", i + 1 == count ? "" : ",");
    }
    fprintf(out_file, "  ]\n}\n");
    if (argc > 1) fclose(out_file);

    volatile float sink = out[0] + y[0];
    (void)sink;
    free(x); free(r); free(out); free(cos_t); free(sin_t);
    free(weights); free(scales); free(vec); free(y);
    return 0;
}
