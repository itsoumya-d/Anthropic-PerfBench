#ifndef HEROS_KERNELS_H
#define HEROS_KERNELS_H
#include <stdint.h>

void rmsnorm_scalar(const float *x, const float *residual, float *out, int n, float eps);
void rmsnorm_unrolled(const float *x, const float *residual, float *out, int n, float eps);
void rmsnorm_neon(const float *x, const float *residual, float *out, int n, float eps);

void rope_scalar(float *x, const float *cos_table, const float *sin_table, int n);
void rope_unrolled(float *x, const float *cos_table, const float *sin_table, int n);
void rope_neon(float *x, const float *cos_table, const float *sin_table, int n);

void int4_gemv_scalar(const uint8_t *weights, const float *scales, const float *x, float *y,
                      int rows, int cols, int group_size);
void int4_gemv_unrolled(const uint8_t *weights, const float *scales, const float *x, float *y,
                        int rows, int cols, int group_size);
void int4_gemv_neon(const uint8_t *weights, const float *scales, const float *x, float *y,
                    int rows, int cols, int group_size);

#endif
