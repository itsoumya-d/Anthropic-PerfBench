#include "kernels.h"
#include <math.h>
#include <stddef.h>

#if defined(USE_NEON)
#include <arm_neon.h>
#endif

void rmsnorm_scalar(const float *x, const float *residual, float *out, int n, float eps) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        float v = x[i] + residual[i];
        sum += v * v;
    }
    float scale = 1.0f / sqrtf(sum / (float)n + eps);
    for (int i = 0; i < n; i++) out[i] = (x[i] + residual[i]) * scale;
}

void rmsnorm_unrolled(const float *x, const float *residual, float *out, int n, float eps) {
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    int i = 0;
    for (; i + 3 < n; i += 4) {
        float v0 = x[i] + residual[i];
        float v1 = x[i + 1] + residual[i + 1];
        float v2 = x[i + 2] + residual[i + 2];
        float v3 = x[i + 3] + residual[i + 3];
        s0 += v0 * v0; s1 += v1 * v1; s2 += v2 * v2; s3 += v3 * v3;
    }
    for (; i < n; i++) { float v = x[i] + residual[i]; s0 += v * v; }
    float scale = 1.0f / sqrtf((s0 + s1 + s2 + s3) / (float)n + eps);
    i = 0;
    for (; i + 3 < n; i += 4) {
        out[i] = (x[i] + residual[i]) * scale;
        out[i + 1] = (x[i + 1] + residual[i + 1]) * scale;
        out[i + 2] = (x[i + 2] + residual[i + 2]) * scale;
        out[i + 3] = (x[i + 3] + residual[i + 3]) * scale;
    }
    for (; i < n; i++) out[i] = (x[i] + residual[i]) * scale;
}

void rmsnorm_neon(const float *x, const float *residual, float *out, int n, float eps) {
#if defined(USE_NEON)
    float32x4_t acc = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t a = vld1q_f32(x + i);
        float32x4_t b = vld1q_f32(residual + i);
        float32x4_t v = vaddq_f32(a, b);
        acc = vfmaq_f32(acc, v, v);
    }
    float sum = vaddvq_f32(acc);
    for (; i < n; i++) { float v = x[i] + residual[i]; sum += v * v; }
    float scale = 1.0f / sqrtf(sum / (float)n + eps);
    float32x4_t scale4 = vdupq_n_f32(scale);
    i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t v = vaddq_f32(vld1q_f32(x + i), vld1q_f32(residual + i));
        vst1q_f32(out + i, vmulq_f32(v, scale4));
    }
    for (; i < n; i++) out[i] = (x[i] + residual[i]) * scale;
#else
    rmsnorm_unrolled(x, residual, out, n, eps);
#endif
}

void rope_scalar(float *x, const float *cos_table, const float *sin_table, int n) {
    int half = n / 2;
    for (int i = 0; i < half; i++) {
        float a = x[i], b = x[i + half];
        x[i] = a * cos_table[i] - b * sin_table[i];
        x[i + half] = a * sin_table[i] + b * cos_table[i];
    }
}

void rope_unrolled(float *x, const float *cos_table, const float *sin_table, int n) {
    int half = n / 2;
    int i = 0;
    for (; i + 3 < half; i += 4) {
        float a0 = x[i], b0 = x[i + half];
        float a1 = x[i + 1], b1 = x[i + 1 + half];
        float a2 = x[i + 2], b2 = x[i + 2 + half];
        float a3 = x[i + 3], b3 = x[i + 3 + half];
        x[i] = a0 * cos_table[i] - b0 * sin_table[i];
        x[i + half] = a0 * sin_table[i] + b0 * cos_table[i];
        x[i + 1] = a1 * cos_table[i + 1] - b1 * sin_table[i + 1];
        x[i + 1 + half] = a1 * sin_table[i + 1] + b1 * cos_table[i + 1];
        x[i + 2] = a2 * cos_table[i + 2] - b2 * sin_table[i + 2];
        x[i + 2 + half] = a2 * sin_table[i + 2] + b2 * cos_table[i + 2];
        x[i + 3] = a3 * cos_table[i + 3] - b3 * sin_table[i + 3];
        x[i + 3 + half] = a3 * sin_table[i + 3] + b3 * cos_table[i + 3];
    }
    for (; i < half; i++) {
        float a = x[i], b = x[i + half];
        x[i] = a * cos_table[i] - b * sin_table[i];
        x[i + half] = a * sin_table[i] + b * cos_table[i];
    }
}

void rope_neon(float *x, const float *cos_table, const float *sin_table, int n) {
#if defined(USE_NEON)
    int half = n / 2;
    int i = 0;
    for (; i + 3 < half; i += 4) {
        float32x4_t a = vld1q_f32(x + i);
        float32x4_t b = vld1q_f32(x + i + half);
        float32x4_t c = vld1q_f32(cos_table + i);
        float32x4_t s = vld1q_f32(sin_table + i);
        float32x4_t out_a = vsubq_f32(vmulq_f32(a, c), vmulq_f32(b, s));
        float32x4_t out_b = vaddq_f32(vmulq_f32(a, s), vmulq_f32(b, c));
        vst1q_f32(x + i, out_a);
        vst1q_f32(x + i + half, out_b);
    }
    for (; i < half; i++) {
        float a = x[i], b = x[i + half];
        x[i] = a * cos_table[i] - b * sin_table[i];
        x[i + half] = a * sin_table[i] + b * cos_table[i];
    }
#else
    rope_unrolled(x, cos_table, sin_table, n);
#endif
}

static inline float nibble_value(uint8_t byte, int high) {
    int q = high ? (byte >> 4) : (byte & 0x0F);
    return (float)(q >= 8 ? q - 16 : q);
}

void int4_gemv_scalar(const uint8_t *weights, const float *scales, const float *x, float *y,
                      int rows, int cols, int group_size) {
    int row_bytes = cols / 2;
    int groups = cols / group_size;
    for (int r = 0; r < rows; r++) {
        float sum = 0.0f;
        const uint8_t *row = weights + (size_t)r * row_bytes;
        const float *row_scales = scales + (size_t)r * groups;
        for (int k = 0; k < cols; k++) {
            uint8_t byte = row[k >> 1];
            float q = nibble_value(byte, k & 1);
            sum += q * row_scales[k / group_size] * x[k];
        }
        y[r] = sum;
    }
}

void int4_gemv_unrolled(const uint8_t *weights, const float *scales, const float *x, float *y,
                        int rows, int cols, int group_size) {
    int row_bytes = cols / 2;
    int groups = cols / group_size;
    for (int r = 0; r < rows; r++) {
        float acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
        const uint8_t *row = weights + (size_t)r * row_bytes;
        const float *row_scales = scales + (size_t)r * groups;
        for (int g = 0; g < groups; g++) {
            float scale = row_scales[g];
            int base = g * group_size;
            for (int j = 0; j < group_size; j += 4) {
                int k = base + j;
                uint8_t b0 = row[k >> 1];
                uint8_t b1 = row[(k + 1) >> 1];
                uint8_t b2 = row[(k + 2) >> 1];
                uint8_t b3 = row[(k + 3) >> 1];
                acc0 += nibble_value(b0, k & 1) * x[k];
                acc1 += nibble_value(b1, (k + 1) & 1) * x[k + 1];
                acc2 += nibble_value(b2, (k + 2) & 1) * x[k + 2];
                acc3 += nibble_value(b3, (k + 3) & 1) * x[k + 3];
            }
            y[r] += 0; /* keep the row alive for clarity */
            acc0 *= scale; acc1 *= scale; acc2 *= scale; acc3 *= scale;
            y[r] = (g == 0 ? 0.0f : y[r]) + acc0 + acc1 + acc2 + acc3;
            acc0 = acc1 = acc2 = acc3 = 0.0f;
        }
    }
}

void int4_gemv_neon(const uint8_t *weights, const float *scales, const float *x, float *y,
                    int rows, int cols, int group_size) {
#if defined(USE_NEON)
    int row_bytes = cols / 2;
    int groups = cols / group_size;
    const uint8x16_t mask = vdupq_n_u8(0x0F);
    const uint8x16_t eight = vdupq_n_u8(8);
    const int8x16_t sign = vdupq_n_s8(8);
    for (int r = 0; r < rows; r++) {
        float32x4_t t0 = vdupq_n_f32(0), t1 = vdupq_n_f32(0);
        float32x4_t t2 = vdupq_n_f32(0), t3 = vdupq_n_f32(0);
        float32x4_t t4 = vdupq_n_f32(0), t5 = vdupq_n_f32(0);
        float32x4_t t6 = vdupq_n_f32(0), t7 = vdupq_n_f32(0);
        const uint8_t *row = weights + (size_t)r * row_bytes;
        const float *row_scales = scales + (size_t)r * groups;
        for (int g = 0; g < groups; g++) {
            uint8x16_t bytes = vld1q_u8(row + g * 16);
            uint8x16_t lo = vandq_u8(bytes, mask);
            uint8x16_t hi = vshrq_n_u8(bytes, 4);
            int8x16_t lo_s = vsubq_s8(vreinterpretq_s8_u8(veorq_u8(lo, eight)), sign);
            int8x16_t hi_s = vsubq_s8(vreinterpretq_s8_u8(veorq_u8(hi, eight)), sign);
            int8x16_t z0 = vzip1q_s8(lo_s, hi_s);
            int8x16_t z1 = vzip2q_s8(lo_s, hi_s);
            int16x8_t w0 = vmovl_s8(vget_low_s8(z0));
            int16x8_t w1 = vmovl_s8(vget_high_s8(z0));
            int16x8_t w2 = vmovl_s8(vget_low_s8(z1));
            int16x8_t w3 = vmovl_s8(vget_high_s8(z1));
            const float *xg = x + g * 32;
            float32x4_t p0 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(w0))), vld1q_f32(xg));
            float32x4_t p1 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(w0))), vld1q_f32(xg + 4));
            float32x4_t p2 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(w1))), vld1q_f32(xg + 8));
            float32x4_t p3 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(w1))), vld1q_f32(xg + 12));
            float32x4_t p4 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(w2))), vld1q_f32(xg + 16));
            float32x4_t p5 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(w2))), vld1q_f32(xg + 20));
            float32x4_t p6 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(w3))), vld1q_f32(xg + 24));
            float32x4_t p7 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(w3))), vld1q_f32(xg + 28));
            float scale = row_scales[g];
            t0 = vfmaq_n_f32(t0, p0, scale); t1 = vfmaq_n_f32(t1, p1, scale);
            t2 = vfmaq_n_f32(t2, p2, scale); t3 = vfmaq_n_f32(t3, p3, scale);
            t4 = vfmaq_n_f32(t4, p4, scale); t5 = vfmaq_n_f32(t5, p5, scale);
            t6 = vfmaq_n_f32(t6, p6, scale); t7 = vfmaq_n_f32(t7, p7, scale);
        }
        float32x4_t sum = vaddq_f32(vaddq_f32(vaddq_f32(t0, t1), vaddq_f32(t2, t3)),
                                    vaddq_f32(vaddq_f32(t4, t5), vaddq_f32(t6, t7)));
        y[r] = vaddvq_f32(sum);
    }
#else
    int4_gemv_unrolled(weights, scales, x, y, rows, cols, group_size);
#endif
}
