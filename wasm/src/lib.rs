use core::slice;
use std::alloc::{alloc, dealloc, Layout};

const ALIGN: usize = 64;

#[inline]
fn nibble(byte: u8, high: bool) -> i8 {
    let q = if high { byte >> 4 } else { byte & 0x0f };
    ((q ^ 8) as i8) - 8
}

#[no_mangle]
pub extern "C" fn pb_alloc(size: usize) -> *mut u8 {
    let layout = Layout::from_size_align(size.max(1), ALIGN).expect("valid layout");
    unsafe { alloc(layout) }
}

#[no_mangle]
pub extern "C" fn pb_dealloc(ptr: *mut u8, size: usize) {
    if ptr.is_null() {
        return;
    }
    let layout = Layout::from_size_align(size.max(1), ALIGN).expect("valid layout");
    unsafe { dealloc(ptr, layout) }
}

#[no_mangle]
pub unsafe extern "C" fn pb_gemv_f32(
    weights: *const f32,
    x: *const f32,
    y: *mut f32,
    rows: usize,
    cols: usize,
) {
    if weights.is_null() || x.is_null() || y.is_null() {
        return;
    }
    let w = slice::from_raw_parts(weights, rows * cols);
    let x = slice::from_raw_parts(x, cols);
    let y = slice::from_raw_parts_mut(y, rows);
    for r in 0..rows {
        let row = &w[r * cols..(r + 1) * cols];
        let mut sum = 0.0f32;
        for k in 0..cols {
            sum += row[k] * x[k];
        }
        y[r] = sum;
    }
}

#[no_mangle]
pub unsafe extern "C" fn pb_int4_gemv_scalar(
    weights: *const u8,
    scales: *const f32,
    x: *const f32,
    y: *mut f32,
    rows: usize,
    cols: usize,
    group: usize,
) {
    if weights.is_null() || scales.is_null() || x.is_null() || y.is_null() || group == 0 {
        return;
    }
    let row_bytes = cols / 2;
    let groups = cols / group;
    let x = slice::from_raw_parts(x, cols);
    let y = slice::from_raw_parts_mut(y, rows);
    for r in 0..rows {
        let row = slice::from_raw_parts(weights.add(r * row_bytes), row_bytes);
        let row_scales = slice::from_raw_parts(scales.add(r * groups), groups);
        let mut sum = 0.0f32;
        for k in 0..cols {
            let q = nibble(row[k >> 1], (k & 1) == 1);
            sum += q as f32 * row_scales[k / group] * x[k];
        }
        y[r] = sum;
    }
}

#[no_mangle]
pub unsafe extern "C" fn pb_int4_gemv_simd(
    weights: *const u8,
    scales: *const f32,
    x: *const f32,
    y: *mut f32,
    rows: usize,
    cols: usize,
    group: usize,
) {
    #[cfg(target_arch = "wasm32")]
    {
        if group == 32 && cols % 32 == 0 {
            int4_gemv_simd_impl(weights, scales, x, y, rows, cols);
            return;
        }
    }
    pb_int4_gemv_scalar(weights, scales, x, y, rows, cols, group);
}

#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
unsafe fn int4_gemv_simd_impl(
    weights: *const u8,
    scales: *const f32,
    x: *const f32,
    y: *mut f32,
    rows: usize,
    cols: usize,
) {
    use core::arch::wasm32::*;

    if weights.is_null() || scales.is_null() || x.is_null() || y.is_null() {
        return;
    }
    let row_bytes = cols / 2;
    let groups = cols / 32;
    let x = slice::from_raw_parts(x, cols);
    let y = slice::from_raw_parts_mut(y, rows);

    for r in 0..rows {
        let row = slice::from_raw_parts(weights.add(r * row_bytes), row_bytes);
        let row_scales = slice::from_raw_parts(scales.add(r * groups), groups);
        let mut row_sum = 0.0f32;

        for g in 0..groups {
            let bytes = v128_load(row.as_ptr().add(g * 16) as *const v128);
            let lo = v128_and(bytes, u8x16_splat(0x0f));
            let hi = u8x16_shr(bytes, 4);
            let eight = i8x16_splat(8);
            let lo_s = i8x16_sub(v128_xor(lo, eight), eight);
            let hi_s = i8x16_sub(v128_xor(hi, eight), eight);

            let z0 = i8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(lo_s, hi_s);
            let z1 = i8x16_shuffle::<8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31>(lo_s, hi_s);

            let mut acc = f32x4_splat(0.0);
            for (j, z) in [z0, z1].into_iter().enumerate() {
                let xp = x.as_ptr().add(g * 32 + j * 16);
                let lo16 = i16x8_extend_low_i8x16(z);
                let hi16 = i16x8_extend_high_i8x16(z);
                let a = f32x4_convert_i32x4(i32x4_extend_low_i16x8(lo16));
                let b = f32x4_convert_i32x4(i32x4_extend_high_i16x8(lo16));
                let c = f32x4_convert_i32x4(i32x4_extend_low_i16x8(hi16));
                let d = f32x4_convert_i32x4(i32x4_extend_high_i16x8(hi16));
                acc = f32x4_add(acc, f32x4_mul(a, v128_load(xp as *const v128)));
                acc = f32x4_add(acc, f32x4_mul(b, v128_load(xp.add(4) as *const v128)));
                acc = f32x4_add(acc, f32x4_mul(c, v128_load(xp.add(8) as *const v128)));
                acc = f32x4_add(acc, f32x4_mul(d, v128_load(xp.add(12) as *const v128)));
            }
            let lane_sum = f32x4_extract_lane::<0>(acc)
                + f32x4_extract_lane::<1>(acc)
                + f32x4_extract_lane::<2>(acc)
                + f32x4_extract_lane::<3>(acc);
            row_sum += lane_sum * row_scales[g];
        }
        y[r] = row_sum;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Lcg(u32);

    impl Lcg {
        fn next_byte(&mut self) -> u8 {
            self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
            (self.0 >> 16) as u8
        }

        fn next_unit(&mut self) -> f32 {
            let v = (self.next_byte() as u32) << 16 | self.next_byte() as u32;
            v as f32 / u32::MAX as f32
        }
    }

    fn pack(weights: &[i8]) -> Vec<u8> {
        weights
            .chunks(2)
            .map(|pair| {
                let lo = (pair[0] as u8) & 0x0f;
                let hi = (pair[1] as u8) & 0x0f;
                lo | (hi << 4)
            })
            .collect()
    }

    fn reference(
        packed: &[u8],
        scales: &[f32],
        x: &[f32],
        rows: usize,
        cols: usize,
        group: usize,
    ) -> Vec<f64> {
        let row_bytes = cols / 2;
        let groups = cols / group;
        (0..rows)
            .map(|r| {
                let mut sum = 0.0f64;
                for k in 0..cols {
                    let byte = packed[r * row_bytes + (k >> 1)];
                    let q = nibble(byte, (k & 1) == 1) as f64;
                    sum += q * scales[r * groups + k / group] as f64 * x[k] as f64;
                }
                sum
            })
            .collect()
    }

    fn assert_close(actual: &[f32], expected: &[f64], tol: f64) {
        for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
            let diff = (*a as f64 - e).abs();
            assert!(diff <= tol * (1.0 + e.abs()), "row {i}: {a} vs {e}");
        }
    }

    #[test]
    fn packed_nibbles_round_trip() {
        let weights: Vec<i8> = (0..64).map(|i| (i % 16) as i8 - 8).collect();
        let packed = pack(&weights);
        for (k, expected) in weights.iter().enumerate() {
            let q = nibble(packed[k >> 1], (k & 1) == 1);
            assert_eq!(q, *expected);
        }
    }

    #[test]
    fn scalar_int4_matches_float64_reference() {
        let rows = 8;
        let cols = 256;
        let group = 32;
        let mut rng = Lcg(12345);
        let weights: Vec<i8> = (0..rows * cols)
            .map(|_| (rng.next_byte() % 16) as i8 - 8)
            .collect();
        let packed = pack(&weights);
        let scales: Vec<f32> = (0..rows * (cols / group))
            .map(|_| 0.001 + rng.next_unit() * 0.01)
            .collect();
        let x: Vec<f32> = (0..cols).map(|_| rng.next_unit() * 2.0 - 1.0).collect();
        let expected = reference(&packed, &scales, &x, rows, cols, group);
        let mut y = vec![0.0f32; rows];
        unsafe {
            pb_int4_gemv_scalar(
                packed.as_ptr(),
                scales.as_ptr(),
                x.as_ptr(),
                y.as_mut_ptr(),
                rows,
                cols,
                group,
            );
        }
        assert_close(&y, &expected, 1e-3);
    }

    #[test]
    fn scalar_fallback_matches_direct_path() {
        let rows = 4;
        let cols = 128;
        let group = 32;
        let mut rng = Lcg(77);
        let weights: Vec<i8> = (0..rows * cols)
            .map(|_| (rng.next_byte() % 16) as i8 - 8)
            .collect();
        let packed = pack(&weights);
        let scales: Vec<f32> = (0..rows * (cols / group))
            .map(|_| 0.001 + rng.next_unit() * 0.01)
            .collect();
        let x: Vec<f32> = (0..cols).map(|_| rng.next_unit() * 2.0 - 1.0).collect();
        let mut direct = vec![0.0f32; rows];
        let mut simd = vec![0.0f32; rows];
        unsafe {
            pb_int4_gemv_scalar(
                packed.as_ptr(),
                scales.as_ptr(),
                x.as_ptr(),
                direct.as_mut_ptr(),
                rows,
                cols,
                group,
            );
            pb_int4_gemv_simd(
                packed.as_ptr(),
                scales.as_ptr(),
                x.as_ptr(),
                simd.as_mut_ptr(),
                rows,
                cols,
                group,
            );
        }
        assert_eq!(direct, simd);
    }

    #[test]
    fn f32_gemv_matches_float64_reference() {
        let rows = 8;
        let cols = 128;
        let mut rng = Lcg(9);
        let w: Vec<f32> = (0..rows * cols).map(|_| rng.next_unit() * 2.0 - 1.0).collect();
        let x: Vec<f32> = (0..cols).map(|_| rng.next_unit() * 2.0 - 1.0).collect();
        let mut y = vec![0.0f32; rows];
        unsafe { pb_gemv_f32(w.as_ptr(), x.as_ptr(), y.as_mut_ptr(), rows, cols) };
        for r in 0..rows {
            let expected: f64 = (0..cols)
                .map(|k| w[r * cols + k] as f64 * x[k] as f64)
                .sum();
            assert!((y[r] as f64 - expected).abs() <= 1e-4 * (1.0 + expected.abs()));
        }
    }
}
