#include <metal_stdlib>
using namespace metal;

constant bool with_variance [[function_constant(0)]];
constant bool weighted [[function_constant(1)]];
constant bool compensated [[function_constant(2)]];

// Two-component FP32 arithmetic. Explicit error-free transforms require safe
// math mode. This improves accumulation, not f32 input/weight representation.
inline float2 add_pair(float2 a, float2 b) {
    float s = a.x + b.x;
    float v = s - a.x;
    float e = (a.x - (s - v)) + (b.x - v) + a.y + b.y;
    float h = s + e;
    return float2(h, e - (h - s));
}

inline float2 product(float a, float b) {
    float h = a * b;
    return float2(h, fma(a, b, -h));
}

inline float2 reduce_pair(float2 a, uint lane, uint width) {
    for (uint offset = width / 2; offset > 0; offset /= 2) {
        float2 b = simd_shuffle_down(a, offset);
        if (lane + offset < width) a = add_pair(a, b);
    }
    return a;
}

// One threadgroup per CSR row. Strided local sums followed by SIMD and
// threadgroup reductions; no atomic floating-point updates or write races.
kernel void integrate_csr(
    device const uint* rows [[buffer(0)]],
    device const uint* pixels [[buffer(1)]],
    device const float* weights [[buffer(2)]],
    device const float* signal [[buffer(3)]],
    device float4* output [[buffer(4)]],
    uint bin [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint simd_width [[threads_per_simdgroup]],
    uint nthreads [[threads_per_threadgroup]]) {
    threadgroup float4 partial[32];
    float sum = 0.0f;
    float variance = 0.0f;
    float2 sum_pair = float2(0.0f), variance_pair = float2(0.0f);
    for (uint k = rows[bin] + tid; k < rows[bin + 1]; k += nthreads) {
        float x = signal[pixels[k]];
        float w = weighted ? weights[k] : 1.0f;
        if (compensated) {
            sum_pair = add_pair(sum_pair, product(w, x));
            if (with_variance) {
                float2 square = product(w, w);
                float positive = max(x, 0.0f);
                float2 v = product(square.x, positive);
                v.y += square.y * positive;
                variance_pair = add_pair(variance_pair, v);
            }
        } else {
            sum = fma(w, x, sum);
            if (with_variance) variance = fma(w * w, max(x, 0.0f), variance);
        }
    }
    float4 reduced;
    if (compensated) {
        reduced = float4(reduce_pair(sum_pair, lane, simd_width),
                         reduce_pair(variance_pair, lane, simd_width));
    } else {
        reduced = float4(simd_sum(sum), 0.0f, simd_sum(variance), 0.0f);
    }
    if (lane == 0) partial[simd] = reduced;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0) {
        float4 value = lane < nthreads / simd_width ? partial[lane] : float4(0.0f);
        float4 total;
        if (compensated) {
            total = float4(reduce_pair(value.xy, lane, simd_width),
                           reduce_pair(value.zw, lane, simd_width));
        } else {
            total = float4(simd_sum(value.x), 0.0f, simd_sum(value.z), 0.0f);
        }
        if (lane == 0) output[bin] = total;
    }
}
