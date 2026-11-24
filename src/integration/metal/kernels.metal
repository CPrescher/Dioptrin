#include <metal_stdlib>
using namespace metal;

constant bool WITH_VARIANCE [[function_constant(0)]];
constant bool WEIGHTED [[function_constant(1)]];
constant uint ACC_MODE [[function_constant(2)]];       // 0 = plain f32, 1 = compensated f32
constant uint BINS_PER_TG [[function_constant(3)]];    // 1 = Block (128), 4 = Warp (32), 16 = SubWarp (8)
constant uint FRAMES_PER_TG [[function_constant(4)]];  // 4 or 8 frames per threadgroup

struct Params {
    uint pixels;
    uint bins;
    uint num_frames;
};

struct PlainAcc {
    float value;

    inline void add(float x) { value += x; }
    inline void combine(PlainAcc x) { value += x.value; }
    inline void product(float w, float x) { value = fma(w, x, value); }
    inline static float square_weight(float w) { return w * w; }
    inline void product_w2(float w2, float x) { value = fma(w2, x, value); }
    inline PlainAcc shuffle_down(ushort offset) const {
        return { simd_shuffle_down(value, offset) };
    }
    inline float2 pair() const { return float2(value, 0.0f); }
};

template<bool CheckFinite>
struct CompAcc {
    float hi;
    float lo;

    // Knuth TwoSum with persistent low component. Compiled with MTLMathMode::Safe
    // so floating-point operations are not contracted or reassociated.
    // For integer pixel inputs (CheckFinite == false), pixels can never be NaN/Inf
    // so the non-finite branch is eliminated from the inner loop.
    inline void combine(CompAcc<CheckFinite> x) {
        float total = hi + x.hi;
        if (CheckFinite && !isfinite(total)) {
            hi = total;
            lo = 0.0f;
            return;
        }
        float part = total - hi;
        float error = (hi - (total - part)) + (x.hi - part);
        float tail = (error + lo) + x.lo;
        hi = total + tail;
        lo = tail - (hi - total);
    }
    inline void add(float x) { combine({ x, 0.0f }); }
    inline static CompAcc<CheckFinite> exact_product(float a, float b) {
        float h = a * b;
        return { h, fma(a, b, -h) };
    }
    inline void product(float w, float x) { combine(exact_product(w, x)); }
    inline static CompAcc<CheckFinite> square_weight(float w) { return exact_product(w, w); }
    inline void product_w2(CompAcc<CheckFinite> square, float x) {
        CompAcc<CheckFinite> term = exact_product(square.hi, x);
        term.lo = fma(square.lo, x, term.lo);
        combine(term);
    }
    inline CompAcc<CheckFinite> shuffle_down(ushort offset) const {
        return { simd_shuffle_down(hi, offset), simd_shuffle_down(lo, offset) };
    }
    inline float2 pair() const { return float2(hi, lo); }
};

inline void combine_from_float2(thread PlainAcc& acc, float2 p) {
    acc.add(p.x);
}
template<bool CheckFinite>
inline void combine_from_float2(thread CompAcc<CheckFinite>& acc, float2 p) {
    acc.combine(CompAcc<CheckFinite>{ p.x, p.y });
}

template<typename Acc>
inline void reduce_and_store(
    Acc signal,
    Acc variance,
    uint thread_in_bin,
    bool valid,
    device float* output,
    ulong out_offset,
    threadgroup float4* shared_buf
) {
    const uint threads_per_bin = 128u / BINS_PER_TG;
    const ushort reduce_start = threads_per_bin >= 32u ? 16u : ushort(threads_per_bin >> 1);

    for (ushort offset = reduce_start; offset > 0; offset >>= 1) {
        signal.combine(signal.shuffle_down(offset));
        if (WITH_VARIANCE) {
            variance.combine(variance.shuffle_down(offset));
        }
    }

    if (BINS_PER_TG == 1u) {
        if ((thread_in_bin & 31u) == 0u) {
            float2 sp = signal.pair();
            float2 vp = WITH_VARIANCE ? variance.pair() : float2(0.0f);
            shared_buf[thread_in_bin >> 5] = float4(sp.x, sp.y, vp.x, vp.y);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (thread_in_bin == 0u) {
            for (uint w = 1u; w < 4u; ++w) {
                float4 p = shared_buf[w];
                combine_from_float2(signal, p.xy);
                if (WITH_VARIANCE) combine_from_float2(variance, p.zw);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (valid && thread_in_bin == 0u) {
        float2 sp = signal.pair();
        output[out_offset + 0] = sp.x;
        output[out_offset + 1] = sp.y;
        if (WITH_VARIANCE) {
            float2 vp = variance.pair();
            output[out_offset + 2] = vp.x;
            output[out_offset + 3] = vp.y;
        }
    }
}

template<typename Pixel, typename Acc>
inline void gather_typed(
    device const uint* rows,
    device const uint* columns,
    device const float* weights,
    device const Pixel* frames,
    device float* output,
    constant Params& params,
    threadgroup float4* shared_buf,
    uint2 tg_pos,
    uint tid
) {
    const uint threads_per_bin = 128u / BINS_PER_TG;
    const uint thread_in_bin = tid % threads_per_bin;
    const uint warp_first_bin = tg_pos.x * BINS_PER_TG + (tid & ~31u) / threads_per_bin;
    if (warp_first_bin >= params.bins) return;

    const uint bin = tg_pos.x * BINS_PER_TG + tid / threads_per_bin;
    const bool valid = bin < params.bins;

    const uint start = valid ? rows[bin] : 0u;
    const uint end = valid ? rows[bin + 1u] : 0u;
    uint f0 = tg_pos.y * FRAMES_PER_TG;
    const uint f_end = min(f0 + FRAMES_PER_TG, params.num_frames);
    const ulong out_stride = WITH_VARIANCE ? 4ul : 2ul;

    if (FRAMES_PER_TG == 8u && f0 + 8u <= f_end) {
        device const Pixel* img0 = frames + ulong(f0 + 0u) * ulong(params.pixels);
        device const Pixel* img1 = frames + ulong(f0 + 1u) * ulong(params.pixels);
        device const Pixel* img2 = frames + ulong(f0 + 2u) * ulong(params.pixels);
        device const Pixel* img3 = frames + ulong(f0 + 3u) * ulong(params.pixels);
        device const Pixel* img4 = frames + ulong(f0 + 4u) * ulong(params.pixels);
        device const Pixel* img5 = frames + ulong(f0 + 5u) * ulong(params.pixels);
        device const Pixel* img6 = frames + ulong(f0 + 6u) * ulong(params.pixels);
        device const Pixel* img7 = frames + ulong(f0 + 7u) * ulong(params.pixels);

        Acc sig0{}, sig1{}, sig2{}, sig3{}, sig4{}, sig5{}, sig6{}, sig7{};
        Acc var0{}, var1{}, var2{}, var3{}, var4{}, var5{}, var6{}, var7{};

        for (uint k = start + thread_in_bin; k < end; k += threads_per_bin) {
            uint col = columns[k];
            float s0 = float(img0[col]);
            float s1 = float(img1[col]);
            float s2 = float(img2[col]);
            float s3 = float(img3[col]);
            float s4 = float(img4[col]);
            float s5 = float(img5[col]);
            float s6 = float(img6[col]);
            float s7 = float(img7[col]);

            if (WEIGHTED) {
                float w = weights[k];
                sig0.product(w, s0);
                sig1.product(w, s1);
                sig2.product(w, s2);
                sig3.product(w, s3);
                sig4.product(w, s4);
                sig5.product(w, s5);
                sig6.product(w, s6);
                sig7.product(w, s7);
                if (WITH_VARIANCE) {
                    auto w2 = Acc::square_weight(w);
                    var0.product_w2(w2, s0 > 0.0f ? s0 : 0.0f);
                    var1.product_w2(w2, s1 > 0.0f ? s1 : 0.0f);
                    var2.product_w2(w2, s2 > 0.0f ? s2 : 0.0f);
                    var3.product_w2(w2, s3 > 0.0f ? s3 : 0.0f);
                    var4.product_w2(w2, s4 > 0.0f ? s4 : 0.0f);
                    var5.product_w2(w2, s5 > 0.0f ? s5 : 0.0f);
                    var6.product_w2(w2, s6 > 0.0f ? s6 : 0.0f);
                    var7.product_w2(w2, s7 > 0.0f ? s7 : 0.0f);
                }
            } else {
                sig0.add(s0);
                sig1.add(s1);
                sig2.add(s2);
                sig3.add(s3);
                sig4.add(s4);
                sig5.add(s5);
                sig6.add(s6);
                sig7.add(s7);
                if (WITH_VARIANCE) {
                    var0.add(s0 > 0.0f ? s0 : 0.0f);
                    var1.add(s1 > 0.0f ? s1 : 0.0f);
                    var2.add(s2 > 0.0f ? s2 : 0.0f);
                    var3.add(s3 > 0.0f ? s3 : 0.0f);
                    var4.add(s4 > 0.0f ? s4 : 0.0f);
                    var5.add(s5 > 0.0f ? s5 : 0.0f);
                    var6.add(s6 > 0.0f ? s6 : 0.0f);
                    var7.add(s7 > 0.0f ? s7 : 0.0f);
                }
            }
        }

        reduce_and_store(
            sig0, var0, thread_in_bin, valid, output,
            (ulong(f0 + 0u) * ulong(params.bins) + ulong(bin)) * out_stride, shared_buf);
        reduce_and_store(
            sig1, var1, thread_in_bin, valid, output,
            (ulong(f0 + 1u) * ulong(params.bins) + ulong(bin)) * out_stride, shared_buf);
        reduce_and_store(
            sig2, var2, thread_in_bin, valid, output,
            (ulong(f0 + 2u) * ulong(params.bins) + ulong(bin)) * out_stride, shared_buf);
        reduce_and_store(
            sig3, var3, thread_in_bin, valid, output,
            (ulong(f0 + 3u) * ulong(params.bins) + ulong(bin)) * out_stride, shared_buf);
        reduce_and_store(
            sig4, var4, thread_in_bin, valid, output,
            (ulong(f0 + 4u) * ulong(params.bins) + ulong(bin)) * out_stride, shared_buf);
        reduce_and_store(
            sig5, var5, thread_in_bin, valid, output,
            (ulong(f0 + 5u) * ulong(params.bins) + ulong(bin)) * out_stride, shared_buf);
        reduce_and_store(
            sig6, var6, thread_in_bin, valid, output,
            (ulong(f0 + 6u) * ulong(params.bins) + ulong(bin)) * out_stride, shared_buf);
        reduce_and_store(
            sig7, var7, thread_in_bin, valid, output,
            (ulong(f0 + 7u) * ulong(params.bins) + ulong(bin)) * out_stride, shared_buf);
        return;
    }

    if (f0 + 4u <= f_end) {
        device const Pixel* img0 = frames + ulong(f0 + 0u) * ulong(params.pixels);
        device const Pixel* img1 = frames + ulong(f0 + 1u) * ulong(params.pixels);
        device const Pixel* img2 = frames + ulong(f0 + 2u) * ulong(params.pixels);
        device const Pixel* img3 = frames + ulong(f0 + 3u) * ulong(params.pixels);

        Acc sig0{}, sig1{}, sig2{}, sig3{};
        Acc var0{}, var1{}, var2{}, var3{};

        for (uint k = start + thread_in_bin; k < end; k += threads_per_bin) {
            uint col = columns[k];
            float s0 = float(img0[col]);
            float s1 = float(img1[col]);
            float s2 = float(img2[col]);
            float s3 = float(img3[col]);

            if (WEIGHTED) {
                float w = weights[k];
                sig0.product(w, s0);
                sig1.product(w, s1);
                sig2.product(w, s2);
                sig3.product(w, s3);
                if (WITH_VARIANCE) {
                    auto w2 = Acc::square_weight(w);
                    var0.product_w2(w2, s0 > 0.0f ? s0 : 0.0f);
                    var1.product_w2(w2, s1 > 0.0f ? s1 : 0.0f);
                    var2.product_w2(w2, s2 > 0.0f ? s2 : 0.0f);
                    var3.product_w2(w2, s3 > 0.0f ? s3 : 0.0f);
                }
            } else {
                sig0.add(s0);
                sig1.add(s1);
                sig2.add(s2);
                sig3.add(s3);
                if (WITH_VARIANCE) {
                    var0.add(s0 > 0.0f ? s0 : 0.0f);
                    var1.add(s1 > 0.0f ? s1 : 0.0f);
                    var2.add(s2 > 0.0f ? s2 : 0.0f);
                    var3.add(s3 > 0.0f ? s3 : 0.0f);
                }
            }
        }

        reduce_and_store(
            sig0, var0, thread_in_bin, valid, output,
            (ulong(f0 + 0u) * ulong(params.bins) + ulong(bin)) * out_stride, shared_buf);
        reduce_and_store(
            sig1, var1, thread_in_bin, valid, output,
            (ulong(f0 + 1u) * ulong(params.bins) + ulong(bin)) * out_stride, shared_buf);
        reduce_and_store(
            sig2, var2, thread_in_bin, valid, output,
            (ulong(f0 + 2u) * ulong(params.bins) + ulong(bin)) * out_stride, shared_buf);
        reduce_and_store(
            sig3, var3, thread_in_bin, valid, output,
            (ulong(f0 + 3u) * ulong(params.bins) + ulong(bin)) * out_stride, shared_buf);
        f0 += 4u;
    }

    for (uint f = f0; f < f_end; ++f) {
        device const Pixel* image = frames + ulong(f) * ulong(params.pixels);
        Acc signal{}, variance{};
        for (uint k = start + thread_in_bin; k < end; k += threads_per_bin) {
            uint col = columns[k];
            float s = float(image[col]);
            if (WEIGHTED) {
                float w = weights[k];
                signal.product(w, s);
                if (WITH_VARIANCE) {
                    variance.product_w2(Acc::square_weight(w), s > 0.0f ? s : 0.0f);
                }
            } else {
                signal.add(s);
                if (WITH_VARIANCE) {
                    variance.add(s > 0.0f ? s : 0.0f);
                }
            }
        }
        reduce_and_store(
            signal, variance, thread_in_bin, valid, output,
            (ulong(f) * ulong(params.bins) + ulong(bin)) * out_stride, shared_buf);
    }
}

template<typename Pixel, bool CheckFinite>
inline void dispatch_gather(
    device const uint* rows,
    device const uint* columns,
    device const float* weights,
    device const Pixel* frames,
    device float* output,
    constant Params& params,
    threadgroup float4* shared_buf,
    uint2 tg_pos,
    uint tid
) {
    if (ACC_MODE == 1u) {
        gather_typed<Pixel, CompAcc<CheckFinite>>(
            rows, columns, weights, frames, output, params, shared_buf, tg_pos, tid);
    } else {
        gather_typed<Pixel, PlainAcc>(
            rows, columns, weights, frames, output, params, shared_buf, tg_pos, tid);
    }
}

kernel void integrate_f32(
    device const uint* rows [[buffer(0)]],
    device const uint* columns [[buffer(1)]],
    device const float* weights [[buffer(2)]],
    device const float* frames [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant Params& params [[buffer(5)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float4 shared_buf[4];
    dispatch_gather<float, true>(rows, columns, weights, frames, output, params, shared_buf, tg_pos, tid);
}

kernel void integrate_u16(
    device const uint* rows [[buffer(0)]],
    device const uint* columns [[buffer(1)]],
    device const float* weights [[buffer(2)]],
    device const ushort* frames [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant Params& params [[buffer(5)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float4 shared_buf[4];
    dispatch_gather<ushort, false>(rows, columns, weights, frames, output, params, shared_buf, tg_pos, tid);
}

kernel void integrate_u32(
    device const uint* rows [[buffer(0)]],
    device const uint* columns [[buffer(1)]],
    device const float* weights [[buffer(2)]],
    device const uint* frames [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant Params& params [[buffer(5)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float4 shared_buf[4];
    dispatch_gather<uint, false>(rows, columns, weights, frames, output, params, shared_buf, tg_pos, tid);
}

kernel void integrate_i32(
    device const uint* rows [[buffer(0)]],
    device const uint* columns [[buffer(1)]],
    device const float* weights [[buffer(2)]],
    device const int* frames [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant Params& params [[buffer(5)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float4 shared_buf[4];
    dispatch_gather<int, false>(rows, columns, weights, frames, output, params, shared_buf, tg_pos, tid);
}

// SIMD-group-per-block Bitshuffle + LZ4 decoder (32 threads per threadgroup).
// Each block descriptor is 4 uints: (src_off, comp_len, dst_off, uncomp_bytes).
// Bit 31 of comp_len marks a verbatim uncompressed tail (< 8 elements).
template<typename Elem, uint ElemBytes>
inline void bshuf_lz4_decode_impl(
    device const uchar* comp_buf,
    device const uint* descs,
    device Elem* out_pixels,
    device atomic_uint* err_flag,
    threadgroup uchar* sh,
    uint blk,
    uint lane
) {
    const uint src_off = descs[blk * 4u + 0u];
    const uint comp_len = descs[blk * 4u + 1u];
    const uint dst_off = descs[blk * 4u + 2u];
    const uint uncomp_bytes = descs[blk * 4u + 3u];

    if (comp_len & 0x80000000u) {
        const uint raw_bytes = comp_len & 0x7fffffffu;
        device uchar* dst = reinterpret_cast<device uchar*>(out_pixels + dst_off);
        for (uint i = lane; i < raw_bytes; i += 32u) {
            dst[i] = comp_buf[src_off + i];
        }
        return;
    }

    device const uchar* ip = comp_buf + src_off;
    threadgroup uchar* op = sh;
    uint ip_off = 0u;
    uint op_off = 0u;

    while (true) {
        uint token = 0u;
        uint lit_len = 0u;
        uint status = 0u; // 0 = ok, 1 = end of block, 2 = error
        if (lane == 0u) {
            if (ip_off >= comp_len) {
                status = 1u;
            } else {
                token = ip[ip_off++];
                lit_len = token >> 4u;
                if (lit_len == 15u) {
                    uint s;
                    do {
                        if (ip_off >= comp_len) { status = 2u; break; }
                        s = ip[ip_off++];
                        if (s > uncomp_bytes - op_off - lit_len) { status = 2u; break; }
                        lit_len += s;
                    } while (s == 255u);
                }
                if (status == 0u
                    && (lit_len > comp_len - ip_off || lit_len > uncomp_bytes - op_off)) {
                    status = 2u;
                }
            }
        }
        status = simd_broadcast(status, 0u);
        if (status == 1u) break;
        if (status == 2u) {
            if (lane == 0u) atomic_store_explicit(err_flag, 1u, memory_order_relaxed);
            return;
        }
        lit_len = simd_broadcast(lit_len, 0u);
        uint lit_src = simd_broadcast(ip_off, 0u);
        for (uint i = lane; i < lit_len; i += 32u) {
            op[op_off + i] = ip[lit_src + i];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        ip_off = lit_src + lit_len;
        op_off += lit_len;

        uint offset = 0u;
        uint match_len = 0u;
        if (lane == 0u) {
            if (ip_off >= comp_len) {
                status = 1u;
            } else if (ip_off + 2u > comp_len) {
                status = 2u;
            } else {
                offset = uint(ip[ip_off]) | (uint(ip[ip_off + 1u]) << 8u);
                ip_off += 2u;
                if (offset == 0u || offset > op_off) {
                    status = 2u;
                } else {
                    match_len = (token & 15u) + 4u;
                    if ((token & 15u) == 15u) {
                        uint s;
                        do {
                            if (ip_off >= comp_len) { status = 2u; break; }
                            s = ip[ip_off++];
                            if (match_len > uncomp_bytes - op_off
                                || s > uncomp_bytes - op_off - match_len) {
                                status = 2u;
                                break;
                            }
                            match_len += s;
                        } while (s == 255u);
                    }
                    if (status == 0u && match_len > uncomp_bytes - op_off) {
                        status = 2u;
                    }
                }
            }
        }
        status = simd_broadcast(status, 0u);
        if (status == 1u) break;
        if (status == 2u) {
            if (lane == 0u) atomic_store_explicit(err_flag, 1u, memory_order_relaxed);
            return;
        }
        offset = simd_broadcast(offset, 0u);
        match_len = simd_broadcast(match_len, 0u);
        ip_off = simd_broadcast(ip_off, 0u);

        if (offset == 1u) {
            uchar fill = op[op_off - 1u];
            for (uint i = lane; i < match_len; i += 32u) {
                op[op_off + i] = fill;
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        } else if (offset >= 32u) {
            for (uint done = 0u; done < match_len; done += 32u) {
                uint idx = done + lane;
                if (idx < match_len) {
                    op[op_off + idx] = op[op_off + idx - offset];
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
        } else {
            for (uint done = 0u; done < match_len; done += offset) {
                if (lane < offset && done + lane < match_len) {
                    op[op_off + done + lane] = op[op_off + done + lane - offset];
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        op_off += match_len;
    }

    if (op_off != uncomp_bytes) {
        if (lane == 0u) atomic_store_explicit(err_flag, 1u, memory_order_relaxed);
        return;
    }

    const uint n_elems = uncomp_bytes / ElemBytes;
    const uint plane_bytes = n_elems >> 3u;
    uint base = 0u;
    for (; base + 32u <= n_elems; base += 32u) {
        uint plane = 0u;
        if (lane < ElemBytes * 8u) {
            uint off = lane * plane_bytes + (base >> 3u);
            plane = uint(sh[off])
                  | (uint(sh[off + 1u]) << 8u)
                  | (uint(sh[off + 2u]) << 16u)
                  | (uint(sh[off + 3u]) << 24u);
        }
        uint my_val = 0u;
        for (uint k = 0u; k < 32u; ++k) {
            uint v = uint(static_cast<simd_vote::vote_t>(simd_ballot(((plane >> k) & 1u) != 0u)));
            if (lane == k) my_val = v;
        }
        out_pixels[dst_off + base + lane] = Elem(my_val);
    }
    const uint rem = n_elems - base;
    if (rem > 0u) {
        uint plane = 0u;
        if (lane < ElemBytes * 8u) {
            uint off = lane * plane_bytes + (base >> 3u);
            for (uint b = 0u; b < (rem >> 3u); ++b) {
                plane |= uint(sh[off + b]) << (b * 8u);
            }
        }
        uint my_val = 0u;
        for (uint k = 0u; k < rem; ++k) {
            uint v = uint(static_cast<simd_vote::vote_t>(simd_ballot(((plane >> k) & 1u) != 0u)));
            if (lane == k) my_val = v;
        }
        if (lane < rem) out_pixels[dst_off + base + lane] = Elem(my_val);
    }
}

kernel void bshuf_lz4_decode_u16(
    device const uchar* comp_buf [[buffer(0)]],
    device const uint* descs [[buffer(1)]],
    device ushort* out_pixels [[buffer(2)]],
    device atomic_uint* err_flag [[buffer(3)]],
    threadgroup uchar* sh [[threadgroup(0)]],
    uint blk [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]
) {
    bshuf_lz4_decode_impl<ushort, 2u>(comp_buf, descs, out_pixels, err_flag, sh, blk, lane);
}

kernel void bshuf_lz4_decode_u32(
    device const uchar* comp_buf [[buffer(0)]],
    device const uint* descs [[buffer(1)]],
    device uint* out_pixels [[buffer(2)]],
    device atomic_uint* err_flag [[buffer(3)]],
    threadgroup uchar* sh [[threadgroup(0)]],
    uint blk [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]
) {
    bshuf_lz4_decode_impl<uint, 4u>(comp_buf, descs, out_pixels, err_flag, sh, blk, lane);
}
