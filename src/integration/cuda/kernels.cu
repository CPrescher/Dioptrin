// Fixed-mask CSR gather. The storage type is independent of the sum policy.
template<int Mode> struct Sum;

template<> struct Sum<0> {
    using Scalar = float;
    using WeightSq = float;
    float value;
    __device__ __forceinline__ void add(float x) { value += x; }
    __device__ __forceinline__ void combine(Sum x) { add(x.value); }
    __device__ __forceinline__ void product(float w, float x) { value = __fmaf_rn(w, x, value); }
    __device__ __forceinline__ static WeightSq square_weight(float w) { return w * w; }
    __device__ __forceinline__ void product_w2(WeightSq w2, float x) { product(w2, x); }
    __device__ __forceinline__ Sum shuffle(int offset) const {
        return {__shfl_down_sync(0xffffffff, value, offset)};
    }
    __device__ __forceinline__ double result() const { return value; }
};

template<> struct Sum<1> {
    using Scalar = float;
    using WeightSq = Sum<1>;
    float hi;
    float lo;
    __device__ __forceinline__ void combine(Sum x) {
        // TwoSum with a persistent low component, also used for tree merges.
        // Explicit round-to-nearest intrinsics keep TwoSum exact with FMA enabled.
        float total = __fadd_rn(hi, x.hi);
        if (!(fabsf(total) <= 3.40282347e+38f)) {
            // Non-finite total (|x| > FLT_MAX or NaN): keep +-inf or NaN as the
            // CPU does; TwoSum would turn an infinite total into NaN.
            hi = total;
            lo = 0.0f;
            return;
        }
        float part = __fsub_rn(total, hi);
        float error = __fadd_rn(__fsub_rn(hi, __fsub_rn(total, part)), __fsub_rn(x.hi, part));
        float tail = __fadd_rn(__fadd_rn(error, lo), x.lo);
        hi = __fadd_rn(total, tail);
        lo = __fsub_rn(tail, __fsub_rn(hi, total));
    }
    __device__ __forceinline__ void add(float x) { combine({x, 0.0f}); }
    __device__ __forceinline__ static Sum exact_product(float a, float b) {
        float hi = __fmul_rn(a, b);
        return {hi, __fmaf_rn(a, b, -hi)};
    }
    __device__ __forceinline__ void product(float w, float x) { combine(exact_product(w, x)); }
    __device__ __forceinline__ static WeightSq square_weight(float w) { return exact_product(w, w); }
    __device__ __forceinline__ void product_w2(WeightSq square, float x) {
        Sum term = exact_product(square.hi, x);
        term.lo = __fmaf_rn(square.lo, x, term.lo);
        combine(term);
    }
    __device__ __forceinline__ Sum shuffle(int offset) const {
        return {__shfl_down_sync(0xffffffff, hi, offset),
                __shfl_down_sync(0xffffffff, lo, offset)};
    }
    __device__ __forceinline__ double result() const { return double(hi) + lo; }
};

template<> struct Sum<2> {
    using Scalar = double;
    using WeightSq = double;
    double value;
    __device__ __forceinline__ void add(double x) { value += x; }
    __device__ __forceinline__ void combine(Sum x) { add(x.value); }
    __device__ __forceinline__ void product(double w, double x) { add(w * x); }
    __device__ __forceinline__ static WeightSq square_weight(double w) { return w * w; }
    __device__ __forceinline__ void product_w2(WeightSq w2, double x) { add(w2 * x); }
    __device__ __forceinline__ Sum shuffle(int offset) const {
        return {__shfl_down_sync(0xffffffff, value, offset)};
    }
    __device__ __forceinline__ double result() const { return value; }
};

template<typename Acc, bool Variance, int BinsPerBlock>
__device__ __forceinline__ void reduce_and_store(
    Acc signal, Acc variance, unsigned thread, bool valid, double denominator,
    double * __restrict__ result, Acc *shared_sig, Acc *shared_var) {
    constexpr unsigned Warps = 128 / 32;
    constexpr unsigned ThreadsPerBin = 128 / BinsPerBlock;
    constexpr int ReduceStart = ThreadsPerBin >= 32 ? 16 : (int)(ThreadsPerBin / 2);
    if constexpr (ReduceStart > 0) {
        for (int offset = ReduceStart; offset > 0; offset /= 2) {
            signal.combine(signal.shuffle(offset));
            if constexpr (Variance) variance.combine(variance.shuffle(offset));
        }
    }
    if constexpr (BinsPerBlock == 1) {
        if (thread % 32 == 0) {
            shared_sig[thread / 32] = signal;
            if constexpr (Variance) shared_var[thread / 32] = variance;
        }
        __syncthreads();
        if (thread == 0) {
            for (unsigned warp = 1; warp < Warps; ++warp) {
                signal.combine(shared_sig[warp]);
                if constexpr (Variance) variance.combine(shared_var[warp]);
            }
        }
        __syncthreads();
    }
    if (valid && thread == 0) {
        result[0] = denominator == 0 ? 0 : signal.result() / denominator;
        if constexpr (Variance) {
            result[1] = denominator == 0 ? 0 : sqrt(fmax(variance.result(), 0.0)) / fabs(denominator);
        }
    }
}

template<typename Weight, typename Pixel, int Mode, bool Variance, int BinsPerBlock, bool Weighted>
__device__ void gather_impl(const unsigned * __restrict__ rows,
                            const unsigned * __restrict__ columns,
                            const Weight * __restrict__ weights,
                            const double * __restrict__ norm,
                            const Pixel * __restrict__ frames,
                            double * __restrict__ output,
                            unsigned pixels, unsigned bins, unsigned num_frames) {
    using Acc = Sum<Mode>;
    using Scalar = typename Acc::Scalar;
    constexpr unsigned ThreadsPerBin = 128 / BinsPerBlock;
    constexpr unsigned Warps = 128 / 32;
    unsigned thread = threadIdx.x % ThreadsPerBin;
    unsigned warp_first_bin = blockIdx.x * BinsPerBlock + (threadIdx.x & ~31u) / ThreadsPerBin;
    if (warp_first_bin >= bins) return;  // Entire warps exit together.
    unsigned bin = blockIdx.x * BinsPerBlock + threadIdx.x / ThreadsPerBin;
    const bool valid = bin < bins;

    const unsigned start = valid ? __ldg(&rows[bin]) : 0u;
    const unsigned end = valid ? __ldg(&rows[bin + 1]) : 0u;
    const double denominator = valid ? __ldg(&norm[bin]) : 0.0;
    const unsigned f0 = blockIdx.y * 4;
    const unsigned out_stride = Variance ? 2 : 1;

    __shared__ Acc shared_sig[BinsPerBlock == 1 ? Warps : 1];
    __shared__ Acc shared_var[BinsPerBlock == 1 ? Warps : 1];

    if (f0 + 4 <= num_frames) {
        const Pixel * __restrict__ img0 = frames + (unsigned long long)(f0 + 0) * pixels;
        const Pixel * __restrict__ img1 = frames + (unsigned long long)(f0 + 1) * pixels;
        const Pixel * __restrict__ img2 = frames + (unsigned long long)(f0 + 2) * pixels;
        const Pixel * __restrict__ img3 = frames + (unsigned long long)(f0 + 3) * pixels;
        Acc sig0{}, sig1{}, sig2{}, sig3{};
        Acc var0{}, var1{}, var2{}, var3{};
        for (unsigned k = start + thread; k < end; k += ThreadsPerBin) {
            unsigned col = __ldg(&columns[k]);
            Scalar s0 = __ldg(&img0[col]);
            Scalar s1 = __ldg(&img1[col]);
            Scalar s2 = __ldg(&img2[col]);
            Scalar s3 = __ldg(&img3[col]);
            if constexpr (Weighted) {
                Scalar w = __ldg(&weights[k]);
                sig0.product(w, s0);
                sig1.product(w, s1);
                sig2.product(w, s2);
                sig3.product(w, s3);
                if constexpr (Variance) {
                    typename Acc::WeightSq w2 = Acc::square_weight(w);
                    var0.product_w2(w2, s0 > 0 ? s0 : 0);
                    var1.product_w2(w2, s1 > 0 ? s1 : 0);
                    var2.product_w2(w2, s2 > 0 ? s2 : 0);
                    var3.product_w2(w2, s3 > 0 ? s3 : 0);
                }
            } else {
                sig0.add(s0);
                sig1.add(s1);
                sig2.add(s2);
                sig3.add(s3);
                if constexpr (Variance) {
                    var0.add(s0 > 0 ? s0 : 0);
                    var1.add(s1 > 0 ? s1 : 0);
                    var2.add(s2 > 0 ? s2 : 0);
                    var3.add(s3 > 0 ? s3 : 0);
                }
            }
        }
        reduce_and_store<Acc, Variance, BinsPerBlock>(
            sig0, var0, thread, valid, denominator,
            output + ((unsigned long long)(f0 + 0) * bins + bin) * out_stride, shared_sig, shared_var);
        reduce_and_store<Acc, Variance, BinsPerBlock>(
            sig1, var1, thread, valid, denominator,
            output + ((unsigned long long)(f0 + 1) * bins + bin) * out_stride, shared_sig, shared_var);
        reduce_and_store<Acc, Variance, BinsPerBlock>(
            sig2, var2, thread, valid, denominator,
            output + ((unsigned long long)(f0 + 2) * bins + bin) * out_stride, shared_sig, shared_var);
        reduce_and_store<Acc, Variance, BinsPerBlock>(
            sig3, var3, thread, valid, denominator,
            output + ((unsigned long long)(f0 + 3) * bins + bin) * out_stride, shared_sig, shared_var);
    } else {
        for (unsigned f = f0; f < num_frames; ++f) {
            const Pixel * __restrict__ image = frames + (unsigned long long)f * pixels;
            Acc signal{}, variance{};
            for (unsigned k = start + thread; k < end; k += ThreadsPerBin) {
                unsigned col = __ldg(&columns[k]);
                Scalar s = __ldg(&image[col]);
                if constexpr (Weighted) {
                    Scalar w = __ldg(&weights[k]);
                    signal.product(w, s);
                    if constexpr (Variance) {
                        variance.product_w2(Acc::square_weight(w), s > 0 ? s : 0);
                    }
                } else {
                    signal.add(s);
                    if constexpr (Variance) {
                        variance.add(s > 0 ? s : 0);
                    }
                }
            }
            reduce_and_store<Acc, Variance, BinsPerBlock>(
                signal, variance, thread, valid, denominator,
                output + ((unsigned long long)f * bins + bin) * out_stride, shared_sig, shared_var);
        }
    }
}

template<typename Weight, typename Pixel, int Mode, bool Variance, int BinsPerBlock>
__device__ void gather(const unsigned * __restrict__ rows,
                       const unsigned * __restrict__ columns,
                       const Weight * __restrict__ weights,
                       const double * __restrict__ norm,
                       const Pixel * __restrict__ frames,
                       double * __restrict__ output,
                       unsigned pixels, unsigned bins, unsigned weighted, unsigned num_frames) {
    if (weighted) {
        gather_impl<Weight, Pixel, Mode, Variance, BinsPerBlock, true>(
            rows, columns, weights, norm, frames, output, pixels, bins, num_frames);
    } else {
        gather_impl<Weight, Pixel, Mode, Variance, BinsPerBlock, false>(
            rows, columns, weights, norm, frames, output, pixels, bins, num_frames);
    }
}

#define STORED_KERNEL(Name, Weight, Pixel, Mode, Variance, Bins) \
extern "C" __global__ void Name(const unsigned * __restrict__ rows, \
                                const unsigned * __restrict__ columns, \
                                const Weight * __restrict__ weights, \
                                const double * __restrict__ norm, \
                                const Pixel * __restrict__ frames, \
                                double * __restrict__ output, \
                                unsigned pixels, unsigned bins, unsigned weighted, \
                                unsigned num_frames) { \
    gather<Weight, Pixel, Mode, Variance, Bins>(rows, columns, weights, norm, frames, output, pixels, bins, weighted, num_frames); \
}

#define KERNEL(Name, Mode, Variance, Bins) STORED_KERNEL(Name, float, float, Mode, Variance, Bins)
KERNEL(plain_block_i, 0, false, 1)
KERNEL(plain_block_iv, 0, true, 1)
KERNEL(compensated_block_i, 1, false, 1)
KERNEL(compensated_block_iv, 1, true, 1)
KERNEL(double_block_i, 2, false, 1)
KERNEL(double_block_iv, 2, true, 1)
KERNEL(plain_warp_i, 0, false, 4)
KERNEL(plain_warp_iv, 0, true, 4)
KERNEL(compensated_warp_i, 1, false, 4)
KERNEL(compensated_warp_iv, 1, true, 4)
KERNEL(double_warp_i, 2, false, 4)
KERNEL(double_warp_iv, 2, true, 4)
KERNEL(plain_subwarp_i, 0, false, 16)
KERNEL(plain_subwarp_iv, 0, true, 16)
KERNEL(compensated_subwarp_i, 1, false, 16)
KERNEL(compensated_subwarp_iv, 1, true, 16)
KERNEL(double_subwarp_i, 2, false, 16)
KERNEL(double_subwarp_iv, 2, true, 16)
#define TYPED_KERNELS(Prefix, Pixel, Mode) \
STORED_KERNEL(Prefix##_block_i, float, Pixel, Mode, false, 1) \
STORED_KERNEL(Prefix##_block_iv, float, Pixel, Mode, true, 1) \
STORED_KERNEL(Prefix##_warp_i, float, Pixel, Mode, false, 4) \
STORED_KERNEL(Prefix##_warp_iv, float, Pixel, Mode, true, 4) \
STORED_KERNEL(Prefix##_subwarp_i, float, Pixel, Mode, false, 16) \
STORED_KERNEL(Prefix##_subwarp_iv, float, Pixel, Mode, true, 16)

TYPED_KERNELS(u16_plain, unsigned short, 0)
TYPED_KERNELS(u32_plain, unsigned, 0)
TYPED_KERNELS(i32_plain, int, 0)
TYPED_KERNELS(u16_compensated, unsigned short, 1)
TYPED_KERNELS(u32_compensated, unsigned, 1)
TYPED_KERNELS(i32_compensated, int, 1)

#define F64_KERNELS(Prefix, Pixel) \
STORED_KERNEL(Prefix##_block_i, double, Pixel, 2, false, 1) \
STORED_KERNEL(Prefix##_block_iv, double, Pixel, 2, true, 1) \
STORED_KERNEL(Prefix##_warp_i, double, Pixel, 2, false, 4) \
STORED_KERNEL(Prefix##_warp_iv, double, Pixel, 2, true, 4) \
STORED_KERNEL(Prefix##_subwarp_i, double, Pixel, 2, false, 16) \
STORED_KERNEL(Prefix##_subwarp_iv, double, Pixel, 2, true, 16)

F64_KERNELS(full_f64, double)
F64_KERNELS(f32_f64, float)
F64_KERNELS(u16_f64, unsigned short)
F64_KERNELS(u32_f64, unsigned)
F64_KERNELS(i32_f64, int)

extern "C" __global__ void cast_f64_input(const double * __restrict__ src,
                                          float * __restrict__ dst,
                                          unsigned long long len) {
    const unsigned long long stride = (unsigned long long)blockDim.x * gridDim.x;
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < len; i += stride) {
        dst[i] = __double2float_rn(src[i]);
    }
}

// Warp-per-block Bitshuffle + LZ4 decoder.
// Each block descriptor is 4 unsigned ints: (src_off, comp_len, dst_off, uncomp_bytes).
// Bit 31 of comp_len marks a verbatim uncompressed tail (< 8 elements).
template<typename Elem, unsigned ElemBytes>
__device__ void bshuf_lz4_decode_impl(const unsigned char * __restrict__ comp_buf,
                                      const unsigned * __restrict__ descs,
                                      Elem * __restrict__ out_pixels,
                                      unsigned * __restrict__ err_flag) {
    extern __shared__ unsigned char sh[];
    const unsigned blk = blockIdx.x;
    const unsigned lane = threadIdx.x;
    const unsigned src_off = __ldg(&descs[blk * 4 + 0]);
    const unsigned comp_len = __ldg(&descs[blk * 4 + 1]);
    const unsigned dst_off = __ldg(&descs[blk * 4 + 2]);
    const unsigned uncomp_bytes = __ldg(&descs[blk * 4 + 3]);

    if (comp_len & 0x80000000u) {
        const unsigned raw_bytes = comp_len & 0x7fffffffu;
        unsigned char *dst = reinterpret_cast<unsigned char *>(out_pixels + dst_off);
        for (unsigned i = lane; i < raw_bytes; i += 32) {
            dst[i] = comp_buf[src_off + i];
        }
        return;
    }

    const unsigned char *ip = comp_buf + src_off;
    unsigned char *op = sh;
    unsigned ip_off = 0;
    unsigned op_off = 0;
    while (true) {
        unsigned token = 0;
        unsigned lit_len = 0;
        unsigned status = 0; // 0 = ok, 1 = end of block, 2 = error
        if (lane == 0) {
            if (ip_off >= comp_len) {
                status = 1;
            } else {
                token = ip[ip_off++];
                lit_len = token >> 4;
                if (lit_len == 15) {
                    unsigned s;
                    do {
                        if (ip_off >= comp_len) { status = 2; break; }
                        s = ip[ip_off++];
                        if (s > uncomp_bytes - op_off - lit_len) { status = 2; break; }
                        lit_len += s;
                    } while (s == 255);
                }
                if (status == 0
                    && (lit_len > comp_len - ip_off || lit_len > uncomp_bytes - op_off)) {
                    status = 2;
                }
            }
        }
        status = __shfl_sync(0xffffffffu, status, 0);
        if (status == 1) break;
        if (status == 2) {
            if (lane == 0) *err_flag = 1;
            return;
        }
        lit_len = __shfl_sync(0xffffffffu, lit_len, 0);
        unsigned lit_src = __shfl_sync(0xffffffffu, ip_off, 0);
        for (unsigned i = lane; i < lit_len; i += 32) {
            op[op_off + i] = ip[lit_src + i];
        }
        __syncwarp();
        ip_off = lit_src + lit_len;
        op_off += lit_len;

        unsigned offset = 0;
        unsigned match_len = 0;
        if (lane == 0) {
            if (ip_off >= comp_len) {
                status = 1;
            } else if (ip_off + 2 > comp_len) {
                status = 2;
            } else {
                offset = (unsigned)ip[ip_off] | ((unsigned)ip[ip_off + 1] << 8);
                ip_off += 2;
                if (offset == 0 || offset > op_off) {
                    status = 2;
                } else {
                    match_len = (token & 15u) + 4u;
                    if ((token & 15u) == 15) {
                        unsigned s;
                        do {
                            if (ip_off >= comp_len) { status = 2; break; }
                            s = ip[ip_off++];
                            if (match_len > uncomp_bytes - op_off
                                || s > uncomp_bytes - op_off - match_len) {
                                status = 2;
                                break;
                            }
                            match_len += s;
                        } while (s == 255);
                    }
                    if (status == 0 && match_len > uncomp_bytes - op_off) {
                        status = 2;
                    }
                }
            }
        }
        status = __shfl_sync(0xffffffffu, status, 0);
        if (status == 1) break;
        if (status == 2) {
            if (lane == 0) *err_flag = 1;
            return;
        }
        offset = __shfl_sync(0xffffffffu, offset, 0);
        match_len = __shfl_sync(0xffffffffu, match_len, 0);
        ip_off = __shfl_sync(0xffffffffu, ip_off, 0);

        if (offset == 1) {
            unsigned char fill = op[op_off - 1];
            for (unsigned i = lane; i < match_len; i += 32) {
                op[op_off + i] = fill;
            }
            __syncwarp();
        } else if (offset >= 32) {
            for (unsigned done = 0; done < match_len; done += 32) {
                unsigned idx = done + lane;
                if (idx < match_len) {
                    op[op_off + idx] = op[op_off + idx - offset];
                }
                __syncwarp();
            }
        } else {
            for (unsigned done = 0; done < match_len; done += offset) {
                if (lane < offset && done + lane < match_len) {
                    op[op_off + done + lane] = op[op_off + done + lane - offset];
                }
                __syncwarp();
            }
        }
        op_off += match_len;
    }
    if (op_off != uncomp_bytes) {
        if (lane == 0) *err_flag = 1;
        return;
    }

    const unsigned n_elems = uncomp_bytes / ElemBytes;
    const unsigned plane_bytes = n_elems >> 3;
    unsigned base = 0;
    for (; base + 32 <= n_elems; base += 32) {
        unsigned plane = 0;
        if (lane < ElemBytes * 8) {
            unsigned off = lane * plane_bytes + (base >> 3);
            plane = (unsigned)sh[off]
                  | ((unsigned)sh[off + 1] << 8)
                  | ((unsigned)sh[off + 2] << 16)
                  | ((unsigned)sh[off + 3] << 24);
        }
        unsigned my_val = 0;
        #pragma unroll
        for (unsigned k = 0; k < 32; ++k) {
            unsigned v = __ballot_sync(0xffffffffu, (plane >> k) & 1u);
            if (lane == k) my_val = v;
        }
        out_pixels[dst_off + base + lane] = (Elem)my_val;
    }
    const unsigned rem = n_elems - base;
    if (rem > 0) {
        unsigned plane = 0;
        if (lane < ElemBytes * 8) {
            unsigned off = lane * plane_bytes + (base >> 3);
            for (unsigned b = 0; b < (rem >> 3); ++b) {
                plane |= (unsigned)sh[off + b] << (b * 8);
            }
        }
        unsigned my_val = 0;
        for (unsigned k = 0; k < rem; ++k) {
            unsigned v = __ballot_sync(0xffffffffu, (plane >> k) & 1u);
            if (lane == k) my_val = v;
        }
        if (lane < rem) out_pixels[dst_off + base + lane] = (Elem)my_val;
    }
}

extern "C" __global__ void bshuf_lz4_decode_u16(const unsigned char * __restrict__ comp_buf,
                                                const unsigned * __restrict__ descs,
                                                unsigned short * __restrict__ out_pixels,
                                                unsigned * __restrict__ err_flag) {
    bshuf_lz4_decode_impl<unsigned short, 2>(comp_buf, descs, out_pixels, err_flag);
}

extern "C" __global__ void bshuf_lz4_decode_u32(const unsigned char * __restrict__ comp_buf,
                                                const unsigned * __restrict__ descs,
                                                unsigned * __restrict__ out_pixels,
                                                unsigned * __restrict__ err_flag) {
    bshuf_lz4_decode_impl<unsigned, 4>(comp_buf, descs, out_pixels, err_flag);
}
