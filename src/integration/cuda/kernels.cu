// Fixed-mask CSR gather. Inputs and weights are f32 in all three policies,
// so comparing the policies isolates accumulation and product precision.
template<int Mode> struct Sum;

template<> struct Sum<0> {
    using Scalar = float;
    float value;
    __device__ void add(float x) { value += x; }
    __device__ void combine(Sum x) { add(x.value); }
    __device__ Sum shuffle(int offset) const {
        return {__shfl_down_sync(0xffffffff, value, offset)};
    }
    __device__ double result() const { return value; }
};

template<> struct Sum<1> {
    using Scalar = float;
    float hi;
    float lo;
    __device__ void add(float x) {
        // TwoSum with a persistent low component, also used for tree merges.
        float total = hi + x;
        float part = total - hi;
        float error = (hi - (total - part)) + (x - part);
        float tail = lo + error;
        hi = total + tail;
        lo = tail - (hi - total);
    }
    __device__ void combine(Sum x) {
        add(x.hi);
        add(x.lo);
    }
    __device__ Sum shuffle(int offset) const {
        return {__shfl_down_sync(0xffffffff, hi, offset),
                __shfl_down_sync(0xffffffff, lo, offset)};
    }
    __device__ double result() const { return double(hi) + lo; }
};

template<> struct Sum<2> {
    using Scalar = double;
    double value;
    __device__ void add(double x) { value += x; }
    __device__ void combine(Sum x) { add(x.value); }
    __device__ Sum shuffle(int offset) const {
        return {__shfl_down_sync(0xffffffff, value, offset)};
    }
    __device__ double result() const { return value; }
};

template<int Mode, bool Variance, int BinsPerBlock>
__device__ void gather(const unsigned *rows, const unsigned *columns,
                       const float *weights, const double *norm, const float *frames,
                       double *output, unsigned pixels, unsigned bins, unsigned weighted) {
    using Acc = Sum<Mode>;
    using Scalar = typename Acc::Scalar;
    constexpr unsigned ThreadsPerBin = 128 / BinsPerBlock;
    constexpr unsigned Warps = 128 / 32;
    unsigned groups = (bins + BinsPerBlock - 1) / BinsPerBlock;
    unsigned frame = blockIdx.x / groups;
    unsigned thread = threadIdx.x % ThreadsPerBin;
    unsigned bin = (blockIdx.x % groups) * BinsPerBlock + threadIdx.x / ThreadsPerBin;
    if (bin >= bins) return;  // Entire warps exit together.
    const float *image = frames + (unsigned long long)frame * pixels;
    double *result = output + ((unsigned long long)frame * bins + bin) * (Variance ? 2 : 1);
    Acc signal{}, variance{};
    for (unsigned k = rows[bin] + thread; k < rows[bin + 1]; k += ThreadsPerBin) {
        Scalar s = image[columns[k]];
        Scalar w = weighted ? weights[k] : 1;
        signal.add(w * s);
        if constexpr (Variance) variance.add(w * w * (s > 0 ? s : 0));
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        signal.combine(signal.shuffle(offset));
        if constexpr (Variance) variance.combine(variance.shuffle(offset));
    }
    if constexpr (BinsPerBlock == 1) {
        __shared__ Acc signals[Warps];
        __shared__ Acc variances[Warps];
        if (thread % 32 == 0) {
            signals[thread / 32] = signal;
            if constexpr (Variance) variances[thread / 32] = variance;
        }
        __syncthreads();
        if (thread == 0) {
            for (unsigned warp = 1; warp < Warps; ++warp) {
                signal.combine(signals[warp]);
                if constexpr (Variance) variance.combine(variances[warp]);
            }
        }
    }
    if (thread == 0) {
        double denominator = norm[bin];
        result[0] = denominator == 0 ? 0 : signal.result() / denominator;
        if constexpr (Variance) {
            result[1] = denominator == 0 ? 0 : sqrt(variance.result()) / denominator;
        }
    }
}

#define KERNEL(Name, Mode, Variance, Bins) \
extern "C" __global__ void Name(const unsigned *rows, const unsigned *columns, \
                                const float *weights, const double *norm, const float *frames, \
                                double *output, unsigned pixels, unsigned bins, unsigned weighted) { \
    gather<Mode, Variance, Bins>(rows, columns, weights, norm, frames, output, pixels, bins, weighted); \
}

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
