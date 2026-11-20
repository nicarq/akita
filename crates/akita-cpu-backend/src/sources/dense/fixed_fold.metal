#include <metal_stdlib>
// NTT arithmetic from LayerZero-Labs Akita c355c74d, mont.h and ntt.h.
using namespace metal;

namespace akita {
struct NttPrime {
    int p;
    int pinv;
    int mont;
    int montsq;
};

inline int mont_mul(NttPrime q, int a, int b) {
    int t = int(uint(a) * uint(b) * uint(q.pinv));
    return int(uint(metal::mulhi(a, b)) - uint(metal::mulhi(t, q.p)));
}

inline int csubp(NttPrime q, int a) { return metal::select(a, int(uint(a) - uint(q.p)), a >= q.p); }

inline int caddp(NttPrime q, int a) { return a + ((a >> 31) & q.p); }

inline int reduce_range(NttPrime q, int a) { return caddp(q, csubp(q, a)); }

inline int normalize(NttPrime q, int a) { return csubp(q, caddp(q, a)); }

inline int from_canonical(NttPrime q, int a) { return mont_mul(q, a, q.montsq); }

inline int to_canonical(NttPrime q, int a) { return normalize(q, mont_mul(q, a, 1)); }

inline int center(NttPrime q, int a) {
    int half_p = q.p / 2;
    return a - (((half_p - a) >> 31) & q.p);
}

template <uint D> inline void stage_barrier(uint len, uint previous) {
    if (D / 2 > 32 && (len > 32 || previous > 32)) {
        metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    } else {
        metal::simdgroup_barrier(metal::mem_flags::mem_threadgroup);
    }
}

template <uint D>
inline void inverse_negacyclic(threadgroup int *a, NttPrime q, device const int *tables, uint lane) {
    static_assert(D >= 2 && (D & (D - 1)) == 0, "D must be a power of two");
    device const int *inv = tables + D;
    device const int *scale = tables + 3 * D;
    constexpr uint HALF = D / 2;

    for (uint len = 1; len < D; len <<= 1) {
        stage_barrier<D>(len, len == 1 ? D / 2 : len / 2);
        uint j = lane & (len - 1);
        uint i0 = ((lane - j) << 1) + j;
        uint i1 = i0 + len;
        int u = a[i0];
        int v = mont_mul(q, a[i1], inv[len - 1 + j]);
        a[i0] = reduce_range(q, u + v);
        a[i1] = reduce_range(q, u - v);
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    a[lane] = mont_mul(q, a[lane], scale[lane]);
    a[lane + HALF] = mont_mul(q, a[lane + HALF], scale[lane + HALF]);
}

} // namespace akita

struct FoldShape {
    uint blocks;
    uint begin;
    uint end;
    uint rows;
    akita::NttPrime prime;
};

kernel void fold_products(device const int *keys [[buffer(0)]], device const int *weights [[buffer(1)]],
                          device int *spectra [[buffer(2)]], constant FoldShape &shape [[buffer(3)]],
                          uint index [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint group = index >> 5;
    uint frequency = group & 127;
    uint quartet = lane & 3;
    ulong row = ulong(group) * shape.blocks;
    int4 sum = 0;
    for (uint block = shape.begin + (lane >> 2); block < shape.end; block += 8) {
        int4 key = ((device const int4 *)keys)[(row + block) * 4 + quartet];
        int weight = weights[frequency * shape.blocks + block];
        for (uint i = 0; i < 4; i++) {
            sum[i] = akita::reduce_range(shape.prime, sum[i] + akita::mont_mul(shape.prime, key[i], weight));
        }
    }
    for (uint offset = 4; offset < 32; offset *= 2) {
        int4 other = simd_shuffle_xor(sum, offset);
        for (uint i = 0; i < 4; i++)
            sum[i] = akita::reduce_range(shape.prime, sum[i] + other[i]);
    }
    if (lane < 4)
        ((device int4 *)spectra)[group * 4 + quartet] = sum;
}

kernel void fold_inverse(device const int *spectra [[buffer(0)]], device int *out [[buffer(1)]],
                         device const int *twiddles [[buffer(2)]], constant FoldShape &shape [[buffer(3)]],
                         uint group [[threadgroup_position_in_grid]],
                         uint lane [[thread_position_in_threadgroup]]) {
    threadgroup int a[128];
    uint dst = (group >> 4) * 2048 + lane * 16 + (group & 15);
    a[lane] = spectra[dst];
    a[lane + 64] = spectra[dst + 64 * 16];
    akita::inverse_negacyclic<128>(a, shape.prime, twiddles, lane);
    out[dst] = akita::center(shape.prime, akita::to_canonical(shape.prime, a[lane]));
    out[dst + 64 * 16] = akita::center(shape.prime, akita::to_canonical(shape.prime, a[lane + 64]));
}

struct PackedCommitShape {
    uint blocks, cols, digits, prefix_bytes;
    akita::NttPrime primes[3];
};

inline uint packed_byte(device const uchar *prefix, device const uchar *tail, uint i, uint n) {
    return i < n ? prefix[i] : tail[i - n];
}

inline int packed_3(device const uchar *prefix, device const uchar *tail, uint i,
                    constant PackedCommitShape &shape) {
    if (i >= shape.digits)
        return 0;
    uint pos = i * 3;
    uint shift = pos & 7;
    uint byte = pos >> 3;
    uint raw = packed_byte(prefix, tail, byte, shape.prefix_bytes);
    if (shift > 5)
        raw |= packed_byte(prefix, tail, byte + 1, shape.prefix_bytes) << 8;
    return int(((raw >> shift) & 7) ^ 4) - 4;
}

kernel void packed_commit_products(device const uchar *prefix [[buffer(0)]],
                                   device const uchar *tail [[buffer(1)]], device const int *matrix [[buffer(2)]],
                                   device const int *tables [[buffer(3)]], device int *partials [[buffer(4)]],
                                   constant PackedCommitShape &shape [[buffer(5)]],
                                   uint3 group [[threadgroup_position_in_grid]],
                                   uint lane [[thread_index_in_threadgroup]]) {
    constexpr uint D = 512, tile = 4;
    uint prime = group.z;
    akita::NttPrime q = shape.primes[prime];
    device const int *tw = tables + prime * 2 * D;
    int psi0 = tw[D + lane], psi1 = tw[D + lane + 256];
    threadgroup int x[tile][D];
    int2 acc[tile][2] = {};
    for (uint col = group.x * 256; col < min(shape.cols, (group.x + 1) * 256); ++col) {
        for (uint t = 0; t < tile; ++t) {
            uint index = ((group.y * tile + t) * shape.cols + col) * D + lane;
            x[t][lane] = akita::mont_mul(q, packed_3(prefix, tail, index, shape), psi0);
            x[t][lane + 256] = akita::mont_mul(q, packed_3(prefix, tail, index + 256, shape), psi1);
        }
        for (uint span = 256; span; span >>= 1) {
            akita::stage_barrier<D>(span, span == 256 ? span : span * 2);
            uint j = lane & (span - 1);
            uint low = ((lane - j) << 1) + j;
            int w = tw[span - 1 + j];
            for (uint t = 0; t < tile; ++t) {
                int a = x[t][low], b = x[t][low + span];
                x[t][low] = akita::reduce_range(q, a + b);
                x[t][low + span] = akita::mont_mul(q, a - b, w);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint r = 0; r < 2; ++r) {
            device const int *row = matrix + ((r * shape.cols + col) * 3 + prime) * D;
            int a0 = row[lane], a1 = row[lane + 256];
            for (uint t = 0; t < tile; ++t) {
                acc[t][r].x = akita::reduce_range(q, acc[t][r].x + akita::mont_mul(q, a0, x[t][lane]));
                acc[t][r].y = akita::reduce_range(q, acc[t][r].y + akita::mont_mul(q, a1, x[t][lane + 256]));
            }
        }
    }
    for (uint t = 0; t < tile; ++t) {
        uint block = group.y * tile + t;
        if (block >= shape.blocks)
            continue;
        for (uint r = 0; r < 2; ++r) {
            device int *row = partials + (((group.x * shape.blocks + block) * 2 + r) * 3 + prime) * D;
            row[lane] = acc[t][r].x;
            row[lane + 256] = acc[t][r].y;
        }
    }
}

kernel void packed_commit_sums(device const int *partials [[buffer(0)]], device int *sums [[buffer(1)]],
                               constant PackedCommitShape &shape [[buffer(2)]],
                               uint i [[thread_position_in_grid]]) {
    uint prime = (i / 512) % 3;
    uint stride = shape.blocks * 2 * 3 * 512;
    int sum = 0;
    for (uint chunk = 0; chunk < (shape.cols + 255) / 256; ++chunk) {
        sum = akita::reduce_range(shape.primes[prime], sum + partials[chunk * stride + i]);
    }
    sums[i] = sum;
}
