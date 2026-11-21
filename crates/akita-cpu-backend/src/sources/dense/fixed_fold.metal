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

inline int4 unpack_spectrum(device const uint *row, uint quartet) {
    uint offset = 4 * quartet - uint(quartet == 3);
    uint4 current = uint4(*(device const packed_uint4 *)(row + offset));
    uint4 previous = simd_shuffle_up(current, 1);
    if (quartet == 3) {
        current = uint4(current.y, (current.y >> 24) | (current.z << 8),
                        (current.z >> 16) | (current.w << 16), current.w >> 8);
    }
    uint shift = 2 * quartet;
    uint4 bits = current << shift;
    if (quartet != 0)
        bits |= previous >> ((32 - shift) & 31);
    return as_type<int4>(bits << 2) >> 2;
}

kernel void fold_products(device const uint *keys [[buffer(0)]], device const int *weights [[buffer(1)]],
                          device int *spectra [[buffer(2)]], constant FoldShape &shape [[buffer(3)]],
                          uint index [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint group = index >> 5;
    uint frequency = group & 127;
    uint quartet = lane & 3;
    ulong row = ulong(group) * shape.blocks;
    int4 sum = 0;
    for (uint block = shape.begin + (lane >> 2); block < shape.end; block += 8) {
        int4 key = unpack_spectrum(keys + (row + block) * 15, quartet);
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

struct LookupShape {
    uint blocks, columns, tile_columns, table_len, column_width, code_count, live_cells;
    akita::NttPrime primes[2];
};
inline int lookup_digit(device const ushort *codes, device const short *digits, uint2 range, uint column, uint lane,
                        constant LookupShape &shape) {
    uint ring = column / 5;
    if (ring >= range.y)
        return 0;
    uint cell = (range.x + ring) * 512 + lane;
    uint offset = cell % shape.column_width;
    if (cell >= shape.live_cells || offset >= shape.code_count)
        return 0;
    uint code = codes[offset];
    if (code >= shape.table_len)
        return 0;
    return int(digits[((cell / shape.column_width) * shape.table_len + code) * 5 + column % 5]);
}
kernel void lookup_commit_products(device const ushort *codes [[buffer(0)]], device const short *digits [[buffer(1)]],
                                   device const int *matrix [[buffer(2)]], device const int *twiddles [[buffer(3)]],
                                   device int *partials [[buffer(4)]], constant LookupShape &shape [[buffer(5)]],
                                   device const uint2 *ranges [[buffer(6)]],
                                   uint3 group [[threadgroup_position_in_grid]],
                                   uint lane [[thread_index_in_threadgroup]]) {
    constexpr uint D = 512, LIMB_ROWS = 9, BLOCKS_PER_GROUP = 2;
    uint prime_index = group.z;
    akita::NttPrime prime = shape.primes[prime_index];
    device const int *tables = twiddles + prime_index * 2 * D;
    int lower_twist = tables[D + lane];
    int upper_twist = tables[D + lane + 256];
    threadgroup int values[BLOCKS_PER_GROUP][D];
    int2 accumulators[BLOCKS_PER_GROUP][LIMB_ROWS] = {};
    for (uint column = group.x * shape.tile_columns; column < min(shape.columns, (group.x + 1) * shape.tile_columns);
         ++column) {
        for (uint block_offset = 0; block_offset < BLOCKS_PER_GROUP; block_offset++) {
            uint block = group.y * BLOCKS_PER_GROUP + block_offset;
            values[block_offset][lane] = akita::mont_mul(
                prime, lookup_digit(codes, digits, ranges[min(block, shape.blocks - 1)], column, lane, shape),
                lower_twist);
            values[block_offset][lane + 256] = akita::mont_mul(
                prime, lookup_digit(codes, digits, ranges[min(block, shape.blocks - 1)], column, lane + 256, shape),
                upper_twist);
        }
        for (uint span = 256; span; span >>= 1) {
            akita::stage_barrier<D>(span, span == 256 ? span : span * 2);
            uint stage_lane = lane & (span - 1);
            uint low = ((lane - stage_lane) << 1) + stage_lane;
            int twiddle = tables[span - 1 + stage_lane];
            for (uint block_offset = 0; block_offset < BLOCKS_PER_GROUP; block_offset++) {
                int lower = values[block_offset][low];
                int upper = values[block_offset][low + span];
                values[block_offset][low] = akita::reduce_range(prime, lower + upper);
                values[block_offset][low + span] = akita::mont_mul(prime, lower - upper, twiddle);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint row = 0; row < LIMB_ROWS; row++) {
            device const int *matrix_row = matrix + ((row * shape.columns + column) * 2 + prime_index) * D;
            int lower = matrix_row[lane];
            int upper = matrix_row[lane + 256];
            for (uint block_offset = 0; block_offset < BLOCKS_PER_GROUP; block_offset++) {
                accumulators[block_offset][row].x =
                    akita::reduce_range(prime, accumulators[block_offset][row].x +
                                                   akita::mont_mul(prime, lower, values[block_offset][lane]));
                accumulators[block_offset][row].y =
                    akita::reduce_range(prime, accumulators[block_offset][row].y +
                                                   akita::mont_mul(prime, upper, values[block_offset][lane + 256]));
            }
        }
    }
    for (uint block_offset = 0; block_offset < BLOCKS_PER_GROUP; block_offset++) {
        uint block = group.y * BLOCKS_PER_GROUP + block_offset;
        if (block >= shape.blocks)
            continue;
        for (uint row = 0; row < LIMB_ROWS; row++) {
            device int *destination =
                partials + (((group.x * shape.blocks + block) * LIMB_ROWS + row) * 2 + prime_index) * D;
            destination[lane] = accumulators[block_offset][row].x;
            destination[lane + 256] = accumulators[block_offset][row].y;
        }
    }
}
kernel void lookup_commit_sums(device const int *partials [[buffer(0)]], device int *sums [[buffer(1)]],
                               constant LookupShape &shape [[buffer(2)]], uint index [[thread_position_in_grid]]) {
    constexpr uint D = 512, LIMB_ROWS = 9;
    uint stride = shape.blocks * LIMB_ROWS * 2 * D;
    int sum = 0;
    akita::NttPrime prime = shape.primes[(index / D) % 2];
    for (uint tile_columns = 0; tile_columns < (shape.columns + shape.tile_columns - 1) / shape.tile_columns;
         tile_columns++)
        sum = akita::reduce_range(prime, sum + partials[tile_columns * stride + index]);
    sums[index] = sum;
}

inline int4 raw_mul(akita::NttPrime q, int4 a, int b) {
    return int4(akita::mont_mul(q,a.x,b), akita::mont_mul(q,a.y,b), akita::mont_mul(q,a.z,b), akita::mont_mul(q,a.w,b));
}

inline int4 raw_reduce(akita::NttPrime q, int4 a) {
    return int4(akita::reduce_range(q,a.x), akita::reduce_range(q,a.y), akita::reduce_range(q,a.z), akita::reduce_range(q,a.w));
}

inline int4 raw_digits(ulong a) {
    ulong bias = a + 0x8000800080008000ul + ulong(a > 0x7fff7fff7fff7ffful) * 59ul;
    return int4(uint(bias) & 65535u, uint(bias >> 16) & 65535u,
                uint(bias >> 32) & 65535u, uint(bias >> 48)) - 32768;
}

kernel void raw_ntt(device const ulong *key [[buffer(0)]], device const int *weights [[buffer(1)]],
                    device const int *tables [[buffer(2)]], device int *out [[buffer(3)]],
                    constant FoldShape &shape [[buffer(4)]], constant uint &last_positions [[buffer(5)]],
                    uint i [[thread_position_in_grid]],
                    uint lane [[thread_index_in_simdgroup]]) {
    const uint positions = shape.rows / 4, residue = (i >> 5) & 15;
    const uint pos = i >> 9;
    const uint end = shape.end - uint(shape.end == shape.blocks && pos >= last_positions);
    int4 sum[4] = {};
    int twist[4];
    for (uint j=0; j<4; ++j)
        twist[j]=akita::mont_mul(shape.prime,tables[256+lane+32*j],shape.prime.montsq);
    for (uint b = shape.begin; b < end; ++b) {
        int4 v[4];
        for (uint j = 0; j < 4; ++j) {
            uint ix = lane + 32*j;
            ulong a = key[(ulong(b) * positions + pos) * 2048 + ix * 16 + residue];
            v[j] = raw_mul(shape.prime, raw_digits(a), twist[j]);
        }
        for (uint j = 0; j < 2; ++j) {
            int4 x=v[j], y=v[j+2];
            v[j]=raw_reduce(shape.prime,x+y);
            v[j+2]=raw_mul(shape.prime,x-y,tables[63+lane+32*j]);
        }
        for (uint j = 0; j < 4; j += 2) {
            int4 x=v[j], y=v[j+1];
            v[j]=raw_reduce(shape.prime,x+y);
            v[j+1]=raw_mul(shape.prime,x-y,tables[31+lane]);
        }
        for (uint span = 16; span; span >>= 1) {
            int tw=tables[span-1+(lane&(span-1))];
            for (uint j=0; j<4; ++j) {
                int4 other=simd_shuffle_xor(v[j],span);
                v[j]=(lane & span) ? raw_mul(shape.prime,other-v[j],tw) : raw_reduce(shape.prime,v[j]+other);
            }
        }
        for (uint j=0; j<4; ++j)
            sum[j]=raw_reduce(shape.prime,sum[j]+raw_mul(shape.prime,v[j],weights[(lane+32*j)*shape.blocks+b]));
    }

    for (uint span=1; span<=16; span+=span) {
        int tw=tables[128+span-1+(lane&(span-1))];
        for (uint j=0; j<4; ++j) {
            int4 mine=(lane&span)?raw_mul(shape.prime,sum[j],tw):sum[j];
            int4 other=simd_shuffle_xor(mine,span);
            sum[j]=raw_reduce(shape.prime,(lane&span)?(other-mine):(mine+other));
        }
    }
    for (uint j=0; j<4; j+=2) {
        int4 x=sum[j], y=raw_mul(shape.prime,sum[j+1],tables[128+31+lane]);
        sum[j]=raw_reduce(shape.prime,x+y);
        sum[j+1]=raw_reduce(shape.prime,x-y);
    }
    for (uint j=0; j<2; ++j) {
        int4 x=sum[j], y=raw_mul(shape.prime,sum[j+2],tables[128+63+lane+32*j]);
        sum[j]=raw_reduce(shape.prime,x+y);
        sum[j+2]=raw_reduce(shape.prime,x-y);
    }
    for (uint j=0; j<4; ++j) {
        int4 vals=raw_mul(shape.prime,sum[j],tables[384+lane+32*j]);
        for (uint d=0; d<4; ++d) {
            out[(pos*4+d)*2048 + (lane+32*j)*16 + residue] = akita::center(shape.prime,akita::to_canonical(shape.prime,vals[d]));
        }
    }
}
