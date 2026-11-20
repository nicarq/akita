#include <metal_stdlib>
using namespace metal;
struct Q {
    int p, pinv, mont, montsq;
};
struct Shape {
    uint width, chunk;
    Q primes[4];
};
inline int reduce(int a, int p) { return a < 0 ? a + p : (a >= p ? a - p : a); }
template <bool small> inline int mont(int a, int b, Q q) {
    if (small) {
        int t = short(uint(a) * uint(b) * uint(q.pinv));
        return ((a * b) >> 16) - ((t * q.p) >> 16);
    }
    int t = int(uint(a) * uint(b) * uint(q.pinv));
    return as_type<int>(uint(mulhi(a, b)) - uint(mulhi(t, q.p)));
}
template <bool small, uint lanes>
inline void products(threadgroup int *x, device const int *z, device const ulong *matrix,
                     device const int *tw, device int *out, constant Shape &s, uint3 id, uint lane) {
    threadgroup int *y = x + 2048;
    const uint D = 2048;
    uint chunk = id.x, k = id.y, side = id.z;
    Q q = s.primes[k];
    device const int *w = tw + k * 2 * D;

    int acc[D / lanes] = {};
    for (uint col = chunk * s.chunk; col < min((chunk + 1) * s.chunk, s.width); ++col) {
        for (uint j = 0; j < D / lanes; j++) {
            uint i = lane + j * lanes;
            int v = z[col * D + i];
            if (small)
                v = v % 12289;
            int scale = side ? q.montsq : w[D + i];
            x[i] = mont<small>(v, scale, q);
            ulong a = matrix[col * D + i];
            bool negative = a > 0x7fffffffffffffe2ul;
            ulong mag = negative ? (0xffffffffffffffc5ul - a) : a;
            int val;
            uint lo = uint(mag), hi = uint(mag >> 32);
            int a0, a1;
            switch (k) {
            case 0:
                a0 = lo % 1073692673u;
                a1 = hi % 1073692673u;
                break;
            case 1:
                a0 = lo % 1073668097u;
                a1 = hi % 1073668097u;
                break;
            case 2:
                a0 = lo % 1073655809u;
                a1 = hi % 1073655809u;
                break;
            case 3:
                a0 = lo % 12289u;
                a1 = hi % 12289u;
                break;
            default:
                a0 = 0;
                a1 = 0;
                break;
            }
            a1 = mont<small>(a1, q.montsq, q);
            if (small)
                a1 = mont<small>(a1, q.montsq, q);
            val = reduce(a0 + a1, q.p);
            y[i] = mont<small>(negative ? -val : val, scale, q);
        }
        for (uint span = D / 2; span; span >>= 1) {
            if (span >= 32)
                threadgroup_barrier(mem_flags::mem_threadgroup);
            else
                simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = lane; i < D / 2; i += lanes) {
                uint j = i & (span - 1), a = 2 * (i - j) + j;
                int u = x[a], v = x[a + span];
                x[a] = reduce(u + v, q.p);
                x[a + span] = mont<small>(u - v, w[span - 1 + j], q);
                u = y[a];
                v = y[a + span];
                y[a] = reduce(u + v, q.p);
                y[a + span] = mont<small>(u - v, w[span - 1 + j], q);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j = 0; j < D / lanes; j++) {
            uint i = lane + j * lanes;
            int key = reduce(y[i], q.p);
            acc[j] = reduce(acc[j] + mont<small>(key, reduce(x[i], q.p), q), q.p);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint j = 0; j < D / lanes; j++) {
        uint i = lane + j * lanes;
        if (small)
            ((device short *)(out + (chunk * 7 + 6) * D))[side * D + i] = short(acc[j]);
        else
            out[(chunk * 7 + side * 3 + k) * D + i] = acc[j];
    }
}

kernel void quotient_products(device const int *z [[buffer(0)]], device const ulong *matrix [[buffer(1)]],
                              device const int *tw [[buffer(2)]], device int *out [[buffer(3)]],
                              constant Shape &s [[buffer(4)]], uint3 g [[threadgroup_position_in_grid]],
                              uint lane [[thread_index_in_threadgroup]]) {
    threadgroup int x[4096];
    if (g.y == 3)
        products<true, 256>(x, z, matrix, tw, out, s, g, lane);
    else
        products<false, 256>(x, z, matrix, tw, out, s, g, lane);
}
