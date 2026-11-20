#include <metal_stdlib>
using namespace metal;

inline ulong fp_add(ulong a, ulong b) {
    ulong s = a + b;
    return min(s + (s < a ? 59ul : 0ul), (s < a ? s + 59ul : s) + 59ul);
}
inline ulong fp_sub(ulong a, ulong b) { return (a - b) - (a < b ? 59ul : 0ul); }
inline ulong fp_mul(ulong a, ulong b) {
    ulong lo = a * b, hi = mulhi(a, b);
    ulong t = lo + 59ul * hi;
    ulong h = mulhi(hi, 59ul) + (t < lo);
    ulong f = t + 59ul * h;
    f += f < t ? 59ul : 0ul;
    return min(f, f + 59ul);
}
inline ulong2 add(ulong2 a, ulong2 b) { return ulong2(fp_add(a.x, b.x), fp_add(a.y, b.y)); }
inline ulong2 sub(ulong2 a, ulong2 b) { return ulong2(fp_sub(a.x, b.x), fp_sub(a.y, b.y)); }
inline ulong2 mul(ulong2 a, ulong2 b) {
    ulong ac = fp_mul(a.x, b.x), bd = fp_mul(a.y, b.y);
    return ulong2(fp_add(ac, fp_add(bd, bd)), fp_add(fp_mul(a.x, b.y), fp_mul(a.y, b.x)));
}
inline ulong2 fold(ulong2 a, ulong2 b, ulong2 r) { return add(a, mul(sub(b, a), r)); }
struct Shape {
    uint len, round, lowlen, stride, digits, mode;
    ulong2 challenge;
};
struct Terms {
    ulong2 v[4];
};
inline uint packed_byte(device const uchar *bytes, device const uchar *tail, uint i, uint prefix, uint count) {
    if (i >= count)
        return 0;
    return i < prefix ? uint(bytes[i]) : uint(tail[i - prefix]);
}
inline ulong2 octet(device const uchar *bytes, device const uchar *tail, uint prefix, device const ulong2 *lut,
                    uint i, uint n) {
    uint base = i * 3;
    uint count = (3 * n + 7) / 8;
    uint raw = packed_byte(bytes, tail, base, prefix, count) |
               (packed_byte(bytes, tail, base + 1, prefix, count) << 8) |
               (packed_byte(bytes, tail, base + 2, prefix, count) << 16);
    uint x = raw ^ (((raw >> 2) & 0x249249u) * 3u);
    uint code = (x & 3u) | ((x >> 1) & 0xcu) | ((x >> 2) & 0x30u) | ((x >> 3) & 0xc0u) | ((x >> 4) & 0x300u) |
                ((x >> 5) & 0xc00u) | ((x >> 6) & 0x3000u) | ((x >> 7) & 0xc000u);
    return lut[code];
}
inline ulong2 after_bind(device const uchar *bytes, device const uchar *tail, device const ulong2 *lut,
                         device ulong2 *table, constant Shape &s, uint i) {
    if (i >= s.len)
        return ulong2(0);
    if (s.round == 3)
        return octet(bytes, tail, s.mode, lut, i, s.digits);
    if (s.round == 4)
        return fold(octet(bytes, tail, s.mode, lut, 2 * i, s.digits),
                    octet(bytes, tail, s.mode, lut, 2 * i + 1, s.digits), s.challenge);
    uint src = 2 * i * s.stride;
    uint prev = (s.digits + (1u << (s.round - 1)) - 1) >> (s.round - 1);
    return fold(table[src], 2 * i + 1 < prev ? table[src + s.stride] : ulong2(0), s.challenge);
}
inline void reduce_terms(threadgroup Terms *t, uint lane, uint width) {
    for (uint d = width / 2; d; d /= 2) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < d)
            for (uint j = 0; j < 4; j++)
                t[lane].v[j] = add(t[lane].v[j], t[lane + d].v[j]);
    }
}
kernel void quartic(device const uchar *bytes [[buffer(0)]], device const ulong2 *lut [[buffer(1)]],
                    device ulong2 *table [[buffer(2)]], device const ulong2 *low [[buffer(3)]],
                    device const ulong2 *high [[buffer(4)]], device Terms *out [[buffer(5)]],
                    constant Shape &s [[buffer(6)]], device const uchar *tail [[buffer(7)]],
                    uint group [[threadgroup_position_in_grid]], uint lane [[thread_position_in_threadgroup]],
                    uint width [[threads_per_threadgroup]]) {
    uint index = group * width + lane;
    Terms t = {};
    if (index < (s.len + 1) / 2) {
        ulong2 a = after_bind(bytes, tail, lut, table, s, 2 * index),
               b = after_bind(bytes, tail, lut, table, s, 2 * index + 1);
        if (s.round >= 4) {
            uint stride = s.round == 4 ? 1 : 2 * s.stride;
            table[2 * index * stride] = a;
            if (2 * index + 1 < s.len)
                table[(2 * index + 1) * stride] = b;
        }
        if (s.len > 1 && any(a != b)) {
            ulong2 w = mul(low[index % s.lowlen], high[index / s.lowlen]);
            ulong2 delta = sub(b, a), delta2 = mul(delta, delta);
            ulong2 y = sub(a, ulong2(5, 0)), y2 = mul(y, y);
            ulong2 wd = mul(w, delta), wd2 = mul(w, delta2);
            t.v[0] = mul(wd, sub(mul(sub(y2, ulong2(21, 0)), y), ulong2(16, 0)));
            t.v[1] = mul(wd2, sub(y2, ulong2(7, 0)));
            t.v[2] = mul(wd2, mul(delta, y));
            t.v[3] = mul(wd2, delta2);
        }
    }
    threadgroup Terms shared[128];
    shared[lane] = t;
    reduce_terms(shared, lane, width);
    if (!lane)
        out[group] = shared[0];
}
kernel void reduce_quartic(device const Terms *v [[buffer(0)]], device Terms *out [[buffer(1)]],
                           constant uint &count [[buffer(2)]], uint group [[threadgroup_position_in_grid]],
                           uint lane [[thread_position_in_threadgroup]], uint width [[threads_per_threadgroup]]) {
    uint i = group * width + lane;
    threadgroup Terms shared[128];
    Terms t = {};
    if (i < count)
        t = v[i];
    shared[lane] = t;
    reduce_terms(shared, lane, width);
    if (!lane)
        out[group] = shared[0];
}
