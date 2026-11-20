#include <metal_stdlib>
using namespace metal;
struct Wide {
    ulong a, b, c, d;
};
inline void accumulate(thread Wide &s, ulong v, ulong p) {
    uint lo = uint(v), hi = uint(v >> 32), pl = uint(p), ph = uint(p >> 32);
    ulong m = ulong(lo) * pl, n = ulong(hi) * pl, o = ulong(lo) * ph, t = ulong(hi) * ph;
    s.a += uint(m);
    s.b += (m >> 32) + uint(n) + uint(o);
    s.c += (n >> 32) + (o >> 32) + uint(t);
    s.d += t >> 32;
}
inline ulong finish(Wide s) {
    s.b += s.a >> 32;
    s.c += s.b >> 32;
    s.d += s.c >> 32;
    ulong b = ulong(uint(s.b)) + 59ul * uint(s.d);
    ulong a = ulong(uint(s.a)) + 59ul * uint(s.c);
    b += a >> 32;
    a = ulong(uint(a)) + 59ul * ((s.d >> 32) * 59ul + (b >> 32));
    ulong lo = ulong(uint(b)) << 32;
    ulong r = lo + a;
    return (r < lo || r >= 0xffffffffffffffc5ul) ? r + 59 : r;
}
inline ulong add(ulong a, ulong b) {
    ulong s = a + b;
    s += s < a ? 59ul : 0ul;
    return min(s, s + 59ul);
}
inline ulong2 sum(ulong2 a, ulong2 b) { return ulong2(add(a.x, b.x), add(a.y, b.y)); }
struct Shape {
    uint tile, positions, blocks;
};
kernel void pack(const device ulong *source [[buffer(0)]], const device ulong2 *weight [[buffer(1)]],
                 device ulong2 *partials [[buffer(2)]], constant Shape &s [[buffer(3)]],
                 uint3 group [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
    uint lane = lid.x;
    uint coordinate = group.x * 8 + lane / 16, residue = lane % 16;
    uint base = (group.z * s.positions + group.y * s.tile) * 2048 + coordinate * 16 + residue;
    uint wi = group.y * s.tile * 16 + residue;
    Wide a = {0, 0, 0, 0}, b = {0, 0, 0, 0};
    for (uint p = 0; p < s.tile; p++) {
        ulong x = source[base + p * 2048];
        ulong2 w = weight[wi + p * 16];
        accumulate(a, x, w.x);
        accumulate(b, x, w.y);
    }
    ulong2 v(finish(a), finish(b));
    for (ushort step = 8; step; step /= 2)
        v = sum(v, as_type<ulong2>(simd_shuffle_down(as_type<uint4>(v), step)));
    if (!residue)
        partials[(group.z * (s.positions / s.tile) + group.y) * 128 + coordinate] = v;
}
kernel void combine(const device ulong2 *partials [[buffer(0)]], device ulong *output [[buffer(1)]],
                    constant Shape &s [[buffer(2)]], uint2 i [[thread_position_in_grid]]) {
    uint tiles = s.positions / s.tile;
    ulong2 r(0);
    for (uint tile = 0; tile < tiles; tile++)
        r = sum(r, partials[(i.y * tiles + tile) * 128 + i.x]);
    output[256 * i.y + i.x] = r.x;
    output[256 * i.y + 128 + i.x] = r.y;
}

kernel void weighted_rows(device const ulong *source [[buffer(0)]], constant ulong2 *weights [[buffer(1)]],
                          device ulong2 *out [[buffer(2)]], constant uint2 &shape [[buffer(3)]],
                          uint column [[thread_position_in_grid]]) {
    Wide a = {}, b = {};
    for (uint row = 0; row < shape.y; ++row) {
        ulong x = source[row * shape.x + column];
        accumulate(a, x, weights[row].x);
        accumulate(b, x, weights[row].y);
    }
    out[column] = ulong2(finish(a), finish(b));
}
