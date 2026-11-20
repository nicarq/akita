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
