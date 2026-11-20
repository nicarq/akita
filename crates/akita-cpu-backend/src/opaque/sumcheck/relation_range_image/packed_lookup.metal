struct PackedQuadShape {
  uint quadsPerEntry, pairCount, classCount, lowLength, groupCount, skipLinear,
      outputMainElements, witnessMainBytes, materialize;
};
struct NormSums {
  ulong2 v[3];
};
inline ulong2 square(ulong2 x) {
  ulong a = fp_mul(x.x, x.x), b = fp_mul(x.y, x.y), c = fp_mul(x.x, x.y);
  return ulong2(fp_add(a, fp_add(b, b)), fp_add(c, c));
}
inline ulong2 reduce32(ulong2 x) {
  for (uint d = 16; d; d >>= 1)
    x = add(x, as_type<ulong2>(simd_shuffle_down(as_type<uint4>(x), d)));
  return x;
}
inline ushort packed_quad(device const uchar *bytes, device const uchar *tail,
                          uint i, constant PackedQuadShape &shape) {
  uint at = (i / 2) * 3 + (i & 1u);
  uint a = at < shape.witnessMainBytes ? bytes[at]
                                       : tail[at - shape.witnessMainBytes];
  at++;
  uint b = at < shape.witnessMainBytes ? bytes[at]
                                       : tail[at - shape.witnessMainBytes];
  uint raw = a | (b << 8);
  return ushort(((raw >> ((i & 1u) * 4u)) & 4095u) ^ 0x924u);
}
kernel void packed_quad_norm(device const uchar *witness [[buffer(0)]],
                             device const uchar *tail [[buffer(1)]],
                             device const ulong2 *lookups [[buffer(2)]],
                             device const ulong2 *lowWeights [[buffer(3)]],
                             device const ulong2 *highWeights [[buffer(4)]],
                             device ulong2 *out [[buffer(5)]],
                             device ulong2 *outputTail [[buffer(6)]],
                             device NormSums *partials [[buffer(7)]],
                             constant PackedQuadShape &shape [[buffer(8)]],
                             uint group [[threadgroup_position_in_grid]],
                             uint threadIndex
                             [[thread_position_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint warpIndex
                             [[simdgroup_index_in_threadgroup]]) {
  threadgroup NormSums warpSums[4];
  NormSums subtotal = {};
  uint rows = (shape.pairCount + shape.lowLength - 1) / shape.lowLength;
  for (uint row = group; row < rows; row += shape.groupCount) {
    NormSums local = {};
    for (uint k = threadIndex; k < shape.lowLength; k += 128) {
      uint pair = row * shape.lowLength + k;
      if (pair >= shape.pairCount)
        break;
      uint base = 2 * pair * shape.quadsPerEntry;
      ulong2 w0 = 0, w1 = 0;
      for (uint j = 0; j < shape.quadsPerEntry; j++) {
        uint i = base + j, h = i + shape.quadsPerEntry;
        ushort ca = packed_quad(witness, tail, i, shape),
               cb = packed_quad(witness, tail, h, shape);
        w0 = add(w0, lookups[j * shape.classCount + ca]);
        w1 = add(w1, lookups[j * shape.classCount + cb]);
      }
      if (shape.materialize) {
        uint i = 2 * pair;
        if (i < shape.outputMainElements)
          out[i] = w0;
        else
          outputTail[i - shape.outputMainElements] = w0;
        i++;
        if (i < shape.outputMainElements)
          out[i] = w1;
        else
          outputTail[i - shape.outputMainElements] = w1;
      }
      if (any(w0 != 0) || any(w1 != 0)) {
        ulong2 e = lowWeights[k], dw = sub(w1, w0);
        local.v[0] = add(local.v[0], mul(e, add(square(w0), w0)));
        local.v[2] = add(local.v[2], mul(e, square(dw)));
        if (!shape.skipLinear)
          local.v[1] =
              add(local.v[1], mul(e, mul(dw, add(add(w0, w0), ulong2(1, 0)))));
      }
    }
    for (uint j = 0; j < 3; j++) {
      if (j == 1 && shape.skipLinear)
        continue;
      ulong2 sum = reduce32(local.v[j]);
      if (lane == 0)
        subtotal.v[j] = add(subtotal.v[j], mul(highWeights[row], sum));
    }
  }
  if (lane == 0)
    warpSums[warpIndex] = subtotal;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (threadIndex < 3) {
    uint j = threadIndex;
    partials[group].v[j] = add(add(warpSums[0].v[j], warpSums[1].v[j]),
                               add(warpSums[2].v[j], warpSums[3].v[j]));
  }
}
kernel void reduce_packed_quad_norm(device const NormSums *partials
                                    [[buffer(0)]],
                                    device NormSums *out [[buffer(1)]],
                                    constant PackedQuadShape &shape
                                    [[buffer(2)]],
                                    uint j [[simdgroup_index_in_threadgroup]],
                                    uint lane [[thread_index_in_simdgroup]]) {
  ulong2 sum = 0;
  for (uint i = lane; i < shape.groupCount; i += 32)
    sum = add(sum, partials[i].v[j]);
  sum = reduce32(sum);
  if (!lane)
    out[0].v[j] = sum;
}
