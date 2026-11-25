#define computeCommandEncoder missingComputeCommandEncoder
#import "../src/sources/quad_lookup.m"
#undef computeCommandEncoder

static BOOL requestedEncoder;

@implementation NSObject (MissingQuadLookupEncoder)
- (id<MTLComputeCommandEncoder>)missingComputeCommandEncoder {
  requestedEncoder = YES;
  return nil;
}
@end

int main(void) {
  @autoreleasepool {
    if (!MTLCreateSystemDefaultDevice())
      return 77;
    NSData *shader =
        [[NSFileHandle fileHandleWithStandardInput] readDataToEndOfFile];
    uint32_t pairs = (uint32_t)getpagesize();
    size_t bytes = 6 * (size_t)pairs;
    void *witness = NULL;
    if (posix_memalign(&witness, getpagesize(), bytes) != 0)
      return 1;
    memset(witness, 0, bytes);
    void *handle = pgl_akita_quad_lookup_open(witness, bytes, shader.bytes);
    if (!handle)
      return 1;

    uint64_t table[2 * 4096][2] = {}, low[1][2] = {{1, 0}}, total[6] = {42};
    NSMutableData *high = [NSMutableData dataWithLength:pairs * sizeof(low)];
    int success = pgl_akita_quad_lookup_round(handle, 2, pairs, 4096, table, low, 1,
                                          high.bytes, pairs, 0, NULL, total);
    pgl_akita_quad_lookup_close(handle);
    free(witness);
    if (!requestedEncoder || success || total[0] != 42) {
      fprintf(stderr, "requested encoder=%d, success=%d, first sum=%llu\n",
              requestedEncoder, success, (unsigned long long)total[0]);
      return 1;
    }
    return 0;
  }
}
