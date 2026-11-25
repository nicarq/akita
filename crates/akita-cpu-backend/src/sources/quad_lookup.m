#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <unistd.h>
typedef struct {
  uint32_t quadsPerEntry, pairCount, classCount, lowLength, groupCount,
      skipLinear, outputMainElements, witnessMainBytes, materialize;
} PackedQuadShape;

@interface PGLAKPackedQuadSource : NSObject {
@public
  id<MTLBuffer> main;
  id<MTLBuffer> tail;
  size_t quadPairs;
}
@end
@implementation PGLAKPackedQuadSource
@end

static id<MTLDevice> device;
static id<MTLCommandQueue> queue;
static id<MTLComputePipelineState> lookupPipeline, reducePipeline;
int pgl_akita_quad_lookup_buffers_fit(size_t limit, size_t body, size_t tail,
                                  size_t largestCopy) {
  return body <= limit && MAX(tail, 16) <= limit && largestCopy <= limit;
}
static id<MTLBuffer> alias(void *pointer, size_t length) {
  return [device newBufferWithBytesNoCopy:pointer
                                   length:length
                                  options:MTLResourceStorageModeShared
                              deallocator:nil];
}
void *pgl_akita_quad_lookup_open(const uint8_t *bytes, size_t length,
                             const char *source) {
  @autoreleasepool {
    static dispatch_once_t once;
    dispatch_once(&once, ^{
      device = MTLCreateSystemDefaultDevice();
      queue = [device newCommandQueue];
      NSError *err = nil;
      id<MTLLibrary> library =
          [device newLibraryWithSource:[NSString stringWithUTF8String:source]
                               options:nil
                                 error:&err];
      if (!library)
        return;
      lookupPipeline =
          [device newComputePipelineStateWithFunction:
                      [library newFunctionWithName:@"packed_quad_norm"]
                                                error:&err];
      reducePipeline =
          [device newComputePipelineStateWithFunction:
                      [library newFunctionWithName:@"reduce_packed_quad_norm"]
                                                error:&err];
    });
    size_t page = getpagesize(), body = length / page * page,
           tail = length - body;
    if (!lookupPipeline || !reducePipeline ||
        lookupPipeline.threadExecutionWidth != 32 ||
        reducePipeline.threadExecutionWidth != 32 ||
        lookupPipeline.maxTotalThreadsPerThreadgroup < 128 ||
        reducePipeline.maxTotalThreadsPerThreadgroup < 96 || !body ||
        length > UINT32_MAX || length % 3 ||
        !pgl_akita_quad_lookup_buffers_fit(device.maxBufferLength, body, tail, 0) ||
        ((uintptr_t)bytes % page))
      return NULL;
    id<MTLBuffer> mainBuffer = alias((void *)bytes, body),
                  tailBuffer = [device newBufferWithLength:MAX(tail, 16)
                                                   options:0];
    if (!mainBuffer || !tailBuffer)
      return NULL;
    memcpy(tailBuffer.contents, (char *)bytes + body, tail);
    PGLAKPackedQuadSource *owner = [PGLAKPackedQuadSource new];
    owner->main = mainBuffer;
    owner->tail = tailBuffer;
    owner->quadPairs = length / 3;
    return (__bridge_retained void *)owner;
  }
}
void pgl_akita_quad_lookup_close(void *h) {
  @autoreleasepool {
    (void)CFBridgingRelease(h);
  }
}
int pgl_akita_quad_lookup_round(void *h, uint32_t quadsPerEntry, uint32_t pairs,
                            uint32_t classCount, const void *lookupValues,
                            const void *lowWeights, uint32_t lowLength,
                            const void *highWeights, uint32_t highLength,
                            int skipLinear, void *target, void *total) {
  @autoreleasepool {
    PGLAKPackedQuadSource *owner = (__bridge PGLAKPackedQuadSource *)h;
    id<MTLBuffer> mainBuffer = owner->main;
    size_t page = getpagesize(), length = target ? (size_t)pairs * 32 : 0,
           body = length / page * page, tail = length - body;
    if (!pairs || (uint64_t)pairs * quadsPerEntry != owner->quadPairs ||
        !lowLength || ((uintptr_t)target % page) ||
        highLength < (pairs + lowLength - 1) / lowLength)
      return 0;
    PackedQuadShape shape = {quadsPerEntry,
                             pairs,
                             classCount,
                             lowLength,
                             MIN((pairs + lowLength - 1) / lowLength, 2048),
                             skipLinear,
                             (uint32_t)(body / 16),
                             (uint32_t)mainBuffer.length,
                             target != NULL};
    size_t lookupBytes = (size_t)quadsPerEntry * classCount * 16,
           lowBytes = (size_t)lowLength * 16,
           highBytes = (size_t)highLength * 16,
           partialBytes = (size_t)shape.groupCount * 48;
    size_t largestCopy =
        MAX(MAX(lookupBytes, lowBytes), MAX(highBytes, partialBytes));
    if (!pgl_akita_quad_lookup_buffers_fit(device.maxBufferLength, body, tail,
                                       largestCopy))
      return 0;
    id<MTLBuffer> lookupTable = [device newBufferWithBytes:lookupValues
                                                    length:lookupBytes
                                                   options:0],
                  lowBuffer = [device newBufferWithBytes:lowWeights
                                                  length:lowBytes
                                                 options:0],
                  highBuffer = [device newBufferWithBytes:highWeights
                                                   length:highBytes
                                                  options:0],
                  edge = [device newBufferWithLength:MAX(tail, 16) options:0],
                  partial = [device
                      newBufferWithLength:partialBytes
                                  options:MTLResourceStorageModePrivate],
                  result = [device newBufferWithLength:48 options:0];
    id<MTLBuffer> out = body ? alias(target, body) : edge;
    if (!lookupTable || !lowBuffer || !highBuffer || !out || !edge ||
        !partial || !result)
      return 0;
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (!encoder)
      return 0;
    [encoder setComputePipelineState:lookupPipeline];
    NSArray *all = @[
      mainBuffer, owner->tail, lookupTable, lowBuffer, highBuffer, out, edge,
      partial
    ];
    for (NSUInteger j = 0; j < all.count; j++)
      [encoder setBuffer:all[j] offset:0 atIndex:j];
    [encoder setBytes:&shape length:sizeof(shape) atIndex:8];
    [encoder dispatchThreadgroups:MTLSizeMake(shape.groupCount, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [encoder setComputePipelineState:reducePipeline];
    [encoder setBuffer:partial offset:0 atIndex:0];
    [encoder setBuffer:result offset:0 atIndex:1];
    [encoder setBytes:&shape length:sizeof(shape) atIndex:2];
    [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(96, 1, 1)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted)
      return 0;
    memcpy(total, result.contents, 48);
    if (tail)
      memcpy((char *)target + body, edge.contents, tail);
    return 1;
  }
}
