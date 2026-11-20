#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdint.h>
#import <unistd.h>

typedef struct {
    uint32_t width, chunk;
    int32_t primes[4][4];
} Shape;
static id<MTLDevice> device;
static id<MTLCommandQueue> queue;
static id<MTLComputePipelineState> pipeline;

static id<MTLBuffer> borrow(const void *ptr, size_t len) {
    size_t page = getpagesize();
    if (!len || len > device.maxBufferLength || ((uintptr_t)ptr | len) % page)
        return nil;
    return [device newBufferWithBytesNoCopy:(void *)ptr
                                     length:len
                                    options:MTLResourceStorageModeShared
                                deallocator:nil];
}

static void prepare(const char *source) {
    @autoreleasepool {
        static dispatch_once_t once;
        dispatch_once(&once, ^{
          device = MTLCreateSystemDefaultDevice();
          queue = [device newCommandQueue];
          id<MTLLibrary> lib = [device newLibraryWithSource:[NSString stringWithUTF8String:source]
                                                    options:nil
                                                      error:NULL];
          id<MTLFunction> f = [lib newFunctionWithName:@"quotient_products"];
          if (f)
              pipeline = [device newComputePipelineStateWithFunction:f error:NULL];
        });
    }
}
int akita_quotient_products(const char *source, const Shape *shape, const uint64_t *matrix, const int32_t *z,
                            const int32_t *tables, int32_t *out, size_t bytes) {
    @autoreleasepool {
        prepare(source);
        if (!queue || !pipeline || pipeline.maxTotalThreadsPerThreadgroup < 256 ||
            pipeline.threadExecutionWidth != 32)
            return 0;
        id<MTLBuffer> key = borrow(matrix, (size_t)shape->width * 2048 * sizeof(*matrix));
        id<MTLBuffer> rhs = borrow(z, (size_t)shape->width * 2048 * sizeof(*z));
        id<MTLBuffer> result = borrow(out, bytes);
        id<MTLBuffer> tw = [device newBufferWithBytes:tables
                                               length:4 * 2 * 2048 * sizeof(*tables)
                                              options:MTLResourceStorageModeShared];
        if (!key || !rhs || !result || !tw)
            return 0;
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!encoder)
            return 0;
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:rhs offset:0 atIndex:0];
        [encoder setBuffer:key offset:0 atIndex:1];
        [encoder setBuffer:tw offset:0 atIndex:2];
        [encoder setBuffer:result offset:0 atIndex:3];
        [encoder setBytes:shape length:sizeof(*shape) atIndex:4];
        size_t jobs = ((size_t)shape->width + shape->chunk - 1) / shape->chunk;
        [encoder dispatchThreadgroups:MTLSizeMake(jobs, 4, 2) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        return command.status == MTLCommandBufferStatusCompleted;
    }
}
