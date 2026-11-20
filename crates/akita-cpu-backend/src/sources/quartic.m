#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <unistd.h>
#include <stdint.h>

typedef struct {
    uint64_t x, y;
} QuarticField;
typedef struct {
    uint32_t length, round, lowLength, stride, digits, prefixBytes, padding[2];
    QuarticField challenge;
} QuarticShape;
typedef struct {
    QuarticField terms[4];
} QuarticSum;

@interface AKQuarticState : NSObject {
  @public
    uint32_t digits, prefixBytes;
    id<MTLBuffer> input, tail, classes, rows, partial, sums;
}
@end
@implementation AKQuarticState
@end

static id<MTLDevice> device;
static id<MTLCommandQueue> queue;
static id<MTLComputePipelineState> quartic, reduce;

static void prepare_quartic(const char *source) {
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        device = MTLCreateSystemDefaultDevice();
        queue = [device newCommandQueue];
        id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:source]
                                                      options:nil
                                                        error:NULL];
        if (!library)
            return;
        id<MTLFunction> first = [library newFunctionWithName:@"quartic"];
        id<MTLFunction> second = [library newFunctionWithName:@"reduce_quartic"];
        if (!first || !second)
            return;
        quartic = [device newComputePipelineStateWithFunction:first error:NULL];
        reduce = [device newComputePipelineStateWithFunction:second error:NULL];
    });
}

void *akita_quartic_open(const char *source, const uint8_t *data, uint32_t n, const QuarticField *values) {
    @autoreleasepool {
        prepare_quartic(source);
        if (!queue || quartic.maxTotalThreadsPerThreadgroup < 128 || reduce.maxTotalThreadsPerThreadgroup < 128)
            return NULL;
        AKQuarticState *s = [[AKQuarticState alloc] init];
        s->digits = n;
        size_t bytes = ((size_t)n * 3 + 7) / 8;
        size_t page = (size_t)getpagesize();
        size_t prefix = bytes / page * page;
        if ((uintptr_t)data % page == 0 && prefix) {
            s->input = [device newBufferWithBytesNoCopy:(void *)data
                                                 length:prefix
                                                options:MTLResourceStorageModeShared
                                            deallocator:nil];
        }
        if (s->input) {
            s->prefixBytes = (uint32_t)prefix;
            if (prefix < bytes) {
                s->tail = [device newBufferWithBytes:data + prefix
                                              length:bytes - prefix
                                             options:MTLResourceStorageModeShared];
                if (!s->tail)
                    return NULL;
            }
        } else {
            s->input = [device newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared];
            s->prefixBytes = (uint32_t)bytes;
        }
        if (!s->tail)
            s->tail = s->input;
        s->classes = [device newBufferWithBytes:values
                                         length:65536 * sizeof(QuarticField)
                                        options:MTLResourceStorageModeShared];
        s->rows = [device newBufferWithLength:((n + 15UL) / 16) * sizeof(QuarticField)
                                      options:MTLResourceStorageModeShared];
        size_t groups = (n + 2047UL) / 2048;
        s->partial = [device newBufferWithLength:groups * sizeof(QuarticSum)
                                         options:MTLResourceStorageModePrivate];
        s->sums = [device newBufferWithLength:((groups + 127) / 128) * sizeof(QuarticSum)
                                      options:MTLResourceStorageModeShared];
        if (!s->input || !s->classes || !s->rows || !s->partial || !s->sums)
            return NULL;
        return (__bridge_retained void *)s;
    }
}

static uint64_t add_coefficient(uint64_t a, uint64_t b) {
    uint64_t sum = a + b;
    sum += (sum < a) * 59;
    return sum < UINT64_MAX - 58 ? sum : sum + 59;
}

int akita_quartic_round(void *handle, uint32_t round, const QuarticField *lo, uint32_t lowLength,
                        const QuarticField *hi, uint32_t highLength, const QuarticField *r, QuarticSum *result) {
    @autoreleasepool {
        AKQuarticState *s = (__bridge AKQuarticState *)handle;
        uint32_t count = (s->digits + (1u << round) - 1) >> round;
        uint32_t stride = round <= 4 ? 1 : 1u << (round - 5);
        QuarticShape shape = {count, round, lowLength, stride, s->digits, s->prefixBytes, {0, 0}, *r};
        uint32_t groups = (count + 255) / 256;
        uint32_t outputGroups = (groups + 127) / 128;
        id<MTLBuffer> low = [device newBufferWithBytes:lo
                                                length:lowLength * sizeof(*lo)
                                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> high = [device newBufferWithBytes:hi
                                                 length:highLength * sizeof(*hi)
                                                options:MTLResourceStorageModeShared];
        if (!low || !high)
            return 0;
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [command computeCommandEncoder];
        if (!enc)
            return 0;
        [enc setComputePipelineState:quartic];
        id<MTLBuffer> buffers[] = {s->input, s->classes, s->rows, low, high, s->partial};
        for (NSUInteger i = 0; i < 6; ++i)
            [enc setBuffer:buffers[i] offset:0 atIndex:i];
        [enc setBytes:&shape length:sizeof(shape) atIndex:6];
        [enc setBuffer:s->tail offset:0 atIndex:7];
        [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [enc setComputePipelineState:reduce];
        [enc setBuffer:s->partial offset:0 atIndex:0];
        [enc setBuffer:s->sums offset:0 atIndex:1];
        [enc setBytes:&groups length:sizeof(groups) atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(outputGroups, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [enc endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted)
            return 0;
        const QuarticSum *parts = s->sums.contents;
        *result = (QuarticSum){0};
        for (uint32_t i = 0; i < outputGroups; ++i) {
            for (size_t c = 0; c < 4; ++c) {
                result->terms[c].x = add_coefficient(result->terms[c].x, parts[i].terms[c].x);
                result->terms[c].y = add_coefficient(result->terms[c].y, parts[i].terms[c].y);
            }
        }
        return 1;
    }
}

void akita_quartic_export(void *handle, uint32_t round, size_t len, QuarticField *out) {
    AKQuarticState *s = (__bridge AKQuarticState *)handle;
    const QuarticField *rows = s->rows.contents;
    for (size_t i = 0; i < len; ++i)
        out[i] = rows[i << (round - 4)];
}

void akita_quartic_release(void *handle) { CFRelease(handle); }
