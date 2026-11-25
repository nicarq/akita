#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <unistd.h>
#include <stdint.h>

typedef struct {
    uint32_t blocks, begin, end, rows;
    int32_t prime[4];
} FoldShape;

static id<MTLBuffer> borrow_pages(id<MTLDevice> device, const void *data, size_t len) {
    size_t page = (size_t)getpagesize();
    if (!len || len > device.maxBufferLength || (uintptr_t)data % page != 0 || len % page != 0)
        return nil;
    return [device newBufferWithBytesNoCopy:(void *)data
                                     length:len
                                    options:MTLResourceStorageModeShared
                                deallocator:nil];
}

static id<MTLDevice> device;
static id<MTLCommandQueue> queue;
static id<MTLComputePipelineState> products, inverse, packedProducts, packedSums, lookupProducts, lookupSums;
static id<MTLComputePipelineState> rawNtt;
static void prepare(const char *source) {
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        device = MTLCreateSystemDefaultDevice();
        if (!device)
            return;
        queue = [device newCommandQueue];
        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:source]
                                                      options:nil
                                                        error:&error];
        if (!library)
            return;
        products = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"fold_products"]
                                                         error:&error];
        inverse = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"fold_inverse"]
                                                        error:&error];
        rawNtt = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"raw_ntt"] error:&error];
        packedProducts =
            [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"packed_commit_products"]
                                                  error:&error];
        packedSums =
            [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"packed_commit_sums"]
                                                  error:&error];
        lookupProducts =
            [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"lookup_commit_products"]
                                                  error:&error];
        lookupSums =
            [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"lookup_commit_sums"]
                                                  error:&error];
    });
}

void *pgl_akita_metal_fold_key(const char *source, const int32_t *keys, size_t keyBytes) {
    @autoreleasepool {
        prepare(source);
        if (!queue || !products || !inverse)
            return NULL;
        return (__bridge_retained void *)borrow_pages(device, keys, keyBytes);
    }
}

void pgl_akita_metal_fold_release(void *key) {
    if (key)
        CFRelease(key);
}

int pgl_akita_metal_fold(void *rawKey, const FoldShape *shape, const int32_t *weights, const int32_t *tables,
                     int32_t *out) {
    @autoreleasepool {
        id<MTLBuffer> key = (__bridge id<MTLBuffer>)rawKey;
        NSUInteger words = (NSUInteger)shape->rows * 2048;
        id<MTLBuffer> output = borrow_pages(device, out, words * 4);
        BOOL copyOutput = output == nil;
        if (copyOutput)
            output = [device newBufferWithLength:words * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> w = [device newBufferWithBytes:weights
                                              length:shape->blocks * 128UL * 4
                                             options:MTLResourceStorageModeShared];
        id<MTLBuffer> t = [device newBufferWithBytes:tables
                                              length:4 * 128 * 4
                                             options:MTLResourceStorageModeShared];
        if (!output || !w || !t)
            return 0;
        id<MTLCommandBuffer> command = [queue commandBuffer];
        if (!command)
            return 0;
        id<MTLComputeCommandEncoder> enc = [command computeCommandEncoder];
        if (!enc)
            return 0;
        [enc setComputePipelineState:products];
        [enc setBuffer:key offset:0 atIndex:0];
        [enc setBuffer:w offset:0 atIndex:1];
        [enc setBuffer:output offset:0 atIndex:2];
        [enc setBytes:shape length:sizeof(FoldShape) atIndex:3];
        [enc dispatchThreads:MTLSizeMake(words * 2, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
        enc = [command computeCommandEncoder];
        if (!enc)
            return 0;
        [enc setComputePipelineState:inverse];
        [enc setBuffer:output offset:0 atIndex:0];
        [enc setBuffer:output offset:0 atIndex:1];
        [enc setBuffer:t offset:0 atIndex:2];
        [enc setBytes:shape length:sizeof(FoldShape) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(words / 128, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [enc endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted)
            return 0;
        if (copyOutput)
            memcpy(out, output.contents, words * 4);
        return 1;
    }
}

typedef struct {
    uint32_t tile, positions, blocks;
} PackingShape;

static id<MTLDevice> packingDevice;
static id<MTLCommandQueue> packingQueue;
static id<MTLComputePipelineState> pack, combine, weightedRows;

void *pgl_akita_metal_pack_key(const char *shader, const uint64_t *source, size_t bytes) {
    @autoreleasepool {
        static dispatch_once_t once;
        dispatch_once(&once, ^{
            packingDevice = MTLCreateSystemDefaultDevice();
            packingQueue = [packingDevice newCommandQueue];
            NSError *error = nil;
            id<MTLLibrary> library = [packingDevice newLibraryWithSource:[NSString stringWithUTF8String:shader]
                                                                 options:nil
                                                                   error:&error];
            if (!library)
                return;
            id<MTLFunction> first = [library newFunctionWithName:@"pack"];
            id<MTLFunction> second = [library newFunctionWithName:@"combine"];
            if (!first || !second)
                return;
            pack = [packingDevice newComputePipelineStateWithFunction:first error:&error];
            combine = [packingDevice newComputePipelineStateWithFunction:second error:&error];
            id<MTLFunction> rows = [library newFunctionWithName:@"weighted_rows"];
            if (rows)
                weightedRows = [packingDevice newComputePipelineStateWithFunction:rows error:&error];
        });
        if (!packingQueue || !combine || pack.threadExecutionWidth != 32 ||
            pack.maxTotalThreadsPerThreadgroup < 128)
            return NULL;
        return (__bridge_retained void *)borrow_pages(packingDevice, source, bytes);
    }
}
int pgl_akita_metal_pack(void *handle, const uint64_t *weights, uint32_t positions, uint32_t blocks,
                     uint64_t *output) {
    @autoreleasepool {
        id<MTLBuffer> key = (__bridge id<MTLBuffer>)handle;
        const PackingShape shape = {64, positions, blocks};
        id<MTLBuffer> w = [packingDevice newBufferWithBytes:weights
                                                     length:positions * 16UL * 16
                                                    options:MTLResourceStorageModeShared];
        id<MTLBuffer> partial =
            [packingDevice newBufferWithLength:(size_t)blocks * (positions / shape.tile) * 128 * 16
                                       options:MTLResourceStorageModePrivate];
        id<MTLBuffer> out = [packingDevice newBufferWithLength:blocks * 256UL * 8
                                                       options:MTLResourceStorageModeShared];
        if (!w || !partial || !out)
            return 0;
        id<MTLCommandBuffer> command = [packingQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [command computeCommandEncoder];
        if (!enc)
            return 0;
        [enc setComputePipelineState:pack];
        [enc setBuffer:key offset:0 atIndex:0];
        [enc setBuffer:w offset:0 atIndex:1];
        [enc setBuffer:partial offset:0 atIndex:2];
        [enc setBytes:&shape length:sizeof(shape) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(16, positions / shape.tile, blocks)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [enc setComputePipelineState:combine];
        [enc setBuffer:partial offset:0 atIndex:0];
        [enc setBuffer:out offset:0 atIndex:1];
        [enc setBytes:&shape length:sizeof(shape) atIndex:2];
        [enc dispatchThreads:MTLSizeMake(128, blocks, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [enc endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted)
            return 0;
        memcpy(output, out.contents, blocks * 256UL * sizeof(uint64_t));
        return 1;
    }
}

typedef struct {
    uint32_t blocks, cols, digits, prefixBytes;
    int32_t primes[3][4];
} PackedCommitShape;

int pgl_akita_metal_packed_commit(const char *source, const PackedCommitShape *input, const uint8_t *digits,
                              const int32_t *matrix, const int32_t *tables, int32_t *out) {
    @autoreleasepool {
        prepare(source);
        if (!queue || !packedProducts || !packedSums)
            return 0;
        PackedCommitShape shape = *input;
        size_t bytes = ((size_t)shape.digits * 3 + 7) / 8;
        size_t page = (size_t)getpagesize();
        shape.prefixBytes = (uint32_t)(bytes / page * page);
        id<MTLBuffer> prefix = borrow_pages(device, digits, shape.prefixBytes);
        id<MTLBuffer> key = borrow_pages(device, matrix, (size_t)shape.cols * 2 * 512 * 3 * 4);
        if (!prefix || !key)
            return 0;
        size_t tailLen = bytes - shape.prefixBytes;
        id<MTLBuffer> tail = tailLen ? [device newBufferWithBytes:digits + shape.prefixBytes
                                                           length:tailLen
                                                          options:MTLResourceStorageModeShared]
                                     : prefix;
        id<MTLBuffer> tw = [device newBufferWithBytes:tables
                                               length:3 * 2 * 512 * 4
                                              options:MTLResourceStorageModeShared];
        uint32_t chunks = (shape.cols + 255) / 256;
        NSUInteger values = (NSUInteger)shape.blocks * 2 * 512 * 3;
        id<MTLBuffer> partials = [device newBufferWithLength:values * chunks * 4
                                                     options:MTLResourceStorageModePrivate];
        id<MTLBuffer> sums = [device newBufferWithLength:values * 4 options:MTLResourceStorageModeShared];
        if (!partials || !sums || !tail || !tw)
            return 0;
        id<MTLCommandBuffer> command = [queue commandBuffer];
        if (!command)
            return 0;
        id<MTLComputeCommandEncoder> enc = [command computeCommandEncoder];
        if (!enc)
            return 0;
        [enc setComputePipelineState:packedProducts];
        [enc setBuffer:prefix offset:0 atIndex:0];
        [enc setBuffer:tail offset:0 atIndex:1];
        [enc setBuffer:key offset:0 atIndex:2];
        [enc setBuffer:tw offset:0 atIndex:3];
        [enc setBuffer:partials offset:0 atIndex:4];
        [enc setBytes:&shape length:sizeof(shape) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(chunks, (shape.blocks + 3) / 4, 3)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
        enc = [command computeCommandEncoder];
        if (!enc)
            return 0;
        [enc setComputePipelineState:packedSums];
        [enc setBuffer:partials offset:0 atIndex:0];
        [enc setBuffer:sums offset:0 atIndex:1];
        [enc setBytes:&shape length:sizeof(shape) atIndex:2];
        [enc dispatchThreads:MTLSizeMake(values, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted)
            return 0;
        memcpy(out, sums.contents, values * 4);
        return 1;
    }
}

int pgl_akita_metal_weighted_rows(void *handle, uint32_t cells, uint32_t rows, const uint64_t *weights,
                              uint64_t *output) {
    @autoreleasepool {
        if (!weightedRows)
            return 0;
        id<MTLBuffer> source = (__bridge id<MTLBuffer>)handle;
        const size_t bytes = (size_t)cells * 16;
        if (source.device != packingQueue.device || bytes > source.device.maxBufferLength)
            return 0;
        id<MTLBuffer> out = borrow_pages(source.device, output, bytes);
        if (!out)
            return 0;
        const uint32_t shape[] = {cells, rows};
        id<MTLCommandBuffer> cmd = [packingQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (!enc)
            return 0;
        [enc setComputePipelineState:weightedRows];
        [enc setBuffer:source offset:0 atIndex:0];
        [enc setBytes:weights length:rows * 16 atIndex:1];
        [enc setBuffer:out offset:0 atIndex:2];
        [enc setBytes:shape length:sizeof(shape) atIndex:3];
        [enc dispatchThreads:MTLSizeMake(cells, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(weightedRows.threadExecutionWidth * 4, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        return cmd.status == MTLCommandBufferStatusCompleted;
    }
}

typedef struct {
    uint32_t blocks, columns, tileColumns, tableLen, columnWidth, codeCount, liveCells;
    int32_t primes[2][4];
} LookupShape;
int pgl_akita_metal_lookup_commit(const char *source, const LookupShape *shape, const uint16_t *codes,
                              const int16_t *lookup, size_t lookupBytes, const int32_t *spectra, size_t matrixBytes,
                              const int32_t *twiddles, const uint32_t *ranges, int32_t *out) {
    @autoreleasepool {
        prepare(source);
        if (!queue || !lookupProducts || !lookupSums || lookupProducts.maxTotalThreadsPerThreadgroup < 256)
            return 0;
        size_t outputBytes = (size_t)shape->blocks * 9 * 2 * 512 * 4;
        size_t groups = (shape->columns + shape->tileColumns - 1) / shape->tileColumns;
        id<MTLBuffer> key = borrow_pages(device, spectra, matrixBytes);
        id<MTLBuffer> input = borrow_pages(device, codes, (size_t)shape->codeCount * 2);
        if (!input || !key)
            return 0;
        id<MTLBuffer> table = [device newBufferWithBytes:lookup
                                                  length:lookupBytes
                                                 options:MTLResourceStorageModeShared];
        id<MTLBuffer> tables = [device newBufferWithBytes:twiddles
                                                   length:2 * 2 * 512 * 4
                                                  options:MTLResourceStorageModeShared];
        id<MTLBuffer> partials = [device newBufferWithLength:outputBytes * groups
                                                     options:MTLResourceStorageModePrivate];
        id<MTLBuffer> result = [device newBufferWithLength:outputBytes options:MTLResourceStorageModeShared];
        if (!table || !tables || !partials || !result)
            return 0;
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!encoder)
            return 0;
        [encoder setComputePipelineState:lookupProducts];
        id<MTLBuffer> buffers[] = {input, table, key, tables, partials};
        for (size_t i = 0; i < 5; i++)
            [encoder setBuffer:buffers[i] offset:0 atIndex:i];
        [encoder setBytes:shape length:sizeof(*shape) atIndex:5];
        [encoder setBytes:ranges length:shape->blocks * 2 * 4 atIndex:6];
        [encoder dispatchThreadgroups:MTLSizeMake(groups, (shape->blocks + 1) / 2, 2)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
        encoder = [command computeCommandEncoder];
        if (!encoder)
            return 0;
        [encoder setComputePipelineState:lookupSums];
        [encoder setBuffer:partials offset:0 atIndex:0];
        [encoder setBuffer:result offset:0 atIndex:1];
        [encoder setBytes:shape length:sizeof(*shape) atIndex:2];
        [encoder dispatchThreads:MTLSizeMake(outputBytes / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted)
            return 0;
        memcpy(out, result.contents, outputBytes);
        return 1;
    }
}

int pgl_akita_metal_raw_ntt(const char *shader, void *rawKey, const uint64_t *source, size_t sourceBytes,
                        const FoldShape *shape, const int32_t *weights, const int32_t *tables, int32_t *out) {
    @autoreleasepool {
        prepare(shader);
        if (!queue || rawNtt.threadExecutionWidth != 32 || rawNtt.maxTotalThreadsPerThreadgroup < 128 ||
            shape->begin >= shape->end ||
            shape->end > shape->blocks || !shape->rows || shape->rows % 4)
            return 0;
        const uint32_t positions = shape->rows / 4;
        const size_t words = (size_t)shape->rows * 2048;
        const size_t ringBytes = 2048 * sizeof(uint64_t);
        const size_t rings = sourceBytes / ringBytes;
        if (!rings || sourceBytes % ringBytes || (rings - 1) / positions + 1 != shape->blocks)
            return 0;
        const uint32_t lastPositions = (uint32_t)((rings - 1) % positions) + 1;
        id<MTLBuffer> key = rawKey ? (__bridge id<MTLBuffer>)rawKey : borrow_pages(device, source, sourceBytes);
        if (!key || key.length < sourceBytes)
            return 0;
        id<MTLBuffer> w = [device newBufferWithBytes:weights length:shape->blocks * 128UL * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> t = [device newBufferWithBytes:tables length:4 * 128 * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> output = borrow_pages(device, out, words * 4);
        BOOL copyOutput = output == nil;
        if (copyOutput)
            output = [device newBufferWithLength:words * 4 options:MTLResourceStorageModeShared];
        if (!w || !t || !output)
            return 0;
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [command computeCommandEncoder];
        if (!enc)
            return 0;
        [enc setComputePipelineState:rawNtt];
        [enc setBuffer:key offset:0 atIndex:0];
        [enc setBuffer:w offset:0 atIndex:1];
        [enc setBuffer:t offset:0 atIndex:2];
        [enc setBuffer:output offset:0 atIndex:3];
        [enc setBytes:shape length:sizeof(FoldShape) atIndex:4];
        [enc setBytes:&lastPositions length:sizeof(lastPositions) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake((size_t)positions * 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [enc endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted)
            return 0;
        if (copyOutput)
            memcpy(out, output.contents, words * 4);
        return 1;
    }
}
