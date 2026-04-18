// Copyright (c) Microsoft. All rights reserved.
// Objective-C++ wrapper for Metal backend

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <torch/extension.h>
#include <vector>
#include <cstring>
#include <dlfcn.h>

// Metal device and pipeline state
static id<MTLDevice> g_device = nil;
static id<MTLCommandQueue> g_commandQueue = nil;
static id<MTLLibrary> g_library = nil;

// Pipeline states for each kernel
static id<MTLComputePipelineState> g_matmulPipeline = nil;
static id<MTLComputePipelineState> g_quantizePipeline = nil;

// Initialize Metal
bool metal_init() {
    if (g_device != nil) return true;

    g_device = MTLCreateSystemDefaultDevice();
    if (g_device == nil) return false;

    g_commandQueue = [g_device newCommandQueue];

    NSError* error = nil;
    g_library = nil;

    // Use dladdr to find the .so's directory and locate shader files next to it.
    // NSBundle mainBundle refers to the Python process bundle which does not
    // contain our Metal resources.
    Dl_info dl_info;
    if (dladdr((void*)metal_init, &dl_info) && dl_info.dli_fname) {
        NSString* so_path = [NSString stringWithUTF8String:dl_info.dli_fname];
        NSString* dir = [so_path stringByDeletingLastPathComponent];

        // Prefer pre-compiled .metallib
        NSString* metallib_path = [dir stringByAppendingPathComponent:@"bitnet_kernels.metallib"];
        if ([[NSFileManager defaultManager] fileExistsAtPath:metallib_path]) {
            g_library = [g_device newLibraryWithURL:[NSURL fileURLWithPath:metallib_path]
                                              error:&error];
        }

        // Fall back to compiling from .metal source at runtime
        if (g_library == nil) {
            NSString* metal_path = [dir stringByAppendingPathComponent:@"bitnet_kernels.metal"];
            if ([[NSFileManager defaultManager] fileExistsAtPath:metal_path]) {
                NSString* source = [NSString stringWithContentsOfFile:metal_path
                                                             encoding:NSUTF8StringEncoding
                                                                error:&error];
                if (source != nil) {
                    g_library = [g_device newLibraryWithSource:source options:nil error:&error];
                }
            }
        }
    }

    // Last-resort inline shader (uses 'bfloat' — the correct MSL type)
    if (g_library == nil) {
        const char* defaultShaders = R"(
#include <metal_stdlib>
using namespace metal;

kernel void bitlinear_int8xint2(
    device const int8_t* A [[buffer(0)]],
    device const uint8_t* B [[buffer(1)]],
    device bfloat* C [[buffer(2)]],
    device const bfloat* s [[buffer(3)]],
    device const bfloat* ws [[buffer(4)]],
    constant int& M [[buffer(5)]],
    constant int& N [[buffer(6)]],
    constant int& K [[buffer(7)]],
    uint2 tid [[thread_position_in_grid]]
) {
    const int m_idx = tid.y;
    const int n_idx = tid.x;
    if (m_idx >= M || n_idx >= N) return;
    int32_t acc = 0;
    for (int k = 0; k < K; k++) {
        uint8_t packed = B[n_idx * (K / 4) + k / 4];
        int8_t w = (int8_t)(((packed >> ((k % 4) * 2)) & 0x03) - 1);
        acc += (int32_t)A[m_idx * K + k] * (int32_t)w;
    }
    float result = (float)acc / (float)s[m_idx] * (float)ws[n_idx];
    C[m_idx * N + n_idx] = bfloat(result);
}
)";
        NSString* source = [NSString stringWithUTF8String:defaultShaders];
        g_library = [g_device newLibraryWithSource:source options:nil error:&error];
    }

    if (g_library == nil) return false;

    id<MTLFunction> matmulFunction = [g_library newFunctionWithName:@"bitlinear_int8xint2"];
    if (matmulFunction == nil) return false;

    g_matmulPipeline = [g_device newComputePipelineStateWithFunction:matmulFunction error:&error];
    return g_device != nil && g_commandQueue != nil && g_matmulPipeline != nil;
}

// ---------------------------------------------------------------------------
// GPU-resident dispatch: callers pass raw Metal buffer pointers (as uintptr_t)
// obtained from PyTorch MPS tensors via mps_storage_ptr().  No CPU copies.
// ---------------------------------------------------------------------------
void metal_matmul_buffers(
    int64_t M, int64_t N, int64_t K,
    uintptr_t A_mtl,   // id<MTLBuffer> cast to uintptr_t  — int8  [M, K]
    uintptr_t B_mtl,   // id<MTLBuffer>                    — uint8 [N, K/4]
    uintptr_t C_mtl,   // id<MTLBuffer>                    — bfloat [M, N]
    uintptr_t s_mtl,   // id<MTLBuffer>                    — bfloat [M]
    uintptr_t ws_mtl   // id<MTLBuffer>                    — bfloat [N]
) {
    if (!metal_init()) {
        throw std::runtime_error("Metal initialization failed");
    }

    id<MTLBuffer> A_buf  = (__bridge id<MTLBuffer>)(void*)A_mtl;
    id<MTLBuffer> B_buf  = (__bridge id<MTLBuffer>)(void*)B_mtl;
    id<MTLBuffer> C_buf  = (__bridge id<MTLBuffer>)(void*)C_mtl;
    id<MTLBuffer> s_buf  = (__bridge id<MTLBuffer>)(void*)s_mtl;
    id<MTLBuffer> ws_buf = (__bridge id<MTLBuffer>)(void*)ws_mtl;

    @autoreleasepool {
        id<MTLCommandBuffer>        cmd     = [g_commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];

        [encoder setComputePipelineState:g_matmulPipeline];

        [encoder setBuffer:A_buf  offset:0 atIndex:0];
        [encoder setBuffer:B_buf  offset:0 atIndex:1];
        [encoder setBuffer:C_buf  offset:0 atIndex:2];
        [encoder setBuffer:s_buf  offset:0 atIndex:3];
        [encoder setBuffer:ws_buf offset:0 atIndex:4];

        int M_i = (int)M, N_i = (int)N, K_i = (int)K;
        [encoder setBytes:&M_i length:sizeof(int) atIndex:5];
        [encoder setBytes:&N_i length:sizeof(int) atIndex:6];
        [encoder setBytes:&K_i length:sizeof(int) atIndex:7];

        MTLSize gridSize        = MTLSizeMake(N, M, 1);
        MTLSize threadgroupSize = MTLSizeMake(32, 8, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];

        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

// ---------------------------------------------------------------------------
// CPU-copy fallback: safe for any tensor regardless of storage location.
// ---------------------------------------------------------------------------
void metal_matmul_copy(
    int64_t M, int64_t N, int64_t K,
    void* A_ptr, void* B_ptr, void* C_ptr, void* s_ptr, void* ws_ptr
) {
    if (!metal_init()) {
        throw std::runtime_error("Metal initialization failed");
    }

    @autoreleasepool {
        size_t A_sz  = M * K;
        size_t B_sz  = N * (K / 4);
        size_t C_sz  = M * N * 2;   // bfloat16 = 2 bytes
        size_t s_sz  = M * 2;
        size_t ws_sz = N * 2;

        id<MTLBuffer> A_buf  = [g_device newBufferWithBytes:A_ptr  length:A_sz  options:MTLResourceStorageModeShared];
        id<MTLBuffer> B_buf  = [g_device newBufferWithBytes:B_ptr  length:B_sz  options:MTLResourceStorageModeShared];
        id<MTLBuffer> C_buf  = [g_device newBufferWithLength:C_sz              options:MTLResourceStorageModeShared];
        id<MTLBuffer> s_buf  = [g_device newBufferWithBytes:s_ptr  length:s_sz  options:MTLResourceStorageModeShared];
        id<MTLBuffer> ws_buf = [g_device newBufferWithBytes:ws_ptr length:ws_sz options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer>        cmd     = [g_commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];

        [encoder setComputePipelineState:g_matmulPipeline];
        [encoder setBuffer:A_buf  offset:0 atIndex:0];
        [encoder setBuffer:B_buf  offset:0 atIndex:1];
        [encoder setBuffer:C_buf  offset:0 atIndex:2];
        [encoder setBuffer:s_buf  offset:0 atIndex:3];
        [encoder setBuffer:ws_buf offset:0 atIndex:4];

        int M_i = (int)M, N_i = (int)N, K_i = (int)K;
        [encoder setBytes:&M_i length:sizeof(int) atIndex:5];
        [encoder setBytes:&N_i length:sizeof(int) atIndex:6];
        [encoder setBytes:&K_i length:sizeof(int) atIndex:7];

        MTLSize gridSize        = MTLSizeMake(N, M, 1);
        MTLSize threadgroupSize = MTLSizeMake(32, 8, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];

        [cmd commit];
        [cmd waitUntilCompleted];

        memcpy(C_ptr, [C_buf contents], C_sz);
    }
}

// Python-callable wrappers
void bitlinear_metal(
    int64_t M, int64_t N, int64_t K,
    uintptr_t A, uintptr_t B, uintptr_t C, uintptr_t s, uintptr_t ws
) {
    metal_matmul_copy(M, N, K,
        reinterpret_cast<void*>(A), reinterpret_cast<void*>(B),
        reinterpret_cast<void*>(C), reinterpret_cast<void*>(s),
        reinterpret_cast<void*>(ws));
}

void bitlinear_metal_mps(
    int64_t M, int64_t N, int64_t K,
    uintptr_t A_mtl, uintptr_t B_mtl, uintptr_t C_mtl,
    uintptr_t s_mtl, uintptr_t ws_mtl
) {
    metal_matmul_buffers(M, N, K, A_mtl, B_mtl, C_mtl, s_mtl, ws_mtl);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bitlinear_metal", &bitlinear_metal,
          "BitNet linear (CPU-copy path — safe with any tensor storage)",
          py::arg("M"), py::arg("N"), py::arg("K"),
          py::arg("A"), py::arg("B"), py::arg("C"), py::arg("s"), py::arg("ws"));
    m.def("bitlinear_metal_mps", &bitlinear_metal_mps,
          "BitNet linear (GPU-resident path — pass raw MTLBuffer handles)",
          py::arg("M"), py::arg("N"), py::arg("K"),
          py::arg("A_mtl"), py::arg("B_mtl"), py::arg("C_mtl"),
          py::arg("s_mtl"), py::arg("ws_mtl"));
    m.def("metal_init", &metal_init, "Initialize Metal device and pipeline");
}
