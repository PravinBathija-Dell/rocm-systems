/* Test stub: no-op anvil device ops for template coverage tests. */
#pragma once

#include "sdma_opcodes.h"

namespace sdma_anvil {

// Declared here. The single definitions live in anvil_stub_quiet_count.cpp
// because every TU that includes gin_anvil_sdma.h would otherwise emit one.
// test/CMakeLists.txt compiles IPC, Suite H and that TU -fgpu-rdc and
// device-links rccl-UnitTestsFixtures whenever ENABLE_ROCSHMEM_GIN is on.
extern __device__ unsigned long long g_sdmaStubQuietCount;

// quietTarget() returns the write index. isFlushed() reports busy for the given
// number of polls, then compares the read index against the target.
extern __device__ unsigned long long g_sdmaStubWriteIndex;
extern __device__ unsigned long long g_sdmaStubReadIndex;
extern __device__ unsigned long long g_sdmaStubBusyPolls;

struct SdmaQueueDeviceHandle {
  int tag;
};

struct SdmaQueueSingleProducerDeviceHandle {
  int tag;
};

__device__ __forceinline__ void memcpyDevice(void* dst, const void* src, size_t size) {
  if (dst == nullptr || src == nullptr || size == 0) return;
  auto* d = static_cast<char*>(dst);
  const auto* s = static_cast<const char*>(src);
  for (size_t i = 0; i < size; ++i) d[i] = s[i];
}

__device__ __forceinline__ void put(SdmaQueueDeviceHandle& handle, void* dst, void* src, size_t size) {
  (void)handle;
  memcpyDevice(dst, src, size);
}

__device__ __forceinline__ void putSignal(SdmaQueueDeviceHandle& handle, void* dst, void* src, size_t size,
                                          uint64_t* signal) {
  (void)signal;
  (void)handle;
  memcpyDevice(dst, src, size);
}

__device__ __forceinline__ void quiet(SdmaQueueDeviceHandle& handle) {
  (void)handle;
  atomicAdd(&g_sdmaStubQuietCount, 1ULL);
}

__device__ __forceinline__ uint64_t quietTarget(SdmaQueueDeviceHandle& handle) {
  (void)handle;
  return g_sdmaStubWriteIndex;
}

__device__ __forceinline__ bool isFlushed(SdmaQueueDeviceHandle& handle, uint64_t upToIndex) {
  (void)handle;
  atomicAdd(&g_sdmaStubQuietCount, 1ULL);
  if (g_sdmaStubBusyPolls != 0) {
    --g_sdmaStubBusyPolls;
    return false;
  }
  return g_sdmaStubReadIndex >= upToIndex;
}

}  // namespace sdma_anvil
