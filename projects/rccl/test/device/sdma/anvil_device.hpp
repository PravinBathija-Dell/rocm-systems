/* Test stub: no-op anvil device ops for template coverage tests. */
#pragma once

#include "sdma_opcodes.h"

namespace sdma_anvil {

// Declared here; the single definition lives in anvil_stub_quiet_count.cpp
// because every TU that includes gin_anvil_sdma.h would otherwise emit one.
// test/CMakeLists.txt compiles IPC, Suite H and that TU -fgpu-rdc and
// device-links rccl-UnitTestsFixtures whenever ENABLE_ROCSHMEM_GIN is on.
extern __device__ unsigned long long g_sdmaStubQuietCount;

// quietTarget() returns writeIndex. isFlushed() reports busy for busyPolls polls, then checks whether the
// read index, readLag behind writeIndex, has reached the target. A zeroed handle has nothing in flight.
struct SdmaQueueDeviceHandle {
  int tag;
  unsigned long long writeIndex;
  unsigned long long readLag;
  unsigned long long busyPolls;
  unsigned long long polls;
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
  return handle.writeIndex;
}

__device__ __forceinline__ bool isFlushed(SdmaQueueDeviceHandle& handle, uint64_t upToIndex) {
  atomicAdd(&g_sdmaStubQuietCount, 1ULL);
  ++handle.polls;
  if (handle.busyPolls != 0) {
    --handle.busyPolls;
    return false;
  }
  return handle.writeIndex - handle.readLag >= upToIndex;
}

}  // namespace sdma_anvil
