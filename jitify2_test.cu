/*
 * Copyright (c) 2017-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * * Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimer in the
 *   documentation and/or other materials provided with the distribution.
 * * Neither the name of NVIDIA CORPORATION nor the names of its
 *   contributors may be used to endorse or promote products derived
 *   from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#define JITIFY_ENABLE_EXCEPTIONS 1
#include "jitify2.hpp"

#include "example_headers/class_arg_kernel.cuh"
#include "example_headers/my_header1.cuh.jit"
#include "jitify2_test_kernels.cu.jit.hpp"

#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#if defined _WIN32 || defined _WIN64
#include <direct.h>
#define chdir _chdir
#else
#include <unistd.h>
#endif

#include "gtest/gtest.h"

#define CHECK_CUDA(call)                                                  \
  do {                                                                    \
    CUresult status = call;                                               \
    if (status != CUDA_SUCCESS) {                                         \
      const char* str;                                                    \
      cuda().GetErrorName()(status, &str);                                \
      std::cout << "(CUDA) returned " << str;                             \
      std::cout << " (" << __FILE__ << ":" << __LINE__ << ":" << __func__ \
                << "())" << std::endl;                                    \
      ASSERT_EQ(status, CUDA_SUCCESS);                                    \
    }                                                                     \
  } while (0)

#define CHECK_CUDART(call)                                                \
  do {                                                                    \
    cudaError_t status = call;                                            \
    if (status != cudaSuccess) {                                          \
      std::cout << "(CUDART) returned " << cudaGetErrorString(status);    \
      std::cout << " (" << __FILE__ << ":" << __LINE__ << ":" << __func__ \
                << "())" << std::endl;                                    \
      ASSERT_EQ(status, cudaSuccess);                                     \
    }                                                                     \
  } while (0)

using namespace jitify2;
using namespace jitify2::reflection;

template <typename ValueType, typename ErrorType>
std::string get_error(
    const jitify2::detail::FallibleValue<ValueType, ErrorType>& x) {
  if (x) return "";
  return x.error();
}

void debug_print(const StringVec& v, const std::string& varname) {
  std::cerr << "--- BEGIN VECTOR " << varname << " ---\n";
  for (const auto& x : v) {
    std::cerr << x << "\n";
  }
  std::cerr << "--- END VECTOR " << varname << " ---" << std::endl;
}

bool contains(const StringVec& v, const std::string& s, const char* varname) {
  bool result = std::find(v.begin(), v.end(), s) != v.end();
  if (!result) debug_print(v, varname);
  return result;
}
bool not_contains(const StringVec& v, const std::string& s,
                  const char* varname) {
  bool result = std::find(v.begin(), v.end(), s) == v.end();
  if (!result) debug_print(v, varname);
  return result;
}

#define CONTAINS(src, target) contains(src, target, #src)
#define NOT_CONTAINS(src, target) not_contains(src, target, #src)

TEST(Jitify2Test, Simple) {
  static const char* const source = R"(
template <int N, typename T>
__global__ void my_kernel(T* data) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  T data0 = data[0];
  for( int i=0; i<N-1; ++i ) {
    data[0] *= data0;
  }
})";
  using dtype = float;
  dtype* d_data;
  CHECK_CUDART(cudaMalloc((void**)&d_data, sizeof(dtype)));
  // Test serialization.
  auto program =
      Program::deserialize(Program("my_program", source)->serialize());
  ASSERT_EQ(get_error(program), "");
  auto preprog =
      PreprocessedProgram::deserialize(program->preprocess()->serialize());
  ASSERT_EQ(get_error(preprog), "");
  std::string kernel_inst =
      Template("my_kernel").instantiate(3, type_of(*d_data));
  auto compiled =
      CompiledProgram::deserialize(preprog->compile(kernel_inst)->serialize());
  ASSERT_EQ(get_error(compiled), "");
  auto linked = LinkedProgram::deserialize(compiled->link()->serialize());
  ASSERT_EQ(get_error(linked), "");

  // Test that kernel instantiation produces correct result.
  Kernel kernel = linked->load()->get_kernel(kernel_inst);
  dim3 grid(1), block(1);
  dtype h_data = 5;
  CHECK_CUDART(
      cudaMemcpy(d_data, &h_data, sizeof(dtype), cudaMemcpyHostToDevice));
  ASSERT_EQ(kernel->configure(grid, block)->launch(d_data), "");
  CHECK_CUDART(
      cudaMemcpy(&h_data, d_data, sizeof(dtype), cudaMemcpyDeviceToHost));
  EXPECT_FLOAT_EQ(h_data, 125.f);

  h_data = 5;
  CHECK_CUDART(
      cudaMemcpy(d_data, &h_data, sizeof(dtype), cudaMemcpyHostToDevice));
  ASSERT_EQ(kernel->configure_1d_max_occupancy()->launch(d_data), "");
  CHECK_CUDART(
      cudaMemcpy(&h_data, d_data, sizeof(dtype), cudaMemcpyDeviceToHost));
  EXPECT_FLOAT_EQ(h_data, 125.f);

  CHECK_CUDART(cudaFree(d_data));
}

bool header_callback(const std::string& filename, std::string* source) {
  // On success, write to *source and return true, otherwise return false.
  if (filename == "example_headers/my_header4.cuh") {
    *source = R"(
#pragma once
template <typename T>
T pointless_func(T x) {
  return x;
};)";
    return true;
  } else {
    // Find this file through other mechanisms.
    return false;
  }
}

// Returns, e.g., "61" for a device of compute capability 6.1.
int get_current_device_arch() {
  int device;
  cudaGetDevice(&device);
  int cc_major;
  cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, device);
  int cc_minor;
  cudaDeviceGetAttribute(&cc_minor, cudaDevAttrComputeCapabilityMinor, device);
  int cc = cc_major * 10 + cc_minor;
  return cc;
}

TEST(Jitify2Test, MultipleKernels) {
  static const char* const source = R"(
#include "example_headers/my_header1.cuh"
#include "example_headers/my_header2.cuh"
#include "example_headers/my_header3.cuh"
#include "example_headers/my_header4.cuh"

__global__ void my_kernel1(const float* indata, float* outdata) {
  outdata[0] = indata[0] + 1;
  outdata[0] -= 1;
}

template <int C, typename T>
__global__ void my_kernel2(const float* indata, float* outdata) {
  for (int i = 0; i < C; ++i) {
    outdata[0] = pointless_func(identity(sqrt(square(negate(indata[0])))));
  }
})";

  enum { C = 123 };
  typedef float T;
  std::string kernel2_inst =
      Template("my_kernel2").instantiate<NonType<int, C>, T>();
  LoadedProgram program = Program("multiple_kernels_program", source)
                              ->preprocess({}, {}, header_callback)
                              ->load({"my_kernel1", kernel2_inst});
  ASSERT_EQ(get_error(program), "");

  T* indata;
  T* outdata;
  CHECK_CUDART(cudaMalloc((void**)&indata, sizeof(T)));
  CHECK_CUDART(cudaMalloc((void**)&outdata, sizeof(T)));
  T inval = 3.14159f;
  CHECK_CUDART(cudaMemcpy(indata, &inval, sizeof(T), cudaMemcpyHostToDevice));

  dim3 grid(1), block(1);
  ASSERT_EQ(program->get_kernel("my_kernel1")
                ->configure(grid, block)
                ->launch(indata, outdata),
            "");
  // These invocations are all equivalent.
  ASSERT_EQ(program->get_kernel(kernel2_inst)
                ->configure(grid, block)
                ->launch(indata, outdata),
            "");
  ASSERT_EQ(program
                ->get_kernel(Template("my_kernel2")
                                 .instantiate({reflect((int)C), reflect<T>()}))
                ->configure(grid, block)
                ->launch(indata, outdata),
            "");
  ASSERT_EQ(
      program->get_kernel(Template("my_kernel2").instantiate((int)C, Type<T>()))
          ->configure(grid, block)
          ->launch(indata, outdata),
      "");
  ASSERT_EQ(
      program
          ->get_kernel(
              Template("my_kernel2").instantiate((int)C, type_of(*indata)))
          ->configure(grid, block)
          ->launch(indata, outdata),
      "");
  ASSERT_EQ(
      program
          ->get_kernel(
              Template("my_kernel2").instantiate((int)C, instance_of(*indata)))
          ->configure(grid, block)
          ->launch(indata, outdata),
      "");

  T outval = 0;
  CHECK_CUDART(cudaMemcpy(&outval, outdata, sizeof(T), cudaMemcpyDeviceToHost));
  CHECK_CUDART(cudaFree(outdata));
  CHECK_CUDART(cudaFree(indata));

  EXPECT_FLOAT_EQ(inval, outval);
}

TEST(Jitify2Test, LaunchLatencyBenchmark) {
  static const char* const source = R"(
template <int N, int M, typename T, typename U>
__global__ void my_kernel(const T*, U*) {}
)";
  const size_t max_size = 2;
  // Note: It's faster (by ~300ns) to use custom keys, but we want to test
  // worst-case perf.
  ProgramCache<> cache(max_size, *Program("my_program", source)->preprocess(),
                       nullptr);
  float* idata = nullptr;
  uint8_t* odata = nullptr;
  dim3 grid(1), block(1);
  Kernel kernel = cache.get_kernel(
      Template("my_kernel")
          .instantiate(3, 4, type_of(*idata), type_of(*odata)));
  ASSERT_EQ(kernel->configure(grid, block)->launch(idata, odata), "");

  void* arg_ptrs[] = {&idata, &odata};

  int nrep = 10000;
  double dt_direct_ns = 1e99, dt_jitify_ns = 1e99;
  static const std::string kernel_inst =
      Template("my_kernel").instantiate(3, 4, type_of(*idata), type_of(*odata));
  for (int i = 0; i < nrep; ++i) {
    // Benchmark direct kernel launch.
    auto t0 = std::chrono::steady_clock::now();
    cuda().LaunchKernel()(kernel->function(), grid.x, grid.y, grid.z, block.x,
                          block.y, block.z, 0, 0, arg_ptrs, nullptr);
    auto dt = std::chrono::steady_clock::now() - t0;
    // Using the minimum is more robust than the average (though this test still
    // remains sensitive to the system environment and has been observed to fail
    // intermittently at a rate of <0.1%).
    dt_direct_ns = std::min(
        dt_direct_ns,
        (double)std::chrono::duration_cast<std::chrono::nanoseconds>(dt)
            .count());

    // Benchmark launch from cache.
    t0 = std::chrono::steady_clock::now();
    cache
        .get_kernel(
            // Note: It's faster to precompute this, but we want to test
            // worst-case perf.
            Template("my_kernel")
                .instantiate(3, 4, type_of(*idata), type_of(*odata)))
        ->configure(grid, block)
        ->launch(idata, odata);
    dt = std::chrono::steady_clock::now() - t0;
    dt_jitify_ns = std::min(
        dt_jitify_ns,
        (double)std::chrono::duration_cast<std::chrono::nanoseconds>(dt)
            .count());
  }
  double launch_time_direct_ns = dt_direct_ns;
  double launch_time_jitify_ns = dt_jitify_ns;
  // Ensure added latency is small.
  double tolerance_ns = 2500;  // 2.5us
  EXPECT_NEAR(launch_time_direct_ns, launch_time_jitify_ns, tolerance_ns);
}

class ScopeGuard {
  std::function<void()> func_;

 public:
  ScopeGuard(std::function<void()> func) : func_(std::move(func)) {}
  ~ScopeGuard() { func_(); }
  ScopeGuard(const ScopeGuard&) = delete;
  ScopeGuard& operator=(const ScopeGuard&) = delete;
  ScopeGuard(ScopeGuard&&) = delete;
  ScopeGuard& operator=(ScopeGuard&&) = delete;
};

inline bool remove_empty_dir(const char* path) {
#if defined(_WIN32) || defined(_WIN64)
  return ::_rmdir(path) == 0;
#else
  return ::rmdir(path) == 0;
#endif
}

TEST(Jitify2Test, ProgramCache) {
  static const char* const source = R"(
template <typename T>
__global__ void my_kernel(const T* __restrict__ idata, T* __restrict__ odata) {}
)";
  using key_type = uint32_t;
  size_t max_size = 2;
  static const char* const cache_path0 = "jitify2_test_cache";
  static const char* const cache_path = "jitify2_test_cache/subdir";
  ProgramCache<key_type> cache(max_size,
                               *Program("my_program", source)->preprocess(),
                               nullptr, cache_path);
  ScopeGuard scoped_cleanup_files([&] {
    cache.clear();
    remove_empty_dir(cache_path);
    remove_empty_dir(cache_path0);
  });

  auto check_hits = [&](size_t expected_hits, size_t expected_misses) {
    size_t num_hits, num_misses;
    cache.get_stats(&num_hits, &num_misses);
    EXPECT_EQ(num_hits, expected_hits);
    EXPECT_EQ(num_misses, expected_misses);
  };

  Kernel kernel;
  Template my_kernel("my_kernel");

  check_hits(0, 0);
  kernel = cache.get_kernel(/* key = */ 0, my_kernel.instantiate<float>());
  ASSERT_EQ(get_error(kernel), "");
  ASSERT_EQ(kernel->configure(1, 1)->launch(nullptr, nullptr), "");
  check_hits(0, 1);
  kernel = cache.get_kernel(/* key = */ 1, my_kernel.instantiate<double>());
  ASSERT_EQ(get_error(kernel), "");
  check_hits(0, 2);
  kernel = cache.get_kernel(/* key = */ 2, my_kernel.instantiate<int>());
  ASSERT_EQ(get_error(kernel), "");
  CUfunction function_int = kernel->function();
  check_hits(0, 3);
  cache.reset_stats();
  check_hits(0, 0);
  kernel = cache.get_kernel(/* key = */ 0, my_kernel.instantiate<float>());
  ASSERT_EQ(get_error(kernel), "");
  CUfunction function_float = kernel->function();
  check_hits(0, 1);
  kernel = cache.get_kernel(/* key = */ 2, my_kernel.instantiate<int>());
  ASSERT_EQ(get_error(kernel), "");
  EXPECT_EQ(kernel->function(), function_int);
  check_hits(1, 1);
  kernel = cache.get_kernel(/* key = */ 0, my_kernel.instantiate<float>());
  ASSERT_EQ(get_error(kernel), "");
  EXPECT_EQ(kernel->function(), function_float);
  check_hits(2, 1);
  LoadedProgram program =
      cache.get_program(/* key = */ 2, {my_kernel.instantiate<int>()});
  ASSERT_EQ(get_error(program), "");
  check_hits(3, 1);

  // Make sure cache dir was created.
  bool cache_path_is_dir;
  ASSERT_TRUE(jitify2::detail::path_exists(cache_path, &cache_path_is_dir));
  ASSERT_TRUE(cache_path_is_dir);
  // Make sure cache dir contains files.
  ASSERT_FALSE(remove_empty_dir(cache_path));
  // Now clear the cache.
  ASSERT_TRUE(cache.clear());
  EXPECT_EQ(cache.max_in_mem(), max_size);
  EXPECT_EQ(cache.max_files(), max_size);
  // Make sure cache dir still exists.
  ASSERT_TRUE(jitify2::detail::path_exists(cache_path, &cache_path_is_dir));
  ASSERT_TRUE(cache_path_is_dir);
  // Make sure cache dir is empty.
  ASSERT_TRUE(remove_empty_dir(cache_path));
  ASSERT_FALSE(jitify2::detail::path_exists(cache_path));

  max_size += 10;
  EXPECT_TRUE(cache.resize(max_size));
  EXPECT_EQ(cache.max_in_mem(), max_size);
  EXPECT_EQ(cache.max_files(), max_size);
  EXPECT_TRUE(cache.resize(max_size + 1, max_size + 2));
  EXPECT_EQ(cache.max_in_mem(), max_size + 1);
  EXPECT_EQ(cache.max_files(), max_size + 2);
}

TEST(Jitify2Test, ProgramCacheAutoKey) {
  static const char* const source = R"(
template <typename T>
__global__ void my_kernel(const T* __restrict__ idata, T* __restrict__ odata) {}
)";
  size_t max_size = 2;
  static const char* const cache_path0 = "jitify2_test_cache";
  static const char* const cache_path = "jitify2_test_cache/subdir";
  ProgramCache<> cache(max_size, *Program("my_program", source)->preprocess(),
                       nullptr, cache_path);
  ScopeGuard scoped_cleanup_files([&] {
    cache.clear();
    remove_empty_dir(cache_path);
    remove_empty_dir(cache_path0);
  });

  auto check_hits = [&](size_t expected_hits, size_t expected_misses) {
    size_t num_hits, num_misses;
    cache.get_stats(&num_hits, &num_misses);
    EXPECT_EQ(num_hits, expected_hits);
    EXPECT_EQ(num_misses, expected_misses);
  };

  Kernel kernel;
  Template my_kernel("my_kernel");

  check_hits(0, 0);
  kernel = cache.get_kernel(my_kernel.instantiate<float>());
  ASSERT_EQ(get_error(kernel), "");
  ASSERT_EQ(kernel->configure(1, 1)->launch(nullptr, nullptr), "");
  check_hits(0, 1);
  kernel = cache.get_kernel(my_kernel.instantiate<double>());
  ASSERT_EQ(get_error(kernel), "");
  check_hits(0, 2);
  kernel = cache.get_kernel(my_kernel.instantiate<int>());
  ASSERT_EQ(get_error(kernel), "");
  CUfunction function_int = kernel->function();
  check_hits(0, 3);
  cache.reset_stats();
  check_hits(0, 0);
  kernel = cache.get_kernel(my_kernel.instantiate<float>());
  ASSERT_EQ(get_error(kernel), "");
  CUfunction function_float = kernel->function();
  check_hits(0, 1);
  kernel = cache.get_kernel(my_kernel.instantiate<int>());
  ASSERT_EQ(get_error(kernel), "");
  EXPECT_EQ(kernel->function(), function_int);
  check_hits(1, 1);
  kernel = cache.get_kernel(my_kernel.instantiate<float>());
  ASSERT_EQ(get_error(kernel), "");
  EXPECT_EQ(kernel->function(), function_float);
  check_hits(2, 1);
  LoadedProgram program = cache.get_program({my_kernel.instantiate<int>()});
  ASSERT_EQ(get_error(program), "");
  check_hits(3, 1);

  // Make sure cache dir was created.
  bool cache_path_is_dir;
  ASSERT_TRUE(jitify2::detail::path_exists(cache_path, &cache_path_is_dir));
  ASSERT_TRUE(cache_path_is_dir);
  // Make sure cache dir contains files.
  ASSERT_FALSE(remove_empty_dir(cache_path));
  // Now clear the cache.
  ASSERT_TRUE(cache.clear());
  EXPECT_EQ(cache.max_in_mem(), max_size);
  EXPECT_EQ(cache.max_files(), max_size);
  // Make sure cache dir still exists.
  ASSERT_TRUE(jitify2::detail::path_exists(cache_path, &cache_path_is_dir));
  ASSERT_TRUE(cache_path_is_dir);
  // Make sure cache dir is empty.
  ASSERT_TRUE(remove_empty_dir(cache_path));
  ASSERT_FALSE(jitify2::detail::path_exists(cache_path));

  max_size += 10;
  EXPECT_TRUE(cache.resize(max_size));
  EXPECT_EQ(cache.max_in_mem(), max_size);
  EXPECT_EQ(cache.max_files(), max_size);
  EXPECT_TRUE(cache.resize(max_size + 1, max_size + 2));
  EXPECT_EQ(cache.max_in_mem(), max_size + 1);
  EXPECT_EQ(cache.max_files(), max_size + 2);
}

TEST(Jitify2Test, ProgramCacheFilenameSanitization) {
  static const char* const source = R"(__global__ void my_kernel() {})";
  const size_t max_size = 1;
  static const char* const cache_path = "jitify2_test_cache";
  // The filename is derived from the program name, so this checks that invalid
  // filename characters are automatically sanitized.
  ProgramCache<> cache(
      max_size, *Program("foo/bar/cat/dog\\:*?|<>", source)->preprocess(),
      nullptr, cache_path);
  ScopeGuard scoped_cleanup_files([&] {
    cache.clear();
    remove_empty_dir(cache_path);
  });
  *cache.get_kernel("my_kernel");
}

TEST(Jitify2Test, OfflinePreprocessing) {
  static const char* const extra_header_source = R"(
#pragma once
template <typename T>
T pointless_func(T x) {
  return x;
};)";
  size_t max_size = 10;
  // These variables come from the header generated by jitify_preprocess.
  ProgramCache<> cache(max_size, *jitify2_test_kernels_cu_jit,
                       jitify2_test_kernels_cu_headers_jit);
  enum { C = 123 };
  typedef float T;
  std::string kernel2_inst =
      Template("my_kernel2").instantiate<NonType<int, C>, T>();
  StringMap extra_headers = {{"my_header4.cuh", extra_header_source}};
  LoadedProgram program = cache.get_program(
      {"my_kernel1", kernel2_inst}, extra_headers, {"-include=my_header4.cuh"});
  ASSERT_EQ(get_error(program), "");

  T* indata;
  T* outdata;
  CHECK_CUDART(cudaMalloc((void**)&indata, sizeof(T)));
  CHECK_CUDART(cudaMalloc((void**)&outdata, sizeof(T)));
  T inval = 3.14159f;
  CHECK_CUDART(cudaMemcpy(indata, &inval, sizeof(T), cudaMemcpyHostToDevice));

  dim3 grid(1), block(1);
  ASSERT_EQ(program->get_kernel("my_kernel1")
                ->configure(grid, block)
                ->launch(indata, outdata),
            "");
  ASSERT_EQ(program->get_kernel(kernel2_inst)
                ->configure(grid, block)
                ->launch(indata, outdata),
            "");

  T outval = 0;
  CHECK_CUDART(cudaMemcpy(&outval, outdata, sizeof(T), cudaMemcpyDeviceToHost));
  CHECK_CUDART(cudaFree(outdata));
  CHECK_CUDART(cudaFree(indata));

  EXPECT_FLOAT_EQ(inval, outval);
}

TEST(Jitify2Test, Sha256) {
  EXPECT_EQ(jitify2::detail::sha256(""),
            "E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855");
  EXPECT_EQ(jitify2::detail::sha256(std::string(1, '\0')),
            "6E340B9CFFB37A989CA544E6BB780A2C78901D3FB33738768511A30617AFA01D");
  EXPECT_EQ(jitify2::detail::sha256("a"),
            "CA978112CA1BBDCAFAC231B39A23DC4DA786EFF8147C4E72B9807785AFEE48BB");
  EXPECT_EQ(jitify2::detail::sha256("abc"),
            "BA7816BF8F01CFEA414140DE5DAE2223B00361A396177A9CB410FF61F20015AD");
  EXPECT_EQ(
      jitify2::detail::sha256("The quick brown fox jumps over the lazy dog."),
      "EF537F25C895BFA782526529A9B63D97AA631564D5D789C2B765448C8635FB6C");
  EXPECT_EQ(
      jitify2::detail::sha256("The quick brown fox jumps over the lazy dog."
                              "The quick brown fox jumps over the lazy dog."
                              "The quick brown fox jumps over the lazy dog."
                              "The quick brown fox jumps over the lazy dog."),
      "F5EA20F5EDD6871D72D699C143C524BF9CEC13D06E9FA5763614EE3BA708C63E");
}

TEST(Jitify2Test, PathBase) {
  EXPECT_EQ(jitify2::detail::path_base("foo/bar/2"), "foo/bar");
  EXPECT_EQ(jitify2::detail::path_base("foo/bar/2/"), "foo/bar/2");
  EXPECT_EQ(jitify2::detail::path_base("foo"), "");
  EXPECT_EQ(jitify2::detail::path_base("/"), "");
#if defined _WIN32 || defined _WIN64
  EXPECT_EQ(jitify2::detail::path_base("foo\\bar\\2"), "foo\\bar");
  EXPECT_EQ(jitify2::detail::path_base("foo\\bar\\2\\"), "foo\\bar\\2");
  EXPECT_EQ(jitify2::detail::path_base("foo"), "");
  EXPECT_EQ(jitify2::detail::path_base("\\"), "");
#endif
}

TEST(Jitify2Test, PathJoin) {
  EXPECT_EQ(jitify2::detail::path_join("foo/bar", "2/1"), "foo/bar/2/1");
  EXPECT_EQ(jitify2::detail::path_join("foo/bar/", "2/1"), "foo/bar/2/1");
  EXPECT_EQ(jitify2::detail::path_join("foo/bar", "/2/1"), "");
#if defined _WIN32 || defined _WIN64
  EXPECT_EQ(jitify2::detail::path_join("foo\\bar", "2\\1"), "foo\\bar/2\\1");
  EXPECT_EQ(jitify2::detail::path_join("foo\\bar\\", "2\\1"), "foo\\bar\\2\\1");
  EXPECT_EQ(jitify2::detail::path_join("foo\\bar", "\\2\\1"), "");
#endif
}

TEST(Jitify2Test, PathSimplify) {
  EXPECT_EQ(jitify2::detail::path_simplify(""), "");
  EXPECT_EQ(jitify2::detail::path_simplify("/"), "/");
  EXPECT_EQ(jitify2::detail::path_simplify("//"), "/");
  EXPECT_EQ(jitify2::detail::path_simplify("/foo/bar"), "/foo/bar");
  EXPECT_EQ(jitify2::detail::path_simplify("foo/bar"), "foo/bar");
  EXPECT_EQ(jitify2::detail::path_simplify("/foo/./bar"), "/foo/bar");
  EXPECT_EQ(jitify2::detail::path_simplify("foo/./bar"), "foo/bar");
  EXPECT_EQ(jitify2::detail::path_simplify("/foo/../bar"), "/bar");
  EXPECT_EQ(jitify2::detail::path_simplify("foo/../bar"), "bar");
  EXPECT_EQ(jitify2::detail::path_simplify("/foo/cat/../../bar"), "/bar");
  EXPECT_EQ(jitify2::detail::path_simplify("foo/cat/../../bar"), "bar");
  EXPECT_EQ(jitify2::detail::path_simplify("/./bar"), "/bar");
  EXPECT_EQ(jitify2::detail::path_simplify("./bar"), "bar");
  EXPECT_EQ(jitify2::detail::path_simplify("../bar"), "../bar");
  EXPECT_EQ(jitify2::detail::path_simplify("../../bar"), "../../bar");
  EXPECT_EQ(jitify2::detail::path_simplify("../.././bar"), "../../bar");
  EXPECT_EQ(jitify2::detail::path_simplify(".././../bar"), "../../bar");
  EXPECT_EQ(jitify2::detail::path_simplify("./../../bar"), "../../bar");
  EXPECT_EQ(jitify2::detail::path_simplify("/foo/bar/.."), "/foo");
  EXPECT_EQ(jitify2::detail::path_simplify("foo/bar/.."), "foo");
  EXPECT_EQ(jitify2::detail::path_simplify("//foo///..////bar"), "/bar");
  EXPECT_EQ(jitify2::detail::path_simplify("foo/"), "foo/");
  EXPECT_EQ(jitify2::detail::path_simplify("/foo/"), "/foo/");
  EXPECT_EQ(jitify2::detail::path_simplify("foo/bar/"), "foo/bar/");
  EXPECT_EQ(jitify2::detail::path_simplify("/foo/bar/"), "/foo/bar/");
  EXPECT_EQ(jitify2::detail::path_simplify("foo/../bar/"), "bar/");
  EXPECT_EQ(jitify2::detail::path_simplify("/foo/../bar/"), "/bar/");
  EXPECT_EQ(jitify2::detail::path_simplify("/../foo"), "");    // Invalid path
  EXPECT_EQ(jitify2::detail::path_simplify("/foo/../../bar"),  // Invalid path
            "");
  EXPECT_EQ(jitify2::detail::path_simplify("/.."), "");         // Invalid path
  EXPECT_EQ(jitify2::detail::path_simplify("/foo/../.."), "");  // Invalid path
#if defined _WIN32 || defined _WIN64
  EXPECT_EQ(jitify2::detail::path_simplify(R"(\)"), R"(\)");
  EXPECT_EQ(jitify2::detail::path_simplify(R"(\\)"), R"(\)");
  EXPECT_EQ(jitify2::detail::path_simplify(R"(\foo\bar)"), R"(\foo\bar)");
  EXPECT_EQ(jitify2::detail::path_simplify(R"(foo\bar)"), R"(foo\bar)");
  EXPECT_EQ(jitify2::detail::path_simplify(R"(\foo\.\bar)"), R"(\foo\bar)");
  EXPECT_EQ(jitify2::detail::path_simplify(R"(foo\.\bar)"), R"(foo\bar)");
  EXPECT_EQ(jitify2::detail::path_simplify(R"(\foo\..\bar)"), R"(\bar)");
  EXPECT_EQ(jitify2::detail::path_simplify(R"(foo\..\bar)"), R"(bar)");

  EXPECT_EQ(jitify2::detail::path_simplify(R"(\foo/.\bar)"), R"(\foo/bar)");
  EXPECT_EQ(jitify2::detail::path_simplify(R"(\foo/.\bar\./cat)"),
            R"(\foo/bar\cat)");
  EXPECT_EQ(jitify2::detail::path_simplify(R"(\foo/.\bar\../cat)"),
            R"(\foo/cat)");
#endif
}

TEST(Jitify2Test, Program) {
  static const char* const name = "my_program";
  static const char* const source = "/* empty source */";
  static const char* const header_name = "my_header";
  static const char* const header_source = "/* empty header */";
  Program program;
  ASSERT_EQ(static_cast<bool>(program), false);
  EXPECT_EQ(program.error(), "Uninitialized");
  EXPECT_THROW(*program, std::runtime_error);
  program = Program(name, source, {{header_name, header_source}});
  ASSERT_EQ(get_error(program), "");
  EXPECT_THROW(program.error(), std::runtime_error);
  EXPECT_EQ(program->name(), name);
  EXPECT_EQ(program->source(), source);
  EXPECT_EQ(program->header_sources().size(), size_t(1));
  ASSERT_EQ(program->header_sources().count(header_name), size_t(1));
  EXPECT_EQ(program->header_sources().at(header_name), header_source);
}

bool contains(const std::string& src, const std::string& target,
              const char* varname) {
  bool result = src.find(target) != std::string::npos;
  if (!result) {
    std::cerr << "--- BEGIN STRING " << varname << " ---\n"
              << src << "\n--- END STRING " << varname << " ---" << std::endl;
  }
  return result;
}

TEST(Jitify2Test, PreprocessedProgram) {
  // Tests source patching, header extraction, use of builtin headers, and basic
  // PreprocessedProgram API functionality.
  static const char* const name = "my_program";
  static const char* const source = R"(
#include <my_header1.cuh>
__global__ void my_kernel() {}
)";
  static const char* const header_name = "my_header1.cuh";

  Program empty_program(name, "");
  ASSERT_EQ(get_error(empty_program), "");
  PreprocessedProgram empty_preprog =
      empty_program->preprocess({"-no-preinclude-workarounds"});
  ASSERT_TRUE(static_cast<bool>(empty_preprog));

  Program program(name, source);
  ASSERT_EQ(get_error(program), "");
  PreprocessedProgram preprog = program->preprocess();
  ASSERT_EQ(static_cast<bool>(preprog), false);
  EXPECT_TRUE(CONTAINS(preprog.error(), "could not open source file"));
  preprog = program->preprocess({"-Iexample_headers"}, {"-lfoo"});
  ASSERT_EQ(get_error(preprog), "");
  EXPECT_EQ(preprog->name(), name);
  EXPECT_EQ(preprog->header_sources().count(header_name), size_t(1));
  EXPECT_TRUE(
      NOT_CONTAINS(preprog->remaining_compiler_options(), "-Iexample_headers"));
  EXPECT_EQ(preprog->remaining_linker_options(), StringVec({"-lfoo"}));
  EXPECT_NE(preprog->header_log(), "");
  EXPECT_EQ(preprog->compile_log(), "");

  // Ensure that the --disable-warnings flag doesn't break things.
  preprog = program->preprocess(
      {"-Iexample_headers", "--disable-warnings", "-w"}, {"-lfoo"});
  ASSERT_EQ(get_error(preprog), "");
  CompiledProgram compiled = preprog->compile();
  ASSERT_EQ(get_error(compiled), "");
}

TEST(Jitify2Test, ExplicitHeaderSources) {
  // This test checks how the keys in a user-provided header_sources map are
  // matched to #include directives in the source.
  static const std::string good_header = R"()";
  static const std::string bad_header =
      R"(#error TEST FAIL: WRONG HEADER FOUND)";
  static const std::string angle_header = R"(#include <bar>)";
  static const std::string quote_header = R"(#include "bar")";
  static const std::string quote_abs_header = R"(#include "/bar")";
  PreprocessedProgram preprog;
  preprog = Program("my_program", R"(#include <foo>)", {{"foo", good_header}})
                ->preprocess();
  ASSERT_EQ(get_error(preprog), "");
  preprog =
      Program("my_program", R"(#include <foo/bar>)", {{"foo/bar", good_header}})
          ->preprocess();
  ASSERT_EQ(get_error(preprog), "");
  preprog = Program("my_program", R"(#include <foo/angle>)",
                    {{"foo/angle", angle_header},
                     {"bar", good_header},
                     {"foo/bar", bad_header}})
                ->preprocess();
  ASSERT_EQ(get_error(preprog), "");
  preprog = Program("my_program", R"(#include "foo")",
                    {{"foo", good_header}, {"/foo", bad_header}})
                ->preprocess();
  ASSERT_EQ(get_error(preprog), "");
  preprog = Program("my_program", R"(#include "foo/bar")",
                    {{"foo/bar", good_header},
                     {"/foo/bar", bad_header},
                     {"bar", bad_header}})
                ->preprocess();
  ASSERT_EQ(get_error(preprog), "");
  preprog = Program("my_program", R"(#include <foo/quote>)",
                    {{"foo/quote", quote_header},
                     {"foo/bar", good_header},
                     {"bar", bad_header},
                     {"/foo/bar", bad_header}})
                ->preprocess();
  ASSERT_EQ(get_error(preprog), "");
  // Relative include takes precedence over -I path.
  preprog = Program("my_program", R"(#include <foo/quote>)",
                    {{"foo/quote", quote_header},
                     {"foo/bar", good_header},
                     {"bar", bad_header},
                     {"/foo/bar", bad_header}})
                ->preprocess({"-I."});
  ASSERT_EQ(get_error(preprog), "");
  // Finding a header at the root from a quote-include in a subdir requires
  // explicitly passing the current dir as an include path ("-I.").
  preprog = Program("my_program", R"(#include <foo/quote>)",
                    {{"foo/quote", quote_header},
                     {"bar", good_header},
                     {"/foo/bar", bad_header}})
                ->preprocess({"-I."});
  ASSERT_EQ(get_error(preprog), "");
  preprog = Program("my_program", R"(#include </foo/bar>)",
                    {{"/foo/bar", good_header},
                     {"foo/bar", bad_header},
                     {"bar", bad_header}})
                ->preprocess();
  ASSERT_EQ(get_error(preprog), "");
  preprog = Program("my_program", R"(#include "/foo/bar")",
                    {{"/foo/bar", good_header},
                     {"foo/bar", bad_header},
                     {"bar", bad_header}})
                ->preprocess();
  ASSERT_EQ(get_error(preprog), "");
  preprog = Program("my_program", R"(#include <foo/quote_abs>)",
                    {{"foo/quote_abs", quote_abs_header},
                     {"/bar", good_header},
                     {"bar", bad_header},
                     {"/foo/bar", bad_header},
                     {"foo/bar", bad_header}})
                ->preprocess();
  ASSERT_EQ(get_error(preprog), "");
}

TEST(Jitify2Test, Preincludes) {
  // This tests that preincludes get preprocessed and that absolute paths are
  // handled correctly. Note that cuda.h includes <stdlib.h>, so it must be
  // preprocessed by Jitify (not simply loaded directly by NVRTC) to work.
  static const std::string source = R"(
#ifndef CUDA_VERSION
#error TEST FAILED
#endif
)";
  PreprocessedProgram preprog;
  preprog = Program("my_program", source)
                ->preprocess({"--pre-include=" CUDA_INC_DIR "/cuda.h"});
  ASSERT_EQ(get_error(preprog), "");
  preprog = Program("my_program", source)
                ->preprocess({"-include=" CUDA_INC_DIR "/cuda.h"});
  ASSERT_EQ(get_error(preprog), "");
}

TEST(Jitify2Test, CurrentExeIncludePath) {
  static const std::string source = R"(
#include <example_headers/my_header1.cuh>
)";
  std::unique_ptr<const char, int (*)(const char*)> cd_back("..", ::chdir);
  ASSERT_EQ(::chdir("example_headers"), 0);
  // This requires -I. to be expanded to the current executable directory, not
  // the current working directory.
  PreprocessedProgram preprog =
      Program("my_program", source)->preprocess({"-I."});
  ASSERT_EQ(get_error(preprog), "");
  ASSERT_EQ(get_error(preprog->compile()), "");
}

TEST(Jitify2Test, CompiledProgram) {
  // Tests compilation, lowered name lookup, and basic CompiledProgram API
  // functionality.
  static const char* const name = "my_program";
  static const char* const source = R"(
template <typename T>
__global__ void my_kernel() {}
)";
  static const char* const instantiation = "my_kernel<float>";
  static const char* const lowered_name = "_Z9my_kernelIfEvv";
  Program program(name, source);
  ASSERT_EQ(get_error(program), "");
  PreprocessedProgram preprog = program->preprocess();
  ASSERT_EQ(get_error(preprog), "");
  // TODO: Check that --remove-unused-globals is still needed.
  // Note: "--remove-unused-globals" is needed to WAR an issue in CUDA 12.0.
  CompiledProgram compiled = preprog->compile(
      instantiation, {}, {"--remove-unused-globals"}, {"-lfoo"});
  ASSERT_EQ(get_error(compiled), "");
  EXPECT_NE(compiled->ptx(), "");
  EXPECT_EQ(compiled->lowered_name_map().size(), size_t(1));
  ASSERT_EQ(compiled->lowered_name_map().count(instantiation), size_t(1));
  EXPECT_EQ(compiled->lowered_name_map().at(instantiation), lowered_name);
  std::vector<std::string> serialized_linker_options =
      compiled->remaining_linker_options().serialize();
  std::unordered_multiset<std::string> linker_options;
  linker_options.insert(serialized_linker_options.begin(),
                        serialized_linker_options.end());
  EXPECT_EQ(linker_options.count("-lfoo"), 1);
  EXPECT_EQ(compiled->log(), "");
}

TEST(Jitify2Test, ConstantMemory) {
  static const char* const source = R"(
__constant__ int a;
__device__ int d;
namespace b { __constant__ int a; __device__ int d; }
namespace c { namespace b { __constant__ int a; __device__ int d; } }
namespace x { __constant__ int a = 3; __device__ int d = 7; }
namespace y { __constant__ int a[] = {4, 5}; __device__ int d[] = {8, 9}; }
namespace z { template <typename T> __constant__ T tv = 10; }

__global__ void constant_test(int* x) {
  x[0] = a;
  x[1] = b::a;
  x[2] = c::b::a;
  x[3] = d;
  x[4] = b::d;
  x[5] = c::b::d;
  x[6] = x::a;
  x[7] = x::d;
  x[8] = y::a[0];
  x[9] = y::a[1];
  x[10] = y::d[0];
  x[11] = y::d[1];
})";

  dim3 grid(1), block(1);
  {  // Test __constant__ look up in kernel using different namespaces.
    Kernel kernel = Program("constmem_program", source)
                        ->preprocess({"-std=c++14"})
                        // TODO: Use z::tv<float> in tests below.
                        ->get_kernel("constant_test", {"&z::tv<float>"});
    const LoadedProgramData& program = kernel->program();
    int dval;
    ASSERT_EQ(program.get_global_value("x::a", &dval), "");
    EXPECT_EQ(dval, 3);
    ASSERT_EQ(program.get_global_value("x::d", &dval), "");
    EXPECT_EQ(dval, 7);
    int darr[2];
    ASSERT_EQ(program.get_global_data("y::a", &darr[0], 2), "");
    EXPECT_EQ(darr[0], 4);
    EXPECT_EQ(darr[1], 5);
    ASSERT_EQ(program.get_global_value("y::d", &darr), "");
    EXPECT_EQ(darr[0], 8);
    EXPECT_EQ(darr[1], 9);
    int inval[] = {2, 4, 8, 12, 14, 18, 22, 26, 30, 34, 38, 42};
    constexpr int n_const = sizeof(inval) / sizeof(int);
    ASSERT_EQ(program.set_global_value("a", inval[0]), "");
    ASSERT_EQ(program.set_global_value("b::a", inval[1]), "");
    ASSERT_EQ(program.set_global_value("c::b::a", inval[2]), "");
    ASSERT_EQ(program.set_global_value("d", inval[3]), "");
    ASSERT_EQ(program.set_global_value("b::d", inval[4]), "");
    ASSERT_EQ(program.set_global_value("c::b::d", inval[5]), "");
    ASSERT_EQ(program.set_global_value("x::a", inval[6]), "");
    ASSERT_EQ(program.set_global_value("x::d", inval[7]), "");
    ASSERT_EQ(program.set_global_data("y::a", &inval[8], 2), "");
    int inarr[] = {inval[10], inval[11]};
    ASSERT_EQ(program.set_global_value("y::d", inarr), "");
    int* outdata;
    CHECK_CUDART(cudaMalloc((void**)&outdata, n_const * sizeof(int)));
    ASSERT_EQ(kernel->configure(grid, block)->launch(outdata), "");
    CHECK_CUDART(cudaDeviceSynchronize());
    int outval[n_const];
    CHECK_CUDART(
        cudaMemcpy(outval, outdata, sizeof(outval), cudaMemcpyDeviceToHost));
    for (int i = 0; i < n_const; i++) {
      EXPECT_EQ(inval[i], outval[i]);
    }
    CHECK_CUDART(cudaFree(outdata));
  }
  {  // Test __constant__ array look up in header nested in both anonymous and
     // explicit namespace.
    static const char* const source2 =
        R"(#include "example_headers/constant_header.cuh")";
    Kernel kernel = Program("constmem_program2", source2)
                        ->preprocess()
                        ->get_kernel("constant_test2");
    const LoadedProgramData& program = kernel->program();
    int inval[] = {3, 5, 9, 13, 15, 19};
    constexpr int n_anon_const = sizeof(inval) / sizeof(int);
    std::string anon_prefix, anon_prefix2;
    if (jitify2::nvrtc().get_version() >= 11030) {
      // Internal linkage names changed in CUDA 11.3 (more robust mangling).
      anon_prefix = "constmem_program2::<unnamed>::";
      anon_prefix2 = "constmem_program2::(anonymous namespace)::";
    } else {
      anon_prefix = "<unnamed>::";
      anon_prefix2 = "(anonymous namespace)::";
    }
    ASSERT_EQ(program.set_global_data(anon_prefix + "b::a", inval, 3), "");
    ASSERT_EQ(program.set_global_data(anon_prefix + "b::d", inval + 3, 3), "");
    // Make sure alternative versions work too.
    ASSERT_EQ(program.set_global_data(anon_prefix2 + "b::a", inval, 3), "");
    ASSERT_EQ(program.set_global_data(anon_prefix2 + "b::d", inval + 3, 3), "");
    int* outdata;
    CHECK_CUDART(cudaMalloc((void**)&outdata, n_anon_const * sizeof(int)));
    ASSERT_EQ(kernel->configure(grid, block)->launch(outdata), "");
    CHECK_CUDART(cudaDeviceSynchronize());
    int outval[n_anon_const];
    CHECK_CUDART(
        cudaMemcpy(outval, outdata, sizeof(outval), cudaMemcpyDeviceToHost));
    for (int i = 0; i < n_anon_const; i++) {
      EXPECT_EQ(inval[i], outval[i]);
    }
    CHECK_CUDART(cudaFree(outdata));
  }
}

TEST(Jitify2Test, InvalidPrograms) {
  // OK.
  EXPECT_EQ(get_error(Program("empty_program", "")->preprocess()), "");
  // OK.
  EXPECT_EQ(
      get_error(Program("found_header", "#include <cstdio>")->preprocess()),
      "");
  // Not OK.
  EXPECT_NE(
      get_error(
          Program("missing_header", "#include <cantfindme>")->preprocess()),
      "");
  // Not OK.
  EXPECT_NE(get_error(Program("bad_program", "NOT CUDA C!")->preprocess()), "");
}

TEST(Jitify2Test, CompileLTO_IR) {
  static const char* const source = R"(
const int arch = __CUDA_ARCH__ / 10;
)";

  if (!jitify2::nvrtc().GetNVVM()) return;  // Skip if not supported
  CompiledProgram program = Program("lto_nvvm_program", source)
                                ->preprocess({"-rdc=true", "-dlto"})
                                ->compile("", {}, {"-arch=compute_."});
  EXPECT_EQ(program->ptx().size(), 0);
  EXPECT_EQ(program->cubin().size(), 0);
  EXPECT_GT(program->nvvm().size(), 0);
  EXPECT_EQ(program->nvvm().size(), program->lto_ir().size());
  int current_arch = get_current_device_arch();
  LinkedProgram linked_program = program->link();
  if (CUDA_VERSION < 11040) {
    ASSERT_FALSE(linked_program.ok());
    ASSERT_TRUE(jitify2::detail::startswith(linked_program.error(),
                                            "Linking LTO IR is not supported"));
  } else {
    int arch;
    ASSERT_EQ(program->link()->load()->get_global_value("arch", &arch), "");
    EXPECT_EQ(arch, current_arch);
  }
}

TEST(Jitify2Test, LinkMultiplePrograms) {
  static const char* const source1 = R"(
__constant__ int c = 5;
__device__ int d = 7;
__device__ int f(int i) { return i + 11; }
)";

  static const char* const source2 = R"(
extern __constant__ int c;
extern __device__ int d;
extern __device__ int f(int);
__global__ void my_kernel(int* data) {
  *data = f(*data + c + d);
}
)";

  CompiledProgram program1 = Program("linktest_program1", source1)
                                 ->preprocess({"-rdc=true"})
                                 ->compile();
  CompiledProgram program2 = Program("linktest_program2", source2)
                                 ->preprocess({"-rdc=true"})
                                 ->compile("my_kernel");
  // TODO: Consider allowing refs not ptrs for programs, and also addding a
  //         get_kernel() shortcut method to LinkedProgram.
  Kernel kernel = LinkedProgram::link({&program1, &program2})
                      ->load()
                      ->get_kernel("my_kernel");
  int* d_data;
  CHECK_CUDART(cudaMalloc((void**)&d_data, sizeof(int)));
  int h_data = 3;
  CHECK_CUDART(
      cudaMemcpy(d_data, &h_data, sizeof(int), cudaMemcpyHostToDevice));
  ASSERT_EQ(kernel->configure(1, 1)->launch(d_data), "");
  CHECK_CUDART(
      cudaMemcpy(&h_data, d_data, sizeof(int), cudaMemcpyDeviceToHost));
  EXPECT_EQ(h_data, 26);
  CHECK_CUDART(cudaFree(d_data));
}

TEST(Jitify2Test, LinkLTO) {
  static const char* const source1 = R"(
__constant__ int c = 5;
__device__ int d = 7;
extern "C"
__device__ int f(int i) { return i + 11; }
)";

  static const char* const source2 = R"(
extern __constant__ int c;
extern __device__ int d;
extern "C" __device__ int f(int);
__global__ void my_kernel(int* data) {
  *data = f(*data + c + d);
}
)";

  if (!jitify2::nvrtc().GetNVVM()) return;  // Skip if not supported

  // **TODO: Work out what code-type mixing is allowed when linking.
  CompiledProgram program1 = Program("linktest_program1", source1)
                                 ->preprocess({"-rdc=true", "-dlto"})
                                 ->compile("");
  CompiledProgram program2 = Program("linktest_program2", source2)
                                 ->preprocess({"-rdc=true", "-dlto"})
                                 ->compile("my_kernel");
  // TODO: Consider allowing refs not ptrs for programs, and also addding a
  //         get_kernel() shortcut method to LinkedProgram.
  LinkedProgram linked_program = LinkedProgram::link({&program1, &program2});
  if (CUDA_VERSION < 11040) {
    ASSERT_FALSE(linked_program.ok());
    ASSERT_TRUE(jitify2::detail::startswith(linked_program.error(),
                                            "Linking LTO IR is not supported"));
    return;
  }
  Kernel kernel = linked_program->load()->get_kernel("my_kernel");
  int* d_data;
  CHECK_CUDART(cudaMalloc((void**)&d_data, sizeof(int)));
  int h_data = 3;
  CHECK_CUDART(
      cudaMemcpy(d_data, &h_data, sizeof(int), cudaMemcpyHostToDevice));
  ASSERT_EQ(kernel->configure(1, 1)->launch(d_data), "");
  CHECK_CUDART(
      cudaMemcpy(&h_data, d_data, sizeof(int), cudaMemcpyDeviceToHost));
  EXPECT_EQ(h_data, 26);
  CHECK_CUDART(cudaFree(d_data));
}

TEST(Jitify2Test, LinkExternalFiles) {
  static const char* const source1 = R"(
__constant__ int c = 5;
__device__ int d = 7;
__device__ int f(int i) { return i + 11; })";

  static const char* const source2 = R"(
extern __constant__ int c;
extern __device__ int d;
extern __device__ int f(int);
__global__ void my_kernel(int* data) {
  *data = f(*data + c + d);
})";

  // Ensure temporary file is deleted at the end.
  std::unique_ptr<const char, int (*)(const char*)> ptx_filename(
      "example_headers/linktest.ptx", std::remove);
  {
    std::ofstream ptx_file(ptx_filename.get());
    ptx_file.exceptions(std::ofstream::failbit | std::ofstream::badbit);
    ptx_file << Program("linktest_program1", source1)
                    ->preprocess({"-rdc=true"})
                    ->compile()
                    ->ptx();
  }
  const std::vector<std::string> linker_options0 = {"-Lexample_headers",
                                                    "-llinktest.ptx"};
  for (bool use_culink : {false, true}) {
    std::vector<std::string> linker_options = linker_options0;
    if (use_culink) {
      linker_options.push_back("--use-culink");
    }
    Kernel kernel = Program("linktest_program2", source2)
                        ->preprocess({"-rdc=true"}, linker_options)
                        ->get_kernel("my_kernel");
    int* d_data;
    CHECK_CUDART(cudaMalloc((void**)&d_data, sizeof(int)));
    int h_data = 3;
    CHECK_CUDART(
        cudaMemcpy(d_data, &h_data, sizeof(int), cudaMemcpyHostToDevice));
    ASSERT_EQ(kernel->configure(1, 1)->launch(d_data), "");
    CHECK_CUDART(
        cudaMemcpy(&h_data, d_data, sizeof(int), cudaMemcpyDeviceToHost));
    EXPECT_EQ(h_data, 26);
    CHECK_CUDART(cudaFree(d_data));
  }
}

namespace a {
__host__ __device__ int external_device_func(int i) { return i + 1; }
}  // namespace a

TEST(Jitify2Test, LinkCurrentExecutable) {
  static const char* const source = R"(
namespace a {
extern __device__ int external_device_func(int);
}
__global__ void my_kernel(int* data) {
  *data = a::external_device_func(*data);
})";
  Kernel kernel = Program("selflink_program", source)
                      ->preprocess({"-rdc=true"}, {"-l."})
                      ->get_kernel("my_kernel");
  int* d_data;
  CHECK_CUDART(cudaMalloc((void**)&d_data, sizeof(int)));
  int h_data = 3;
  CHECK_CUDART(
      cudaMemcpy(d_data, &h_data, sizeof(int), cudaMemcpyHostToDevice));
  ASSERT_EQ(kernel->configure(1, 1)->launch(d_data), "");
  CHECK_CUDART(
      cudaMemcpy(&h_data, d_data, sizeof(int), cudaMemcpyDeviceToHost));
  EXPECT_EQ(h_data, 4);
  CHECK_CUDART(cudaFree(d_data));
}

TEST(Jitify2Test, ClassKernelArg) {
  static const char* const source = R"(
#include "example_headers/class_arg_kernel.cuh"
)";

  int h_data;
  int* d_data;
  CHECK_CUDART(cudaMalloc((void**)&d_data, sizeof(int)));

  PreprocessedProgram preprog =
      Program("class_kernel_arg_program", source)->preprocess();
  ConfiguredKernel configured_kernel =
      preprog->get_kernel(Template("class_arg_kernel").instantiate<Arg>())
          ->configure(1, 1);

  {  // Test that we can pass an arg object to a kernel.
    Arg arg(-1);
    ASSERT_EQ(configured_kernel->launch(d_data, arg), "");
    CHECK_CUDART(cudaDeviceSynchronize());
    CHECK_CUDART(
        cudaMemcpy(&h_data, d_data, sizeof(int), cudaMemcpyDeviceToHost));
    EXPECT_EQ(arg.x, h_data);
  }

  {  // Test that we can pass an arg object rvalue to a kernel.
    int value = -2;
    ASSERT_EQ(configured_kernel->launch(d_data, Arg(value)), "");
    CHECK_CUDART(cudaDeviceSynchronize());
    CHECK_CUDART(
        cudaMemcpy(&h_data, d_data, sizeof(int), cudaMemcpyDeviceToHost));
    EXPECT_EQ(value, h_data);
  }

  {  // Test that we can pass an arg object reference to a kernel.
    std::unique_ptr<Arg> arg(new Arg(-3));
    // References are passed as pointers since refernces are just pointers from
    // an ABI point of view.
    ASSERT_EQ(
        preprog->get_kernel(Template("class_arg_ref_kernel").instantiate<Arg>())
            ->configure(1, 1)
            ->launch(d_data, arg.get()),
        "");
    CHECK_CUDART(
        cudaMemcpy(&h_data, d_data, sizeof(int), cudaMemcpyDeviceToHost));
    EXPECT_EQ(arg->x, h_data);
  }

  {  // Test that we can pass an arg object reference to a kernel
    std::unique_ptr<Arg> arg(new Arg(-4));
    ASSERT_EQ(
        preprog->get_kernel(Template("class_arg_ptr_kernel").instantiate<Arg>())
            ->configure(1, 1)
            ->launch(d_data, arg.get()),
        "");
    CHECK_CUDART(
        cudaMemcpy(&h_data, d_data, sizeof(int), cudaMemcpyDeviceToHost));
    EXPECT_EQ(arg->x, h_data);
  }

  CHECK_CUDART(cudaFree(d_data));
}

TEST(Jitify2Test, GetAttribute) {
  static const char* const source = R"(
__global__ void get_attribute_kernel(int* out, const int* in) {
  __shared__ int buffer[4096];
  buffer[threadIdx.x] = in[threadIdx.x];
  __syncthreads();
  out[threadIdx.y] = buffer[threadIdx.x];
}
)";

  // Checks that we can get function attributes.
  int attrval;
  ASSERT_EQ(Program("get_attribute_program", source)
                ->preprocess()
                ->get_kernel("get_attribute_kernel")
                ->get_attribute(CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, &attrval),
            "");
  EXPECT_EQ(attrval, 4096 * (int)sizeof(int));
}

TEST(Jitify2Test, SetAttribute) {
  static const char* const source = R"(
__global__ void set_attribute_kernel(int* out, int* in) {
  extern __shared__ int buffer[];
  buffer[threadIdx.x] = in[threadIdx.x];
  __syncthreads();
  out[threadIdx.y] = buffer[threadIdx.x];
}
)";

  int* in;
  CHECK_CUDART(cudaMalloc((void**)&in, sizeof(int)));
  int* out;
  CHECK_CUDART(cudaMalloc((void**)&out, sizeof(int)));

  // Query the maximum supported shared bytes per block.
  CUdevice device;
  CHECK_CUDA(cuda().DeviceGet()(&device, 0));
  int shared_bytes;
  CHECK_CUDA(cuda().DeviceGetAttribute()(
      &shared_bytes, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
      device));

  Kernel kernel = Program("set_attribute_program", source)
                      ->preprocess()
                      ->get_kernel("set_attribute_kernel");
  ASSERT_EQ(kernel->set_attribute(
                CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared_bytes),
            "");

  dim3 grid(1), block(1);
  // This kernel will fail on Volta+ unless the set attribute succeeded.
  ASSERT_EQ(kernel->configure(grid, block, shared_bytes)->launch(out, in), "");

  CHECK_CUDART(cudaFree(out));
  CHECK_CUDART(cudaFree(in));
}

TEST(Jitify2Test, RemoveUnusedGlobals) {
  static const char* const source = R"(
struct Foo { static const int value = 7; };
struct Bar { int a; double b; };
__device__ float used_scalar;
__device__ float used_array[2];
__device__ Bar used_struct;
__device__ int used_scalar_init = 3;
__device__ int used_array_init[] = {4, 5};
__device__ Bar used_struct_init = {6, 0.0};
__device__ float unused_scalar;
__device__ float unused_array[3];
__device__ Bar unused_struct;
__device__ int unused_scalar_init = 3;
__device__ int unused_array_init[] = {4, 5};
__device__ Bar unused_struct_init = {6, 0.0};
__device__ float reg, ret, bra;  // Tricky name
__global__ void foo_kernel(int* data) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  used_scalar = 1.f;
  used_array[1] = 2.f;
  used_struct.b = 3.f;
  used_scalar_init = 1;
  used_array_init[1] = 2;
  used_struct_init.b = 3.f;
  __syncthreads();
  *data += Foo::value + used_scalar + used_array[1] + used_struct.b;
  // printf produces global symbols named $str.
  printf("printf test: *data = %i\n", *data);
})";
  CompiledProgram compiled =
      Program("unused_globals_source", source)
          ->preprocess(
              // Note: Flag added twice to test handling of repeats.
              {"-remove-unused-globals", "--remove-unused-globals"})
          ->compile("foo_kernel");
  const std::string& ptx = compiled->ptx();
  EXPECT_TRUE(ptx.find(".global .align 4 .f32 used_scalar;") !=
              std::string::npos);
  // Note: PTX represents arrays and structs as .b8 instead of the actual type.
  EXPECT_TRUE(ptx.find(".global .align 4 .b8 used_array[8];") !=
              std::string::npos);
  EXPECT_TRUE(ptx.find(".global .align 8 .b8 used_struct[16];") !=
              std::string::npos);
  EXPECT_TRUE(ptx.find(".global .align 4 .u32 used_scalar_init = 3;") !=
              std::string::npos);
  // Note: Since CUDA 12.3, the array initialization values do not include
  // trailing zeros (e.g., "{6};" instead of "{6, 0, 0, 0, ...};".
  EXPECT_TRUE(
      ptx.find(".global .align 4 .b8 used_array_init[8] = {4, 0, 0, 0, 5") !=
      std::string::npos);
  EXPECT_TRUE(ptx.find(".global .align 8 .b8 used_struct_init[16] = {6") !=
              std::string::npos);
  EXPECT_FALSE(ptx.find("_ZN3Foo5valueE") != std::string::npos);
  EXPECT_FALSE(ptx.find("unused_scalar;") != std::string::npos);
  EXPECT_FALSE(ptx.find("unused_array;") != std::string::npos);
  EXPECT_FALSE(ptx.find("unused_struct;") != std::string::npos);
  EXPECT_FALSE(ptx.find("unused_scalar_init;") != std::string::npos);
  EXPECT_FALSE(ptx.find("unused_array_init;") != std::string::npos);
  EXPECT_FALSE(ptx.find("unused_struct_init;") != std::string::npos);
  EXPECT_FALSE(ptx.find(".global .align 4 .f32 reg;") != std::string::npos);
  EXPECT_FALSE(ptx.find(".global .align 4 .f32 ret;") != std::string::npos);
  EXPECT_FALSE(ptx.find(".global .align 4 .f32 bra;") != std::string::npos);
  int* d_data;
  CHECK_CUDART(cudaMalloc((void**)&d_data, sizeof(int)));
  int h_data = 3;
  CHECK_CUDART(
      cudaMemcpy(d_data, &h_data, sizeof(int), cudaMemcpyHostToDevice));
  // TODO: Should redirect stdout to avoid the printf message in the test log.
  ASSERT_EQ(compiled->link()
                ->load()
                ->get_kernel("foo_kernel")
                ->configure(1, 1)
                ->launch(d_data),
            "");
  CHECK_CUDART(
      cudaMemcpy(&h_data, d_data, sizeof(int), cudaMemcpyDeviceToHost));
  EXPECT_EQ(h_data, 16);
  CHECK_CUDART(cudaFree(d_data));
}

TEST(Jitify2Test, OptionsVec) {
  OptionsVec options0;
  EXPECT_TRUE(options0.ok());
  OptionsVec options1({Option("-arch", "sm_50"), Option("-G")});
  EXPECT_TRUE(options1.ok());
  StringVec options_sv({"-arch", "sm_50", "-G"});
  OptionsVec options2(options_sv);
  EXPECT_TRUE(options2.ok());
  OptionsVec options3({"-arch", "sm_50", "-G"});
  EXPECT_TRUE(options3.ok());

  OptionsVec options({"--gpu-architecture", "compute_50", "-arch", "sm_50",
                      "-maxrregcount=100", "-Ifoo", "-I=foo2", "--device-debug",
                      "-G", "--restrict", "-restrict", "-lbar", "-l=bar2",
                      "-lineinfo"});
  EXPECT_TRUE(options.ok());

  EXPECT_EQ(options.size(), 12);
  EXPECT_EQ(options.serialize(),
            StringVec({"--gpu-architecture", "compute_50", "-arch", "sm_50",
                       "-maxrregcount=100", "-Ifoo", "-I=foo2",
                       "--device-debug", "-G", "--restrict", "-restrict",
                       "-lbar", "-l=bar2", "-lineinfo"}));
  EXPECT_EQ(options.serialize_canonical(),
            StringVec({"--gpu-architecture=compute_50", "-arch=sm_50",
                       "-maxrregcount=100", "-I=foo", "-I=foo2",
                       "--device-debug", "-G", "--restrict", "-restrict",
                       "-l=bar", "-l=bar2", "-lineinfo"}));
  StringVec implicit_conversion = options;
  EXPECT_EQ(implicit_conversion, options.serialize());
  EXPECT_TRUE(options.pop({"--gpu-architecture", "-arch"}));
  EXPECT_EQ(options.serialize(),
            StringVec({"-maxrregcount=100", "-Ifoo", "-I=foo2",
                       "--device-debug", "-G", "--restrict", "-restrict",
                       "-lbar", "-l=bar2", "-lineinfo"}));
  options.push_back(Option("-foo=bar"));
  EXPECT_EQ(options.serialize(),
            StringVec({"-maxrregcount=100", "-Ifoo", "-I=foo2",
                       "--device-debug", "-G", "--restrict", "-restrict",
                       "-lbar", "-l=bar2", "-lineinfo", "-foo=bar"}));
  options.pop_back();
  EXPECT_EQ(options.serialize(),
            StringVec({"-maxrregcount=100", "-Ifoo", "-I=foo2",
                       "--device-debug", "-G", "--restrict", "-restrict",
                       "-lbar", "-l=bar2", "-lineinfo"}));
  options.erase(1);
  options.erase(1);
  EXPECT_EQ(
      options.serialize(),
      StringVec({"-maxrregcount=100", "--device-debug", "-G", "--restrict",
                 "-restrict", "-lbar", "-l=bar2", "-lineinfo"}));
  OptionsVec extra({"-Ifoo", "-I=foo2"});
  options.insert(options.begin() + 1, extra.begin(), extra.end());
  EXPECT_EQ(options.serialize(),
            StringVec({"-maxrregcount=100", "-Ifoo", "-I=foo2",
                       "--device-debug", "-G", "--restrict", "-restrict",
                       "-lbar", "-l=bar2", "-lineinfo"}));
  EXPECT_EQ(options[2], Option("-I", "foo2"));
  EXPECT_EQ(options, options);
  EXPECT_NE(options, extra);
  std::vector<int> inds = options.find({"--restrict", "-l"});
  EXPECT_EQ(inds, std::vector<int>({5, 7, 8}));
}

TEST(Jitify2Test, ArchFlags) {
  static const char* const source = R"(
const int arch = __CUDA_ARCH__ / 10;
)";
  int current_arch = get_current_device_arch();
  int arch;
  // Test default behavior (automatic architecture detection).
  PreprocessedProgram preprocessed =
      Program("arch_flags_program", source)->preprocess();
  CompiledProgram program = preprocessed->compile();
  // Expect virtual architecture (compile to PTX).
  ASSERT_GT(program->ptx().size(), 0);
  ASSERT_EQ(program->cubin().size(), 0);
  ASSERT_EQ(program->link()->load()->get_global_value("arch", &arch), "");
  EXPECT_EQ(arch, current_arch);

  // Test explicit virtual architecture (compile to PTX).
  // Note: PTX is forwards compatible.
  program = preprocessed->compile("", {}, {"-arch=compute_50"});
  ASSERT_GT(program->ptx().size(), 0);
  ASSERT_EQ(program->cubin().size(), 0);
  ASSERT_EQ(program->link()->load()->get_global_value("arch", &arch), "");
  EXPECT_EQ(arch, 50);

  auto expect_cubin_size_if_available = [](size_t cubin_size) {
    if (jitify2::nvrtc().GetCUBIN()) {
      EXPECT_GT(cubin_size, 0);
    } else {
      EXPECT_EQ(cubin_size, 0);
    }
  };

  // Test explicit real architecture (may compile directly to CUBIN).
  program = preprocessed->compile(
      "", {}, {"-arch", "sm_" + std::to_string(current_arch)});
  EXPECT_GT(program->ptx().size(), 0);
  expect_cubin_size_if_available(program->cubin().size());
  ASSERT_EQ(program->link()->load()->get_global_value("arch", &arch), "");
  EXPECT_EQ(arch, current_arch);

  // Test automatic virtual architecture (compile to PTX).
  program = preprocessed->compile("", {}, {"-arch", "compute_."});
  EXPECT_GT(program->ptx().size(), 0);
  EXPECT_EQ(program->cubin().size(), 0);
  ASSERT_EQ(program->link()->load()->get_global_value("arch", &arch), "");
  EXPECT_EQ(arch, current_arch);

  // Test automatic real architecture (may compile directly to CUBIN).
  program = preprocessed->compile("", {}, {"-arch=sm_."});
  EXPECT_GT(program->ptx().size(), 0);
  expect_cubin_size_if_available(program->cubin().size());
  ASSERT_EQ(program->link()->load()->get_global_value("arch", &arch), "");
  EXPECT_EQ(arch, current_arch);

  // Test that preprocessing and compilation use separate arch flags.
  program = Program("arch_flags_program", source)
                ->preprocess({"-arch=sm_50"})
                ->compile("", {}, {"-arch=sm_."});
  EXPECT_GT(program->ptx().size(), 0);
  expect_cubin_size_if_available(program->cubin().size());
  ASSERT_EQ(program->link()->load()->get_global_value("arch", &arch), "");
  EXPECT_EQ(arch, current_arch);

  // Test that multiple architectures can be specified for preprocessing.
  program = Program("arch_flags_program", source)
                ->preprocess({"-arch=compute_50", "-arch=compute_52",
                              "-arch=compute_61"})
                ->compile("", {}, {"-arch=compute_."});
  EXPECT_GT(program->ptx().size(), 0);
  EXPECT_EQ(program->cubin().size(), 0);
  ASSERT_EQ(get_error(program), "");

  // Test that certain compiler options are automatically passed to the linker.
  LinkedProgram linked =
      Program("arch_flags_program", source)
          ->preprocess({"-maxrregcount=100", "-lineinfo", "-G"})
          ->compile()
          ->link();
  ASSERT_EQ(get_error(linked), "");
  std::vector<std::string> serialized_linker_options =
      linked->linker_options().serialize();
  std::unordered_multiset<std::string> linker_options(
      serialized_linker_options.begin(), serialized_linker_options.end());
  ASSERT_EQ(linker_options.count("-maxrregcount=100"), 1);
  EXPECT_EQ(linker_options.count("--generate-line-info"), 1);
  EXPECT_EQ(linker_options.count("-G"), 1);

  // Test with different option formats.
  linked = Program("arch_flags_program", source)
               ->preprocess({"--maxrregcount", "100", "--generate-line-info",
                             "--device-debug"})
               ->compile()
               ->link();
  ASSERT_EQ(get_error(linked), "");
  serialized_linker_options = linked->linker_options().serialize_canonical();
  linker_options.clear();
  linker_options.insert(serialized_linker_options.begin(),
                        serialized_linker_options.end());
  EXPECT_EQ(linker_options.count("--maxrregcount=100"), 1);
  EXPECT_EQ(linker_options.count("--generate-line-info"), 1);
  EXPECT_EQ(linker_options.count("--device-debug"), 1);
}

struct Base {
  virtual ~Base() {}
};
template <typename T>
struct Derived : public Base {};

TEST(Jitify2Test, Reflection) {
  static const char* const source = R"(
struct Base { virtual ~Base() {} };
template <typename T>
struct Derived : public Base {};
template <typename T>
__global__ void type_kernel() {}
template <unsigned short N>
__global__ void nontype_kernel() {}
)";

  PreprocessedProgram preprog =
      Program("reflection_program", source)->preprocess();

  Template type_kernel("type_kernel");

#define JITIFY_TYPE_REFLECTION_TEST(T)                                   \
  EXPECT_EQ(                                                             \
      preprog->get_kernel(type_kernel.instantiate<T>())->lowered_name(), \
      preprog->get_kernel(type_kernel.instantiate({#T}))->lowered_name())

  JITIFY_TYPE_REFLECTION_TEST(const volatile float);
  JITIFY_TYPE_REFLECTION_TEST(const volatile float*);
  JITIFY_TYPE_REFLECTION_TEST(const volatile float&);
  JITIFY_TYPE_REFLECTION_TEST(Base * (const volatile float));
  JITIFY_TYPE_REFLECTION_TEST(const volatile float[4]);

#undef JITIFY_TYPE_REFLECTION_TEST

  typedef Derived<float> derived_type;
  const Base& base = derived_type();
  EXPECT_EQ(preprog->get_kernel(type_kernel.instantiate(instance_of(base)))
                ->lowered_name(),
            preprog->get_kernel(type_kernel.instantiate<derived_type>())
                ->lowered_name());

  Template nontype_kernel("nontype_kernel");

#define JITIFY_NONTYPE_REFLECTION_TEST(N)                                 \
  EXPECT_EQ(                                                              \
      preprog->get_kernel(nontype_kernel.instantiate(N))->lowered_name(), \
      preprog->get_kernel(nontype_kernel.instantiate({#N}))->lowered_name())

  JITIFY_NONTYPE_REFLECTION_TEST(7);
  JITIFY_NONTYPE_REFLECTION_TEST('J');

#undef JITIFY_NONTYPE_REFLECTION_TEST
}

TEST(Jitify2Test, BuiltinNumericLimitsHeader) {
  static const char* const source = R"(
#include <limits>
struct MyType {};
namespace std {
template<> class numeric_limits<MyType> {
 public:
  static MyType min() { return {}; }
  static MyType max() { return {}; }
};
}  // namespace std
template <typename T>
__global__ void my_kernel(T* data) {
  data[0] = std::numeric_limits<T>::min();
  data[1] = std::numeric_limits<T>::max();
}
)";
  PreprocessedProgram preprog =
      Program("builtin_numeric_limits_program", source)->preprocess();
  for (const auto& type :
       {"float", "double", "char", "signed char", "unsigned char", "short",
        "unsigned short", "int", "unsigned int", "long", "unsigned long",
        "long long", "unsigned long long", "MyType"}) {
    std::string kernel_inst = Template("my_kernel").instantiate(type);
    Kernel kernel =
        preprog->compile(kernel_inst)->link()->load()->get_kernel(kernel_inst);
    (void)kernel;
  }
}

TEST(Jitify2Test, CuRandKernel) {
  static const char* const source = R"(
#include <curand_kernel.h>
__global__ void my_kernel() {}
)";
  Kernel kernel =
      Program("curand_program", source)
          // Note: --remove-unused-globals is added to remove huge precomputed
          // arrays that come from CURAND.
          ->preprocess({"-I" CUDA_INC_DIR, "--remove-unused-globals"})
          ->get_kernel("my_kernel");
  // TODO: Expand this test to actually call curand kernels and check outputs.
  (void)kernel;
}

TEST(Jitify2Test, Thrust) {
  // clang-format off
  static const char* const source = R"(
// WAR for NVRTC issue causing compilation error:
//   `namespace "thrust" has no actual member "iterator_core_access"`.
#define THRUST_WRAPPED_NAMESPACE jitify_thrust_ns_war
#define THRUST_DISABLE_ABI_NAMESPACE

#include <thrust/iterator/counting_iterator.h>

using namespace jitify_thrust_ns_war;  // Part of WAR above

__global__ void my_kernel(thrust::counting_iterator<int> begin,
                          thrust::counting_iterator<int> end) {
})";
  // clang-format on
  // Checks that basic Thrust headers can be compiled.
#if CUDA_VERSION < 11000
  const char* cppstd = "-std=c++03";
#else
  const char* cppstd = "-std=c++14";
#endif
  PreprocessedProgram preprog = Program("thrust_program", source)
                                    ->preprocess({"-I" CUDA_INC_DIR, cppstd});
  ASSERT_EQ(get_error(preprog), "");
  ASSERT_EQ(get_error(preprog->compile()), "");
}

#if CUDA_VERSION >= 11000
TEST(Jitify2Test, CubBlockPrimitives) {
  static const char* const cub_program_source = R"(
// WAR for issue in CUB shipped with CUDA 11.4
// (https://github.com/NVIDIA/cub/issues/334)
// Note: We can't easily work around this inside Jitify itself.
// TODO(benbarsdell): Check exactly when this issue is fixed in CUB (<1.15.0?).
#include <cub/version.cuh>
#if CUB_VERSION >= 101200 && CUB_VERSION < 101500
#define ProcessFloatMinusZero BaseDigitExtractor<KeyT>::ProcessFloatMinusZero
#endif

// WAR for header include issue (note: order of includes matters):
//   https://github.com/NVIDIA/jitify/issues/107#issuecomment-1225617951
#include <cuda/std/cstdint>
#include <cuda/std/cstddef>
#include <cuda/std/type_traits>
#include <cuda/std/limits>

#include <cub/block/block_load.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_store.cuh>

template <int BLOCK_SIZE, int PER_THREAD>
__global__ void my_kernel(float* data) {
  typedef cub::BlockLoad<float, BLOCK_SIZE, PER_THREAD,
                         cub::BLOCK_LOAD_VECTORIZE>
      BlockLoad;
  typedef cub::BlockRadixSort<float, BLOCK_SIZE, PER_THREAD> BlockSort;
  typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
  typedef cub::BlockStore<float, BLOCK_SIZE, PER_THREAD,
                          cub::BLOCK_STORE_VECTORIZE>
      BlockStore;
  __shared__ union {
    typename BlockLoad::TempStorage load;
    typename BlockSort::TempStorage sort;
    typename BlockReduce::TempStorage reduce;
    typename BlockStore::TempStorage store;
    float sum;
  } temp_storage;
  float thread_data[PER_THREAD];
  BlockLoad(temp_storage.load).Load(data, thread_data);
  __syncthreads();
  BlockSort(temp_storage.sort).Sort(thread_data);
  __syncthreads();
  float sum = BlockReduce(temp_storage.reduce).Sum(thread_data);
  __syncthreads();
  if (threadIdx.x == 0) {
    temp_storage.sum = sum;
  }
  __syncthreads();
  sum = temp_storage.sum;
#pragma unroll
  for (int i = 0; i < PER_THREAD; ++i) {
    thread_data[i] *= 1.f / sum;
  }
  __syncthreads();
  BlockStore(temp_storage.store).Store(data, thread_data);
}
)";
  int block_size = 64;
  int per_thread = 4;
  int n = block_size * per_thread;
  std::vector<float> h_data(n);
  float sum = 0;
  for (int i = 0; i < n; ++i) {
    // Start with values sorted in reverse.
    h_data[i] = (float)(n - 1 - i);
    sum += h_data[i];
  }
  // Shuffle the values a bit.
  std::swap(h_data[3], h_data[7]);
  std::swap(h_data[10], h_data[20]);
  std::vector<float> h_expected(n);
  for (int i = 0; i < n; ++i) {
    // Expected sorted and normalized.
    h_expected[i] = (float)i / sum;
  }
  std::vector<float> h_result(n);
  float* d_data;
  CHECK_CUDART(cudaMalloc((void**)&d_data, n * sizeof(float)));
  CHECK_CUDART(cudaMemcpy(d_data, h_data.data(), n * sizeof(float),
                          cudaMemcpyHostToDevice));

  std::string kernel_inst =
      Template("my_kernel").instantiate(block_size, per_thread);
  Kernel kernel = Program("cub_program", cub_program_source)
                      ->preprocess({"-I" CUB_DIR, "-I" CUDA_INC_DIR})
                      ->compile(kernel_inst)
                      ->link()
                      ->load()
                      ->get_kernel(kernel_inst);
  kernel->configure(1, block_size)->launch(d_data);

  CHECK_CUDART(cudaMemcpy(h_result.data(), d_data, n * sizeof(float),
                          cudaMemcpyDeviceToHost));
  for (int i = 0; i < n; ++i) {
    EXPECT_FLOAT_EQ(h_result[i], h_expected[i]);
  }
  CHECK_CUDART(cudaFree(d_data));
}
#endif  // CUDA_VERSION >= 11000

#if CUDA_VERSION >= 11000
TEST(Jitify2Test, LibCudaCxx) {
  // Test that each libcudacxx header can be compiled on its own.
  for (const std::string header :
       {"atomic", "barrier", "cassert", "cfloat", "chrono", "climits",
        "cstddef", "cstdint", "ctime", "functional", "latch",
        /*"limits",*/ "ratio", "semaphore", "type_traits", "utility"}) {
    std::string source =
        "#include <cuda/std/" + header + ">\n__global__ void my_kernel() {}";
    // Note: The -arch flag here is required because "CUDA atomics are
    // only supported for sm_60 and up on *nix and sm_70 and up on
    // Windows."
    Program("libcudacxx_program", source)
        ->preprocess({"-I" CUDA_INC_DIR, "-arch=compute_70",
                      "-no-builtin-headers", "-no-preinclude-workarounds",
                      "-no-system-headers-workaround",
                      "-no-replace-pragma-once"})
        ->get_kernel("my_kernel");
  }
  // WAR for bug in cuda/std/limits that is missing include cuda/std/climits.
  static const char* const source = R"(
#include <cuda/std/climits>
#include <cuda/std/limits>
__global__ void my_kernel() {}
)";
  Program("libcudacxx_program", source)
      ->preprocess({"-I" CUDA_INC_DIR, "-arch=compute_70",
                    "-no-builtin-headers", "-no-preinclude-workarounds",
                    "-no-system-headers-workaround", "-no-replace-pragma-once"})
      ->get_kernel("my_kernel");
}

// See GitHub issue #107.
TEST(Jitify2Test, LibCudaCxxAndBuiltinLimits) {
  static const char* const source = R"(
#include <limits>
#include <cuda/std/limits>
)";

  PreprocessedProgram preprog =
    Program("limits_program", source)->preprocess({"-I" CUDA_INC_DIR});
  ASSERT_EQ(get_error(preprog), "");
  CompiledProgram compiled = preprog->compile();
  ASSERT_EQ(get_error(compiled), "");
}
#endif  // CUDA_VERSION >= 11000

TEST(Jitify2Test, AssertHeader) {
  static const char* const source = R"(
#include <cassert>
__global__ void my_assert_kernel() {
  assert(0 == 1);
}
)";
  // TODO: Should temporarily redirect stderr while executing this kernel and
  // check that the assertion message is printed (this will also avoid printing
  // the assertion message to the test log).
  // Checks that cassert works as expected.
  Program("assert_program", source)
      ->preprocess()
      ->get_kernel("my_assert_kernel")
      ->configure(1, 1)
      ->launch();
  ASSERT_EQ(cudaDeviceSynchronize(), cudaErrorAssert);
  // NOTE: Assertion failure is a sticky error in CUDA, so the process can no
  // longer be used for CUDA operations after this point.
}

TEST(Jitify2Test, Minify) {
  static const char* const name = "my_program";
  // This source is intentionally tricky to parse so that it stresses the
  // minification algorithm.
  static const std::string source = R"(
//#define FOO foo
//#define BAR(call)                             \
//  do {                                        \
//    call;                                     \
//  } while (0)

#ifndef __CUDACC_RTC__
    #define FOOBAR
    #define BARFOO
#else
    #define MY_CHAR_BIT 8
    #define __MY_CHAR_UNSIGNED__ ('\xff' > 0) // CURSED
    #if __MY_CHAR_UNSIGNED__
        #define MY_CHAR_MIN 0
        #define MY_CHAR_MAX UCHAR_MAX
    #else
        #define MY_CHAR_MIN SCHAR_MIN
        #define MY_CHAR_MAX SCHAR_MAX
    #endif
#endif
/*
This will
all be
"trickily"
removed
hopefully.*/

const char* const foo = R"foo(abc\def
ghi"')foo";  // )'

const char* const linecont_str = "line1 \
line2";
const char c = '\xff';

#if CUDA_VERSION >= 11000

// WAR for header include issue (note: order of includes matters):
//   https://github.com/NVIDIA/jitify/issues/107#issuecomment-1225617951
#include <cuda/std/cstdint>
#include <cuda/std/cstddef>
#include <cuda/std/type_traits>
#include <cuda/std/limits>

// CUB headers can be tricky to parse.
#include <cub/block/block_load.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_store.cuh>
#endif  // CUDA_VERSION >= 11000

#include <cuda.h>

  #include <iterator>  // Here's a comment
  #include <tuple>  // Here's another comment

#include "example_headers/my_header1.cuh"
__global__ void my_kernel() {}
)";
  PreprocessedProgram preprog =
      Program(name, source)->preprocess({"-I" CUB_DIR, "-I" CUDA_INC_DIR});
  ASSERT_EQ(get_error(preprog), "");
  CompiledProgram compiled = preprog->compile();
  ASSERT_EQ(get_error(compiled), "");
  std::string orig_ptx = compiled->ptx();

  preprog = Program(name, source)
                ->preprocess({"-I" CUB_DIR, "-I" CUDA_INC_DIR, "--minify"});
  ASSERT_EQ(get_error(preprog), "");
  EXPECT_LT(preprog->source().size(), source.size());
  compiled = preprog->compile();
  ASSERT_EQ(get_error(compiled), "");
  ASSERT_EQ(compiled->ptx(), orig_ptx);
}

bool read_binary_file(const char* filename, std::string* contents) {
  std::ifstream file(filename, std::ios::binary | std::ios::ate);
  if (!file) return false;
  const std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  contents->resize(size);
  return static_cast<bool>(file.read(&(*contents)[0], size));
}

// If kSerializationVersion hasn't changed, this checks that the existing golden
// files can still be deserialized correctly (i.e., that the serialization
// format wasn't inadvertently changed). If the version number has changed, this
// generates new golden files and adds them to git.
template <class JitifyObjectMaker>
void check_or_update_serialization_goldens(
    JitifyObjectMaker make_jitify_object) {
  using JitifyObject = typename std::result_of<JitifyObjectMaker()>::type;
  constexpr size_t version = jitify2::serialization::kSerializationVersion;
  std::string object_type_name = jitify2::reflection::reflect<JitifyObject>();
  // Remove namespace prefix from type name.
  object_type_name = object_type_name.substr(object_type_name.rfind("::") + 2);
  const std::string filename = "../serialization_goldens/" + object_type_name +
                               "." + std::to_string(version) + ".jitify";
  std::string serialized;
  if (read_binary_file(filename.c_str(), &serialized)) {
    JitifyObject object = JitifyObject::deserialize(serialized);
    ASSERT_TRUE(object) << "Failed to deserialize \"" << filename
                        << "\".\nIf you have changed any "
                           "serialization code, you need to bump "
                           "jitify2::serialization::kSerializationVersion.";
    ASSERT_EQ(make_jitify_object()->serialize(), serialized)
        << "Serialized object does not match golden in \"" << filename << "\"";
  } else {
    std::ofstream file(filename.c_str(), std::ios::binary);
    ASSERT_TRUE(file) << "Failed to open file for writing: " << filename;
    JitifyObject object = make_jitify_object();
    serialized = object->serialize();
    file.write(serialized.data(), serialized.size());
    ASSERT_TRUE(file) << "Failed to write to " << filename;
    file.close();
    std::cout << "Generated new golden file \"" << filename
              << "\". This should be committed alongside the changes to the "
                 "serialization format."
              << std::endl;
    if (std::system(("git add " + filename).c_str())) {
      std::cerr << "WARNING: Failed to `git add " << filename << "`"
                << std::endl;
    }
  }
}

TEST(Jitify2Test, SerializationGoldensProgram) {
  static const char* const source = R"(
#include "header1.cuh"
__global__ void kernel1(dtype*) {}
)";
  check_or_update_serialization_goldens([&] {
    return Program("program1", source,
                   {{"header1.cuh", "using dtype = float;"}});
  });
}

TEST(Jitify2Test, SerializationGoldensPreprocessedProgram) {
  static const char* const source = R"(
#include "header1.cuh"
__global__ void kernel1(dtype*) {}
)";
  check_or_update_serialization_goldens([&] {
    return PreprocessedProgram(PreprocessedProgramData(
        "program1", source, {{"header1.cuh", "using dtype = float;"}},
        {"-std=c++14"}, {"-lfoo"}, "header log", "compile log"));
  });
}

TEST(Jitify2Test, SerializationGoldensCompiledProgram) {
  check_or_update_serialization_goldens([&] {
    // Note: It doesn't matter what values we put in here, it's only testing the
    // de/serialization of them.
    return CompiledProgram(CompiledProgramData(
        "some ptx", "some cubin", "some nvvm", {{"name", "lowered_name"}},
        {"-lfoo"}, "log", {"-std=c++14"}));
  });
}

TEST(Jitify2Test, SerializationGoldensLinkedProgram) {
  check_or_update_serialization_goldens([&] {
    return LinkedProgram(LinkedProgramData(
        "some cubin", {{"name", "lowered_name"}}, "log", {"-lfoo"}));
  });
}

void expect_tokenization(const char* source,
                         std::vector<parser::Token::Type> expected_types,
                         std::vector<std::string> expected_token_strings) {
  using namespace jitify2::parser;
  CppLexer lexer(source);
  std::vector<Token> tokens;
  Token cur_token = lexer.next();
  while (cur_token) {
    tokens.push_back(cur_token);
    cur_token = lexer.next();
  }
  std::vector<Token::Type> types;
  std::vector<std::string> token_strings;
  types.reserve(tokens.size());
  token_strings.reserve(tokens.size());
  for (const Token& token : tokens) {
    types.push_back(token.type());
    token_strings.push_back(std::string(token.token_string()));
  }
  EXPECT_EQ(types, expected_types);
  EXPECT_EQ(token_strings, expected_token_strings);
}

TEST(Jitify2ParserTest, SingleTokens) {
  using namespace jitify2::parser;
  using Tt = Token::Type;
  expect_tokenization("(", {Tt::kLParen}, {"("});
  expect_tokenization(")", {Tt::kRParen}, {")"});
  expect_tokenization("[", {Tt::kLBracket}, {"["});
  expect_tokenization("<:", {Tt::kLBracket}, {"<:"});
  expect_tokenization("]", {Tt::kRBracket}, {"]"});
  expect_tokenization(":>", {Tt::kRBracket}, {":>"});
  expect_tokenization("{", {Tt::kLBrace}, {"{"});
  expect_tokenization("<%", {Tt::kLBrace}, {"<%"});
  expect_tokenization("}", {Tt::kRBrace}, {"}"});
  expect_tokenization("%>", {Tt::kRBrace}, {"%>"});
  expect_tokenization(".", {Tt::kDot}, {"."});
  expect_tokenization(".*", {Tt::kDotStar}, {".*"});
  expect_tokenization("->", {Tt::kArrow}, {"->"});
  expect_tokenization("->*", {Tt::kArrowStar}, {"->*"});
  expect_tokenization(",", {Tt::kComma}, {","});
  expect_tokenization("+", {Tt::kPlus}, {"+"});
  expect_tokenization("++", {Tt::kPlusPlus}, {"++"});
  expect_tokenization("+=", {Tt::kPlusEq}, {"+="});
  expect_tokenization("-", {Tt::kMinus}, {"-"});
  expect_tokenization("--", {Tt::kMinusMinus}, {"--"});
  expect_tokenization("-=", {Tt::kMinusEq}, {"-="});
  expect_tokenization("*", {Tt::kStar}, {"*"});
  expect_tokenization("*=", {Tt::kStarEq}, {"*="});
  expect_tokenization("/", {Tt::kSlash}, {"/"});
  expect_tokenization("/=", {Tt::kSlashEq}, {"/="});
  expect_tokenization("%", {Tt::kPercent}, {"%"});
  expect_tokenization("%=", {Tt::kPercentEq}, {"%="});
  expect_tokenization("?", {Tt::kQuestion}, {"?"});
  expect_tokenization(":", {Tt::kColon}, {":"});
  expect_tokenization("::", {Tt::kColonColon}, {"::"});
  expect_tokenization("&", {Tt::kAmp}, {"&"});
  expect_tokenization("&&", {Tt::kAmpAmp}, {"&&"});
  expect_tokenization("&=", {Tt::kAmpEq}, {"&="});
  expect_tokenization("|", {Tt::kBar}, {"|"});
  expect_tokenization("||", {Tt::kBarBar}, {"||"});
  expect_tokenization("|=", {Tt::kBarEq}, {"|="});
  expect_tokenization("^", {Tt::kCaret}, {"^"});
  expect_tokenization("^=", {Tt::kCaretEq}, {"^="});
  expect_tokenization("~", {Tt::kTilde}, {"~"});
  expect_tokenization("=", {Tt::kEq}, {"="});
  expect_tokenization("==", {Tt::kEqEq}, {"=="});
  expect_tokenization("!", {Tt::kBang}, {"!"});
  expect_tokenization("!=", {Tt::kBangEq}, {"!="});
  expect_tokenization("<", {Tt::kLt}, {"<"});
  expect_tokenization("<<", {Tt::kLtLt}, {"<<"});
  expect_tokenization("<=", {Tt::kLtEq}, {"<="});
  expect_tokenization("<<=", {Tt::kLtLtEq}, {"<<="});
  expect_tokenization(">", {Tt::kGt}, {">"});
  expect_tokenization(">>", {Tt::kGtGt}, {">>"});
  expect_tokenization(">=", {Tt::kGtEq}, {">="});
  expect_tokenization(">>=", {Tt::kGtGtEq}, {">>="});
  expect_tokenization("#", {Tt::kHash}, {"#"});
  expect_tokenization("%:", {Tt::kHash}, {"%:"});
  expect_tokenization("##", {Tt::kHashHash}, {"##"});
  expect_tokenization("%:%:", {Tt::kHashHash}, {"%:%:"});
  expect_tokenization(";", {Tt::kSemicolon}, {";"});
  expect_tokenization(" ", {Tt::kWhitespace}, {" "});
  expect_tokenization("\f", {Tt::kWhitespace}, {"\f"});
  expect_tokenization("\r", {Tt::kWhitespace}, {"\r"});
  expect_tokenization("\t", {Tt::kWhitespace}, {"\t"});
  expect_tokenization("\v", {Tt::kWhitespace}, {"\v"});
  expect_tokenization("\n", {Tt::kWhitespace}, {"\n"});
  expect_tokenization("0123", {Tt::kNumber}, {"0123"});
  expect_tokenization("123", {Tt::kNumber}, {"123"});
  expect_tokenization("0x1F", {Tt::kNumber}, {"0x1F"});
  expect_tokenization("0b10", {Tt::kNumber}, {"0b10"});
  expect_tokenization("123u", {Tt::kNumber}, {"123u"});
  expect_tokenization("123LLu", {Tt::kNumber}, {"123LLu"});
  expect_tokenization("123'45'6", {Tt::kNumber}, {"123'45'6"});
  expect_tokenization("0x12'34'56", {Tt::kNumber}, {"0x12'34'56"});
  expect_tokenization("\"str\"", {Tt::kString}, {"\"str\""});
  expect_tokenization("u\"str\"", {Tt::kString}, {"u\"str\""});
  expect_tokenization("u8\"str\"", {Tt::kString}, {"u8\"str\""});
  expect_tokenization("U\"str\"", {Tt::kString}, {"U\"str\""});
  expect_tokenization("L\"str\"", {Tt::kString}, {"L\"str\""});
  expect_tokenization(R"("a \n\"b\"")", {Tt::kString}, {R"("a \n\"b\"")"});
  expect_tokenization("R\"xx(str)xx\"", {Tt::kRawString}, {"R\"xx(str)xx\""});
  expect_tokenization("uR\"xx(str)xx\"", {Tt::kRawString}, {"uR\"xx(str)xx\""});
  expect_tokenization("u8R\"xx(str)xx\"", {Tt::kRawString},
                      {"u8R\"xx(str)xx\""});
  expect_tokenization("UR\"xx(str)xx\"", {Tt::kRawString}, {"UR\"xx(str)xx\""});
  expect_tokenization("LR\"xx(str)xx\"", {Tt::kRawString}, {"LR\"xx(str)xx\""});
  expect_tokenization(R"yy(R"xx(a\nb
\c\\")xx")yy",
                      {Tt::kRawString}, {R"yy(R"xx(a\nb
\c\\")xx")yy"});
  expect_tokenization("'c'", {Tt::kCharacter}, {"'c'"});
  expect_tokenization("u'c'", {Tt::kCharacter}, {"u'c'"});
  expect_tokenization("u8'c'", {Tt::kCharacter}, {"u8'c'"});
  expect_tokenization("U'c'", {Tt::kCharacter}, {"U'c'"});
  expect_tokenization("L'c'", {Tt::kCharacter}, {"L'c'"});
  expect_tokenization(R"('\'')", {Tt::kCharacter}, {R"('\'')"});
  expect_tokenization(R"('\\')", {Tt::kCharacter}, {R"('\\')"});
  expect_tokenization(R"('\n')", {Tt::kCharacter}, {R"('\n')"});
  expect_tokenization("abc_DEF1", {Tt::kIdentifier}, {"abc_DEF1"});
  expect_tokenization("u", {Tt::kIdentifier}, {"u"});
  expect_tokenization("u8", {Tt::kIdentifier}, {"u8"});
  expect_tokenization("U", {Tt::kIdentifier}, {"U"});
  expect_tokenization("L", {Tt::kIdentifier}, {"L"});
  expect_tokenization("uabc", {Tt::kIdentifier}, {"uabc"});
  expect_tokenization("u8abc", {Tt::kIdentifier}, {"u8abc"});
  expect_tokenization("Uabc", {Tt::kIdentifier}, {"Uabc"});
  expect_tokenization("Labc", {Tt::kIdentifier}, {"Labc"});
  expect_tokenization("class", {Tt::kKeyword}, {"class"});
  expect_tokenization("not", {Tt::kKeyword}, {"not"});
  expect_tokenization("consteval", {Tt::kKeyword}, {"consteval"});
  expect_tokenization("__device__", {Tt::kKeyword}, {"__device__"});
  expect_tokenization("__constant__", {Tt::kKeyword}, {"__constant__"});
  expect_tokenization("// A comment", {Tt::kComment}, {"// A comment"});
  expect_tokenization("// A \"comment\"", {Tt::kComment}, {"// A \"comment\""});
  expect_tokenization("/* A comment\n*/", {Tt::kComment}, {"/* A comment\n*/"});
  expect_tokenization("/* A \"comment\"\n*/", {Tt::kComment},
                      {"/* A \"comment\"\n*/"});
}

TEST(Jitify2ParserTest, MultipleTokens) {
  using namespace jitify2::parser;
  using Tt = Token::Type;
  // Make sure escaped backslashes don't break string tokenization.
  expect_tokenization(R"('\\';)", {Tt::kCharacter, Tt::kSemicolon},
                      {R"('\\')", ";"});
  expect_tokenization(R"("\\";)", {Tt::kString, Tt::kSemicolon},
                      {R"("\\")", ";"});
  // Make sure unterminated string doesn't run on into the next line.
  expect_tokenization(R"("foo
";)",
                      {Tt::kString, Tt::kWhitespace, Tt::kString},
                      {R"("foo)", "\n", R"(";)"});
  // Make sure #include strings treat backslashes literally.
  expect_tokenization(
      R"(#include "x\n\\y\"//comment)",
      {Tt::kHash, Tt::kIdentifier, Tt::kWhitespace, Tt::kString, Tt::kComment},
      {"#", "include", " ", R"("x\n\\y\")", "//comment"});
  expect_tokenization(
      R"(#include <x\n\\y\>//comment)",
      {Tt::kHash, Tt::kIdentifier, Tt::kWhitespace, Tt::kString, Tt::kComment},
      {"#", "include", " ", R"(<x\n\\y\>)", "//comment"});
}

TEST(Jitify2ParserTest, AlternativeOperatorRepresentations) {
  using namespace jitify2::parser;
  static const char* const source1 = "vector<::std::string>";
  CppLexer lexer(source1);
  ASSERT_EQ(lexer.next().token_string(), "vector");
  ASSERT_EQ(lexer.next().token_string(), "<");
  ASSERT_EQ(lexer.next().token_string(), "::");
  ASSERT_EQ(lexer.next().token_string(), "std");
  ASSERT_EQ(lexer.next().token_string(), "::");
  ASSERT_EQ(lexer.next().token_string(), "string");

  static const char* const source2 = "(argv<::>)";
  lexer = CppLexer(source2);
  ASSERT_EQ(lexer.next().token_string(), "(");
  ASSERT_EQ(lexer.next().token_string(), "argv");
  ASSERT_EQ(lexer.next().token_string(), "<:");
  ASSERT_EQ(lexer.next().token_string(), ":>");
  ASSERT_EQ(lexer.next().token_string(), ")");

  static const char* const source3 = "foo<:::std::string>";
  lexer = CppLexer(source3);
  ASSERT_EQ(lexer.next().token_string(), "foo");
  ASSERT_EQ(lexer.next().token_string(), "<:");
  ASSERT_EQ(lexer.next().token_string(), "::");
  ASSERT_EQ(lexer.next().token_string(), "std");
  ASSERT_EQ(lexer.next().token_string(), "::");
  ASSERT_EQ(lexer.next().token_string(), "string");
}

TEST(Jitify2ParserTest, Minify) {
  using namespace jitify2::parser;
  using Tt = Token::Type;
  static const char* const source = R"~(#pragma once
#define BAR -1
#define CAT ( x )
#define CAT2 (x) + (y)
#define DOG( x )
#define DOG2( x ) x
#define DOG3( x ) ( x )
#define DOG4( x ) (x) + (y)
#define DOG5( x )x 
#define DOG6( x )( x )

#define FOO            \
  do {                 \
    printf("error\n"); \
  while (0)

#define _STR(x) #x
#define STR(x) _STR(x)
#define PLUS +
const char* str_suf1 = "foo"s;
const char* str_suf2 = "foo" STR(bar) "bar";
#pragma once
const char* rstr_suf1 = R"(foo)"s;
const char* rstr_suf2 = R"(foo)" s;
char char_suf = 'c's;
#pragma once
unsigned num_suf1 = 123u;
unsigned num_suf2 = 123 PLUS 1;

#pragma once
c += a++ + b;
c += a + ++b;
c += a +++ b;
c += a++++ + ++++b;

a : ::b;
)~";
  static const char* const minified_source = R"~(#pragma once
#define BAR -1
#define CAT (x)
#define CAT2 (x)+(y)
#define DOG(x)
#define DOG2(x)x
#define DOG3(x)(x)
#define DOG4(x)(x)+(y)
#define DOG5(x)x
#define DOG6(x)(x)
#define FOO do{printf("error\n");while(0)
#define _STR(x)#x
#define STR(x)_STR(x)
#define PLUS +
const char*str_suf1="foo"s;const char*str_suf2="foo" STR(bar)"bar";
#pragma once
const char*rstr_suf1=R"(foo)"s;const char*rstr_suf2=R"(foo)" s;char char_suf='c's;
#pragma once
unsigned num_suf1=123u;unsigned num_suf2=123 PLUS 1;
#pragma once
c+=a+++b;c+=a+ ++b;c+=a+++b;c+=a+++++ ++++b;a: ::b;)~";
  auto tokens = CppLexer::tokenize<TokenSequence>(source);
  const int cxx_standard_year = 20;
  std::string processed_source;
  minify_cuda_source(tokens.begin(), tokens.end(), cxx_standard_year,
                     &processed_source);
  EXPECT_EQ(processed_source, minified_source);
}

TEST(Jitify2ParserTest, ProcessCudaSource) {
  using namespace jitify2::parser;
  static const char* const source = R"~(
# /*blah*/pragma /*blah*/once  // blah
const char* include = "#include <x.h>";
#pragma once
#pragma once
// A comment.
//using std::array;
using std::array;
using ::std::array;
# /*blah*/ inclu\
de /*blah*/ <a.h> /*blah*/ // blah
#line 1
const char* bar = "#include \"y.h\"";
# /*blah*/ include /*blah*/ "b.h" /*blah*/ // blah
#include "foo/c.h"
#include <foo/c.h>
const char* cat = R"blah(#include "z.h")blah" "dog";
int i = cat[0 + 1];
)~";
  static const char* const expected =
      R"~(#ifndef JITIFY_INCLUDE_GUARD_D17F1E6F8466B0A8F5157A76D6618008AF6353BBABC40BE8FC2AFF6B38D21883
#define JITIFY_INCLUDE_GUARD_D17F1E6F8466B0A8F5157A76D6618008AF6353BBABC40BE8FC2AFF6B38D21883
#ifdef JITIFY_USED_HEADER_WARNINGS
#warning JITIFY_USED_HEADER "./my_header.cuh"
#endif
#line 1

const char* include = "#include <x.h>";
// A comment.
//using std::array;
using cuda::std::array;
using ::cuda::std::array;
# /*blah*/ inclu\
de /*blah*/ <a.h> /*blah*/ // blah
#line 1
const char* bar = "#include \"y.h\"";
# /*blah*/ include /*blah*/ <__jitify_rel_inc:.:__jitify_name:b.h> /*blah*/ // blah
#include <__jitify_rel_inc:.:__jitify_name:foo/c.h>
#include <foo/c.h>
const char* cat = R"blah(#include "z.h")blah" "dog";
int i = cat[0 + 1];
#endif // JITIFY_INCLUDE_GUARD_D17F1E6F8466B0A8F5157A76D6618008AF6353BBABC40BE8FC2AFF6B38D21883
)~";
  static const char* const expected_minified =
      R"~(#ifndef JITIFY_INCLUDE_GUARD_D17F1E6F8466B0A8F5157A76D6618008AF6353BBABC40BE8FC2AFF6B38D21883
#define JITIFY_INCLUDE_GUARD_D17F1E6F8466B0A8F5157A76D6618008AF6353BBABC40BE8FC2AFF6B38D21883
#ifdef JITIFY_USED_HEADER_WARNINGS
#warning JITIFY_USED_HEADER "./my_header.cuh"
#endif
#line 1
const char*include="#include <x.h>";using cuda::std::array;using::cuda::std::array;
#include<a.h>
#line 1
const char*bar="#include \"y.h\"";
#include<__jitify_rel_inc:.:__jitify_name:b.h>
#include<__jitify_rel_inc:.:__jitify_name:foo/c.h>
#include<foo/c.h>
const char*cat=R"blah(#include "z.h")blah""dog";int i=cat[0+1];
#endif
)~";
  const int cxx_standard_year = -1;
  std::string processed_source;
  std::vector<IncludeName> includes;
  std::string include_name = "my_header.cuh";
  std::string current_dir = ".";
  std::string include_fullpath = current_dir + "/" + include_name;
  EXPECT_TRUE(process_cuda_source(
                  source, include_fullpath,
                  ProcessFlags::kReplacePragmaOnce | ProcessFlags::kReplaceStd |
                      ProcessFlags::kAddUsedHeaderWarning,
                  cxx_standard_year, &processed_source,
                  [&](IncludeName include) { includes.push_back(include); })
                  .empty());
  std::vector<IncludeName> expected_includes = {
      IncludeName("a.h"), IncludeName("b.h", current_dir),
      IncludeName("foo/c.h", "."), IncludeName("foo/c.h")};
  ASSERT_EQ(includes, expected_includes);
  EXPECT_EQ(includes[0].location().file_name(), include_fullpath);
  EXPECT_EQ(includes[0].location().line(), 11);
  EXPECT_EQ(includes[2].location().file_name(), include_fullpath);
  EXPECT_EQ(includes[2].location().line(), 3);
  EXPECT_EQ(processed_source, expected);
  includes.clear();
  EXPECT_TRUE(process_cuda_source(
                  source, include_fullpath,
                  ProcessFlags::kReplacePragmaOnce | ProcessFlags::kReplaceStd |
                      ProcessFlags::kAddUsedHeaderWarning |
                      ProcessFlags::kMinify,
                  cxx_standard_year, &processed_source,
                  [&](IncludeName include) { includes.push_back(include); })
                  .empty());
  ASSERT_EQ(includes, expected_includes);
  EXPECT_EQ(includes[0].location().file_name(), include_fullpath);
  EXPECT_EQ(includes[0].location().line(), 11);
  EXPECT_EQ(includes[2].location().file_name(), include_fullpath);
  EXPECT_EQ(includes[2].location().line(), 3);
  EXPECT_EQ(processed_source, expected_minified);
}

TEST(Jitify2ParserTest, CppParserIterator) {
  static const char* const source =
      R"(
#/*blah*/incl\
ude/*blah*/<foo> //blah


#/*blah*/line/*blah*/10
#/*blah*/include/*blah*/"bar" //blah
#/*blah*/li\
ne/*blah*/20 "newfilename"
# include "cat"
)";
  using namespace jitify2::parser;
  using Tt = Token::Type;
  auto tokens = CppLexer::tokenize<TokenSequence>(source);
  auto iter = make_cpp_parser_iterator(tokens.begin(), tokens.end());
  EXPECT_EQ(iter.line_number(), 2);
  ASSERT_TRUE(iter.match(Tt::kHash));
  EXPECT_EQ(iter.line_number(), 2);
  ASSERT_TRUE(iter.match_identifier("include"));
  EXPECT_EQ(iter.line_number(), 3);
  ASSERT_TRUE(iter.match(Tt::kString));
  EXPECT_EQ(iter.line_number(), 3);
  ASSERT_EQ(iter.previous_token().token_string(), "<foo>");
  ASSERT_TRUE(iter.match(Tt::kEndOfDirective));

  EXPECT_EQ(iter.line_number(), 6);
  ASSERT_TRUE(iter.match(Tt::kHash));
  EXPECT_EQ(iter.line_number(), 6);
  ASSERT_TRUE(iter.match_identifier("line"));
  EXPECT_EQ(iter.line_number(), 6);
  ASSERT_TRUE(iter.match(Tt::kNumber));
  EXPECT_EQ(iter.line_number(), 6);
  ASSERT_EQ(iter.previous_token().token_string(), "10");
  ASSERT_TRUE(iter.match(Tt::kEndOfDirective));

  EXPECT_EQ(iter.line_number(), 10);
  ASSERT_TRUE(iter.match(Tt::kHash));
  EXPECT_EQ(iter.line_number(), 10);
  ASSERT_TRUE(iter.match_identifier("include"));
  EXPECT_EQ(iter.line_number(), 10);
  ASSERT_TRUE(iter.match(Tt::kString));
  EXPECT_EQ(iter.line_number(), 10);
  ASSERT_EQ(iter.previous_token().token_string(), "\"bar\"");
  ASSERT_TRUE(iter.match(Tt::kEndOfDirective));

  EXPECT_EQ(iter.line_number(), 11);
  ASSERT_TRUE(iter.match(Tt::kHash));
  EXPECT_EQ(iter.line_number(), 11);
  ASSERT_TRUE(iter.match_identifier("line"));
  EXPECT_EQ(iter.line_number(), 12);
  ASSERT_TRUE(iter.match(Tt::kNumber));
  EXPECT_EQ(iter.line_number(), 12);
  ASSERT_EQ(iter.previous_token().token_string(), "20");
  ASSERT_TRUE(iter.match(Tt::kString));
  ASSERT_EQ(iter.previous_token().token_string(), "\"newfilename\"");
  EXPECT_EQ(iter.line_number(), 12);
  ASSERT_TRUE(iter.match(Tt::kEndOfDirective));

  EXPECT_EQ(iter.line_number(), 20);
  ASSERT_TRUE(iter.match(Tt::kHash));
  EXPECT_EQ(iter.line_number(), 20);
  ASSERT_TRUE(iter.match_identifier("include"));
  EXPECT_EQ(iter.line_number(), 20);
  ASSERT_TRUE(iter.match(Tt::kString));
  EXPECT_EQ(iter.line_number(), 20);
  ASSERT_EQ(iter.previous_token().token_string(), "\"cat\"");
  ASSERT_TRUE(iter.match(Tt::kEndOfDirective));
}

int main(int argc, char** argv) {
  cudaSetDevice(0);
  // Initialize the driver context (avoids "initialization error"/"context is
  // destroyed").
  cudaFree(0);
  ::testing::InitGoogleTest(&argc, argv);
  // Test order is actually undefined, so we use filters to force the
  // AssertHeader test to run last.
  ::testing::GTEST_FLAG(filter) += ":-Jitify2Test.AssertHeader";
  int result = RUN_ALL_TESTS();
  ::testing::GTEST_FLAG(filter) = "Jitify2Test.AssertHeader";
  return result | RUN_ALL_TESTS();
}
