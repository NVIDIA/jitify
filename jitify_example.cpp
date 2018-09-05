/*
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
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

/*
  Simple examples demonstrating different ways to load source code
    and call kernels.
 */

#define JITIFY_PRINT_INSTANTIATION 1
#define JITIFY_PRINT_SOURCE 1
#define JITIFY_PRINT_LOG 1
#define JITIFY_PRINT_PTX 1
#define JITIFY_PRINT_LAUNCH 1
#include "jitify.hpp"

#include "example_headers/my_header1.cuh.jit"
#ifdef LINUX  // Only supported by gcc on Linux
JITIFY_INCLUDE_EMBEDDED_FILE(example_headers_my_header2_cuh);
#endif

#include <cassert>
#include <cmath>
#include <iostream>

#define CHECK_CUDA(call)        \
  do {                          \
    if (call != CUDA_SUCCESS) { \
      return false;             \
    }                           \
  } while (0)

template <typename T>
bool are_close(T in, T out) {
  return fabs(in - out) <= 1e-5f * fabs(in);
}

std::istream* file_callback(std::string filename, std::iostream& tmp_stream) {
  // User returns NULL or pointer to stream containing file source
  // Note: tmp_stream is provided for convenience
  if (filename == "example_headers/my_header4.cuh") {
    tmp_stream << "#pragma once\n"
                  "template<typename T>\n"
                  "T pointless_func(T x) {\n"
                  "	return x;\n"
                  "}\n";
    return &tmp_stream;
  } else {
    // Find this file through other mechanisms
    return 0;
  }
}

#if __cplusplus >= 201103L

template <typename T>
bool test_simple() {
  const char* program_source =
      "my_program\n"
      "template<int N, typename T>\n"
      "__global__\n"
      "void my_kernel(T* data) {\n"
      "    T data0 = data[0];\n"
      "    for( int i=0; i<N-1; ++i ) {\n"
      "        data[0] *= data0;\n"
      "    }\n"
      "}\n";
  static jitify::JitCache kernel_cache;
  jitify::Program program = kernel_cache.program(program_source, 0);
  T h_data = 5;
  T* d_data;
  cudaMalloc((void**)&d_data, sizeof(T));
  cudaMemcpy(d_data, &h_data, sizeof(T), cudaMemcpyHostToDevice);
  dim3 grid(1);
  dim3 block(1);
  using jitify::reflection::type_of;
  CHECK_CUDA(program.kernel("my_kernel")
                 .instantiate(3, type_of(*d_data))
                 .configure(grid, block)
                 .launch(d_data));
  cudaMemcpy(&h_data, d_data, sizeof(T), cudaMemcpyDeviceToHost);
  cudaFree(d_data);
  std::cout << h_data << std::endl;
  return are_close(h_data, 125.f);
}

template <typename T>
bool test_kernels() {
  // Note: The name is specified first, followed by a newline, then the code
  const char* program1 =
      "my_program1\n"
      "#include <example_headers/my_header1.cuh>\n"
      "#include <example_headers/my_header2.cuh>\n"
      "#include <example_headers/my_header3.cuh>\n"
      "#include <example_headers/my_header4.cuh>\n"
      "\n"
      "__global__\n"
      "void my_kernel1(float const* indata, float* outdata) {\n"
      "    outdata[0] = indata[0] + 1;\n"
      "    outdata[0] -= 1;\n"
      "}\n"
      "\n"
      "template<int C, typename T>\n"
      "__global__\n"
      "void my_kernel2(float const* indata, float* outdata) {\n"
      "    for( int i=0; i<C; ++i ) {\n"
      "        outdata[0] = "
      "pointless_func(identity(sqrt(square(negate(indata[0])))));\n"
      "    }\n"
      "}\n";

  using jitify::reflection::instance_of;
  using jitify::reflection::NonType;
  using jitify::reflection::reflect;
  using jitify::reflection::Type;
  using jitify::reflection::type_of;

  thread_local static jitify::JitCache kernel_cache;
  jitify::Program program = kernel_cache.program(
      program1,                          // Code string specified above
      {example_headers_my_header1_cuh},  // Code string generated by stringify
      {"--use_fast_math", "-I/usr/local/cuda/include"}, file_callback);

  T* indata;
  T* outdata;
  cudaMalloc((void**)&indata, sizeof(T));
  cudaMalloc((void**)&outdata, sizeof(T));
  T inval = 3.14159f;
  cudaMemcpy(indata, &inval, sizeof(T), cudaMemcpyHostToDevice);

  dim3 grid(1);
  dim3 block(1);
  CHECK_CUDA(program.kernel("my_kernel1")
                 .instantiate()
                 .configure(grid, block)
                 .launch(indata, outdata));
  enum { C = 123 };
  // These invocations are all equivalent and will come from cache after the 1st
  CHECK_CUDA((program.kernel("my_kernel2")
                  .instantiate<NonType<int, C>, T>()
                  .configure(grid, block)
                  .launch(indata, outdata)));
  CHECK_CUDA(program.kernel("my_kernel2")
                 .instantiate({reflect((int)C), reflect<T>()})
                 .configure(grid, block)
                 .launch(indata, outdata));
  // Recommended versions
  CHECK_CUDA(program.kernel("my_kernel2")
                 .instantiate((int)C, Type<T>())
                 .configure(grid, block)
                 .launch(indata, outdata));
  CHECK_CUDA(program.kernel("my_kernel2")
                 .instantiate((int)C, type_of(*indata))
                 .configure(grid, block)
                 .launch(indata, outdata));
  CHECK_CUDA(program.kernel("my_kernel2")
                 .instantiate((int)C, instance_of(*indata))
                 .configure(grid, block)
                 .launch(indata, outdata));

  T outval = 0;
  cudaMemcpy(&outval, outdata, sizeof(T), cudaMemcpyDeviceToHost);
  cudaFree(outdata);
  cudaFree(indata);

  std::cout << inval << " -> " << outval << std::endl;

  return are_close(inval, outval);
}

template <typename T>
bool test_parallel_for() {
  int n = 10000;
  T* d_out;
  cudaMalloc((void**)&d_out, n * sizeof(T));
  T val = 3.14159f;

  jitify::ExecutionPolicy policy(jitify::DEVICE);
  auto lambda = JITIFY_LAMBDA((d_out, val), d_out[i] = i * val);
  CHECK_CUDA(jitify::parallel_for(policy, 0, n, lambda));

  std::vector<T> h_out(n);
  cudaMemcpy(&h_out[0], d_out, n * sizeof(T), cudaMemcpyDeviceToHost);

  cudaFree(d_out);

  for (int i = 0; i < n; ++i) {
    if (!are_close(h_out[i], i * val)) {
      std::cout << h_out[i] << " != " << i * val << std::endl;
      return false;
    }
  }
  return true;
}

#endif  // C++11

int main(int argc, char* argv[]) {
#if __cplusplus >= 201103L
#define TEST_RESULT(result) (result ? "PASSED" : "FAILED")

  // Uncached
  bool test_simple_result = test_simple<float>();
  bool test_kernels_result = test_kernels<float>();
  bool test_parallel_for_result = test_parallel_for<float>();
  // Cached
  test_simple_result &= test_simple<float>();
  test_kernels_result &= test_kernels<float>();
  test_parallel_for_result &= test_parallel_for<float>();

  std::cout << "test_simple<float>:       " << TEST_RESULT(test_simple_result)
            << std::endl;
  std::cout << "test_kernels<float>:      " << TEST_RESULT(test_kernels_result)
            << std::endl;
  std::cout << "test_parallel_for<float>: "
            << TEST_RESULT(test_parallel_for_result) << std::endl;

  return (!test_simple_result + !test_kernels_result +
          !test_parallel_for_result);
#else
  std::cout << "Tests require building with C++11 support (make CXX11=1)"
            << std::endl;
  return 0;
#endif
}
