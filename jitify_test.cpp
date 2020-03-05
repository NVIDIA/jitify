/*
 * Copyright (c) 2017-2019, NVIDIA CORPORATION. All rights reserved.
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

#ifdef LINUX  // Only supported by gcc on Linux (defined in Makefile)
#define JITIFY_ENABLE_EMBEDDED_FILES 1
#endif
#define JITIFY_PRINT_INSTANTIATION 1
#define JITIFY_PRINT_SOURCE 1
#define JITIFY_PRINT_LOG 1
#define JITIFY_PRINT_PTX 1
#define JITIFY_PRINT_LINKER_LOG 1
#define JITIFY_PRINT_LAUNCH 1
#define JITIFY_PRINT_HEADER_PATHS 1
#include "jitify.hpp"

#include "example_headers/my_header1.cuh.jit"
#ifdef LINUX  // Only supported by gcc on Linux (defined in Makefile)
JITIFY_INCLUDE_EMBEDDED_FILE(example_headers_my_header2_cuh);
#endif

#include "gtest/gtest.h"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>

#define CHECK_CUDA(call)                                                  \
  do {                                                                    \
    CUresult status = call;                                               \
    if (status != CUDA_SUCCESS) {                                         \
      const char* str;                                                    \
      cuGetErrorName(status, &str);                                       \
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

static const char* const simple_program_source =
    "my_program\n"
    "template<int N, typename T>\n"
    "__global__\n"
    "void my_kernel(T* data) {\n"
    "    if (blockIdx.x != 0 || threadIdx.x != 0) return;\n"
    "    T data0 = data[0];\n"
    "    for( int i=0; i<N-1; ++i ) {\n"
    "        data[0] *= data0;\n"
    "    }\n"
    "}\n";

TEST(JitifyTest, Simple) {
  static jitify::JitCache kernel_cache;
  jitify::Program program = kernel_cache.program(simple_program_source);
  typedef float T;
  T* d_data;
  CHECK_CUDART(cudaMalloc((void**)&d_data, sizeof(T)));
  dim3 grid(1);
  dim3 block(1);
  using jitify::reflection::type_of;
  auto kernel_inst =
      program.kernel("my_kernel").instantiate(3, type_of(*d_data));
  T h_data = 5;
  CHECK_CUDART(cudaMemcpy(d_data, &h_data, sizeof(T), cudaMemcpyHostToDevice));
  CHECK_CUDA(kernel_inst.configure(grid, block).launch(d_data));
  CHECK_CUDART(cudaMemcpy(&h_data, d_data, sizeof(T), cudaMemcpyDeviceToHost));
  EXPECT_FLOAT_EQ(h_data, 125.f);

  h_data = 5;
  CHECK_CUDART(cudaMemcpy(d_data, &h_data, sizeof(T), cudaMemcpyHostToDevice));
  CHECK_CUDA(kernel_inst.configure_1d_max_occupancy().launch(d_data));
  CHECK_CUDART(cudaMemcpy(&h_data, d_data, sizeof(T), cudaMemcpyDeviceToHost));
  EXPECT_FLOAT_EQ(h_data, 125.f);

  CHECK_CUDART(cudaFree(d_data));
}

TEST(JitifyTest, Simple_experimental) {
  std::vector<std::string> opts;
  jitify::experimental::Program program_orig(simple_program_source, {}, opts);
  auto program =
      jitify::experimental::Program::deserialize(program_orig.serialize());
  typedef float T;
  T* d_data;
  CHECK_CUDART(cudaMalloc((void**)&d_data, sizeof(T)));
  dim3 grid(1);
  dim3 block(1);
  using jitify::reflection::type_of;
  auto kernel_inst_orig =
      program.kernel("my_kernel").instantiate(3, type_of(*d_data));
  auto kernel_inst = jitify::experimental::KernelInstantiation::deserialize(
      kernel_inst_orig.serialize());
  T h_data = 5;
  CHECK_CUDART(cudaMemcpy(d_data, &h_data, sizeof(T), cudaMemcpyHostToDevice));
  CHECK_CUDA(kernel_inst.configure(grid, block).launch(d_data));
  CHECK_CUDART(cudaMemcpy(&h_data, d_data, sizeof(T), cudaMemcpyDeviceToHost));
  EXPECT_FLOAT_EQ(h_data, 125.f);

  h_data = 5;
  CHECK_CUDART(cudaMemcpy(d_data, &h_data, sizeof(T), cudaMemcpyHostToDevice));
  CHECK_CUDA(kernel_inst.configure_1d_max_occupancy().launch(d_data));
  CHECK_CUDART(cudaMemcpy(&h_data, d_data, sizeof(T), cudaMemcpyDeviceToHost));
  EXPECT_FLOAT_EQ(h_data, 125.f);

  CHECK_CUDART(cudaFree(d_data));
}

static const char* const multiple_kernels_program_source =
    "my_program1\n"
    "#include \"example_headers/my_header1.cuh\"\n"
    "#include \"example_headers/my_header2.cuh\"\n"
    "#include \"example_headers/my_header3.cuh\"\n"
    "#include \"example_headers/my_header4.cuh\"\n"
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

TEST(JitifyTest, MultipleKernels) {
  using jitify::reflection::instance_of;
  using jitify::reflection::NonType;
  using jitify::reflection::reflect;
  using jitify::reflection::Type;
  using jitify::reflection::type_of;

  thread_local static jitify::JitCache kernel_cache;
  jitify::Program program = kernel_cache.program(
      multiple_kernels_program_source,   // Code string specified above
      {example_headers_my_header1_cuh},  // Code string generated by stringify
      {"--use_fast_math", "-I/usr/local/cuda/include"}, file_callback);

  typedef float T;
  T* indata;
  T* outdata;
  CHECK_CUDART(cudaMalloc((void**)&indata, sizeof(T)));
  CHECK_CUDART(cudaMalloc((void**)&outdata, sizeof(T)));
  T inval = 3.14159f;
  CHECK_CUDART(cudaMemcpy(indata, &inval, sizeof(T), cudaMemcpyHostToDevice));

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
  CHECK_CUDART(cudaMemcpy(&outval, outdata, sizeof(T), cudaMemcpyDeviceToHost));
  CHECK_CUDART(cudaFree(outdata));
  CHECK_CUDART(cudaFree(indata));

  EXPECT_FLOAT_EQ(inval, outval);
}

TEST(JitifyTest, MultipleKernels_experimental) {
  using jitify::reflection::instance_of;
  using jitify::reflection::NonType;
  using jitify::reflection::reflect;
  using jitify::reflection::Type;
  using jitify::reflection::type_of;

  jitify::experimental::Program program_orig(
      multiple_kernels_program_source,   // Code string specified above
      {example_headers_my_header1_cuh},  // Code string generated by stringify
      {"--use_fast_math", "-I/usr/local/cuda/include"}, file_callback);
  auto program =
      jitify::experimental::Program::deserialize(program_orig.serialize());

  typedef float T;
  T* indata;
  T* outdata;
  CHECK_CUDART(cudaMalloc((void**)&indata, sizeof(T)));
  CHECK_CUDART(cudaMalloc((void**)&outdata, sizeof(T)));
  T inval = 3.14159f;
  CHECK_CUDART(cudaMemcpy(indata, &inval, sizeof(T), cudaMemcpyHostToDevice));

  dim3 grid(1);
  dim3 block(1);
  CHECK_CUDA(program.kernel("my_kernel1")
                 .instantiate()
                 .configure(grid, block)
                 .launch(indata, outdata));
  enum { C = 123 };
  // These invocations are all equivalent.
  CHECK_CUDA(jitify::experimental::KernelInstantiation::deserialize(
                 program.kernel("my_kernel2")
                     .instantiate<NonType<int, C>, T>()
                     .serialize())
                 .configure(grid, block)
                 .launch(indata, outdata));
  CHECK_CUDA(jitify::experimental::KernelInstantiation::deserialize(
                 program.kernel("my_kernel2")
                     .instantiate({reflect((int)C), reflect<T>()})
                     .serialize())
                 .configure(grid, block)
                 .launch(indata, outdata));
  // Recommended versions
  CHECK_CUDA(jitify::experimental::KernelInstantiation::deserialize(
                 program.kernel("my_kernel2")
                     .instantiate((int)C, Type<T>())
                     .serialize())
                 .configure(grid, block)
                 .launch(indata, outdata));
  CHECK_CUDA(jitify::experimental::KernelInstantiation::deserialize(
                 program.kernel("my_kernel2")
                     .instantiate((int)C, type_of(*indata))
                     .serialize())
                 .configure(grid, block)
                 .launch(indata, outdata));
  CHECK_CUDA(jitify::experimental::KernelInstantiation::deserialize(
                 program.kernel("my_kernel2")
                     .instantiate((int)C, instance_of(*indata))
                     .serialize())
                 .configure(grid, block)
                 .launch(indata, outdata));

  T outval = 0;
  CHECK_CUDART(cudaMemcpy(&outval, outdata, sizeof(T), cudaMemcpyDeviceToHost));
  CHECK_CUDART(cudaFree(outdata));
  CHECK_CUDART(cudaFree(indata));

  EXPECT_FLOAT_EQ(inval, outval);
}

static const char* const constmem_program_source =
    "constmem_program\n"
    "#pragma once\n"
    "\n"
    "__constant__ int a;\n"
    "__device__ int d;\n"
    "namespace b { __constant__ int a; __device__ int d; }\n"
    "namespace c { namespace b { __constant__ int a; __device__ int d; } }\n"
    "\n"
    "__global__ void constant_test(int *x) {\n"
    "  x[0] = a;\n"
    "  x[1] = b::a;\n"
    "  x[2] = c::b::a;\n"
    "  x[3] = d;\n"
    "  x[4] = b::d;\n"
    "  x[5] = c::b::d;\n"
    "}\n";

TEST(JitifyTest, ConstantMemory) {
  using jitify::reflection::Type;
  thread_local static jitify::JitCache kernel_cache;

  constexpr int n_const = 6;
  int* outdata;
  CHECK_CUDART(cudaMalloc((void**)&outdata, n_const * sizeof(int)));

  dim3 grid(1);
  dim3 block(1);
  {  // test __constant__ look up in kernel string using diffrent namespaces
    jitify::Program program =
        kernel_cache.program(constmem_program_source, 0,
                             {"--use_fast_math", "-I/usr/local/cuda/include"});
    auto instance = program.kernel("constant_test").instantiate();
    int inval[] = {2, 4, 8, 12, 14, 18};
    CHECK_CUDA(
        cuMemcpyHtoD(instance.get_constant_ptr("a"), &inval[0], sizeof(int)));
    CHECK_CUDA(cuMemcpyHtoD(instance.get_constant_ptr("b::a"), &inval[1],
                            sizeof(int)));
    CHECK_CUDA(cuMemcpyHtoD(instance.get_constant_ptr("c::b::a"), &inval[2],
                            sizeof(int)));
    CHECK_CUDA(
        cuMemcpyHtoD(instance.get_global_ptr("d"), &inval[3], sizeof(int)));
    CHECK_CUDA(cuMemcpyHtoD(instance.get_global_ptr("b::d"), &inval[4],
                            sizeof(int)));
    CHECK_CUDA(cuMemcpyHtoD(instance.get_global_ptr("c::b::d"), &inval[5],
                            sizeof(int)));
    CHECK_CUDA(instance.configure(grid, block).launch(outdata));
    CHECK_CUDART(cudaDeviceSynchronize());
    int outval[n_const];
    CHECK_CUDART(
        cudaMemcpy(outval, outdata, sizeof(outval), cudaMemcpyDeviceToHost));

    for (int i = 0; i < n_const; i++) {
      EXPECT_EQ(inval[i], outval[i]);
    }
  }

  {  // test __constant__ array look up in header nested in both anonymous and
     // explicit namespace
    jitify::Program program =
        kernel_cache.program("example_headers/constant_header.cuh", 0,
                             {"--use_fast_math", "-I/usr/local/cuda/include"});
    auto instance = program.kernel("constant_test2").instantiate();
    int inval[] = {3, 5, 9, 13, 15, 19};
    CHECK_CUDA(
        cuMemcpyHtoD(instance.get_constant_ptr("(anonymous namespace)::b::a"),
                     inval, sizeof(inval) / 2));
    CHECK_CUDA(
        cuMemcpyHtoD(instance.get_global_ptr("(anonymous namespace)::b::d"),
                     inval + 3, sizeof(inval) / 2));
    CHECK_CUDA(instance.configure(grid, block).launch(outdata));

    int outval[n_const];
    CHECK_CUDART(
        cudaMemcpy(outval, outdata, sizeof(outval), cudaMemcpyDeviceToHost));

    for (int i = 0; i < n_const; i++) {
      EXPECT_EQ(inval[i], outval[i]);
    }
  }

  CHECK_CUDART(cudaFree(outdata));
}

TEST(JitifyTest, ConstantMemory_experimental) {
  using jitify::reflection::Type;

  constexpr int n_const = 6;
  int* outdata;
  CHECK_CUDART(cudaMalloc((void**)&outdata, n_const * sizeof(int)));

  dim3 grid(1);
  dim3 block(1);
  {  // test __constant__ look up in kernel string using different namespaces
    jitify::experimental::Program program_orig(
        constmem_program_source, {},
        {"--use_fast_math", "-I/usr/local/cuda/include"});
    auto program =
        jitify::experimental::Program::deserialize(program_orig.serialize());
    auto instance = jitify::experimental::KernelInstantiation::deserialize(
        program.kernel("constant_test").instantiate().serialize());
    int inval[] = {2, 4, 8, 12, 14, 18};
    CHECK_CUDA(
        cuMemcpyHtoD(instance.get_constant_ptr("a"), &inval[0], sizeof(int)));
    CHECK_CUDA(cuMemcpyHtoD(instance.get_constant_ptr("b::a"), &inval[1],
                            sizeof(int)));
    CHECK_CUDA(cuMemcpyHtoD(instance.get_constant_ptr("c::b::a"), &inval[2],
                            sizeof(int)));
    CHECK_CUDA(
        cuMemcpyHtoD(instance.get_global_ptr("d"), &inval[3], sizeof(int)));
    CHECK_CUDA(cuMemcpyHtoD(instance.get_global_ptr("b::d"), &inval[4],
                            sizeof(int)));
    CHECK_CUDA(cuMemcpyHtoD(instance.get_global_ptr("c::b::d"), &inval[5],
                            sizeof(int)));
    CHECK_CUDA(instance.configure(grid, block).launch(outdata));
    CHECK_CUDART(cudaDeviceSynchronize());
    int outval[n_const];
    CHECK_CUDART(
        cudaMemcpy(outval, outdata, sizeof(outval), cudaMemcpyDeviceToHost));

    for (int i = 0; i < n_const; i++) {
      EXPECT_EQ(inval[i], outval[i]);
    }
  }

  {  // test __constant__ array look up in header nested in both anonymous and
     // explicit namespace
    jitify::experimental::Program program_orig(
        "example_headers/constant_header.cuh", {},
        {"--use_fast_math", "-I/usr/local/cuda/include"});
    auto program =
        jitify::experimental::Program::deserialize(program_orig.serialize());
    auto instance = jitify::experimental::KernelInstantiation::deserialize(
        program.kernel("constant_test2").instantiate().serialize());
    int inval[] = {3, 5, 9, 13, 15, 19};
    CHECK_CUDA(
        cuMemcpyHtoD(instance.get_constant_ptr("(anonymous namespace)::b::a"),
                     inval, sizeof(inval) / 2));
    CHECK_CUDA(
        cuMemcpyHtoD(instance.get_global_ptr("(anonymous namespace)::b::d"),
                     inval + 3, sizeof(inval) / 2));
    CHECK_CUDA(instance.configure(grid, block).launch(outdata));

    int outval[n_const];
    CHECK_CUDART(
        cudaMemcpy(outval, outdata, sizeof(outval), cudaMemcpyDeviceToHost));

    for (int i = 0; i < n_const; i++) {
      EXPECT_EQ(inval[i], outval[i]);
    }
  }

  CHECK_CUDART(cudaFree(outdata));
}

TEST(JitifyTest, ParallelFor) {
  int n = 10000;
  typedef float T;
  T* d_out;
  CHECK_CUDART(cudaMalloc((void**)&d_out, n * sizeof(T)));
  T val = 3.14159f;

  jitify::ExecutionPolicy policy(jitify::DEVICE);
  auto lambda = JITIFY_LAMBDA((d_out, val), d_out[i] = (float)i * val);
  CHECK_CUDA(jitify::parallel_for(policy, 0, n, lambda));

  std::vector<T> h_out(n);
  CHECK_CUDART(
      cudaMemcpy(&h_out[0], d_out, n * sizeof(T), cudaMemcpyDeviceToHost));

  CHECK_CUDART(cudaFree(d_out));

  for (int i = 0; i < n; ++i) {
    EXPECT_FLOAT_EQ(h_out[i], (T)i * val);
  }
}

TEST(JitifyTest, InvalidPrograms) {
  jitify::JitCache kernel_cache;
  auto program_v1 = kernel_cache.program("empty_program\n");  // OK
  EXPECT_THROW(auto program_v1 = kernel_cache.program("missing_filename"),
               std::runtime_error);
  EXPECT_THROW(
      auto program_v1 = kernel_cache.program("bad_program\nNOT CUDA C!"),
      std::runtime_error);
  jitify::experimental::Program program_v2("empty_program\n");  // OK
  EXPECT_THROW(jitify::experimental::Program program_v2("missing_filename"),
               std::runtime_error);
  EXPECT_THROW(
      jitify::experimental::Program program_v2("bad_program\nNOT CUDA C!"),
      std::runtime_error);
}

// TODO: Expand this to include more Thrust code.
static const char* const thrust_program_source =
    "thrust_program\n"
    "#include <thrust/iterator/counting_iterator.h>\n"
    "__global__ void my_kernel(thrust::counting_iterator<int> begin,\n"
    "                          thrust::counting_iterator<int> end) {\n"
    "}\n";

TEST(JitifyTest, ThrustHeaders) {
  // Checks that basic Thrust headers can be compiled.
  jitify::JitCache kernel_cache;
  auto program_v1 = kernel_cache.program(thrust_program_source, {},
                                         {"-I" CUDA_INC_DIR, "-std=c++98"});
  auto program_v2 = jitify::experimental::Program(
      thrust_program_source, {}, {"-I" CUDA_INC_DIR, "-std=c++98"});
}

static const char* const cub_program_source =
    "cub_program\n"
    "#include <cub/block/block_load.cuh>\n"
    "#include <cub/block/block_radix_sort.cuh>\n"
    "#include <cub/block/block_reduce.cuh>\n"
    "#include <cub/block/block_store.cuh>\n"
    "\n"
    "template<int BLOCK_SIZE, int PER_THREAD>\n"
    "__global__ void my_kernel(float* data) {\n"
    "    typedef cub::BlockLoad<float, BLOCK_SIZE, PER_THREAD,\n"
    "                           cub::BLOCK_LOAD_VECTORIZE> BlockLoad;\n"
    "    typedef cub::BlockRadixSort<float, BLOCK_SIZE, PER_THREAD>\n"
    "        BlockSort;\n"
    "    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;\n"
    "    typedef cub::BlockStore<float, BLOCK_SIZE, PER_THREAD,\n"
    "                            cub::BLOCK_STORE_VECTORIZE> BlockStore;\n"
    "    __shared__ union {\n"
    "        typename BlockLoad::TempStorage load;\n"
    "        typename BlockSort::TempStorage sort;\n"
    "        typename BlockReduce::TempStorage reduce;\n"
    "        typename BlockStore::TempStorage store;\n"
    "        float sum;\n"
    "      } temp_storage;\n"
    "    float thread_data[PER_THREAD];\n"
    "    BlockLoad(temp_storage.load).Load(data, thread_data);\n"
    "    __syncthreads();\n"
    "    BlockSort(temp_storage.sort).Sort(thread_data);\n"
    "    __syncthreads();\n"
    "    float sum = BlockReduce(temp_storage.reduce).Sum(thread_data);\n"
    "    __syncthreads();\n"
    "    if (threadIdx.x == 0) {\n"
    "      temp_storage.sum = sum;\n"
    "    }\n"
    "    __syncthreads();\n"
    "    sum = temp_storage.sum;\n"
    "    #pragma unroll\n"
    "    for (int i = 0; i < PER_THREAD; ++i) {\n"
    "        thread_data[i] *= 1.f / sum;\n"
    "    }\n"
    "    __syncthreads();\n"
    "    BlockStore(temp_storage.store).Store(data, thread_data);\n"
    "}\n";

TEST(JitifyTest, CubBlockPrimitives) {
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

  jitify::JitCache kernel_cache;
  auto program_v1 = kernel_cache.program(cub_program_source, {},
                                         {"-I" CUB_DIR, "-I" CUDA_INC_DIR});
  CHECK_CUDART(cudaMemcpy(d_data, h_data.data(), n * sizeof(float),
                          cudaMemcpyHostToDevice));
  CHECK_CUDA(program_v1.kernel("my_kernel")
                 .instantiate(block_size, per_thread)
                 .configure(1, block_size)
                 .launch(d_data));
  CHECK_CUDART(cudaMemcpy(h_result.data(), d_data, n * sizeof(float),
                          cudaMemcpyDeviceToHost));
  for (int i = 0; i < n; ++i) {
    EXPECT_FLOAT_EQ(h_result[i], h_expected[i]);
  }

  auto program_v2 = jitify::experimental::Program::deserialize(
      jitify::experimental::Program(cub_program_source, {},
                                    {"-I" CUB_DIR, "-I" CUDA_INC_DIR})
          .serialize());
  auto kernel_inst_v2 = jitify::experimental::KernelInstantiation::deserialize(
      program_v2.kernel("my_kernel")
          .instantiate(block_size, per_thread)
          .serialize());
  CHECK_CUDART(cudaMemcpy(d_data, h_data.data(), n * sizeof(float),
                          cudaMemcpyHostToDevice));
  CHECK_CUDA(kernel_inst_v2.configure(1, block_size).launch(d_data));
  CHECK_CUDART(cudaMemcpy(h_result.data(), d_data, n * sizeof(float),
                          cudaMemcpyDeviceToHost));
  for (int i = 0; i < n; ++i) {
    EXPECT_FLOAT_EQ(h_result[i], h_expected[i]);
  }

  CHECK_CUDART(cudaFree(d_data));
}

static const char* const unused_globals_source =
    "unused_globals_program\n"
    "struct Foo { static const int value = 7; };\n"
    "struct Bar { int a; double b; };\n"
    "__device__ float used_scalar;\n"
    "__device__ float used_array[2];\n"
    "__device__ Bar used_struct;\n"
    "__device__ float unused_scalar;\n"
    "__device__ float unused_array[3];\n"
    "__device__ Bar unused_struct;\n"
    "__device__ float reg, ret, bra;\n"  // Tricky names
    "__global__ void foo_kernel(int* data) {\n"
    "  if (blockIdx.x != 0 || threadIdx.x != 0) return;\n"
    "  used_scalar = 1.f;\n"
    "  used_array[1] = 2.f;\n"
    "  used_struct.b = 3.f;\n"
    "  __syncthreads();\n"
    "  *data += Foo::value + used_scalar + used_array[1] + used_struct.b;\n"
  "  printf(\"*data = %i\\n\", *data);\n"  // Produces global symbols named $str
    "}\n";

TEST(JitifyTest, RemoveUnusedGlobals) {
  cudaFree(0);
  auto program_v2 = jitify::experimental::Program(
      unused_globals_source, {},
      // Note: Flag added twice to test handling of repeats.
      {"-remove-unused-globals", "--remove-unused-globals"});
  auto kernel_inst_v2 = program_v2.kernel("foo_kernel").instantiate();
  std::string ptx = kernel_inst_v2.ptx();
  EXPECT_TRUE(ptx.find(".global .align 4 .f32 used_scalar;") !=
              std::string::npos);
  // Note: PTX represents arrays and structs as .b8 instead of the actual type.
  EXPECT_TRUE(ptx.find(".global .align 4 .b8 used_array[8];") !=
              std::string::npos);
  EXPECT_TRUE(ptx.find(".global .align 8 .b8 used_struct[16];") !=
              std::string::npos);
  EXPECT_FALSE(ptx.find("_ZN3Foo5valueE") != std::string::npos);
  EXPECT_FALSE(ptx.find("unused_scalar;") != std::string::npos);
  EXPECT_FALSE(ptx.find("unused_array;") != std::string::npos);
  EXPECT_FALSE(ptx.find("unused_struct;") != std::string::npos);
  EXPECT_FALSE(ptx.find(".global .align 4 .f32 reg;") != std::string::npos);
  EXPECT_FALSE(ptx.find(".global .align 4 .f32 ret;") != std::string::npos);
  EXPECT_FALSE(ptx.find(".global .align 4 .f32 bra;") != std::string::npos);
  int* d_data;
  CHECK_CUDART(cudaMalloc((void**)&d_data, sizeof(int)));
  int h_data = 3;
  CHECK_CUDART(
      cudaMemcpy(d_data, &h_data, sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(kernel_inst_v2.configure(1, 1).launch(d_data));
  CHECK_CUDART(
      cudaMemcpy(&h_data, d_data, sizeof(int), cudaMemcpyDeviceToHost));
  EXPECT_EQ(h_data, 16);
  CHECK_CUDART(cudaFree(d_data));
}

static const char* const curand_program_source =
    "curand_program\n"
    "#include <curand_kernel.h>\n"
    "__global__ void my_kernel() {}\n"
    "\n";

TEST(JitifyTest, CuRandKernel) {
  auto program_v2 = jitify::experimental::Program(
      curand_program_source, {},
      // Note: --remove-unused-globals is added to remove huge precomputed
      // arrays that come from CURAND.
      {"-I" CUDA_INC_DIR, "--remove-unused-globals"});
  auto kernel_inst_v2 = program_v2.kernel("my_kernel").instantiate();
  // TODO: Expand this test to actually call curand kernels and check outputs.
}

static const char* const linktest_program1_source =
    "linktest_program1\n"
    "__constant__ int c = 5;\n"
    "__device__ int d = 7;\n"
    "__device__ int f(int i) { return i + 11; }\n"
    "\n";

static const char* const linktest_program2_source =
    "linktest_program2\n"
    "extern __constant__ int c;\n"
    "extern __device__ int d;\n"
    "extern __device__ int f(int);\n"
    "__global__ void my_kernel(int* data) {\n"
    "  *data = f(*data + c + d);\n"
    "}\n"
    "\n";

TEST(JitifyTest, LinkExternalFiles) {
  cudaFree(0);
  // Ensure temporary file is deleted at the end.
  std::unique_ptr<const char, int(*)(const char*)> ptx_filename(
      "example_headers/linktest.ptx", std::remove);
  {
    std::ofstream ptx_file(ptx_filename.get());
    ptx_file.exceptions(std::ofstream::failbit | std::ofstream::badbit);
    ptx_file << jitify::experimental::Program(linktest_program1_source, {},
                                              {"-rdc=true"})
                    .kernel("")
                    .instantiate()
                    .ptx();
  }
  auto program_v2 = jitify::experimental::Program(
      linktest_program2_source, {},
      {"-rdc=true", "-Lexample_headers", "-llinktest.ptx"});
  auto kernel_inst_v2 = program_v2.kernel("my_kernel").instantiate();
  int* d_data;
  CHECK_CUDART(cudaMalloc((void**)&d_data, sizeof(int)));
  int h_data = 3;
  CHECK_CUDART(
      cudaMemcpy(d_data, &h_data, sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(kernel_inst_v2.configure(1, 1).launch(d_data));
  CHECK_CUDART(
      cudaMemcpy(&h_data, d_data, sizeof(int), cudaMemcpyDeviceToHost));
  EXPECT_EQ(h_data, 26);
  CHECK_CUDART(cudaFree(d_data));
}

// NOTE: Keep this as the last test in the file, in case the env var is sticky.
TEST(JitifyTest, EnvVarOptions) {
  setenv("JITIFY_OPTIONS", "-bad_option", true);
  EXPECT_THROW(jitify::JitCache kernel_cache;
               auto program = kernel_cache.program(simple_program_source),
               std::runtime_error);
  EXPECT_THROW(jitify::experimental::Program program(simple_program_source),
               std::runtime_error);
  setenv("JITIFY_OPTIONS", "", true);
}
