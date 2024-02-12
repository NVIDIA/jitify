/*
 * Copyright (c) 2017-2024, NVIDIA CORPORATION. All rights reserved.
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

#pragma once

class Managed {
 public:
  void* operator new(size_t len) {
    void* ptr = nullptr;
#ifndef __CUDACC_RTC__
    cudaMallocManaged(&ptr, len);
    cudaDeviceSynchronize();
#endif
    return ptr;
  }

  void operator delete(void* ptr) {
#ifndef __CUDACC_RTC__
    cudaDeviceSynchronize();
    cudaFree(ptr);
#endif
  }
};

struct Arg : public Managed {
  const int x;
  Arg(int x_) : x(x_) {}

  // there can be no call to the copy constructor
  Arg(const Arg& arg) = delete;
};

template <typename T>
__global__ void class_arg_kernel(int* x, T arg) {
  *x = arg.x;
}

template <typename T>
__global__ void class_arg_ref_kernel(int* x, T& arg) {
  *x = arg.x;
}

template <typename T>
__global__ void class_arg_ptr_kernel(int* x, T* arg) {
  *x = arg->x;
}
