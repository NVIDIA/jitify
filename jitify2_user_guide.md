
Jitify User Guide
=================

### Table of Contents

[Basic usage](#basic_usage)

[Error handling](#error_handling)

[Basic workflow](#basic_workflow)

[Advanced workflow](#advanced_workflow)

[Unit tests](#unit_tests)

[Build options](#build_options)

[Compiler options](#compiler_options)

<a name="basic_usage"/>

## Basic usage

Jitify is just a single header file:

```c++
#include <jitify2.hpp>
```

which only requires linking with the CUDA Driver (and NVRTC at runtime):

```bash
# with NVCC:
$ nvcc ...
# or with GCC:
$ g++ ... -I$CUDA_INC_DIR -L$CUDA_LIB_DIR -lcuda
```

It provides a simple API for compiling and executing CUDA source code at runtime:

```c++
  std::string program_name = "my_program";
  std::string program_source = R"(
  template <typename T>
  __global__ void my_kernel(T* data) { *data = T{7}; }
  )";
  dim3 grid(1), block(1);
  float* data;
  cudaMalloc((void**)&data, sizeof(float));
  jitify2::LoadedProgram program =
      jitify2::Program(program_name, program_source)
          // Preprocess source code and load all included headers.
          ->preprocess({"-std=c++14"})
          // Compile, link, and load the program, and obtain the loaded kernel.
          ->get_kernel("my_kernel<float>")
          // Configure the kernel launch.
          ->configure(grid, block)
          // Launch the kernel.
          ->launch(data);
```

<a name="error_handling"/>

## Error handling

All Jitify APIs such as `preprocess()`, `compile()`, `link()`,
`load()`, and `get_kernel()` return special objects that wrap either a
valid data object (if the call succeeds) or an error state (if the
call fails). The error state can be inspected using `operator bool()`
and the `error()` method. If the macro `JITIFY_ENABLE_EXCEPTIONS` is not
defined to 0 before jitify.hpp is included in your application, an
exception will be thrown when attempting to use the result of a failed
call or when a method such as `launch()` fails:

```c++
  jitify2::PreprocessedProgram preprog =
      jitify2::Program(program_name, program_source)
          ->preprocess({"-std=c++14"});
  if (!preprog) {
    // The call failed, we can access the error.
    std::cerr << preprog.error() << std::endl;
    // This will either throw an exception or terminate the application.
    *preprog;
  } else {
    // The call succeeded, we can access the data object.
    jitify2::PreprocessedProgramData preprog_data = *preprog;
    // Or we can directly call a method on the data object.
    jitify2::CompiledProgram compiled = preprog->compile("my_kernel");
    // This will throw (or terminate) if any of the chained methods fails.
    preprog->compile("my_kernel")
        ->link()
        ->load()
        ->get_kernel("my_kernel")
        ->configure(1, 1)
        ->launch();
  }
```

<a name="basic_workflow"/>

## Basic workflow example

Here we describe a complete workflow for integrating Jitify into an
application. There are many ways to use Jitify, but this is the
recommended approach.

The jitify_preprocess tool allows CUDA source to be transformed and
headers to be loaded and baked into the application during offline
compilation, avoiding the need to perform these transformations or
to load any headers at runtime.

First run jitify_preprocess to generate JIT headers for your runtime
sources:

```bash
$ ./jitify_preprocess -i myprog1.cu myprog2.cu
```

Then include the headers in your application:

```c++
#include "myprog1.cu.jit.hpp"
#include "myprog2.cu.jit.hpp"
```

And use the variables they define to construct a `ProgramCache` object:

```c++
  using jitify2::ProgramCache;
  static ProgramCache<> myprog1_cache(/*max_size = */ 100, *myprog1_cu_jit);
```

Kernels can then be obtained directly from the cache:

```c++
  using jitify2::reflection::Template;
  using jitify2::reflection::Type;
  myprog1_cache
    .get_kernel(Template("my_kernel").instantiate(123, Type<float>()))
    ->configure(grid, block)
    ->launch(idata, odata);
```

<a name="advanced_workflow"/>

## Advanced workflow example

The jitify_preprocess tool also supports automatic minification of source code as
well as generation of a separate source file for sharing runtime headers between
different runtime programs:

```bash
$ ./jitify_preprocess -i --minify -s myheaders myprog1.cu myprog2.cu
```

The generated source file should be linked with your application:

```bash
$ g++ -o myapp myapp.cpp myheaders.jit.cpp ...
```

And the generated variable should be passed to the ProgramCache constructor.
A directory name can also be specified to enable caching of compiled binaries on
disk:

```c++
#include "myprog1.cu.jit.hpp"
#include "myprog2.cu.jit.hpp"
...
  using jitify2::ProgramCache;
  static ProgramCache<> myprog1_cache(
      /*max_size = */ 100, *myprog1_cu_jit, myheaders_jit, "/tmp/my_jit_cache");
```

For advanced use-cases, multiple kernels can be instantiated in a single program:

```c++
  using jitify2::reflection::Template;
  using jitify2::reflection::Type;
  using jitify2::Program;
  std::string kernel1 = Template("my_kernel1").instantiate(123, Type<float>());
  std::string kernel2 =
      Template("my_kernel2").instantiate(45, Type<int>(), Type<int>());
  Program myprog1 = myprog1_cache.get_program({kernel1, kernel2});
  myprog1->set_global_value("my::value", 3.14f);
  myprog1->get_kernel(kernel1)->configure(grid, block)->launch(idata, odata);
  myprog1->get_kernel(kernel2)->configure(grid, block)->launch(idata, odata);
```

For improved performance, the cache can be given user-defined keys:

```c++
  using jitify2::ProgramCache;
  using jitify2::Kernel;
  using MyKeyType = uint32_t;
  static ProgramCache<MyKeyType> myprog1_cache(
      /*max_size = */ 100, *myprog1_cu_jit, myheaders_jit, "/tmp/my_jit_cache");
  std::string kernel1 = Template("my_kernel1").instantiate(123, Type<float>());
  Kernel kernel = myprog1_cache.get_kernel(MyKeyType(7), kernel1);
```

<a name="unit_tests"/>

## Unit tests

The unit tests can be built and run using CMake as follows:

```bash
$ mkdir build && cd build && cmake ..
$ make check
```

<a name="build_options"/>

## Build options

- `JITIFY_ENABLE_EXCEPTIONS=1`

  Defining this macro to 0 before including the jitify header disables
  the use of exceptions throughout the API, requiring the user to
  explicitly check for errors. See [Error handling](#error_handling)
  for more details.

- `JITIFY_THREAD_SAFE=1`

  Defining this macro to 0 before including the jitify header disables
  the use of mutexes in the ProgramCache class.

- `JITIFY_LINK_NVRTC_STATIC=0`

  Defining this macro to 1 before including the jitify header disables
  dynamic loading of the NVRTC dynamic library and allows the
  library to be linked statically.

- `JITIFY_FAIL_IMMEDIATELY=0`

  Defining this macro to 1 before including the jitify header causes
  errors to trigger exceptions/termination immediately instead of
  only when a jitify object is dereferenced. This is useful for
  debugging, as it allows the origin of an error to be found via a
  backtrace.

<a name="compiler_options"/>

## Compiler options

The Jitify API accepts options that can be used to control compilation
and linking. While most options are simply passed through to NVRTC
(for compiler options) or the CUDA cuLink APIs (for linker options),
some trigger special behavior in Jitify as detailed below:

- `-I<dir>`

  Specifies a directory to search for include files. Jitify intercepts
  these flags and handles searching for include files itself instead
  of relying on NVRTC, in order to provide more flexibility.

- `-remove-unused-globals (-remove-unused-globals)`

  Causes all unused `.global (__device__)` and `.const (__constant__)`
  variable declarations to be removed from the compiled PTX
  source. This is useful for avoiding bloated PTX when there are many
  static constants or large precomputed arrays that appear in headers
  but are not used in the compiled code.

- `--device-as-default-execution-space (-default-device)`

  This flag is automatically passed to NVRTC for all kernels. It
  avoids compiler errors arising from accidental inclusion of host
  code (a common problem).

- `--gpu-architecture=<arch> (-arch)`

  This flag controls the GPU architecture for which the program is
  preprocessed or compiled (note that it is treated separately for
  these two operations and is not automatically forwarded from
  preprocessing to compilation). If not specified, this flag will
  automatically be added with a value set to a virtual compute
  architecture corresponding to the current CUDA context (i.e., the
  device returned by `cuCtxGetDevice`). The user may specify this flag
  with either a virtual ("compute_XX") or real ("sm_XX") architecture:
  a virtual architecture will trigger compilation to PTX, while a real
  architecture will trigger direct-to-CUBIN compilation (if it is
  supported; otherwise it will fall back to PTX compilation and the
  real architecture will be passed to the CUDA driver for PTX to CUBIN
  compilation).

  For preprocessing (but not compilation), multiple architecture
  values may be specified by repeating the flag. The source will be
  preprocessed for all specified architectures to produce a single
  complete set of header dependencies. This functionality only needs
  to be used if the source contains `#include` statements that depend
  on the value of the `__CUDA_ARCH__` macro; in all other cases there
  should not be any need to specify multiple architectures.
  Note that an architecture flag specified for preprocessing is _not_
  automatically passed through to the compilation phase (because it
  would be ambiguous in the case of multiple architectures); the flag
  must be specified separately for the compilation phase.

  For compilation (but not preprocessing), the architecture value may
  be specified using the special syntax "compute_."  or "sm_." to
  explicitly select the preferred type of compilation (PTX or
  direct-to-CUBIN respectively) while still relying on automatic
  detection of the architecture.

  Note that when compiling with the "-dlto" flag (generating NVVM for
  link-time optimization, which is supported as of CUDA 11.4), it does
  not matter whether a real or virtual architecture is specified (but
  the architecture number does matter as sets the __CUDA_ARCH__ flag).
  However, some versions of NVRTC have an issue where they only accept
  virtual architectures when compiling with "-dlto".

- `-std=<std>`

  Unless otherwise specified, this flag is automatically passed to
  NVRTC for all kernels and is set to `c++11` (which is the minimum
  requirement for Jitify itself). Jitify also supports the value
  `-std=c++03` for explicitly selecting the `C++03` standard.

- `--minify (-m)`

  This option is supported by Jitify only at preprocessing time and
  causes all runtime source code and headers to be "minified"
  (all comments and most whitespace is removed). This reduces the
  size of the source code and achieves a basic level of code
  obfuscation.

- `--no-replace-pragma-once (-no-replace-pragma-once)`

  This option is supported by Jitify only at preprocessing time and
  disables the automatic replacement of `#pragma once` with
  `#ifndef ...`.

- `--no-builtin-headers (-no-builtin-headers)`

  This option is supported by Jitify only at preprocessing time and
  disables the use of Jitify's built-in standard library header
  implementations.

- `--no-preinclude-workarounds (-no-preinclude-workarounds)`

  This option is supported by Jitify only at preprocessing time and
  disables the use of builtin workarounds for certain libraries
  (e.g., Thrust and CUB).

- `--cuda-std (-cuda-std)`

  [EXPERIMENTAL]
  This option is supported by Jitify only at preprocessing time and
  causes all instances of `std::foo` to be automatically replaced
  with `::cuda::std::foo`, with the intention of supporting the use of
  the libcudacxx header implementations instead of the system
  implementations. This is experimental because it does not currently
  support the transformation of `namespace std {` (as is used for
  specializations of standard library templates).

Linker options:

- `-l<library>`

  Specifies a device library to link with. This can be a static
  library, in which case the prefix/suffix will be added automatically
  if none is present (e.g., `-lfoo` is equivalent to `-llibfoo.a` in
  Linux systems), or a .ptx/.cubin/.fatbin/.o file, in which case the
  file type will be inferred from the extension.

- `-l.`

  Specifies that the current executable itself should be linked as a
  device library. The executable must have been compiled with `nvcc
  -rdc=true` for this to work. Note that this only links against the
  executable _file_, not the running application;
  while this provides
  access to offline-compiled device functions, it is not possible to
  share symbols between offline- and runtime-compiled code.

- `-L<dir>`

  Specifies a directory to search for linker files. For a given
  `-l<library>`, the unmodified `<library>` name is tried first,
  before searching for the file in the `-L` directories in the order
  they are listed.
