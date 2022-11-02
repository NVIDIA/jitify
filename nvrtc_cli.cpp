/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

// This provides a command-line interface for the NVIDIA CUDA C++ runtime
// compilation library (NVRTC). The interface and behavior are designed to
// match NVCC as closely as possible, so that the two can be used
// interchangeably in supported cases (e.g., for device compilation to PTX or
// CUBIN).

#include <cuda_runtime_api.h>  // For CUDART_VERSION
#include <nvrtc.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <list>
#include <sstream>
#include <string>
#include <vector>

#define CHECK_NVRTC(call)                                                  \
  do {                                                                     \
    nvrtcResult nvrtc_ret = call;                                          \
    if (nvrtc_ret != NVRTC_SUCCESS) {                                      \
      fatal_error() << #call " failed: " << nvrtcGetErrorString(nvrtc_ret) \
                    << std::endl;                                          \
      return nvrtc_ret;                                                    \
    }                                                                      \
  } while (0)

std::ostream& fatal_error() { return std::cerr << "nvrtc_cli fatal   : "; }

void print_usage(int argc, char* argv[]) {
  (void)argc;
  (void)argv;
  std::cout << R"(
Usage  : nvrtc_cli [options] <inputfile or '-'>

Options for managing the compiler driver
========================================
<inputfile or '-'>
        Specifies the input CUDA C++ source file to compile. If this is '-' (a
        single dash) then the source is read from standard input.

--help                                          (-h)
        Print this help information on this tool.

--version                                       (-V)
        Print version information of this tool.

--output-file <file>                            (-o)
        Specify name and location of the output file.

--ptx                                           (-ptx)
        Output PTX file.

--cubin                                         (-cubin)
        Output CUBIN file. This option is ignored if --ptx is specified.

--name-expression "<name>"                      (-ne)
        Specify a name expression denoting a __global__ function or
        __device__/__constant__ variable to be instantiated in the compilation.
        This option can be repeated.

--list-gpu-arch                                 (-arch-ls)
        List the virtual device architectures (compute_XX) supported by the
        compiler and exit.

--list-gpu-code                                 (-code-ls)
        List the real device architectures (sm_XX) supported by the compiler and
        exit.

--time <file or '-'>                            (-time)
        Generate a comma separated value table with the time taken by each
        compilation phase, and append it at the end of the file given as the
        option argument. If the file is empty, the column headings are generated
        in the first row of the table. If the file name is '-', the timing data
        is generated in stdout.

Options for NVRTC compilation
=============================
All other options are passed directly through to NVRTC.
See https://docs.nvidia.com/cuda/nvrtc/index.html#group__options
NOTE: Options to NVRTC that accept arguments must be specified using the form
  "--opt=val", not "--opt" "val".
)";
}

using milliseconds_double = std::chrono::duration<double, std::milli>;

struct CompiledProgram {
  std::string ptx;
  std::vector<char> cubin;
  std::vector<char> nvvm;
};

nvrtcResult compile_program(
    const char* name, const char* source,
    const std::vector<const char*>& options, std::string* log,
    CompiledProgram* compiled,
    const std::vector<const char*>& name_expressions = {},
    std::vector<std::string>* lowered_names = nullptr,
    milliseconds_double* compilation_time_ms = nullptr) {
  nvrtcProgram nvrtc_program;
  const int num_headers = 0;
  const char** headers = nullptr;
  const char** header_include_names = nullptr;
  CHECK_NVRTC(nvrtcCreateProgram(&nvrtc_program, source, name, num_headers,
                                 headers, header_include_names));
  for (const char* name_expr : name_expressions) {
    CHECK_NVRTC(nvrtcAddNameExpression(nvrtc_program, name_expr));
  }
  auto start_time = std::chrono::steady_clock::now();
  // Note: We name it the same as the function name so that the (deferred)
  // generated error message is correct.
  nvrtcResult nvrtcCompileProgram =
      ::nvrtcCompileProgram(nvrtc_program, (int)options.size(), options.data());
  if (compilation_time_ms) {
    auto compilation_time = std::chrono::steady_clock::now() - start_time;
    *compilation_time_ms = compilation_time;
  }
  if (log) {
    size_t log_size;
    CHECK_NVRTC(nvrtcGetProgramLogSize(nvrtc_program, &log_size));
    // Note: log_size includes null-terminator, and std::string is guaranteed to
    // include its own.
    log->resize(log_size - 1);
    CHECK_NVRTC(nvrtcGetProgramLog(nvrtc_program, &(*log)[0]));
  }
  CHECK_NVRTC(nvrtcCompileProgram);

  size_t ptx_size;
  CHECK_NVRTC(nvrtcGetPTXSize(nvrtc_program, &ptx_size));
  if (ptx_size == 1) ptx_size = 0;  // WAR for issue in NVRTC 11.4 -dlto
  if (ptx_size) {
    compiled->ptx.resize(ptx_size - 1);
    CHECK_NVRTC(nvrtcGetPTX(nvrtc_program, &compiled->ptx[0]));
  }

#if CUDART_VERSION >= 11010
  size_t cubin_size;
  CHECK_NVRTC(nvrtcGetCUBINSize(nvrtc_program, &cubin_size));
  if (cubin_size) {
    compiled->cubin.resize(cubin_size);
    CHECK_NVRTC(nvrtcGetCUBIN(nvrtc_program, compiled->cubin.data()));
  }
#endif

#if CUDART_VERSION >= 11040
  size_t nvvm_size;
  CHECK_NVRTC(nvrtcGetNVVMSize(nvrtc_program, &nvvm_size));
  if (nvvm_size) {
    compiled->nvvm.resize(nvvm_size);
    CHECK_NVRTC(nvrtcGetNVVM(nvrtc_program, compiled->nvvm.data()));
  }
#endif

  if (lowered_names) {
    lowered_names->reserve(name_expressions.size());
    lowered_names->clear();
    for (const char* name : name_expressions) {
      const char* lowered_name_c;
      CHECK_NVRTC(nvrtcGetLoweredName(nvrtc_program, name, &lowered_name_c));
      lowered_names->push_back(lowered_name_c);
    }
  }

  return NVRTC_SUCCESS;
}

bool startswith(const std::string& str, const std::string& prefix,
                size_t* prefix_length = nullptr) {
  bool match = str.size() >= prefix.size() &&
               std::equal(prefix.begin(), prefix.end(), str.begin());
  if (prefix_length) {
    *prefix_length = prefix.size();
  }
  return match;
}

std::string get_arch_string(const CompiledProgram& compiled,
                            int* arch_ptr = nullptr) {
  if (!compiled.ptx.empty()) {
    std::istringstream iss(compiled.ptx);
    for (std::string line; std::getline(iss, line);) {
      if (startswith(line, ".target")) {
        std::string arch_str = line.substr(line.find("_") + 1);
        if (arch_ptr) {
          *arch_ptr = std::stoi(arch_str);
        }
        return (compiled.cubin.empty() ? "compute_" : "sm_") + arch_str;
      }
    }
  }
  return "<unknown>";
}

nvrtcResult get_default_arch(int* default_arch) {
  CompiledProgram compiled;
  CHECK_NVRTC(compile_program("nvrtc_cli_default_arch",
                              /*source=*/"",
                              /*options=*/{}, /*log=*/nullptr, &compiled));
  get_arch_string(compiled, default_arch);
  return NVRTC_SUCCESS;
}

nvrtcResult get_supported_archs(std::vector<int>* supported_archs) {
#if CUDART_VERSION >= 11020
  int num_supported_archs;
  CHECK_NVRTC(nvrtcGetNumSupportedArchs(&num_supported_archs));
  supported_archs->resize(num_supported_archs);
  CHECK_NVRTC(nvrtcGetSupportedArchs(supported_archs->data()));
#elif CUDART_VERSION >= 11000
  *supported_archs = {35, 37, 50, 52, 53, 60, 61, 62, 70, 72, 75, 80};
#elif CUDART_VERSION >= 10010
  *supported_archs = {30, 32, 35, 37, 50, 52, 53, 60, 61, 62, 70, 72, 75};
#elif CUDART_VERSION >= 9000
  *supported_archs = {30, 32, 35, 37, 50, 52, 53, 60, 61, 62, 70, 72};
#else
  fatal_error() << "get_supported_archs() is not implemented for NVRTC < 9.0";
  return NVRTC_ERROR_INTERNAL_ERROR;
#endif
  return NVRTC_SUCCESS;
}

nvrtcResult print_supported_archs(const char* arch_type) {
  std::vector<int> supported_archs;
  int default_arch;
  CHECK_NVRTC(get_default_arch(&default_arch));
  CHECK_NVRTC(get_supported_archs(&supported_archs));
  for (int arch : supported_archs) {
    std::cout << arch_type << "_" << arch
              << (arch == default_arch ? " (default)" : "") << std::endl;
  }
  return NVRTC_SUCCESS;
}

std::string path_filename(const std::string& p) {
  // "/usr/local/myfile.dat" -> "myfile.dat"
  // "foo/bar" -> "bar"
  // "foo/bar/" -> ""
#if defined _WIN32 || defined _WIN64
  // Note that Windows supports both forward and backslash path separators.
  const char* sep = "\\/";
#else
  char sep = '/';
#endif
  size_t i = p.find_last_of(sep);
  if (i != std::string::npos) {
    return p.substr(i + 1);
  } else {
    return p;
  }
}

std::string remove_extension(const std::string& filename) {
  // "foo.bar" -> "foo"
  // "foo" -> "foo"
  // "foo.cat.bar" -> "foo.cat"
  return filename.substr(0, filename.find_last_of('.'));
}

nvrtcResult print_version() {
  std::cout << "nvrtc_cli: NVIDIA CUDA C++ runtime compilation library "
               "(NVRTC) compiler driver"
            << std::endl;
  CompiledProgram compiled;
  CHECK_NVRTC(compile_program("nvrtc_cli_build_version",
                              /*source=*/"",
                              /*options=*/{}, /*log=*/nullptr, &compiled));
  std::istringstream iss(compiled.ptx);
  for (std::string line; std::getline(iss, line);) {
    if (startswith(line, "// Compiler Build ID:") ||
        startswith(line, "// Cuda compilation tools, release") ||
        startswith(line, "// Based on")) {
      std::cout << line.substr(3) << std::endl;
    }
  }
  return NVRTC_SUCCESS;
}

void write_time_info(std::ostream& stream, const std::string& input_filename,
                     const std::string& phase_name,
                     const std::string& output_filename,
                     const std::string& arch,
                     std::chrono::duration<double, std::milli> time_ms,
                     bool include_header = false) {
  if (include_header) {
    stream << "source file name , phase name , phase input files , phase "
              "output file , arch , tool, metric , unit"
           << std::endl;
  }
  stream << input_filename << " , " << phase_name << " , " << input_filename
         << " , " << output_filename << " , " << arch << " , nvrtc_cli , "
         << time_ms.count() << " , ms" << std::endl;
}

bool parse_option_argument(const char* arg, const std::string& sarg,
                           size_t optname_length, int argc, char* argv[],
                           int* arg_i, const char** result) {
  if (sarg.size() == optname_length) {
    if (*arg_i == argc) {
      fatal_error() << "argument expected after '" << sarg << "'" << std::endl;
      return false;
    }
    *result = argv[(*arg_i)++];
  } else {
    if (sarg[optname_length] != '=') {
      fatal_error() << "argument expected for '" << sarg << "'" << std::endl;
      return false;
    }
    *result = &arg[optname_length + 1];
  }
  return true;
}

bool failed_to(const std::string& verb, const std::string& filename) {
  fatal_error() << "Failed to " << verb << " file '" << filename << "'"
                << std::endl;
  return false;
}

bool write_to_file(const std::string& filename, const std::string& data) {
  std::ofstream stream(filename.c_str());
  if (!stream) return failed_to("open", filename);
  stream << data;
  if (!stream) return failed_to("write to", filename);
  return true;
}

bool write_to_file(const std::string& filename, const std::vector<char>& data) {
  std::ofstream stream(filename.c_str(), std::ios::out | std::ios::binary);
  if (!stream) return failed_to("open", filename);
  stream.write(data.data(), data.size());
  if (!stream) return failed_to("write to", filename);
  return true;
}

int main(int argc, char* argv[]) {
  auto start_time = std::chrono::steady_clock::now();
  std::string output_filename;
  std::vector<const char*> name_expressions;
  std::vector<const char*> options;
  const char* input_filename = nullptr;
  const char* time_output_filename = nullptr;
  bool output_ptx = false;
  bool output_cubin = false;
  bool output_nvvm = false;
  bool have_arch_flag = false;
  size_t optname_length;
  int i = 1;
  while (i < argc) {
    const char* arg = argv[i++];
    std::string sarg = arg;
    if (sarg == "-h" || sarg == "--help") {
      print_usage(argc, argv);
      return EXIT_SUCCESS;
    } else if (sarg == "-V" || sarg == "--version") {
      print_version();
      return EXIT_SUCCESS;
    } else if (sarg == "-arch-ls" || sarg == "--list-gpu-arch") {
      CHECK_NVRTC(print_supported_archs("compute"));
      return EXIT_SUCCESS;
    } else if (sarg == "-code-ls" || sarg == "--list-gpu-code") {
      CHECK_NVRTC(print_supported_archs("sm"));
      return EXIT_SUCCESS;
    } else if (startswith(sarg, "-o", &optname_length) ||
               startswith(sarg, "--output-file", &optname_length)) {
      const char* output_filename_c;
      if (!parse_option_argument(arg, sarg, optname_length, argc, argv, &i,
                                 &output_filename_c)) {
        return EXIT_FAILURE;
      }
      output_filename = output_filename_c;
    } else if (sarg == "-ptx" || sarg == "--ptx") {
      output_ptx = true;
    } else if (sarg == "-cubin" || sarg == "--cubin") {
#if CUDART_VERSION >= 11010
      output_cubin = true;
#else
      fatal_error()
          << "compilation to CUBIN is not supported by this version of NVRTC"
          << std::endl;
      return EXIT_FAILURE;
#endif
    } else if (startswith(sarg, "-ne", &optname_length) ||
               startswith(sarg, "--name-expression", &optname_length)) {
      const char* name_expression;
      if (!parse_option_argument(arg, sarg, optname_length, argc, argv, &i,
                                 &name_expression)) {
        return EXIT_FAILURE;
      }
      name_expressions.push_back(name_expression);
    } else if (startswith(sarg, "-time", &optname_length) ||
               startswith(sarg, "--time", &optname_length)) {
      if (!parse_option_argument(arg, sarg, optname_length, argc, argv, &i,
                                 &time_output_filename)) {
        return EXIT_FAILURE;
      }
    } else if (startswith(sarg, "-L", &optname_length) ||
               startswith(sarg, "--library-path", &optname_length)) {
      // Ignore library paths (-l will still cause an error).
      if (sarg.size() == optname_length) {
        i++;  // Skip over library name argument
      }
    } else if (sarg == "-") {
      if (input_filename) {
        fatal_error() << "unexpected argument '" << arg
                      << "'; inputfile already specified as '" << input_filename
                      << "'" << std::endl;
        return EXIT_FAILURE;
      }
      input_filename = arg;
    } else if (sarg[0] == '-') {
      if (startswith(sarg, "-arch") || startswith(sarg, "--gpu-architecture")) {
        have_arch_flag = true;
      } else if (sarg == "-dlto" || sarg == "--dlink-time-opt") {
        output_nvvm = true;
      }
      // Pass option through to NVRTC.
      options.push_back(arg);
    } else {
      if (input_filename) {
        fatal_error() << "unexpected argument '" << arg
                      << "'; inputfile already specified as '" << input_filename
                      << "'" << std::endl;
        return EXIT_FAILURE;
      }
      input_filename = arg;
    }
  }

  if (!input_filename) {
    fatal_error()
        << "No input files specified; use option --help for more information"
        << std::endl;
    return EXIT_FAILURE;
  }

  std::string default_arch_flag;
  if (output_cubin && !have_arch_flag) {
    // Use real instead of virtual default architecture to enable CUBIN output.
    int default_arch;
    CHECK_NVRTC(get_default_arch(&default_arch));
    default_arch_flag = "-arch=sm_" + std::to_string(default_arch);
    options.push_back(default_arch_flag.c_str());
  }

  if (output_nvvm && (output_ptx || output_cubin)) {
    fatal_error()
        << "Link-time optimization (--dlink-time-opt/-dlto) is not compatible "
           "with PTX or CUBIN output (--ptx/-ptx or --cubin/-cubin)"
        << std::endl;
    return EXIT_FAILURE;
  }

  if (!output_ptx && !output_cubin && !output_nvvm) {
    output_ptx = true;
  }

  if (input_filename != std::string("-") && output_filename.empty()) {
    const char* extension =
        output_ptx ? ".ptx" : output_cubin ? ".cubin" : ".nvvm";
    output_filename =
        remove_extension(path_filename(input_filename)) + extension;
  }

  std::istream* input_stream;
  std::ifstream input_file_stream;
  if (input_filename != std::string("-")) {
    input_file_stream.open(input_filename);
    if (!input_file_stream) {
      fatal_error() << "Failed to open input file '" << input_filename << "'"
                    << std::endl;
      return EXIT_FAILURE;
    }
    input_stream = &input_file_stream;
  } else {
    // Read from stdin.
    input_stream = &std::cin;
    input_filename = "<stdin>";
  }
  std::stringstream source_buffer;
  source_buffer << input_stream->rdbuf();
  std::string source_str = source_buffer.str();
  const char* source = source_str.c_str();

  std::string log;
  CompiledProgram compiled;
  std::vector<std::string> lowered_names;
  milliseconds_double compilation_time;
  nvrtcResult result =
      compile_program(input_filename, source, options, &log, &compiled,
                      name_expressions, &lowered_names, &compilation_time);
  std::cerr << log;

  std::ofstream time_file_stream;
  std::ostream* time_stream = &time_file_stream;
  if (time_output_filename) {
    bool write_header;
    if (time_output_filename == std::string("-")) {
      time_stream = &std::cout;
      write_header = true;
    } else {
      time_file_stream.open(time_output_filename,
                            std::ios::out | std::ios::app);
      if (!time_file_stream) {
        fatal_error() << "Failed to open time output file '"
                      << time_output_filename << "'" << std::endl;
        return EXIT_FAILURE;
      }
      write_header = time_file_stream.tellp() == 0;
    }
    write_time_info(*time_stream, input_filename, "nvrtcCompileProgram",
                    output_filename, get_arch_string(compiled),
                    compilation_time, write_header);
    if (!*time_stream) {
      fatal_error() << "Failed to write to time output file '"
                    << time_output_filename << "'" << std::endl;
      return EXIT_FAILURE;
    }
  }

  if (result != NVRTC_SUCCESS) return EXIT_FAILURE;

  if (!output_filename.empty()) {
    if (output_ptx) {
      if (!write_to_file(output_filename, compiled.ptx)) return EXIT_FAILURE;
    } else if (output_cubin) {
      if (!write_to_file(output_filename, compiled.cubin)) return EXIT_FAILURE;
    } else if (output_nvvm) {
      if (!write_to_file(output_filename, compiled.nvvm)) return EXIT_FAILURE;
    }
  }

  if (time_output_filename) {
    auto total_time = std::chrono::steady_clock::now() - start_time;
    write_time_info(*time_stream, /*input_filename=*/"", "nvrtc_cli (driver)",
                    /*output_filename=*/"", /*arch=*/"",
                    total_time - compilation_time);
    if (!*time_stream) {
      fatal_error() << "Failed to write to time output file '"
                    << time_output_filename << "'" << std::endl;
      return EXIT_FAILURE;
    }
  }

  if (!name_expressions.empty()) {
    std::cout << "Lowered name expressions:" << std::endl;
    for (size_t i = 0; i < name_expressions.size(); ++i) {
      std::cout << "\"" << name_expressions[i] << "\" -> " << lowered_names[i]
                << std::endl;
    }
  }

  if (output_filename.empty() && output_ptx) {
    std::cout << compiled.ptx;
  }

  return EXIT_SUCCESS;
}
