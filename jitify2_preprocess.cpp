/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
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
  jitify2_preprocess is a tool to preprocess CUDA source files with Jitify and
    generate serialized Jitify PreprocessedProgram objects that include all
    header dependencies. These serialized programs can then be shipped with a
    Jitify application and loaded efficiently at runtime ready for compilation.
 */

#include "jitify2.hpp"

#include <cctype>
#include <cstdlib>
#include <iostream>
#include <regex>
#include <string>

void write_binary_as_c_array(const std::string& varname, std::istream& istream,
                             std::ostream& ostream) {
  static const char* const hex_digits = "0123456789abcdef";
  std::string varname_data = varname;
  ostream << "const unsigned char " << varname_data << "[] = {\n";
  unsigned int n = 0;
  while (true) {
    unsigned char c;
    istream.read((char*)&c, 1);
    if (!istream.good()) break;
    ostream << '0' << 'x' << hex_digits[c >> 4] << hex_digits[c & 0xF] << ",";
    if (++n % 16 == 0) ostream << "\n";
  }
  ostream << "\n};\n";
}

void write_serialized_program_as_cpp_header(
    std::istream& istream, std::ostream& ostream,
    const std::string& source_varname,
    const std::string& shared_headers_varname) {
  std::string include_guard_name =
      "JITIFY_INCLUDE_GUARD_PROGRAM_" + source_varname;
  ostream << "#ifndef " << include_guard_name << "\n#define "
          << include_guard_name << "\n";
  ostream << "#include <jitify2.hpp>\n";
  if (!shared_headers_varname.empty()) {
    ostream << "extern const jitify2::StringMap* " << shared_headers_varname
            << ";\n";
  }
  std::string vs = source_varname + "_serialized";
  write_binary_as_c_array(vs, istream, ostream);
  auto ind = [](int n) { return std::string(2 * n, ' '); };
  ostream << "const jitify2::PreprocessedProgram " << source_varname << " =\n"
          << ind(2) << "jitify2::PreprocessedProgram::deserialize(\n"
          << ind(4) << "{reinterpret_cast<const char*>(\n"
          << ind(6) << vs << "),\n"
          << ind(4) << " sizeof(" << vs << ")});\n";
  ostream << "#endif  // " << include_guard_name << "\n";
}

void write_serialized_headers_as_cpp_source(std::istream& istream,
                                            std::ostream& ostream,
                                            const std::string& varname) {
  ostream << "#define JITIFY_SERIALIZATION_ONLY\n#include <jitify2.hpp>\n";
  std::string vs = varname + "_serialized";
  auto ind = [](int n) { return std::string(2 * n, ' '); };
  ostream << "static ";
  write_binary_as_c_array(vs, istream, ostream);
  ostream << "const jitify2::StringMap* " << varname << " = [] {\n"
          << ind(1) << "static jitify2::StringMap header_sources;\n"
          << ind(1) << "bool ok = jitify2::serialization::deserialize(\n"
          << ind(3) << "{reinterpret_cast<const char*>(\n"
          << ind(5) << vs << "),\n"
          << ind(3) << " sizeof(" << vs << ")},\n"
          << ind(3) << "&header_sources);\n"
          << ind(1) << "return ok ? &header_sources : nullptr;\n"
          << "}();\n";
}

// Replaces non-alphanumeric characters with '_' and prepends '_' if the string
// begins with a digit.
std::string sanitize_varname(const std::string& s) {
  std::string r = s;
  if (std::isdigit(r[0])) {
    r = '_' + r;
  }
  for (std::string::iterator it = r.begin(); it != r.end(); ++it) {
    if (!std::isalnum(*it)) {
      *it = '_';
    }
  }
  return r;
}

bool read_file(const std::string& fullpath, std::string* content) {
  std::ifstream file(fullpath.c_str(), std::ios::binary | std::ios::ate);
  if (!file) return false;
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  content->resize(size);
  file.read(&(*content)[0], size);
  return true;
}

bool make_directories_for(const std::string& filename) {
  using jitify2::detail::make_directories;
  using jitify2::detail::path_base;
  if (!make_directories(path_base(filename))) {
    std::cerr << "Error creating directories for output file " << filename
              << std::endl;
    return false;
  }
  return true;
}

void print_usage() {
  std::cout << "jitify2_preprocess - Preprocesses source files into serialized "
               "Jitify programs."
            << std::endl;
  std::cout << R"(
Usage:
jitify2_preprocess \
  source.cu [source2.cu ...]          Input source files.
  [-option -option=value ...]         Compiler options.
  [-i / --include-style]              Write output for inclusion in C++ source.
  [-o / --output-directory <dir>]     Write output files to the specified dir.
  [-p / --variable-prefix <prefix>]   Prefix to add to variable names (see -i).
  [-s / --shared-headers <filename>]  Write headers to a separate file.
  [-v / --verbose]                    Print header locations.
  [-h / --help]                       Show this help.

jitify2_preprocess is a tool to preprocess CUDA source files with jitify and
generate serialized jitify PreprocessedProgram objects that include all header
dependencies. These serialized programs can then be shipped with a jitify
application and loaded efficiently at runtime ready for compilation.

Each input source file is preprocessed and serialized with jitify to
"<filename>.jit".

The -i option causes program output files to be written as C++ headers with the
extension ".jit.hpp". These can then be #include'd in an application and the
deserialized programs can be accessed directly as global PreprocessedProgram
variables. The embedded variables are named based on a sanitized version of
"<filename>.jit". E.g., "my-source.cu" generates "my-source.cu.jit.hpp"
containing "const jitify2::PreprocessedProgram <prefix>my_source_cu_jit;".
(Note that this variable will be in an error state if deserialization fails).

The -s / --shared-headers option causes all headers to be serialized to a
separate file "<filename>.jit" instead of being stored inside each serialized
program. When combined with the -i option, this output is written as a C++
source file "<filename>.jit.cpp" to be linked into a C++ application. The
embedded variable is named based on a sanitized version of "<filename>.jit".
E.g., "--shared-headers my-headers" generates "my-headers.jit.cpp" containing
"const jitify2::StringMap* <prefix>my_headers_jit;". (Note that this variable
will be null if deserialization fails).)"
            << std::endl;
}

int main(int argc, char* argv[]) {
  (void)argc;
  using namespace jitify2;
  using jitify2::detail::path_join;
  std::string shared_headers_filename;
  std::string output_dir;
  std::string varname_prefix;
  StringVec options;
  StringVec source_filenames;
  bool write_as_cpp_headers = false;
  bool verbose = false;
  const char* arg_c;
  while ((arg_c = *++argv)) {
    std::string arg = arg_c;
    if (arg[0] == '-') {
      if (arg == "-h" || arg == "--help") {
        print_usage();
        return EXIT_SUCCESS;
      } else if (arg == "-s" || arg == "--shared-headers") {
        arg_c = *++argv;
        if (!arg_c) {
          std::cerr << "Expected filename after -s" << std::endl;
          return EXIT_FAILURE;
        }
        shared_headers_filename = arg_c;
      } else if (arg == "-o" || arg == "--output-directory") {
        arg_c = *++argv;
        if (!arg_c) {
          std::cerr << "Expected directory after -o / --output-directory"
                    << std::endl;
          return EXIT_FAILURE;
        }
        output_dir = arg_c;
      } else if (arg == "-p" || arg == "--variable-prefix") {
        arg_c = *++argv;
        if (!arg_c) {
          std::cerr << "Expected prefix after -p / --variable-prefix"
                    << std::endl;
          return EXIT_FAILURE;
        }
        varname_prefix = arg_c;
      } else if (arg == "-i" || arg == "--include-style") {
        write_as_cpp_headers = true;
      } else if (arg == "-v" || arg == "--verbose") {
        verbose = true;
      } else {
        options.push_back(arg);
      }
    } else {
      source_filenames.push_back(arg);
    }
  }
  if (source_filenames.empty()) {
    std::cerr << "Expected at least one source file" << std::endl;
    return EXIT_FAILURE;
  }
  bool share_headers = !shared_headers_filename.empty();
  std::string shared_headers_varname;
  if (share_headers) {
    shared_headers_varname =
        sanitize_varname(varname_prefix + shared_headers_filename + ".jit");
  }
  StringMap all_header_sources;
  for (const std::string& source_filename : source_filenames) {
    std::string source;
    if (!read_file(source_filename, &source)) {
      std::cerr << "Error reading source file " << source_filename << std::endl;
      return EXIT_FAILURE;
    }

    PreprocessedProgram preprocessed =
        Program(source_filename, source)->preprocess(options);
    if (!preprocessed) {
      std::cerr << "Error processing source file " << source_filename << "\n"
                << preprocessed.error() << std::endl;
      return EXIT_FAILURE;
    }
    if (verbose && !preprocessed->header_log().empty()) {
      std::cout << preprocessed->header_log() << std::endl;
    }
    if (!preprocessed->compile_log().empty()) {
      std::cout << preprocessed->compile_log() << std::endl;
    }

    if (share_headers) {
      // TODO: Check that there aren't weird header name issues when merging
      // multiple source files like this.
      all_header_sources.insert(preprocessed->header_sources().begin(),
                                preprocessed->header_sources().end());
    }

    if (write_as_cpp_headers) {
      std::stringstream ss(std::stringstream::in | std::stringstream::out |
                           std::stringstream::binary);
      preprocessed->serialize(ss, /*include_headers = */ !share_headers);
      std::string source_varname =
          sanitize_varname(varname_prefix + source_filename + ".jit");
      std::string output_filename =
          path_join(output_dir, source_filename + ".jit.hpp");
      if (!make_directories_for(output_filename)) return EXIT_FAILURE;
      std::ofstream file(output_filename, std::ios::binary);
      write_serialized_program_as_cpp_header(ss, file, source_varname,
                                             shared_headers_varname);
      if (!file) {
        std::cerr << "Error writing output to " << output_filename << std::endl;
        return EXIT_FAILURE;
      }
    } else {
      std::string output_filename =
          path_join(output_dir, source_filename + ".jit");
      if (!make_directories_for(output_filename)) return EXIT_FAILURE;
      std::ofstream file(output_filename, std::ios::binary);
      preprocessed->serialize(file, /*include_headers = */ !share_headers);
      if (!file) {
        std::cerr << "Error writing output to " << output_filename << std::endl;
        return EXIT_FAILURE;
      }
    }
  }
  if (share_headers) {
    if (write_as_cpp_headers) {
      std::stringstream ss(std::stringstream::in | std::stringstream::out |
                           std::stringstream::binary);
      serialization::serialize(ss, all_header_sources);
      std::string output_filename =
          path_join(output_dir, shared_headers_filename + ".jit.cpp");
      if (!make_directories_for(output_filename)) return EXIT_FAILURE;
      std::ofstream file(output_filename, std::ios::binary);
      write_serialized_headers_as_cpp_source(ss, file, shared_headers_varname);
      if (!file) {
        std::cerr << "Error writing output to " << output_filename << std::endl;
        return EXIT_FAILURE;
      }
    } else {
      std::string output_filename = shared_headers_filename + ".jit";
      if (!make_directories_for(output_filename)) return EXIT_FAILURE;
      std::ofstream file(output_filename, std::ios::binary);
      serialization::serialize(file, all_header_sources);
      if (!file) {
        std::cerr << "Error writing output to " << output_filename << std::endl;
        return EXIT_FAILURE;
      }
    }
  }
  return EXIT_SUCCESS;
}
