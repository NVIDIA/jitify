cmake_minimum_required(VERSION 3.9)
project(jitify LANGUAGES CXX CUDA)

option(ASAN "Enable address sanitizer in debug build" ON)

# C++ compiler options.
set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CUDA_STANDARD 17)  # Doesn't work?
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ")
if (MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /WX -D_CRT_SECURE_NO_WARNINGS")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2")
else()
  set(CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS} -O3 \
      -Wall -Wextra -Wconversion -Wshadow -fmessage-length=80 \
      -D_FILE_OFFSET_BITS=64 \
      ")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
  if (ASAN)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} \
        -fsanitize=undefined,address")
  else()
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} \
        -fsanitize=undefined")
  endif()
endif()

# CUDA compiler options.
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
if (MSVC)
  set(CMAKE_CUDA_FLAGS
    "${CMAKE_CUDA_FLAGS} -Xcompiler=\"/WX\" -D_CRT_SECURE_NO_WARNINGS -rdc=true")
  set(CMAKE_CUDA_FLAGS_RELEASE
    "${CMAKE_CUDA_FLAGS_RELEASE} -O3 -Xcompiler=\"/O2\"")
else()
  set(CMAKE_CUDA_FLAGS
    "${CMAKE_CUDA_FLAGS} -O3 \
    -Xcompiler=\"-Wall -Wextra -Wconversion -Wshadow -fmessage-length=80 \" \
    -D_FILE_OFFSET_BITS=64 \
    -rdc=true \
    ")
  set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g")
endif()

find_package(CUDA REQUIRED) # Required for CUDA_INCLUDE_DIRS

# Add macro definitions used in tests.
if (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0.0)
  # CCCL's include directories have moved in CUDA 13 compared to CUDA 12.
  # On Windows, CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES includes both include paths, which must be separated before including in compile definitions.
  # On *nix, only the main ctk include dir is included
  # It may be cleaner to switch to the more modern find_package(CUDAToolkit) and find_package(CCCL)
  list(LENGTH CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES ctk_inc_dirs_length)
  if (ctk_inc_dirs_length GREATER 1)
    list(GET CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES 0 cuda_inc_dir)
    list(GET CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES 1 cccl_inc_dir)
  else()
    list(GET CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES 0 cuda_inc_dir)
    # Check the default location within the CTK if not in the variable.
    if (EXISTS "${cuda_inc_dir}/cccl")
      set(cccl_inc_dir "${cuda_inc_dir}/cccl")
    else()
      set(cccl_inc_dir "${cuda_inc_dir}")
    endif()
  endif()
  add_compile_definitions(
      CUDA_INC_DIR="${cuda_inc_dir}"
      CUB_DIR="${cccl_inc_dir}")
  unset(cccl_inc_dir)
  unset(cuda_inc_dir)
  unset(inc_dirs_length)
else()
  add_compile_definitions(
      CUDA_INC_DIR="${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
      CUB_DIR=${CUDA_INC_DIR})
endif()
# Copy the example_headers directory for use at runtime by tests.
file(COPY example_headers DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# Windows requires linking to DbgHelp for UnDecorateSymbolName.
if (MSVC)
link_libraries(DbgHelp)
endif()

# -----------
# Google Test
# -----------
# Download and unpack googletest at configure time.
configure_file(CMakeLists.txt.in googletest-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
if(result)
  message(FATAL_ERROR "CMake step for googletest failed: ${result}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
if(result)
  message(FATAL_ERROR "Build step for googletest failed: ${result}")
endif()
# Prevent overriding the parent project's compiler/linker
# settings on Windows.
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
# Add googletest directly to our build. This defines
# the gtest and gtest_main targets.
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googletest-src
                 ${CMAKE_CURRENT_BINARY_DIR}/googletest-build
                 EXCLUDE_FROM_ALL)

# ----
# NVTX
# ----
# Download and unpack nvtx at configure time.
configure_file(CMakeLists.txt.in nvtx-download/CMakeLists.txt)
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/nvtx-src/c
                 ${CMAKE_CURRENT_BINARY_DIR}/nvtx-build
                 EXCLUDE_FROM_ALL)

# ----
# Executable utilities
# ----
add_executable(stringify stringify.cpp)
function(add_stringify_command arg)
  add_custom_command(
      OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${arg}.jit
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      COMMAND $<TARGET_FILE:stringify> ${CMAKE_CURRENT_SOURCE_DIR}/${arg} > ${arg}.jit
      DEPENDS stringify)
endfunction()
add_executable(jitify2_preprocess jitify2_preprocess.cpp)
target_include_directories(jitify2_preprocess PRIVATE ${CUDA_INCLUDE_DIRS})
if (NOT WIN32)
  find_package(Threads REQUIRED)
  target_link_libraries(jitify2_preprocess PRIVATE ${CMAKE_DL_LIBS} ${CMAKE_THREAD_LIBS_INIT})
endif()

# ----
# Generate files for use in tests
# ----
add_stringify_command(example_headers/my_header1.cuh)
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/jitify2_test_kernels.cu.jit.hpp
           ${CMAKE_CURRENT_BINARY_DIR}/jitify2_test_kernels.cu.headers.jit.cpp
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMAND $<TARGET_FILE:jitify2_preprocess> -i --minify
                -o ${CMAKE_CURRENT_BINARY_DIR}
                -s jitify2_test_kernels.cu.headers
                jitify2_test_kernels.cu
    DEPENDS jitify2_preprocess)
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/jitify_2nd_compilation_unit.cu
     "#include \"jitify2.hpp\"")

# -----
# Tests
# -----
enable_testing()
set(TESTS
    jitify2_test
    jitify2_test_static
)
foreach(test ${TESTS})
  if (${test} MATCHES "_static$")
    STRING(REGEX REPLACE "_static$" "" test_base ${test})
  else()
    set(test_base ${test})
  endif()
  # Note that generated headers are listed as source files to force dependency.
  add_executable(
      ${test} EXCLUDE_FROM_ALL ${test_base}.cu
      ${CMAKE_CURRENT_BINARY_DIR}/jitify_2nd_compilation_unit.cu
      ${CMAKE_CURRENT_BINARY_DIR}/example_headers/my_header1.cuh.jit
      ${CMAKE_CURRENT_BINARY_DIR}/jitify2_test_kernels.cu.jit.hpp
      ${CMAKE_CURRENT_BINARY_DIR}/jitify2_test_kernels.cu.headers.jit.cpp)
  # Ensure the generated headers can be found.
  target_include_directories(${test} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
  # Ensure the main jitify header can be found.
  target_include_directories(${test} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(${test} gtest_main nvtx3-cpp)
  set_property(TARGET ${test} PROPERTY CUDA_ARCHITECTURES OFF)
  target_compile_definitions(${test}
                               PUBLIC JITIFY_ENABLE_NVTX=1)
  if (${test} MATCHES "_static$")
    target_compile_definitions(${test}
                               PUBLIC JITIFY_LINK_CUDA_STATIC=1
                               PUBLIC JITIFY_LINK_NVRTC_STATIC=1
                               PUBLIC JITIFY_LINK_NVJITLINK_STATIC=1)
    target_link_libraries(${test} cuda nvrtc)
    if (CUDA_VERSION_MAJOR GREATER_EQUAL 12)
      target_link_libraries(${test} nvJitLink)
    endif()
  endif()
  if (NOT WIN32)
    target_link_libraries(${test} ${CMAKE_DL_LIBS})
  endif()
  add_test(NAME ${test} COMMAND ${test})
endforeach(test)
# Add "check" command that *builds and* runs tests, with verbose output.
# (The default "test" command neither builds nor gives verbose output).
# --build-config is required for multi-config generators, and uses COMMAND_EXPAND_LISTS to ensure that the flag and value are not provided as a single string to ctest
add_custom_target(check ALL
    COMMAND ${CMAKE_CTEST_COMMAND} --verbose "$<IF:$<STREQUAL:$<CONFIG>,>,,--build-config;$<CONFIG>>"
    DEPENDS ${TESTS}
    COMMAND_EXPAND_LISTS
)

# ----
# Docs
# ----
find_package(Doxygen)
if(DOXYGEN_FOUND)
  # Copy source file so that Doxygen can find it.
  # TODO: A better approach would be to configure a Doxyfile.in template file
  # with the appropriate input name and/or output path.
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/jitify2.hpp
                 ${CMAKE_CURRENT_BINARY_DIR}/
                 COPYONLY)
  set(doxyfile ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile2)
  add_custom_target(
      doc
      COMMAND ${DOXYGEN_EXECUTABLE} ${doxyfile}
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      DEPENDS ${doxyfile})
endif()
