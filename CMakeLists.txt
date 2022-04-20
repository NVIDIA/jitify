cmake_minimum_required(VERSION 3.9)
project(jitify LANGUAGES CXX CUDA)

# C++ compiler options.
set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CUDA_STANDARD 11)  # Doesn't work?
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ")
if (MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /WX")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2")
else()
  set(CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -Wconversion -Wshadow")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
endif()

# CUDA compiler options.
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
if (MSVC)
  set(CMAKE_CUDA_FLAGS
    "${CMAKE_CUDA_FLAGS} -Xcompiler=\"/WX\" -rdc=true")
  set(CMAKE_CUDA_FLAGS_RELEASE
    "${CMAKE_CUDA_FLAGS_RELEASE} -O3 -Xcompiler=\"/O2\"")
else()
  set(CMAKE_CUDA_FLAGS
    "${CMAKE_CUDA_FLAGS} -Xcompiler=\"-Wall -Wextra -Wconversion -Wshadow\" -O3 -rdc=true")
  set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g")
endif()

find_package(CUDA REQUIRED) # Required for CUDA_INCLUDE_DIRS

# Add macro definitions used in tests.
add_compile_definitions(
    CUDA_INC_DIR="${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
    CUB_DIR=${CUDA_INC_DIR})
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
# Executable utilities
# ----
add_executable(stringify stringify.cpp)
function(add_stringify_command arg)
  add_custom_command(
      OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${arg}.jit
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      COMMAND ./stringify ${CMAKE_CURRENT_SOURCE_DIR}/${arg} > ${arg}.jit
      DEPENDS stringify)
endfunction()
add_executable(jitify2_preprocess jitify2_preprocess.cpp)
target_include_directories(jitify2_preprocess PRIVATE ${CUDA_INCLUDE_DIRS})
if (NOT WIN32)
  target_link_libraries(jitify2_preprocess PRIVATE ${CMAKE_DL_LIBS})
endif()

# ----
# Generate files for use in tests
# ----
add_stringify_command(example_headers/my_header1.cuh)
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/jitify2_test_kernels.cu.jit.hpp
           ${CMAKE_CURRENT_BINARY_DIR}/jitify2_test_kernels.cu.headers.jit.cpp
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/jitify2_preprocess -i --minify
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
)
foreach(test ${TESTS})
  # Note that generated headers are listed as source files to force dependency.
  add_executable(
      ${test} EXCLUDE_FROM_ALL ${test}.cu
      ${CMAKE_CURRENT_BINARY_DIR}/jitify_2nd_compilation_unit.cu
      ${CMAKE_CURRENT_BINARY_DIR}/example_headers/my_header1.cuh.jit
      ${CMAKE_CURRENT_BINARY_DIR}/jitify2_test_kernels.cu.jit.hpp
      ${CMAKE_CURRENT_BINARY_DIR}/jitify2_test_kernels.cu.headers.jit.cpp)
  # Ensure the generated headers can be found.
  target_include_directories(${test} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
  # Ensure the main jitify header can be found.
  target_include_directories(${test} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(${test} gtest_main)
  set_property(TARGET ${test} PROPERTY CUDA_ARCHITECTURES OFF)
  if (NOT WIN32)
    target_link_libraries(${test} ${CMAKE_DL_LIBS})
  endif()
  add_test(NAME ${test} COMMAND ${test})
endforeach(test)
# Add "check" command that *builds and* runs tests, with verbose output.
# (The default "test" command neither builds nor gives verbose output).
add_custom_target(check ALL COMMAND ${CMAKE_CTEST_COMMAND} --verbose
                  DEPENDS ${TESTS})

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
