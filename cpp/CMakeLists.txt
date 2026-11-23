# libgpap — the GPAP engine core.
#
# Build requirements are deliberately thin: a C++17 compiler, CMake and the `cuda.h` driver
# header. NO NVIDIA driver (libcuda is opened at run time), NO CUDA toolkit and no nvcc: every
# CUDA kernel is Triton, compiled ahead of time by tools/build_kernels.py.
# If no toolkit is installed we fall back to the `cuda.h` bundled inside the triton wheel.
#
#   cmake -S cpp -B build && cmake --build build -j
#   ./build/gpap_c_api
cmake_minimum_required(VERSION 3.18)
# 0.2.0 / SOVERSION 1 — the ABI break. gpap_mode's values were renumbered (AUTO is 0
# now, so a zeroed options struct dispatches), gpap_flags / gpap_num_propagated gained a capacity
# argument, gpap_probe_count takes the options struct, and gpap_probe_options.distinct_vars became
# same_var with the sense INVERTED at the same offset. The rename is a compile error for a caller
# that recompiles -- but a caller that only RELINKS gets none of that, so the soname has to move:
# under SOVERSION 0 an old binary would have read GPAP_MODE_WAVE as SCALAR and passed the integer
# 1 where a gpap_probe_options* is now dereferenced. Bump both again on the next break.
# 0.3.0 — ADDITIVE: the gpap_sampler_* sampling entry points (cpp/src/sample.cpp). No
# existing symbol or struct changed, so SOVERSION stays 1.
# 0.4.0 — ADDITIVE: gpap_backend, gpap_engine_backend, gpap_engine_device_name (the OpenCL
# backend). No existing symbol or struct changed, so SOVERSION stays 1.
project(gpap VERSION 0.4.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------- embedding
# libgpap builds on its own (`cmake -S cpp`) or inside a host project (add_subdirectory or
# FetchContent with SOURCE_SUBDIR cpp). A host typically wants it static, without the examples,
# and, when it brings its own GPU layer, without CUDA: GPAP_WITH_CUDA=OFF needs neither cuda.h
# nor the generated kernels, and leaves the OpenCL plug, which is then the default backend.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(GPAP_TOP_LEVEL ON)
else()
  set(GPAP_TOP_LEVEL OFF)
endif()
option(GPAP_WITH_CUDA "Build the CUDA plug (needs cuda.h and the generated kernels)" ON)
option(GPAP_SHARED "Build libgpap as a shared library (OFF: static)" ON)
option(GPAP_BUILD_EXAMPLES "Build the examples" ${GPAP_TOP_LEVEL})

if(GPAP_TOP_LEVEL AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

if(GPAP_WITH_CUDA)
# ---------------------------------------------------------------- CUDA driver API
# The header only: the CUDA plug opens libcuda at run time (src/gpu_cuda.cpp), so neither the
# build nor loading the library needs the NVIDIA driver.
if(NOT GPAP_CUDA_INCLUDE_DIR)
  find_path(GPAP_CUDA_INCLUDE_DIR cuda.h
    HINTS /usr/local/cuda/include /usr/include ENV CUDA_HOME
    PATH_SUFFIXES include)
endif()
if(NOT GPAP_CUDA_INCLUDE_DIR)
  # No toolkit: use the driver headers triton ships (they are what the AOT kernels target).
  find_package(Python3 COMPONENTS Interpreter)
  if(Python3_Interpreter_FOUND)
    execute_process(
      COMMAND ${Python3_EXECUTABLE} -c
              "import os,triton;print(os.path.join(os.path.dirname(triton.__file__),'backends','nvidia','include'))"
      OUTPUT_VARIABLE _triton_inc OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if(_triton_inc AND EXISTS "${_triton_inc}/cuda.h")
      set(GPAP_CUDA_INCLUDE_DIR "${_triton_inc}")
      message(STATUS "Using cuda.h bundled with triton: ${GPAP_CUDA_INCLUDE_DIR}")
    endif()
  endif()
endif()
if(NOT GPAP_CUDA_INCLUDE_DIR)
  message(FATAL_ERROR "cuda.h not found. Pass -DGPAP_CUDA_INCLUDE_DIR=/path/to/include")
endif()

# ---------------------------------------------------------------- generated kernels
set(GPAP_GENERATED "${CMAKE_CURRENT_SOURCE_DIR}/src/generated/gpap_kernels.c")
if(NOT EXISTS "${GPAP_GENERATED}")
  message(FATAL_ERROR
    "Generated kernels missing. Run:\n"
    "    python cpp/tools/build_kernels.py\n"
    "(needs triton; add --arch NN to cross-compile for another GPU)")
endif()
endif()  # GPAP_WITH_CUDA

# ---------------------------------------------------------------- OpenCL kernel source
# The OpenCL backend compiles kernels/bcp.cl on the device at run time, so the library carries
# the source as a string. Nothing else is needed to BUILD that backend: the OpenCL library is
# opened at run time (src/gpu_opencl.cpp).
set(GPAP_CL_EMBEDDED "${CMAKE_CURRENT_BINARY_DIR}/gpap_cl_source.c")
add_custom_command(
  OUTPUT "${GPAP_CL_EMBEDDED}"
  COMMAND ${CMAKE_COMMAND}
          -DIN=${CMAKE_CURRENT_SOURCE_DIR}/kernels/bcp.cl
          -DOUT=${GPAP_CL_EMBEDDED}
          -DSYMBOL=gpap_cl_source
          -P ${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_source.cmake
  DEPENDS kernels/bcp.cl tools/embed_source.cmake
  COMMENT "Embedding kernels/bcp.cl"
  VERBATIM)

# ---------------------------------------------------------------- library
set(GPAP_SOURCES
  src/runtime.cpp
  src/formula.cpp
  src/engine.cpp
  src/readout.cpp
  src/calibrate.cpp
  src/probe.cpp
  src/check.cpp
  src/sample.cpp)
if(GPAP_WITH_CUDA)
  list(APPEND GPAP_SOURCES src/gpu_cuda.cpp)
endif()
list(APPEND GPAP_SOURCES src/gpu_opencl.cpp ${GPAP_CL_EMBEDDED})
if(GPAP_WITH_CUDA)
  list(APPEND GPAP_SOURCES ${GPAP_GENERATED})
endif()
if(GPAP_SHARED)
  add_library(gpap SHARED ${GPAP_SOURCES})
else()
  add_library(gpap STATIC ${GPAP_SOURCES})
endif()
add_library(gpap::gpap ALIAS gpap)
target_include_directories(gpap
  PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
          $<INSTALL_INTERFACE:include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
if(GPAP_WITH_CUDA)
  target_include_directories(gpap PRIVATE ${GPAP_CUDA_INCLUDE_DIR})
endif()
# PUBLIC so it travels through the exported target: <gpap/gpap.hpp> is C++17, and a consumer
# that only set CMAKE_CXX_STANDARD 11 would otherwise fail to compile it. (C consumers include
# gpap.h instead and are unaffected — that header is plain C99.)
target_compile_features(gpap PUBLIC cxx_std_17)
target_compile_options(gpap PRIVATE -Wall -Wextra -Wno-unused-parameter)
# ONE spelling of the version: gpap_version() reports this, rather than a second literal in
# runtime.cpp that nobody remembers to bump alongside it.
target_compile_definitions(gpap PRIVATE GPAP_VERSION_STR="${PROJECT_VERSION}"
                                         GPAP_WITH_CUDA=$<BOOL:${GPAP_WITH_CUDA}>)
# dlopen, for the GPU libraries (both plugs open theirs at run time), and dladdr, so
# gpap_build_id() can find and fingerprint the library's own file.
target_link_libraries(gpap PRIVATE ${CMAKE_DL_LIBS})
set_target_properties(gpap PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION 1)

# ---------------------------------------------------------------- examples
if(GPAP_BUILD_EXAMPLES)
add_executable(gpap_c_api examples/c_api.c)                   # plain C, the raw ABI
target_link_libraries(gpap_c_api PRIVATE gpap)

add_executable(gpap_sample examples/sample.cpp)              # C++, gpap::Sampler
target_link_libraries(gpap_sample PRIVATE gpap)

add_executable(gpap_assumptions examples/assumptions.cpp)    # C++, gpap::AssumptionChecker
target_link_libraries(gpap_assumptions PRIVATE gpap)

add_executable(gpap_probe examples/probe.cpp)                # C++, gpap::Prober
target_link_libraries(gpap_probe PRIVATE gpap)

add_executable(gpap_host_plug examples/host_plug.cpp)        # C++, a host's own gpap::Gpu
target_link_libraries(gpap_host_plug PRIVATE gpap)
endif()

# ---------------------------------------------------------------- install / find_package
# Only for libgpap built on its own: a host that embeds it installs what it needs itself.
if(GPAP_TOP_LEVEL)
# Downstream projects then just:
#     find_package(gpap REQUIRED)
#     target_link_libraries(my_app PRIVATE gpap::gpap)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS gpap EXPORT gpapTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT gpapTargets FILE gpapTargets.cmake NAMESPACE gpap::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/gpap)

# SameMINORVersion, not SameMajorVersion. This project encodes an ABI break in the MINOR
# version (0.1.0 -> 0.2.0 moved SOVERSION 0 -> 1; see the note at the top of this file), and
# CMake reads a 0.x major as 0 — so SameMajorVersion would have told a downstream
# `find_package(gpap 0.1 REQUIRED)` that the ABI-incompatible 0.2.0 satisfies it. The rule the
# soname states and the rule the package config enforces have to be the same rule.
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/gpapConfigVersion.cmake"
  VERSION ${PROJECT_VERSION} COMPATIBILITY SameMinorVersion)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/gpapConfig.cmake"
"include(CMakeFindDependencyMacro)\n\
include(\"\${CMAKE_CURRENT_LIST_DIR}/gpapTargets.cmake\")\n")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/gpapConfig.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/gpapConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/gpap)
endif()  # GPAP_TOP_LEVEL
