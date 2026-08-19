# libgpap — the GPAP engine core.
#
# Build requirements are deliberately thin: a C++17 compiler, CMake, the CUDA *driver* library
# (libcuda, which ships with the NVIDIA driver) and the `cuda.h` driver header. NO CUDA toolkit
# and no nvcc: every GPU kernel is Triton, compiled ahead of time by tools/build_kernels.py.
# If no toolkit is installed we fall back to the `cuda.h` bundled inside the triton wheel.
#
#   cmake -S cpp -B build && cmake --build build -j
#   ./build/gpap_c_api
cmake_minimum_required(VERSION 3.18)
project(gpap VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ---------------------------------------------------------------- CUDA driver API
find_library(GPAP_CUDA_DRIVER
  NAMES cuda
  HINTS /usr/lib/wsl/lib /usr/lib/x86_64-linux-gnu /usr/local/cuda/lib64
        /usr/local/cuda/lib64/stubs
  DOC "libcuda (NVIDIA driver library)")
if(NOT GPAP_CUDA_DRIVER)
  message(FATAL_ERROR "libcuda not found. Install the NVIDIA driver, or pass "
                      "-DGPAP_CUDA_DRIVER=/path/to/libcuda.so.1")
endif()

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

# ---------------------------------------------------------------- library
add_library(gpap SHARED
  src/runtime.cpp
  src/formula.cpp
  src/engine.cpp
  src/readout.cpp
  src/calibrate.cpp
  src/probe.cpp
  ${GPAP_GENERATED})
target_include_directories(gpap
  PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
          $<INSTALL_INTERFACE:include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${GPAP_CUDA_INCLUDE_DIR})
target_link_libraries(gpap PRIVATE ${GPAP_CUDA_DRIVER})
# PUBLIC so it travels through the exported target: <gpap/gpap.hpp> is C++17, and a consumer
# that only set CMAKE_CXX_STANDARD 11 would otherwise fail to compile it. (C consumers include
# gpap.h instead and are unaffected — that header is plain C99.)
target_compile_features(gpap PUBLIC cxx_std_17)
target_compile_options(gpap PRIVATE -Wall -Wextra -Wno-unused-parameter)
set_target_properties(gpap PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION 0)

# ---------------------------------------------------------------- examples
add_executable(gpap_c_api examples/c_api.c)                   # plain C, the raw ABI
target_link_libraries(gpap_c_api PRIVATE gpap)

add_executable(gpap_sample examples/sample.cpp)              # C++, gpap::Sampler
target_link_libraries(gpap_sample PRIVATE gpap)

add_executable(gpap_assumptions examples/assumptions.cpp)    # C++, gpap::AssumptionChecker
target_link_libraries(gpap_assumptions PRIVATE gpap)

add_executable(gpap_probe examples/probe.cpp)                # C++, gpap::Prober
target_link_libraries(gpap_probe PRIVATE gpap)

# ---------------------------------------------------------------- install / find_package
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

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/gpapConfigVersion.cmake"
  VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/gpapConfig.cmake"
"include(CMakeFindDependencyMacro)\n\
include(\"\${CMAKE_CURRENT_LIST_DIR}/gpapTargets.cmake\")\n")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/gpapConfig.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/gpapConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/gpap)
