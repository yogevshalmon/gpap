# A host project that embeds libgpap the way mallob does: from source, by the host's own CMake
# (add_subdirectory here; FetchContent with SOURCE_SUBDIR cpp works the same), static, and
# without CUDA. tests/engine/test_embed.py configures, builds and runs it:
#
#   cmake -S cpp/examples/embed -B build-embed
#   cmake --build build-embed -j && build-embed/embed_host
cmake_minimum_required(VERSION 3.18)
project(gpap_embed_host CXX)

set(HOST_BUILD_TYPE "${CMAKE_BUILD_TYPE}")
set(GPAP_WITH_CUDA OFF CACHE BOOL "" FORCE)
set(GPAP_SHARED OFF CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../.. gpap)

# What embedding libgpap must not do to its host.
if(NOT "${CMAKE_BUILD_TYPE}" STREQUAL "${HOST_BUILD_TYPE}")
  message(FATAL_ERROR "libgpap changed the host's build type to '${CMAKE_BUILD_TYPE}'")
endif()
if(TARGET gpap_c_api)
  message(FATAL_ERROR "libgpap built its examples inside a host")
endif()
get_target_property(GPAP_TYPE gpap TYPE)
if(NOT GPAP_TYPE STREQUAL "STATIC_LIBRARY")
  message(FATAL_ERROR "GPAP_SHARED=OFF built libgpap as ${GPAP_TYPE}")
endif()

add_executable(embed_host host.cpp)
target_link_libraries(embed_host PRIVATE gpap::gpap)
