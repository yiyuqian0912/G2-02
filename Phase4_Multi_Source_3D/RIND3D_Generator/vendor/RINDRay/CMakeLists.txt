cmake_minimum_required(VERSION 3.19)
project(rindray VERSION 0.1.0 LANGUAGES CXX)

option(RINDRAY_EMBREE "Build the optional Embree 4 backend when available" ON)
option(RINDRAY_PYTHON "Build the Python extension" ON)
option(RINDRAY_BUILD_TESTS "Build the native public-API smoke test" OFF)
option(RINDRAY_VULKAN "Build the optional headless Vulkan Ray Query backend" OFF)

add_library(rindray STATIC cpp/core.cpp cpp/vulkan.cpp)
target_include_directories(rindray PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_compile_features(rindray PUBLIC cxx_std_20)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/cpp/core.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/cpp/bindings.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/cpp/vulkan.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/cpp/vulkan.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/cpp/query.comp"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/rindray/rindray.hpp")
file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/cpp/core.cpp" core_hash)
file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/cpp/bindings.cpp" binding_hash)
file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/include/rindray/rindray.hpp" header_hash)
file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/cpp/vulkan.cpp" vulkan_hash)
file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/cpp/vulkan.hpp" vulkan_header_hash)
file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/cpp/query.comp" shader_hash)
file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_spirv.cmake" embedding_hash)
set_target_properties(rindray PROPERTIES POSITION_INDEPENDENT_CODE ON)
find_package(Threads REQUIRED)
target_link_libraries(rindray PRIVATE Threads::Threads)
if(RINDRAY_EMBREE)
  find_package(embree 4 CONFIG QUIET)
  if(embree_FOUND)
    target_link_libraries(rindray PRIVATE embree)
    target_compile_definitions(rindray PUBLIC RINDRAY_HAS_EMBREE=1)
  endif()
endif()
if(RINDRAY_VULKAN)
  find_package(Vulkan REQUIRED)
  find_program(RINDRAY_GLSLANG glslangValidator REQUIRED)
  execute_process(COMMAND "${RINDRAY_GLSLANG}" --version OUTPUT_VARIABLE shader_compiler_version
    OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
  add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/query_spv.hpp"
    COMMAND "${RINDRAY_GLSLANG}" -V --target-env vulkan1.2
      -o "${CMAKE_CURRENT_BINARY_DIR}/query.spv" "${CMAKE_CURRENT_SOURCE_DIR}/cpp/query.comp"
    COMMAND "${CMAKE_COMMAND}" "-DSPIRV=${CMAKE_CURRENT_BINARY_DIR}/query.spv"
      "-DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/query_spv.hpp" -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_spirv.cmake"
    DEPENDS cpp/query.comp cmake/embed_spirv.cmake VERBATIM)
  target_sources(rindray PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/query_spv.hpp")
  target_include_directories(rindray PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
  target_link_libraries(rindray PRIVATE Vulkan::Vulkan)
  target_compile_definitions(rindray PUBLIC RINDRAY_HAS_VULKAN=1)
endif()
string(SHA256 source_hash "${core_hash}${binding_hash}${header_hash}${vulkan_hash}${vulkan_header_hash}${shader_hash}${embedding_hash}${shader_compiler_version}${CMAKE_CXX_COMPILER_ID}${CMAKE_CXX_COMPILER_VERSION}${CMAKE_BUILD_TYPE}${CMAKE_CXX_FLAGS}${CMAKE_CXX_FLAGS_RELEASE}${CMAKE_CXX_FLAGS_DEBUG}${RINDRAY_EMBREE}${RINDRAY_VULKAN}")
target_compile_definitions(rindray PRIVATE RINDRAY_BUILD_ID="${source_hash}")
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(rindray PRIVATE -Wall -Wextra -Wpedantic)
  set_source_files_properties(cpp/vulkan.cpp PROPERTIES COMPILE_OPTIONS -Wno-missing-field-initializers)
endif()

if(RINDRAY_PYTHON)
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
  execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_VARIABLE nanobind_ROOT OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
  find_package(nanobind CONFIG REQUIRED)
  nanobind_add_module(_core cpp/bindings.cpp)
  target_link_libraries(_core PRIVATE rindray)
  install(TARGETS _core LIBRARY DESTINATION rindray)
endif()
install(TARGETS rindray ARCHIVE DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
if(RINDRAY_BUILD_TESTS)
  enable_testing()
  add_executable(rindray_native_smoke tests/native_smoke.cpp)
  target_link_libraries(rindray_native_smoke PRIVATE rindray)
  add_test(NAME rindray_native_smoke COMMAND rindray_native_smoke)
endif()
