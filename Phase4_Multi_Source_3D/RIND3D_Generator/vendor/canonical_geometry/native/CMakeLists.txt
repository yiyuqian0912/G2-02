cmake_minimum_required(VERSION 3.23)
project(rind_exact_geometry LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CGAL 6.2.1 EXACT REQUIRED COMPONENTS Core)

add_executable(rind_capability_probe capability_probe.cpp)
target_link_libraries(rind_capability_probe PRIVATE CGAL::CGAL CGAL::CGAL_Core)
target_compile_options(rind_capability_probe PRIVATE -Wall -Wextra -Wpedantic)

enable_testing()
add_test(NAME rind_capability_probe COMMAND rind_capability_probe)

add_executable(rind_capability_probe_3d capability_probe_3d.cpp)
target_link_libraries(rind_capability_probe_3d PRIVATE CGAL::CGAL CGAL::CGAL_Core)
target_compile_options(rind_capability_probe_3d PRIVATE -Wall -Wextra -Wpedantic -O0)
add_test(NAME rind_capability_probe_3d COMMAND rind_capability_probe_3d)

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_exact2d exact2d.cpp)
target_link_libraries(_exact2d PRIVATE CGAL::CGAL CGAL::CGAL_Core)
target_compile_options(_exact2d PRIVATE -Wall -Wextra -Wpedantic -O0)
set_target_properties(_exact2d PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../src/rind")

pybind11_add_module(_exact3d exact3d.cpp)
target_link_libraries(_exact3d PRIVATE CGAL::CGAL CGAL::CGAL_Core)
target_compile_options(_exact3d PRIVATE -Wall -Wextra -Wpedantic -O0)
set_target_properties(_exact3d PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../src/rind")
