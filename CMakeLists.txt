cmake_minimum_required(VERSION 3.16)
project(raymap_renderer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# library core code
add_library(core
    src/aabb.cpp
    src/kdtree.cpp
)

target_include_directories(core PUBLIC include)

# tests
add_executable(tests
    tests/tests.cpp
)

target_link_libraries(tests PRIVATE core)

# Optional: enable testing via ctest
enable_testing()
add_test(NAME tests COMMAND tests)