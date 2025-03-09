cmake_minimum_required(VERSION 3.25.1)
project(sigmoid)

set(CMAKE_CXX_STANDARD 17)

include_directories(
    test/include
    src/include
)

add_executable(test
    src/neuron.cpp
    src/neuronlayer.cpp
    src/neuronnetwork.cpp
    test/test.cpp
    test/parsedata.cpp
)

target_compile_definitions(test PRIVATE PROJ_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
