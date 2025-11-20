cmake_minimum_required(VERSION 3.15)
project(packets)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)

set(SOURCES
    src/packets.cpp
    src/bindings.cpp
)

pybind11_add_module(_packets ${SOURCES})

target_include_directories(_packets PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(_packets PRIVATE pthread fmt)
target_compile_options(_packets PRIVATE -Wall -Wextra)

install(TARGETS _packets LIBRARY DESTINATION .)
