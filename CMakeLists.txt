cmake_minimum_required(VERSION 3.15)
include(FetchContent)
project(parser LANGUAGES CXX)

set(PYBIND11_FINDPYTHON ON)

# Try to use a local pybind11; fall back on FetchContent
find_package(pybind11 CONFIG QUIET)
if (NOT TARGET pybind11)
  message(STATUS "pybind11 not found - fetching via FetchContent...")
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG 3.0.0
  )
FetchContent_MakeAvailable(pybind11)
endif()

# Try to use a local fmt::format installation. If that didn't work, try via FetchContent.
find_package(fmt CONFIG QUIET)
if (NOT TARGET fmt::fmt)
  message(STATUS "fmt not found - fetching via FetchContent...")
  include(FetchContent)
  FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG 10.1.1
  )
FetchContent_MakeAvailable(fmt)
endif()

Python_add_library(_parser MODULE src/parser.cpp WITH_SOABI)
install(TARGETS _parser DESTINATION ${SKBUILD_PROJECT_NAME})

target_compile_features(_parser PUBLIC cxx_std_20)
if (MSVC)
  target_compile_options(_parser PRIVATE /W4 /permissive-)
else()
  target_compile_options(_parser PRIVATE -Wall -Wextra -Wpedantic -O3 -flto=auto)
endif()

target_link_libraries(_parser PRIVATE fmt::fmt)
