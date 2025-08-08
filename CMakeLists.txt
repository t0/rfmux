cmake_minimum_required(VERSION 3.15)
include(FetchContent)
project(parser LANGUAGES CXX)

set(PYBIND11_FINDPYTHON ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)

# Pybind11 is installed as a Python dependency, so we need to make sure we look
# in the right places for pybind11 headers
execute_process(
  COMMAND python -m pybind11 --cmakedir
  OUTPUT_VARIABLE _PYBIND11_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${_PYBIND11_DIR}")
find_package(pybind11 CONFIG REQUIRED)

# Try to use a local fmt::format installation. If that didn't work, try via FetchContent.
find_package(fmt CONFIG QUIET)
if (NOT TARGET fmt::fmt-header-only)
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

target_link_libraries(_parser PRIVATE pybind11::pybind11 fmt::fmt-header-only)
