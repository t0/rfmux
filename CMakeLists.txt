cmake_minimum_required(VERSION 3.15)
project(parser LANGUAGES CXX)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)

Python_add_library(_parser MODULE src/parser.cpp WITH_SOABI)
install(TARGETS _parser DESTINATION ${SKBUILD_PROJECT_NAME})

target_compile_features(_parser PUBLIC cxx_std_20)
if (MSVC)
  target_compile_options(_parser PRIVATE /W4 /permissive-)
else()
  target_compile_options(_parser PRIVATE -Wall -Wextra -Wpedantic -O3 -flto=auto)
endif()

target_link_libraries(_parser PRIVATE fmt::fmt)
