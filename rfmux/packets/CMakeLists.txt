cmake_minimum_required(VERSION 3.15)
project(packets LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Try to find fmt first, otherwise fetch from source
find_package(fmt CONFIG QUIET)
if(NOT fmt_FOUND)
	include(FetchContent)
	set(CMAKE_POSITION_INDEPENDENT_CODE ON)
	FetchContent_Declare(
		fmt
		GIT_REPOSITORY https://github.com/fmtlib/fmt.git
		GIT_TAG 11.0.2
	)
	FetchContent_MakeAvailable(fmt)
endif()

set(SOURCES
    src/packets.cpp
    src/bindings.cpp
)

pybind11_add_module(_packets ${SOURCES})

target_include_directories(_packets PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(_packets PRIVATE fmt::fmt)

# Link Winsock on Windows
if(WIN32)
    target_link_libraries(_packets PRIVATE ws2_32)
endif()

# Compiler warnings (cross-platform)
if(MSVC)
    target_compile_options(_packets PRIVATE /W4)
    # Workaround for MSVC 17.10+ constexpr mutex breaking change
    # See: https://github.com/microsoft/STL/issues/4730
    #      https://github.com/microsoft/STL/issues/4875
    # Without this, mutexes crash when Python uses older VC++ runtime
    target_compile_definitions(_packets PRIVATE _DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)
else()
    target_compile_options(_packets PRIVATE -Wall -Wextra)
endif()

# Install to rfmux/packets subdirectory
install(TARGETS _packets LIBRARY DESTINATION rfmux/packets)
