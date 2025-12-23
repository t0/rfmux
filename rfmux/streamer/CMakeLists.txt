cmake_minimum_required(VERSION 3.15)
project(packets LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Use Release build with optimizations by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

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

pybind11_add_module(_receiver ${SOURCES})

target_include_directories(_receiver PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(_receiver PRIVATE fmt::fmt)

# Link Winsock on Windows
if(WIN32)
    target_link_libraries(_receiver PRIVATE ws2_32)
endif()

# Compiler warnings and optimizations (cross-platform)
if(MSVC)
    target_compile_options(_receiver PRIVATE
        /W4                  # Warning level 4
        /O2                  # Optimize for speed
        /fp:fast            # Fast floating-point math
        /arch:AVX2          # Use AVX2 instructions if available
    )
    # Workaround for MSVC 17.10+ constexpr mutex breaking change
    # See: https://github.com/microsoft/STL/issues/4730
    #      https://github.com/microsoft/STL/issues/4875
    # Without this, mutexes crash when Python uses older VC++ runtime
    target_compile_definitions(_receiver PRIVATE _DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)
else()
    target_compile_options(_receiver PRIVATE
        -Wall -Wextra       # Warnings
        -O3                 # Aggressive optimization with auto-vectorization
        -march=native       # Use all available CPU instructions
        -ffast-math         # Fast floating-point (convert divisions to multiplications)
    )
endif()

# Install to rfmux/streamer subdirectory
install(TARGETS _receiver LIBRARY DESTINATION rfmux/streamer)
