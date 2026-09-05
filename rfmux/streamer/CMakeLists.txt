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
		GIT_TAG 11.1.4
	)
	# Don't install fmt (it's statically linked into _receiver)
	set(FMT_INSTALL OFF CACHE BOOL "" FORCE)
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
        # No -ffast-math: the batch getters hand back NaN for an
        # undisciplined timestamp, and under fast-math clang folds a
        # select against NaN into the other branch.
    )
endif()

# Install to rfmux/streamer subdirectory
install(TARGETS _receiver LIBRARY DESTINATION rfmux/streamer)

#
# fastrx: Linux-only, AF_XDP-based receiver fastpath for fancy NICs
#
# Built only on Linux when clang, libxdp, libbpf and liburing are all present.
# When fastrxd is mandatory (i.e. for wheel builds), use
#
#   $ uv pip install . --config-settings=cmake.define.FASTRX_REQUIRED=ON
#
# or
#
#   $ SKBUILD_CMAKE_DEFINE="FASTRX_REQUIRED=ON" uv pip install .
#
# but note that --config-settings don't seem to stick - uv will happily
# re-install without fastrx during subsequent "uv run" commands.

option(FASTRX_REQUIRED "Fail configure if fastrxd/_fastrx cannot be built" OFF)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    find_program(FASTRX_CLANG NAMES clang-19 clang)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(FASTRX_LIBXDP QUIET IMPORTED_TARGET libxdp>=1.3)
        pkg_check_modules(FASTRX_LIBBPF QUIET IMPORTED_TARGET libbpf>=1.0)
        pkg_check_modules(FASTRX_LIBURING QUIET IMPORTED_TARGET liburing>=2.0)
    endif()
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux"
	AND CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64"
        AND FASTRX_CLANG
	AND FASTRX_LIBXDP_FOUND
	AND FASTRX_LIBBPF_FOUND
	AND FASTRX_LIBURING_FOUND)

    message(STATUS "fastrx: building fastrxd and _fastrx (clang=${FASTRX_CLANG})")

    # pthread must be linked in explicitly on pre-2.34 glibc
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)

    # ABI hash, used to ensure the shared-memory region between fastrxd and its
    # clients is interpreted consistently
    set(FASTRX_ABI_HEADERS
        ${CMAKE_CURRENT_SOURCE_DIR}/include/fastrx.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/packet.h)
    set(FASTRX_ABI_HASH "")
    foreach(hdr ${FASTRX_ABI_HEADERS})
        file(SHA256 ${hdr} _h)
        string(APPEND FASTRX_ABI_HASH ${_h})
    endforeach()
    string(SHA256 FASTRX_ABI_HASH ${FASTRX_ABI_HASH})
    string(SUBSTRING ${FASTRX_ABI_HASH} 0 8 FASTRX_ABI_HASH)

    set(FASTRX_BPF_CFLAGS
        -O2 -g
        -target bpf
        -Wall
        -Wno-unused-value
        -Wno-pointer-sign
        -Wno-compare-distinct-pointer-types
        -Wno-missing-declarations
    )

    # clang with "-target bpf" does not add the host's arch-specific include
    # path, so on Debian/Ubuntu multiarch layouts <asm/types.h> (which lives
    # under /usr/include/<triple>/) is not found.
    if(CMAKE_LIBRARY_ARCHITECTURE AND
            EXISTS /usr/include/${CMAKE_LIBRARY_ARCHITECTURE})
        list(APPEND FASTRX_BPF_CFLAGS -I/usr/include/${CMAKE_LIBRARY_ARCHITECTURE})
    endif()

    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/fastrx_bpf.o
        COMMAND ${FASTRX_CLANG} ${FASTRX_BPF_CFLAGS}
                -I${CMAKE_CURRENT_SOURCE_DIR}/include -c
                ${CMAKE_CURRENT_SOURCE_DIR}/src/bpf.c
                -o ${CMAKE_CURRENT_BINARY_DIR}/fastrx_bpf.o
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/bpf.c
                ${CMAKE_CURRENT_SOURCE_DIR}/include/packet.h
        COMMENT "fastrx: compiling BPF program"
    )

    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/fastrx_bpf_embed.o
        COMMAND ${CMAKE_LINKER} -r -b binary -o fastrx_bpf_embed.o fastrx_bpf.o
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/fastrx_bpf.o
        COMMENT "fastrx: embedding BPF ELF as linkable object"
    )

    add_custom_target(fastrx_bpf_embed ALL
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/fastrx_bpf_embed.o
    )

    add_executable(fastrxd src/fastrxd.cpp)
    target_include_directories(fastrxd PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_definitions(fastrxd PRIVATE FASTRXD_ABI_VERSION=0x${FASTRX_ABI_HASH}u)
    target_compile_options(fastrxd PRIVATE
        -Wall -Wextra -Wshadow -Wcast-qual -Wcast-align
        -Wnon-virtual-dtor -Wextra-semi -Werror
    )
    target_link_libraries(fastrxd PRIVATE
        PkgConfig::FASTRX_LIBXDP PkgConfig::FASTRX_LIBBPF elf z
        Threads::Threads)
    target_link_options(fastrxd PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/fastrx_bpf_embed.o)
    add_dependencies(fastrxd fastrx_bpf_embed)
    install(TARGETS fastrxd DESTINATION rfmux/streamer)

    pybind11_add_module(_fastrx src/fastrx_py.cpp)
    target_include_directories(_fastrx PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_definitions(_fastrx PRIVATE FASTRXD_ABI_VERSION=0x${FASTRX_ABI_HASH}u)
    target_compile_options(_fastrx PRIVATE
        -Wall -Wextra -Wshadow -Wcast-qual -Wcast-align
        -Wnon-virtual-dtor -Wextra-semi -Werror
    )
    target_link_libraries(_fastrx
	    PRIVATE atomic Threads::Threads PkgConfig::FASTRX_LIBURING)
    install(TARGETS _fastrx LIBRARY DESTINATION rfmux/streamer)

elseif(FASTRX_REQUIRED
	AND CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")

    message(FATAL_ERROR "fastrx: FASTRX_REQUIRED=ON, but pre-requisites were missing")
endif()
