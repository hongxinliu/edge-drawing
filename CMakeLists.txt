cmake_minimum_required(VERSION 3.0)
project(EdgeDrawing)

# general
add_compile_options(-Wall -g -fPIC)
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++11" COMPILER_SUPPORTS_CXX11)
if(COMPILER_SUPPORTS_CXX11)
    add_compile_options(-std=c++11)
else()
	message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support.")
endif()

# packages
find_package(OpenCV REQUIRED)

# targets
add_subdirectory(ED)
add_subdirectory(examples)
