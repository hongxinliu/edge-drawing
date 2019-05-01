cmake_minimum_required(VERSION 3.0)
project(ed)

# general
add_compile_options(-O3 -g -Wall)
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++11" COMPILER_SUPPORTS_CXX11)
if(COMPILER_SUPPORTS_CXX11)
    add_compile_options(-std=c++11)
else()
	message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support.")
endif()

# packages
find_package(OpenCV REQUIRED)

# directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# ed
add_library(${PROJECT_NAME} src/ed.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

# example
add_executable(example example/example.cpp)
target_link_libraries(example ${PROJECT_NAME})

# install
install(TARGETS ${PROJECT_NAME} example 
        RUNTIME DESTINATION bin 
        LIBRARY DESTINATION lib 
        ARCHIVE DESTINATION lib)

install(FILES include/ed.hpp DESTINATION include/ed)