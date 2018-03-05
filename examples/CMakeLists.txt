include_directories(${CMAKE_SOURCE_DIR})

add_executable(example example.cpp)
target_link_libraries(example ${OpenCV_LIBS} ED)
