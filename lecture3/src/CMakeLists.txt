cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
project(ex3)

find_package(OpenCV REQUIRED)

add_executable(ex3 ex3.cpp)

target_link_libraries(ex3 ${OpenCV_LIBS})
