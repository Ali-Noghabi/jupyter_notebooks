cmake_minimum_required(VERSION 3.10)

project(MatrixMultiplication)

set(CMAKE_CXX_STANDARD 17)

include_directories(include)

add_executable(MatrixMultiplication src/main.cpp src/Mutex.cpp src/matrix_operations.cpp src/matrices_generator.cpp)
