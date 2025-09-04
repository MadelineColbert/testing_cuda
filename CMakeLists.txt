cmake_minimum_required(VERSION 3.15 FATAL_ERROR) # Recommended minimum version for CMAKE_CUDA_ARCHITECTURES

set(CMAKE_CUDA_ARCHITECTURES "native")

set(PROGRAM_NAME "my_cublas_app")

project(MyCublasProject LANGUAGES CXX CUDA)

add_executable(${PROGRAM_NAME} test.cu) # Assuming a CUDA source file

target_include_directories(${PROGRAM_NAME}
    PUBLIC
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_link_libraries(${PROGRAM_NAME}
    PUBLIC
        cublas
)


