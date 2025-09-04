cmake_minimum_required(VERSION 3.15 FATAL_ERROR) # Recommended minimum version for CMAKE_CUDA_ARCHITECTURES

set(CMAKE_CUDA_ARCHITECTURES "native")

set(PROGRAM_NAME "my_cublas_app")

project(MyCublasProject LANGUAGES CXX CUDA)

add_executable(${PROGRAM_NAME} test.cu) # Assuming a CUDA source file

# add_library(${PROGRAM_NAME} STATIC test.cu)

target_include_directories(${PROGRAM_NAME}
    PUBLIC
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_link_libraries(${PROGRAM_NAME}
    PUBLIC
        cublas
)

# install(TARGETS MMLib DESTINATION lib)
# install(DIRECTORY include/ DESTINATION include)