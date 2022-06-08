cmake_minimum_required(VERSION 3.9)

project(test LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 11)

#if(CUDA_ENABLE)
#    enable_language(CUDA)
#endif()
add_executable(
        ${PROJECT_NAME}
        main.cu
        vec3.h
        ray.h
)

set_target_properties (
        ${PROJECT_NAME}
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON )