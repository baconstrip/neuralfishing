cmake_minimum_required(VERSION 2.8.12)
project(NeuralFishing)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
find_package(OpenMP)
add_compile_options("-O3")

#add_compile_options(
  #"-Wall" "-Wpedantic" "-Wextra" "-fexceptions"
  #"$<$<CONFIG:DEBUG>:-O0;-g3;-ggdb>"
#)

add_definitions("-std=c++17")

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

set(HEADER_FILES
    neural/fake_layer.hpp 
    neural/layer.hpp 
    neural/real_layer.hpp 
    neural/mnist_input_layer.hpp 
    neural/images.hpp
    math/util.hpp
)

set(NEURAL_SOURCES 
    neural/real_layer.cpp
)

set(MATH_SOURCES
    math/util.cpp
)

add_library(neural_net OBJECT ${NEURAL_SOURCES})
target_include_directories(neural_net PRIVATE neural)

add_library(nn_math OBJECT ${MATH_SOURCES})
target_include_directories(nn_math PRIVATE math)

set(SOURCE_FILES main.cpp ${HEADER_FILES})

add_executable(fishin ${SOURCE_FILES})
target_include_directories(fishin PRIVATE neural math)
target_link_libraries(fishin neural_net nn_math OpenMP::OpenMP_CXX ${CONAN_LIBS})