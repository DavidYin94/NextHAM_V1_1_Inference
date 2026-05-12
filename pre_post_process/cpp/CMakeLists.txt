cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(NextHAM_Preprocessing LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -ffast-math")

# Find LibTorch
set(CMAKE_PREFIX_PATH "/your_path/NextHAM_V1_1_Inference/pkg/libtorch")
find_package(Torch REQUIRED)

# Find Eigen3
set(EIGEN3_INCLUDE_DIR "/your_path/NextHAM_V1_1_Inference/pkg/eigen-3.4.0")
include_directories(${EIGEN3_INCLUDE_DIR})

# Find OpenMP
find_package(OpenMP REQUIRED)

# Add Source Files
set(SOURCES
    main.cpp
    stru_parser.cpp
    csr_parser.cpp
    block_parser.cpp
    neighbor_list.cpp
    transformer.cpp
)

add_executable(nextham_preprocess ${SOURCES})

# Target Properties
target_link_libraries(nextham_preprocess "${TORCH_LIBRARIES}" OpenMP::OpenMP_CXX)
set_property(TARGET nextham_preprocess PROPERTY CXX_STANDARD 17)

# Set runtime paths for shared libraries
if(UNIX)
  set_target_properties(nextham_preprocess PROPERTIES INSTALL_RPATH "${CMAKE_PREFIX_PATH}/lib")
endif()
