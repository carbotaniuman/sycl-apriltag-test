cmake_minimum_required(VERSION 3.22)

set(CMAKE_CXX_COMPILER acpp)
# set(CMAKE_C_COMPILER icx)
# set(CMAKE_CXX_COMPILER icpx)

project(sycl_apriltag_test VERSION 1.0 LANGUAGES CXX)
find_package(AdaptiveCpp CONFIG REQUIRED)
# find_package(IntelSYCL REQUIRED)
# find_package(oneDPL REQUIRED)
# find_package(TBB REQUIRED)
# find_package(OpenCL REQUIRED)

# Add a library
add_executable(sycl_apriltag_test src/main.cpp src/segmentation.cpp src/find_boundaries.cpp src/fit_quad.cpp src/threshold.cpp src/image_impl.cpp)
target_include_directories(sycl_apriltag_test PUBLIC include)
target_compile_features(sycl_apriltag_test PRIVATE cxx_std_20)
# target_link_libraries(sycl_apriltag_test TBB::tbb OpenCL::OpenCL)
set_target_properties(sycl_apriltag_test PROPERTIES
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)
target_compile_options(sycl_apriltag_test PRIVATE )
add_sycl_to_target(TARGET sycl_apriltag_test SOURCES src/main.cpp src/find_boundaries.cpp src/segmentation.cpp src/threshold.cpp)
