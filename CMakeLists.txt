cmake_minimum_required(VERSION 3.22)

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
set(CMAKE_CXX_COMPILER acpp)
# SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} --acpp-targets=cuda:sm_70")

project(sycl_apriltag_test VERSION 1.0 LANGUAGES CXX)

# find_package(IntelSYCL REQUIRED)
find_package(AdaptiveCpp CONFIG REQUIRED)
# find_package(oneDPL REQUIRED)

include(FetchContent)
FetchContent_Declare(
  apriltag
  GIT_REPOSITORY https://github.com/carbotaniuman/apriltag.git
  GIT_TAG        04ebca938bb16bd453965ddfdf30554745245045
)
FetchContent_MakeAvailable(apriltag)

add_executable(sycl_apriltag_test src/main.cpp src/segmentation.cpp src/find_boundaries.cpp src/fit_quad.cpp src/threshold.cpp src/image_impl.cpp src/quad_decoding.cpp)
# add_executable(sycl_apriltag_test src/maind.cpp)
target_include_directories(sycl_apriltag_test PUBLIC include)
target_compile_features(sycl_apriltag_test PRIVATE cxx_std_17)
set_target_properties(sycl_apriltag_test PROPERTIES
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
)
target_link_libraries(sycl_apriltag_test PRIVATE apriltag)
# target_link_libraries(sycl_apriltag_test PRIVATE oneDPL apriltag)
# target_compile_options(sycl_apriltag_test
#  PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda
# )
# target_link_options(sycl_apriltag_test
#  PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda
# )
add_sycl_to_target(TARGET sycl_apriltag_test SOURCES src/main.cpp src/segmentation.cpp src/find_boundaries.cpp src/fit_quad.cpp src/threshold.cpp src/quad_decoding.cpp)
# add_sycl_to_target(TARGET sycl_apriltag_test SOURCES src/maind.cpp)
