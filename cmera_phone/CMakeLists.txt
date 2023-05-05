cmake_minimum_required(VERSION 3.23)
project (camera)
find_package( OpenCV REQUIRED )
find_package(Eigen3 REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})
set(CMAKE_CXX_STANDARD 14)

add_executable(camera main.cpp Compute_RT.cpp Compute_RT.h)
target_link_libraries(camera  ${OpenCV_LIBS} )