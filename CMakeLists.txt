cmake_minimum_required(VERSION 2.8.3)
project(auto_car)

set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3 -march=native")

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    add_definitions(-DCOMPILEDWITHC11)
    message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
    add_definitions(-DCOMPILEDWITHC0X)
    message(STATUS "Using flag -std=c++0x.")
else()
    message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()
# 添加c++ 11标准支持
set( CMAKE_CXX_FLAGS "-std=c++11" )

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  geometry_msgs
  image_transport
  nav_msgs
  pcl_conversions
  pcl_ros
  roscpp
  rospy
  sensor_msgs
  serial
  std_msgs
  tf
)

# 寻找OpenCV库
find_package( OpenCV 3 REQUIRED)
# 添加头文件
catkin_package(
        CATKIN_DEPENDS
        serial
        std_msgs
)

include_directories(
        include
        srv
        msg
        ${catkin_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        $(Eigen_INCLUDE_DIRS)
)
set(sweepRobot_srclib
        src/utility.cpp
        )

add_library(${PROJECT_NAME} ${sweepRobot_srclib} )
target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} ${OpenCV_LIBS})

add_executable( stereo_ky_node stereo_cali.cpp)
target_link_libraries( stereo_ky_node
        ${catkin_LIBRARIES}
        ${PROJECT_NAME}
        )
add_executable( stereo_video_node stereo_vo.cpp)
target_link_libraries( stereo_video_node
        ${catkin_LIBRARIES}
        ${OpenCV_LIBS}
        ${PROJECT_NAME}
        )

