cmake_minimum_required(VERSION 3.5)
project(localization)

find_package(catkin REQUIRED COMPONENTS  
  roscpp
  rospy
  nav_msgs
  sensor_msgs
  std_msgs
  tf
  cmake_modules
)
find_package(Boost REQUIRED COMPONENTS system)
find_package(Eigen3 REQUIRED)

catkin_package(
 INCLUDE_DIRS ${EIGEN3_INCLUDE_DIRS}
 LIBRARIES ${PROJECT_NAME}
 CATKIN_DEPENDS nav_msgs roscpp sensor_msgs std_msgs tf
 DEPENDS #Eigen3
)

include_directories(
  SYSTEM
  ${catkin_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)
set(CMAKE_CXX_STANDARD 14)

add_executable(localization Clocalization.cpp)
target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES})
add_compile_options(-std=c++0x)
