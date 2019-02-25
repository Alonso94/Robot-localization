cmake_minimum_required(VERSION 2.8.3)
project(localization)
find_package(catkin REQUIRED COMPONENTS
  rospy
  message_generation
)

add_message_files(
  FILES
  Coordinates.msg
)
generate_messages(
  DEPENDENCIES
)
catkin_package()
include_directories(include ${catkin_INCLUDE_DIRS})


