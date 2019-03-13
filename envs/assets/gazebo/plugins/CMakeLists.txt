cmake_minimum_required(VERSION 2.6)
project(LabWorldPlugin)

find_package(gazebo REQUIRED)

include_directories(include ${GAZEBO_INCLUDE_DIRS})
link_directories(${GAZEBO_LIBRARY_DIRS})
list(APPEND CMAKE_CXX_FLAGS ${GAZEBO_CXX_FLAGS})

add_library(LabWorldPlugin SHARED LabWorldPlugin.cpp)
target_link_libraries(LabWorldPlugin ${GAZEBO_LIBRARIES})
