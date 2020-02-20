cmake_minimum_required(VERSION 2.8.3)
project(sawyer_bopt)

find_package(catkin REQUIRED COMPONENTS
  gazebo_msgs
  intera_core_msgs
  rospy
)

catkin_package(
   LIBRARIES sawyer_bopt 
   DEPENDS system_lib
)

catkin_package(
  CATKIN_DEPENDS
  rospy
  intera_core_msgs
  gazebo_msgs
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)


#############
## Install ##
#############

install(PROGRAMS
   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

foreach(dir launch models)
   install(DIRECTORY ${dir}/
      DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/${dir})
endforeach(dir)
