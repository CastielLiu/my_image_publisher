cmake_minimum_required(VERSION 2.8.3)
project(my_image_publisher)


find_package(catkin REQUIRED COMPONENTS roscpp cv_bridge image_transport sensor_msgs roscpp rospy std_msgs)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES my_image_transport
#  CATKIN_DEPENDS cv_bridge image_transport opencv2 sensor_msgs
#  DEPENDS system_lib
)


find_package(OpenCV REQUIRED)
include_directories(  ${catkin_INCLUDE_DIRS}  ${OpenCV_INCLUDE_DIRS} )


add_executable(my_image_publisher_node src/my_image_publisher.cpp)
add_dependencies(my_image_publisher_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(my_image_publisher_node ${catkin_LIBRARIES} ${OpenCV_LIBRARIES}
)

add_executable(publish_camera_info  src/publish_camera_info.cpp)
add_dependencies(publish_camera_info ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(publish_camera_info ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})



