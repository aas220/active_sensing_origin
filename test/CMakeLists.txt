cmake_minimum_required(VERSION 2.8)

add_executable(test_entropy_estimation test_entropy_estimation.cpp)
target_link_libraries(test_entropy_estimation entropy_estimation multivariate_gaussian)
target_link_libraries(test_entropy_estimation gtest gtest_main)
target_link_libraries(test_entropy_estimation ${HDF_LIBRARIES})

add_executable(test_abstract_robot test_abstract_robot.cpp)
target_link_libraries(test_abstract_robot abstract_robot_model)
target_link_libraries(test_abstract_robot ${catkin_LIBRARIES} gtest gtest_main)

#add_executable(test_peg_hole_2d test_peg_hole_2d.cpp)
#target_link_libraries(test_peg_hole_2d peg_hole_2d_model peg_hole_2d_planner)
#target_link_libraries(test_peg_hole_2d ${catkin_LIBRARIES} gtest gtest_main)

add_executable(test_peg_hole_3d test_peg_hole_3d.cpp)
target_link_libraries(test_peg_hole_3d peg_hole_3d_model)
target_link_libraries(test_peg_hole_3d ${catkin_LIBRARIES} gtest gtest_main)

add_executable(test_potential_field test_potential_field.cpp)
target_link_libraries(test_potential_field potential_field abstract_robot_model)
target_link_libraries(test_potential_field ${catkin_LIBRARIES} gtest gtest_main)

add_executable(test_particle_filter test_particle_filter.cpp)
target_link_libraries(test_particle_filter particle_filter abstract_robot_model)
target_link_libraries(test_particle_filter ${catkin_LIBRARIES} gtest gtest_main)
target_link_libraries(test_particle_filter ${HDF_LIBRARIES})

add_executable(test_active_sensing test_active_sensing.cpp)
target_link_libraries(test_active_sensing particle_filter abstract_robot_model potential_field
        planner active_sensing entropy_estimation)
target_link_libraries(test_active_sensing ${catkin_LIBRARIES} gtest gtest_main)
target_link_libraries(test_active_sensing ${HDF_LIBRARIES})

add_executable(test_simulator test_simulator.cpp)
target_link_libraries(test_simulator particle_filter abstract_robot_model potential_field planner
        active_sensing entropy_estimation simulator)
target_link_libraries(test_simulator ${catkin_LIBRARIES} gtest gtest_main)
target_link_libraries(test_simulator ${HDF_LIBRARIES})
