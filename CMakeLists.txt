cmake_minimum_required(VERSION 2.8)
project(active_sensing_continuous)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Find Packages.
find_package(Eigen3 REQUIRED)
find_package(HDF5 REQUIRED)
find_package(CUDA REQUIRED)

if(CUDA_FOUND)
    message(STATUS "CUDA found (include: ${CUDA_INCLUDE_DIRS}, lib: ${CUDA_LIBRARIES})")
    include_directories(${CUDA_INCLUDE_DIR})
else(CUDA_FOUND)
    message(FATAL_ERROR "This project requires CUDA configured with CMake")
endif(CUDA_FOUND)

if(HDF5_FOUND)
    message(Found HDF5)
    set(HDF_LIBRARIES hdf5_hl hdf5)
    include_directories(${HDF5_INCLUDE_DIR})
else(HDF5_FOUND)
    message(FATAL_ERROR "This project requires HDF5 1.8.5 (or later) configured with CMake")
endif(HDF5_FOUND)

find_package(OpenMP REQUIRED)

if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

#find_package(fcl REQUIRED)

find_package(Boost COMPONENTS filesystem system REQUIRED)
find_package(Threads REQUIRED)
find_package(YamlCpp REQUIRED)
find_package(catkin REQUIRED COMPONENTS roslz4 roscpp rospy std_msgs visualization_msgs tf)

catkin_package(
    CATKIN_DEPENDS roscpp rospy std_msgs visualization_msgs tf
)

# add gprof
add_compile_options(-pg)
set(catkin_LIBRARIES ${catkin_LIBRARIES} -pg)

# Add subdirectories.
add_subdirectory(lib/random)
#add_subdirectory(lib/cwru_catheter_kinematics)

# Include directories.
include_directories(
        include
        lib/random/include
        ${catkin_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${fcl_INCLUDE_DIR}
        ${Boost_INCLUDE_DIR}
        ${YAMLCPP_INCLUDE_DIR}

)


## link directories
link_directories(/usr/local/lib)
link_directories(/usr/local/cuda/targets/x86_64-linux/lib)


# Add libraries
add_library(particle_filter src/particle_filter.cpp)
target_link_libraries(particle_filter rng pthread)

cuda_add_library(entropy_estimation src/entropy_estimation.cpp)
target_link_libraries(entropy_estimation particle_filter ${catkin_LIBRARIES} libflann_cuda.so libflann_cuda_s.a libcudart_static.a)

add_library(abstract_robot_model src/models/abstract_robot_model.cpp)
target_link_libraries(abstract_robot_model multivariate_gaussian)

add_library(potential_field src/planners/potential_field.cpp)

add_library(active_sensing src/active_sensing.cpp)
target_link_libraries(active_sensing entropy_estimation particle_filter)

add_library(planner src/planner.cpp)
target_link_libraries(planner active_sensing particle_filter)

add_library(simulator src/simulator.cpp)
target_link_libraries(simulator planner ${catkin_LIBRARIES})

add_library(peg_hole_2d_model src/models/peg_hole_2d_model.cpp)
target_link_libraries(
        peg_hole_2d_model
        multivariate_gaussian
        ${Boost_FILESYSTEM_LIBRARY}
        ${Boost_SYSTEM_LIBRARY}
        ${catkin_LIBRARIES}
        fcl
)

add_library(peg_hole_3d_model src/models/peg_hole_3d_model.cpp)
target_link_libraries(
        peg_hole_3d_model
        fcl
        multivariate_gaussian
        ${Boost_FILESYSTEM_LIBRARY}
        ${Boost_SYSTEM_LIBRARY}
        ${catkin_LIBRARIES}
)

add_library(multitier_peg_hole_2d_model src/models/multitier_peg_hole_2d_model.cpp)
target_link_libraries(
        multitier_peg_hole_2d_model
        multivariate_gaussian
        fcl
        ${Boost_FILESYSTEM_LIBRARY}
        ${Boost_SYSTEM_LIBRARY}
        ${catkin_LIBRARIES}
)

add_library(capsule_robot_model src/models/capsule_robot_model.cpp)
target_link_libraries(
        capsule_robot_model
        multivariate_gaussian
        fcl
        ${Boost_FILESYSTEM_LIBRARY}
        ${Boost_SYSTEM_LIBRARY}
        ${catkin_LIBRARIES}
)

add_library(peg_hole_2d_planner src/planners/peg_hole_2d_planner.cpp)

add_library(peg_hole_3d_planner src/planners/peg_hole_3d_planner.cpp)

add_library(multitier_peg_hole_2d_planner src/planners/multitier_peg_hole_2d_planner.cpp)

add_library(capsule_robot_planner src/planners/capsule_robot_planner.cpp)

add_executable(peg_hole_2d_sim src/problems/peg_hole_2d_sim.cpp)
target_link_libraries(
        peg_hole_2d_sim
        peg_hole_2d_model
        peg_hole_2d_planner
        simulator
        particle_filter
        entropy_estimation
        active_sensing
        ${HDF_LIBRARIES}
        ${YAMLCPP_LIBRARY}
        libcudart_static.a
)
#[[
add_executable(abstract_robot_sim src/problems/abstract_robot_sim.cpp)
target_link_libraries(
        abstract_robot_sim
        abstract_robot_model
        potential_field
        simulator
        particle_filter
        entropy_estimation
        active_sensing
        ${HDF_LIBRARIES}
)

add_executable(peg_hole_3d_sim src/problems/peg_hole_3d_sim.cpp)
target_link_libraries(
        peg_hole_3d_sim
        peg_hole_3d_model
        peg_hole_3d_planner
        simulator
        particle_filter
        entropy_estimation
        active_sensing
        ${HDF_LIBRARIES}
        ${YAMLCPP_LIBRARY}
        ${catkin_LIBRARIES}
)

add_executable(multitier_peg_hole_2d_sim src/problems/multitier_peg_hole_2d_sim.cpp)
target_link_libraries(
        multitier_peg_hole_2d_sim
        multitier_peg_hole_2d_model
        multitier_peg_hole_2d_planner
        simulator
        particle_filter
        entropy_estimation
        active_sensing
        ${HDF_LIBRARIES}
        ${YAMLCPP_LIBRARY}
        ${catkin_LIBRARIES}
)

add_executable(capsule_robot_sim src/problems/capsule_robot_sim.cpp)
target_link_libraries(
        capsule_robot_sim
        capsule_robot_model
        capsule_robot_planner
        simulator
        particle_filter
        entropy_estimation
        active_sensing
        ${HDF_LIBRARIES}
        ${YAMLCPP_LIBRARY}
        ${catkin_LIBRARIES}
)

add_executable(peg_hole_3d_node src/ros_nodes/peg_hole_3d_node.cpp)
target_link_libraries(peg_hole_3d_node ${catkin_LIBRARIES})
]]
# Add tests
# add_subdirectory(test)
