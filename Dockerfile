FROM ros:noetic
MAINTAINER Lexi Scott aas220@case.edu
RUN sudo apt-get -y update                                                                                              
RUN sudo apt-get -y upgrade
RUN sudo apt-get install -y libfcl-dev
RUN sudo apt-get install -y libeigen3-dev
RUN sudo apt-get install -y ros-noetic-tf
RUN sudo apt-get install -y libyaml-cpp-dev
RUN sudo apt-get install -y libmpich-dev
RUN sudo apt-get install -y libhdf5-serial-dev
RUN sudo apt-get install -y mpich
RUN sudo apt-get install -y libopenmpi-dev
RUN sudo ln -s /usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5.so /usr/lib/libhdf5.so
RUN sudo ln -s /usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5_hl.so /usr/lib/libhdf5_hl.so
COPY . .
WORKDIR "/catkin_ws"
RUN /bin/bash -c ' . /opt/ros/noetic/setup.bash ; catkin_make'
#CMD rosrun active_sensing_continuous peg_hole_2d_sim src/active_sensing_origin/active_sensing_yml/peg_hole_2d/peg_hole_2d/yml outputfile.txt
