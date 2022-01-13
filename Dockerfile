FROM ros:noetic
MAINTAINER Lexi Scott aas220@case.edu
RUN sudo apt-get -y update                                                                                              
RUN sudo apt-get -y upgrade
RUN sudo apt-get install -y libfcl-dev
RUN sudo apt-get install -y libeigen3-dev
RUN sudo apt-get install -y ros-noetic-tf
RUN sudo apt-get install -y libyaml-cpp-dev
RUN sudo apt-get install -y libmpich-dev
RUN sudo apt-get install -y mpich
RUN sudo apt install -y libopenmpi-dev
COPY . .
WORKDIR "/catkin_ws"
RUN /bin/bash -c ' . /opt/ros/noetic/setup.bash ; catkin_make'

