#!/bin/bash

# Save path to directory
cwd=$(pwd)

# Install FCL and CCD.
sudo apt-get install libfcl-0.5-dev

# Install FLANN.
cd ~/Downloads
wget http://www.cs.ubc.ca/research/flann/uploads/FLANN/flann-1.8.4-src.zip
unzip flann-1.8.4-src.zip
cd flann-1.8.4-src
mkdir build
cd build
cmake ..
make
sudo make install

# Install HDF5 headers.
sudo apt-get install libhdf5-dev

# Install random
cd $cwd
mkdir lib
cd lib
git clone https://github.com/tipakorng/random.git
