# Active Sensing in Continuous Space

## Dependencies
- FCL
- HDF5
- FLANN
- https://github.com/tipakorng/random.git

## Some changes have been made in order to run this package in Ubuntu 16.04 ros-kinetic

- Install FCL and CCD using:
`sudo apt-get install libfcl-0.5-dev`

- Change in code:

	'boost::share_ptr' to 'std::share_ptr'

	In target_link_libraries:

		'{$fcl_libraries}' to 'fcl'

	In test_particle_filer:

		'Eigen::VectorXd observation({1});' to 'Eigen::VectorXd observation(1);'

	For all test nodes:

		add 'ros::init()' in main func

- Some of links missing:

`sudo ln -s /usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5.so /usr/lib/libhdf5.so`

`sudo ln -s /usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5_hl.so /usr/lib/libhdf5_hl.so`

`sudo ln -s /opt/ros/kinetic/lib/liboctomap.so.1.8.1 /usr/lib/liboctomap.so`

`sudo ln -s /opt/ros/kinetic/lib/liboctomath.so.1.8.1 /usr/lib/liboctomath.so`