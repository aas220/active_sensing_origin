//
// Created by tipakorng on 11/23/15.
//

#ifndef ACTIVE_SENSING_CONTINUOUS_UTILS_H
#define ACTIVE_SENSING_CONTINUOUS_UTILS_H

#include <iostream>
#include <vector>
#include <flann/flann.hpp>
#include <flann/io/hdf5.h>
#include <math_utils.h>
#include "particle_filter.h"

/*
 * \brief Put positions of the particles to a FLANN matrix.
 */
void vectorToFlann(const std::vector<Particle> &particles, flann::Matrix<double> &points);

/*
 * \brief Calculate ball volume R^n.
 */
double ballVolume(int dimension, double radius);

/*
 * \brief Estimate entropy of a set of particles using k-nearest neighbor method in Ajgl et al. 2011.
 *
 * @param particles A vector containing particles defined in particle_filter.h.
 * @param k The number of nearest neighbors used in estimating the entropy.
 */
double estimateEntropy(const std::vector<Particle> &particles, int num_nearest_neighbors, int num_cores);

#endif //ACTIVE_SENSING_CONTINUOUS_UTILS_H
