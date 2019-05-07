//
// Created by tipakorng on 11/23/15.
//

#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include "entropy_estimation.h"


void vectorToFlann(const std::vector<Particle> &particles, flann::Matrix<double> &mat)
{
    double element;

    for (int i = 0; i < mat.rows; i++)
    {
        for (int j = 0; j < mat.cols; j++)
        {
            mat[i][j] = particles[i].getValue()[j];
            element = particles[i].getValue()[j];
        }
    }
}

double digamma_function(double x)
{
    return 0;
}

double ballVolume(int dimension, double radius)
{
    return pow(M_PI, (0.5 * dimension)) * pow(radius, dimension) / boost::math::tgamma(0.5 * dimension + 1);
}

double estimateEntropy(const std::vector<Particle> &particles, int num_nearest_neighbor, int num_cores)
{
    /* Setup kd tree and find k nearest neighbors and their distances */
    int dimension = particles[0].getDimension();
    flann::Matrix<double> dataset(new double[particles.size()*dimension], particles.size(), dimension);
    vectorToFlann(particles, dataset);
    flann::Matrix<int> indices(new int[dataset.rows * num_nearest_neighbor], dataset.rows, num_nearest_neighbor);
    flann::Matrix<double> distances(new double[dataset.rows * num_nearest_neighbor], dataset.rows, num_nearest_neighbor);
    flann::Index<flann::L2<double> > index(dataset, flann::KDTreeIndexParams(4));
    index.buildIndex();
    flann::SearchParams search_params;
    search_params.checks = 128;
    search_params.cores = num_cores;
    index.knnSearch(dataset, indices, distances, num_nearest_neighbor, search_params);

    /* Entropy calculation */
    double entropy = std::log(num_nearest_neighbor) - boost::math::digamma(num_nearest_neighbor);
    int idx;  // Index for nearest neighbors
    double sum_weight;
    double knn_distance;

    for (int i = 0; i < particles.size(); ++i)
    {
        sum_weight = 0;

        for (int j = 0; j < num_nearest_neighbor; ++j)
        {
            idx = indices[i][j];
            sum_weight += particles[idx].getWeight();
        }

        knn_distance = std::sqrt(distances[i][num_nearest_neighbor - 1]);

        if (sum_weight > 1e-16)
        {
            entropy -= sum_weight * std::log(sum_weight / ballVolume(dimension, knn_distance)) / num_nearest_neighbor;
        }
    }

    /* Clean up */
    delete[] dataset.ptr();
    delete[] indices.ptr();
    delete[] distances.ptr();

    return entropy;
}
