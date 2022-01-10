//
// Created by tipakorng on 12/2/15.
//

#include <vector>
#include <cstdlib>
#include <stdlib.h>
#include <time.h>
#include "gtest/gtest.h"
#include "entropy_estimation.h"
#include "particle_filter.h"
#include "multivariate_gaussian.h"
#include "rng.h"
#include <lz4.h>
using std::vector;

TEST(test_utils, test_vector2matrix)
{
    vector<Particle> particles;
    int num_particles = 5;
    int size_particle = 3;

    for (int i = 0; i < num_particles; ++i)
    {
        Eigen::VectorXd point(size_particle);
        point << 0, 1, 2;
        double weight = 1;
        Particle particle(point, weight);
        particles.push_back(particle);
    }

    flann::Matrix<double> mat(new double[size_particle * num_particles], num_particles, size_particle);
    vectorToFlann(particles, mat);

    for (int i = 0; i < num_particles; ++i)
    {
        for (int j = 0; j < size_particle; ++j)
        {
            ASSERT_EQ(mat[i][j], j);
        }
    }
}

TEST(test_utils, test_ball_volume)
{
    int dim = 2;
    double radius = 2;
    ASSERT_EQ(M_PI * pow(radius, 2), ballVolume(dim, radius));
}

TEST(test_utils, test_estimate_entropy)
{
    unsigned int num_trials = 10;
    unsigned long long seed = 0;
    Rng rng(seed);

    unsigned int dim = 3;
    Eigen::VectorXd mean = Eigen::VectorXd::Zero(dim);
    Eigen::MatrixXd cov(dim, dim);

    unsigned int num_particles = 10000;
    unsigned int num_nearest = 20;
    unsigned int num_cores = 1;
    double weight = 1.0 / static_cast<double>(num_particles);
    vector<Particle> particles;

    for (unsigned int i = 0; i < num_trials; i++)
    {
        // Generate the Gaussian distribution.
        cov.setZero();

        for (unsigned int j = 0; j < dim; j++)
        {
            cov(j, j) = rng.doub();
        }

        MultivariateGaussian gaussian_pdf(mean, cov, seed);

        // Generate particles.
        particles.clear();

        for (unsigned int j = 0; j < num_particles; j++)
        {
            Eigen::VectorXd sample(dim);
            gaussian_pdf.dev(sample);
            particles.push_back(Particle(sample, weight));
        }

        // Calculate the entropy and check the errors.
        double estimated_entropy = estimateEntropy(particles, num_nearest, num_cores);
        double entropy = 0.5 * dim * std::log(2 * M_PI * std::exp(1)) + 0.5 * std::log(cov.determinant());
        double error = entropy - estimated_entropy;
        ASSERT_LT(error, 0.1);
    }
}

TEST(test_utils, test_compare_entropy)
{
    unsigned int num_trials = 10;
    unsigned long long seed = 0;
    Rng rng(seed);

    unsigned int dim = 2;
    Eigen::VectorXd mean = Eigen::VectorXd::Zero(dim);
    Eigen::MatrixXd cov_0(dim, dim);
    Eigen::MatrixXd cov_1(dim, dim);

    unsigned int num_particles = 100;
    unsigned int num_nearest = 10;
    unsigned int num_cores = 4;
    double weight = 1.0 / static_cast<double>(num_particles);

    vector<Particle> particles_0;
    vector<Particle> particles_1;

    for (unsigned int i = 0; i < num_trials; i++)
    {
        // Set covariance matrices.
        cov_0.setZero();
        cov_1.setZero();

        for (unsigned int j = 0; j < dim; j++)
        {
            cov_0(j, j) = rng.doub();
        }

        double min_cov = cov_0.diagonal().minCoeff();

        for (unsigned int j = 0; j < dim; j++)
        {
            cov_1(j, j) = rng.doub() * min_cov;
        }

        MultivariateGaussian gaussian_pdf_0(mean, cov_0, seed);
        MultivariateGaussian gaussian_pdf_1(mean, cov_1, seed);

        // Sample from the distributions.
        particles_0.clear();
        particles_1.clear();

        for (unsigned int j = 0; j < num_particles; j++)
        {
            Eigen::VectorXd sample_0(dim);
            Eigen::VectorXd sample_1(dim);
            gaussian_pdf_0.dev(sample_0);
            gaussian_pdf_1.dev(sample_1);
            particles_0.push_back(Particle(sample_0, weight));
            particles_1.push_back(Particle(sample_1, weight));
        }

        // Calculate the entropy and compare the results.
        double estimated_entropy_0 = estimateEntropy(particles_0, num_nearest, num_cores);
        double estimated_entropy_1 = estimateEntropy(particles_1, num_nearest, num_cores);
        ASSERT_GT(estimated_entropy_0, estimated_entropy_1);
    }
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
