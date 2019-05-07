//
// Created by tipakorng on 3/8/16.
//

#include "gtest/gtest.h"
#include "particle_filter.h"
#include "active_sensing.h"
#include "models/abstract_robot_model.h"
#include "planners/potential_field.h"
#include "rng.h"

struct ActiveSensingTest : testing::Test
{

    AbstractRobot *robot;
    ParticleFilter *particle_filter;
    PotentialField *potential_field;
    ActiveSensing *state_entropy_active_sensing;
    ActionEntropyActiveSensing *action_entropy_active_sensing;
    ros::NodeHandle *nh;

    unsigned int dim;

    unsigned int num_particles;

    double step_size;
    double obstacle_threshold;
    double attractive_coeff;
    double repulsive_coeff;

    unsigned int horizon;
    double discount;
    unsigned int num_observations;
    unsigned int num_nearest_neighbors;
    unsigned int num_cores;

    Eigen::VectorXd init_mean;
    Eigen::MatrixXd init_cov;
    Eigen::MatrixXd action_cov;
    Eigen::MatrixXd sensing_cov;

    ActiveSensingTest()
    {
        // ROS
        nh = new ros::NodeHandle("active_sensing");

        // Model
        dim = 2;

        // Particle Filter
        num_particles = 10000;

        // State-space Planner
        step_size = 0.1;
        obstacle_threshold = 0.3;
        attractive_coeff = 1.0;
        repulsive_coeff = 1e-3;

        // Active Sensing
        horizon = 1;
        discount = 1.0;
        num_observations = 10;
        num_nearest_neighbors = 20;
        num_cores = 4;
    }

    virtual void SetUp(const Eigen::MatrixXd &new_init_cov, const Eigen::MatrixXd &new_sensing_cov)
    {
        // Create mobile robot
        Eigen::Vector2d goal_center;
        init_mean.resize(dim);
        action_cov.resize(dim, dim);
        init_mean << 0, 0;
        init_cov = new_init_cov;
        action_cov.diagonal() << 0.1, 0.1;
        sensing_cov = new_sensing_cov;
        goal_center << 100, 0;  // The goal is on the x-axis.
        double goal_radius = 0.1;
        Hypersphere goal(dim, goal_radius, goal_center);
        robot = new AbstractRobot(init_mean, init_cov, action_cov, sensing_cov);
        robot->setGoal(goal);

        // Create particle filter
        particle_filter = new ParticleFilter(*robot, num_particles, nh);

        // Create planner
        potential_field = new PotentialField(*robot, step_size, obstacle_threshold, attractive_coeff, repulsive_coeff);

        // Create active sensing
        state_entropy_active_sensing = new ActiveSensing(*robot, *potential_field, *particle_filter, horizon, discount,
                                                         num_observations, num_nearest_neighbors, num_cores);

        // Create action entropy action sensing
        action_entropy_active_sensing = new ActionEntropyActiveSensing(*robot, *potential_field, *particle_filter,
                                                                       horizon, discount, num_observations,
                                                                       num_nearest_neighbors, num_cores);
    }

    virtual void TearDown()
    {
        delete robot;
        delete particle_filter;
        delete potential_field;
        delete state_entropy_active_sensing;
        delete action_entropy_active_sensing;
        delete nh;
    }

    virtual ~ActiveSensingTest()
    {}
};

TEST_F(ActiveSensingTest, testStateEntropyActiveSensing)
{
    unsigned int num_trials = 10;
    Eigen::MatrixXd new_init_cov(dim, dim);
    Eigen::MatrixXd new_sensing_cov(1, 1);
    new_init_cov.setZero();
    new_sensing_cov.setZero();
    Rng rng(0);

    for (unsigned int i = 0; i < num_trials; i++)
    {
        new_init_cov(0, 0) = rng.doub();
        new_init_cov(1, 1) = 0.5 * rng.doub() * new_init_cov(0, 0);  // Make y-axis cov smaller than x-axis cov.
        new_sensing_cov(0, 0) = 0.5 * rng.doub() * new_init_cov(1, 1);  // Make sensing cov smaller than y-axis cov.
        SetUp(new_init_cov, new_sensing_cov);

        particle_filter->initParticles();
        std::vector<Particle> particles = particle_filter->getParticles();

        // The conditional entropy of sensing_action = 0 should be lower because the initial belief
        // spreads more along the x-axis.
        double entropy_0 = state_entropy_active_sensing->getConditionalCumulativeEntropy(particles, 0);
        double entropy_1 = state_entropy_active_sensing->getConditionalCumulativeEntropy(particles, 1);
        ASSERT_LT(entropy_0, entropy_1);
    }
}

TEST_F(ActiveSensingTest, testActionEntropyActiveSensing)
{
    unsigned int num_trials = 10;
    Eigen::MatrixXd new_init_cov(dim, dim);
    Eigen::MatrixXd new_sensing_cov(1, 1);
    new_init_cov.setZero();
    new_sensing_cov.setZero();
    Rng rng(0);

    for (unsigned int i = 0; i < num_trials; i++)
    {
        new_init_cov(0, 0) = rng.doub();
        new_init_cov(1, 1) = 0.5 * rng.doub() * new_init_cov(0, 0);  // Make y-axis cov smaller than x-axis cov.
        new_sensing_cov(0, 0) = 0.5 * rng.doub() * new_init_cov(1, 1);  // Make sensing cov smaller than y-axis cov.
        SetUp(new_init_cov, new_sensing_cov);

        particle_filter->initParticles();
        std::vector<Particle> particles = particle_filter->getParticles();

        // The conditional entropy of sensing_action = 0 should be lower because the initial belief
        // spreads more along the x-axis.
        double entropy_0 = action_entropy_active_sensing->getConditionalCumulativeEntropy(particles, 0);
        double entropy_1 = action_entropy_active_sensing->getConditionalCumulativeEntropy(particles, 1);
        ASSERT_LT(entropy_1, entropy_0);
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "test_active_sensing");
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
