//
// Created by tipakorng on 12/4/15.
//

#include <Eigen/Dense>
#include <cmath>
#include "gtest/gtest.h"
#include "models/abstract_robot_model.h"

struct AbstractRobotTest : testing::Test
{
    AbstractRobot *robot_;

    unsigned int dimensions_ = 2;

    unsigned int observation_size_ = 1;

    double goal_radius_ = 0.1;

    Eigen::VectorXd init_mean_;

    Eigen::MatrixXd init_cov_;

    Eigen::MatrixXd action_cov_;

    Eigen::MatrixXd sensing_cov_;

    Eigen::VectorXd goal_center_;

    AbstractRobotTest()
    {
        init_mean_.resize(dimensions_);
        init_cov_.resize(dimensions_, dimensions_);
        action_cov_.resize(dimensions_, dimensions_);
        sensing_cov_.resize(observation_size_, observation_size_);
        goal_center_.resize(dimensions_);
        init_mean_ << 0, 0;
        init_cov_ << 1e-4, 0, 0, 1e-4;
        action_cov_ << 1e-4, 0, 0, 1e-4;
        sensing_cov_ << 1e-4;
        goal_center_ << 1, 1;
        Hypersphere goal(dimensions_, goal_radius_, goal_center_);
        robot_ = new AbstractRobot(init_mean_, init_cov_, action_cov_, sensing_cov_);
        robot_->setGoal(goal);
    }

    virtual ~AbstractRobotTest()
    {
        delete robot_;
    }
};

TEST_F(AbstractRobotTest, testHypersphere)
{
    // Initialize a hypersphere in 3D (so a sphere in this case).
    unsigned int dimensions = 3;
    double radius = 1;
    Eigen::VectorXd center(dimensions);
    center << 0, 0, 0;
    Hypersphere hypersphere(dimensions, radius, center);

    // Initialize the points for testing.
    Eigen::VectorXd x1(dimensions);
    Eigen::VectorXd x2(dimensions);

    // Test the case when the line intersects the sphere at two points.
    x1 << -2, 0, 0;
    x2 << 2, 0, 0;
    ASSERT_TRUE(hypersphere.intersects(x1, x2));
    ASSERT_EQ(x2(0), -1);
    ASSERT_EQ(x2(1), 0);
    ASSERT_EQ(x2(2), 0);

    // Test the case when the line intersects the sphere at one points.
    x1 << -2, 0, 0;
    x2 << 0.5, 0, 0;
    ASSERT_TRUE(hypersphere.intersects(x1, x2));
    ASSERT_EQ(x2(0), -1);
    ASSERT_EQ(x2(1), 0);
    ASSERT_EQ(x2(2), 0);

    // Test the case when the line is tangent to the sphere.
    x1 << -2, 1, 0;
    x2 << 2, 1, 0;
    ASSERT_TRUE(hypersphere.intersects(x1, x2));
    ASSERT_EQ(x2(0), 0);
    ASSERT_EQ(x2(1), 1);
    ASSERT_EQ(x2(2), 0);

    // Test the case when the x1 and x2 are the same point not on the line.
    x1 << 2, 0, 0;
    x2 << 2, 0, 0;
    ASSERT_FALSE(hypersphere.intersects(x1, x2));

    // Test the case when the x1 and x2 are the same point on the line.
    x1 << 1, 0, 0;
    x2 << 1, 0, 0;
    ASSERT_TRUE(hypersphere.intersects(x1, x2));

    // Test the case when the line does not intersects the sphere.
    x1 << 2, 2, 0;
    x2 << -2, 2, 0;
    ASSERT_FALSE(hypersphere.intersects(x1, x2));

    // Test the case when the line does not intersects the sphere.
    x1 << -3, 0, 0;
    x2 << -2, 0, 0;
    ASSERT_FALSE(hypersphere.intersects(x1, x2));

    // Test the case when the line does not intersects the sphere.
    x1 << 2, 0, 0;
    x2 << 3, 0, 0;
    ASSERT_FALSE(hypersphere.intersects(x1, x2));
}

double gaussian_pdf(const Eigen::VectorXd &x, const Eigen::VectorXd &mean, const Eigen::MatrixXd &cov)
{
    return exp(-0.5 * (x-mean).transpose() * cov.inverse() * (x-mean)) / std::sqrt((2 * M_PI * cov).determinant());
}

TEST_F(AbstractRobotTest, getObservationProbability)
{
    // Initialize variables.
    Eigen::VectorXd state(dimensions_);
    state << 0, 1;
    unsigned int sensing_action;
    Eigen::VectorXd true_observation(observation_size_);
    Eigen::VectorXd observation(observation_size_);
    observation << 1;

    // Check pdf calculation.
    for (sensing_action = 0; sensing_action < dimensions_; sensing_action++)
    {
        true_observation = robot_->getObservation(state, sensing_action);
        ASSERT_EQ(robot_->getObservationProbability(state, sensing_action, observation),
                  gaussian_pdf(observation, true_observation, sensing_cov_));
    }
}

TEST_F(AbstractRobotTest, getTransitionProbability)
{
    // Initialize variables.
    Eigen::VectorXd currentState(dimensions_);
    currentState << 0, 0;
    Eigen::VectorXd nextState(dimensions_);
    nextState << 1, 0;
    Eigen::VectorXd action(dimensions_);
    action << 1, 0;

    // Transition probability must be greater than 0.
    ASSERT_GE(robot_->getTransitionProbability(nextState, currentState, action), 0.0);

    // Check the numerical value of the pdf against a pre-computed number.
    ASSERT_LT(robot_->getTransitionProbability(currentState, currentState, action), 1e-3);
}

TEST_F(AbstractRobotTest, getReward)
{
    // Initialize variables.
    Eigen::VectorXd state(dimensions_);
    state << 1, 1;
    Eigen::VectorXd action(dimensions_);
    action << 1, 0;

    // The reward equals negative distance traveled.
    ASSERT_EQ(robot_->getReward(state, action), -action.norm());
}

TEST_F(AbstractRobotTest, getObservation)
{
    // Initialize variables.
    Eigen::VectorXd state(dimensions_);
    state << 0, 1;
    Eigen::VectorXd observation(observation_size_);
    unsigned int sensing_action;

    // Observe x[0].
    sensing_action = 0;
    observation = robot_->getObservation(state, sensing_action);
    ASSERT_EQ(observation[0], state[sensing_action]);

    // Observe x[1].
    sensing_action = 1;
    observation = robot_->getObservation(state, sensing_action);
    ASSERT_EQ(observation[0], state[sensing_action]);
}

TEST_F(AbstractRobotTest, getNextState)
{
    // Initialize variables.
    Eigen::VectorXd state(dimensions_);
    state << 0, 0;
    Eigen::VectorXd action(dimensions_);
    action << 0, 1;

    // Calculate and check next state.
    Eigen::VectorXd nextState(robot_->getNextState(state, action));
    ASSERT_EQ(nextState[0], state[0] + action[0]);
    ASSERT_EQ(nextState[1], state[1] + action[1]);
}

TEST_F(AbstractRobotTest, sampleNextState)
{
    // Initialize variables.
    Eigen::VectorXd state(dimensions_);
    state << 0, 0;
    Eigen::VectorXd action(dimensions_);
    action << 1, 0;
    Eigen::VectorXd next_state_noiseless(robot_->getNextState(state, action));
    Eigen::VectorXd next_state_average = Eigen::VectorXd::Zero(dimensions_);
    int num_samples = 1000;

    for (int i = 0; i < num_samples; i++)
    {
        next_state_average += robot_->sampleNextState(state, action) / num_samples;
    }

    // The noiseless state and average noisy state should be relatively close to each other.
    ASSERT_LT((next_state_noiseless-next_state_average).norm(), 1e-3);
}

TEST_F(AbstractRobotTest, sampleInitState)
{
    // Initialize variables.
    Eigen::VectorXd init_state_average = Eigen::VectorXd::Zero(dimensions_);
    int num_samples = 1000;

    // Average sampled initial state should be close to the mean of the initial belief.
    for (int i = 0; i < num_samples; i++)
    {
        init_state_average += robot_->sampleInitState() / num_samples;
    }

    ASSERT_LT((init_state_average-init_mean_).norm(), 1e-3);
}

TEST_F(AbstractRobotTest, sampleObservation)
{
    // Init variables.
    Eigen::VectorXd state(dimensions_);
    state << 1, 0;
    Eigen::VectorXd observation_average(observation_size_);
    int num_samples = 1000;

    // For each sensing action, check if the noisy observation is close to the noiseless observation.
    for (unsigned int sensing_action = 0; sensing_action < dimensions_; sensing_action++)
    {
        Eigen::VectorXd observation_noiseless(robot_->getObservation(state, sensing_action));
        observation_average = Eigen::VectorXd::Zero(observation_size_);

        // Average sampled initial state should be close to the mean of the initial belief.
        for (int i = 0; i < num_samples; i++)
        {
            observation_average += robot_->sampleObservation(state, sensing_action) / num_samples;
        }

        ASSERT_LT((observation_average-observation_noiseless).norm(), 1e-3);
    }

}

TEST_F(AbstractRobotTest, boundary)
{
    // Initialize boundary.
    Eigen::VectorXd boundary_min(dimensions_);
    Eigen::VectorXd boundary_max(dimensions_);
    boundary_min << 0, 0;
    boundary_max << 1, 1;
    robot_->setBoundary(boundary_min, boundary_max);

    // Test moving a state inside the boundary.
    Eigen::VectorXd state(dimensions_);
    state << -1, 2;
    robot_->moveInBoundary(state);
    ASSERT_EQ(state(0), 0);
    ASSERT_EQ(state(1), 1);
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
