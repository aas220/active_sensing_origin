//
// Created by tipakorng on 7/19/17.
//

#include <Eigen/Dense>
#include "gtest/gtest.h"
#include "models/abstract_robot_model.h"
#include "planners/potential_field.h"


struct PotentialFieldTest : testing::Test
{
    AbstractRobot *robot_;

    PotentialField *planner_;

    PotentialFieldTest()
    {
        // Initialize the robot.
        unsigned int dimensions = 2;
        unsigned int observation_size = 1;
        Eigen::VectorXd init_mean(dimensions);
        Eigen::MatrixXd init_cov(dimensions, dimensions);
        Eigen::MatrixXd action_cov(dimensions, dimensions);
        Eigen::MatrixXd sensing_cov(observation_size, observation_size);
        init_mean << 1, 0;
        init_cov << 1e-4, 0, 0, 1e-4;
        action_cov << 1e-4, 0, 0, 1e-4;
        sensing_cov << 1e-4;
        robot_ = new AbstractRobot(init_mean, init_cov, action_cov, sensing_cov);

        // Initialize the planner.
        double step_size = 0.1;
        double threshold = 0.3;
        double attractive_coefficient = 1;
        double repulsive_coefficient = 1e-3;
        planner_ = new PotentialField(*robot_, step_size, threshold, attractive_coefficient, repulsive_coefficient);
    }
};

TEST_F(PotentialFieldTest, testGoal)
{
    // Add goal to the robot.
    Eigen::VectorXd goal_center(robot_->getStateSize());
    goal_center.setZero();
    double goal_radius = 0.1;
    Hypersphere goal(robot_->getStateSize(), goal_radius, goal_center);
    robot_->setGoal(goal);

    // Calculate and check attractive and repulsive forces.
    Eigen::VectorXd state(robot_->getStateSize());
    state << 1, 0;
    Eigen::VectorXd att_force = planner_->attractiveForce(state);
    Eigen::VectorXd rep_force = planner_->repulsiveForce(state);
    ASSERT_LT(att_force(0), 0);
    ASSERT_EQ(att_force(1), 0);
    ASSERT_EQ(rep_force(0), 0);
    ASSERT_EQ(rep_force(1), 0);
}

TEST_F(PotentialFieldTest, testAttractors)
{
    // Add an attractor.
    Eigen::VectorXd attractor(robot_->getStateSize());
    attractor.setZero();
    planner_->addAttractor(attractor);

    // Calculate and check attractive and repulsive forces.
    Eigen::VectorXd state(robot_->getStateSize());
    state << 1, 0;
    Eigen::VectorXd att_force = planner_->attractiveForce(state);
    Eigen::VectorXd rep_force = planner_->repulsiveForce(state);
    ASSERT_LT(att_force(0), 0);
    ASSERT_EQ(att_force(1), 0);
    ASSERT_EQ(rep_force(0), 0);
    ASSERT_EQ(rep_force(1), 0);
}

TEST_F(PotentialFieldTest, testObstacles)
{
    // Add an obstacle to the robot.
    Eigen::VectorXd obstacle_center(robot_->getStateSize());
    obstacle_center << 0.2, 0;
    double obstacle_radius = 0.1;
    Hypersphere obstacle(robot_->getStateSize(), obstacle_radius, obstacle_center);
    robot_->addObstacle(obstacle);

    // Calculate and check attractive and repulsive forces.
    Eigen::VectorXd state(robot_->getStateSize());
    state << 0, 0;
    Eigen::VectorXd att_force = planner_->attractiveForce(state);
    Eigen::VectorXd rep_force = planner_->repulsiveForce(state);
    ASSERT_EQ(att_force(0), 0);
    ASSERT_EQ(att_force(1), 0);
    ASSERT_LT(rep_force(0), 0);
    ASSERT_EQ(rep_force(1), 0);
}

TEST_F(PotentialFieldTest, testRepulsers)
{
    // Add a repulser.
    Eigen::VectorXd repulser(robot_->getStateSize());
    repulser << 0.1, 0;
    planner_->addRepulser(repulser);

    // Calculate and check attractive and repulsive forces.
    Eigen::VectorXd state(robot_->getStateSize());
    state << 0, 0;
    Eigen::VectorXd att_force = planner_->attractiveForce(state);
    Eigen::VectorXd rep_force = planner_->repulsiveForce(state);
    ASSERT_EQ(att_force(0), 0);
    ASSERT_EQ(att_force(1), 0);
    ASSERT_LT(rep_force(0), 0);
    ASSERT_EQ(rep_force(1), 0);
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
