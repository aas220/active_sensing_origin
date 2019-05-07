//
// Created by tipakorng on 8/1/17.
//

#include <Eigen/Dense>
#include "gtest/gtest.h"
#include "models/peg_hole_3d_model.h"


struct PegHole3dTest : testing::Test
{
    PegHole3d *robot;

    unsigned int state_size;

    Eigen::VectorXd init_mean;

    Eigen::MatrixXd init_cov;

    Eigen::MatrixXd motion_cov;

    double sensing_cov;

    PegHole3dTest()
    {
        state_size = 6;

        init_mean.resize(state_size, 1);
        init_mean.setZero();
        init_cov.resize(state_size, state_size);
        init_cov = 1e-2 * Eigen::MatrixXd::Identity(state_size, state_size);
        motion_cov.resize(state_size, state_size);
        motion_cov = 1e-2 * Eigen::MatrixXd::Identity(state_size, state_size);
        sensing_cov = 1e-2;
    }

    virtual void SetUp(double peg_dim_1, double peg_dim_2, double peg_dim_3,
                       double wall_distance, double new_hole_tolerance)
    {
        robot = new PegHole3d(peg_dim_1, peg_dim_2, peg_dim_3, wall_distance, wall_distance, wall_distance,
                              new_hole_tolerance, init_mean, init_cov, motion_cov, sensing_cov, 0);
    }

    virtual void TearDown()
    {
        delete robot;
    }
};

TEST_F(PegHole3dTest, testCollisionDetection)
{
    double peg_dim_1 = 1;
    double peg_dim_2 = 1;
    double peg_dim_3 = 2;
    double wall_distance = 100;
    double hole_tolerance = 0.1;
    SetUp(peg_dim_1, peg_dim_2, peg_dim_3, wall_distance, hole_tolerance);

    Eigen::VectorXd state(state_size);

    // Test simple collision.
    state << 10, 0, 0, 0, 0, 0;
    ASSERT_TRUE(robot->isCollision(state));

    // Test simple no collision.
    state << 10, 0, peg_dim_3, 0, 0, 0;
    ASSERT_FALSE(robot->isCollision(state));

    // Test no collision when peg is inside hole.
    state << 0, 0, 0, 0, 0, 0;
    ASSERT_FALSE(robot->isCollision(state));
    state << -0.5 * hole_tolerance, 0, 0, 0, 0, 0;
    ASSERT_FALSE(robot->isCollision(state));
    state << 0.5 * hole_tolerance, 0, 0, 0, 0, 0;
    ASSERT_FALSE(robot->isCollision(state));
    state << 0, -0.5 * hole_tolerance, 0, 0, 0, 0;
    ASSERT_FALSE(robot->isCollision(state));
    state << 0, 0.5 * hole_tolerance, 0, 0, 0, 0;
    ASSERT_FALSE(robot->isCollision(state));

    // Test collision with the walls.
    state << wall_distance, 0, peg_dim_3, 0, 0, 0;
    ASSERT_TRUE(robot->isCollision(state));
    state << -wall_distance, 0, peg_dim_3, 0, 0, 0;
    ASSERT_TRUE(robot->isCollision(state));
    state << 0, wall_distance, peg_dim_3, 0, 0, 0;
    ASSERT_TRUE(robot->isCollision(state));
    state << 0, -wall_distance, peg_dim_3, 0, 0, 0;
    ASSERT_TRUE(robot->isCollision(state));
    state << 0, 0, wall_distance, 0, 0, 0;
    ASSERT_TRUE(robot->isCollision(state));
    state << 0, 0, -hole_tolerance, 0, 0, 0;
    ASSERT_TRUE(robot->isCollision(state));

    // Test collision with sides of the hole.
    state << -hole_tolerance, 0, 0, 0, 0, 0;
    ASSERT_TRUE(robot->isCollision(state));
    state << hole_tolerance, 0, 0, 0, 0, 0;
    ASSERT_TRUE(robot->isCollision(state));
    state << 0, -hole_tolerance, 0, 0, 0, 0;
    ASSERT_TRUE(robot->isCollision(state));
    state << 0, hole_tolerance, 0, 0, 0, 0;
    ASSERT_TRUE(robot->isCollision(state));

    // Test rotation.
    state << 0, 0, 0, 0.25 * M_PI, 0, 0;
    ASSERT_TRUE(robot->isCollision(state));
    state << 0, 0, 0, 0, 0.25 * M_PI, 0;
    ASSERT_TRUE(robot->isCollision(state));
    state << 0, 0, 0, 0, 0, 0.25 * M_PI;
    ASSERT_TRUE(robot->isCollision(state));
}

TEST_F(PegHole3dTest, testNextState)
{
    double peg_dim_1 = 1;
    double peg_dim_2 = 1;
    double peg_dim_3 = 2;
    double wall_distance = 100;
    double hole_tolerance = 0.1;
    SetUp(peg_dim_1, peg_dim_2, peg_dim_3, wall_distance, hole_tolerance);

    Eigen::VectorXd state(state_size);
    Eigen::VectorXd action(state_size);
    Eigen::VectorXd next_state(state_size);

    // The distance from peg to ground is 0.1.
    // We can't move the peg down by 1.
    state << 10, 0, 1.1, 0, 0, 0;
    action << 0, 0, -1, 0, 0, 0;
    next_state = robot->getNextState(state, action);
    ASSERT_FALSE(robot->isCollision(state));
    ASSERT_EQ(next_state(0), state(0));
    ASSERT_EQ(next_state(1), state(1));
    ASSERT_NEAR(next_state(2), 1, 0.1);

    // However, we can move the peg down when it is above the hole.
    state << 0, 0, 1.1, 0, 0, 0;
    action << 0, 0, -1, 0, 0, 0;
    next_state = robot->getNextState(state, action);
    ASSERT_FALSE(robot->isCollision(state));
    ASSERT_EQ(next_state(0), state(0));
    ASSERT_EQ(next_state(1), state(1));
    ASSERT_NEAR(next_state(2), 0.1, 1e-8);
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

