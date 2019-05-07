//
// Created by tipakorng on 7/26/17.
//

#include <Eigen/Dense>
#include "gtest/gtest.h"
#include "models/peg_hole_2d_model.h"
#include "planners/peg_hole_2d_planner.h"
#include "math_utils.h"


struct PegHole2dTest : testing::Test
{
    PegHole2d *robot;

    PegHole2dPlanner *planner;

    unsigned int state_size;

    double peg_width;

    double peg_height;

    double hole_tolerance;

    double trans_step_size;

    double rot_step_size;

    Eigen::VectorXd init_mean;

    Eigen::MatrixXd init_cov;

    Eigen::MatrixXd motion_cov;

    double sensing_cov;

    PegHole2dTest()
    {
        state_size = 3;
        init_mean.resize(state_size, 1);
        init_mean << 2, 2, 0;
        init_cov.resize(state_size, state_size);
        init_cov = 1e-2 * Eigen::MatrixXd::Identity(state_size, state_size);
        motion_cov.resize(state_size, state_size);
        motion_cov = 1e-2 * Eigen::MatrixXd::Identity(state_size, state_size);
        sensing_cov = 1e-2;

        trans_step_size = 0.1;
        rot_step_size = 0.1;
    }

    virtual void SetUp(double new_peg_width, double new_peg_height, double new_hole_tolerance)
    {
        robot = new PegHole2d(new_peg_width, new_peg_height, new_hole_tolerance,
                              init_mean, init_cov, motion_cov, sensing_cov);

        planner = new PegHole2dPlanner(new_peg_width, new_peg_height, new_hole_tolerance,
                                       trans_step_size, rot_step_size);
    }

    virtual void SetUp(double new_peg_width, double new_peg_height, double new_hole_tolerance,
                       double new_trans_step_size, double new_rot_step_size)
    {
        robot = new PegHole2d(new_peg_width, new_peg_height, new_hole_tolerance,
                              init_mean, init_cov, motion_cov, sensing_cov);

        planner = new PegHole2dPlanner(new_peg_width, new_peg_height, new_hole_tolerance,
                                       new_trans_step_size, new_rot_step_size);
    }

    virtual void TearDown()
    {
        delete robot;
        delete planner;
    }
};

TEST_F(PegHole2dTest, testCollisionDetection)
{
    double peg_width = 1;
    double peg_height = 2;
    double hole_tolerance = 0.1;
    SetUp(peg_width, peg_height, hole_tolerance);

    Eigen::VectorXd state(3);

    state << 2, 0, 0;
    ASSERT_TRUE(robot->isCollision(state));

    state << 2, 2, 0;
    ASSERT_FALSE(robot->isCollision(state));

    state << 0, 0, 0.25 * M_PI;
    ASSERT_TRUE(robot->isCollision(state));

    state << 0, 0, M_PI;
    ASSERT_FALSE(robot->isCollision(state));
}

TEST_F(PegHole2dTest, testNextState)
{
    double peg_width = 1;
    double peg_height = 1;
    double hole_tolerance = 0;
    SetUp(peg_width, peg_height, hole_tolerance);

    Eigen::VectorXd state(state_size);
    Eigen::VectorXd action(state_size);
    Eigen::VectorXd next_state(state_size);

    state << 10, 1, 0;
    action << 0, -1, 0;
    next_state = robot->getNextState(state, action);
    ASSERT_EQ(next_state(0), state(0));
    ASSERT_NEAR(next_state(1), 0.5, 0.1);
    ASSERT_EQ(next_state(2), state(2));

    state << 10, 2, 0.25 * M_PI;
    action << 0, -2, 0;
    next_state = robot->getNextState(state, action);
    ASSERT_EQ(next_state(0), state(0));
    ASSERT_NEAR(next_state(1), sqrt(2)/2, 0.1);
    ASSERT_EQ(next_state(2), state(2));
}

TEST_F(PegHole2dTest, testPlanner)
{
    double peg_width = 1;
    double peg_height = 1;
    double hole_tolerance = 0;
    double trans_step_size = 0.1;
    double rot_step_size = 0.1;
    SetUp(peg_width, peg_height, hole_tolerance, trans_step_size, rot_step_size);

    Eigen::VectorXd state(state_size);
    Eigen::VectorXd action(state_size);

    state << 10, 5, M_PI;
    action = planner->policy(state);
    ASSERT_NEAR(action(0), -trans_step_size * math::sgn(state(0)), std::numeric_limits<double>::epsilon());
    ASSERT_EQ(action(1), 0);
    ASSERT_EQ(action(2), 0);

    state << 0, 5, 0;
    action = planner->policy(state);
    ASSERT_EQ(action(0), 0);
    ASSERT_NEAR(action(1), -trans_step_size * math::sgn(state(1)), std::numeric_limits<double>::epsilon());
    ASSERT_EQ(action(2), 0);

    state << 0, 5, M_PI;
    action = planner->policy(state);
    ASSERT_EQ(action(0), 0);
    ASSERT_NEAR(action(1), -trans_step_size * math::sgn(state(1)), std::numeric_limits<double>::epsilon());
    ASSERT_NEAR(action(2), -rot_step_size * math::sgn(state(2)), std::numeric_limits<double>::epsilon());
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "test_peghole2d_node");
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
