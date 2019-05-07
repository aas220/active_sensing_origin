//
// Created by tipakorng on 3/8/16.
//

#include "gtest/gtest.h"
#include "particle_filter.h"
#include "active_sensing.h"
#include "models/abstract_robot_model.h"
#include "planners/potential_field.h"
#include "simulator.h"

struct ActiveSensingTest : testing::Test {

    AbstractRobot *robot;

    ParticleFilter *particle_filter;

    PotentialField *potential_field;

    BeliefSpacePlanner *planner;

    BeliefSpacePlanner *aePlanner;

    ActiveSensing *state_entropy_active_sensing;

    ActionEntropyActiveSensing *action_entropy_active_sensing;

    Simulator *sim;

    Simulator *aeSim;

    ActiveSensingTest() {
        // Create mobile robot
        Eigen::VectorXd initMean(2);
        Eigen::MatrixXd initCov(2, 2);
        Eigen::MatrixXd actionCov(2, 2);
        Eigen::MatrixXd sensingCov(1, 1);
        Eigen::Vector2d goalCenter;
        initMean << 0, 0;
        initCov << 1e-3, 0, 0, 1e-3;  // Robot's initial belief spreads less along the "x" direction, so sensing action 0 makes more sense
        actionCov << 1e-4, 0, 0, 1e-4;
        sensingCov << 1e-4;
        goalCenter << 0, 2;
        double goalRadius = 0.1;
        Hypersphere goal(goalCenter.size(), goalRadius, goalCenter);
        robot = new AbstractRobot(initMean, initCov, actionCov, sensingCov);
        robot->setGoal(goal);
        // Create particle filter
        unsigned int numParticles = 100;
        particle_filter = new ParticleFilter(*robot, numParticles);
        // Create planner
        double stepSize = 0.1;
        double obstacleDistanceThreshold = 0.3;
        double attCoeff = 1.0;
        double repCoeff = 1e-3;
        potential_field = new PotentialField(*robot, stepSize, obstacleDistanceThreshold, attCoeff, repCoeff);
        // Create active sensing
        unsigned int horizon = 1;
        double discount = 1.0;
        unsigned int numObservations = 10;
        state_entropy_active_sensing = new ActiveSensing(*robot, *potential_field, *particle_filter, horizon, discount);
        state_entropy_active_sensing->setNumObservations(numObservations);
        // Create action entropy action sensing
        action_entropy_active_sensing = new ActionEntropyActiveSensing(*robot, *potential_field, *particle_filter, horizon, discount);
        action_entropy_active_sensing->setNumObservations(numObservations);
        // Create belief space planner
        planner = new BeliefSpacePlanner(*potential_field, *state_entropy_active_sensing, *particle_filter);
        aePlanner = new BeliefSpacePlanner(*potential_field, *action_entropy_active_sensing, *particle_filter);
        // Create simulator
        sim = new Simulator(*robot, *planner);
        aeSim = new Simulator(*robot, *aePlanner);
    }

    virtual ~ActiveSensingTest(){
        delete robot;
        delete particle_filter;
        delete potential_field;
        delete state_entropy_active_sensing;
        delete action_entropy_active_sensing;
        delete planner;
        delete aePlanner;
        delete sim;
        delete aeSim;
    }
};

TEST_F(ActiveSensingTest, simulate) {
    unsigned int numSteps = 20;
    sim->simulate(robot->getInitState(), numSteps);
    std::vector<Eigen::VectorXd> states = sim->getStates();
    Eigen::VectorXd lastState = states.back();
    std::vector<unsigned int> sensingActions = sim->getSensingActions();
    std::cout << "last state = " << lastState[0] << ", " << lastState[1] << std::endl;
    std::cout << "sensing action = ";

    for (int i = 0; i < sim->getSensingActions().size(); i++) {
        std::cout << sensingActions[i] << ", ";
    }

    std::cout << std::endl;
}

TEST_F(ActiveSensingTest, aeSimulate) {
    unsigned int numSteps = 20;
    aeSim->simulate(robot->getInitState(), numSteps);
    std::vector<Eigen::VectorXd> states = aeSim->getStates();
    Eigen::VectorXd lastState = states.back();
    std::vector<unsigned int> sensingActions = aeSim->getSensingActions();
    std::cout << "last state = " << lastState[0] << ", " << lastState[1] << std::endl;
    std::cout << "sensing action = ";

    for (int i = 0; i < aeSim->getSensingActions().size(); i++) {
        std::cout << sensingActions[i] << ", ";
    }

    std::cout << std::endl;
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "test_simulator_node");

//    ros::NodeHandle nh;
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
