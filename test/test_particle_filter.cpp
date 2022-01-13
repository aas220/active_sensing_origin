//
// Created by tipakorng on 12/4/15.
//

#include "gtest/gtest.h"
#include "particle_filter.h"
#include "models/abstract_robot_model.h"

struct ParticleFilterTest : testing::Test {

    AbstractRobot *robot;

    ParticleFilter *pf;

    ParticleFilterTest() {
        // Create mobile robot
        Eigen::VectorXd init_mean(2);
        Eigen::MatrixXd init_cov(2, 2);
        Eigen::MatrixXd action_cov(2, 2);
        Eigen::MatrixXd sensing_cov(1, 1);
        Eigen::Vector2d goal_center;
        init_mean << 0, 0;
        init_cov << 1e-6, 0, 0, 1;
        action_cov << 1e-6, 0, 0, 1e-6;
        sensing_cov << 1e-6;
        goal_center << 1, 1;
        double goal_radius = 0.1;
        Hypersphere goal(init_mean.size(), goal_radius, goal_center);
        robot = new AbstractRobot(init_mean, init_cov, action_cov, sensing_cov);
        robot->setGoal(goal);
        // Create particle filter
        unsigned int numParticles = 100;
        pf = new ParticleFilter(*robot, numParticles);
    }

    virtual ~ParticleFilterTest(){
        delete robot;
        delete pf;
    }
};

void printParticles(const std::vector<Particle> &particles) {
    for (int i = 0; i < particles.size(); i++) {
        std::cout << "(value, weight) = " << particles[i].getValue()[0] << ", " << particles[i].getValue()[1]
        << ", " << particles[i].getWeight() << ", " << std::endl;
    }
}

TEST_F(ParticleFilterTest, initParticles) {
    ASSERT_EQ(pf->getNumParticles(), 100);
}

TEST_F(ParticleFilterTest, updateWeights) {
    std::vector<Particle> particles = pf->getParticles();
    std::vector<Particle> newParticles = particles;
    unsigned int sensingAction = 0;
    Eigen::VectorXd observation(1);
    observation << 0;
    pf->updateWeights(newParticles, sensingAction, observation);

    for (int i = 0; i < particles.size(); i++) {
        ASSERT_EQ(particles[i].getValue()[0], newParticles[i].getValue()[0]);
        ASSERT_EQ(particles[i].getValue()[1], newParticles[i].getValue()[1]);
        ASSERT_NE(particles[i].getWeight(), newParticles[i].getWeight());
    }
}

TEST_F(ParticleFilterTest, resample) {
    int numParticles = 100;
    std::vector<Particle> particles;
    Eigen::VectorXd value(2);
    double weight;

    for (int i = 0; i < numParticles - 1; i++) {
        value << 0, 0;
        weight = 1e-6;
        Particle particle(value, weight);
        particles.push_back(particle);
    }

    value << 3.14, 6.28;
    weight = 1;
    Particle particle(value, weight);
    particles.push_back(particle);

    std::vector<Particle> newParticles = particles;
    pf->resample(newParticles);

    for (int i = 0; i < particles.size(); i++) {
        ASSERT_EQ(newParticles[i].getValue()[0], 3.14);
        ASSERT_EQ(newParticles[i].getValue()[1], 6.28);
        ASSERT_EQ(newParticles[i].getWeight(), 1);
    }
}

TEST_F(ParticleFilterTest, propagate) {
    std::vector<Particle> particles = pf->getParticles();
    std::vector<Particle> newParticles = particles;
    Eigen::VectorXd taskAction(2);
    taskAction << 1, 0;
    pf->propagate(newParticles, taskAction);
}

TEST_F(ParticleFilterTest, update) {
    std::vector<Particle> particles = pf->getParticles();
    std::vector<Particle> newParticles = particles;
    Eigen::VectorXd taskAction(2);
    Eigen::VectorXd observation(1);
    unsigned int sensingAction = 0;
    taskAction << 1, 0;
    observation << 0;
    pf->update(newParticles, taskAction, sensingAction, observation);

    for (int i = 0; i < particles.size(); i++) {
        ASSERT_LT((newParticles[i].getValue()-particles[i].getValue()-taskAction).norm(), 1e-2);  // Initial belief is centered around (0, 0)
    }
}

TEST_F(ParticleFilterTest, normalize) {
    std::vector<Particle> particles = pf->getParticles();
    std::vector<Particle> newParticles = particles;
    Eigen::VectorXd taskAction(2);
    taskAction << 1, 0;
    unsigned int sensingAction = 0;
//    Eigen::VectorXd observation({1}); --------------

    Eigen::VectorXd observation(1);
    pf->update(newParticles, taskAction, sensingAction, observation);
    pf->normalize(newParticles);
    double sumWeight = 0;

    for (int i = 0; i < newParticles.size(); i++) {
        sumWeight += newParticles[i].getWeight();
    }

    ASSERT_DOUBLE_EQ(1, sumWeight);
}

TEST_F(ParticleFilterTest, importanceSampling) {
    int numParticles = 100;
    std::vector<Particle> particles;
    Eigen::VectorXd value(2);
    double weight;

    for (int i = 0; i < numParticles - 1; i++) {
        value << 0, 0;
        weight = 1e-6;
        Particle particle(value, weight);
        particles.push_back(particle);
    }

    value << 3.14, 6.28;
    weight = 1;
    Particle particle(value, weight);
    particles.push_back(particle);

    Particle p = pf->importanceSampling(particles);

    ASSERT_EQ(3.14, p.getValue()[0]);
    ASSERT_EQ(6.28, p.getValue()[1]);
    ASSERT_EQ(1, p.getWeight());
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "test_particlefilter_node");
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
