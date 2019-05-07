//
// Created by tipakorng on 7/12/17.
//

#include <iostream>
#include <fstream>
#include <ctime>
#include <Eigen/Dense>
#include "models/abstract_robot_model.h"
#include "planners/potential_field.h"
#include "simulator.h"
#include "particle_filter.h"


void simulate(Simulator &simulator, AbstractRobot &model, unsigned int num_trials, unsigned int max_steps,
              std::ofstream &file, bool display)
{
    double total_reward = 0;
    double total_sensing_time = 0;
    unsigned int num_successes = 0;
    Eigen::VectorXd init_state;

    for (unsigned int i = 0; i < num_trials; i++)
    {
        // Run simulation.
        init_state = model.getInitState();
        simulator.simulate(init_state, max_steps, display);

        // Print to screen.
        std::cout << "trial " << i << ", reward = " << simulator.getCumulativeReward();
        std::cout << ", sensing time = " << simulator.getAverageActiveSensingTime();
        std::cout << ", goal reached = " << model.isGoal(simulator.getStates().back()) << std::endl;

        // Print to file.
        file << "trial " << i << ", reward = " << simulator.getCumulativeReward();
        file << ", sensing time = " << simulator.getAverageActiveSensingTime();
        file << ", goal reached = " << model.isGoal(simulator.getStates().back()) << std::endl;

        if (model.isGoal(simulator.getStates().back()))
        {
            num_successes++;
            total_reward += simulator.getCumulativeReward();
            total_sensing_time += simulator.getAverageActiveSensingTime();
        }
    }

    // Print to screen.
    std::cout << "average reward = " << total_reward / num_successes << std::endl;
    std::cout << "average sensing time = " << total_sensing_time / num_successes << std::endl;
    std::cout << "number of success trials = " << num_successes << std::endl;

    // Print to file.
    file << "average reward = " << total_reward / num_successes << std::endl;
    file << "average sensing time = " << total_sensing_time / num_successes << std::endl;
    file << "number of success trials = " << num_successes << std::endl;
}


int main(int argc, char** argv)
{
    // Simulation serial number.
    std::time_t raw_time;
    struct tm *time_info;
    char buffer[80];
    time(&raw_time);
    time_info = localtime(&raw_time);
    strftime(buffer, 80, "%Y-%m-%d-%H:%M:%S", time_info);
    std::string file_name(buffer);

    // Initialize the model.
    unsigned int dimensions = 8;
    Eigen::VectorXd init_mean = Eigen::VectorXd::Zero(dimensions);
    init_mean(0) = 1;
    Eigen::MatrixXd init_cov = 1e-3 * Eigen::MatrixXd::Identity(dimensions, dimensions);
    Eigen::MatrixXd action_cov = 1e-3 * Eigen::MatrixXd::Identity(dimensions, dimensions);
    Eigen::MatrixXd sensing_cov = 1e-3 * Eigen::MatrixXd::Identity(1, 1);
    AbstractRobot model(init_mean, init_cov, action_cov, sensing_cov);

    // Initialize goal.
    double goal_radius = 0.1;
    Hypersphere goal(dimensions, goal_radius, Eigen::VectorXd::Zero(dimensions));
    model.setGoal(goal);

    // Initialize obstacles
    Eigen::VectorXd obstacle_center = Eigen::VectorXd::Zero(dimensions);
    obstacle_center(0) = 0.5;
    double obstacle_radius = 0.1;
    Hypersphere obstacle(dimensions, obstacle_radius, obstacle_center);
    model.addObstacle(obstacle);

    // Initialize particle filter.
    unsigned int num_particles = 100;
    ParticleFilter particle_filter(model, num_particles);

    // Initialize state-space planner.
    double step_size = 0.1;
    double threshold = 0.3;
    double attractive_coeff = 1;
    double repulsive_coeff = 1e-3;
    PotentialField state_space_planner(model, step_size, threshold, attractive_coeff, repulsive_coeff);

    // Initialize active sensing.
    unsigned int num_sensing_actions = dimensions;
    unsigned int horizon = 1;
    double discount = 1.0;
    unsigned int num_observation_samples = 10;
    unsigned int num_nearest_neighbors = 5;
    unsigned int num_cores = 2;
    RandomActiveSensing random_action_sensing(model, state_space_planner, particle_filter, horizon, discount);
    ActiveSensing state_entropy_active_sensing(model, state_space_planner, particle_filter, horizon, discount,
                                               num_observation_samples, num_nearest_neighbors, num_cores);
    ActionEntropyActiveSensing action_entropy_active_sensing(model, state_space_planner, particle_filter, horizon,
                                                             discount, num_observation_samples, num_nearest_neighbors,
                                                             num_cores);

    // Initialize belief space planners.
    BeliefSpacePlanner random_planner(state_space_planner, random_action_sensing, particle_filter);
    BeliefSpacePlanner state_entropy_planner(state_space_planner, state_entropy_active_sensing, particle_filter);
    BeliefSpacePlanner action_entropy_planner(state_space_planner, action_entropy_active_sensing, particle_filter);

    // Initialize simulators.
    unsigned int num_trials = 1000;
    unsigned int max_steps = 100;
    unsigned int sensing_interval = 0;
    bool display = false;
    Simulator random_simulator(model, random_planner, sensing_interval);
    Simulator state_entropy_simulator(model, state_entropy_planner, sensing_interval);
    Simulator action_entropy_simulator(model, action_entropy_planner, sensing_interval);

    // Open output file and write simulation configuration.
    std::ofstream file;
    file.open("../output/abstract_robot/" + file_name + ".txt");
    file << "Abstract Robot Simulation: " << file_name << std::endl;
    file << "==========" << std::endl;
    file << "Model Parameters" << std::endl;
    file << "init_mean = \n" << init_mean.transpose() << std::endl;
    file << "init_cov = \n" << init_cov << std::endl;
    file << "action_cov = \n" << action_cov << std::endl;
    file << "sensing_cov = \n" << sensing_cov << std::endl;
    file << "obstacle_center = \n" << obstacle_center.transpose() << std::endl;
    file << "obstacle_radius = " << obstacle_radius << std::endl;
    file << "goal_radius = " << goal_radius << std::endl;
    file << "num_sensing_actions = " << num_sensing_actions << std::endl;
    file << "==========" << std::endl;
    file << "Particle Filter Parameters" << std::endl;
    file << "num_particles = " << num_particles << std::endl;
    file << "==========" << std::endl;
    file << "State-Space Planner Parameters" << std::endl;
    file << "step_size = " << step_size << std::endl;
    file << "threshold = " << threshold << std::endl;
    file << "attractive_coeff = " << attractive_coeff << std::endl;
    file << "repulsive_coeff = " << repulsive_coeff << std::endl;
    file << "==========" << std::endl;
    file << "Active-Sensing Parameters" << std::endl;
    file << "horizon = " << horizon << std::endl;
    file << "discount = " << discount << std::endl;
    file << "num_observation_samples = " << num_observation_samples << std::endl;
    file << "num_nearest_neighbors = " << num_nearest_neighbors << std::endl;
    file << "num_cores = " << num_cores << std::endl;
    file << "==========" << std::endl;
    file << "Simulator Parameters" << std::endl;
    file << "num_trials = " << num_trials << std::endl;
    file << "max_steps = " << max_steps << std::endl;
    file << "sensing_interval = " << sensing_interval << std::endl;
    file << "==========" << std::endl;
    file << std::endl;

    // Run random active sensing simulation.
    std::cout << "Running random active sensing simulations..." << std::endl;
    file << "Random Active Sensing Simulations" << std::endl;
    simulate(random_simulator, model, num_trials, max_steps, file, display);
    std::cout << "Finished random active sensing simulations." << std::endl << std::endl;
    file << std::endl;

    // Run random active sensing simulation.
    std::cout << "Running state-entropy active sensing simulations..." << std::endl;
    file << "State-Entropy Active Sensing Simulations" << std::endl;
    simulate(state_entropy_simulator, model, num_trials, max_steps, file, display);
    std::cout << "Finished state-entropy active sensing simulations." << std::endl << std::endl;
    file << std::endl;

    // Run random active sensing simulation.
    std::cout << "Running action-entropy active sensing simulations..." << std::endl;
    file << "Action-Entropy Active Sensing Simulations" << std::endl;
    simulate(action_entropy_simulator, model, num_trials, max_steps, file, display);
    std::cout << "Finished action-entropy active sensing simulations." << std::endl << std::endl;
    file << std::endl;

    // Close the output file.
    file.close();

    return 0;
}
