//
// Created by tipakorng on 8/19/17.
//

#include <iostream>
#include <fstream>
#include <ctime>
#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>
#include <string>

#include <ros/ros.h>
#include <tf/transform_broadcaster.h>

#include "models/multitier_peg_hole_2d_model.h"
#include "planners/multitier_peg_hole_2d_planner.h"
#include "simulator.h"
#include "particle_filter.h"

void simulate(Simulator &simulator, Model &model, unsigned int num_trials, unsigned int max_steps,
              std::ofstream &file, unsigned int display)
{
    double total_reward = 0;
    double total_sensing_time = 0;
    unsigned int num_successes = 0;
    Eigen::VectorXd init_state;

    for (unsigned int i = 0; i < num_trials; i++)
    {
        // Run simulation.
        init_state = model.sampleInitState();
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
    // Initialize yml node.
    if (argc < 2)
        throw std::invalid_argument("Need problem definition file!");
    else if (argc < 3)
        throw std::invalid_argument("Need path to output folder!");

    // Get root node of the yml file.
    std::string definition_file(argv[1]);
    YAML::Node root = YAML::LoadFile(definition_file);

    // Set output file name as the current date and time.
    std::time_t raw_time;
    struct tm *time_info;
    char buffer[80];
    time(&raw_time);
    time_info = localtime(&raw_time);
    strftime(buffer, 80, "%Y-%m-%d-%H:%M:%S", time_info);
    std::string file_name(buffer);
    std::string output_path = argv[2] + file_name + ".txt";

    // Initialize ROS node.
    ros::init(argc, argv, "multitier_peg_hole_2d_sim");
    ros::NodeHandle node_handle("active_sensing");

    // Initialize the model.
    MultitierPegHole2dModel model(argv[1], &node_handle);
    model.publishMap();

    // Initilaize particle filter.
    unsigned int num_particles = root["particle_filter"]["num_particles"].as<uint>();
    ParticleFilter particle_filter(model, num_particles, &node_handle);

    // Initialize state-space planner.
    MultitierPegHole2dPlanner state_space_planner(argv[1]);

    // Initialize active sensing.
    unsigned long num_sensing_actions = model.getSensingActions().size();
    unsigned int horizon = root["active_sensing"]["horizon"].as<uint>();
    double discount = root["active_sensing"]["discount"].as<double>();
    unsigned int num_observation_samples = root["active_sensing"]["num_observation_samples"].as<uint>();
    unsigned int num_nearest_neighbors = root["active_sensing"]["num_nearest_neighbors"].as<uint>();
    unsigned int num_cores = root["active_sensing"]["num_cores"].as<uint>();
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
    unsigned int num_trials = root["simulator"]["num_trials"].as<uint>();
    unsigned int max_steps = root["simulator"]["max_steps"].as<uint>();
    unsigned int sensing_interval = root["simulator"]["sensing_intervals"].as<uint>();
    unsigned int verbosity = root["simulator"]["verbosity"].as<uint>();
    Simulator random_simulator(model, random_planner, &node_handle, sensing_interval);
    Simulator state_entropy_simulator(model, state_entropy_planner, &node_handle, sensing_interval);
    Simulator action_entropy_simulator(model, action_entropy_planner, &node_handle, sensing_interval);

    // Open output file and write simulation configuration.
    std::ofstream file;
    file.open(output_path);
    file << "Peg-Hole Simulation in 2D" << std::endl;
    file << "==========" << std::endl;
    model.printParameters(file);
    file << "==========" << std::endl;
    file << "Particle Filter Parameters" << std::endl;
    file << "num_particles = " << num_particles << std::endl;
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
   simulate(random_simulator, model, num_trials, max_steps, file, verbosity);
   std::cout << "Finished random active sensing simulations." << std::endl << std::endl;
   file << std::endl;

    // Run state-entropy active sensing simulation.
   std::cout << "Running state-entropy active sensing simulations..." << std::endl;
   file << "State-Entropy Active Sensing Simulations" << std::endl;
   simulate(state_entropy_simulator, model, num_trials, max_steps, file, verbosity);
   std::cout << "Finished state-entropy active sensing simulations." << std::endl << std::endl;
   file << std::endl;

    // Run action-entropy active sensing simulation.
    std::cout << "Running action-entropy active sensing simulations..." << std::endl;
    file << "Action-Entropy Active Sensing Simulations" << std::endl;
    simulate(action_entropy_simulator, model, num_trials, max_steps, file, verbosity);
    std::cout << "Finished action-entropy active sensing simulations." << std::endl << std::endl;
    file << std::endl;

    // Close the output file.
    file.close();
}
