//
// Created by tipakorng on 3/3/16.
//

#ifndef ACTIVE_SENSING_CONTINUOUS_SIMULATOR_H
#define ACTIVE_SENSING_CONTINUOUS_SIMULATOR_H

#include <iostream>
#include <chrono>
#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include "model.h"
#include "planner.h"


class Simulator {

public:

    /**
     * \brief Constructor.
     *
     * @param model The model of the system.
     * @param planner The state-space planner.
     */
    explicit Simulator(Model &model, BeliefSpacePlanner &planner, unsigned int sensing_interval=0);

    /**
     * \brief Constructor.
     *
     * @param model The model of the system.
     * @param planner The state-space planner.
     */
    explicit Simulator(Model &model, BeliefSpacePlanner &planner, ros::NodeHandle *node_handle,
                       unsigned int sensing_interval=0);

    /**
     * \brief Destructor.
     */
    virtual ~Simulator();

    /**
     * \brief Initialize simulator.
     */
    void initSimulator();

    /**
     * \brief Update simulator states with sensing and observation.
     *
     * @param sensing_action A sensing action.
     * @param observation An observation.
     * @param task_action A task action.
     */
    void updateSimulator(unsigned int sensing_action, Eigen::VectorXd observation, Eigen::VectorXd task_action);

    /**
     * \brief Update simulator states without sensing and observation.
     *
     * @param task_action A task action.
     */
    void updateSimulator(Eigen::VectorXd task_action);

    /**
     * \brief Run simulation for a number of steps.
     *
     * @param num_steps The number of steps the simulator will run.
     */
    void simulate(const Eigen::VectorXd &init_state, unsigned int num_steps, unsigned int verbosity=0);

    /**
     * \brief Return the state trajectory.
     *
     * @return The state trajectory.
     */
    std::vector<Eigen::VectorXd> getStates();

    /**
     * \brief Return the sensing action trajectory.
     *
     * @return The sensing action trajectory.
     */
    std::vector<unsigned int> getSensingActions();

    /**
     * \brief Return the task action trajectory.
     *
     * @return The task action trajectory.
     */
    std::vector<Eigen::VectorXd> getTaskActions();

    /**
     * \brief Return the observation trajectory.
     *
     * @return The observation trajectory.
     */
    std::vector<Eigen::VectorXd> getObservations();

    /**
     * \brief Return the cumulative reward.
     *
     * @return The cumulative reward.
     */
    double getCumulativeReward();

    /**
     * \brief Return the time the active sensing algorithm took in seconds.
     *
     * @return The time the active sensing took (s).
     */
    double getAverageActiveSensingTime();
    double getAvgObservationTime();
    double getAvgUpdatebeliefTime();
    double getAvgTotalUpdatebeliefTime();
    double getAvgTaskactionTime();
    double getAvgPredictbeliefTime();
    double getAvgTotalPredictbeliefTime();

    void publishState();

private:

    /**
     * \brief The model of the system.
     */
    Model &model_;

    /**
     * \brief The state-space planner.
     */
    BeliefSpacePlanner &planner_;

    /**
     * \brief The state trajectory.
     */
    std::vector<Eigen::VectorXd> states_;

    /**
     * \brief The task action trajectory.
     */
    std::vector<Eigen::VectorXd> task_actions_;

    /**
     * \brief The sensing action trajectory.
     */
    std::vector<unsigned int> sensing_actions_;

    /**
     * \brief The observation trajectory.
     */
    std::vector<Eigen::VectorXd> observations_;

    /**
     * \brief The cumulative reward of the trajectory.
     */
    double cumulative_reward_;

    /**
     * \brief The cumulative active sensing time.
     */
    double active_sensing_time_;

    /**
     * \brief This is the number of steps between two consecutive sensing actions.
     */
    double observation_time_;
    double updatebelief_time_;
    double total_updatebelief_time_;
    double taskaction_time_;
    double predictbelief_time_;
    double total_predictbelief_time_;

    unsigned int sensing_interval_;

    ros::NodeHandle *node_handle_;

    tf::TransformBroadcaster tf_broadcaster_;

    ros::Publisher publisher_;

    bool has_publisher_;
};

#endif //ACTIVE_SENSING_CONTINUOUS_SIMULATOR_H
