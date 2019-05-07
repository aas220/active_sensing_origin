//
// Created by tipakorng on 3/3/16.
//

#ifndef ACTIVE_SENSING_CONTINUOUS_BELIEF_SPACE_PLANNER_H
#define ACTIVE_SENSING_CONTINUOUS_BELIEF_SPACE_PLANNER_H

#include <ros/ros.h>

#include "model.h"
#include "active_sensing.h"
#include "particle_filter.h"
#include "state_space_planner.h"


class BeliefSpacePlanner {

public:

    /**
     * \brief The constructor.
     *
     * @param state_space_planner A state-space planner.
     * @param active_sensing An active-sensing object.
     * @param particle_filter A particle filter.
     */
    explicit BeliefSpacePlanner(StateSpacePlanner &state_space_planner, ActiveSensing &active_sensing,
                           ParticleFilter &particle_filter);

    /**
     * \brief The constructor.
     *
     * @param state_space_planner A state-space planner.
     * @param active_sensing An active-sensing object.
     * @param particle_filter A particle filter.
     */
    explicit BeliefSpacePlanner(StateSpacePlanner &state_space_planner, ActiveSensing &active_sensing,
                                ParticleFilter &particle_filter, ros::NodeHandle *node_handle);

    /**
     * \brief The destructor.
     */
    virtual ~BeliefSpacePlanner();

    /**
     * \brief Reset the planner and its components.
     */
    virtual void reset();
    
    /**
     * \brief Get the maximum-likelihood state.
     * 
     * @return The maximumlikelihood state.
     */
    Eigen::VectorXd getMaximumLikelihoodState() const;

    /**
     * \brief Get task action from the maximum-likelihood state.
     *
     * @return A task action.
     */
    Eigen::VectorXd getTaskAction();

    /**
     * \brief Get a sensing action.
     *
     * @return A sensing action.
     */
    unsigned int getSensingAction() const;

    /**
     * \brief Propagate the particles according to the task action and the motion model.
     *
     * @param task_action A task action.
     */
    void predictBelief(const Eigen::VectorXd &task_action);

    /**
     * \brief Update the particles' weight with the sensing action, the observation, and the observation model.
     *
     * @param sensing_action A sensing action.
     * @param observation An observation
     */
    void updateBelief(unsigned int sensing_action, const Eigen::VectorXd &observation);

    /**
     * \brief Normalize the particles' weight.
     */
    void normalizeBelief();

    void publishParticles() const;

    bool isPublishable() const;

protected:

    /**
     * \brief The state-space planner.
     */
    StateSpacePlanner &state_space_planner_;

    /**
     * \brief The active-sensing algorithm.
     */
    ActiveSensing &active_sensing_;

    /**
     * \brief The particle filter.
     */
    ParticleFilter &particle_filter_;

    /**
     * \brief Pointer to ROS node handle.
     */
    ros::NodeHandle *node_handle_;

    tf::TransformBroadcaster tf_broadcaster_;

    ros::Publisher publisher_;

    bool has_publisher_;

};

#endif //ACTIVE_SENSING_CONTINUOUS_BELIEF_SPACE_PLANNER_H
