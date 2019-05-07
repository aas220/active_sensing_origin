//
// Created by tipakorng on 3/8/16.
//

#include <visualization_msgs/MarkerArray.h>
#include <tf/transform_broadcaster.h>
#include "planner.h"

BeliefSpacePlanner::BeliefSpacePlanner(StateSpacePlanner &state_space_planner, ActiveSensing &active_sensing,
                                       ParticleFilter &particle_filter) :
    state_space_planner_(state_space_planner),
    active_sensing_(active_sensing),
    particle_filter_(particle_filter)
{
    has_publisher_ = false;
}

BeliefSpacePlanner::BeliefSpacePlanner(StateSpacePlanner &state_space_planner, ActiveSensing &active_sensing,
                                       ParticleFilter &particle_filter, ros::NodeHandle *node_handle) :
        state_space_planner_(state_space_planner),
        active_sensing_(active_sensing),
        particle_filter_(particle_filter),
        node_handle_(node_handle)
{
    publisher_ = node_handle_->advertise<visualization_msgs::MarkerArray>("planner", 1);
    has_publisher_ = true;
}

BeliefSpacePlanner::~BeliefSpacePlanner()
{}

void BeliefSpacePlanner::reset()
{
    state_space_planner_.reset();
    particle_filter_.initParticles();
}

Eigen::VectorXd BeliefSpacePlanner::getMaximumLikelihoodState() const
{
    return particle_filter_.getBestParticle().getValue();
}

Eigen::VectorXd BeliefSpacePlanner::getTaskAction()
{
    return state_space_planner_.policy(getMaximumLikelihoodState());
}

unsigned int BeliefSpacePlanner::getSensingAction() const
{
    return active_sensing_.getSensingAction(particle_filter_.getParticles());
}

void BeliefSpacePlanner::predictBelief(const Eigen::VectorXd &task_action)
{
    particle_filter_.propagate(task_action);
}

void BeliefSpacePlanner::updateBelief(unsigned int sensing_action, const Eigen::VectorXd &observation)
{
    particle_filter_.updateWeights(sensing_action, observation);
    particle_filter_.resample();
}

void BeliefSpacePlanner::normalizeBelief()
{
    particle_filter_.normalize();
}

void BeliefSpacePlanner::publishParticles() const
{
    particle_filter_.publish();
}

bool BeliefSpacePlanner::isPublishable() const
{
    return has_publisher_;
}
