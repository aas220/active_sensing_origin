//
// Created by tipakorng on 3/8/16.
//

#include "../../include/planners/potential_field.h"

PotentialField::PotentialField(const AbstractRobot &robot, double step_size, double threshold,
                               double attractive_coefficient, double repulsive_coefficient) :
    StateSpacePlanner(),
    step_size_(step_size),
    threshold_(threshold),
    att_(attractive_coefficient),
    rep_(repulsive_coefficient)
{
    robot_ = &robot;
}

PotentialField::~PotentialField() { }

Eigen::VectorXd PotentialField::policy(const Eigen::VectorXd &state)
{
    Eigen::VectorXd force = repulsiveForce(state) + attractiveForce(state);
    double force_magnitude = force.norm();

    if (force_magnitude > step_size_)
        return step_size_ * force / force_magnitude;

    else
        return force;
}

void PotentialField::setStepSize(double new_step_size)
{
    step_size_ = new_step_size;
}

double PotentialField::getStepSize() const
{
    return step_size_;
}

Eigen::VectorXd PotentialField::repulsiveForce(const Eigen::VectorXd &state) const
{
    Eigen::VectorXd relative_position = Eigen::VectorXd::Zero(robot_->getStateSize());
    Eigen::VectorXd force = Eigen::VectorXd::Zero(robot_->getStateSize());
    double distance;

    for (unsigned int i = 0; i < robot_->getNumObstacles(); i++)
    {
        relative_position = state - robot_->obstacle(i)->getCenter();
        distance = robot_->obstacle(i)->distanceClosestPoint(state) + 1e-8;

        if (distance < threshold_)
        {
            force -= rep_ * (1/threshold_ - 1/distance) * relative_position / relative_position.norm() / pow(distance, 2);
        }
    }

    for (Eigen::VectorXd repulser : repulser_list_)
    {
        relative_position = state - repulser;
        distance = relative_position.norm() + 1e-8;

        if (distance < threshold_)
        {
            force -= rep_ * (1/threshold_ - 1/distance) * relative_position / relative_position.norm() / pow(distance, 2);
        }
    }

    return force;
}

Eigen::VectorXd PotentialField::attractiveForce(const Eigen::VectorXd &state) const
{
    Eigen::VectorXd force(state.size());
    force.setZero();

    if (robot_->isGoalDefined())
        force -= att_ * (state - robot_->goal()->getCenter());

    for (Eigen::VectorXd attractor : attractor_list_)
        force -= att_ * (state - attractor);

    return force;
}

void PotentialField::addRepulser(const Eigen::VectorXd &state)
{
    repulser_list_.push_back(state);
}

void PotentialField::addAttractor(const Eigen::VectorXd &state)
{
    attractor_list_.push_back(state);
}
