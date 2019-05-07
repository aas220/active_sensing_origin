//
// Created by tipakorng on 12/4/15.
//
#include <iostream>
#include "models/abstract_robot_model.h"
#include "math_utils.h"

Hypersphere::Hypersphere(unsigned int dimensions) :
    dimensions_(dimensions), radius_(0), center_(Eigen::VectorXd::Zero(dimensions)) {}

Hypersphere::Hypersphere(unsigned int dimensions, double radius, Eigen::VectorXd center) :
    dimensions_(dimensions), radius_(radius), center_(center) {}

Hypersphere::~Hypersphere() {}

double Hypersphere::distanceCenter(const Eigen::VectorXd &state) const
{
    return (center_-state).norm();
}

double Hypersphere::distanceClosestPoint(const Eigen::VectorXd &state) const
{
    return distanceCenter(state) - radius_;
}

Eigen::VectorXd Hypersphere::getClosestPoint(const Eigen::VectorXd &state) const
{
    return center_ + radius_ * (state - center_) / distanceCenter(state);
}

void Hypersphere::setCenter(const Eigen::VectorXd &center)
{
    center_ = center;
}

void Hypersphere::setRadius(double radius)
{
    radius_ = radius;
}

Eigen::VectorXd Hypersphere::getCenter() const
{
    return center_;
}

double Hypersphere::getRadius() const
{
    return radius_;
}

bool Hypersphere::contains(const Eigen::VectorXd &state) const
{

    if (distanceCenter(state) < radius_)
        return true;

    else
        return false;

}

bool Hypersphere::intersects(const Eigen::VectorXd &x1, Eigen::VectorXd &x2) const
{
    double distance = (x2-x1).norm();

    // If x1 and x2 are the same point and x1 (or x2) is on the hypersphere, return true.
    if (distance < std::numeric_limits<double>::epsilon())
    {
        if (distanceCenter(x1) == radius_)
            return true;
        else
            return false;
    }

    Eigen::VectorXd direction = (x2 - x1) / distance;

    double f = pow(direction.dot(x1-center_), 2) - (x1-center_).squaredNorm() + pow(radius_, 2);

    if (f < 0)
    {
        return false;
    }
    else
    {
        double d = -direction.dot(x1-center_) - sqrt(f);

        if (d > distance || d < 0)
        {
            return false;
        }
        else
        {
            x2 = x1 + d * direction;
            return true;
        }
    }
}


AbstractRobot::AbstractRobot(const Eigen::VectorXd &init_mean, const Eigen::MatrixXd &init_cov,
                         const Eigen::MatrixXd &action_cov, const Eigen::MatrixXd &sensing_cov):
    Model(),
    state_size_(static_cast<unsigned int>(init_mean.size())),
    task_action_size_(static_cast<unsigned int>(action_cov.rows())),
    observation_size_(static_cast<unsigned int>(sensing_cov.rows())),
    boundary_min_(Eigen::VectorXd::Constant(state_size_, 1, -std::numeric_limits<double>::infinity())),
    boundary_max_(Eigen::VectorXd::Constant(state_size_, 1, std::numeric_limits<double>::infinity())),
    action_cov_(action_cov),
    sensing_cov_(sensing_cov),
    goal_(state_size_)
{
    is_goal_defined_ = false;
    unsigned long long seed = 0;
    Eigen::VectorXd action_mean = Eigen::VectorXd::Zero(action_cov.rows());  // Action noise has zero mean
    Eigen::VectorXd sensing_mean = Eigen::VectorXd::Zero(sensing_cov.rows());  // Sensing noise has zero mean
    init_belief_ = new MultivariateGaussian(init_mean, init_cov, seed);
    motion_model_ = new MultivariateGaussian(action_mean, action_cov, seed);
    sensing_model_ = new MultivariateGaussian(sensing_mean, sensing_cov, seed);
    initSensingActions();
}

AbstractRobot::~AbstractRobot()
{
    delete init_belief_;
    delete motion_model_;
    delete sensing_model_;
}

unsigned int AbstractRobot::getStateSize() const
{
    return state_size_;
}

void AbstractRobot::initSensingActions()
{

    for (unsigned int i = 0; i < this->getStateSize(); i++)
    {
        sensing_actions_.push_back(i);
    }

}

bool AbstractRobot::isGoal(const Eigen::VectorXd &state) const
{

    if (is_goal_defined_ && goal_.distanceClosestPoint(state) < 0)
        return true;

    else
        return false;
}

void AbstractRobot::fillMarker(const Eigen::VectorXd &state, visualization_msgs::Marker &marker) const
{
    marker.type = visualization_msgs::Marker::SPHERE;
    marker.action = visualization_msgs::Marker::ADD;

    marker.pose.position.x = state(0);

    if (state_size_ > 1)
        marker.pose.position.y = state(1);
    else
        marker.pose.position.y = 0;

    if (state_size_ > 2)
        marker.pose.position.z = state(2);
    else
        marker.pose.position.z = 0;

    marker.pose.orientation.x = 0;
    marker.pose.orientation.y = 0;
    marker.pose.orientation.z = 0;
    marker.pose.orientation.w = 1;

    marker.scale.x = 0.1;
    marker.scale.y = 0.1;
    marker.scale.z = 0.1;
}

void AbstractRobot::publishMap()
{}

bool AbstractRobot::isGoalDefined() const
{
    return is_goal_defined_;
}

bool AbstractRobot::isFreeState(const Eigen::VectorXd &state) const
{
    for (Hypersphere obstacle: obstacle_list_)
        if (obstacle.distanceCenter(state) < obstacle.getRadius())
            return false;

    return true;

}

double AbstractRobot::getObservationProbability(const Eigen::VectorXd &state, unsigned int sensing_action,
                                              const Eigen::VectorXd &observation) const
{
    // TODO: Change back when done debugging.
    Eigen::VectorXd noiseless_observation = getObservation(state, sensing_action);
    double p = math::gaussianPdf(observation, noiseless_observation, sensing_cov_);
    return p;
}

double AbstractRobot::getTransitionProbability(const Eigen::VectorXd &nextState, const Eigen::VectorXd &current_state,
                                             const Eigen::VectorXd &task_action) const
{
    return math::gaussianPdf(nextState, getNextState(current_state, task_action), action_cov_);
}

double AbstractRobot::getReward(const Eigen::VectorXd &state, const Eigen::VectorXd &task_action) const
{
    return -task_action.norm();
}

Eigen::VectorXd AbstractRobot::getInitState() const
{
    return init_belief_->mean_;
}

Eigen::VectorXd AbstractRobot::getObservation(const Eigen::VectorXd &state, unsigned int sensing_action) const
{
    Eigen::VectorXd observation(observation_size_);
    unsigned int sensing_start = sensing_action * observation_size_;
    observation << state.segment(sensing_start, observation_size_);
    return observation;
}

Eigen::VectorXd AbstractRobot::getNextState(const Eigen::VectorXd &state, const Eigen::VectorXd &task_action) const
{
    Eigen::VectorXd next_state(state.size());
    next_state = state + task_action;
    moveInBoundary(next_state);

    // Check for collision. Assume no overlapping obstacles.
    for (Hypersphere obstacle: obstacle_list_)
    {
        if (obstacle.intersects(state, next_state))
        {
            return next_state;
        }
    }

    return next_state;
}

Eigen::VectorXd AbstractRobot::sampleNextState(const Eigen::VectorXd &state, const Eigen::VectorXd &task_action) const
{
    Eigen::VectorXd motion_noise(task_action_size_);
    motion_model_->dev(motion_noise);
    Eigen::VectorXd next_state = state + task_action + motion_noise;
    moveInBoundary(next_state);

    // Check for collision. Assume no overlapping obstacles.
    for (Hypersphere obstacle: obstacle_list_)
    {
        if (obstacle.intersects(state, next_state))
        {
            return next_state;
        }
    }

    return next_state;
}

Eigen::VectorXd AbstractRobot::sampleInitState() const
{
    Eigen::VectorXd sample(state_size_);
    init_belief_->dev(sample);
    moveInBoundary(sample);
    return sample;
}

Eigen::VectorXd AbstractRobot::sampleObservation(const Eigen::VectorXd &state, unsigned int sensing_action) const
{
    Eigen::VectorXd sample(observation_size_);
    sensing_model_->dev(sample);
    return getObservation(state, sensing_action) + sample;
}

bool AbstractRobot::isTerminal(const Eigen::VectorXd &state) const
{
    return isGoal(state);
}

Eigen::VectorXd AbstractRobot::getObstacleCenter(int obstacle_id) const
{
    return obstacle_list_[obstacle_id].getCenter();
}

double AbstractRobot::getObstacleRadius(int obstacle_id) const
{
    return obstacle_list_[obstacle_id].getRadius();
}

unsigned int AbstractRobot::getNumObstacles() const
{
    return obstacle_list_.size();
}

const Hypersphere* AbstractRobot::obstacle(unsigned int i) const
{
    return &obstacle_list_[i];
}

Eigen::VectorXd AbstractRobot::getGoalCenter() const
{
    return goal_.getCenter();
}

double AbstractRobot::getGoalRadius() const
{
    return goal_.getRadius();
}

const Hypersphere* AbstractRobot::goal() const
{
    return &goal_;
}

unsigned int AbstractRobot::getTaskActionSize() const
{
    return task_action_size_;
}

double AbstractRobot::getProbability(const Eigen::VectorXd &x, const Eigen::VectorXd &mean,
                                   const Eigen::MatrixXd &cov) const
{
    double n = (x-mean).transpose() * cov.inverse() * (x-mean);
    return exp(-0.5 * n) / sqrt(pow(2 * M_PI, cov.rows()) * cov.determinant());
}

void AbstractRobot::addObstacle(const Hypersphere &obstacle)
{
    obstacle_list_.push_back(obstacle);
}

void AbstractRobot::setGoal(const Hypersphere &goal)
{
    goal_ = goal;
    is_goal_defined_ = true;
}

void AbstractRobot::setBoundary(const Eigen::VectorXd &min, const Eigen::VectorXd &max)
{
    assert(min.size() == state_size_);
    assert(max.size() == state_size_);
    
    for (int i = 0; i < state_size_; i++)
    {
        assert(min(i) < max(i));
    }
    
    boundary_min_ = min;
    boundary_max_ = max;
}

void AbstractRobot::moveInBoundary(Eigen::VectorXd &state) const
{
    for (int i = 0; i < state_size_; i++)
    {
        state(i) = std::max(state(i), boundary_min_(i));
        state(i) = std::min(state(i), boundary_max_(i));
    }
}
