//
// Created by tipakorng on 7/25/17.
//

#include "models/peg_hole_2d_model.h"
#include "math_utils.h"
#include <iostream>
#include <tf/tf.h>

Rectangle::Rectangle(double dimension1, double dimension2) :
    dimension1_(dimension1),
    dimension2_(dimension2)
{}

Rectangle::~Rectangle()
{}

void Rectangle::corners(double x, double y, double angle, std::vector<Eigen::Vector2d> &corner_list) const
{
    Eigen::Matrix3d g;
    Eigen::Vector3d p;
    Eigen::Vector3d q;
    g.block<2, 2>(0, 0) << std::cos(angle), -std::sin(angle), std::sin(angle), std::cos(angle);
    g.block<2, 1>(0, 2) << x, y;

    // Populating the list of corners in the clockwise direction starting from the top left corner.
    corner_list.clear();
    p << -0.5 * dimension1_, 0.5 * dimension2_, 1;
    q = g * p;
    corner_list.push_back(q.segment<2>(0));
    p << 0.5 * dimension1_, 0.5 * dimension2_, 1;
    q = g * p;
    corner_list.push_back(q.segment<2>(0));
    p << 0.5 * dimension1_, -0.5 * dimension2_, 1;
    q = g * p;
    corner_list.push_back(q.segment<2>(0));
    p << -0.5 * dimension1_, -0.5 * dimension2_, 1;
    q = g * p;
    corner_list.push_back(q.segment<2>(0));
}

void Rectangle::axes(double x, double y, double angle, std::vector<Eigen::Vector2d> &axis_list) const
{
    std::vector<Eigen::Vector2d> corner_list;
    corners(x, y, angle, corner_list);

    Eigen::Matrix2d rot90;
    rot90 << 0, -1, 1, 0;

    axis_list.clear();
    axis_list.push_back(rot90 * (corner_list[0] - corner_list[1]));
    axis_list.push_back(rot90 * (corner_list[1] - corner_list[2]));
}


PegHole2d::PegHole2d(double peg_width, double peg_height, double hole_tolerance, const Eigen::VectorXd &init_mean,
                    const Eigen::MatrixXd &init_cov, const Eigen::MatrixXd &motion_cov,
                    double sensing_cov, double trans_step_size, double rot_step_size, double collision_tol):
    peg_width_(peg_width),
    peg_height_(peg_height),
    peg_(peg_width, peg_height),
    hole_tolerance_(hole_tolerance),
    state_size_(3),
    observation_size_(1),
    trans_step_size_(trans_step_size),
    rot_step_size_(rot_step_size),
    collision_tol_(collision_tol)
{
    // Initialize probability distribution.
    Eigen::MatrixXd sensing_cov_matrix(1, 1);
    sensing_cov_matrix << sensing_cov;
    init_belief_ = new MultivariateGaussian(init_mean, init_cov, 0);
    motion_noise_ = new MultivariateGaussian(Eigen::VectorXd::Zero(3), motion_cov, 0);
    sensing_noise_ = new MultivariateGaussian(Eigen::VectorXd::Zero(1), sensing_cov_matrix, 0);

    // Initialize sensing actions.
    for (unsigned int i = 0; i < this->getStateSize(); i++)
    {
        sensing_actions_.push_back(i);
    }

    // Initialize obstacles (left wall, hole, and right wall).
    // The corners go in the clockwise direction starting from the top left corner.
    Eigen::Vector2d corner;
    corner << -std::numeric_limits<double>::infinity(), 0;
    left_wall_.push_back(corner);
    corner << -0.5 * peg_width_ - hole_tolerance_, 0;
    left_wall_.push_back(corner);
    corner << -0.5 * peg_width_ - hole_tolerance_, -0.5 * peg_height_ - hole_tolerance_;
    left_wall_.push_back(corner);
    corner << -std::numeric_limits<double>::infinity(), -0.5 * peg_height_ - hole_tolerance_;
    left_wall_.push_back(corner);

    corner << -0.5 * peg_width_ - hole_tolerance_, -0.5 * peg_height_ - hole_tolerance_;
    hole_floor_.push_back(corner);
    corner << 0.5 * peg_width_ + hole_tolerance_, -0.5 * peg_height_ - hole_tolerance_;
    hole_floor_.push_back(corner);
    corner << 0.5 * peg_width_ + hole_tolerance_, -peg_height_ - hole_tolerance_;
    hole_floor_.push_back(corner);
    corner << -0.5 * peg_width_ - hole_tolerance_, -peg_height_ - hole_tolerance_;
    hole_floor_.push_back(corner);

    corner << 0.5 * peg_width_ + hole_tolerance_, 0;
    right_wall_.push_back(corner);
    corner << std::numeric_limits<double>::infinity(), 0;
    right_wall_.push_back(corner);
    corner << std::numeric_limits<double>::infinity(), -0.5 * peg_height_ - hole_tolerance_;
    right_wall_.push_back(corner);
    corner << 0.5 * peg_width_ + hole_tolerance_, -0.5 * peg_height_ - hole_tolerance_;
    right_wall_.push_back(corner);
}

PegHole2d::~PegHole2d()
{
    delete init_belief_;
    delete motion_noise_;
    delete sensing_noise_;
}

unsigned int PegHole2d::getStateSize() const
{
    return state_size_;
}

double PegHole2d::getObservationProbability(const Eigen::VectorXd &state, unsigned int sensing_action,
                                          const Eigen::VectorXd &observation) const
{
    Eigen::VectorXd noiseless_observation = getObservation(state, sensing_action);
    return math::gaussianPdf(observation, noiseless_observation, sensing_noise_->cov_);
}

double PegHole2d::getTransitionProbability(const Eigen::VectorXd &next_state, const Eigen::VectorXd &current_state,
                                         const Eigen::VectorXd &task_action) const
{
    Eigen::VectorXd noiseless_next_state = getNextState(current_state, task_action);
    return math::gaussianPdf(next_state, noiseless_next_state, motion_noise_->cov_);
}

double PegHole2d::getReward(const Eigen::VectorXd &state, const Eigen::VectorXd &task_action) const
{
    Eigen::VectorXd normalized_action = normalizeAction(task_action);
    return -normalized_action.norm();
}

Eigen::VectorXd PegHole2d::getInitState() const
{
    Eigen::VectorXd state = init_belief_->mean_;
    return state;
}

Eigen::VectorXd PegHole2d::getObservation(const Eigen::VectorXd &state, unsigned int sensing_action) const
{
    Eigen::VectorXd observation(observation_size_);
    observation << state(sensing_action);
    return observation;
}

Eigen::VectorXd PegHole2d::getNextState(const Eigen::VectorXd &state, const Eigen::VectorXd &task_action) const
{
    Eigen::VectorXd normalized_action = normalizeAction(task_action);
    Eigen::VectorXd next_state = state + normalized_action;

    if (isCollision(next_state))
        next_state = moveToFreeState(state, normalized_action, 0, 1, collision_tol_);

    return next_state;
}

Eigen::VectorXd PegHole2d::sampleInitState() const
{
    Eigen::VectorXd init_state(state_size_);
    init_belief_->dev(init_state);

    while (isCollision(init_state))
        init_belief_->dev(init_state);

    return init_state;
}

Eigen::VectorXd PegHole2d::sampleNextState(const Eigen::VectorXd &state, const Eigen::VectorXd &task_action) const
{
    Eigen::VectorXd noiseless_next_state = getNextState(state, task_action);
    Eigen::VectorXd noise(state_size_);
    motion_noise_->dev(noise);
    Eigen::VectorXd next_state = noiseless_next_state + noise;

    while (isCollision(next_state))
    {
        motion_noise_->dev(noise);
        next_state = noiseless_next_state + noise;
    }

    return next_state;
}

Eigen::VectorXd PegHole2d::sampleObservation(const Eigen::VectorXd &state, unsigned int sensing_action) const
{
    Eigen::VectorXd noise(observation_size_);
    sensing_noise_->dev(noise);
    return getObservation(state, sensing_action) + noise;
}

bool PegHole2d::isTerminal(const Eigen::VectorXd &state) const
{
    return isGoal(state);
}

bool PegHole2d::isGoal(const Eigen::VectorXd &state) const
{
    // Only have to check the position because the angle must be correct before the peg can be inserted.
    return std::abs(state(0)) < hole_tolerance_ && std::abs(state(1)) < hole_tolerance_;
}

void PegHole2d::fillMarker(const Eigen::VectorXd &state, visualization_msgs::Marker &marker) const
{
    tf::Quaternion quaternion;

    marker.type = visualization_msgs::Marker::CUBE;
    marker.action = visualization_msgs::Marker::ADD;

    quaternion.setRPY(0, state(2), 0);

    marker.pose.position.x = state(0);
    marker.pose.position.y = 0;
    marker.pose.position.z = state(1);
    marker.pose.orientation.x = quaternion.getX();
    marker.pose.orientation.y = quaternion.getY();
    marker.pose.orientation.z = quaternion.getZ();
    marker.pose.orientation.w = quaternion.getW();

    marker.scale.x = peg_width_;
    marker.scale.y = 1.0;
    marker.scale.z = peg_height_;
}

void PegHole2d::publishMap()
{}

bool PegHole2d::isCollision(const Eigen::VectorXd &state) const
{
    std::vector<Eigen::Vector2d> peg_corners;
    peg_.corners(state(0), state(1), state(2), peg_corners);

    return !(separatingAxisTheorem(peg_corners, left_wall_) &&
             separatingAxisTheorem(peg_corners, right_wall_) &&
             separatingAxisTheorem(peg_corners, hole_floor_));
}

bool PegHole2d::separatingAxisTheorem(const std::vector<Eigen::Vector2d> &corner_list_1,
                                      const std::vector<Eigen::Vector2d> &corner_list_2) const
{
    // Calculate the axes to project the corners on.
    // Rectangle is a special case of the separating axis theorem where we only have to check 4 axes.
    // The axes are orthogonal to the edges of the rectangles.
    Eigen::Matrix2d rot90;
    rot90 << 0, -1, 1, 0;
    Eigen::Vector2d axis0 = rot90 * (corner_list_1[0] - corner_list_1[1]);
    Eigen::Vector2d axis1 = rot90 * (corner_list_1[1] - corner_list_1[2]);
    Eigen::Vector2d axis2;
    axis2 << 0, 1;
    Eigen::Vector2d axis3;
    axis3 << 1, 0;

    return isSeparated(axis0, corner_list_1, corner_list_2) || isSeparated(axis1, corner_list_1, corner_list_2) ||
           isSeparated(axis2, corner_list_1, corner_list_2) || isSeparated(axis3, corner_list_1, corner_list_2);
}

bool PegHole2d::isSeparated(const Eigen::Vector2d &axis,
                            const std::vector<Eigen::Vector2d> &corner_list_1,
                            const std::vector<Eigen::Vector2d> &corner_list_2) const
{
    double min_1 = std::numeric_limits<double>::infinity();
    double max_1 = -std::numeric_limits<double>::infinity();
    double min_2 = std::numeric_limits<double>::infinity();
    double max_2 = -std::numeric_limits<double>::infinity();
    double temp;

    for (Eigen::Vector2d corner : corner_list_1)
    {
        temp = axis.dot(corner);

        if (temp < min_1)
            min_1 = temp;
        if (temp > max_1)
            max_1 = temp;
    }

    for (Eigen::Vector2d corner : corner_list_2)
    {
        temp = axis.dot(corner);

        if (temp < min_2)
            min_2 = temp;
        if (temp > max_2)
            max_2 = temp;
    }

    return max_1 < min_2 || max_2 < min_1;
}

Eigen::VectorXd PegHole2d::moveToFreeState(const Eigen::VectorXd &state, const Eigen::VectorXd &action,
                                           double lo, double hi, double tol) const
{
    if (isCollision(state))
    {
        std::cout << "state = " << state.transpose() << std::endl;
        throw std::invalid_argument("state is not free!");
    }

    Eigen::VectorXd next_state = state + 0.5 * (lo + hi) * action;

    if (isCollision(next_state))
    {
        hi = 0.5 * (lo + hi);
        return moveToFreeState(state, action, lo, hi, tol);
    }
    else
    {
        if (hi - lo < tol)
        {
            return next_state;
        }
        else
        {
            lo = 0.5 * (lo + hi);
            return moveToFreeState(state, action, lo, hi, tol);
        }
    }
}

Eigen::VectorXd PegHole2d::normalizeAction(const Eigen::VectorXd &task_action) const
{
    Eigen::VectorXd normalized_action(state_size_);
    normalized_action(0) = task_action(0) * std::min(1.0, trans_step_size_ / std::abs(task_action(0)));
    normalized_action(1) = task_action(1) * std::min(1.0, trans_step_size_ / std::abs(task_action(1)));
    normalized_action(2) = task_action(2) * std::min(1.0, rot_step_size_ / std::abs(task_action(2)));
    return normalized_action;
}
