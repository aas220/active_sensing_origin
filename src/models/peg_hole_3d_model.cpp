//
// Created by tipakorng on 7/31/17.
//

#include "models/peg_hole_3d_model.h"
#include "math_utils.h"
#include <boost/shared_ptr.hpp>  //use std::shared_ptr here?
#include <tf/tf.h>

PegHole3d::PegHole3d(double peg_dim_1, double peg_dim_2, double peg_dim_3,
                     double map_dim_1, double map_dim_2, double map_dim_3, double hole_tolerance,
                     const Eigen::VectorXd &init_mean, const Eigen::MatrixXd &init_cov,
                     const Eigen::MatrixXd &motion_cov, double sensing_cov, unsigned long long seed):
    peg_dim_1_(peg_dim_1),
    peg_dim_2_(peg_dim_2),
    peg_dim_3_(peg_dim_3),
    hole_tolerance_(hole_tolerance),
    collision_tolerance_(1e-2),
    state_size_(6),
    observation_size_(1)
{
    // Initialize probability density functions.
    Eigen::MatrixXd sensing_cov_matrix(observation_size_, observation_size_);
    sensing_cov_matrix << sensing_cov;
    init_belief_ = new MultivariateGaussian(init_mean, init_cov, seed);
    motion_noise_model_ = new MultivariateGaussian(Eigen::VectorXd::Zero(state_size_), motion_cov, seed);
    sensing_noise_model_ = new MultivariateGaussian(Eigen::VectorXd::Zero(observation_size_), sensing_cov_matrix, seed);

    // Initialize sensing actions.
    for (unsigned int i = 0; i < state_size_; i++)
        sensing_actions_.push_back(i);

    // Initialize peg.
    std::shared_ptr<fcl::CollisionGeometry> peg_geometry_ptr(new fcl::Box(peg_dim_1, peg_dim_2, peg_dim_3));
    peg_ = new fcl::CollisionObject(peg_geometry_ptr);

    // Initialize hole.
    double a = 0.5 * map_dim_1 - (0.5 * peg_dim_1 + hole_tolerance);
    double b = 0.5 * map_dim_2 + (0.5 * peg_dim_2 + hole_tolerance);
    double c = 0.5 * map_dim_1 + (0.5 * peg_dim_1 + hole_tolerance);
    double d = 0.5 * map_dim_2 - (0.5 * peg_dim_2 + hole_tolerance);
    double h = 0.5 * peg_dim_3 + hole_tolerance;
    fcl::Vec3f translation;
    fcl::Matrix3f rotation;
    rotation.setIdentity();
    std::shared_ptr<fcl::CollisionGeometry> hole_1(new fcl::Box(a, b, h));
    std::shared_ptr<fcl::CollisionGeometry> hole_2(new fcl::Box(c, d, h));
    std::shared_ptr<fcl::CollisionGeometry> hole_3(new fcl::Box(a, b, h));
    std::shared_ptr<fcl::CollisionGeometry> hole_4(new fcl::Box(c, d, h));
    translation[2] = -0.5 * h;
    translation[0] = -(0.5 * peg_dim_1 + hole_tolerance) - 0.5 * a;
    translation[1] = -(0.5 * peg_dim_2 + hole_tolerance) + 0.5 * b;
    hole_sides_.push_back(new fcl::CollisionObject(hole_1, rotation, translation));
    translation[0] = -(0.5 * peg_dim_1 + hole_tolerance) + 0.5 * c;
    translation[1] =  (0.5 * peg_dim_2 + hole_tolerance) + 0.5 * d;
    hole_sides_.push_back(new fcl::CollisionObject(hole_2, rotation, translation));
    translation[0] =  (0.5 * peg_dim_1 + hole_tolerance) + 0.5 * a;
    translation[1] =  (0.5 * peg_dim_2 + hole_tolerance) - 0.5 * b;
    hole_sides_.push_back(new fcl::CollisionObject(hole_3, rotation, translation));
    translation[0] =  (0.5 * peg_dim_1 + hole_tolerance) - 0.5 * c;
    translation[1] = -(0.5 * peg_dim_2 + hole_tolerance) - 0.5 * d;
    hole_sides_.push_back(new fcl::CollisionObject(hole_4, rotation, translation));

    // Initialize world edges.
    std::shared_ptr<fcl::CollisionGeometry> wall_1l(new fcl::Plane(1, 0, 0, -map_dim_1));
    std::shared_ptr<fcl::CollisionGeometry> wall_1h(new fcl::Plane(-1, 0, 0, -map_dim_1));
    std::shared_ptr<fcl::CollisionGeometry> wall_2l(new fcl::Plane(0, 1, 0, -map_dim_2));
    std::shared_ptr<fcl::CollisionGeometry> wall_2h(new fcl::Plane(0, -1, 0, -map_dim_2));
    std::shared_ptr<fcl::CollisionGeometry> wall_3l(new fcl::Plane(0, 0, 1, -0.5*peg_dim_3-hole_tolerance));
    std::shared_ptr<fcl::CollisionGeometry> wall_3h(new fcl::Plane(0, 0, -1, -map_dim_3));
    walls_.push_back(new fcl::CollisionObject(wall_1l));
    walls_.push_back(new fcl::CollisionObject(wall_1h));
    walls_.push_back(new fcl::CollisionObject(wall_2l));
    walls_.push_back(new fcl::CollisionObject(wall_2h));
    walls_.push_back(new fcl::CollisionObject(wall_3l));
    walls_.push_back(new fcl::CollisionObject(wall_3h));
}

PegHole3d::~PegHole3d()
{
    delete init_belief_;
    delete motion_noise_model_;
    delete sensing_noise_model_;
    delete peg_;

    for (fcl::CollisionObject *object : hole_sides_)
        delete object;

    for (fcl::CollisionObject *object : walls_)
        delete object;
}

unsigned int PegHole3d::getStateSize() const
{
    return state_size_;
}

double PegHole3d::getObservationProbability(const Eigen::VectorXd &state, unsigned int sensing_action,
                                            const Eigen::VectorXd &observation) const
{
    Eigen::VectorXd noiseless_observation = getObservation(state, sensing_action);
    return math::gaussianPdf(observation, noiseless_observation, sensing_noise_model_->cov_);
}

double PegHole3d::getTransitionProbability(const Eigen::VectorXd &next_state, const Eigen::VectorXd &current_state,
                                           const Eigen::VectorXd &task_action) const
{
    Eigen::VectorXd noiseless_next_state = getNextState(current_state, task_action);
    return math::gaussianPdf(next_state, noiseless_next_state, motion_noise_model_->cov_);
}

double PegHole3d::getReward(const Eigen::VectorXd &state, const Eigen::VectorXd &task_action) const
{
    return -task_action.norm();
}

Eigen::VectorXd PegHole3d::getInitState() const
{
    // TODO: Should check if the init_mean is collision free in the constructor.
    Eigen::VectorXd state = init_belief_->mean_;
    return state;
}

Eigen::VectorXd PegHole3d::getObservation(const Eigen::VectorXd &state, unsigned int sensing_action) const
{
    Eigen::VectorXd observation(observation_size_);
    observation(0) = state(sensing_action);
    return observation;
}

Eigen::VectorXd PegHole3d::getNextState(const Eigen::VectorXd &state, const Eigen::VectorXd &task_action) const
{
    Eigen::VectorXd next_state = state + task_action;

    while (isCollision(next_state))
        next_state = moveToFreeState(state, task_action, 0, 1, collision_tolerance_);

    return next_state;
}

Eigen::VectorXd PegHole3d::sampleInitState() const
{
    Eigen::VectorXd init_state(state_size_);
    init_belief_->dev(init_state);

    while (isCollision(init_state))
        init_belief_->dev(init_state);

    return init_state;
}

Eigen::VectorXd PegHole3d::sampleNextState(const Eigen::VectorXd &state, const Eigen::VectorXd &task_action) const
{
    Eigen::VectorXd noiseless_next_state = getNextState(state, task_action);
    Eigen::VectorXd noise(state_size_);
    motion_noise_model_->dev(noise);
    Eigen::VectorXd next_state = noiseless_next_state + noise;

    while (isCollision(next_state))
    {
        motion_noise_model_->dev(noise);
        next_state = noiseless_next_state + noise;
    }

    return next_state;
}

Eigen::VectorXd PegHole3d::sampleObservation(const Eigen::VectorXd &state, unsigned int sensing_action) const
{
    Eigen::VectorXd noise(observation_size_);
    sensing_noise_model_->dev(noise);
    return getObservation(state, sensing_action) + noise;
}

bool PegHole3d::isTerminal(const Eigen::VectorXd &state) const
{
    return isGoal(state);
}

bool PegHole3d::isGoal(const Eigen::VectorXd &state) const
{
    // Only have to check the position because the angle must be correct before the peg can be inserted.
    return std::abs(state(0)) < hole_tolerance_ &&
           std::abs(state(1)) < hole_tolerance_ &&
           std::abs(state(2)) < hole_tolerance_;
}

void PegHole3d::fillMarker(const Eigen::VectorXd &state, visualization_msgs::Marker &marker) const
{
    tf::Quaternion quaternion;

    marker.type = visualization_msgs::Marker::CUBE;
    marker.action = visualization_msgs::Marker::ADD;

    quaternion.setRPY(state(5), state(4), state(3));

    marker.pose.position.x = state(0);
    marker.pose.position.y = state(1);
    marker.pose.position.z = state(2);
    marker.pose.orientation.x = quaternion.getX();
    marker.pose.orientation.y = quaternion.getY();
    marker.pose.orientation.z = quaternion.getZ();
    marker.pose.orientation.w = quaternion.getW();

    marker.scale.x = peg_dim_1_;
    marker.scale.y = peg_dim_2_;
    marker.scale.z = peg_dim_3_;
}

void PegHole3d::publishMap()
{}

bool PegHole3d::isCollision(const Eigen::VectorXd &state) const
{
    fcl::Vec3f translation;
    fcl::Matrix3f rotation;
    translation.setValue(state(0), state(1), state(2));
    rotation.setEulerYPR(state(3), state(4), state(5));
    peg_->setTransform(rotation, translation);
    fcl::CollisionRequest request;
    fcl::CollisionResult result;

    for (fcl::CollisionObject *object : hole_sides_)
    {
        if (fcl::collide(peg_, object, request, result) > 0)
            return true;
    }

    for (fcl::CollisionObject *object : walls_)
    {
        if (fcl::collide(peg_, object, request, result) > 0)
            return true;
    }

    return false;
}

Eigen::VectorXd PegHole3d::moveToFreeState(const Eigen::VectorXd &state, const Eigen::VectorXd &action,
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
