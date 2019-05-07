//
// Created by tipakorng on 8/29/17.
//

#include <vector>
#include <boost/shared_ptr.hpp>
#include <yaml-cpp/yaml.h>

#include <ros/ros.h>
#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>

#include <fcl/narrowphase/narrowphase.h>
#include <fcl/shape/geometric_shape_to_BVH_model.h>
#include <fcl/shape/geometric_shapes_utility.h>
#include <fcl/collision.h>

#include "models/capsule_robot_model.h"
#include "math_utils.h"


CapsuleRobotModel::CapsuleRobotModel(const std::string &file_path) :
    state_size_(2),
    observation_size_(2)
{
    // Initialize yaml node.
    YAML::Node root = YAML::LoadFile(file_path);
    YAML::Node model = root["model"];

    // Initialize the robot.
    robot_radius_ = model["robot_radius"].as<double>();
    std::shared_ptr<fcl::CollisionGeometry> robot_geom(new fcl::Sphere(robot_radius_));
    robot_ = new fcl::CollisionObject(robot_geom);

    // Initialize wall structs and FCL collision objects.
    int num_walls = model["num_walls"].as<int>();
    Eigen::Vector2d wall_end_0;
    Eigen::Vector2d wall_end_1;
    double wall_thickness;
    fcl::Vec3f translation;
    translation.setZero();
    fcl::Matrix3f rotation;
    rotation.setZero();

    for (int i = 0; i < num_walls; i++)
    {
        // Create a wall struct.
        wall_end_0(0) = model["walls"][i]["end_0"][0].as<double>();
        wall_end_0(1) = model["walls"][i]["end_0"][1].as<double>();
        wall_end_1(0) = model["walls"][i]["end_1"][0].as<double>();
        wall_end_1(1) = model["walls"][i]["end_1"][1].as<double>();
        wall_thickness = model["walls"][i]["thickness"].as<double>();
        LineSegment wall(wall_end_0, wall_end_1, wall_thickness);
        wall_line_segments_.push_back(wall);

        // Create an FCL collision object.
        translation.setValue(wall.center(0), wall.center(1), 0);
        rotation.setEulerYPR(wall.angle, 0, 0);
        std::shared_ptr<fcl::CollisionGeometry> wall_geom(new fcl::Box(wall.bounding_box(0),
                                                                         wall.bounding_box(1),
                                                                         0.1));
        walls_.push_back(new fcl::CollisionObject(wall_geom, rotation, translation));
    }

    // Initialize probability density functions.
    Eigen::VectorXd init_mean = Eigen::VectorXd::Zero(state_size_);
    Eigen::MatrixXd init_cov = Eigen::MatrixXd::Zero(state_size_, state_size_);
    Eigen::MatrixXd motion_cov = Eigen::MatrixXd::Zero(state_size_, state_size_);
    Eigen::MatrixXd sensing_cov = Eigen::MatrixXd::Zero(observation_size_, observation_size_);

    for (int i = 0; i < state_size_; i++)
    {
        init_mean(i) = model["init_mean"][i].as<double>();
        init_cov(i, i) = model["init_cov"][i].as<double>();
        motion_cov(i, i) = model["motion_cov"][i].as<double>();
    }

    unsigned long long seed = model["seed"].as<unsigned long long>();
    init_belief_ = new MultivariateGaussian(init_mean, init_cov, seed);
    motion_noise_model_ = new MultivariateGaussian(Eigen::VectorXd::Zero(state_size_), motion_cov, seed);

    // Initialize sensing actions and noise model.
    for (unsigned int i = 0; i < state_size_; i++)
    {
        sensing_actions_.push_back(i);
        sensing_cov(0, 0) = model["sensing_covs"][i][0].as<double>();
        sensing_cov(1, 1) = model["sensing_covs"][i][1].as<double>();
        sensing_noise_models_.push_back(new MultivariateGaussian(Eigen::VectorXd::Zero(observation_size_), sensing_cov,
                                                                 seed));
    }

    // Initialize other stuff.
    collision_tolerance_ = model["collision_tol"].as<double>();

    // Initialize planner parameters.
    step_size_ = root["model"]["step_size"].as<double>();

    // Initialize goal.
    goal_.resize(state_size_, 1);
    goal_(0) = model["goal"]["center"][0].as<double>();
    goal_(1) = model["goal"]["center"][1].as<double>();
    goal_radius_ = model["goal"]["radius"].as<double>();

    has_publisher_ = false;
}

CapsuleRobotModel::CapsuleRobotModel(const std::string &file_path, ros::NodeHandle *node_handle) :
    CapsuleRobotModel(file_path)
{
    node_handle_ = node_handle;
    publisher_ = node_handle_->advertise<visualization_msgs::MarkerArray>("model", 1);
    has_publisher_ = true;
}

CapsuleRobotModel::~CapsuleRobotModel()
{
    delete init_belief_;
    delete motion_noise_model_;
    delete robot_;

    for (int i = 0; i < walls_.size(); i++)
        delete walls_[i];

    for (int i = 0; i < sensing_noise_models_.size(); i++)
        delete sensing_noise_models_[i];
}

unsigned int CapsuleRobotModel::getStateSize() const
{
    return state_size_;
}

double CapsuleRobotModel::getObservationProbability(const Eigen::VectorXd &state, unsigned int sensing_action,
                                                    const Eigen::VectorXd &observation) const
{
    Eigen::VectorXd noiseless_observation = getObservation(state, sensing_action);
    return math::gaussianPdf(observation, noiseless_observation, sensing_noise_models_[sensing_action]->cov_);
}

double CapsuleRobotModel::getTransitionProbability(const Eigen::VectorXd &next_state,
                                                   const Eigen::VectorXd &current_state,
                                                   const Eigen::VectorXd &task_action) const
{
    Eigen::VectorXd noiseless_next_state = getNextState(current_state, task_action);
    return math::gaussianPdf(next_state, current_state, motion_noise_model_->cov_);
}

double CapsuleRobotModel::getReward(const Eigen::VectorXd &state, const Eigen::VectorXd &task_action) const
{
    Eigen::VectorXd normalized_action = normalizeAction(task_action);
    Eigen::VectorXd next_state = state + normalized_action;

    if (isCollision(next_state))
        return -1;

    else
        return 0;
}

Eigen::VectorXd CapsuleRobotModel::getInitState() const
{
    Eigen::VectorXd state = init_belief_->mean_;
    return state;
}

Eigen::VectorXd CapsuleRobotModel::getObservation(const Eigen::VectorXd &state, unsigned int sensing_action) const
{
    Eigen::VectorXd observation(observation_size_);
    observation = state;
    return observation;
}

Eigen::VectorXd CapsuleRobotModel::getNextState(const Eigen::VectorXd &state, const Eigen::VectorXd &task_action) const
{
    Eigen::VectorXd normalized_action = normalizeAction(task_action);
    Eigen::VectorXd next_state = state + normalized_action;

    while (isCollision(next_state))
        next_state = moveToFreeState(state, normalized_action, 0, 1, collision_tolerance_);

    return next_state;
}

Eigen::VectorXd CapsuleRobotModel::sampleInitState() const
{
    Eigen::VectorXd init_state(state_size_);
    init_belief_->dev(init_state);

    while (isCollision(init_state))
        init_belief_->dev(init_state);

    return init_state;
}

Eigen::VectorXd CapsuleRobotModel::sampleNextState(const Eigen::VectorXd &state,
                                                   const Eigen::VectorXd &task_action) const
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

Eigen::VectorXd CapsuleRobotModel::sampleObservation(const Eigen::VectorXd &state, unsigned int sensing_action) const
{
    Eigen::VectorXd noise(observation_size_);
    sensing_noise_models_[sensing_action]->dev(noise);
    return getObservation(state, sensing_action) + noise;
}

bool CapsuleRobotModel::isTerminal(const Eigen::VectorXd &state) const
{
    return isGoal(state);
}

bool CapsuleRobotModel::isGoal(const Eigen::VectorXd &state) const
{
    return (goal_ - state).norm() < goal_radius_;
}

void CapsuleRobotModel::fillMarker(const Eigen::VectorXd &state, visualization_msgs::Marker &marker) const
{
    marker.type = visualization_msgs::Marker::SPHERE;
    marker.action = visualization_msgs::Marker::ADD;

    marker.pose.orientation.x = 0;
    marker.pose.orientation.y = 0;
    marker.pose.orientation.z = 0;
    marker.pose.orientation.w = 1;

    marker.pose.position.x = state(0);
    marker.pose.position.y = state(1);
    marker.pose.position.z = 0;

    marker.scale.x = 2 * robot_radius_;
    marker.scale.y = 2 * robot_radius_;
    marker.scale.z = 2 * robot_radius_;
}

void CapsuleRobotModel::publishMap()
{
    if (has_publisher_)
    {
        tf::Transform transform;
        transform.setOrigin(tf::Vector3(0, 0, 0));
        tf::Quaternion q;
        q.setRPY(0, 0, 0);
        transform.setRotation(q);
        tf_broadcaster_.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "world", "my_frame"));

        visualization_msgs::MarkerArray marker_array;
        visualization_msgs::Marker marker;

        unsigned int marker_id = 0;

        // Set markers for the walls.
        for (int i = 0; i < wall_line_segments_.size(); i++)
        {
            marker.header.frame_id = "world";
            marker.header.stamp = ros::Time::now();
            marker.ns = "basic_shapes";
            marker.id = marker_id++;
            marker.type = visualization_msgs::Marker::CUBE;
            marker.action = visualization_msgs::Marker::ADD;

            tf::Quaternion quaternion;
            quaternion.setRPY(0, 0, wall_line_segments_[i].angle);
            marker.pose.orientation.x = quaternion.getX();
            marker.pose.orientation.y = quaternion.getY();
            marker.pose.orientation.z = quaternion.getZ();
            marker.pose.orientation.w = quaternion.getW();

            marker.pose.position.x = wall_line_segments_[i].center(0);
            marker.pose.position.y = wall_line_segments_[i].center(1);
            marker.pose.position.z = 0;

            marker.scale.x = wall_line_segments_[i].bounding_box(0);
            marker.scale.y = wall_line_segments_[i].bounding_box(1);
            marker.scale.z = 0.1;

            marker.color.r = 1.0;
            marker.color.g = 0.0;
            marker.color.b = 0.0;
            marker.color.a = 1.0;

            marker.lifetime = ros::Duration();

            marker_array.markers.push_back(marker);
        }

        publisher_.publish(marker_array);
    }
}

bool CapsuleRobotModel::isCollision(const Eigen::VectorXd &state) const
{
    fcl::Vec3f translation;
    fcl::Matrix3f rotation;
    translation.setValue(state(0), state(1), 0);
    rotation.setEulerYPR(0, 0, 0);
    robot_->setTransform(rotation, translation);
    fcl::CollisionRequest request;
    fcl::CollisionResult result;

    for (fcl::CollisionObject *object : walls_)
    {
        if (fcl::collide(robot_, object, request, result) > 0)
            return true;
    }

    return false;
}

void CapsuleRobotModel::printParameters(std::ofstream &file) const
{
    file << "Model Parameters" << std::endl;
    file << "state_size = " << state_size_ << std::endl;
    file << "observation_size = " << observation_size_ << std::endl;
    file << "collision_tolerance = " << collision_tolerance_ << std::endl;
    file << "has_publisher = " << has_publisher_ << std::endl;
    file << "goal_center = " << goal_.transpose() << std::endl;
    file << "goal_radius = " << goal_radius_ << std::endl;
    file << "step_size = " << step_size_ << std::endl;
    file << "robot_radius = " << robot_radius_ << std::endl;
    file << "init_mean = " << init_belief_->mean_.transpose() << std::endl;
    file << "init_cov = " << init_belief_->cov_.diagonal().transpose() << std::endl;
    file << "motion_noise_cov = " << motion_noise_model_->cov_.diagonal().transpose() << std::endl;

    for (int i = 0; i < sensing_actions_.size(); i++)
        file << "observation_noise_cov = " << sensing_noise_models_[i]->cov_.diagonal().transpose() << std::endl;

    for (LineSegment wall : wall_line_segments_)
    {
        file << "wall_end_0 = " << wall.end_0.transpose() << std::endl;
        file << "wall_end_1 = " << wall.end_1.transpose() << std::endl;
        file << "wall_center = " << wall.center.transpose() << std::endl;
        file << "wall_normal = " << wall.normal.transpose() << std::endl;
        file << "wall_bounding_box = " << wall.bounding_box.transpose() << std::endl;
        file << "thickness = " << wall.thickness << std::endl;
        file << "length = " << wall.length << std::endl;
        file << "angle = " << wall.angle << std::endl;
    }
}

Eigen::VectorXd CapsuleRobotModel::moveToFreeState(const Eigen::VectorXd &state, const Eigen::VectorXd &action,
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

Eigen::VectorXd CapsuleRobotModel::normalizeAction(const Eigen::VectorXd &action) const
{
    Eigen::VectorXd normalized_action = action * std::min(1.0, step_size_ / action.norm());
    return normalized_action;
}
