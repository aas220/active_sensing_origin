//
// Created by tipakorng on 8/18/17.
//

#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>

#include <ros/ros.h>
#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
#include <visualization_msgs/MarkerArray.h>


#include "models/multitier_peg_hole_2d_model.h"
#include "math_utils.h"
#include "fcl/narrowphase/narrowphase.h"
#include "fcl/shape/geometric_shape_to_BVH_model.h"
#include "fcl/shape/geometric_shapes_utility.h"
#include "fcl/collision.h"


bool Hole::isInside(double x, double y) const
{
    return std::abs(x - center[0]) < tol && y < center[1];
}


MultitierPegHole2dModel::MultitierPegHole2dModel(const std::string &file_path) :
    state_size_(3),
    observation_size_(1)
{
    // Initialize the yml node.
    YAML::Node root = YAML::LoadFile(file_path);
    YAML::Node model = root["model"];

    // Initialize the peg.
    std::vector<double> peg_dim = model["peg_dim"].as<std::vector<double> >();
    peg_dim_x_ = peg_dim[0];
    peg_dim_y_ = peg_dim[1];
    std::shared_ptr<fcl::CollisionGeometry> peg_geometry_ptr(new fcl::Box(peg_dim_x_, peg_dim_y_, 1.0));
    peg_ = new fcl::CollisionObject(peg_geometry_ptr);


    // Initialize world edges.
    map_dim_ = model["map_dim"].as<std::vector<double> >();
    std::shared_ptr<fcl::CollisionGeometry> wall_x_min(new fcl::Plane(1, 0, 0, map_dim_[0]));
    std::shared_ptr<fcl::CollisionGeometry> wall_y_min(new fcl::Plane(0, 1, 0, map_dim_[1]));
    std::shared_ptr<fcl::CollisionGeometry> wall_x_max(new fcl::Plane(-1, 0, 0, map_dim_[2]));
    std::shared_ptr<fcl::CollisionGeometry> wall_y_max(new fcl::Plane(0, -1, 0, map_dim_[3]));
    walls_.push_back(new fcl::CollisionObject(wall_x_min));
    walls_.push_back(new fcl::CollisionObject(wall_y_min));
    walls_.push_back(new fcl::CollisionObject(wall_x_max));
    walls_.push_back(new fcl::CollisionObject(wall_y_max));

    // Initialize holes
    int num_holes = static_cast<int>(model["holes"].size());
    fcl::Vec3f translation;
    fcl::Matrix3f rotation;
    rotation.setIdentity();

    for (int i = 0; i < num_holes; i++)
    {
        Hole hole_struct;
        hole_struct.center[0] = model["holes"][i]["center"][0].as<double>();
        hole_struct.center[1] = model["holes"][i]["center"][1].as<double>();
        hole_struct.tol = model["holes"][i]["tol"].as<double>();
        hole_struct.depth = model["holes"][i]["depth"].as<double>();
        hole_struct.edges[0] = map_dim_[0];
        hole_struct.edges[1] = hole_struct.center[0] - 0.5 * peg_dim[0] - hole_struct.tol;
        hole_struct.edges[2] = hole_struct.center[0] + 0.5 * peg_dim[0] + hole_struct.tol;
        hole_struct.edges[3] = map_dim_[2];
        holes_.push_back(hole_struct);

        // Left side of the hole.
        translation[0] = 0.5 * (hole_struct.edges[0] + hole_struct.edges[1]);
        translation[1] = hole_struct.center[1];
        translation[2] = 0;
        std::shared_ptr<fcl::CollisionGeometry> hole_geom_0(new fcl::Box(hole_struct.edges[1] - hole_struct.edges[0], hole_struct.depth, 1));
        hole_sides_.push_back(new fcl::CollisionObject(hole_geom_0, rotation, translation));

        // Right side of the hole.
        translation[0] = 0.5 * (hole_struct.edges[2] + hole_struct.edges[3]);
        translation[1] = hole_struct.center[1];
        translation[2] = 0;
        std::shared_ptr<fcl::CollisionGeometry> hole_geom_1(new fcl::Box(hole_struct.edges[3] - hole_struct.edges[2], hole_struct.depth, 1));
        hole_sides_.push_back(new fcl::CollisionObject(hole_geom_1, rotation, translation));
    }

    // Initialize probability density functions.
    Eigen::VectorXd init_mean = Eigen::VectorXd::Zero(state_size_);
    Eigen::MatrixXd init_cov = Eigen::MatrixXd::Zero(state_size_, state_size_);
    Eigen::MatrixXd motion_cov = Eigen::MatrixXd::Zero(state_size_, state_size_);
    Eigen::MatrixXd sensing_cov(observation_size_, observation_size_);

    for (int i = 0; i < state_size_; i++)
    {
        init_mean(i) = model["init_mean"][i].as<double>();
        init_cov(i, i) = model["init_cov"][i].as<double>();
        motion_cov(i, i) = model["motion_cov"][i].as<double>();
    }

    sensing_cov(0, 0) = model["sensing_cov"].as<double>();
    init_belief_ = new MultivariateGaussian(init_mean, init_cov, model["init_seed"].as<unsigned long long>());
    motion_noise_model_ = new MultivariateGaussian(Eigen::VectorXd::Zero(state_size_), motion_cov,
                                                   model["motion_seed"].as<unsigned long long>());
    sensing_noise_model_ = new MultivariateGaussian(Eigen::VectorXd::Zero(observation_size_), sensing_cov,
                                                    model["sensing_seed"].as<unsigned long long>());

    // Initialize sensing actions.
    for (unsigned int i = 0; i < state_size_; i++)
        sensing_actions_.push_back(i);

    // Initialize other stuff.
    collision_tolerance_ = model["collision_tol"].as<double>();

    // Initialize planner parameters.
    trans_step_size_ = root["state_space_planner"]["translation_step_size"].as<double>();
    rot_step_size_ = root["state_space_planner"]["rotation_step_size"].as<double>();
}

MultitierPegHole2dModel::MultitierPegHole2dModel(const std::string &file_path, ros::NodeHandle *node_handle) :
    MultitierPegHole2dModel(file_path)
{
    node_handle_ = node_handle;
    publisher_ = node_handle_->advertise<visualization_msgs::MarkerArray>("model", 1);
    has_publisher_ = true;
}

MultitierPegHole2dModel::~MultitierPegHole2dModel()
{}

unsigned int MultitierPegHole2dModel::getStateSize() const
{
    return state_size_;
}

double MultitierPegHole2dModel::getObservationProbability(const Eigen::VectorXd &state, unsigned int sensing_action,
                                                          const Eigen::VectorXd &observation) const
{
    Eigen::VectorXd noiseless_observation = getObservation(state, sensing_action);
    return math::gaussianPdf(observation, noiseless_observation, sensing_noise_model_->cov_);
}

double MultitierPegHole2dModel::getTransitionProbability(const Eigen::VectorXd &next_state,
                                                         const Eigen::VectorXd &current_state,
                                                         const Eigen::VectorXd &task_action) const
{
    Eigen::VectorXd noiseless_next_state = getNextState(current_state, task_action);
    return math::gaussianPdf(next_state, current_state, motion_noise_model_->cov_);
}

double MultitierPegHole2dModel::getReward(const Eigen::VectorXd &state, const Eigen::VectorXd &task_action) const
{
    Eigen::VectorXd normalized_action = normalizeAction(task_action);
    return -normalized_action.norm();
}

Eigen::VectorXd MultitierPegHole2dModel::getInitState() const
{
    Eigen::VectorXd state = init_belief_->mean_;
    return state;
}

Eigen::VectorXd MultitierPegHole2dModel::getObservation(const Eigen::VectorXd &state, unsigned int sensing_action) const
{
    Eigen::VectorXd observation(observation_size_);
    observation(0) = state(sensing_action);
    return observation;
}

Eigen::VectorXd MultitierPegHole2dModel::getNextState(const Eigen::VectorXd &state,
                                                      const Eigen::VectorXd &task_action) const
{
    Eigen::VectorXd normalized_action = normalizeAction(task_action);
    Eigen::VectorXd next_state = state + normalized_action;

    while (isCollision(next_state))
        next_state = moveToFreeState(state, normalized_action, 0, 1, collision_tolerance_);

    return next_state;
}

Eigen::VectorXd MultitierPegHole2dModel::sampleInitState() const
{
    Eigen::VectorXd init_state(state_size_);
    init_belief_->dev(init_state);

    while (isCollision(init_state))
        init_belief_->dev(init_state);

    return init_state;
}

Eigen::VectorXd MultitierPegHole2dModel::sampleNextState(const Eigen::VectorXd &state,
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

Eigen::VectorXd MultitierPegHole2dModel::sampleObservation(const Eigen::VectorXd &state,
                                                           unsigned int sensing_action) const
{
    Eigen::VectorXd noise(observation_size_);
    sensing_noise_model_->dev(noise);
    return getObservation(state, sensing_action) + noise;
}

bool MultitierPegHole2dModel::isTerminal(const Eigen::VectorXd &state) const
{
    return isGoal(state);
}

bool MultitierPegHole2dModel::isGoal(const Eigen::VectorXd &state) const
{
    return holes_.back().isInside(state(0), state(1));
}

void MultitierPegHole2dModel::fillMarker(const Eigen::VectorXd &state, visualization_msgs::Marker &marker) const
{
    marker.type = visualization_msgs::Marker::CUBE;
    marker.action = visualization_msgs::Marker::ADD;

    tf::Quaternion quaternion;
    quaternion.setRPY(0, 0, state(2));
    marker.pose.orientation.x = quaternion.getX();
    marker.pose.orientation.y = quaternion.getY();
    marker.pose.orientation.z = quaternion.getZ();
    marker.pose.orientation.w = quaternion.getW();

    marker.pose.position.x = state(0);
    marker.pose.position.y = state(1);
    marker.pose.position.z = 0;

    marker.scale.x = peg_dim_x_;
    marker.scale.y = peg_dim_y_;
    marker.scale.z = 1.0;
}

void MultitierPegHole2dModel::publishMap()
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

        for (int i = 0; i < getNumHoles(); i++)
        {
            for (int j = 0; j < 2; j++)
            {
                marker.header.frame_id = "world";
                marker.header.stamp = ros::Time::now();
                marker.ns = "basic_shapes";
                marker.id = i * 2 + j;
                marker.type = visualization_msgs::Marker::CUBE;
                marker.action = visualization_msgs::Marker::ADD;

                marker.pose.orientation.x = 0;
                marker.pose.orientation.y = 0;
                marker.pose.orientation.z = 0;
                marker.pose.orientation.w = 1;

                // Left edge.
                if (j == 0)
                {
                    marker.pose.position.x = 0.5 * (holes_[i].edges[0] + holes_[i].edges[1]);
                    marker.pose.position.y = holes_[i].center[1];
                    marker.pose.position.z = 0;

                    marker.scale.x = holes_[i].edges[1] - holes_[i].edges[0];
                    marker.scale.y = holes_[i].depth;
                    marker.scale.z = 1.0;
                }

                // Right edge.
                else
                {
                    marker.pose.position.x = 0.5 * (holes_[i].edges[2] + holes_[i].edges[3]);
                    marker.pose.position.y = holes_[i].center[1];
                    marker.pose.position.z = 0;

                    marker.scale.x = holes_[i].edges[3] - holes_[i].edges[2];
                    marker.scale.y = holes_[i].depth;
                    marker.scale.z = 1.0;
                }

                marker.color.r = 1.0;
                marker.color.g = 0.0;
                marker.color.b = 0.0;
                marker.color.a = 1.0;

                marker.lifetime = ros::Duration();

                marker_array.markers.push_back(marker);
            }
        }

        publisher_.publish(marker_array);
    }
}

bool MultitierPegHole2dModel::isCollision(const Eigen::VectorXd &state) const
{
    fcl::Vec3f translation;
    fcl::Matrix3f rotation;
    translation.setValue(state(0), state(1), 0);
    rotation.setEulerYPR(state(2), 0, 0);
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

Eigen::VectorXd MultitierPegHole2dModel::moveToFreeState(const Eigen::VectorXd &state, const Eigen::VectorXd &action,
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

unsigned int MultitierPegHole2dModel::getNumHoles() const
{
    return static_cast<unsigned int>(holes_.size());
}

void MultitierPegHole2dModel::printParameters(std::ofstream &file) const
{
    file << "Model Parameters" << std::endl;
    file << "peg_width = " << peg_dim_x_ << std::endl;
    file << "peg_height = " << peg_dim_y_ << std::endl;
    file << "init_mean = \n" << init_belief_->mean_.transpose() << std::endl;
    file << "init_cov = \n" << init_belief_->cov_.diagonal().transpose() << std::endl;
    file << "action_cov = \n" << motion_noise_model_->cov_.diagonal().transpose() << std::endl;
    file << "sensing_cov = \n" << sensing_noise_model_->cov_.diagonal().transpose() << std::endl;

    for (int i = 0; i < getNumHoles(); i++)
    {
        file << "hole_center = " << holes_[i].center.transpose() << std::endl;
        file << "hole_tolerance = " << holes_[i].tol << std::endl;
        file << "hole_edges = " << holes_[i].edges.transpose() << std::endl;
    }

    file << "translation_step_size = " << trans_step_size_ << std::endl;
    file << "rotation_step_size = " << rot_step_size_ << std::endl;
    file << "collision_tol = " << collision_tolerance_ << std::endl;
}

Eigen::VectorXd MultitierPegHole2dModel::normalizeAction(const Eigen::VectorXd &task_action) const
{
    Eigen::VectorXd normalized_action(state_size_);
    normalized_action(0) = task_action(0) * std::min(1.0, trans_step_size_ / std::abs(task_action(0)));
    normalized_action(1) = task_action(1) * std::min(1.0, trans_step_size_ / std::abs(task_action(1)));
    normalized_action(2) = task_action(2) * std::min(1.0, rot_step_size_ / std::abs(task_action(2)));
    return normalized_action;
}
