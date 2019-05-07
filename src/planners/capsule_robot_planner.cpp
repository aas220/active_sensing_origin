//
// Created by tipakorng on 9/8/17.
//

#include <yaml-cpp/yaml.h>

#include "planners/capsule_robot_planner.h"

CapsuleRobotPlanner::CapsuleRobotPlanner(const std::string &file_path)
{
    YAML::Node root = YAML::LoadFile(file_path);
    YAML::Node planner = root["state_space_planner"];

    unsigned int num_via_points = planner["num_via_points"].as<unsigned int>();
    Eigen::Vector2d via_point;

    for (unsigned int i = 0; i < num_via_points; i++)
    {
        via_point[0] = planner["via_points"][i]["center"][0].as<double>();
        via_point[1] = planner["via_points"][i]["center"][1].as<double>();
        via_points_.push_back(via_point);
        radii_.push_back(planner["via_points"][i]["radius"].as<double>());
        axes_.push_back(planner["via_points"][i]["axis"].as<unsigned int>());
    }

    via_point[0] = root["model"]["goal"]["center"][0].as<double>();
    via_point[1] = root["model"]["goal"]["center"][1].as<double>();
    via_points_.push_back(via_point);
    radii_.push_back(root["model"]["goal"]["radius"].as<double>());
    axes_.push_back(root["model"]["goal"]["axis"].as<unsigned int>());

    gain_0_ = planner["gain_0"].as<double>();
    gain_1_ = planner["gain_1"].as<double>();

    reset();
}

CapsuleRobotPlanner::~CapsuleRobotPlanner()
{}

Eigen::VectorXd CapsuleRobotPlanner::policy(const Eigen::VectorXd &state)
{
    Eigen::VectorXd action(state.size());

    if ((state - via_points_[id_]).norm() < radii_[id_] && id_ < via_points_.size() - 1)
        id_++;

    if (axes_[id_] == 0)
    {
        action(0) = gain_0_ * (via_points_[id_](0) - state(0));
        action(1) = gain_1_ * (via_points_[id_](1) - state(1));
    }

    if (axes_[id_] == 1)
    {
        action(0) = gain_1_ * (via_points_[id_](0) - state(0));
        action(1) = gain_0_ * (via_points_[id_](1) - state(1));
    }

    return action;
}

void CapsuleRobotPlanner::reset()
{
    id_ = 0;
}
