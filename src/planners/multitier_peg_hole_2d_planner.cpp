//
// Created by tipakorng on 8/19/17.
//

#include <yaml-cpp/yaml.h>

#include "planners/multitier_peg_hole_2d_planner.h"


MultitierPegHole2dPlanner::MultitierPegHole2dPlanner(const std::string &file_path)
{
    // Initialize the yml node.
    YAML::Node root = YAML::LoadFile(file_path);

    // Initialize peg.
    state_size_ = 3;
    peg_dim_x_ = root["model"]["peg_dim"][0].as<double>();
    peg_dim_y_ = root["model"]["peg_dim"][1].as<double>();

    // Initialize holes.
    std::vector<double> map_dim = root["model"]["map_dim"].as<std::vector<double> >();

    for (int i = 0; i < root["model"]["holes"].size(); i++)
    {
        Hole hole_struct;
        hole_struct.center[0] = root["model"]["holes"][i]["center"][0].as<double>();
        hole_struct.center[1] = root["model"]["holes"][i]["center"][1].as<double>();
        hole_struct.tol = root["model"]["holes"][i]["tol"].as<double>();
        hole_struct.depth = root["model"]["holes"][i]["depth"].as<double>();
        hole_struct.edges[0] = map_dim[0];
        hole_struct.edges[1] = hole_struct.center[0] - 0.5 * peg_dim_x_ - hole_struct.tol;
        hole_struct.edges[2] = hole_struct.center[0] + 0.5 * peg_dim_x_ + hole_struct.tol;
        hole_struct.edges[3] = map_dim[2];
        holes_.push_back(hole_struct);
    }

    // Reset planner states.
    reset();
}

MultitierPegHole2dPlanner::~MultitierPegHole2dPlanner()
{}

Eigen::VectorXd MultitierPegHole2dPlanner::policy(const Eigen::VectorXd &state)
{
    Eigen::VectorXd action(3);
    action.setZero();

    if (state(1) + peg_dim_y_ < holes_[id_].center(1) - holes_[id_].depth - holes_[id_].tol)
    {
        if (id_ < holes_.size() - 1)
            id_ ++;
    }

    // Move the peg to the center of the i-th hole.
    action(0) = holes_[id_].center(0) - state(0);

    // If the peg is close enough to the center of the hole, rotate and insert it.
    if (std::abs(holes_[id_].center(0) - state(0)) < 0.5 * peg_dim_x_)
    {
        // Rotate the peg to align it to the hole.
        action(2) = -state(2);

        // If the alignment is good enough, insert the peg.
        if (std::abs(state(2)) < std::asin((holes_[id_].tol / (0.5 * peg_dim_y_))))
        {
            action(1) = holes_[id_].center(1) - holes_[id_].depth - holes_[id_].tol - state(1) - peg_dim_y_;
        }
    }

    return action;
}

void MultitierPegHole2dPlanner::reset()
{
    id_ = 0;
}

void MultitierPegHole2dPlanner::printParameters(std::ofstream &file) const
{}
