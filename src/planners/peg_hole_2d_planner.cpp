//
// Created by tipakorng on 7/26/17.
//

#include "planners/peg_hole_2d_planner.h"

PegHole2dPlanner::PegHole2dPlanner(double peg_width, double peg_height, double hole_tolerance) :
    peg_width_(peg_width),
    peg_height_(peg_height),
    hole_tolerance_(hole_tolerance)
{}

PegHole2dPlanner::~PegHole2dPlanner()
{}

Eigen::VectorXd PegHole2dPlanner::policy(const Eigen::VectorXd &state)
{
    Eigen::VectorXd action = Eigen::VectorXd::Zero(3);

    action(0) = -state(0);
    
    if (std::abs(state(0)) < 0.5 * peg_width_)
    {
        action(2) = -state(2);

        if (std::abs(state(2)) < std::asin(hole_tolerance_ / (0.5 * peg_height_)))
        {
            action(1) = -state(1);
        }
    }

    return action;
}
