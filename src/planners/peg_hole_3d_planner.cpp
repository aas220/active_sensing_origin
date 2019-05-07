//
// Created by tipakorng on 8/2/17.
//

#include "planners/peg_hole_3d_planner.h"


PegHole3dPlanner::PegHole3dPlanner(double peg_dim_1, double peg_dim_2, double peg_dim_3, double hole_tolerance,
                                   double translation_step_size, double rotation_step_size) :
        peg_dim_1_(peg_dim_1),
        peg_dim_2_(peg_dim_2),
        peg_dim_3_(peg_dim_3),
        hole_tolerance_(hole_tolerance),
        tran_step_size_(translation_step_size),
        rot_step_size_(rotation_step_size)
{}

PegHole3dPlanner::~PegHole3dPlanner()
{}

Eigen::VectorXd PegHole3dPlanner::policy(const Eigen::VectorXd &state)
{
    Eigen::VectorXd action(state.size());
    action.setZero();

    // If the peg is not above the hole, move the peg toward the hole.
    // Otherwise, re-orient the peg and lower it down.
    if (state.segment<2>(0).norm() > hole_tolerance_)
    {
        action.segment<2>(0) = -state.segment<2>(0) * std::min(1.0, tran_step_size_ / state.segment<2>(0).norm());
    }

    else
    {
        action.segment<3>(3) = -state.segment<3>(3) * std::min(1.0, rot_step_size_ / state.segment<3>(3).norm());

        // If angle error is small according to a simple heuristic, lower the peg.
        if (state.segment<3>(3).norm() < std::asin(hole_tolerance_ / (0.5 * peg_dim_3_)))
            action(2) = -state(2) * std::min(1.0, tran_step_size_ / std::abs(state(2)));
    }

    return action;
}
