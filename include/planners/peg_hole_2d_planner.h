//
// Created by tipakorng on 7/26/17.
//

#ifndef ACTIVE_SENSING_CONTINUOUS_PEG_HOLE_2D_PLANNER_H
#define ACTIVE_SENSING_CONTINUOUS_PEG_HOLE_2D_PLANNER_H

#include "state_space_planner.h"


class PegHole2dPlanner : public StateSpacePlanner
{
public:

    explicit PegHole2dPlanner(double peg_width, double peg_height, double hole_tolerance);

    virtual ~PegHole2dPlanner();

    virtual Eigen::VectorXd policy(const Eigen::VectorXd &state);

private:

    template<typename T>
    int sgn(T val)
    {
        return (T(0) < val) - (val < T(0));
    }

    double peg_width_;

    double peg_height_;

    double hole_tolerance_;
};

#endif //ACTIVE_SENSING_CONTINUOUS_PEG_HOLE_2D_PLANNER_H
