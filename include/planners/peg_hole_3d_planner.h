//
// Created by tipakorng on 8/2/17.
//

#ifndef ACTIVE_SENSING_CONTINUOUS_PEG_HOLE_3D_PLANNER_H
#define ACTIVE_SENSING_CONTINUOUS_PEG_HOLE_3D_PLANNER_H

#include "state_space_planner.h"
#include "math_utils.h"

class PegHole3dPlanner : public StateSpacePlanner
{
public:
    explicit PegHole3dPlanner(double peg_dim_1, double peg_dim_2, double peg_dim_3, double hole_tolerance,
                              double translation_step_size, double rotation_step_size);

    virtual ~PegHole3dPlanner();

    virtual Eigen::VectorXd policy(const Eigen::VectorXd &state);

private:
    double peg_dim_1_;

    double peg_dim_2_;

    double peg_dim_3_;

    double hole_tolerance_;

    double tran_step_size_;

    double rot_step_size_;

};

#endif //ACTIVE_SENSING_CONTINUOUS_PEG_HOLE_3D_PLANNER_H
