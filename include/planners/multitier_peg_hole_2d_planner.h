//
// Created by tipakorng on 8/19/17.
//

#ifndef ACTIVE_SENSING_CONTINUOUS_MULTITIER_PEG_HOLE_2D_PLANNER_H
#define ACTIVE_SENSING_CONTINUOUS_MULTITIER_PEG_HOLE_2D_PLANNER_H

#include <fstream>

#include "state_space_planner.h"
#include "models/multitier_peg_hole_2d_model.h"


class MultitierPegHole2dPlanner : public StateSpacePlanner
{
public:
    explicit MultitierPegHole2dPlanner(const std::string &file_path);

    virtual ~MultitierPegHole2dPlanner();

    virtual Eigen::VectorXd policy(const Eigen::VectorXd &state);

    void reset();

    void printParameters(std::ofstream &file) const;

private:
    unsigned int state_size_;

    double peg_dim_x_;

    double peg_dim_y_;

    std::vector<Hole> holes_;

    unsigned int id_;
};

#endif //ACTIVE_SENSING_CONTINUOUS_MULTITIER_PEG_HOLE_2D_PLANNER_H
