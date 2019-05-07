//
// Created by tipakorng on 9/8/17.
//

#ifndef ACTIVE_SENSING_CONTINUOUS_CAPSULE_ROBOT_PLANNER_H
#define ACTIVE_SENSING_CONTINUOUS_CAPSULE_ROBOT_PLANNER_H

#include "planner.h"

class CapsuleRobotPlanner : public StateSpacePlanner
{
public:
    explicit CapsuleRobotPlanner(const std::string &file_path);

    virtual ~CapsuleRobotPlanner();

    virtual Eigen::VectorXd policy(const Eigen::VectorXd &state);

    void reset();

    void printParameters(std::ofstream &file) const;

private:
    unsigned int id_;

    std::vector<Eigen::Vector2d> via_points_;

    std::vector<double> radii_;

    std::vector<unsigned int> axes_;

    double gain_0_;

    double gain_1_;
};

#endif //ACTIVE_SENSING_CONTINUOUS_CAPSULE_ROBOT_PLANNER_H
