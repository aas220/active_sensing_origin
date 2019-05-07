//
// Created by tipakorng on 2/24/17.
//

#ifndef ACTIVE_SENSING_CONTINUOUS_STATE_SPACE_PLANNER_H
#define ACTIVE_SENSING_CONTINUOUS_STATE_SPACE_PLANNER_H

#include <Eigen/Dense>

class StateSpacePlanner {

public:

    /**
     * \brief The policy.
     *
     * @param state A state.
     * @return A task action.
     */
    virtual Eigen::VectorXd policy(const Eigen::VectorXd &state) = 0;

    virtual void reset()
    {}

};

#endif //ACTIVE_SENSING_CONTINUOUS_STATE_SPACE_PLANNER_H
