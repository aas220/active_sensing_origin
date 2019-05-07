//
// Created by tipakorng on 3/8/16.
//

#ifndef ACTIVE_SENSING_CONTINUOUS_POTENTIAL_FIELD_H
#define ACTIVE_SENSING_CONTINUOUS_POTENTIAL_FIELD_H

#include "state_space_planner.h"
#include "models/abstract_robot_model.h"


class PotentialField : public StateSpacePlanner {

public:

    /**
     * \brief The constructor.
     *
     * @param robot A robot object.
     * @param step_size The step size of the planner.
     * @param threshold The obstacle threshold
     * @param attractive_coefficient The attractive coefficient of the planner.
     * @param repulsive_coefficient The repulsive coefficient of the planner.
     */
    PotentialField(const AbstractRobot &robot, double step_size, double threshold, double attractive_coefficient,
                   double repulsive_coefficient);

    /**
     * \brief The destructor.
     */
    ~PotentialField();

    /**
     * \brief The policy calculates the action at at state.
     *
     * @param state The state vector.
     * @return The action at the state.
     */
    virtual Eigen::VectorXd policy(const Eigen::VectorXd &state);

    /**
     * \brief Calculate the repulsive force at a state.
     *
     * @param state The state vector.
     * @return The repulsive force vector at the state.
     */
    virtual Eigen::VectorXd repulsiveForce(const Eigen::VectorXd &state) const;

    /**
     * \brief Calculate the attractive force at a state.
     *
     * @param state The state vector.
     * @return The attractive force vector at the state.
     */
    virtual Eigen::VectorXd attractiveForce(const Eigen::VectorXd &state) const;

    /**
     * \brief Set step size.
     *
     * @param new_step_size Set the step size.
     */
    void setStepSize(double new_step_size);

    /**
     * \brief Return the step size.
     *
     * @return The step size.
     */
    double getStepSize() const;

    /**
     * \brief Add a point that generates repulsive force.
     *
     * @param state A repulsive state.
     */
    void addRepulser(const Eigen::VectorXd &state);

    /**
     * \brief Add a point that generates attractive force.
     *
     * @param state An attractive state.
     */
    void addAttractor(const Eigen::VectorXd &state);

protected:

    // Maximum step size the robot can make in a turn.
    double step_size_;

    // Obstacle distance threshold.
    double threshold_;

    // Attractive coefficient.
    double att_;

    // Repulsive coefficient.
    double rep_;

    // Robot
    const AbstractRobot *robot_;

    // List of repulsers.
    std::vector<Eigen::VectorXd> repulser_list_;

    // List of attractors.
    std::vector<Eigen::VectorXd> attractor_list_;
};

#endif //ACTIVE_SENSING_CONTINUOUS_POTENTIAL_FIELD_H
