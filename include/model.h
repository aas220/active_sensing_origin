//
// Created by tipakorng on 12/3/15.
//

#ifndef ACTIVE_SENSING_CONTINUOUS_MODEL_H
#define ACTIVE_SENSING_CONTINUOUS_MODEL_H

#include <visualization_msgs/Marker.h>
#include "Eigen/Dense"

//abstract class
class Model {

public:

    /**
     * \brief Return the dimension of the state space.
     *
     * @return The dimension of the state space.
     */
    virtual unsigned int getStateSize() const = 0;

    /**
     * \brief Calculate the observation probability.
     *
     * @param state A state.
     * @param sensing_action A sensing action.
     * @param observation An observation.
     * @return The probability of getting the observation.
     */
    virtual double getObservationProbability(const Eigen::VectorXd &state, unsigned int sensing_action, 
                                             const Eigen::VectorXd &observation) const = 0;

    /**
     * \brief Calculate the transition probability
     *
     * @param next_state The next state.
     * @param current_state The current state.
     * @param task_action A task action
     * @return The probability of transitioning to the next state from the current state given the task action.
     */
    virtual double getTransitionProbability(const Eigen::VectorXd &next_state, const Eigen::VectorXd &current_state, 
                                            const Eigen::VectorXd &task_action) const = 0;

    /**
     * \brief Return the reward obtained from performing an action at a state.
     *
     * @param state A state.
     * @param task_action A task action.
     * @return The reward from performing the action at the state.
     */
    virtual double getReward(const Eigen::VectorXd &state, const Eigen::VectorXd &task_action) const = 0;

    /**
     * \brief Return the mean of the initial belief.
     *
     * @return The mean of the initial belief.
     */
    virtual Eigen::VectorXd getInitState() const = 0;

    /**
     * \brief Return the maximum-likelihood observation given state and sensing action.
     *
     * @param state A state.
     * @param sensing_action A sensing action.
     * @return The maximum likelihood observation.
     */
    virtual Eigen::VectorXd getObservation(const Eigen::VectorXd &state, unsigned int sensing_action) const = 0;

    /**
     * \brief Return the maximum-likelihood next state give the current state and a task action.
     *
     * @param state The current state.
     * @param task_action A task action.
     * @return The next state.
     */
    virtual Eigen::VectorXd getNextState(const Eigen::VectorXd &state, const Eigen::VectorXd &task_action) const = 0;

    /**
     * \brief Return the next state sampled according to the transition probability.
     *
     * @param state The current state.
     * @param task_action A task action.
     * @return The next state.
     */
    virtual Eigen::VectorXd sampleNextState(const Eigen::VectorXd &state, const Eigen::VectorXd &task_action) const = 0;

    /**
     * \brief Sample an initial state.
     *
     * @return An initial state
     */
    virtual Eigen::VectorXd sampleInitState() const = 0;

    /**
     * \brief Sample an observation from the sensing model.
     *
     * @param state The current state.
     * @param sensing_action A sensing action.
     * @return An observation.
     */
    virtual Eigen::VectorXd sampleObservation(const Eigen::VectorXd &state, unsigned int sensing_action) const = 0;

    /**
     * \brief Return true iff the state is terminal (e.g., goal or obstacle).
     *
     * @param state The current state.
     * @return True iff the state is terminal,
     */
    virtual bool isTerminal(const Eigen::VectorXd &state) const = 0;

    virtual bool isGoal(const Eigen::VectorXd &state) const = 0;

    /**
     * \brief Return the list of possible sensing actions.
     *
     * @return The list of possible sensing action.
     */
    std::vector<unsigned int> getSensingActions()
    {
        return sensing_actions_;
    }

    virtual void fillMarker(const Eigen::VectorXd &state, visualization_msgs::Marker &marker) const = 0;

    virtual void publishMap() = 0;

protected:

    /**
     * \brief The list of sensing actions.
     */
    std::vector<unsigned int> sensing_actions_;
};


#endif //ACTIVE_SENSING_CONTINUOUS_MODEL_H
