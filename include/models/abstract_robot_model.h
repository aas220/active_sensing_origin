//
// Created by tipakorng on 12/4/15.
//

#ifndef ACTIVE_SENSING_CONTINUOUS_MOBILE_ROBOT_H
#define ACTIVE_SENSING_CONTINUOUS_MOBILE_ROBOT_H

#include <time.h>
#include "model.h"
#include "multivariate_gaussian.h"


// Circular object for goal and obstacles
class Hypersphere
{

public:

    Hypersphere(unsigned int dimensions);

    Hypersphere(unsigned int dimensions, double radius, Eigen::VectorXd center);

    ~Hypersphere();

    double distanceCenter(const Eigen::VectorXd &state) const;

    double distanceClosestPoint(const Eigen::VectorXd &state) const;

    Eigen::VectorXd getClosestPoint(const Eigen::VectorXd &state) const;

    void setCenter(const Eigen::VectorXd &center);

    void setRadius(double radius);

    Eigen::VectorXd getCenter() const;

    double getRadius() const;

    bool contains(const Eigen::VectorXd &state) const;

    /**
     * \brief Determines if the line from x1 to x2 intersects the hypersphere or not.
     * 
     * This function also calculate the new end point of the line segment from x1, where the x2 becomes the new end 
     * point on the hypersphere.
     * 
     * @param x1 The starting point of the line.
     * @param x2 The end point of the line.
     * @return True iff there is an intersection.
     */
    bool intersects(const Eigen::VectorXd &x1, Eigen::VectorXd &x2) const;

private:

    unsigned int dimensions_;

    double radius_;

    Eigen::VectorXd center_;

};


/**
 * \brief Mobile robot with gaussian noise in transition and sensing model.
 */
class AbstractRobot: public Model
{

public:

    /**
     * \brief The constructor.
     *
     * @param init_mean The mean of the initial belief.
     * @param init_cov The covariance of the initial belief.
     * @param action_cov The covariance of the motion model.
     * @param sensing_cov The covariance of the sensing model.
     */
    explicit AbstractRobot(const Eigen::VectorXd &init_mean, const Eigen::MatrixXd &init_cov, 
                         const Eigen::MatrixXd &action_cov, const Eigen::MatrixXd &sensing_cov);

    /**
     * \brief The destructor.
     */
    virtual ~AbstractRobot();

    /**
     * \brief Return the dimension of the state space.
     *
     * @return The dimension of the state space.
     */
    virtual unsigned int getStateSize() const;

    /**
     * \brief Calculate the probability of the observation given the state and the sensing action.
     *
     * @param state A state.
     * @param sensing_action A sensing action.
     * @param observation A observation.
     * @return Probability of the observation.
     */
    virtual double getObservationProbability(const Eigen::VectorXd &state, unsigned int sensing_action,
                                             const Eigen::VectorXd &observation) const;

    /**
     * \brief Calculate the probability of being at the next state given the current state and the task action.
     *
     * @param next_state The next state.
     * @param current_state The current state.
     * @param task_action The task action.
     * @return Probability of the task action.
     */
    virtual double getTransitionProbability(const Eigen::VectorXd &next_state, const Eigen::VectorXd &current_state,
                                            const Eigen::VectorXd &task_action) const;

    /**
     * \brief Calculate the reward from performing a given task action at a given state.
     *
     * @param state A state.
     * @param task_action A task action.
     * @return The reward of the task action performed at the state.
     */
    virtual double getReward(const Eigen::VectorXd &state, const Eigen::VectorXd &task_action) const;

    /**
     * \brief Return the mean of the initial belief.
     *
     * @return The mean of the initial belief.
     */
    virtual Eigen::VectorXd getInitState() const;

    /**
     * \brief Return an observation assuming noiseless sensing model.
     *
     * @param state A state.
     * @param sensing_action A sensing action.
     * @return The observation obtained from the state and sensing action.
     */
    virtual Eigen::VectorXd getObservation(const Eigen::VectorXd &state, unsigned int sensing_action) const;

    /**
     * \brief Return the state reached from the current state by a given task action assuming noiseless motion model.
     *
     * @param state A state.
     * @param task_action A task action.
     * @return The next state.
     */
    virtual Eigen::VectorXd getNextState(const Eigen::VectorXd &state, const Eigen::VectorXd &task_action) const;

    /**
     * \brief Sample the next state from the motion model.
     *
     * @param state A state.
     * @param task_action A task action.
     * @return The next state.
     */
    virtual Eigen::VectorXd sampleNextState(const Eigen::VectorXd &state, const Eigen::VectorXd &task_action) const;

    /**
     * \brief Sample a state from the initial belief.
     *
     * @return A state sampled from the initial belief.
     */
    virtual Eigen::VectorXd sampleInitState() const;

    /**
     * \brief Sample an observation from the observation model.
     *
     * @param state A state.
     * @param sensing_action A sensing action.
     * @return An observation sampled from the observation model.
     */
    virtual Eigen::VectorXd sampleObservation(const Eigen::VectorXd &state, unsigned int sensing_action) const;

    /**
     * \brief Check if the state is a terminal state or not.
     *
     * @param state A state.
     * @return True iff the state is terminal.
     */
    virtual bool isTerminal(const Eigen::VectorXd &state) const;

    /**
     * \brief Check if the goal is reached.
     *
     * @param state A state.
     * @return True iff the state is a goal state.
     */
    virtual bool isGoal(const Eigen::VectorXd &state) const;

    virtual void fillMarker(const Eigen::VectorXd &state, visualization_msgs::Marker &marker) const;

    virtual void publishMap();

    /**
     * \brief Check if the goal is defined.
     *
     * @return True iff the goal is defined.
     */
    virtual bool isGoalDefined() const;

    /**
     * \brief Check if the state is collision-free or not.
     *
     * @param state A state.
     * @return True iff the state is collision-free.
     */
    virtual bool isFreeState(const Eigen::VectorXd &state) const;

    /**
     * \brief Return the coordinate position of an obstacle.
     *
     * @param obstacle_id The ID of an obstacle.
     * @return The coordinate position of an obstacle.
     */
    Eigen::VectorXd getObstacleCenter(int obstacle_id) const;

    /**
     * \brief Return obstacle radius.
     *
     * @param obstacle_id The ID of an obstacle
     * @return The radius of the obstacle.
     */
    double getObstacleRadius(int obstacle_id) const;

    /**
     * \brief Return the number of obstacles.
     *
     * @return The number of obstacles.
     */
    unsigned int getNumObstacles() const;

    const Hypersphere* obstacle(unsigned int i) const;

    /**
     * \brief Return the coordinate position of the goal.
     *
     * @return The coordinate position of the goal.
     */
    Eigen::VectorXd getGoalCenter() const;

    /**
     * \brief Return the radius of the goal.
     *
     * @return The radius of the goal.
     */
    double getGoalRadius() const;

    const Hypersphere* goal() const;

    /**
     * \brief Return the dimensions of the task-action space.
     *
     * @return The dimensions of the task-action space.
     */
    unsigned int getTaskActionSize() const;

    /**
     * \brief Add an obstacle.
     *
     * @param obstacle The obstacle hypersphere.
     */
    void addObstacle(const Hypersphere &obstacle);

    /**
     * \brief Set the goal.
     *
     * @param goal The goal hypersphere.
     */
    void setGoal(const Hypersphere &goal);

    /**
     * \brief Set boundary of the problem.
     *
     * Assume axis-aligned boundary.
     *
     * @param min The lower bound of the state space.
     * @param max The upper bound of the state space.
     */
    void setBoundary(const Eigen::VectorXd &min, const Eigen::VectorXd &max);

    void moveInBoundary(Eigen::VectorXd &state) const;

protected:

    /**
     * \brief Initialize sensing action.
     */
    virtual void initSensingActions();

    /**
     * \brief Get the value of the pdf at x.
     *
     * @param x The value of a random variable.
     * @param mean The mean of the pdf.
     * @param cov The covariance of the pdf.
     * @return The value of the pdf at x.
     */
    double getProbability(const Eigen::VectorXd &x, const Eigen::VectorXd &mean, const Eigen::MatrixXd &cov) const;

    double goal_radius_;

    unsigned int state_size_;

    unsigned int task_action_size_;

    unsigned int observation_size_;

    bool is_goal_defined_;

    Eigen::VectorXd boundary_min_;

    Eigen::VectorXd boundary_max_;

    Eigen::MatrixXd action_cov_;

    Eigen::MatrixXd sensing_cov_;

    Hypersphere goal_;

    std::vector<Hypersphere> obstacle_list_;

    MultivariateGaussian *init_belief_;

    MultivariateGaussian *motion_model_;

    MultivariateGaussian *sensing_model_;
};

#endif //ACTIVE_SENSING_CONTINUOUS_MOBILE_ROBOT_H
