//
// Created by tipakorng on 7/25/17.
//

#ifndef ACTIVE_SENSING_CONTINUOUS_PEG_HOLE_2D_MODEL_H
#define ACTIVE_SENSING_CONTINUOUS_PEG_HOLE_2D_MODEL_H

#include "model.h"
#include "multivariate_gaussian.h"


class Rectangle
{
public:
    Rectangle(double dimension1, double dimension2);

    ~Rectangle();

    void corners(double x, double y, double angle, std::vector<Eigen::Vector2d> &corner_list) const;

    void axes(double x, double y, double angle, std::vector<Eigen::Vector2d> &axis_list) const;

private:
    double dimension1_;

    double dimension2_;

    double x_;

    double y_;

    double angle_;

};


class PegHole2d: public Model
{
public:
    explicit PegHole2d(double peg_width, double peg_height, double hole_tolerance,
                       const Eigen::VectorXd &init_mean, const Eigen::MatrixXd &init_cov,
                       const Eigen::MatrixXd &motion_cov, double sensing_cov,
                       double trans_step_size, double rot_step_size, double collision_tol);

    virtual ~PegHole2d();

    virtual unsigned int getStateSize() const;

    virtual double getObservationProbability(const Eigen::VectorXd &state, unsigned int sensing_action,
                                             const Eigen::VectorXd &observation) const;

    virtual double getTransitionProbability(const Eigen::VectorXd &next_state, const Eigen::VectorXd &current_state,
                                            const Eigen::VectorXd &task_action) const;

    virtual double getReward(const Eigen::VectorXd &state, const Eigen::VectorXd &task_action) const;

    virtual Eigen::VectorXd getInitState() const;

    virtual Eigen::VectorXd getObservation(const Eigen::VectorXd &state, unsigned int sensing_action) const;

    virtual Eigen::VectorXd getNextState(const Eigen::VectorXd &state, const Eigen::VectorXd &task_action) const;

    virtual Eigen::VectorXd sampleInitState() const;

    virtual Eigen::VectorXd sampleNextState(const Eigen::VectorXd &state, const Eigen::VectorXd &task_action) const;

    virtual Eigen::VectorXd sampleObservation(const Eigen::VectorXd &state, unsigned int sensing_action) const;

    virtual bool isTerminal(const Eigen::VectorXd &state) const;

    virtual bool isGoal(const Eigen::VectorXd &state) const;

    virtual void fillMarker(const Eigen::VectorXd &state, visualization_msgs::Marker &marker) const;

    virtual void publishMap();

    bool isCollision(const Eigen::VectorXd &state) const;

private:
    bool separatingAxisTheorem(const std::vector<Eigen::Vector2d> &corner_list_1,
                     const std::vector<Eigen::Vector2d> &corner_list_2) const;

    bool isSeparated(const Eigen::Vector2d &axis,
                     const std::vector<Eigen::Vector2d> &corner_list_1,
                     const std::vector<Eigen::Vector2d> &corner_list_2) const;

    Eigen::VectorXd moveToFreeState(const Eigen::VectorXd &state, const Eigen::VectorXd &action,
                                    double lo, double hi, double tol) const;

    Eigen::VectorXd normalizeAction(const Eigen::VectorXd &action) const;

    double peg_width_;

    double peg_height_;

    Rectangle peg_;

    double hole_tolerance_;

    unsigned int state_size_;

    unsigned int observation_size_;

    MultivariateGaussian *init_belief_;

    MultivariateGaussian *motion_noise_;

    MultivariateGaussian *sensing_noise_;

    std::vector<Eigen::Vector2d> left_wall_;

    std::vector<Eigen::Vector2d> hole_floor_;

    std::vector<Eigen::Vector2d> right_wall_;

    double trans_step_size_;

    double rot_step_size_;
    
    double collision_tol_;

};

#endif //ACTIVE_SENSING_CONTINUOUS_PEG_HOLE_2D_MODEL_H


