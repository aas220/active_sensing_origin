//
// Created by tipakorng on 7/31/17.
//

#ifndef ACTIVE_SENSING_CONTINUOUS_PEG_HOLE_3D_MODEL_H
#define ACTIVE_SENSING_CONTINUOUS_PEG_HOLE_3D_MODEL_H
#include <ros/ros.h>
#include "model.h"
#include "multivariate_gaussian.h"
#include "fcl/shape/geometric_shapes.h"
#include "fcl/narrowphase/narrowphase.h"
#include "fcl/shape/geometric_shape_to_BVH_model.h"
#include "fcl/shape/geometric_shapes_utility.h"
#include "fcl/collision.h"

class PegHole3d : public Model
{
public:
    explicit PegHole3d(double peg_dim_1, double peg_dim_2, double peg_dim_3,
                       double map_dim_1, double map_dim_2, double map_dim_3, double hole_tolerance,
                       const Eigen::VectorXd &init_mean, const Eigen::MatrixXd &init_cov,
                       const Eigen::MatrixXd &motion_cov, double sensing_cov, unsigned long long seed);

    virtual ~PegHole3d();

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
    Eigen::VectorXd moveToFreeState(const Eigen::VectorXd &state, const Eigen::VectorXd &action,
                                    double lo, double hi, double tol) const;

    double peg_dim_1_;

    double peg_dim_2_;

    double peg_dim_3_;

    double hole_tolerance_;

    double collision_tolerance_;

    unsigned int state_size_;

    unsigned int observation_size_;

    MultivariateGaussian *init_belief_;

    MultivariateGaussian *motion_noise_model_;

    MultivariateGaussian *sensing_noise_model_;

    fcl::CollisionObject *peg_;

    std::vector<fcl::CollisionObject*> hole_sides_;

    std::vector<fcl::CollisionObject*> walls_;
};

#endif //ACTIVE_SENSING_CONTINUOUS_PEG_HOLE_3D_MODEL_H
