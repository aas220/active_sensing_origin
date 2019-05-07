//
// Created by tipakorng on 8/18/17.
//

#ifndef ACTIVE_SENSING_CONTINUOUS_MULTITIER_PEG_HOLE_2D_MODEL_H
#define ACTIVE_SENSING_CONTINUOUS_MULTITIER_PEG_HOLE_2D_MODEL_H

#include <fstream>

#include <ros/ros.h>
#include <tf/transform_broadcaster.h>

#include "model.h"
#include "multivariate_gaussian.h"
#include "fcl/shape/geometric_shapes.h"


struct Hole
{
    Eigen::Vector2d center;

    Eigen::Vector4d edges;

    double depth;

    double tol;

    bool isInside(double x, double y) const;
};


class MultitierPegHole2dModel : public Model
{
public:
    explicit MultitierPegHole2dModel(const std::string &file_path);

    explicit MultitierPegHole2dModel(const std::string &file_path, ros::NodeHandle *node_handle);

    virtual ~MultitierPegHole2dModel();

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

    bool isGoal(const Eigen::VectorXd &state) const;

    void fillMarker(const Eigen::VectorXd &state, visualization_msgs::Marker &marker) const;

    void publishMap();

    bool isCollision(const Eigen::VectorXd &state) const;

    unsigned int getNumHoles() const;

    void printParameters(std::ofstream &file) const;

private:
    Eigen::VectorXd moveToFreeState(const Eigen::VectorXd &state, const Eigen::VectorXd &action,
                                    double lo, double hi, double tol) const;

    Eigen::VectorXd normalizeAction(const Eigen::VectorXd &action) const;

    unsigned int state_size_;

    unsigned int observation_size_;

    double collision_tolerance_;

    double peg_dim_x_;

    double peg_dim_y_;

    std::vector<double> map_dim_;

    MultivariateGaussian *init_belief_;

    MultivariateGaussian *motion_noise_model_;

    MultivariateGaussian *sensing_noise_model_;

    fcl::CollisionObject *peg_;

    std::vector<fcl::CollisionObject*> hole_sides_;

    std::vector<fcl::CollisionObject*> walls_;

    std::vector<Hole> holes_;

    ros::NodeHandle *node_handle_;

    tf::TransformBroadcaster tf_broadcaster_;

    ros::Publisher publisher_;

    bool has_publisher_;

    double trans_step_size_;

    double rot_step_size_;
};

#endif //ACTIVE_SENSING_CONTINUOUS_MULTITIER_PEG_HOLE_2D_MODEL_H
