//
// Created by tipakorng on 8/29/17.
//

#ifndef ACTIVE_SENSING_CONTINUOUS_CAPSULE_ROBOT_MODEL_H
#define ACTIVE_SENSING_CONTINUOUS_CAPSULE_ROBOT_MODEL_H

#include <fstream>

#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <fcl/shape/geometric_shapes.h>

#include "model.h"
#include "multivariate_gaussian.h"


struct LineSegment
{
    LineSegment(const Eigen::Vector2d &new_end_0, const Eigen::Vector2d &new_end_1, double new_thickness)
    {
        end_0 = new_end_0;
        end_1 = new_end_1;

        center = 0.5 * (end_0 + end_1);

        Eigen::Vector2d temp = end_1 - end_0;
        length = temp.norm();

        normal(0) = temp(1);
        normal(1) = -temp(0);

        thickness = new_thickness;

        bounding_box(0) = length;
        bounding_box(1) = thickness;

        angle = std::atan2(temp(1), temp(0));
    }

    Eigen::Vector2d end_0;

    Eigen::Vector2d end_1;

    Eigen::Vector2d center;

    Eigen::Vector2d normal;

    Eigen::Vector2d bounding_box;

    double thickness;

    double length;

    double angle;

};


class CapsuleRobotModel : public Model
{
public:
    explicit CapsuleRobotModel(const std::string &file_path);

    explicit CapsuleRobotModel(const std::string &file_path, ros::NodeHandle *node_handle);

    virtual ~CapsuleRobotModel();

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

    void printParameters(std::ofstream &file) const;

private:
    Eigen::VectorXd moveToFreeState(const Eigen::VectorXd &state, const Eigen::VectorXd &action,
                                    double lo, double hi, double tol) const;

    Eigen::VectorXd normalizeAction(const Eigen::VectorXd &action) const;

    unsigned int state_size_;

    unsigned int observation_size_;

    double collision_tolerance_;

    MultivariateGaussian *init_belief_;

    MultivariateGaussian *motion_noise_model_;

    std::vector<MultivariateGaussian*> sensing_noise_models_;

    ros::NodeHandle *node_handle_;

    tf::TransformBroadcaster tf_broadcaster_;

    ros::Publisher publisher_;

    bool has_publisher_;

    std::vector<fcl::CollisionObject *> walls_;

    fcl::CollisionObject *robot_;

    Eigen::VectorXd goal_;

    double goal_radius_;

    double step_size_;

    std::vector<LineSegment> wall_line_segments_;

    double robot_radius_;

};

#endif //ACTIVE_SENSING_CONTINUOUS_CAPSULE_ROBOT_MODEL_H
