//
// Created by tipakorng on 11/23/15.
//

#ifndef ACTIVE_SENSING_CONTINUOUS_PARTICLE_FILTER_H
#define ACTIVE_SENSING_CONTINUOUS_PARTICLE_FILTER_H

#include <vector>
#include <time.h>
#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include "rng.h"
#include "Eigen/Dense"
#include "model.h"


class Particle{

public:

    /**
     * \brief The constructor.
     *
     * @param value The value (or coordinates) of the particle.
     * @param weight The weight of the particle.
     */
    Particle(const Eigen::VectorXd &value, double weight);

    /**
     * \brief The destructor.
     */
    ~Particle();

    /**
     * \brief Set a new value.
     *
     * @param value A new value.
     */
    void setValue(const Eigen::VectorXd &value);

    /**
     * \brief Set a new weight.
     *
     * @param weight A new weight.
     */
    void setWeight(double weight);

    /**
     * \brief Return the dimension of the std::vector space containing the particle value.
     *
     * @return The dimension.
     */
    unsigned int getDimension() const;

    /**
     * \brief Return the value of the particle.
     *
     * @return The value.
     */
    Eigen::VectorXd getValue() const;

    double getWeight() const;

private:

    /**
     * \brief The value (or coordinates) of the particle.
     */
    Eigen::VectorXd value_;

    /**
     * \brief The weight of the particle.
     */
    double weight_;

    /**
     * The dimension of the std::vector space the value of the particle is in.
     */
    const unsigned int dimension_;
};

class ParticleFilter{
public:

    /**
     * \brief The constructor.
     *
     * @param model
     * @param num_particles
     */
    explicit ParticleFilter(const Model &model, unsigned int num_particles);

    /**
     * \brief The constructor.
     *
     * @param model
     * @param num_particles
     */
    explicit ParticleFilter(const Model &model, unsigned int num_particles, ros::NodeHandle *node_handle);

    /**
     * \brief The destructor.
     */
    virtual ~ParticleFilter();

    /**
     * \brief Sample the initial particles from the initial belief.
     */
    void initParticles();

    /**
     * \brief Update the weight of the filter's particles.
     *
     * @param sensing_action A sensing action
     * @param observation An observation
     */
    void updateWeights(unsigned int sensing_action, const Eigen::VectorXd &observation);

    /**
     * \brief Update particles' weight.
     *
     * @param particles A list of particles.
     * @param sensing_action A sensing action
     * @param observation An observation
     */
    void updateWeights(std::vector<Particle> &particles, unsigned int sensing_action,
                       const Eigen::VectorXd &observation) const;

    /**
     * \brief Resample the filter's particles based on the weights.
     */
    void resample();

    /**
     * \brief Resample the particles based on the weights.
     *
     * @param particles A list of particles.
     */
    void resample(std::vector<Particle> &particles);

    /**
     * \brief Propagate the filter's particles.
     *
     * @param task_action A task action.
     */
    void propagate(const Eigen::VectorXd task_action);

    /**
     * \brief Propagate the particles based on the task action.
     *
     * @param particles A list of particles.
     * @param task_action A task action.
     */
    void propagate(std::vector<Particle> &particles, const Eigen::VectorXd &task_action) const;

    /**
     * Update the weights and propagate the filter's particles.
     *
     * @param task_action A task action.
     * @param sensing_action A sensing action.
     * @param observation An observation.
     */
    void update(const Eigen::VectorXd task_action, unsigned int sensing_action, const Eigen::VectorXd observation);

    /**
     * \brief Update the weights and propagates the particles.
     *
     * @param particles A list of particles.
     * @param task_action A task action.
     * @param sensing_action A sensing action.
     * @param observation An observation.
     */
    void update(std::vector<Particle> &particles, const Eigen::VectorXd task_action, unsigned int sensing_action,
                const Eigen::VectorXd &observation) const;

    /**
     * \brief Normalize the filter's particles.
     */
    void normalize();

    /**
     * \brief Normalize the particles.
     *
     * @param particles A list of particles.
     */
    void normalize(std::vector<Particle> &particles) const;

    /**
     * \brief Sample a particle based on the normalized weights.
     *
     * @param particles A list of particles.
     * @return A sampled particle.
     */
    Particle importanceSampling(const std::vector<Particle> &particles);

    /**
     * Return the number of particles.
     *
     * @return The number of particles.
     */
    unsigned int getNumParticles();

    /**
     * Return the particles.
     * @return The filter's particles
     */
    std::vector<Particle> getParticles();

    /**
     * \brief Return particle with largest weight.
     *
     * TODO: Should use search tree to store particles.
     *
     * @return The particle with largest weight.
     */
    Particle getBestParticle() const;

    /**
     * \brief Publish the particles as MarkerArray.
     */
    void publish();

    /**
     * \brief Publish the particles as MarkerArray.
     */
    void publish(const std::vector<Particle> &particles);

    /**
     * \brief Check if the particle filter has a publisher.
     *
     * @return True iff the particle filter has a publisher.
     */
    bool isPublishable()const ;

private:

    /**
     * \brief The number of particles the filter has.
     */
    const unsigned int numParticles_;

    /**
     * \brief The filter's particles.
     */
    std::vector<Particle> particles_;
    /**
     * \brief The model of the system.
     */
    const Model &model_;

    /**
     * \brief The random number generator.
     */
    Rng rng_;

    /**
     * \brief ROS node handle.
     */
    ros::NodeHandle *node_handle_;

    /**
     * \brief ROS tranform broadcaster.
     */
    tf::TransformBroadcaster tf_broadcaster_;

    /**
     * \brief ROS publisher.
     */
    ros::Publisher publisher_;

    /**
     * \brief True iff the particle filter has a publisher.
     */
    bool has_publisher_;
};

#endif //ACTIVE_SENSING_CONTINUOUS_PARTICLE_FILTER_H
