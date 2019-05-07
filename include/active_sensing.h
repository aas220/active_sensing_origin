//
// Created by tipakorng on 12/3/15.
//

#ifndef ACTIVE_SENSING_CONTINUOUS_ACTIVE_SENSING_H
#define ACTIVE_SENSING_CONTINUOUS_ACTIVE_SENSING_H

#include <vector>
#include "Eigen/Dense"
#include "model.h"
#include "state_space_planner.h"
#include "particle_filter.h"
#include "entropy_estimation.h"


class ActiveSensing{

public:

    /**
     * \brief Constructor.
     *
     * @param model The model of the system.
     * @param planner The state-space planner.
     * @param particle_filter The particle filter.
     * @param horizon The search horizon for the active sensing algorithm.
     * @param discount The discount factor of the active-sensing algorithm.
     */
    explicit ActiveSensing(Model &model, StateSpacePlanner &planner, ParticleFilter &particle_filter, 
                           unsigned int horizon, double discount, unsigned int num_observations=100,
                           unsigned int num_nearest_neighbors=5, unsigned int num_cores=4);

    /**
     * \brief Destructor.
     */
    virtual ~ActiveSensing();

    /**
     * \brief Return sensing action that minimizes the entropy.
     *
     * @param particles A list of state particles.
     */
    virtual unsigned int getSensingAction(const std::vector<Particle> &particles) const;

    /**
     * \brief Set the number of nearest neighbors for entropy estimation.
     *
     * @param k The number of nearest neighbors.
     */
    void setNumNearestNeighbors(unsigned int k);

    /**
     * \brief Set the number of observations.
     *
     * @param n The number of observations.
     */
    void setNumObservations(unsigned int n);

    /**
     * \brief Return entropy of the particles conditioned on the sensing action.
     *
     * @param particles A list of state entropy.
     * @param sensing_action A sensing action.
     */
    double getConditionalCumulativeEntropy(const std::vector<Particle> &particles, unsigned int sensing_action) const;

protected:

    /**
     * \brief Return entropy of the particles.
     *
     * @param particles A list of state particles.
     */
    virtual double getCumulativeEntropy(const std::vector<Particle> &particles) const;

    /**
     * \brief Return task action with lowest entropy.
     *
     * @param particles A list of state particles.
     */
    Eigen::VectorXd getTaskAction(const std::vector<Particle> &particles) const;

    /**
     * \brief The model of the system.
     */
    Model &model_;

    /**
     * \brief The state space planner.
     */
    StateSpacePlanner &planner_;

    /**
     * \brief The particle filter.
     */
    ParticleFilter &particle_filter_;

    /**
     * \brief The sensing horizon.
     */
    unsigned int horizon_;

    /**
     * \brief The discount factor of the active sensing algorithm.
     */
    double discount_;

    /**
     * \brief The number of observation samples in conditional entropy estimation.
     */
    unsigned int num_observations_;

    /**
     * \brief The number of nearest neighbors for entropy estimation.
     */
    unsigned int num_nearest_neighbors_;

    /**
     * \brief The number of cores used in entropy estimation.
     */
    unsigned int num_cores_;

    Rng *rng_;
};


class ActionEntropyActiveSensing : public ActiveSensing {

public:

    /**
     * \brief Constructor.
     *
     * @param model The model of the system.
     * @param planner The state-space planner.
     * @param particle_filter The particle filter.
     * @param horizon The search horizon for the active sensing algorithm.
     * @param discount The discount factor of the active-sensing algorithm.
     */
    explicit ActionEntropyActiveSensing(Model &model, StateSpacePlanner &planner, ParticleFilter &particle_filter, 
                                        unsigned int horizon, double discount, unsigned int num_observations=100,
                                        unsigned int num_nearest_neighbors=5, unsigned int num_cores=4);

    /**
     * \brief Destructor.
     */
    ~ActionEntropyActiveSensing();

    /**
     * \brief This function returns the conditional entropy of the task action particles.
     *
     * @param particles A list of state particles.
     * @return The cumulative entropy of the task action particles.
     */
    double getCumulativeEntropy(const std::vector<Particle> &particles) const;

private:

    /**
     * \brief Calculate task-action particles from the state particles.
     *
     * @param state_particles A list of state particles.
     * @return A list of task-action particles.
     */
    std::vector<Particle> getTaskActionParticles(const std::vector<Particle> &state_particles) const;

};


class RandomActiveSensing : public ActiveSensing {

public:

    /**
     * \brief Constructor.
     *
     * @param model The model of the system.
     * @param planner The state-space planner.
     * @param particle_filter The particle filter.
     * @param horizon The search horizon for the active sensing algorithm.
     * @param discount The discount factor of the active-sensing algorithm.
     */
    explicit RandomActiveSensing(Model &model, StateSpacePlanner &planner, ParticleFilter &particle_filter,
                                        unsigned int horizon, double discount);

    /**
     * \brief Destructor.
     */
    ~RandomActiveSensing();

    /**
     * \brief Return a randomly selected sensing action.
     *
     * @param particles A list of state particles.
     */
    virtual unsigned int getSensingAction(const std::vector<Particle> &particles) const;

};

#endif //ACTIVE_SENSING_CONTINUOUS_ACTIVE_SENSING_H
