//
// Created by tipakorng on 3/4/16.
//

#include "../include/active_sensing.h"
#include "entropy_estimation.h"
#include <omp.h>
#include <chrono>

ActiveSensing::ActiveSensing(Model &model, StateSpacePlanner &planner, ParticleFilter &particle_filter,
                             unsigned int horizon, double discount, unsigned int num_observations,
                             unsigned int num_nearest_neighbors, unsigned int num_cores) :
    model_(model),
    planner_(planner),
    particle_filter_(particle_filter),
    horizon_(horizon),
    discount_(discount),
    num_observations_(num_observations),
    num_nearest_neighbors_(num_nearest_neighbors),
    num_cores_(num_cores)
{
    rng_ = new Rng(0);
}

ActiveSensing::~ActiveSensing()
{
    delete rng_;
}

unsigned int ActiveSensing::getSensingAction(const std::vector<Particle> &particles) const {
    std::vector<unsigned int> sensing_actions = model_.getSensingActions();
    unsigned int best_sensing_action = 0;
    double entropy;
    double min_entropy = INFINITY;
    double max_entropy = -INFINITY;

    // for (unsigned int sensing_action : sensing_actions) {
    //     entropy = getConditionalCumulativeEntropy(particles, sensing_action);

    //     if (entropy < min_entropy) {
    //         min_entropy = entropy;
    //         best_sensing_action = sensing_action;
    //     }

    //     if (entropy > max_entropy) {
    //         max_entropy = entropy;
    //     }
    // }

    // if (max_entropy == -INFINITY)
    // {
    //     unsigned long rand = rng_->int64();
    //     unsigned long index = rand % sensing_actions.size();
    //     return sensing_actions[index];
    // }

    double cumulative_entropy = 0;
    double pf_max = 0;
    double flann_max = 0;
    double pf_min = 9999;
    double flann_min = 9999;
    double pf_total = 0;
    double flann_total = 0;

    #pragma omp parallel for 
    for(int i=0;i<sensing_actions.size();i++){
        //entropy = getConditionalCumulativeEntropy(particles, sensing_actions.at(i));
        for (int j = 0; j < num_observations_; j++) {
            Eigen::VectorXd state;
    	    Eigen::VectorXd observation;
            state = particle_filter_.importanceSampling(particles).getValue();
            observation = model_.sampleObservation(state, sensing_actions.at(i));
            std::vector<Particle> updated_particles = particles;

            double pf_start = omp_get_wtime();
            particle_filter_.updateWeights(updated_particles, sensing_actions.at(i), observation);
            double pf_end = omp_get_wtime();
            double pf_new = pf_end - pf_start;
            pf_total += pf_new;
            pf_max = pf_max > pf_new ? pf_max : pf_new;
            pf_min = pf_min < pf_new ? pf_min : pf_new;
            // std::cout << "particle filter time" << pf_new << std::endl;

            double flann_start = omp_get_wtime();
            cumulative_entropy += getCumulativeEntropy(updated_particles);
            double flann_end = omp_get_wtime();
            double flann_new = flann_end - flann_start;
            flann_total += flann_new;
            flann_max = flann_max > flann_new ? flann_max : flann_new;
            flann_min = flann_min < flann_new ? flann_min : flann_new;
            // std::cout << "flann gpu time" << flann_new << std::endl;

            if(j==num_observations_-1){
                entropy = cumulative_entropy;
                if (entropy < min_entropy) {
                    min_entropy = entropy;
                    best_sensing_action = sensing_actions.at(i);
                }

                if (entropy > max_entropy) {
                    max_entropy = entropy;
                }
            }
        }
    }

    std::cout << "particle filter minimum time: " << pf_min << std::endl;
    std::cout << "particle filter maximum time: " << pf_max << std::endl;
    std::cout << "flann minimum time: " << flann_min << std::endl;
    std::cout << "flann maximum time: " << flann_max << std::endl;
    std::cout << "particle filter average time: " << pf_total / 30 << std::endl;
    std::cout << "flann average time: " << flann_total / 30 << std::endl;
    std::cout << "  " << std::endl;

    return best_sensing_action;
}

void ActiveSensing::setNumNearestNeighbors(unsigned int k) {
    num_nearest_neighbors_ = k;
}

void ActiveSensing::setNumObservations(unsigned int n) {
    num_observations_ = n;
}

double ActiveSensing::getConditionalCumulativeEntropy(const std::vector<Particle> &particles,
                                                      unsigned int sensing_action) const {
    double cumulative_entropy = 0;
    Eigen::VectorXd state;
    Eigen::VectorXd observation;

    for (int i = 0; i < num_observations_; i++) {
        state = particle_filter_.importanceSampling(particles).getValue();
        observation = model_.sampleObservation(state, sensing_action);
        std::vector<Particle> updated_particles = particles;
        particle_filter_.updateWeights(updated_particles, sensing_action, observation);
        cumulative_entropy += getCumulativeEntropy(updated_particles);
    }

    return cumulative_entropy;
}

double ActiveSensing::getCumulativeEntropy(const std::vector<Particle> &particles) const {
    double cumulative_entropy = 0;
    std::vector<Particle> new_particles = particles;
    Eigen::VectorXd task_action;

    for (int i = 0; i < horizon_; i++) {
        particle_filter_.normalize(new_particles);
        cumulative_entropy += discount_ * estimateEntropy(new_particles, num_nearest_neighbors_, num_cores_);

        if (i < horizon_ - 1) {
            task_action = getTaskAction(new_particles);
            particle_filter_.propagate(new_particles, task_action);
        }
    }

    return cumulative_entropy;
}

Eigen::VectorXd ActiveSensing::getTaskAction(const std::vector<Particle> &particles) const {
    double max_weight = 0;
    Eigen::VectorXd state;

    for (const Particle &particle : particles) {

        if (particle.getWeight() > max_weight) {
            max_weight = particle.getWeight();
            state = particle.getValue();
        }
    }

    return planner_.policy(state);

}

ActionEntropyActiveSensing::ActionEntropyActiveSensing(Model &model, StateSpacePlanner &planner, ParticleFilter &particle_filter,
                                                       unsigned int horizon, double discount, unsigned int num_observations,
                                                       unsigned int num_nearest_neighbors, unsigned int num_cores) :
    ActiveSensing(model, planner, particle_filter, horizon, discount, num_observations, num_nearest_neighbors, num_cores)
{}

ActionEntropyActiveSensing::~ActionEntropyActiveSensing()
{}

double ActionEntropyActiveSensing::getCumulativeEntropy(const std::vector<Particle> &particles) const {
    double cumulative_entropy = 0;
    std::vector<Particle> new_particles = particles;
    std::vector<Particle> task_action_particles;
    Eigen::VectorXd task_action;

    for (int i = 0; i < horizon_; i++) {
        task_action_particles = getTaskActionParticles(new_particles);
        particle_filter_.normalize(task_action_particles);
        cumulative_entropy += discount_ * estimateEntropy(task_action_particles, num_nearest_neighbors_, num_cores_);

        if (i < horizon_-1) {
            task_action = getTaskAction(new_particles);
            particle_filter_.propagate(new_particles, task_action);
        }
    }

    return cumulative_entropy;
}

std::vector<Particle> ActionEntropyActiveSensing::getTaskActionParticles(const std::vector<Particle> &state_particles) const {
    std::vector<Particle> task_action_particles;
    Eigen::VectorXd task_actions;

    for (const Particle &particle : state_particles) {
        task_actions = planner_.policy(particle.getValue());
        task_action_particles.push_back(Particle(task_actions, particle.getWeight()));
    }
    return task_action_particles;
}

RandomActiveSensing::RandomActiveSensing(Model &model, StateSpacePlanner &planner, ParticleFilter &particle_filter,
                                         unsigned int horizon, double discount) :
        ActiveSensing(model, planner, particle_filter, horizon, discount)
{}

RandomActiveSensing::~RandomActiveSensing()
{}

unsigned int RandomActiveSensing::getSensingAction(const std::vector<Particle> &particles) const
{
    std::vector<unsigned int> sensing_actions = model_.getSensingActions();

    unsigned long rand = rng_->int64();
    unsigned long index = rand % sensing_actions.size();
    return sensing_actions[index];
}
