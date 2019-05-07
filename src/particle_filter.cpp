//
// Created by tipakorng on 11/23/15.
//

#include <visualization_msgs/MarkerArray.h>
#include "../include/particle_filter.h"


Particle::Particle(const Eigen::VectorXd &value, double weight) :
    value_(value),
    dimension_(static_cast<unsigned int>(value.size()))
{
    weight_ = weight;
}

Particle::~Particle() {}

void Particle::setValue(const Eigen::VectorXd &value)
{
    value_ = value;
}

void Particle::setWeight(double weight)
{
    weight_ = weight;
}

unsigned int Particle::getDimension() const
{
    return dimension_;
}

Eigen::VectorXd Particle::getValue() const
{
    return value_;
}

double Particle::getWeight() const
{
    return weight_;
}

ParticleFilter::ParticleFilter(const Model &model, unsigned int num_particles):
    model_(model),
    numParticles_(num_particles),
    rng_(Rng(0)),
    node_handle_(nullptr)
{
    has_publisher_ = false;
    initParticles();
}

ParticleFilter::ParticleFilter(const Model &model, unsigned int num_particles, ros::NodeHandle *node_handle):
        model_(model),
        numParticles_(num_particles),
        rng_(Rng(0)),
        node_handle_(node_handle)
{
    publisher_ = node_handle_->advertise<visualization_msgs::MarkerArray>("particle_filter", 1);
    has_publisher_ = true;
    initParticles();
}

ParticleFilter::~ParticleFilter() {}

void ParticleFilter::initParticles()
{
    particles_.clear();
    int dimension = model_.getStateSize();
    double weight = 1.0 / numParticles_;

    for(int i  = 0; i < numParticles_; ++i)
    {
        Eigen::VectorXd value(dimension);
        value = model_.sampleInitState();
        Particle particle(value, weight);
        particles_.push_back(particle);
    }

}

void ParticleFilter::updateWeights(unsigned int sensing_action, const Eigen::VectorXd &observation)
{
    updateWeights(particles_, sensing_action, observation);
}

void ParticleFilter::updateWeights(std::vector<Particle> &particles, unsigned int sensing_action,
                                   const Eigen::VectorXd &observation) const
{

    double max_weight = 0;

    for (Particle &particle: particles)
    {
        double weight = model_.getObservationProbability(particle.getValue(), sensing_action, observation);
        particle.setWeight(weight);

        if (weight > max_weight)
            max_weight = weight;
    }

//    if (max_weight < std::numeric_limits<double>::epsilon())
//        resetWeights(particles);
//    else
//        normalize(particles);

}

void ParticleFilter::resample()
{
    resample(particles_);
}

void ParticleFilter::resample(std::vector<Particle> &particles)
{
    // Copy particles.
    std::vector<Particle> old_particles = particles;
    unsigned long num_particles = particles.size();
    particles.clear();

    double sum_old_weights = 0;

    for (Particle particle : old_particles)
    {
        sum_old_weights += particle.getWeight();
    }

    // Low variance sampling from Thrun's Probabilistic Robotics.
    double r = rng_.doub() * sum_old_weights;
    double c = old_particles[0].getWeight();
    unsigned long i = 0;
    double u;

    for (unsigned long j = 0; j < num_particles; j++)
    {
        u = r + j / num_particles;

        while (u > c)
        {
            if (i < old_particles.size())
                i += 1;
            else
                i = 0;

            c += old_particles[i].getWeight();
        }

        particles.push_back(old_particles[i]);
    }
}

void ParticleFilter::propagate(const Eigen::VectorXd task_action)
{
    propagate(particles_, task_action);
}

void ParticleFilter::propagate(std::vector<Particle> &particles, const Eigen::VectorXd &task_action) const
{

    for (Particle &particle: particles)
    {
        Eigen::VectorXd new_value = model_.sampleNextState(particle.getValue(), task_action);
        particle.setValue(new_value);
    }

}

void ParticleFilter::update(const Eigen::VectorXd task_action, unsigned int sensing_action,
                            const Eigen::VectorXd observation)
{
    update(particles_, task_action, sensing_action, observation);
}

void ParticleFilter::update(std::vector<Particle> &particles, const Eigen::VectorXd task_action,
                            unsigned int sensing_action, const Eigen::VectorXd &observation) const
{
    updateWeights(particles, sensing_action, observation);
    propagate(particles, task_action);
}

void ParticleFilter::normalize()
{
    normalize(particles_);
}

void ParticleFilter::normalize(std::vector<Particle> &particles) const
{
    /* Find sum of all weights */
    double sum_weight = 0.0;

    for (Particle &particle: particles)
    {
        sum_weight += particle.getWeight();
    }

    /* Normalize weights */
    for (Particle &particle: particles)
    {
        particle.setWeight(particle.getWeight() / sum_weight);
    }
}

Particle ParticleFilter::importanceSampling(const std::vector<Particle> &particles)
{

    /* Build CDF for sampling */
    std::vector<double> cdf(numParticles_, 0);
    double sum_weight = 0;

    for(int i = 0; i < numParticles_; ++i)
    {
        sum_weight += particles[i].getWeight();
        cdf[i] = sum_weight;
    }

    /* Sample from the CDF */
    double random_number = cdf.back() * rng_.doub();
    std::vector<double>::iterator sample_idx = lower_bound(cdf.begin(), cdf.end(), random_number);
    int idx = static_cast<int>(sample_idx-cdf.begin());
    return particles[idx];
}

unsigned int ParticleFilter::getNumParticles()
{
    return numParticles_;
}

std::vector<Particle> ParticleFilter::getParticles()
{
    return particles_;
}

Particle ParticleFilter::getBestParticle() const
{
    double max_weight = 0;
    int best_particle_id = -1;

    for (int i = 0; i < particles_.size(); i++)
    {
        if (particles_[i].getWeight() > max_weight)
        {
            max_weight = particles_[i].getWeight();
            best_particle_id = i;
        }
    }

    if (max_weight == 0)
    {
        return particles_[0];
    }

    return particles_[best_particle_id];
}

void ParticleFilter::publish()
{
    publish(particles_);
}

void ParticleFilter::publish(const std::vector<Particle> &particles)
{
    if (has_publisher_)
    {
        tf::Transform transform;
        transform.setOrigin(tf::Vector3(0, 0, 0));
        tf::Quaternion q;
        q.setRPY(0, 0, 0);
        transform.setRotation(q);
        tf_broadcaster_.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "world", "my_frame"));

        visualization_msgs::Marker marker;
        visualization_msgs::MarkerArray marker_array;

        for (int i = 0; i < particles.size(); i++)
        {
            marker.header.frame_id = "world";
            marker.header.stamp = ros::Time::now();
            marker.ns = "basic_shapes";
            marker.id = i;

            Eigen::VectorXd state = particles[i].getValue();

            model_.fillMarker(state, marker);

            marker.lifetime = ros::Duration();

            marker.color.r = 0.0;
            marker.color.g = 1.0;
            marker.color.b = 0.0;
            marker.color.a = 0.2;

            marker_array.markers.push_back(marker);

        }

        publisher_.publish(marker_array);
        ros::Duration(1e-3).sleep();
    }
}

bool ParticleFilter::isPublishable() const
{
    return has_publisher_;
}
