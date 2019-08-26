//
// Created by tipakorng on 3/3/16.
//

#include <visualization_msgs/MarkerArray.h>
#include "simulator.h"
#include<iostream>
using namespace std;


Simulator::Simulator(Model &model, BeliefSpacePlanner &planner, unsigned int sensing_interval) :
        model_(model),
        planner_(planner),
        sensing_interval_(sensing_interval)
{
    has_publisher_ = false;
}

Simulator::Simulator(Model &model, BeliefSpacePlanner &planner, ros::NodeHandle *node_handle,
                     unsigned int sensing_interval) :
        model_(model),
        planner_(planner),
        sensing_interval_(sensing_interval),
        node_handle_(node_handle)
{
    publisher_ = node_handle_->advertise<visualization_msgs::Marker>("simulator", 1);
    has_publisher_ = true;
}

Simulator::~Simulator()
{}

void Simulator::initSimulator()
{
    // Clear stuff
    states_.clear();
    task_actions_.clear();
    sensing_actions_.clear();
    observations_.clear();
    cumulative_reward_ = 0;
    // Reset planner
    planner_.reset();
    // Reset active sensing time
    active_sensing_time_ = 0;
    observation_time_=0;
    updatebelief_time_=0;
    total_updatebelief_time_=0;
    taskaction_time_=0;
    predictbelief_time_=0;
    total_predictbelief_time_=0;
}

void Simulator::updateSimulator(unsigned int sensing_action, Eigen::VectorXd observation, Eigen::VectorXd task_action)
{
    Eigen::VectorXd new_state = model_.sampleNextState(states_.back(), task_action);
    states_.push_back(new_state);
    sensing_actions_.push_back(sensing_action);
    observations_.push_back(observation);
    task_actions_.push_back(task_action);
    cumulative_reward_ += model_.getReward(new_state, task_action);
}

void Simulator::updateSimulator(Eigen::VectorXd task_action)
{
    Eigen::VectorXd new_state = model_.sampleNextState(states_.back(), task_action);
    states_.push_back(new_state);
    task_actions_.push_back(task_action);
    cumulative_reward_ += model_.getReward(new_state, task_action);
}

void Simulator::simulate(const Eigen::VectorXd &init_state, unsigned int num_steps, unsigned int verbosity)
{
    initSimulator();
    unsigned int n = 0;
    unsigned int sensing_action;
    double active_sensing_time = 0;
    double observation_time = 0;
    double updatebelief_time =0;
    double taskaction_time = 0;
    double predictbelief_time = 0;
    double total_updatebelief_time = 0;
    double total_predictbelief_time = 0;

    double avg_observation_time = 0;
    double avg_updatebelief_time = 0;
    double avg_total_updatebelief_time=0;
    double avg_taskaction_time = 0;
    double avg_predictbelief_time = 0;
    double avg_total_predictbelief_time = 0;
    Eigen::VectorXd observation;
    Eigen::VectorXd task_action;
    states_.push_back(init_state);
    std::chrono::high_resolution_clock::time_point active_sensing_start;
    std::chrono::high_resolution_clock::time_point active_sensing_finish;
    std::chrono::high_resolution_clock::time_point observation_start;
    std::chrono::high_resolution_clock::time_point observation_finish;
    std::chrono::high_resolution_clock::time_point updatebelief_start;
    std::chrono::high_resolution_clock::time_point updatebelief_finish;
    std::chrono::high_resolution_clock::time_point taskaction_start;
    std::chrono::high_resolution_clock::time_point taskaction_finish;
    std::chrono::high_resolution_clock::time_point predictbelief_start;
    std::chrono::high_resolution_clock::time_point predictbelief_finish;
    std::chrono::duration<double> active_sensing_elapsed_time;
    std::chrono::duration<double> observation_elapsed_time;
    std::chrono::duration<double> updatebelief_elapsed_time;
    std::chrono::duration<double> taskaction_elapsed_time;
    std::chrono::duration<double> predictbelief_elapsed_time;
    std::chrono::duration<double> total_updatebelief_elapsed_time;
    std::chrono::duration<double> total_predictbelief_elapsed_time;

    if (verbosity > 0)
        std::cout << "state = \n" << states_.back().transpose() << std::endl;

    planner_.publishParticles();

    while (!model_.isTerminal(states_.back()) && n < num_steps)
    {
        // Normalize particle weights.
        planner_.normalizeBelief();

        // If sensing is allowed in this step.
        if (n % (sensing_interval_ + 1) == 0)
        {
            // Get sensing action.
            active_sensing_start = std::chrono::high_resolution_clock::now();

            sensing_action = planner_.getSensingAction();

            active_sensing_finish = std::chrono::high_resolution_clock::now();

            active_sensing_elapsed_time = std::chrono::duration_cast<std::chrono::duration<double> >
                    (active_sensing_finish-active_sensing_start);
            active_sensing_time += active_sensing_elapsed_time.count();

            // Update the belief.
            observation_start = std::chrono::high_resolution_clock::now();
	
            observation = model_.sampleObservation(states_.back(), sensing_action);
	    
            observation_finish = std::chrono::high_resolution_clock::now();

	    observation_elapsed_time = std::chrono::duration_cast<std::chrono::duration<double> >
                    (observation_finish-observation_start);
            observation_time += observation_elapsed_time.count();
	
	    updatebelief_start = std::chrono::high_resolution_clock::now();
		

            planner_.updateBelief(sensing_action, observation);
	    
            updatebelief_finish = std::chrono::high_resolution_clock::now();

	    updatebelief_elapsed_time = std::chrono::duration_cast<std::chrono::duration<double> >
                    (updatebelief_finish-updatebelief_start);
	    updatebelief_time += updatebelief_elapsed_time.count();

	    total_updatebelief_elapsed_time = std::chrono::duration_cast<std::chrono::duration<double> >(updatebelief_finish-observation_start);
	total_updatebelief_time += total_updatebelief_elapsed_time.count();

	    
            


            // Predict the new belief.
	    taskaction_start = std::chrono::high_resolution_clock::now();
            task_action = planner_.getTaskAction();
	    taskaction_finish = std::chrono::high_resolution_clock::now();

	    taskaction_elapsed_time = std::chrono::duration_cast<std::chrono::duration<double> >(taskaction_finish-taskaction_start);
	    taskaction_time +=taskaction_elapsed_time.count();
	    
            predictbelief_start = std::chrono::high_resolution_clock::now();
            planner_.predictBelief(task_action);
            updateSimulator(sensing_action, observation, task_action);
	    predictbelief_finish = std::chrono::high_resolution_clock::now();

	    predictbelief_elapsed_time = std::chrono::duration_cast<std::chrono::duration<double> >(predictbelief_finish-predictbelief_start);
	    predictbelief_time +=predictbelief_elapsed_time.count();

	    total_predictbelief_elapsed_time=std::chrono::duration_cast<std::chrono::duration<double> >(predictbelief_finish-taskaction_start);
	    total_predictbelief_time += total_predictbelief_elapsed_time.count();
            

            if (verbosity > 0)
            {
                std::cout << "n = " << n << std::endl;
                std::cout << "sensing_action = " << sensing_action << std::endl;
                std::cout << "observation = " << observation.transpose() << std::endl;
                std::cout << "most_likely_state = " << planner_.getMaximumLikelihoodState().transpose() << std::endl;
                std::cout << "task_action = " << task_action.transpose() << std::endl;
                std::cout << "state = " << states_.back().transpose() << std::endl;
            }
        }

        // Otherwise, set sensing action to DO NOTHING.
        else
        {
            task_action = planner_.getTaskAction();
            planner_.predictBelief(task_action);
            updateSimulator(task_action);

            if (verbosity > 0)
            {
                std::cout << "n = " << n << std::endl;
                std::cout << "sensing_action = " << "n/a" << std::endl;
                std::cout << "observation = " << "n/a" << std::endl;
                std::cout << "most_likely_state = " << planner_.getMaximumLikelihoodState().transpose() << std::endl;
                std::cout << "task_action = " << task_action.transpose() << std::endl;
                std::cout << "state = " << states_.back().transpose() << std::endl;
            }
        }

        planner_.publishParticles();
        model_.publishMap();

        if (has_publisher_)
        {
            publishState();
        }

        n++;
    }

    int num_sensing_steps = (n + 1) / (sensing_interval_ + 1);

    if (num_sensing_steps > 0)
{
        active_sensing_time_ = active_sensing_time / num_sensing_steps;

	avg_observation_time = observation_time/num_sensing_steps;
        observation_time_=avg_observation_time;

        avg_updatebelief_time = updatebelief_time/num_sensing_steps;
        updatebelief_time_ = avg_updatebelief_time;

	avg_total_updatebelief_time = total_updatebelief_time/num_sensing_steps;
        total_updatebelief_time_ = avg_total_updatebelief_time;

	avg_taskaction_time = taskaction_time/num_sensing_steps;
        taskaction_time_=avg_taskaction_time;

	avg_predictbelief_time = predictbelief_time/num_sensing_steps;
        predictbelief_time_ = avg_predictbelief_time;

	avg_total_predictbelief_time = total_predictbelief_time/num_sensing_steps;
        total_predictbelief_time_=avg_total_predictbelief_time;
	        
}
    else
{
        active_sensing_time_ = 0;
}

cout<<"active_sensing_time is "<<active_sensing_time_<<endl;
cout<<"avg_observation_time is "<<avg_observation_time<<endl;
cout<<"avg_updatebelief_time is "<<avg_updatebelief_time<<endl;
cout<<"avg_total_updatebelief_time is "<<avg_total_updatebelief_time<<endl;
cout<<"avg_taskaction_time is "<<avg_taskaction_time<<endl;
cout<<"avg_predictbelief_time is "<<avg_predictbelief_time<<endl;
cout<<"avg_total_predictbelief_time is "<<avg_total_predictbelief_time<<endl;

  
}

std::vector<Eigen::VectorXd> Simulator::getStates()
{
    return states_;
}

std::vector<unsigned int> Simulator::getSensingActions()
{
    return sensing_actions_;
}

std::vector<Eigen::VectorXd> Simulator::getTaskActions()
{
    return task_actions_;
}

std::vector<Eigen::VectorXd> Simulator::getObservations()
{
    return observations_;
}

double Simulator::getCumulativeReward()
{
    return cumulative_reward_;
}

double Simulator::getAverageActiveSensingTime()
{
    return active_sensing_time_;
}

double Simulator::getAvgObservationTime()
{
  return observation_time_;
}
double Simulator::getAvgUpdatebeliefTime()
{
 return updatebelief_time_;
}
double Simulator::getAvgTotalUpdatebeliefTime()
{
 return total_updatebelief_time_;
}
double Simulator::getAvgTaskactionTime()
{
return taskaction_time_;
}
double Simulator::getAvgPredictbeliefTime()
{
return predictbelief_time_;
}
double Simulator::getAvgTotalPredictbeliefTime()
{
return total_predictbelief_time_;
}


void Simulator::publishState()
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

        uint32_t shape = visualization_msgs::Marker::CUBE;
        marker.header.frame_id = "world";
        marker.header.stamp = ros::Time::now();
        marker.ns = "basic_shapes";
        marker.id = 0;
        marker.type = shape;
        marker.action = visualization_msgs::Marker::ADD;

        Eigen::VectorXd state = states_.back();

        model_.fillMarker(state, marker);

        marker.color.r = 0.0;
        marker.color.g = 0.0;
        marker.color.b = 1.0;
        marker.color.a = 1.0;

        marker.lifetime = ros::Duration();

        publisher_.publish(marker);
    }
}
