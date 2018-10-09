/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	default_random_engine def_gen;
	// Set Standard daviations
	double std_x = std[0];
	double std_y = std[1];
	double std_theta = std[2];

	// Set number of Particles
	num_particles = 15;
	// Initialize weights with number of Particles
	weights.resize(num_particles);
	// Create normal distributions for x,y,theta
	normal_distribution<double>dist_x(x, std_x);
	normal_distribution<double>dist_y(y, std_y);
	normal_distribution<double>dist_theta(theta, std_theta);

	// Creating inital Particles
	for (unsigned int i = 0; i < num_particles; i++) {
		Particle particle;
		particle.id = i;
		particle.x = dist_x(def_gen);
		particle.y = dist_y(def_gen);
		particle.theta = dist_theta(def_gen);
		particle.weight = 1.0;

		particles.push_back(particle);
	}
	// Set Initialization flag to true
	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	default_random_engine def_gen_predict;
	double std_x = std_pos[0];
	double std_y = std_pos[1];
	double std_theta = std_pos[2];
	// Create gaussian noise distribution
	normal_distribution<double>dist_noise_x(0.0, std_x);
	normal_distribution<double>dist_noise_y(0.0, std_y);
	normal_distribution<double>dist_noise_theta(0.0, std_theta);
	
	// Predition of each particle
	// Check yaw rate
	for (unsigned int i = 0; i < num_particles; i++) {
		if (fabs(yaw_rate) < 0.0001) {
			particles[i].x += velocity * delta_t * cos(particles[i].theta) + dist_noise_x(def_gen_predict);
			particles[i].y += velocity * delta_t * sin(particles[i].theta) + dist_noise_y(def_gen_predict);
			particles[i].theta += dist_noise_theta(def_gen_predict);
		}
		else {
			particles[i].x += (velocity / yaw_rate) * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta)) + dist_noise_x(def_gen_predict);
			particles[i].y += (velocity / yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t)) + dist_noise_y(def_gen_predict);
			particles[i].theta += yaw_rate * delta_t + dist_noise_theta(def_gen_predict);
		}
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for (unsigned int i = 0; i < observations.size(); i++) {

		// starting with possinle highst distance
		double min_dist = numeric_limits<double>::max();
		// init id of landmark from map placeholder to be associated with the observation
		int closest_id = -1;
		for (unsigned int j = 0; j < predicted.size(); j++) {
			// get distance between current/predicted landmarks
			double distBetween = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
			// find the predicted landmark closest the current observed landmark
			if (distBetween < min_dist) {
				min_dist = distBetween;
				closest_id = predicted[j].id;
			}
		}

		// set the observation's id to the nearest predicted landmark's id
		observations[i].id = closest_id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	// Initialize landmark uncertainities
	double var_x = pow(std_landmark[0], 2);
	double var_y = pow(std_landmark[1], 2);
	double cov_xy = std_landmark[0] * std_landmark[1];
	double normalizer = 2 * M_PI * cov_xy;
	double weights_sum = 0;

	// weight updating
	for (int i = 0; i < num_particles; i++) {
		//Particle particle = particles[i];
		// Landmarks in sensor range
		vector<LandmarkObs> predictions;
		for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {
			double land_x = map_landmarks.landmark_list[j].x_f;
			double land_y = map_landmarks.landmark_list[j].y_f;
			int land_id = map_landmarks.landmark_list[j].id_i;

			// Take the landmarks in sensor range +/- sensor_range
			//if (fabs(land_x - particles[i].x) <= sensor_range && fabs(land_y - particles[i].y) <= sensor_range) {
			predictions.push_back(LandmarkObs{ land_id,land_x,land_y });
			//}
		}
		vector<LandmarkObs>transform_obs;
		for (int j = 0; j < observations.size(); j++) {
			//LandmarkObs observation = observations[j];
			// Homogenous transformation
			double transform_x = particles[i].x + observations[j].x * cos(particles[i].theta) - observations[j].y * sin(particles[i].theta);
			double transform_y = particles[i].y + observations[j].x * sin(particles[i].theta) + observations[j].y * cos(particles[i].theta);
			transform_obs.push_back(LandmarkObs{ observations[j].id,transform_x,transform_y });
		}
		dataAssociation(predictions, transform_obs);
		// reinit weight
		double updated_weight = 1;

		for (unsigned int j = 0; j < transform_obs.size(); j++) {

			// placeholders for observation and associated prediction coordinates
			double o_x, o_y, pr_x, pr_y;
			o_x = transform_obs[j].x;
			o_y = transform_obs[j].y;

			int associated_prediction = transform_obs[j].id;

			// get the x,y coordinates of the prediction associated with the current observation
			for (unsigned int k = 0; k < predictions.size(); k++) {
				if (predictions[k].id == associated_prediction) {
					pr_x = predictions[k].x;
					pr_y = predictions[k].y;
				}
			}
			// Calculate the mean of Multivariate-Gaussian probability density
			double x_diff = pr_x - o_x;
			double y_diff = pr_y - o_y;
			double probability = exp(-0.5 * ((x_diff * x_diff) / var_x + (y_diff * y_diff) / var_y));

			updated_weight *= probability / normalizer;
		}
		// Update particle weight
		particles[i].weight = updated_weight;

		// Update weight vector
		weights[i] = updated_weight;

		// Add weight for normalization purpose
		weights_sum += updated_weight;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	// Set default random engine
	// Vector for new particles
	vector<Particle> new_particles(num_particles);

	// Use discrete distribution to return particles by weight
	random_device rd;
	default_random_engine gen(rd());
	for (int i = 0; i < num_particles; i++) {
		discrete_distribution<int> index(weights.begin(), weights.end());
		new_particles[i] = particles[index(gen)];
	}

	// Replace old particles with the resampled particles
	particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
