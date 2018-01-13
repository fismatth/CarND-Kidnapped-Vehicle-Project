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

ostream& operator<<(ostream& os, const Particle& p) {
	os << "x = " << p.x << ", y = " << p.y << ", theta = " << p.theta
			<< "; weight = " << p.weight << endl;
	return os;
}

ostream& operator<<(ostream& os, const LandmarkObs& l) {
	os << "x = " << l.x << ", y = " << l.y << ", id = " << l.id << endl;
	return os;
}

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 1000;
	particles.resize(num_particles);
	weights.resize(num_particles);

	default_random_engine gen;

	// Extract standard deviations for x, y, and theta
	double std_x, std_y, std_theta;
	std_x = std[0];
	std_y = std[1];
	std_theta = std[2];

	// Create normal distributions for x, y and theta
	normal_distribution<double> dist_x(x, std_x);
	normal_distribution<double> dist_y(y, std_y);
	normal_distribution<double> dist_theta(theta, std_theta);

	for (int i = 0; i < num_particles; ++i) {
		double sample_x, sample_y, sample_theta;
		sample_x = dist_x(gen);
		sample_y = dist_y(gen);
		sample_theta = dist_theta(gen);

		Particle p;
		p.x = sample_x;
		p.y = sample_y;
		p.theta = sample_theta;
		p.id = i;
		p.weight = 1.0;
		particles[i] = p;
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
		double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	default_random_engine gen;

	// Extract standard deviations for x, y, and theta
	double std_x, std_y, std_theta;
	std_x = std_pos[0];
	std_y = std_pos[1];
	std_theta = std_pos[2];

	// Create normal distributions for x, y and theta
	normal_distribution<double> dist_x(0.0, std_x);
	normal_distribution<double> dist_y(0.0, std_y);
	normal_distribution<double> dist_theta(0.0, std_theta);

	for (int i = 0; i < num_particles; ++i) {
		double sample_x, sample_y, sample_theta;
		sample_x = dist_x(gen);
		sample_y = dist_y(gen);
		sample_theta = dist_theta(gen);
		double theta = particles[i].theta;

		if (fabs(yaw_rate) > 1e-4) {
			double yaw_rate_dt = yaw_rate * delta_t;
			particles[i].x += (velocity / yaw_rate)
					* (sin(theta + yaw_rate_dt) - sin(theta)) + sample_x;
			particles[i].y += (velocity / yaw_rate)
					* (cos(theta) - cos(theta + yaw_rate_dt)) + sample_y;
			particles[i].theta += yaw_rate_dt + sample_theta;
		} else {
			particles[i].x += velocity * delta_t * cos(theta) + sample_x;
			particles[i].y += velocity * delta_t * sin(theta) + sample_y;
			particles[i].theta += yaw_rate * delta_t + sample_theta;
		}
	}
}

double distance(const LandmarkObs& l1, const LandmarkObs& l2) {
	double dist_x = l1.x - l2.x;
	double dist_y = l1.y - l2.y;
	return sqrt(dist_x * dist_x + dist_y * dist_y);
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted,
		std::vector<LandmarkObs>& observations, int num_predicted) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for (int i = 0; i < observations.size(); ++i) {
		double min_dist = distance(observations[i], predicted[0]);
		int min_idx = 0;
		for (int j = 1; j < num_predicted; ++j) {
			double dist = distance(observations[i], predicted[j]);
			if (dist < min_dist) {
				min_dist = dist;
				min_idx = j;
			}
		}
		observations[i].id = min_idx;
	}
}

void getPredicted(const Particle& p, const Map& map_landmarks,
		double sensor_range, std::vector<LandmarkObs>& predicted,
		int& num_predicted) {
	// avoid resizing ((de)allocating memory)
	num_predicted = 0;
	// transform each landmark from map to vehicle coordinates
	for (int i = 0; i < map_landmarks.landmark_list.size(); ++i) {
		double x_map = map_landmarks.landmark_list[i].x_f;
		double y_map = map_landmarks.landmark_list[i].y_f;
		int id = map_landmarks.landmark_list[i].id_i;
		double theta = p.theta;
		// inverse operation (world -> vehicle): do translation first
		double x_t = x_map - p.x;
		double y_t = y_map - p.y;
		// rotate by -theta
		double x_veh = x_t * cos(-theta) - y_t * sin(-theta);
		double y_veh = x_t * sin(-theta) + y_t * cos(-theta);

		if (dist(x_veh, y_veh, 0.0, 0.0) <= sensor_range) {
			predicted[num_predicted].x = x_veh;
			predicted[num_predicted].y = y_veh;
			predicted[num_predicted].id = id;
			++num_predicted;
		}
	}
}

double Gaussian2D(double mu[], double sigma[], double x[]) {
	double x_mu_0 = mu[0] - x[0];
	double x_mu_1 = mu[1] - x[1];
	double sigma2_0 = sigma[0] * sigma[0];
	double sigma2_1 = sigma[1] * sigma[1];
	double exponent = -0.5
			* (x_mu_0 * x_mu_0 / sigma2_0 + x_mu_1 * x_mu_1 / sigma2_1);
	double normalizer = 2.0 * M_PI * sigma[0] * sigma[1];
	return exp(exponent) / normalizer;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		const std::vector<LandmarkObs> &observations,
		const Map &map_landmarks) {
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
	std::vector<LandmarkObs> predicted(map_landmarks.landmark_list.size());
	int num_predicted;

	for (int i = 0; i < num_particles; ++i) {
		getPredicted(particles[i], map_landmarks, sensor_range, predicted,
				num_predicted);
		// observations declared as const ref, dataAssociation changes the associated id -> copy
		std::vector<LandmarkObs> matched_observations(observations);
		dataAssociation(predicted, matched_observations, num_predicted);
		double prob = 1.0;
		particles[i].associations.resize(matched_observations.size());
		particles[i].sense_x.resize(matched_observations.size());
		particles[i].sense_y.resize(matched_observations.size());

		for (int j = 0; j < matched_observations.size(); ++j) {
			int nn_idx = matched_observations[j].id;
			double mu[2];
			mu[0] = predicted[nn_idx].x;
			mu[1] = predicted[nn_idx].y;
			double x[2];
			x[0] = matched_observations[j].x;
			x[1] = matched_observations[j].y;
			double prob1O = Gaussian2D(mu, std_landmark, x);
			prob *= prob1O;
			// only for visualization
			particles[i].associations[j] = predicted[nn_idx].id;
			// transformation vehicle -> world
			double theta = particles[i].theta;
			particles[i].sense_x[j]= mu[0] * cos(theta) - mu[1] * sin(theta) + particles[i].x;
			particles[i].sense_y[j] = mu[0] * sin(theta) + mu[1] * cos(theta) + particles[i].y;
		}

		particles[i].weight = prob;
		weights[i] = prob;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	std::vector<Particle> resampled(num_particles);
	default_random_engine gen;
	double sum_weights = 0.0;
	for (int i = 0; i < num_particles; ++i) {
		sum_weights += weights[i];
	}

	uniform_real_distribution<double> uni(0.0, sum_weights);
	uniform_real_distribution<double> uni_np(0.0, num_particles);

	int index = int(uni_np(gen));
	double beta = 0.0;
	for (int i = 0; i < num_particles; ++i) {
		beta += uni(gen);
		while (weights[index] < beta) {
			beta -= weights[index];
			index++;
			index %= num_particles;
		}
		resampled[i] = particles[index];
	}
	std::swap(particles, resampled);
	for (int i = 0; i < num_particles; ++i) {
		weights[i] = particles[i].weight;
	}
}

Particle ParticleFilter::SetAssociations(Particle& particle,
		const std::vector<int>& associations,
		const std::vector<double>& sense_x,
		const std::vector<double>& sense_y) {
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	particle.associations = associations;
	particle.sense_x = sense_x;
	particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
	vector<int> v = best.associations;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1);  // get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseX(Particle best) {
	vector<double> v = best.sense_x;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1);  // get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseY(Particle best) {
	vector<double> v = best.sense_y;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1);  // get rid of the trailing space
	return s;
}
