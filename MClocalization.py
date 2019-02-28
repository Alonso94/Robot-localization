#!/usr/bin/env python
import matplotlib.pyplot as plt
import scipy.optimize

import roslib
import rospy

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from localization.msg import Coordinates
from threading import Lock
import numpy as np
import tf.transformations as tr

def cvt_local2global(local_point, sc_point):
    point = np.zeros((3, len(local_point)))
    x, y, a = local_point.T
    X, Y, A = sc_point
    point[0] = x * np.cos(A) - y * np.sin(A) + X
    point[1] = x * np.sin(A) + y * np.cos(A) + Y
    point[2] = a + A
    return point.T


def cvt_global2local(global_point, sc_point):
    point = np.zeros((3, len(global_point)))
    x, y, a = global_point.T
    X, Y, A = sc_point
    point[0] = x * np.cos(A) + y * np.sin(A) - X * np.cos(A) - Y * np.sin(A)
    point[1] = -x * np.sin(A) + y * np.cos(A) + X * np.sin(A) - Y * np.cos(A)
    point[2] = a - A
    return point.T


class ParticleFilter(object):
    def __init__(self):

        self.num_particles=100
        self.start_x=250
        self.start_y=1360
        #second robot x=250 y=1660
        self.start_theta=0.0

        self.beacons=[[3094,1000],[-94,50],[-94,1950]]
        # or
        #self.beacons=[[3094,50],[3094,1950],[-94,1000]]
        self.beac_dist_thresh=300
        self.num_is_near_thresh=0.1
        self.sense_noise=50



        self.distance_noise=5
        self.angle_noise=0.04
        xrand = np.random.normal(self.start_x, self.distance_noise, self.num_particles)
        yrand = np.random.normal(self.start_y, self.distance_noise, self.num_particles)
        trand = np.random.normal(0.0, self.angle_noise, self.num_particles)
        self.particles=np.array([xrand,yrand,trand])
        self.weights=[1]*self.num_particles

    @staticmethod
    def gaus(x, mu=0, sigma=1):
        return np.exp(- ((x - mu) ** 2) / (sigma ** 2) / 2.0) / np.sqrt(2.0 * np.pi * (sigma ** 2))

    def move_particles(self,dx,dy,dtheta):

        x_noise = np.random.normal(0, self.distance_noise, self.num_particles)
        move_x=dx+x_noise
        y_noise = np.random.normal(0, self.distance_noise, self.num_particles)
        move_y=dy+y_noise
        angle_noise = np.random.normal(0, self.angle_noise, self.num_particles)
        move_theta=dtheta+angle_noise
        move_point=np.array([move_x,move_y,move_theta]).T
        self.particles = cvt_local2global(move_point, self.particles)
        self.particles[self.particles[:, 1] > 2000 - 100, 1] = 2000 - 100
        self.particles[self.particles[:, 1] < 0, 1] = 0
        self.particles[self.particles[:, 0] > 3000 - 100, 0] = 3000 - 100
        self.particles[self.particles[:, 0] < 100, 0] = 100


    def calc_weights(self,ranges,intens,angles):

        x_coords=ranges*np.cos(angles)
        y_coords=ranges*np.sin(angles)
        landmarks = np.array([x_coords, y_coords]).T
        particles=self.particles.copy()

        """Calculate particle weights based on their pose and landmards"""
        # determines 3 beacon positions (x,y) for every particle in it's local coords
        res = self.beacons[np.newaxis, :, :] - particles[:, np.newaxis, :2]
        X = (res[:, :, 0] * np.cos(particles[:, 2])[:, np.newaxis]
             + res[:, :, 1] * np.sin(particles[:, 2])[:, np.newaxis])
        Y = (-res[:, :, 0] * np.sin(particles[:, 2])[:, np.newaxis]
             + res[:, :, 1] * np.cos(particles[:, 2])[:, np.newaxis])
        beacons = np.concatenate((X[:, :, np.newaxis], Y[:, :, np.newaxis]), axis=2)

        # find closest beacons to landmark
        dist_from_beacon = np.linalg.norm(beacons[:, np.newaxis, :, :] -
                                          landmarks[np.newaxis, :, np.newaxis, :], axis=3)
        ind_closest_beacon = np.argmin(dist_from_beacon, axis=2)
        closest_beacons = beacons[np.arange(beacons.shape[0])[:, np.newaxis], ind_closest_beacon]

        # Calculate cos of angle between landmark, beacon and particle
        scalar_product = np.sum((closest_beacons - particles[:, np.newaxis, :2]) *
                                (closest_beacons - landmarks[np.newaxis, :, :2]), axis=2)
        dist_from_closest_beacon_to_particle = np.linalg.norm(closest_beacons - particles[:, np.newaxis, :2], axis=2)
        dist_from_closest_beacon_to_landmark = np.linalg.norm(closest_beacons - landmarks[np.newaxis, :, :2], axis=2)
        cos_landmarks = scalar_product / \
                        np.where(dist_from_closest_beacon_to_landmark, dist_from_closest_beacon_to_landmark, 1) / \
                        np.where(dist_from_closest_beacon_to_particle, dist_from_closest_beacon_to_particle, 1)

        # From local minimum
        res = closest_beacons - landmarks[np.newaxis, :, :2]
        X = (res[:, :, 0] * np.cos(particles[:, 2])[:, np.newaxis]
             - res[:, :, 1] * np.sin(particles[:, 2])[:, np.newaxis])
        Y = (res[:, :, 0] * np.sin(particles[:, 2])[:, np.newaxis]
             + res[:, :, 1] * np.cos(particles[:, 2])[:, np.newaxis])
        delta_beacon_landmark = np.concatenate((X[:, :, np.newaxis], Y[:, :, np.newaxis]), axis=2)
        is_bad_beacon_landmark_x = \
            (ind_closest_beacon == 0) * (delta_beacon_landmark[:, :, 0] < 0) + \
            (ind_closest_beacon == 1) * (delta_beacon_landmark[:, :, 0] > 0) + \
            (ind_closest_beacon == 2) * (delta_beacon_landmark[:, :, 0] > 0)

        is_bad_beacon_landmark_y = \
            (ind_closest_beacon == 1) * (delta_beacon_landmark[:, :, 1] < 0) + \
            (ind_closest_beacon == 2) * (delta_beacon_landmark[:, :, 1] > 0)
        # Calculate errors of position of landmarks
        errors = np.abs(dist_from_closest_beacon_to_landmark - 48) ** 2 + \
                 4 * np.abs(1 - cos_landmarks) ** 2 + \
                 0.5 * is_bad_beacon_landmark_x * delta_beacon_landmark[:, :, 0] ** 2 + \
                 0.5 * is_bad_beacon_landmark_y * delta_beacon_landmark[:, :, 1] ** 2

        # too far real beacons go away: non valid
        is_near = dist_from_closest_beacon_to_landmark < self.beac_dist_thresh
        is_near_sum = np.sum(is_near, axis=0)
        is_near_or = (is_near_sum > is_near.shape[0] * self.num_is_near_thresh)
        self.is_near_or = is_near_or
        num_good_landmarks = np.sum(is_near_or)
        sum_errors = np.sum(errors * is_near_or[np.newaxis, :], axis=1)

        if num_good_landmarks:
            self.cost_function = np.sqrt(sum_errors) / num_good_landmarks
        else:
            self.cost_function = np.ones(sum_errors.shape[0]) * 1000

        is_beacon_seeing = np.ones(3) * False
        for i in range(3):
            is_beacon_seeing[i] = np.any(i == ind_closest_beacon[:, is_near_or])
        self.num_seeing_beacons = np.sum(is_beacon_seeing)

        weights = self.gaus(self.cost_function, mu=0, sigma=self.sense_noise)
        if np.sum(weights) > 0:
            weights /= np.sum(weights)
        else:
            weights = np.ones(particles.shape[0], dtype=np.float) / particles.shape[0]

        self.particles=particles
        self.weights=weights

    def resample_and_update(self):

        n = self.num_particles
        weigths = np.array(self.weights)
        indices = []
        C = np.append([0.], np.cumsum(weigths))
        j = 0
        u0 = (np.random.rand() + np.arange(n)) / n
        for u in u0:
            while j < len(C) and u > C[j]:
                j += 1
            indices += [j - 1]

        self.particles=self.particles[indices,:]

    def calc_pose(self):
        x = np.mean(self.particles[:, 0])
        y = np.mean(self.particles[:, 1])
        zero_elem = self.particles[0, 2]
        # this helps if particles angles are close to 0 or 2*pi
        temporary = ((self.particles[:, 2] - zero_elem + np.pi) % (2.0 * np.pi)) + zero_elem - np.pi
        angle = np.mean(temporary) % (2.0 * np.pi)
        return np.array((x, y, angle))

class Montecarlo(object):
    def __init__(self):

        self.curr_odom = None
        self.last_odom = None

        self.dx = 0.0
        self.dy = 0.0
        self.dtheta = 0.0

        self.pf=ParticleFilter()

        rospy.init_node('localization', anonymous=True)
        self.position_pub = rospy.Publisher('/robot_position', Coordinates, queue_size=1)
        self.laser_sub = rospy.Subscriber('/scan', LaserScan, self.laser_callback, queue_size=1)
        self.odom_sub = rospy.Subscriber('/real', Odometry, self.odom_callback, queue_size=1)
        self.mutex = Lock()


    def handle_observation(self,laser_scan_msg):
        # prediction,weight updat and resampling
        # predict errors
        max_range = 3800
        min_inten = 1200
        ranges = list(laser_scan_msg.ranges)
        ranges=[x*1000 for x in ranges]
        intens = list(laser_scan_msg.intensities)
        angles = np.arange(laser_scan_msg.angle_min, laser_scan_msg.angle_max, laser_scan_msg.angle_increment)

        x = []
        y = []
        inten = []
        for i in range(len(ranges)):
            if (ranges[i] < max_range) and (intens > min_inten):
                x.append(ranges[i] * np.cos(angles[i]))
                y.append(ranges[i] * np.sin(angles[i]))
                inten.append(intens[i])

        self.pf.move_particles(self.dx,self.dy,self.dtheta)
        self.pf.calc_weights(ranges,intens,angles)
        self.pf.resample_and_update()
        res=self.pf.calc_pose()

        self.position_pub.publish(res)

    def laser_callback(self,msg):

        self.mutex.acquire()
        self.handle_observation(msg)
        self.mutex.release()

    def handle_odometry(self,odom):

        self.last_odom = self.curr_odom
        self.curr_odom = odom
        if self.last_odom:
            p_curr = np.array([self.curr_odom.pose.pose.position.x,
                               self.curr_odom.pose.pose.position.y,
                               self.curr_odom.pose.pose.position.z])
            p_last = np.array([self.last_odom.pose.pose.position.x,
                               self.last_odom.pose.pose.position.y,
                               self.last_odom.pose.pose.position.z])
            q_curr = np.array([self.curr_odom.pose.pose.orientation.x,
                               self.curr_odom.pose.pose.orientation.y,
                               self.curr_odom.pose.pose.orientation.z,
                               self.curr_odom.pose.pose.orientation.w])
            q_last = np.array([self.last_odom.pose.pose.orientation.x,
                               self.last_odom.pose.pose.orientation.y,
                               self.last_odom.pose.pose.orientation.z,
                               self.last_odom.pose.pose.orientation.w])
            rot_last = tr.quaternion_matrix(q_last)[0:3, 0:3]
            p_last_curr = rot_last.transpose().dot(p_curr - p_last)
            q_last_curr = tr.quaternion_multiply(tr.quaternion_inverse(q_last), q_curr)
            _, _, diff = tr.euler_from_quaternion(q_last_curr)
            self.dtheta += diff
            self.dx += (p_last_curr[0]) * 1000
            self.dy += (p_last_curr[1]) * 1000

    def odom_callback(self,msg):
        self.mutex.acquire()
        self.handle_odometry(msg)
        self.mutex.release()


MCL=Montecarlo()
rospy.spin()