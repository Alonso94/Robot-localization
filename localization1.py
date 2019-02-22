#!/usr/bin/env python
import math

import rospy
import numpy as np
import matplotlib.pyplot as plt
from sensor_msgs.msg import LaserScan
import scipy.optimize

import actionlib
import time

import ros_stm_driver.srv
import struct

from commands_lists.big_commands import dictCommands

import numpy as np
import tf.transformations as tr
from math import sqrt, cos, sin, pi, atan2, log, exp
import random
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from threading import Lock


class Particale(object):
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta

class Beacon(object):
    def __init__(self,x,y):
        self.x=x
        self.y=y

class ParticleFilter(object):
    def __init__(self, num_p):
        # number of particles
        self.num_p = num_p

        # workspace
        self.xmin = 0
        self.xmax = 3000
        self.ymin = 0
        self.ymax = 2000

        # odometry
        self.curr_odom = None
        self.last_odom = None

        # relative motion since the last time particles were updated
        self.dx = 0
        self.dy = 0
        self.dtheta = 0

        #start
        self.start_x=293
        self.start_y=425
        self.distance_noise=5
        self.angle_noise=0.05

        # initialization arrays
        self.particles = []
        self.weights = []

    def init_particles(self):
        xrand=np.random.normal(self.start_x,self.distance_noise,self.num_p)
        yrand=np.random.normal(self.start_y,self.distance_noise,self.num_p)
        trand=np.random.normal(0.0,self.angle_noise,self.num_p)
        self.particles.append(Particale(xrand, yrand, trand))

    def gaus(x, mu=0, sigma=1):
        """calculates the probability of x for 1-dim Gaussian with mean mu and var. sigma"""
        return np.exp(- ((x - mu) ** 2) / (sigma ** 2) / 2.0) / np.sqrt(2.0 * np.pi * (sigma ** 2))


    def handle_odometry(self, odom):
        # relative motion from last odometry
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
            _, _, dtheta = tr.euler_from_quaternion(q_last_curr)
            self.dtheta += dtheta
            self.dx += p_last_curr[0]
            self.dy += p_last_curr[1]

    def predict_particle_odometry(self, p):
        # predict particle position
        # random
        nx = random.gauss(0, 0.1)
        ny = random.gauss(0, 0.1)
        ntheta = random.gauss(0, 0.1)
        # velocity
        v = sqrt(self.dx ** 2 + self.dy ** 2)
        # particle move is not dominated by noise
        if abs(v) < 1e-10 and abs(self.dtheta) < 1e-5:
            return
        p.x += v * cos(p.theta) + nx
        p.y += v * sin(p.theta) + ny
        p.theta += self.dtheta + ntheta

    def intersect(self,particle, beacon):
        x = beacon.x * np.cos(particle.theta) + beacon.y * np.sin(particle.theta) - particle.x * np.cos(particle.theta) - particle.y * np.sin(particle.theta)
        y = -beacon.x * np.sin(particle.theta) + beacon.y * np.cos(particle.theta) + particle.x * np.sin(particle.theta) - particle.y * np.cos(particle.theta)
        return x, y

    @staticmethod
    def get_prediction_error_squared(self, laser_scan_msg,particle):

        max_range = 3500
        min_inten = 800
        actual_ranges = laser_scan_msg.ranges
        n = len(laser_scan_msg.ranges)
        for i in range(n):
            if actual_ranges[i]>max_range or laser_scan_msg.intinsities[i]<min_inten:
                actual_ranges[i]=max_range
        d = (laser_scan_msg.angle_max - laser_scan_msg.angle_min) / n

        predict_ranges = [max_range]*n
        beacons=[]
        beacons.append(Beacon(0,0))
        beacons.append(Beacon(0,2000))
        beacons.append(Beacon(1500,0))
        beacons.append(Beacon(3000,1000))
        for b in beacons:
            x, y = self.intersect(particle, b)
            distance = sqrt(x * x + y * y)
            angle = atan2(y, x)
            ind_angle = math.ceil((angle - laser_scan_msg.angle_min)/d)
            if ind_angle>0 and ind_angle<n:
                for i in range(-5,5,1):
                    if ind_angle+i>0 and ind_angle+i<n:
                        predict_ranges[ind_angle+i]=distance
        diff = [actual_range - predict_range for actual_range, predict_range in zip(actual_ranges, predict_ranges)]
        norm_error = np.linalg.norm(diff)
        return norm_error ** 2

    def sigmoid(self, x):
        if x >= 0:
            z = exp(-x)
            return 1 / (1 + z)
        else:
            z = exp(x)
            return z / (1 + z)

    def resample(self, new_particles):
        # sample particle with probability that is proportional to its weight
        sample = np.random.uniform(0, 1)
        index = int(sample * (self.num_p - 1))
        beta = 0.0
        if not self.weights:
            self.weights = [1] * self.num_p
        max_w = max(self.weights)
        for p in self.particles:
            beta += np.random.uniform(0, 1) * 2.0 * max_w

            while beta > self.weights[index]:
                beta -= self.weights[index]
                index = (index + 1) % self.num_p

            p = self.particles[index]
            new_particles.append(Particale(p.x, p.y, p.theta))

    def handle_observation(self, laser_scan, dt):
        # prediction,weight updat and resampling
        # predict errors
        errors = []
        x=[]
        y=[]
        for p in self.particles:
            self.predict_particle_odometry(p)
            error = self.get_prediction_error_squared(laser_scan, p)
            errors.append(error)
            x.append(p.x)
            y.append(p.y)

        self.best_estimate=[np.mean(x),np.mean(y)]

        # update weights
        self.weights = [exp(-error) for error in errors]

        # compute effective sample size by weights
        sig_weight = [self.sigmoid(error) for error in errors]
        N_eff = sum([1 / (weight ** 2) for weight in sig_weight])

        # resample only when size_eff>thresh
        if N_eff > 90:
            new_particles = []
            self.resample(new_particles)
            self.particles = new_particles


class MonteCarlo(object):
    def __init__(self, num_p, xmin, xmax, ymin, ymax):
        rospy.init_node('localization', anonymous=True)
        trans_noise_std = 0.45
        rot_noise_std = 0.03

        self.min_inten = 800
        self.odom_sub = rospy.Subscriber('/real', Odometry, self.odom_callback, queue_size=1)
        self.laser_sub = rospy.Subscriber('/scan', LaserScan, self.laser_callback, queue_size=1)

        self.pf = ParticleFilter(num_p)

        dtheta = pi/15.0
        self.pf.init_particles()
        self.last_scan = None

        self.mutex = Lock()

    def odom_callback(self, msg):
        self.mutex.acquire()
        self.pf.handle_odometry(msg)
        self.mutex.release()

    def laser_callback(self, msg):
        self.pf.laser_min_angle = msg.angle_min
        self.pf.laser_max_angle = msg.angle_max
        self.pf.laser_min_range = msg.range_min
        self.pf.laser_max_range = msg.range_max

        dt_since_last_scan = 0
        if self.last_scan:
            dt_since_last_scan = (msg.header.stamp - self.last_scan.header.stamp).to_sec()

        self.mutex.acquire()
        self.pf.handle_observation(msg, dt_since_last_scan)

        self.pf.dx = 0
        self.pf.dy = 0
        self.pf.dtheta = 0

        self.mutex.release()
        self.last_scan = msg


def fibonacci_client(command, param):
    if param != '':
        par = struct.pack(dictCommands[command][0], *param)
        param = par
    send_command_to_stm = rospy.ServiceProxy('comm_to_stm', ros_stm_driver.srv.stm_command)
    res = send_command_to_stm(command, param)
    # try:
    realResult = struct.unpack(
        dictCommands[command][1], bytearray(res.parameters))
    # except struct.error:
    #     rospy.logerr("wrong result parameters")
    #     return 0
    rospy.loginfo("result = %s", realResult)
    return realResult


# rospy.init_node('stmaction_client',disable_signals=True)
# print("stm driver loaded")
rospy.wait_for_service('comm_to_stm')
print("seervice initialized")
num_particles = 100
xmin = 0
xmax = 3.2
ymin = 0
ymax = 2
mcl = MonteCarlo(num_particles, xmin, xmax, ymin, ymax)
print("MCL began")
# send_command_to_stm=rospy.ServiceProxy('comm_to_stm',ros_stm_driver.srv.stm_command)
dx = dy = dtheta = 0
while (1):
    command = int("0x11", 16)
    print("enter parameters")
    dx = input("x ")
    dy = input("y ")
    dtheta = input("theta ")
    param = [dx, dy, dtheta, 1]
    # send_command_to_stm=rospy.ServiceProxy('comm_to_stm',ros_stm_driver.srv.stm_command)
    print(command, param)
    # res=send_command_to_stm(command,param)
    fibonacci_client(command, param)
    print(mcl.pf.best_estimate[0], mcl.pf.best_estimate[1])
