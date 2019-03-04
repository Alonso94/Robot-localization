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
from math import sqrt,atan2 ,exp
from scipy.stats import norm
from std_msgs.msg import String, Header, ColorRGBA
from geometry_msgs.msg import Twist, PoseStamped, Point,Pose2D
from visualization_msgs.msg import Marker, MarkerArray


def cvt_local2global(local_point, sc_point):
    x, y, a = local_point.T
    X, Y, A = sc_point.T
    x1 = x * np.cos(A) - y * np.sin(A) + X
    y1 = x * np.sin(A) + y * np.cos(A) + Y
    a1 = (a + A) % (2 * np.pi)
    return np.array([x1, y1, a1]).T


def cvt_global2local(global_point, sc_point):
    x, y, a = global_point.T
    X, Y, A = sc_point.T
    x1 = x * np.cos(A) + y * np.sin(A) - X * np.cos(A) - Y * np.sin(A)
    y1 = -x * np.sin(A) + y * np.cos(A) + X * np.sin(A) - Y * np.cos(A)
    a1 = (a - A) % (2 * np.pi)
    return np.array([x1, y1, a1]).T

def find_src(global_point, local_point):
    x, y, a = local_point.T
    x1, y1, a1 = global_point.T
    A = (a1 - a) % (2 * np.pi)
    X = x1 - x * np.cos(A) + y * np.sin(A)
    Y = y1 - x * np.sin(A) - y * np.cos(A)
    return np.array([X, Y, A]).T


class ParticleFilter(object):
    def __init__(self):

        self.num_particles=100
        self.start_x=300
        self.start_y=1250
        #second robot x=250 y=1660
        self.start_theta=0.0

        self.beacons=np.array([[3094,1000],[-94,50],[-94,1950]])
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
        self.particles=np.array([xrand,yrand,trand]).T
        self.weights=[1]*self.num_particles

    @staticmethod
    def gaus(x, mu=0, sigma=1):
        return np.exp(- ((x - mu) ** 2) / (sigma ** 2) / 2.0) / np.sqrt(2.0 * np.pi * (sigma ** 2))

    def move_particles(self,dx,dy,dtheta):

        x_noise = np.random.normal(0, self.distance_noise/4, self.num_particles)
        move_x=dx+x_noise
        y_noise = np.random.normal(0, self.distance_noise/4, self.num_particles)
        move_y=dy+y_noise
        angle_noise = np.random.normal(0, self.angle_noise/2, self.num_particles)
        move_theta=dtheta+angle_noise
        move_point=np.array([move_x,move_y,move_theta]).T
        self.particles = cvt_local2global(move_point, self.particles)
        self.particles[self.particles[:, 1] > 2000 - 100, 1] = 2000 - 100
        self.particles[self.particles[:, 1] < 0, 1] = 0
        self.particles[self.particles[:, 0] > 3000 - 100, 0] = 3000 - 100
        self.particles[self.particles[:, 0] < 100, 0] = 100
        self.particles[self.particles[:, 2] > (2*np.pi) ] %=(2*np.pi)

    def calc_weights(self,ranges,angle_min):

        diffs=[]
        p_sum=np.zeros(self.num_particles)
        for p in self.particles:
            diff=0
            for b in self.beacons:
                dx=p[0]-b[0]
                dy=p[1]-b[1]
                distance=sqrt(dx*dx+dy*dy)
                angle=atan2(dy,dx)+p[2]+np.pi
                while angle<-np.pi:
                    angle+=np.pi*2
                while angle>np.pi:
                    angle-=np.pi*2
                angle=angle-angle_min
                sigma=0.3
                p=np.random.normal(angle,sigma,self.num_particles)
                p_sum+=(p/np.max(p))*distance
            for r in range(self.num_particles):
                if ranges[r]>50:
                    diff+=(range[r]-p_sum[r])**2
            diffs.append(diff)
        diffs/=np.sum(diffs)
        self.weights = [(1/error) for error in diffs]
        self.weights/=np.sum(self.weights)

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

    def weighted_mean(self):
        x = np.sum(self.particles[:, 0]*self.weights)/np.sum(self.weights)
        y = np.sum(self.particles[:, 1] * self.weights) / np.sum(self.weights)
        zero_elem = self.particles[0, 2]
        # this helps if particles angles are close to 0 or 2*pi
        temporary = ((self.particles[:, 2] - zero_elem + np.pi) % (2.0 * np.pi)) + zero_elem - np.pi
        angle = (np.sum(temporary*self.weights)/np.sum(self.weights)) % (2.0 * np.pi)
        return np.array((x,y,angle))

class Montecarlo(object):
    def __init__(self):

        self.curr_odom = None
        self.last_odom = None

        self.dx = 0.0
        self.dy = 0.0
        self.dtheta = 0.0

        self.pf=ParticleFilter()

        rospy.init_node('localization', anonymous=True)
        self.position_pub = rospy.Publisher('/robot_position', Pose2D, queue_size=1)
        self.laser_sub = rospy.Subscriber('/scan', LaserScan, self.laser_callback, queue_size=1)
        self.odom_sub = rospy.Subscriber('/real', Odometry, self.odom_callback, queue_size=1)
        self.particles_pub = rospy.Publisher('/husky_1/particle_filter/particles', MarkerArray, queue_size=1)
        self.mutex = Lock()


    def handle_observation(self,laser_scan_msg):
        # prediction,weight updat and resampling
        # predict errors
        max_range = 4000
        min_inten = 2500
        ranges = list(laser_scan_msg.ranges)
        ranges=[x*1000 for x in ranges]
        intens = list(laser_scan_msg.intensities)
        angles = np.arange(laser_scan_msg.angle_min, laser_scan_msg.angle_max, laser_scan_msg.angle_increment)
        angle_min=laser_scan_msg.angle_min

        laser_ranges=[]
        for i in range(len(ranges)):
            if (ranges[i] < max_range) and (intens > min_inten):
                laser_ranges.append(range[i])


        self.pf.move_particles(self.dx,self.dy,self.dtheta)
        self.pf.calc_weights(laser_ranges,angle_min)
        res = self.pf.weighted_mean()
        self.pf.resample_and_update()
        #res=self.pf.calc_pose()

        self.position_pub.publish(Pose2D(res[0]/1000,res[1]/1000,res[2]))

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

            self.dtheta = diff% (2*np.pi)
            self.dx = (p_last_curr[0]) * 1000
            self.dy = (p_last_curr[1]) * 1000

    def odom_callback(self,msg):
        self.mutex.acquire()
        self.handle_odometry(msg)
        self.mutex.release()


MCL=Montecarlo()
rospy.spin()