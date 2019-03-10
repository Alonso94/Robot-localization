#!/usr/bin/env python
import matplotlib.pyplot as plt
import scipy.optimize

import roslib
import rospy

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, Point,Pose2D,PoseArray, Quaternion
from threading import Lock
import numpy as np
import tf.transformations as tr
import tf
import time
from math import sqrt,atan2,exp

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

        self.num_particles=500
        self.start_x=150
        self.start_y=1250
        #second robot x=250 y=1660
        self.start_theta= np.pi

        self.beacons=np.array([[3094,1000],[-94,50],[-94,1950]])
        self.beacon_r=50
        # or
        #self.beacons=[[3094,50],[3094,1950],[-94,1000]]

        self.beac_dist_thresh=300
        self.num_is_near_thresh=0.1
        self.sense_noise=50



        self.distance_noise=50
        self.angle_noise=0.08

        #init_particles
        xrand = np.random.normal(self.start_x, self.distance_noise, self.num_particles)
        yrand = np.random.normal(self.start_y, self.distance_noise, self.num_particles)
        trand = np.random.normal(self.start_theta, self.angle_noise, self.num_particles)
        self.particles=np.array([xrand,yrand,trand]).T
        self.weights=[1]*self.num_particles

        """
        #beacons points in global frame
        self.beacons_points=[]
        r = self.beacon_r
        for b in self.beacons:
            beacon_points=[]
            x=b[0]-r
            while x<b[0]+r:
                y1=b[1]+sqrt(r*r+x*x)
                y2=b[1]-sqrt(r*r+x*x)
                point1=np.array([x,y1,0.0]).T
                point2=np.array([x,y2,0.0]).T
                beacon_points.append(point1)
                beacon_points.append(point2)
                x+=1
            self.beacons_points.append(beacon_points)
        """

    def move_particles(self,dx,dy,dtheta):
        if dx<0.00001 and dy<0.00001 and dtheta<0.00001:
            return
        x_noise = np.random.normal(0, self.distance_noise/2, self.num_particles)
        move_x=dx+x_noise
        y_noise = np.random.normal(0, self.distance_noise/2, self.num_particles)
        move_y=dy+y_noise
        angle_noise = np.random.normal(0, self.angle_noise/4, self.num_particles)
        move_theta=dtheta+angle_noise
        move_point=np.array([move_x,move_y,move_theta]).T
        self.particles = cvt_local2global(move_point, self.particles)
        self.particles[self.particles[:, 1] > 2000 - 100, 1] = 2000 - 100
        self.particles[self.particles[:, 1] < 0, 1] = 0
        self.particles[self.particles[:, 0] > 3000 - 100, 0] = 3000 - 100
        self.particles[self.particles[:, 0] < 100, 0] = 100
        self.particles[self.particles[:, 2] > (2*np.pi) ] %=(2*np.pi)


    def calc_weights(self,real_points):

        #beacons in particles frame
        #bs=[]
        #for i in range(3):
            #bs.append(cvt_global2local(self.beacons_points[:,i], self.particles))
        """
        errors=[]
        for particle in self.particles:
            diff=0

            for i in range(3):
                # beacon center in particle frame
                x = self.beacons[i, 0]
                y = self.beacons[i, 1]
                beacon_center = np.array([x, y, 0.0]).T
                center = cvt_global2local(beacon_center, particle)
                #determine which real set to comapre
                for set in real_points:
                    dx=center[0]-set[0][0]
                    dy=center[1]-set[0][1]
                    distance=sqrt(dx*dx+dy*dy)
                    #print(distance)
                    #print(ind,angle,start_angles[ind])
                    if distance<350:
                        # distance between each points  from real set and the beacon center
                        for point in set:
                            #print(point,center)
                            dx1=point[0]-center[0]
                            dy1=point[1]-center[1]
                            distance=sqrt(dx1*dx1+dy1*dy1)-50
                            #print(distance)
                            diff+=distance**2
            if diff==0:
                diff= 10**9
            errors.append(diff)
        self.weights = [exp(-error) for error in errors]
        #print(self.particles[np.argmax(self.weights)])
        """
        probs=[]
        for particle in self.particles:
            I=0
            for b in self.beacons:
                x = b[0]
                y = b[1]
                beacon_center = np.array([x, y, 0.0]).T
                center = cvt_global2local(beacon_center, particle)
                for point in real_points:
                    dx=point[0]-center[0]
                    dy=point[1]-center[1]
                    distance=abs(sqrt(dx*dx+dy*dy)-50)
                    I+=(1/(1+distance))
            probs.append(I)
        probs/=np.sum(probs)
        self.weights=[prob for prob in probs]

    def resample_and_update(self):

        n = self.num_particles
        weigths = np.array(self.weights)
        indices = []
        C = np.append([0.], np.cumsum(weigths))
        j = 0
        u0 = (np.random.rand() + np.arange(n)) / n
        for u in u0:
            while j < len(C)-1 and u > C[j]:
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
        self.position_pub = rospy.Publisher('/pose', Pose2D, queue_size=1)
        self.laser_sub = rospy.Subscriber('/scan', LaserScan, self.laser_callback, queue_size=1)
        self.odom_sub = rospy.Subscriber('/real', Odometry, self.odom_callback, queue_size=1)

        self.particles_pub = rospy.Publisher('/particles', PoseArray,queue_size=1)
        self.mutex = Lock()



    def handle_observation(self,laser_scan_msg):
        # prediction,weight updat and resampling
        # predict errors
        max_range = 4000
        min_inten = 2500
        ranges = list(laser_scan_msg.ranges)
        ranges = [x * 1000 for x in ranges]
        intens = list(laser_scan_msg.intensities)
        angles = np.arange(laser_scan_msg.angle_min, laser_scan_msg.angle_max, laser_scan_msg.angle_increment)

        real_points=[]
        real_set=[]
        old=0
        """
        for i in range(len(ranges)):
            if (ranges[i] < max_range) and (intens[i] > min_inten):
                if old==0:
                    old=1
                x=ranges[i] * np.cos(angles[i])
                y=ranges[i] * np.sin(angles[i])
                point=np.array([x,y]).T
                #print(point)
                real_set.append(point)
            elif old==1:
                old=0
                real_points.append(real_set)
                real_set=[]
        #x=input()
        #print(len(real_x))
        """

        for i in range(len(ranges)) :
            if (ranges[i] < max_range) and (intens[i] > min_inten):
                x = ranges[i] * np.cos(angles[i])
                y = ranges[i] * np.sin(angles[i])
                point=np.array([x,y]).T
                real_points.append(point)

        self.pf.move_particles(self.dx,self.dy,self.dtheta)
        self.pf.calc_weights(real_points)
        self.pf.resample_and_update()
        res=self.pf.calc_pose()
        print(res)
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
            self.dtheta = diff % (2*np.pi)
            self.dx = (p_last_curr[0]) * 1000
            self.dy = (p_last_curr[1]) * 1000

    def odom_callback(self,msg):
        self.mutex.acquire()
        self.handle_odometry(msg)
        self.mutex.release()

    def publish_particle_rviz(self):
        # Publishes the particles of the particle filter in rviz
        poses = PoseArray()
        poses.header.stamp = rospy.Time.now()
        poses.header.frame_id = "map"
        for p in self.pf.particles:
            point=Point(p[0]/1000,p[1]/1000,0)
            direction=p[2]
            quat = Quaternion(*tf.transformations.quaternion_from_euler(0, 0, direction))
            poses.poses.append(Pose(point, quat))

        self.particles_pub.publish(poses)

    def run(self):
        while not rospy.is_shutdown():
            self.publish_particle_rviz()
        time.sleep(1 / 30)

MCL=Montecarlo()
MCL.run()
