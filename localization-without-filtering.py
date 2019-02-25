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

def handle_observation(laser_scan_msg):
    # prediction,weight updat and resampling
    # predict errors
    max_range = 3800
    min_inten = 1200
    ranges = list(laser_scan_msg.ranges)
    ranges*=1000
    intens = list(laser_scan_msg.intensities)
    angles = np.arange(laser_scan_msg.angle_min, laser_scan_msg.angle_max, laser_scan_msg.angle_increment)

    cond = (ranges < max_range) * (intens > min_inten)
    x = (ranges * np.cos(angles))[cond]
    y = (ranges * np.sin(angles))[cond]
    inten = intens[cond]

    Beacons = np.array([[-100, 50], [-100, 1950], [3100, 1000]])#, [1500, 220], [1500, 2000]])

    r = 44
    print(angles)
    points = np.zeros((len(x), 3))
    points[:, 0] = x
    points[:, 1] = y

    global robot_position
    # points in robot frame (false frame)
    apr_points = cvt_local2global(points, robot_position)[:, 0:2]

    beacons_len = np.sum((Beacons[np.newaxis, :, :] - apr_points[:, np.newaxis, :]) ** 2, axis=2) ** 0.5

    # label points
    num_beacons = np.argmin(beacons_len, axis=1)

    init_pos=[x,y,z]

    def fun(X, points, num_beacons):
        beacon = Beacons[num_beacons]
        points = cvt_local2global(points, X)[:, 0:2]
        total_r = np.sum((beacon - points) ** 2, axis=1) ** 0.5 - r
        return total_r

    robot_position = scipy.optimize.least_squares(fun, robot_position, loss="cauchy", args=[points, num_beacons], ftol=1e-6)
    res=Coordinates(robot_position[0],robot_position[1],robot_position[2])
    position_pub.publish(res)

def laser_callback(msg):

    mutex.acquire()
    handle_observation(msg)
    mutex.release()

def handle_odometry(odom):
    global robot_position
    global curr_odom,last_odom,dx,dy,dtheta
    last_odom = curr_odom
    curr_odom = odom
    if last_odom:
        p_curr = np.array([curr_odom.pose.pose.position.x,
                           curr_odom.pose.pose.position.y,
                           curr_odom.pose.pose.position.z])
        p_last = np.array([last_odom.pose.pose.position.x,
                           last_odom.pose.pose.position.y,
                           last_odom.pose.pose.position.z])
        q_curr = np.array([curr_odom.pose.pose.orientation.x,
                           curr_odom.pose.pose.orientation.y,
                           curr_odom.pose.pose.orientation.z,
                           curr_odom.pose.pose.orientation.w])
        q_last = np.array([last_odom.pose.pose.orientation.x,
                           last_odom.pose.pose.orientation.y,
                           last_odom.pose.pose.orientation.z,
                           last_odom.pose.pose.orientation.w])
        rot_last = tr.quaternion_matrix(q_last)[0:3, 0:3]
        p_last_curr = rot_last.transpose().dot(p_curr - p_last)
        q_last_curr = tr.quaternion_multiply(tr.quaternion_inverse(q_last), q_curr)
        _, _, diff = tr.euler_from_quaternion(q_last_curr)
        dtheta += diff
        dx += (p_last_curr[0]) * 1000
        dy += (p_last_curr[1]) * 1000
        v = np.sqrt(dx ** 2 +dy ** 2)
        robot_position[0]+=v*np.cos(theta)
        robot_position[1]+=v*np.sin(theta)
        robot_position[2]+=dtheta

def odom_callback(msg):
    mutex.acquire()
    handle_odometry(msg)
    mutex.release()


x=200
y=1180
theta=0.0
robot_position=[x,y,theta]
curr_odom=None
last_odom=None
dx=0
dy=0
rospy.init_node('localization', anonymous=True)
position_pub = rospy.Publisher('/robot_position',Coordinates, queue_size = 1)
laser_sub = rospy.Subscriber('/scan', LaserScan, laser_callback, queue_size=1)
odom_sub = rospy.Subscriber('/real', Odometry, odom_callback, queue_size=1)
mutex = Lock()