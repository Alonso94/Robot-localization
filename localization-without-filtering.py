#!/usr/bin/env python
import matplotlib.pyplot as plt
import scipy.optimize

import roslib
import rospy

from localization.srv import GetCoords
from sensor_msgs.msg import LaserScan
from threading import Lock
import numpy as np

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

    beacons_len = np.sum((Beacons[np.newaxis, :, :] - apr_points[:, np.newaxis, :]) ** 2, axis=2) ** 0.5

    # points in robot frame (false frame)
    apr_points = cvt_local2global(points, init_X)[:, 0:2]

    # label points
    num_beacons = np.argmin(beacons_len, axis=1)

    def fun(X, points, num_beacons):
        beacon = Beacons[num_beacons]
        points = cvt_local2global(points, X)[:, 0:2]
        total_r = np.sum((beacon - points) ** 2, axis=1) ** 0.5 - r
        return total_r

    res = scipy.optimize.least_squares(fun, init_X, loss="cauchy", args=[points, num_beacons], ftol=1e-6)

    return res

def laser_callback(self, msg):

    mutex.acquire()
    handle_observation(msg)
    mutex.release()

def get_coords_callback():
    return GetCoordsResponse(True, x,y,z)

x=200
y=1180
thata=0.0

rospy.init_node('localization', anonymous=True)
rospy.Service('get_coordinates', GetCoords, get_coords_callback)
laser_sub = rospy.Subscriber('/scan', LaserScan, laser_callback, queue_size=1)
mutex = Lock()