#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

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

scan=np.load("scan1.npy")

data=scan.tolist()
angles=np.arange(data.angle_min,data.angle_max,data.angle_increment)
ranges=np.asarray(data.ranges)*1000
intens=np.asarray(data.intensities)

max_range=3200
min_inten=1000
cond = (ranges < max_range) * (intens > min_inten)
x = (ranges * np.cos(angles))[cond]
y = (ranges * np.sin(angles))[cond]
inten = intens[cond]

fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=300)
ax.scatter(x, y, s=1, c=inten)

#plt.show()

Beacons=np.array([[-100,50],[-100,1950],[3100,1000],[1500,220],[1500,2000]])
r=44
print(angles)
points = np.zeros((len(x), 3))
points[:, 0] = x
points[:, 1] = y

fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=300)
ax.scatter(*cvt_local2global(points, [190,1190,0])[:, 0:2].T,  s=1, c=inten)
for b in Beacons:
    ax.add_artist(plt.Circle(b, r, linewidth=1, fill=False, color="r"))
plt.show()

######################

init_X = np.array([180, 1100, 0])
init_X_true = np.array([190, 1190, 0])
# points in robot frame (false frame)
apr_points = cvt_local2global(points, init_X)[:, 0:2]
print(apr_points)
#distance to each beacon
beacons_len = np.sum((Beacons[np.newaxis, :, :] - apr_points[:, np.newaxis, :]) ** 2, axis=2) ** 0.5
print(beacons_len)
# label points
num_beacons = np.argmin(beacons_len, axis=1)
print(num_beacons)
def fun(X, points, num_beacons):
    beacon = Beacons[num_beacons]
    points = cvt_local2global(points, X)[:, 0:2]
    total_r = np.sum((beacon - points) ** 2, axis=1) ** 0.5 - r
    return total_r

res = scipy.optimize.least_squares(fun, init_X, loss="cauchy",args=[points, num_beacons], ftol=1e-6)
print(res['x'])

fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=300)
ax.scatter(*cvt_local2global(points, res['x'])[:, 0:2].T,  s=1, c=inten)
for b in Beacons:
    ax.add_artist(plt.Circle(b, r, linewidth=1, fill=False, color="r"))

plt.show()




