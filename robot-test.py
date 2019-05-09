#!/usr/bin/env python
import rospy


import ros_stm_driver.srv
import struct

from commands_lists.big_commands import dictCommands
from geometry_msgs.msg import Pose2D
import time


def fibonacci_client(command, param):
    add_two_ints = rospy.ServiceProxy('comm_to_stm', ros_stm_driver.srv.stm_command)
    if param != '':
        par = struct.pack(dictCommands[command][0], *param)
        param = par
    res = add_two_ints(command, param)
    realResult = struct.unpack(
        dictCommands[command][1], bytearray(res.parameters))
    return realResult

def glob_fib_client(command, param):
    finished = 0
    while not finished:
        try:
            result = fibonacci_client(command, param)
        except Exception:
            time.sleep(0.1)
            finished = 0
        else:
            finished = 1
    return result




position_known=False
param1=[]
def position_correction(msg):
    global position_known
    global param1
    position_known=True
    param1=[]
    param1.append(msg.x)
    param1.append(msg.y)
    param1.append(msg.theta)

points=[]
# points.append([2.85,0.2,0.0,1])
# points.append([2.75,0.2,0.0,1])
# points.append([2.75,0.6,0.0,1])
# points.append([1.77,0.6,0.0,1])
# points.append([2.0,1.0,2.17,1])
# points.append([2.85,1.25,0.0,1])

points.append([0.15,0.2,3.14,1])
points.append([0.25,0.2,3.14,1])
points.append([0.25,0.6,3.14,1])
points.append([1.25,0.6,3.14,1])
points.append([1.0,1.0,2.14,1])
points.append([0.15,1.25,3.14,1])

# rospy.init_node('stmaction_client',disable_signals=True)
# print("stm driver loaded")
rospy.wait_for_service('comm_to_stm')
rospy.init_node('path_test',anonymous=True)
position_sub = rospy.Subscriber('/robot_position', Pose2D,position_correction,queue_size=1)
while not position_known:
    x=1
print("position is found")
command=int("0x3",16)
glob_fib_client(command,param1)
for p in points:
    while 1:
        if abs(param1[0]-p[0])<0.01 and abs(param1[1]-p[1])<0.01 and abs(param1[2]-p[2])<0.05:
            break
        command = int("0x11", 16)
        print("to point")
        print(p)
        glob_fib_client(command,p)
        command=int("0x32",16)
        time.sleep(0.2)

        while not glob_fib_client(command,'')[0]:
            time.sleep(0.1)
        position_known = False
        while not position_known:
            time.sleep(0.1)
        print(param1)
        command = int("0x3", 16)
        glob_fib_client(command, param1)
        rospy.sleep(1)