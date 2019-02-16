#!/usr/bin/env python
import rospy
import numpy as np
import tf.transformations as tr
from math import sqrt,cos,sin,pi,atan2,log,exp
import random
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from threading import Lock

class Particale(object):
    def __init__(self,id,x,y,theta):
        self.id=id
        self.x=x
        self.y=y
        self.theta=theta

class ParticleFilter(object):
    def __init__(self,num_p,xmin,xmax,ymin,ymax,laser_min_range,laser_max_range,
                 laser_min_angle,laser_max_angle,trans_noise_std,rot_noise_std):
        # number of particles
        self.num_p=num_p
        # make a grid
        self.origin_x=0
        self.origin_y=0
        self.height=2000
        self.width=3200
        self.resolution=20
        #cell is true if the probability of being occuped is zero
        self.grid_bin=np.zeros((2000,3200),dtype='int8')
        # beacons
        self.grid_bin[0:100,0:100]=np.ones((100,100),dtype='int8')
        self.grid_bin[1900:2000, 0:1000] = np.ones((100, 100), dtype='int8')
        self.grid_bin[950:1050,3100:3200] = np.ones((100, 100), dtype='int8')
        self.grid_bin[1500:100, 1600:200] = np.ones((100, 100), dtype='int8')
        self.grid_bin[1500:0, 1600:1] = np.ones((100, 1), dtype='int8')

        # workspace
        self.xmin=xmin
        self.xmax=xmax
        self.ymin=ymin
        self.ymax=ymax
        # laser
        self.laser_min_range=laser_min_range
        self.laser_max_range=laser_max_range
        self.laser_min_angle=laser_min_angle
        self.laser_max_angle=laser_max_angle
        # noise
        self.trans_noise_std=trans_noise_std
        self.rot_noise_std=rot_noise_std

        # number of laser beams to evaluate laser beams of a particle
        self.eval_beams=32
        # odometry
        self.curr_odom=None
        self.last_odom=None
        # relative motion since the last time prticles were updated
        self.dx=0
        self.dy=0
        self.dtheta=0
        # initialization arrays
        self.particles=[]
        self.weights=[]

    def metric_to_grid(self,x,y):
        gx=(x-self.origin_x)/self.resolution
        gy=(y-self.origin_y)/self.resolution
        row=min(max(int(gy),0),self.height)
        col=min(max(int(gx),0),self.width)
        return row,col

    def get_random_p(self,r_xmin,r_xmax,r_ymin,r_ymax,thetamax):
        while True:
            xrand=np.random.uniform(r_xmin,r_xmax)
            yrand=np.random.uniform(r_ymin,r_ymax)
            row,col=self.metric_to_grid(xrand,yrand)
            if self.grid_bin[row,col]:
                theta=np.random.uniform(-thetamax,thetamax)
                return xrand,yrand,theta

    def init_particles(self,r_xmin,r_xmax,r_ymin,r_ymax,thetamax):
        for i in range(self.num_p):
            xrand,yrand,theta=self.get_random_p(r_xmin,r_xmax,r_ymin,r_ymax,thetamax)
            self.particles.append(Particale(i,xrand,yrand,theta))

    def handle_odometry(self,odom):
        # relative motion from last odometry
        self.last_odom=self.curr_odom
        self.curr_odom=odom
        if self.last_odom:
            p_curr=np.array([self.curr_odom.pose.pose.position.x,
                                self.curr_odom.pose.pose.position.y,
                                self.curr_odom.pose.pose.position.z])
            p_last=np.array([self.last_odom.pose.pose.position.x,
                                 self.last_odom.pose.pose.position.y,
                                 self.last_odom.pose.pose.position.z])
            q_curr=np.array([self.curr_odom.pose.pose.position.x,
                                self.curr_odom.pose.pose.position.y,
                                self.curr_odom.pose.pose.position.z,
                                self.curr_odom.pose.pose.position.w])
            q_last=np.array([self.last_odom.pose.pose.position.x,
                                self.last_odom.pose.pose.position.y,
                                self.last_odom.pose.pose.position.z,
                                self.last_odom.pose.pose.position.w])
            rot_last=tr.quaternion_matrix(q_last)[0:3,0:3]
            p_last_curr=rot_last.transpose().dot(p_curr-p_last)
            q_last_curr=tr.quaternion_multiply(tr.quaternion_inverse(q_last),q_curr)
            _,_,dtheta=tr.euler_from_quaternion(q_last_curr)
            self.dtheta+=dtheta
            self.dx+=p_last_curr[0]
            self.dy+=p_last_curr[1]

    def predict_particle_odometry(self,p):
        # predict particle position
        # random
        nx=random.gauss(0,self.trans_noise_std)
        ny=random.gauss(0,self.trans_noise_std)
        ntheta=random.gauss(0,self.rot_noise_std)
        # velocity
        v=sqrt(self.dx**2+self.dy**2)
        # particle move is not dominated by noise
        if abs(v) < 1e-10 and abs(self.dtheta) < 1e-5:
            return
        p.x+=v*cos(p.theta)+nx
        p.y+=v*sin(p.theta)+ny
        p.theta+=self.dtheta+ntheta

    def subsample_laser_scan(self,laser_scan_msg):

        max_range = 3050
        min_inten = 800
        intens=laser_scan_msg.intensities
        ranges = laser_scan_msg.ranges
        cond = (ranges < max_range) * (intens > min_inten)
        ranges=ranges[cond]

        n = len(laser_scan_msg.ranges)
        d = (laser_scan_msg.angle_max - laser_scan_msg.angle_min) / n
        angles=[d*float(i)+laser_scan_msg.angle_min for i in range(n)]
        step=n/self.eval_beams
        angles=angles[::step]
        ranges=ranges[::-step]

        actual_ranges=[]
        for r in ranges:
            if r<self.laser_min_range and r<=self.laser_max_range:
                actual_ranges.append(r)
            elif r<self.laser_min_range:
                actual_ranges.append(self.laser_min_range)
            elif r>self.laser_max_range:
                actual_ranges.append(self.laser_max_range)
        return actual_ranges,angles

    def simulate_laser_for_p(self,x,y,theta,angles,min_range,max_range):
        # laser scan if the robot at the particle p
        ranges=[]
        range_step=self.resolution
        for angle in angles:
            phi=theta+angle
            r=min_range
            while r<=max_range:
                xm=x+r*cos(phi)
                ym=y+r*sin(phi)
                if xm>self.xmax or xm<self.xmin:
                    break
                if ym>self.ymax or ym<self.ymin:
                    break
                row,col=self.metric_to_grid(xm,ym)
                free=self.grid_bin[row,col].all()
                if not free:
                    break
                r+=range_step
            ranges.append(r)
        return ranges

    def get_prediction_error_squared(self,laser_scan_msg,p):
        if p.x<self.xmin or p.x>self.xmax:
            return 300
        if p.y<self.ymin or p.y>self.ymax:
            return 300
        row,col=self.metric_to_grid(p.x,p.y)
        if not self.grid_bin[row,col]:
            return 300
        # get actual ranges
        [actual_ranges,angles]=self.subsample_laser_scan(laser_scan_msg)
        # get predicted
        predict_ranges=self.simulate_laser_for_p(p.x,p.y,p.theta,angles,self.laser_min_range,self.laser_max_range)
        # the error
        diff=[actual_range-predict_range for actual_range,predict_range in zip(actual_ranges,predict_ranges)]
        norm_error=np.linalg.norm(diff)

        return norm_error**2

    def sigmoid(self,x):
        if x>=0:
            z=exp(-x)
            return 1/(1+z)
        else:
            z=exp(x)
            return z/(1+z)

    def resample(self,new_particles):
        # sample particle with probability that is proportional to its weight
        sample=np.random.uniform(0,1)
        index=int(sample*(self.num_p-1))
        beta=0.0
        if not self.weights:
            self.weights=[1]*self.num_p
        max_w=max(self.weights)
        for p in self.particles:
            beta+=np.random.uniform(0,1)*2.0*max_w

            while beta > self.weights[index]:
                beta-= self.weights[index]
                index=(index+1)%self.num_p

            p=self.particles[index]
            new_particles.append(Particale(p.id,p.x,p.y,p.theta))

    def handle_observation(self,laser_scan,dt):
        # prediction,weight updat and resampling
        # predict errors
        errors=[]
        for p in self.particles:
            self.predict_particle_odometry(p)
            error=self.get_prediction_error_squared(laser_scan,p)
            errors.append(error)

        ind=np.argmin(errors)
        best_estimate=self.particles[ind]
        print(best_estimate.x,best_estimate.y)

        # update weights
        self.weights=[exp(-error) for error in errors]

        # compute effective sample size by weights
        sig_weight=[self.sigmoid(error) for error in errors]
        N_eff=sum([1/(weight**2) for weight in sig_weight])

        # resample only when size_eff>thresh
        if N_eff>50:
            new_particles=[]
            self.resample(new_particles)
            self.particles=new_particles

class MonteCarlo(object):
    def __init__(self,num_p,xmin,xmax,ymin,ymax):
        rospy.init_node('localization',anonymous=True)
        trans_noise_std=0.45
        rot_noise_std=0.03

        self.min_inten=800
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback, queue_size=1)
        self.laser_sub = rospy.Subscriber('/scan', LaserScan, self.laser_callback, queue_size=1)

        self.build_map()

        self.pf=ParticleFilter(num_p,xmin,xmax,ymin,ymax,10,3050,0,0,
                               trans_noise_std,rot_noise_std)

        r_xmin, r_xmax, r_ymin, r_ymax,dtheta=[200,1000,0,600,pi/4]
        self.pf.init_particles(r_xmin, r_xmax, r_ymin, r_ymax,dtheta)
        self.last_scan=None

        self.mutex = Lock()

    def odom_callback(self,msg):
        self.mutex.acquire()
        self.pf.handle_odometry(msg)
        self.mutex.release()

    def laser_callback(self,msg):
        self.pf.laser_min_angle=msg.angle_min
        self.pf.laser_max_angle=msg.angle_max
        self.pf.laser_min_range=msg.range_min
        self.pf.laser_max_range=msg.range_max

        dt_since_last_scan=0
        if self.last_scan:
            dt_since_last_scan=(msg.header.stamp-self.last_scan.header.stamp).to_sec()

        self.mutex.acquire()
        self.pf.handle_observation(msg,dt_since_last_scan)

        self.pf.dx=0
        self.pf.dy=0
        self.pf.dtheta=0

        self.mutex.release()
        self.last_scan=msg


