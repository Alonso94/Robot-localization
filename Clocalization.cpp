//
// Created by ali on 20.03.19.
//
#include <cstdlib>
#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <iterator>
#include <numeric>

using namespace std;

// ROS headers
#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/LaserScan.h>
#include <geometry_msgs/Pose2D.h>
#include <geometry_msgs/PoseArray.h>

// Eigen headers
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <unsupported/Eigen/MatrixFunctions>

#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2/LinearMath/Quaternion.h>

#include <ecl/threads.hpp>

using ecl::Mutex;

#define x_ first
#define y_ second.first
#define th_ second.second
#define pi acos(-1)

typedef pair<double,pair<double,double>> point;

point local2global(point local_point,point sc_point)
{
    point res;
    res.x_=local_point.x_*cos(sc_point.th_)-local_point.y_*sin(sc_point.th_)+sc_point.x_;
    res.y_=local_point.x_*sin(sc_point.th_)+local_point.y_*cos(sc_point.th_)+sc_point.y_;
    res.th_=fmod(local_point.th_+sc_point.th_,2*pi);
    return res;
}

point global2local(point global_point,point sc_point)
{
    point res;
    res.x_=global_point.x_*cos(sc_point.th_)+global_point.y_*sin(sc_point.th_)-sc_point.x_*cos(sc_point.th_)-sc_point.y_*sin(sc_point.th_);
    res.y_=-global_point.x_*sin(sc_point.th_)+global_point.y_*cos(sc_point.th_)+sc_point.x_*sin(sc_point.th_)-sc_point.y_*cos(sc_point.th_);
    res.th_=fmod(global_point.th_-sc_point.th_,2*pi);
    return res;
}

class Particle_filter{
public:
    int num_particles=1000;
    double start_x=280,start_y=1250,start_th=3.14;
    point res=make_pair(start_x,make_pair(start_y,start_th));
    vector<pair<double,double>> beacons{{3094,1000},{-94,50},{-94,1950}};
    double distance_noise=50,angle_noise=0.08;
    vector<point> particles;
    vector<double> weights;

    Particle_filter(){
        init_particles(res,1,1);
    }
    void init_particles(point pose,double d_scale,double angle_scale){
        random_device rd;
        mt19937 gen(rd());
        normal_distribution<double> noise_x(pose.x_,d_scale*distance_noise);
        normal_distribution<double> noise_y(pose.y_,d_scale*distance_noise);
        normal_distribution<double> noise_th(pose.th_,angle_scale*angle_noise);
        for(int i=0;i<num_particles;++i){
            particles[i].x_=noise_x(gen);
            particles[i].y_=noise_y(gen);
            particles[i].th_=noise_th(gen);
        }
    }
    void move_particles(double dx,double dy,double dth,double d_scale,double angle_scale){
        random_device rd;
        mt19937 gen(rd());
        normal_distribution<double> noise_x(0,d_scale*distance_noise);
        normal_distribution<double> noise_y(0,d_scale*distance_noise);
        normal_distribution<double> noise_th(0,angle_scale*angle_noise);
        for(int i=0;i<num_particles;++i){
            particles[i].x_+=dx+noise_x(gen);
            if (particles[i].x_>19500) particles[i].x_=1950;
            if (particles[i].x_<50) particles[i].x_=50;
            particles[i].y_+=dy+noise_y(gen);
            if (particles[i].y_>2950) particles[i].y_=2950;
            if (particles[i].y_<50) particles[i].y_=50;
            particles[i].th_+=dth+noise_th(gen);
            if (particles[i].th_>2*pi) particles[i].th_=fmod(particles[i].th_,2*pi);
        }
    }
    int calc_weights(vector<pair<double,double> > real_points){
        vector<double> probs;
        int max_seen_beacons=0;
        int seen_beacons;
        double I,xb,yb,dx,dy,distance;
        point beacon_center,center;
        for(int i=0;i<num_particles;i++){
            seen_beacons=0;
            I=0;
            for(auto&b :beacons){
                beacon_center.x_=b.first;
                beacon_center.y_=b.second;
                beacon_center.th_=0.0;
                center=global2local(beacon_center,particles[i]);
                for(auto&point : real_points){
                    dx=point.first-center.x_;
                    dy=point.second-center.y_;
                    distance=abs(sqrt(dx*dx+dy*dy)-50);
                    I+=(1/(1+pow(distance,1.2)));
                }
                if(I>0.04) seen_beacons+=1;
            }
            if (seen_beacons>=3) probs.push_back(pow(10,(seen_beacons-1))*I);
            else if (seen_beacons==2) probs.push_back(I);
            else probs.push_back(0.005*I);
            max_seen_beacons=max(max_seen_beacons,seen_beacons);
        }
        double sum=0;
        for(auto&n :probs) sum+=n;
        for(int i=0;i<num_particles;++i){
            weights[i]=probs[i]/sum;
        }
        return max_seen_beacons;
    }
    void resample_and_update(){
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<double> r(0.0,0.9);
        vector<double> cum_sum,u;
        cum_sum[0]=0;
        for(int i=0;i<weights.size();++i){
            cum_sum[i+1]=cum_sum[i]+weights[i];
            u[i]=float(r(gen)+i)/num_particles;
        }
        vector<point> new_particles;
        int j=0;
        for(auto&v:u){
            while (j<cum_sum.size()-1 and v<cum_sum[j]){
                j+=1;
            }
            new_particles.push_back(particles[j-1]);
        }
        particles.clear();
        particles=new_particles;
    }
    point calc_pose(){
        double th0=particles[0].th_;
        double sum_x=0,sum_y=0,sum_th=0;
        for(auto&p:particles){
            sum_x+=p.x_;
            sum_y+=p.y_;
            sum_th+=fmod(p.th_-th0+pi,2.0*pi)+th0-pi;
        }
        point res;
        res.x_=sum_x/num_particles;
        res.y_=sum_y/num_particles;
        res.th_=fmod(sum_th/num_particles,2.0*pi);
        return res;
    }
};
class MonteCarlo{
public:
    double dx=0.0,dy=0.0,dth=0.0;
    vector<pair<double,double>> real_points;
    Particle_filter pf;
    Mutex mutex;

    ros::Publisher poistion_pub;
    ros::Publisher laser_pub;
    ros::Publisher beacon_pub;
    ros::Publisher particles_pub;

    ros::Subscriber laser_sub;
    ros::Subscriber odom_sub;

    nav_msgs::Odometry last_odom;
    nav_msgs::Odometry curr_odom;
    bool first;

    MonteCarlo(int& argc,char **argv){
        ros::init(argc,argv,"localization");
        ros::NodeHandle n;
        ros::Rate r(30);

        poistion_pub=n.advertise<geometry_msgs::Pose2D>("/robot_position",1);
        laser_pub=n.advertise<geometry_msgs::PoseArray>("/laser",1);
        beacon_pub=n.advertise<geometry_msgs::PoseArray>("/beacons",1);
        particles_pub=n.advertise<geometry_msgs::PoseArray>("/particles",1);

        laser_sub=n.subscribe<sensor_msgs::LaserScan>("/scan",1,&MonteCarlo::laser_callback,this);
        odom_sub=n.subscribe<nav_msgs::Odometry>("/real",1,&MonteCarlo::odom_callback,this);
        first=true;
    }

    vector<pair<double,double>> get_laser_points(vector<float> ranges,vector<float> intens,vector<double> angles,double min_inten) {
        double max_range=4;
        vector<pair<double ,double >> real_points;
        double x,y;
        for(int i=0;i<ranges.size();++i){
            if(ranges[i]<max_range && intens[i]>min_inten){
                x=ranges[i]*1000*cos(angles[i]);
                y=ranges[i]*1000*sin(angles[i]);
                real_points.emplace_back(x,-y);
            }
        }
        return real_points;
    }

    void handle_observation(sensor_msgs::LaserScan laser_scan_msg){
        vector<double> angles;
        auto count=int(laser_scan_msg.scan_time/laser_scan_msg.time_increment);
        double angle=laser_scan_msg.angle_min;
        for (int i=0;i<count;++i){
            angles.push_back(angle);
            angle+=laser_scan_msg.angle_increment;
        }
        vector<pair<double,double>> real_points;
        real_points=get_laser_points(laser_scan_msg.ranges,laser_scan_msg.intensities,angles,2500);

        int seen_beacons;
        seen_beacons=pf.calc_weights(real_points);
        if(seen_beacons==0){
            pf.beacons.emplace_back(1500,200);
            pf.beacons.emplace_back(1500,2050);
            while(seen_beacons<3){
                pf.init_particles(pf.res,8,8);
                real_points.clear();
                real_points=get_laser_points(laser_scan_msg.ranges,laser_scan_msg.intensities,angles,1200);
                seen_beacons=pf.calc_weights(real_points);
                pf.resample_and_update();
            }
            pf.beacons.pop_back();
            pf.beacons.pop_back();
            pf.calc_weights(real_points);
        }
        pf.resample_and_update();
        pf.res=pf.calc_pose();

        geometry_msgs::Pose2D pose;
        pose.x=pf.res.x_/1000;
        pose.y=pf.res.y_/1000;
        pose.theta=pf.res.th_;
        poistion_pub.publish(pose);
    }

    void laser_callback(sensor_msgs::LaserScan &msg){
        mutex.lock();
        handle_observation(msg);
        mutex.unlock();
    }

    void handle_odometry(nav_msgs::Odometry &odom){
        first=false;
        last_odom=curr_odom;
        curr_odom=odom;
        double p_curr[3],p_last[3],q_curr[4],q_last[4];
        if (!first) {
            p_curr[0] = curr_odom.pose.pose.position.x;
            p_curr[1] = curr_odom.pose.pose.position.y;
            p_curr[2] = curr_odom.pose.pose.position.z;

            p_last[0] = last_odom.pose.pose.position.x;
            p_last[1] = last_odom.pose.pose.position.y;
            p_last[2] = last_odom.pose.pose.position.z;

            q_curr[0] = curr_odom.pose.pose.orientation.x;
            q_curr[1] = curr_odom.pose.pose.orientation.y;
            q_curr[2] = curr_odom.pose.pose.orientation.z;
            q_curr[3] = curr_odom.pose.pose.orientation.w;

            q_last[0] = last_odom.pose.pose.orientation.x;
            q_last[1] = last_odom.pose.pose.orientation.y;
            q_last[2] = last_odom.pose.pose.orientation.z;
            q_last[3] = last_odom.pose.pose.orientation.w;



        }
    }
};
int main()
{
    point r(make_pair(1,make_pair(1,3.14)));
    point k(make_pair(1,make_pair(1,1.3)));
    cout<<r.x_<<" "<<r.y_<<" "<<r.th_<<endl;
    point res=local2global(r,k);
    cout<<res.x_<<" "<<res.y_<<" "<<res.th_<<endl;
    res=global2local(res,k);
    cout<<res.x_<<" "<<res.y_<<" "<<res.th_<<endl;
    return 0;
}
