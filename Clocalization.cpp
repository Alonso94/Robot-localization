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
#include <geometry_msgs/PoseStamped.h>

// Eigen headers
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <unsupported/Eigen/MatrixFunctions>

#include <tf/tf.h>
#include <tf/transform_datatypes.h>

#include <geometry_msgs/Twist.h>

#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>
using boost::mutex;

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

    /////////////////////////
    // start position
    //////////////////////////
    // purple - green
    double start_x=300,start_y=1300,start_th=3.14;
    // purple - red
    // double start_x=300,start_y=1400,start_th=3.14;
    // yellow - green
    //double start_x=2700,start_y=1300,start_th=0.0;
    // yellow - red
    //double start_x=2700,start_y=1400,start_th=0.0;

    point res;

    /////////////////////////////////////////
    // zone
    //////////////////////////////////////////
    // purple
    vector<pair<double,double>> beacons{{3094,1000},{-94,50},{-94,1950}};
    // yellow
    //vector<pair<double,double>> beacons{{-94,1000},{3094,50},{3094,1950}};

    double distance_noise=50,angle_noise=0.05;
    vector<point> particles;
    vector<point> new_particles;
    vector<double> weights;

    Particle_filter(){
        res.x_=start_x;
        res.y_=start_y;
        res.th_=start_th;
        init_particles(res,2,2);
    }
    void init_particles(point pose,double d_scale,double angle_scale){
        // initilize particles randomly (normal distribution)
        random_device rd;
        mt19937 gen(rd());
        normal_distribution<double> noise_x(pose.x_,d_scale*distance_noise);
        normal_distribution<double> noise_y(pose.y_,d_scale*distance_noise);
        normal_distribution<double> noise_th(pose.th_,angle_scale*angle_noise);
        point p;
        for(int i=0;i<num_particles;++i){
            p.x_=noise_x(gen);
            p.y_=noise_y(gen);
            p.th_=noise_th(gen);
            particles.push_back(p);
        }
    }
    void move_particles(double dx,double dy,double dth,double d_scale,double angle_scale){
        // move particles by odometry
        res.x_+=dx;
        res.y_+=dy;
        res.th_+=dth;
        res.th_=fmod(res.th_,2*pi);
        random_device rd;
        mt19937 gen(rd());
        normal_distribution<double> noise_x(0,d_scale*distance_noise);
        normal_distribution<double> noise_y(0,d_scale*distance_noise);
        normal_distribution<double> noise_th(0,angle_scale*angle_noise);
        for(int i=0;i<num_particles;++i){
            particles[i].x_+=dx+noise_x(gen);
            if (particles[i].x_>2950) particles[i].x_=2950;
            if (particles[i].x_<0) particles[i].x_=0;
            particles[i].y_+=dy+noise_y(gen);
            if (particles[i].y_>1950) particles[i].y_=1950;
            if (particles[i].y_<0) particles[i].y_=0;
            particles[i].th_+=dth+noise_th(gen);
            if (particles[i].th_>2*pi) particles[i].th_=fmod(particles[i].th_,2*pi);
        }
    }
    int calc_weights(vector<pair<double,double>> real_points){
        // calculate weight for each particle depending on the LIDAR data
        // real points: data from lidar, as points in the general frame
        vector<double> probs;
        int max_seen_beacons=0;
        int seen_beacons;
        double I,xb,yb,dx,dy,distance;
        point beacon_center,center;
        // for each particle
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
                    // function 1/(1+d^1.2)
                    I+=(1/(1+pow(distance,1.2)));
                }
                if(I>0.04) seen_beacons+=1;
            }
            if (seen_beacons>=3) probs.push_back(pow(10,(seen_beacons-1))*I);
            else if (seen_beacons==2) probs.push_back(I);
            else probs.push_back(0.05*I);
            max_seen_beacons=max(max_seen_beacons,seen_beacons);
        }
        double sum=0;
        for(auto&n :probs) {
            sum+=n;
        }
        weights.clear();
        // normalize weights
        for(int i=0;i<num_particles;++i){
            weights.push_back(probs[i]/sum);
        }
        return max_seen_beacons;
    }

    void resample_and_update(){
        // resample particle based on the weights
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<double> r(0.0,0.9);
        vector<double> cum_sum,u;
        cum_sum.push_back(0);
        for(int i=0;i<weights.size();++i){
            cum_sum.push_back(cum_sum[i]+weights[i]);
            u.push_back(float(r(gen)+i)/num_particles);
        }
        int j=0;
        new_particles.clear();
        for(auto&v:u){
            while (j<cum_sum.size()-1 && v>cum_sum[j]){
                j+=1;
            }
            new_particles.push_back(particles[j-1]);
        }
        particles.clear();
        for(int i=0;i<num_particles;++i){
            particles.push_back(new_particles[i]);
        }
    }

    point calc_pose(){
        // mean of the particles position along x,y ,theta
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
    double vx=0.0,vy=0.0;
    vector<pair<double,double>> real_points;
    Particle_filter pf;
    mutex mutex1;

    ros::Publisher position_pub;
    ros::Publisher laser_pub;
    ros::Publisher beacon_pub;
    ros::Publisher particles_pub;

    ros::Subscriber laser_sub;
    ros::Subscriber odom_sub;

    nav_msgs::OdometryConstPtr last_odom;
    nav_msgs::OdometryConstPtr curr_odom;
    bool first;



    MonteCarlo(int argc, char** argv){
        ros::init(argc,argv,"localization");
        ros::NodeHandle n;
        ros::Rate r(30);
        cout<<"contruct"<<endl;
        //poistion_pub=n.advertise<geometry_msgs::PoseStamped>("/robot_position",1);
        // output
        position_pub=n.advertise<nav_msgs::Odometry>("/real_corr",1);

        // visualization
        laser_pub=n.advertise<geometry_msgs::PoseArray>("/laser",1);
        beacon_pub=n.advertise<geometry_msgs::PoseArray>("/beacons",1);
        particles_pub=n.advertise<geometry_msgs::PoseArray>("/particles",1);

        laser_sub=n.subscribe<sensor_msgs::LaserScan>("/scan",1, &MonteCarlo::laser_callback,this);
        odom_sub=n.subscribe<nav_msgs::Odometry>("/real",1,&MonteCarlo::odom_callback,this);
        first=true;
    }

    vector<pair<double,double>> get_laser_points(vector<float> ranges,vector<float> intens,vector<double> angles,double min_inten) {
        double max_range=4;
        vector<pair<double ,double >> real_points;
        double x,y;
        for(int i=0;i<ranges.size();++i){
            if(ranges[i]>0.05 && ranges[i]<max_range && intens[i]>min_inten){
                x=ranges[i]*1000*cos(angles[i]);
                y=ranges[i]*1000*sin(angles[i]);
                real_points.emplace_back(x,-y);
            }
        }
        return real_points;
    }

    void handle_observation(const sensor_msgs::LaserScanConstPtr &laser_scan_msg,ros::Time time){
        //cout<<"recieved"<<endl;
        vector<double> angles;
        int count=1081;
        double angle=laser_scan_msg->angle_min;
        double add=laser_scan_msg->angle_increment;
        for (int i=0;i<count;++i){
            angles.push_back(angle);
            angle+=add;
        }
        // get real points with 3500 as a threshold over intensity
        real_points=get_laser_points(laser_scan_msg->ranges,laser_scan_msg->intensities,angles,3500);
        if(real_points.size()==0){
            publish_pose(time);
            publish_beacons();
            return;
        }
        publish_real_points();
        int seen_beacons;
        seen_beacons=pf.calc_weights(real_points);
        if(seen_beacons==0){
            // using central tower
            pf.beacons.emplace_back(1500,200);
            pf.beacons.emplace_back(1500,2050);
            while(seen_beacons<3){
                pf.init_particles(pf.res,8,8);
                real_points.clear();
                real_points=get_laser_points(laser_scan_msg->ranges,laser_scan_msg->intensities,angles,1200);
                publish_real_points();
                seen_beacons=pf.calc_weights(real_points);
                pf.resample_and_update();
                //publish_particles();
                publish_pose(time);
                publish_beacons();
            }
            pf.beacons.pop_back();
            pf.beacons.pop_back();
            pf.calc_weights(real_points);
            return;
        }
        pf.resample_and_update();
        //publish_particles();
        pf.res=pf.calc_pose();
        publish_pose(time);
        publish_beacons();
        //cout<<"pose: "<<pf.res.x_/1000<<" "<<pf.res.y_/1000<<" "<<pf.res.th_<<endl;
    }

    void publish_pose(ros::Time time){
        nav_msgs::Odometry pose;
        pose.header.stamp=time;
        pose.header.frame_id="map";
        pose.pose.pose.position.x=pf.res.x_/1000;
        pose.pose.pose.position.y=pf.res.y_/1000-0.02;
        pose.pose.pose.position.z=0.0 ;
        Eigen::Quaterniond quat(Eigen::AngleAxis<double>(fmod((pi+pf.res.th_),2*pi), Eigen::Vector3d(0,0,1)));
        pose.pose.pose.orientation.x=quat.x();
        pose.pose.pose.orientation.y=quat.y();
        pose.pose.pose.orientation.z=quat.z();
        pose.pose.pose.orientation.w=quat.w();
        position_pub.publish(pose);
    }

    void laser_callback(const sensor_msgs::LaserScanConstPtr &msg){
        mutex1.lock();
        ros::Time time;
        time=ros::Time::now();
        handle_observation(msg,time);
        mutex1.unlock();
    }

    void handle_odometry(const nav_msgs::OdometryConstPtr &odom){
        last_odom=curr_odom;
        curr_odom=odom;
        if (!first) {
            tf::Vector3 p_curr(curr_odom->pose.pose.position.x,
                               curr_odom->pose.pose.position.y,
                               curr_odom->pose.pose.position.z);

            tf::Vector3 p_last(last_odom->pose.pose.position.x,
                               last_odom->pose.pose.position.y,
                               last_odom->pose.pose.position.z);


            tf::Quaternion q_curr(curr_odom->pose.pose.orientation.x,
                                  curr_odom->pose.pose.orientation.y,
                                  curr_odom->pose.pose.orientation.z,
                                  curr_odom->pose.pose.orientation.w);

            tf::Quaternion q_last(last_odom->pose.pose.orientation.x,
                                  last_odom->pose.pose.orientation.y,
                                  last_odom->pose.pose.orientation.z,
                                  last_odom->pose.pose.orientation.w);

            //tf::Matrix3x3 rot_last(q_last);
            //dx=rot_last.transpose().tdotx(p_last-p_curr)*1000;
            //dy=rot_last.transpose().tdoty(p_last-p_curr)*1000;
            double x1=curr_odom->pose.pose.position.x;
            double x2=last_odom->pose.pose.position.x;
            dx=(x1-x2)*1000;
            double y1=curr_odom->pose.pose.position.y;
            double y2=last_odom->pose.pose.position.y;
            dy=(y1-y2)*1000;
            double th1,th2;
            th1=tf::getYaw(q_curr);
            th2=tf::getYaw(q_last);
            dth=th1-th2;
        }
        first=false;
    }

    void odom_callback(const nav_msgs::OdometryConstPtr &msg){
        mutex1.lock();
        ros::Time time;
        time=ros::Time::now();
        handle_odometry(msg);
        //vx=msg->twist.twist.linear.x;
        //vy=msg->twist.twist.linear.y;
        double scale,angle_scale;
        scale=1;
        angle_scale=1;
        if (abs(dx)>0.5 || abs(dy)>0.5) {
            scale=2;
            angle_scale=2;
        }
        if (dth>0.05) angle_scale=4;
        //cout<<"odom: "<<dx<<" "<<dy<<" "<<dth<<endl;
        pf.move_particles(dx,dy,dth,scale,angle_scale);
        //publish_pose(time);
        // publish_beacons();
        mutex1.unlock();
    }

    void publish_particles(){
        ros::Rate r(30);
        geometry_msgs::PoseArray poses;
        poses.header.stamp=ros::Time::now();
        poses.header.frame_id="map";
        for(auto&p:pf.particles){
            geometry_msgs::Pose pose;
            pose.position.x=p.x_/1000;
            pose.position.y=p.y_/1000;
            pose.position.z=0.0 ;
            Eigen::Quaterniond quat(Eigen::AngleAxis<double>(p.th_, Eigen::Vector3d(0,0,1)));
            pose.orientation.x=quat.x();
            pose.orientation.y=quat.y();
            pose.orientation.z=quat.z();
            pose.orientation.w=quat.w();
            poses.poses.insert(poses.poses.begin(),pose);
        }
        particles_pub.publish(poses);
        r.sleep();
    }

    void publish_real_points(){
        if(real_points.empty()) return;
        ros::Rate r(30);
        geometry_msgs::PoseArray poses;
        poses.header.stamp=ros::Time::now();
        poses.header.frame_id="map";
        for(auto&p:real_points){
            point real;
            real.x_=p.first;
            real.y_=p.second;
            real.th_=0.0;
            point r_real;
            r_real=local2global(real,pf.res);
            geometry_msgs::Pose pose;
            pose.position.x=r_real.x_/1000;
            pose.position.y=r_real.y_/1000;
            pose.position.z=0.0 ;
            Eigen::Quaterniond quat(Eigen::AngleAxis<double>(0.0, Eigen::Vector3d(0,0,1)));
            pose.orientation.x=quat.x();
            pose.orientation.y=quat.y();
            pose.orientation.z=quat.z();
            pose.orientation.w=quat.w();
            poses.poses.insert(poses.poses.begin(),pose);
        }
        laser_pub.publish(poses);
        r.sleep();
    }

    void publish_beacons(){
        ros::Rate r(30);
        geometry_msgs::PoseArray poses;
        poses.header.stamp=ros::Time::now();
        poses.header.frame_id="map";
        geometry_msgs::Pose pose;
        pose.position.x=pf.res.x_/1000;
        pose.position.y=pf.res.y_/1000;
        pose.position.z=0.0 ;
        Eigen::Quaterniond quat(Eigen::AngleAxis<double>(pf.res.th_, Eigen::Vector3d(0,0,1)));
        pose.orientation.x=quat.x();
        pose.orientation.y=quat.y();
        pose.orientation.z=quat.z();
        pose.orientation.w=quat.w();
        poses.poses.push_back(pose);
        for(auto&p:pf.beacons){
            pose.position.x=p.first/1000;
            pose.position.y=p.second/1000;
            pose.position.z=0.0 ;
            quat=Eigen::AngleAxis<double>(0.0, Eigen::Vector3d(0,0,1));
            pose.orientation.x=quat.x();
            pose.orientation.y=quat.y();
            pose.orientation.z=quat.z();
            pose.orientation.w=quat.w();
            poses.poses.insert(poses.poses.begin(),pose);
        }
        beacon_pub.publish(poses);
        r.sleep();
    }

};
int main(int argc, char** argv)
{
    MonteCarlo MCL(argc,argv);
    ros::spin();
return 0;