# Mobile robot localization project:
#### Used tools: C++, Python, ROS, Hokuyo UST-10LX LIDAR, Odroid XU4, mobile robot
#### Algorithm implemented : Monte Carlo Localization
### Problem formulation:
The goal is to localize a mobile robot in Eurobot environment; a table which has an area of 200x300 cm^2 (blue rectangel in the image).
Two teams are competing in each match, The team could start in the yellow (right) or in the purple area (left) according to the results of a draft.
The team has the right to mount the beacons to use them in the localization, in the following image an illustration for the case of starting at the yellow area.</br>
(The robot is in the green, beacons are in yellow color)</br>
<img src="https://github.com/Alonso94/Robot-localization/blob/master/PF.png"/> </br>
We were asked to use a Hokuyo LIDAR, mounted on top of the robot, to localize the robot in the playground with an accuracy less than 1 cm.

### The steps of work:
1. Get introduced to LIDAR and how it is work;</br>
    Firstly you have to clone the following packages (inside this ROS package):
    ```
    git clone https://github.com/ros-drivers/urg_c
    git clone https://github.com/ros-perception/laser_proc
    ```
    After that, you have to change interface file by accessing it:
    ```
    sudo nano /etc/network/interfaces
    ```
    and add the following at the end of it:
    ```
    auto eth0:0
    iface eth0:0 inet static
      address 192.168.0.10
      netmask 255.255.255.0
      gateway 192.168.0.1
    source directory /etc/network/interfaces
    ```
    Enter to the network manager (after connecting the LIDAR)
    ```
      sudo nmtui
    ```
    After that add a new wired connection with the following parameters
    ```
    Name : LIDAR (it is up to you)
    IPv4 tab : IPv4 Configuration
      address 192.168.0.10
      gate: 192.168.0.1
    ```
    Add the following to ~/.bashrc file
    ```
    export ROS_HOSTNAME=192.168.0.15
    export ROS_MASTER_URI=http://192.168.0.15:11311/
    ```
    Every time you have to start by running the following
    ```
    roscore &
    rosrun urg_node urg_node _ip_address:="192.168.0.10"
    ```
    You will recieve a /scan topic in which there are data from the LIDAR.</br>
2. The first try, was to localize the robot by simple __triangulation method__; we have chosen beacons with reflecation surface,
  then we can filter them by the intensity of the reflected lasre rays.</br>
  Accordin to the charasteristics of the LIDAR, the error margin of the range for each point could be +-40 mm, which makes the results from this method very bad;
  Where the robot could be in a point inside a hexagon the distance between its opposite sides about 8cm, the position received from just triangulation was floating inside that hexagon.</br>
3. We decided to use __particle filter__ to acheive a better accuracy , the idea of using particle filter in localization was first addressed by [1], for better understanding to particle filter we recommend watching the video from the author of MCL __Wolfram burgrad__ (see [2])</br>



[1]Fox, D., Burgard, W., Dellaert, F., & Thrun, S. (1999). Monte carlo localization: Efficient position estimation for mobile robots. AAAI/IAAI, 1999(343-349), 2-2.</br>
[2] lecture 9; http://ais.informatik.uni-freiburg.de/teaching/ss19/robotics/
