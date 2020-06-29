from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import time
import rospy
import sys

import numpy as np
from dotmap import DotMap
from math import pi

from std_msgs.msg import Float64, Float64MultiArray
from sensor_msgs.msg import JointState, Imu
from mav_msgs.msg import Actuators
from geometry_msgs.msg import Twist, TwistStamped, PoseArray, Pose, Point, PointStamped
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from visualization_msgs.msg import *

from dmbrl.env.myTF import MyTF
from dmbrl.env.gazeboConnection import GazeboConnection

class BlimpActionSpace():
    def __init__(self):
        # m1 m2 m3 s ftop fbot fleft fright
        self.action_space = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        self.high = np.array([70, 70, 30, pi/2, pi/9, pi/9, pi/9, pi/9])
        self.low = -self.high
        self.shape = self.action_space.shape
        self.dU = self.action_space.shape[0]


class BlimpObservationSpace():
    def __init__(self):
        self.observation_space = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.high = np.array([pi, pi, pi, pi, pi, pi, 10, 10, 10, 5, 5, 5, 3 ,3 ,3])
        self.low = -self.high
        self.shape = self.observation_space.shape
        self.dO = self.observation_space.shape[0]

class BlimpEnv():

    def __init__(self):
        rospy.init_node('RL_node', anonymous=False)
        rospy.loginfo("[RL Node] Initialising...")

        self._load()
        self.pub_and_sub = False
        self._create_pubs_subs()
        self.pub_and_sub = True

        self.gaz = GazeboConnection(True, "WORLD")
        # self.gaz.unpauseSim()

        rospy.loginfo("[RL Node] Initialized")

    def _load(self):
        rospy.loginfo("[RL Node] Load and Initialize Parameters...")

        self.RATE = rospy.Rate(2) # loop frequency
        self.GRAVITY = -9.8

        # action noise
        noise_stddev = 0.1
        self.noise_stddev = noise_stddev

        # action space
        self.action_space = BlimpActionSpace()
        self.dU = self.action_space.dU
        # self.action = (self.action_space.high + self.action_space.low)/2

        # observation space
        '''
        state
        0:2 relative angle
        3:5 angular velocity
        6:8 relative position
        9:11 velocity
        12:14 acceleration
        '''
        self.observation_space = BlimpObservationSpace()
        self.dO = self.observation_space.dO

        # msgs initialize
        self.angle = [0,0,0]
        self.target_angle = [0,0,0]
        self.angular_velocity = [0,0,0]
        self.position = [0,0,0]
        self.target_position = [0,0,0]
        self.velocity = [0,0,0]
        self.linear_acceleration = [0,0,0]
        self.reward = Float64()

        # MPC
        self.MPC_HORIZON = 15
        self.SELECT_MPC_TARGET = 10
        self.MPC_position_target = []
        self.MPC_attitude_target = []

        rospy.loginfo("[RL Node] Load and Initialize Parameters Finished")

    def _create_pubs_subs(self):
        rospy.loginfo("[RL Node] Create Subscribers and Publishers...")

        """ create subscribers """
        rospy.Subscriber(
            "/target/update_full",
            InteractiveMarkerInit,
            self._interactive_target_callback)
        rospy.Subscriber(
            "/moving_target",
            Pose,
            self._moving_target_callback)
        rospy.Subscriber(
            "/blimp/ground_truth/imu",
            Imu,
            self._imu_callback)
        rospy.Subscriber(
            "/blimp/ground_truth/position",
            PointStamped,
            self._gps_callback)
        rospy.Subscriber(
            "/blimp/ground_speed",
            TwistStamped,
            self._velocity_callback)
        rospy.Subscriber(
            "/blimp/reward",
            Float64,
            self._reward_callback)
        rospy.Subscriber(
            "/machine_1/mpc_calculated/pose_traj",
            PoseArray,
            self._trajectory_callback)

        """ create publishers """
        self.action_publisher = rospy.Publisher(
            "/blimp/controller_cmd",
            Float64MultiArray,
            queue_size=1)
        self.MPC_target_publisher = rospy.Publisher(
            "/actorpose",
            Odometry,
            queue_size=1)
        self.MPC_rviz_trajectory_publisher = rospy.Publisher(
            "/blimp/MPC_rviz_trajectory",
            PoseArray,
            queue_size=60)

        rospy.loginfo("[RL Node] Subscribers and Publishers Created")

    def _reward_callback(self,msg):
        """
        blimp/reward:
        Float64

        :param msg:
        :return:
        """
        self.reward = msg

    def _imu_callback(self, msg):
        """
        sensor_msgs/Imu:
        std_msgs/Header header
          uint32 seq
          time stamp
          string frame_id
        geometry_msgs/Quaternion orientation
          float64 x
          float64 y
          float64 z
          float64 w
        float64[9] orientation_covariance
        geometry_msgs/Vector3 angular_velocity
          float64 x
          float64 y
          float64 z
        float64[9] angular_velocity_covariance
        geometry_msgs/Vector3 linear_acceleration
          float64 x
          float64 y
          float64 z
        float64[9] linear_acceleration_covariance

        :param msg:
        :return:
        """
        a = msg.orientation.x
        b = msg.orientation.y
        c = msg.orientation.z
        d = msg.orientation.w

        # NED Frame
        p = msg.angular_velocity.x
        q = -1*msg.angular_velocity.y
        r = -1*msg.angular_velocity.z

        ax = -1*msg.linear_acceleration.x
        ay = msg.linear_acceleration.y
        az = -1*msg.linear_acceleration.z + self.GRAVITY

        # from Quaternion to Euler Angle
        euler = MyTF.euler_from_quaternion(a,b,c,d)

        phi = euler[0]
        the = -1*euler[1]
        psi = -1*euler[2]

        self.angle = np.array((phi,the,psi))
        self.angular_velocity = np.array((p,q,r))
        self.linear_acceleration = np.array((ax,ay,az))

    def _gps_callback(self, msg):
        """
        geometry_msgs/PointStamped:
        std_msgs/Header header
          uint32 seq
          time stamp
          string frame_id
        geometry_msgs/Point point
          float64 x
          float64 y
          float64 z

        :param msg:
        :return:
        """
        location = msg

        # NED Frame
        location.point.y = location.point.y * -1
        location.point.z = location.point.z * -1
        self.position = np.array((location.point.x, location.point.y, location.point.z))

        if (self.pub_and_sub):
            self.MPC_target_publish()

    def _velocity_callback(self, msg):
        """
        std_msgs/Header header
          uint32 seq
          time stamp
          string frame_id
        geometry_msgs/Twist twist
          geometry_msgs/Vector3 linear
            float64 x
            float64 y
            float64 z
          geometry_msgs/Vector3 angular
            float64 x
            float64 y
            float64 z
        """
        velocity = msg

        # NED Frame
        velocity.twist.linear.y = velocity.twist.linear.y * -1
        velocity.twist.linear.z = velocity.twist.linear.z * -1
        self.velocity = np.array((velocity.twist.linear.x, velocity.twist.linear.y, velocity.twist.linear.z))

    def _interactive_target_callback(self, msg):
        """
        InteractiveMarkerInit

        string server_id
        uint64 seq_num
        visualization_msgs/InteractiveMarker[] markers
          std_msgs/Header header
            uint32 seq
            time stamp
            string frame_id
          geometry_msgs/Pose pose
            geometry_msgs/Point position
              float64 x
              float64 y
              float64 z
            geometry_msgs/Quaternion orientation
              float64 x
              float64 y
              float64 z
              float64 w

        :param msg:
        :return:
        """
        target_pose = msg.markers[0].pose
        
        # NED Frame
        euler = self._euler_from_pose(target_pose)
        target_phi, target_the, target_psi = 0, 0, -1*euler[2]
        self.target_angle = np.array((target_phi, target_the, target_psi))
        target_pose.position.y = target_pose.position.y*-1
        target_pose.position.z = target_pose.position.z*-1
        self.target_position = np.array((target_pose.position.x, target_pose.position.y, target_pose.position.z))

        print("Interactive Target Pose")
        print("=============================")
        print("position = ",self.target_position)
        print("angle = ",self.target_angle)
    
    def _moving_target_callback(self, msg):
        """
        geometry_msgs/Point position
          float64 x
          float64 y
          float64 z
        geometry_msgs/Quaternion orientation
          float64 x
          float64 y
          float64 z
          float64 w

        :param msg:
        :return:
        """

        target_pose = msg
        euler = self._euler_from_pose(target_pose)
        target_phi, target_the, target_psi = 0, 0, euler[2]
        self.target_angle = np.array((target_phi, target_the, target_psi))

        # NED Frame
        target_pose.position.y = target_pose.position.y*-1
        target_pose.position.z = target_pose.position.z*-1
        self.target_position = np.array((target_pose.position.x, target_pose.position.y, target_pose.position.z))

    def _trajectory_callback(self, msg):
        """
        15 waypoint for the next 3 secs

        geometry_msgs/Pose: 
        geometry_msgs/Point position
          float64 x
          float64 y
          float64 z
        geometry_msgs/Quaternion orientation
          float64 x
          float64 y
          float64 z
          float64 w

        :param msg:
        :return:
        """
        data=[]
        time_mult=1

        position_trajectory = []
        time_trajectory = []
        yaw_trajectory = [] 
        MPC_position_trajectory = []

        MPC_rviz_trajectory = PoseArray()  
        MPC_rviz_trajectory.header.frame_id="world"
        MPC_rviz_trajectory.header.stamp=rospy.Time.now()
        MPC_rviz_trajectory.poses=[]

        current_time = time.time()
        for i in range(self.MPC_HORIZON):
            x = msg.poses[i].position.x
            y = msg.poses[i].position.y
            z = msg.poses[i].position.z
            position_trajectory.append([y,-x,z])
            time_trajectory.append(0.1*i*time_mult+current_time)
            
            temp_pose_msg = Pose()
            temp_pose_msg.position.x = y 
            temp_pose_msg.position.y = x 
            temp_pose_msg.position.z = -z 
            MPC_rviz_trajectory.poses.append(temp_pose_msg)

        for i in range(0, self.MPC_HORIZON-1):
            yaw_trajectory.append(np.arctan2(position_trajectory[i+1][1]-position_trajectory[i][1],position_trajectory[i+1][0]-position_trajectory[i][0]))
        yaw_trajectory.append(yaw_trajectory[-1])

        position_trajectory = np.array(position_trajectory)
        time_trajectory = np.array(time_trajectory)
        yaw_trajectory = np.array(yaw_trajectory)
        self.MPC_rviz_trajectory_publisher.publish(MPC_rviz_trajectory)

        (_,
         _,
         MPC_yaw_cmd) = self.trajectory_control(
                 position_trajectory,
                 yaw_trajectory,
                 time_trajectory, time.time())
        self.MPC_position_target = position_trajectory[self.SELECT_MPC_TARGET]
        self.MPC_attitude_target = np.array((0.0, 0.0, MPC_yaw_cmd))

    def MPC_target_publish(self):
        """
        std_msgs/Header header
          uint32 seq
          time stamp
          string frame_id
        string child_frame_id
        geometry_msgs/PoseWithCovariance pose
          geometry_msgs/Pose pose
            geometry_msgs/Point position
              float64 x
              float64 y
              float64 z
            geometry_msgs/Quaternion orientation
              float64 x
              float64 y
              float64 z
              float64 w
          float64[36] covariance
        geometry_msgs/TwistWithCovariance twist
          geometry_msgs/Twist twist
            geometry_msgs/Vector3 linear
              float64 x
              float64 y
              float64 z
            geometry_msgs/Vector3 angular
              float64 x
              float64 y
              float64 z
          float64[36] covariance
        """
        target_pose = Odometry()
        #NED
        target_pose.header.frame_id="world"
        target_pose.pose.pose.position.x = -self.target_position[1]; 
        target_pose.pose.pose.position.y = self.target_position[0]; 
        target_pose.pose.pose.position.z = self.target_position[2];
        self.MPC_target_publisher.publish(target_pose)

    def trajectory_control(self, position_trajectory, yaw_trajectory, time_trajectory, current_time):
        """Generate a commanded position, velocity and yaw based on the trajectory

        Args:
            position_trajectory: list of 3-element numpy arrays, NED positions
            yaw_trajectory: list yaw commands in radians
            time_trajectory: list of times (in seconds) that correspond to the position and yaw commands
            current_time: float corresponding to the current time in seconds

        Returns: tuple (commanded position, commanded velocity, commanded yaw)

        """

        ind_min = np.argmin(np.abs(np.array(time_trajectory) - current_time))
        time_ref = time_trajectory[ind_min]


        if current_time < time_ref:
            position0 = position_trajectory[ind_min - 1]
            position1 = position_trajectory[ind_min]

            time0 = time_trajectory[ind_min - 1]
            time1 = time_trajectory[ind_min]
            yaw_cmd = yaw_trajectory[ind_min - 1]

        else:
            yaw_cmd = yaw_trajectory[ind_min]
            if ind_min >= len(position_trajectory) - 1:
                position0 = position_trajectory[ind_min]
                position1 = position_trajectory[ind_min]

                time0 = 0.0
                time1 = 1.0
            else:

                position0 = position_trajectory[ind_min]
                position1 = position_trajectory[ind_min + 1]
                time0 = time_trajectory[ind_min]
                time1 = time_trajectory[ind_min + 1]

        position_cmd = (position1 - position0) * \
                        (current_time - time0) / (time1 - time0) + position0
        velocity_cmd = (position1 - position0) / (time1 - time0)

        return (position_cmd, velocity_cmd, yaw_cmd)

    def _euler_from_pose(self, pose):
        a = pose.orientation.x
        b = pose.orientation.y
        c = pose.orientation.z
        d = pose.orientation.w
        euler = MyTF.euler_from_quaternion(a,b,c,d)
        return euler     
          
    def step(self,action):
        act = Float64MultiArray()
        self.action = action
        act.data = action
        self.action_publisher.publish(act)

        self.RATE.sleep()

        obs, reward, done = self._get_obs()
        return obs, reward, done, {}

    def reset(self):
        self.gaz.resetSim()
        obs, reward, done = self._get_obs()
        return obs

    def _get_acts(self):
        action = self.action

        return action

    def _get_obs(self):
        # relative_angle = self.target_angle - self.angle 
        # relative_distance = self.target_position - self.position;
        relative_angle = self.MPC_attitude_target - self.angle
        relative_distance = self.MPC_position_target - self.position

        #extend state
        state = []
        state.extend(relative_angle)
        state.extend(self.angular_velocity)
        state.extend(relative_distance)
        state.extend(self.velocity)
        state.extend(self.linear_acceleration)

        # state.extend(self.target_angle)
        # state.extend(self.target_position)

        #extend reward
        if self.reward is None:
            reward = -1
        else:
            reward = self.reward.data

        #done is not used in this experiment
        done = False

        return state, reward, done
