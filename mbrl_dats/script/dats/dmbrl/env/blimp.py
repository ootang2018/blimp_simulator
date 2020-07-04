from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import time
import rospy
import sys

import numpy as np
import gym
from gym import spaces

from math import pi

from std_msgs.msg import Float64, Float64MultiArray
from sensor_msgs.msg import JointState, Imu
from mav_msgs.msg import Actuators
from geometry_msgs.msg import Twist, TwistStamped, PoseArray, Pose, Point, PointStamped
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from visualization_msgs.msg import *

from .myTF import MyTF
from .gazeboConnection import GazeboConnection

class BlimpActionSpace():
    def __init__(self):
        '''
        0: left motor 
        1: right motor
        2: back motor
        3: servo
        4: top fin
        5: bottom fin 
        6: left fin
        7: right fin
        '''
        STICK_LIMIT = pi/2
        FIN_LIMIT = 0#pi/9
        MOTOR_LIMIT = 70
        MOTOR3_LIMIT = 0#30
        self.action_space = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        self.act_bnd = np.array([MOTOR_LIMIT, MOTOR_LIMIT, MOTOR3_LIMIT, STICK_LIMIT, FIN_LIMIT, FIN_LIMIT, FIN_LIMIT, FIN_LIMIT])
        self.shape = self.action_space.shape
        self.dU = self.action_space.shape[0]

class BlimpObservationSpace():
    def __init__(self):
        '''
        state
        0:2 relative angle
        3:5 angular velocity
        6:8 relative position
        9:11 velocity
        12:14 acceleration
        '''
        DISTANCE_BND = 10#50
        ORIENTATION_BND = pi 
        ORIENTATION_VELOCITY_BND = pi
        VELOCITY_BND = 10
        ACCELERATION_BND = 4
        self.observation_space = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.obs_bnd = np.array([ORIENTATION_BND, ORIENTATION_BND, ORIENTATION_BND,
            ORIENTATION_VELOCITY_BND, ORIENTATION_VELOCITY_BND, ORIENTATION_VELOCITY_BND,
            DISTANCE_BND, DISTANCE_BND, DISTANCE_BND,
            VELOCITY_BND, VELOCITY_BND, VELOCITY_BND,
            ACCELERATION_BND ,ACCELERATION_BND ,ACCELERATION_BND])
        self.shape = self.observation_space.shape
        self.dO = self.observation_space.shape[0]

class BlimpEnv(gym.Env):

    def __init__(self, SLEEP_RATE = 10, USE_MPC=True):
        super(BlimpEnv, self).__init__()

        rospy.init_node('RL_node', anonymous=False)
        rospy.loginfo("[RL Node] Initialising...")

        self.SLEEP_RATE = SLEEP_RATE
        self.RATE = rospy.Rate(SLEEP_RATE) # loop frequency
        self.EPISODE_TIME = 30 # 30 sec
        self.EPISODE_LENGTH = self.EPISODE_TIME * self.SLEEP_RATE 
        self.use_MPC = USE_MPC

        self._load()
        self._create_pubs_subs()

        self.gaz = GazeboConnection(True, "WORLD")

        rospy.loginfo("[RL Node] Initialized")

    def _load(self):
        rospy.loginfo("[RL Node] Load and Initialize Parameters...")

        self.GRAVITY = 9.81

        # action noise
        noise_stddev = 0.1
        self.noise_stddev = noise_stddev

        # action space
        act_space = BlimpActionSpace()
        self.act_bnd = act_space.act_bnd
        self.action_space = spaces.Box(low=-1, high=1,
                                        shape=act_space.shape, dtype=np.float32)

        # observation space
        obs_space = BlimpObservationSpace()
        self.obs_bnd = obs_space.obs_bnd
        self.observation_space = spaces.Box(low=-1, high=1,
                                            shape=obs_space.shape, dtype=np.float32)

        # msgs initialize
        self.angle = np.array((0,0,0))
        self.target_angle = np.array((0,0,0))
        self.angular_velocity = np.array((0,0,0))
        self.position = np.array((0,0,0))
        self.target_position = np.array((0,0,0))
        self.velocity = np.array((0,0,0))
        self.linear_acceleration = np.array((0,0,0))
        self.reward = Float64()

        # MPC
        self.MPC_HORIZON = 15
        self.SELECT_MPC_TARGET = 14
        self.MPC_TARGET_UPDATE_RATE = self.SLEEP_RATE * 2 
        self.MPC_position_target = np.array((0,0,0))
        self.MPC_attitude_target = np.array((0,0,0))

        # misc
        self.cnt=0
        self.timestep=1
        self.pub_and_sub = False

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
        self.pub_and_sub = True

    def _reward_callback(self, msg):
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
        az = msg.linear_acceleration.z - self.GRAVITY

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
            # NED to my frame
            x = msg.poses[i].position.x
            y = msg.poses[i].position.y
            z = msg.poses[i].position.z
            position_trajectory.append([y,-x,z])
            time_trajectory.append(0.1*i*time_mult+current_time)
            
            # NED to world frame
            temp_pose_msg = Pose()
            temp_pose_msg.position.x = y 
            temp_pose_msg.position.y = x 
            temp_pose_msg.position.z = -z 
            MPC_rviz_trajectory.poses.append(temp_pose_msg)

        for i in range(0, self.MPC_HORIZON-1):
            yaw_trajectory.append(np.arctan2(position_trajectory[i+1][1]-position_trajectory[i][1],position_trajectory[i+1][0]-position_trajectory[i][0]))
        yaw_trajectory.append(yaw_trajectory[-1])

        position_trajectory = np.array(position_trajectory)
        yaw_trajectory = np.array(yaw_trajectory)
        self.MPC_rviz_trajectory_publisher.publish(MPC_rviz_trajectory)

        # Update MPC target 
        if (self.timestep%self.MPC_TARGET_UPDATE_RATE ==0):
            self.MPC_position_target = position_trajectory[self.SELECT_MPC_TARGET]
            self.MPC_attitude_target = yaw_trajectory[self.SELECT_MPC_TARGET] # to avoid dramatic yaw change

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

    def _euler_from_pose(self, pose):
        a = pose.orientation.x
        b = pose.orientation.y
        c = pose.orientation.z
        d = pose.orientation.w
        euler = MyTF.euler_from_quaternion(a,b,c,d)
        return euler     
          
    def step(self,action):
        self.timestep += 1
        action = self.act_bnd * action

        act = Float64MultiArray()
        self.action = action
        act.data = action
        self.action_publisher.publish(act)

        self.RATE.sleep()
        obs, reward, done = self._get_obs()
        return obs, reward, done, {}

    def reset(self):
        self.gaz.resetSim()
        self.timestep = 1
        obs, reward, done = self._get_obs()
        return obs

    def _get_acts(self):
        action = self.action

        return action

    def _get_obs(self):
        if (self.use_MPC):
            relative_angle = self.MPC_attitude_target - self.angle
            relative_distance = self.MPC_position_target - self.position
        else:
            relative_angle = self.target_angle - self.angle 
            relative_distance = self.target_position - self.position
        
        if relative_angle[0] > np.pi:
            relative_angle[0] -= 2*np.pi
        elif  relative_angle[0] < -np.pi:
            relative_angle[0] += 2*np.pi

        if relative_angle[1] > np.pi:
            relative_angle[1] -= 2*np.pi
        elif  relative_angle[1] < -np.pi:
            relative_angle[1] += 2*np.pi

        if relative_angle[2] > np.pi:
            relative_angle[2] -= 2*np.pi
        elif  relative_angle[2] < -np.pi:
            relative_angle[2] += 2*np.pi

        #extend state
        state = []
        state.extend(relative_angle)
        state.extend(self.angular_velocity)
        state.extend(relative_distance)
        state.extend(self.velocity)
        state.extend(self.linear_acceleration)
        state = np.array(state)
        state = state / (2*self.obs_bnd) #normalize

        #extend reward
        if self.reward is None:
            reward = -1
        else:
            reward = self.reward.data

        #done is used to reset environment when episode finished
        done = False
        if (self.timestep%(self.EPISODE_LENGTH+1)==0):
            done = True

        #reset if blimp fly too far away
        if any(np.abs(self.position)>= 50) :
            self.gaz.resetSim()
            reward -= 10

        return state, reward, done
