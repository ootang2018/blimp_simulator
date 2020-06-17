#!/usr/bin/env python
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
from geometry_msgs.msg import Twist, TwistStamped, Point, PointStamped
from std_srvs.srv import Empty
from visualization_msgs.msg import *

# mine
from pets.misc.myTF import MyTF
# from pets.controller import MPC
from gazeboConnection import GazeboConnection


class Agent:
    """An general class for RL agents.
    """
    def __init__(self):
        """Initializes an agent.

        Arguments:
            params:
                .env: The environment for this agent.
                .noisy_actions: (bool) Indicates whether random Gaussian noise will
                    be added to the actions of this agent.
                .noise_stddev: (float) The standard deviation to be used for the
                    action noise if params.noisy_actions is True.
        """
        # rospy.init_node('Agent_node', anonymous=False)
        rospy.loginfo("[Agent Node] Initialising...")

        self._load()
        self._create_pubs_subs()

        self.gaz = GazeboConnection(True, "WORLD")
        # self.gaz.unpauseSim()

        rospy.loginfo("[Agent Node] Initialized")

    def _load(self):
        rospy.loginfo("[Agent Node] Load and Initialize Parameters...")

        self.RATE = rospy.Rate(2) # loop frequency
        self.GRAVITY = -9.8

        # action noise
        noise_stddev = 0.1
        self.noise_stddev = noise_stddev

        # action space
        # m1 m2 m3 s ftop fbot fleft fright
        self.action_space = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        # self.ac_ub = np.array([70, 70, 50, pi/2, pi/36, pi/36, pi/36, pi/36])
        self.ac_ub = np.array([70, 70, 50, pi/2, 0, 0, 0, 0])
        self.ac_lb = -self.ac_ub
        self.dU = 8
        self.action = (self.ac_ub + self.ac_lb)/2

        # observation space
        # phi the psi phitarget thetarget psitarget p q r x y z xtarget ytarget ztarget vx vy vz ax ay az
        self.observation_space = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.ob_ub = np.array([pi, pi, pi, pi, pi, pi, pi/2, pi/2, pi/2, 5, 5, 5, 5 ,5 ,5 , 2.5, 2.5, 2.5, 1.25, 1.25, 1.25])
        self.ob_lb = -self.ob_ub
        self.dO = 21

        # msgs initialize
        self.angle = [0,0,0]
        self.target_angle = [0,0,0]
        self.angular_velocity = [0,0,0]
        self.position = [0,0,0]
        self.target_position = [0,0,0]
        self.velocity = [0,0,0]
        self.linear_acceleration = [0,0,0]
        self.reward = Float64()

        rospy.loginfo("[Agent Node] Load and Initialize Parameters Finished")

    def _create_pubs_subs(self):
        rospy.loginfo("[Agent Node] Create Subscribers and Publishers...")

        """ create subscribers """
        rospy.Subscriber(
            "/target/update_full",
            InteractiveMarkerInit,
            self._target_callback)
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

        """ create publishers """
        self.action_publisher = rospy.Publisher(
            "/blimp/controller_cmd",
            Float64MultiArray,
            queue_size=1)

        rospy.loginfo("[Agent Node] Subscribers and Publishers Created")

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

        p = msg.angular_velocity.x
        q = msg.angular_velocity.y
        r = -1*msg.angular_velocity.z

        ax = msg.linear_acceleration.x
        ay = -1*msg.linear_acceleration.y
        az = msg.linear_acceleration.z+self.GRAVITY

        # from Quaternion to Euler Angle
        euler = MyTF.euler_from_quaternion(a,b,c,d)

        phi = euler[0]
        the = -1*euler[1]
        psi = -1*euler[2]
        self.angle = [phi,the,psi]
        self.angular_velocity = [p,q,r]
        self.linear_acceleration = [ax,ay,az]

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
        self.position = [location.point.x, location.point.y, location.point.z]

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
        self.velocity = [velocity.twist.linear.x, velocity.twist.linear.y, velocity.twist.linear.z]

    def _target_callback(self, msg):
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

        # extract orientaion and convert to euler angle
        a = target_pose.orientation.x
        b = target_pose.orientation.y
        c = target_pose.orientation.z
        d = target_pose.orientation.w
        euler = MyTF.euler_from_quaternion(a,b,c,d)
        target_phi = 0
        target_the = 0
        # target_psi = 0
        target_psi = -1*euler[2]
        self.target_angle = [target_phi, target_the, target_psi]

        # extract position
        self.target_position = [target_pose.position.x, target_pose.position.y, target_pose.position.z]
        # self.target_position = [0, 0, 7]

    def reset(self):
        self.gaz.resetSim()
        obs, reward, done = self._get_obs()
        return obs

    def step(self,action):
        act = Float64MultiArray()
        self.action = action
        act.data = action
        self.action_publisher.publish(act)

        obs, reward, done = self._get_obs()
        return obs, reward, done

    def _get_acts(self):
        action = self.action

        return action

    def _get_obs(self):

        #extend state
        state = []
        state.extend(self.angle)
        state.extend(self.target_angle)
        state.extend(self.angular_velocity)
        state.extend(self.position)
        state.extend(self.target_position)
        state.extend(self.velocity)
        state.extend(self.linear_acceleration)
        # print(state)

        #extend reward
        if self.reward is None:
            reward = -1
        else:
            reward = self.reward.data
        # print(reward)

        #done is not used in this experiment
        done = False

        return state, reward, done


    def sample(self, horizon, policy, record_fname=None):
        """Samples a rollout from the agent.

        Arguments:
            horizon: (int) The length of the rollout to generate from the agent.
            policy: (policy) The policy that the agent will use for actions.
            record_fname: (str/None) The name of the file to which a recording of the rollout
                will be saved. If None, the rollout will not be recorded.

        Returns: (dict) A dictionary containing data from the rollout.
            The keys of the dictionary are 'obs', 'ac', and 'reward_sum'.
        """
        times, rewards = [], []
        O, A, reward_sum, done = [self.reset()], [], 0, False

        policy.reset()
        for t in range(horizon):
            start = time.time()
            A.append(policy.act(O[t], t))
            times.append(time.time() - start)
            # print(self.action)

            if self.noise_stddev is None:
                obs, reward, done = self.step(A[t])

            else:
                na = np.random.normal(loc=0, scale=self.noise_stddev, size=[self.dU]) # noise stdv
                nb = self.ac_ub # noise scale
                action = A[t] + na*nb # add scaled action noise
                action = np.minimum(np.maximum(action, self.ac_lb), self.ac_ub)
                obs, reward, done = self.step(action)
            O.append(obs)
            reward_sum += reward
            rewards.append(reward)
            if done:
                break
            self.RATE.sleep()

        self.record = False
        print("Average action selection time: ", np.mean(times))
        print("Rollout length: ", len(A))

        return {
            "obs": np.array(O),
            "ac": np.array(A),
            "reward_sum": reward_sum,
            "rewards": np.array(rewards),
        }

# if __name__ == "__main__":
#     rospy.init_node('agent_node', anonymous=False)
#     rospy.loginfo("Agent Node Initialising...")

#     cfg = create_config('blimp', "MPC", [], [], '/home/yliu_local/blimpRL_ws/src/RL_log/pets_log')

#     task_hor = 100
#     policy = MPC(cfg.ctrl_cfg)

#     # Agent()
#     # Agent.sample(task_hor, policy)

