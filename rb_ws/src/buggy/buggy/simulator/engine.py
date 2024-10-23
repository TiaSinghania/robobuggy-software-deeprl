#! /usr/bin/env python3
import re
import sys
import threading
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Twist, PoseWithCovariance, TwistWithCovariance
from std_msgs.msg import Float64
from sensor_msgs.msg import NavSatFix
from nav_msgs.msg import Odometry
import numpy as np
import utm

class Simulator(Node):
    # simulator constants:

    def __init__(self):
        super().__init__('sim_single')
        UTM_EAST_ZERO = 589702.87
        UTM_NORTH_ZERO = 4477172.947
        UTM_ZONE_NUM = 17
        UTM_ZONE_LETTER = "T"
        #TODO: make these values accurate
        WHEELBASE_SC = 1.17
        WHEELBASE_NAND= 1.17

        self.starting_poses = {
            "Hill1_NAND": (Simulator.UTM_EAST_ZERO + 0, Simulator.UTM_NORTH_ZERO + 0, -110),
            "Hill1_SC": (Simulator.UTM_EAST_ZERO + 20, Simulator.UTM_NORTH_ZERO + 30, -110),
            "WESTINGHOUSE": (589647, 4477143, -150),
            "UC_TO_PURNELL": (589635, 4477468, 160),
            "UC": (589681, 4477457, 160),
            "TRACK_EAST_END": (589953, 4477465, 90),
            "TRACK_RESNICK": (589906, 4477437, -20),
            "GARAGE": (589846, 4477580, 180),
            "PASS_PT": (589491, 4477003, -160),
            "FREW_ST": (589646, 4477359, -20),
            "FREW_ST_PASS": (589644, 4477368, -20),
            "RACELINE_PASS": (589468.02, 4476993.07, -160),
        }

        self.number_publisher = self.create_publisher(Float64, 'numbers', 1)
        self.i = 0

        # for X11 matplotlib (direction included)
        self.plot_publisher = self.create_publisher(Pose, "sim_2d/utm", 1)

        # simulate the INS's outputs (noise included)
        self.pose_publisher = self.create_publisher(Odometry, "nav/odom", 1)

        self.steering_subscriber = self.create_subscription(
            Float64, "buggy/input/steering", self.update_steering_angle, 1
        )
        # To read from velocity
        self.velocity_subscriber = self.create_subscription(
            Float64, "velocity", self.update_velocity, 1
        )
        self.navsatfix_noisy_publisher = self.create_publisher(
                NavSatFix, "state/pose_navsat_noisy", 1
        )
        if (self.get_namespace() == "/SC"):
            self.buggy_name = "SC"
            init_pose = self.starting_poses["Hill_1SC"]


        if (self.get_namespace() == "/NAND"):
            self.buggy_name = "NAND"
            init_pose = self.starting_poses["Hill_1NAND"]

        self.e_utm, self.n_utm, self.heading = init_pose
        # TODO: do we want to not hard code this
        self.velocity = 12 # m/s
        self.steering_angle = 0  # degrees
        self.rate = 200  # Hz
        self.pub_skip = 2  # publish every pub_skip ticks

        self.lock = threading.Lock()

        freq = 10
        timer_period = 1/freq  # seconds
        self.timer = self.create_timer(timer_period, self.loop)

    def update_steering_angle(self, data: Float64):
        with self.lock:
            self.steering_angle = data.data

    def update_velocity(self, data: Float64):
        with self.lock:
            self.velocity = data.data

    def get_steering_arc(self):
        with self.lock:
            steering_angle = self.steering_angle
        if steering_angle == 0.0:
            return np.inf

        return Simulator.WHEELBASE / np.tan(np.deg2rad(steering_angle))

    def dynamics(self, state, v):
        l = Simulator.WHEELBASE
        _, _, theta, delta = state

        return np.array([v * np.cos(theta),
                         v * np.sin(theta),
                         v / l * np.tan(delta),
                         0])

    def step(self):
        with self.lock:
            heading = self.heading
            e_utm = self.e_utm
            n_utm = self.n_utm
            velocity = self.velocity
            steering_angle = self.steering_angle

        h = 1/self.rate
        state = np.array([e_utm, n_utm, np.deg2rad(heading), np.deg2rad(steering_angle)])
        k1 = self.dynamics(state, velocity)
        k2 = self.dynamics(state + h/2 * k1, velocity)
        k3 = self.dynamics(state + h/2 * k2, velocity)
        k4 = self.dynamics(state + h/2 * k3, velocity)

        final_state = state + h/6 * (k1 + 2 * k2 + 2 * k3 + k4)

        e_utm_new, n_utm_new, heading_new, _ = final_state
        heading_new = np.rad2deg(heading_new)

        with self.lock:
            self.e_utm = e_utm_new
            self.n_utm = n_utm_new
            self.heading = heading_new

    def publish(self):
        p = Pose()
        time_stamp = self.get_clock().now().to_msg()
        with self.lock:
            p.position.x = self.e_utm
            p.position.y = self.n_utm
            p.position.z = self.heading
            velocity = self.velocity

        self.plot_publisher.publish(p)

        (lat, long) = utm.to_latlon(
            p.position.x,
            p.position.y,
            Simulator.UTM_ZONE_NUM,
            Simulator.UTM_ZONE_LETTER,
        )

        nsf = NavSatFix()
        nsf.latitude = lat
        nsf.longitude = long
        nsf.altitude = 260
        nsf.header.stamp = time_stamp
        self.navsatfix_publisher.publish(nsf)

        if Simulator.NOISE:
            lat_noisy = lat + np.random.normal(0, 1e-6)
            long_noisy = long + np.random.normal(0, 1e-6)
            nsf_noisy = NavSatFix()
            nsf_noisy.latitude = lat_noisy
            nsf_noisy.longitude = long_noisy
            nsf_noisy.header.stamp = time_stamp
            self.navsatfix_noisy_publisher.publish(nsf_noisy)

        odom = Odometry()
        odom.header.stamp = time_stamp

        odom_pose = Pose()
        odom_pose.position.x = long
        odom_pose.position.y = lat
        odom_pose.position.z = 260

        odom_pose.orientation.z = np.sin(np.deg2rad(self.heading) / 2)
        odom_pose.orientation.w = np.cos(np.deg2rad(self.heading) / 2)

        odom_twist = Twist()
        odom_twist.linear.x = velocity

        odom.pose = PoseWithCovariance(pose=odom_pose)
        odom.twist = TwistWithCovariance(twist=odom_twist)

        self.pose_publisher.publish(odom)

    def loop(self):
        rate = self.create_rate(self.rate)
        pub_tick_count = 0

        self.step()

        if pub_tick_count == self.pub_skip:
            self.publish()
            pub_tick_count = 0
        else:
            pub_tick_count += 1



def main(args=None):
    rclpy.init(args=args)
    sim = Simulator()
    rclpy.spin(sim)
    rclpy.shutdown()

if __name__ == "__main__":
    main()