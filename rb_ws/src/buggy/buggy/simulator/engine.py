#! /usr/bin/env python3
import sys
import time
import rclpy
from rclpy.node import Node

from std_msgs.msg import String, Float64

class Simulator(Node):
    # simulator constants:

    def __init__(self):
        super().__init__('SC_sim_single')
        self.get_logger().info("INITIALIZING")
        self.number_publisher = self.create_publisher(Float64, 'numbers', 1)
        self.test_publisher = self.create_publisher(String, 'test', 1)
        self.i = 0

        self.buggy_name = "NONE"

        if (self.get_namespace() == "/SC"):
            self.buggy_name = "SC"

        if (self.get_namespace() == "/NAND"):
            self.buggy_name = "NAND"

        freq = 10
        timer_period = 1/freq  # seconds
        self.timer = self.create_timer(timer_period, self.loop)

    def loop(self):

        self.get_logger().info("LOOPING")

        msg = String()
        msg.data = self.buggy_name
        self.test_publisher.publish(msg)
        msg2 = Float64()
        msg2.data = float(self.i)
        self.number_publisher.publish(msg2)
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    sim = Simulator()
    sim.get_logger().info("CREATED NODE")
    rclpy.spin(sim)


