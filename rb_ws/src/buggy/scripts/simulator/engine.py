#! /usr/bin/env python3
import sys
import time
import rclpy
from std_msgs.msg import String

class Simulator():
    # simulator constants:

    def __init__(self):
        self.test_publisher = self.create_publisher(String, 'test', 10)
        if (self.get_namespace() == "SC"):
            self.buggy_name = "SC"

        if (self.get_namespace() == "NAND"):
            self.buggy_name = "NAND"

    def loop(self):
        print("hello")
        self.test_publisher.publish(self.buggy_name)

if __name__ == "__main__":
    rclpy.init()
    sim = Simulator()

    # publish initial position, then sleep
    # so that auton stack has time to initialize
    # before buggy moves
    time.sleep(15.0)
    sim.loop()


