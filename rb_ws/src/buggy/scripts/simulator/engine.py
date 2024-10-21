#! /usr/bin/env python3
import sys
import time
import rclpy
from rclpy.node import Node

class Simulator(Node):
    # simulator constants:

    def __init__(self):
        if (self.get_namespace() == "SC"):
            self.buggy_name = "SC"
        if (self.get_namespace() == "NAND"):
            self.buggy_name = "NAND"

    if __name__ == "__main__":
        rclpy.init()
        sim = Simulator()
        rclpy.spin(sim)

        # publish initial position, then sleep
        # so that auton stack has time to initialize
        # before buggy moves
        time.sleep(15.0)
        sim.loop()


