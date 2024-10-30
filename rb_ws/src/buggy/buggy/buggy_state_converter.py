#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import numpy as np
import pyproj


class BuggyStateConverter(Node):
    def __init__(self):
        super().__init__("buggy_state_converter")

        # Determine namespace to configure the subscriber and publisher correctly
        namespace = self.get_namespace()

        # Create publisher and subscriber for "self" namespace
        self.self_raw_state_subscriber = self.create_subscription(
            Odometry, "self/raw_state", self.self_raw_state_callback, 10
        )
        self.self_state_publisher = self.create_publisher(Odometry, "self/state", 10)

        # Check if namespace is "/SC" to add an additional "other" subscriber/publisher
        if namespace == "/SC":
            self.other_raw_state_subscriber = self.create_subscription(
                Odometry, "other/raw_state", self.other_raw_state_callback, 10
            )
            self.other_state_publisher = self.create_publisher(Odometry, "other/state", 10)

        # Initialize pyproj Transformer for ECEF -> UTM conversion for /SC
        self.ecef_to_utm_transformer = pyproj.Transformer.from_crs(
            "epsg:4978", "epsg:32617", always_xy=True
        )  # Check UTM EPSG code, using EPSG:32617 for UTM Zone 17N

    def self_raw_state_callback(self, msg):
        """ Callback for processing self/raw_state messages and publishing to self/state """
        namespace = self.get_namespace()

        if namespace == "/SC":
            converted_msg = self.convert_SC_state(msg)
        elif namespace == "/NAND":
            converted_msg = self.convert_NAND_state(msg)
        else:
            self.get_logger().warn("Namespace not recognized for buggy state conversion.")
            return

        # Publish the converted message to self/state
        self.self_state_publisher.publish(converted_msg)

    def other_raw_state_callback(self, msg):
        """ Callback for processing other/raw_state messages and publishing to other/state """
        # Convert the /SC message and publish to other/state
        converted_msg = self.convert_SC_state(msg)
        self.other_state_publisher.publish(converted_msg)

    def convert_SC_state(self, msg):
        """ Converts /SC namespace raw state to clean state units and structure """
        converted_msg = Odometry()
        converted_msg.header = msg.header

        # ---- 1. Convert ECEF Position to UTM Coordinates ----
        ecef_x = msg.pose.pose.position.x
        ecef_y = msg.pose.pose.position.y
        ecef_z = msg.pose.pose.position.z

        # Convert ECEF to UTM
        utm_x, utm_y, _ = self.ecef_to_utm_transformer.transform(ecef_x, ecef_y, ecef_z)
        converted_msg.pose.pose.position.x = utm_x  # UTM Easting
        converted_msg.pose.pose.position.y = utm_y  # UTM Northing
        converted_msg.pose.pose.position.z = 0.0    # ignored

        # ---- 2. Convert Quaternion to Heading (Radians) ----
        qx, qy, qz, qw = msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w
        yaw = self.quaternion_to_yaw(qx, qy, qz, qw)
        converted_msg.pose.pose.orientation.x = yaw
        # ignored:
        # converted_msg.pose.pose.orientation.y = qy
        # converted_msg.pose.pose.orientation.z = qz
        # converted_msg.pose.pose.orientation.w = qw

        # ---- 3. Copy Covariances (Unchanged) ----
        converted_msg.pose.covariance = msg.pose.covariance
        converted_msg.twist.covariance = msg.twist.covariance

        # ---- 4. Copy Linear Velocities ----
        converted_msg.twist.twist.linear.x = msg.twist.twist.linear.x   # m/s in x-direction
        converted_msg.twist.twist.linear.y = msg.twist.twist.linear.y   # m/s in x-direction
        converted_msg.twist.twist.linear.z = msg.twist.twist.linear.z   # Keep original Z velocity (??)

        # ---- 5. Copy Angular Velocities ----
        converted_msg.twist.twist.angular.x = msg.twist.twist.angular.z   # rad/s for heading change rate (using yaw from twist.angular)
        converted_msg.twist.twist.angular.y = 0.0   # ignored
        converted_msg.twist.twist.angular.z = 0.0   # ignored

        return converted_msg

    def convert_NAND_state(self, msg):
        """ Converts /NAND namespace raw state to clean state units and structure """
        converted_msg = Odometry()
        converted_msg.header = msg.header

        # ---- 1. Directly use UTM Coordinates ----
        converted_msg.pose.pose.position.x = msg.pose.pose.position.x   # UTM Easting
        converted_msg.pose.pose.position.y = msg.pose.pose.position.y   # UTM Northing
        converted_msg.pose.pose.position.z = 0.0                        # ignored

        # ---- 2. Orientation in Radians with 0 at East ----
        converted_msg.pose.pose.orientation.x = msg.pose.pose.orientation.x
        # ignored:
        # converted_msg.pose.pose.orientation.y = msg.pose.pose.orientation.y
        # converted_msg.pose.pose.orientation.z = msg.pose.pose.orientation.z
        # converted_msg.pose.pose.orientation.w = msg.pose.pose.orientation.w

        # ---- 3. Copy Covariances (Unchanged) ----
        converted_msg.pose.covariance = msg.pose.covariance
        converted_msg.twist.covariance = msg.twist.covariance

        # ---- 4. Copy Linear Velocities ----
        # CHECK: ROS serial translator node must change scalar speed to velocity x/y components before pushing to raw_state
        converted_msg.twist.twist.linear.x = msg.twist.twist.linear.x   # m/s in x-direction
        converted_msg.twist.twist.linear.y = msg.twist.twist.linear.y   # m/s in y-direction
        converted_msg.twist.twist.linear.z = msg.twist.twist.linear.z   # Keep original Z velocity (??)

        # ---- 5. Copy Angular Velocities ----
        converted_msg.twist.twist.angular.x = msg.twist.twist.angular.x
        converted_msg.twist.twist.angular.y = 0.0
        converted_msg.twist.twist.angular.z = 0.0

        return converted_msg

    def quaternion_to_yaw(self, qx, qy, qz, qw):
        """Convert a quaternion to yaw (heading) in radians."""
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        return np.arctan2(siny_cosp, cosy_cosp)


def main(args=None):
    rclpy.init(args=args)

    # Create the BuggyStateConverter node and spin it
    state_converter = BuggyStateConverter()
    rclpy.spin(state_converter)

    # Shutdown when done
    state_converter.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
