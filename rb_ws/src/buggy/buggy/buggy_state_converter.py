#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import numpy as np
import pyproj


class BuggyStateConverter(Node):
    def __init__(self):
        super().__init__("buggy_state_converter")

        # ROS2 Subscriber: /ekf/odometry_earth
        self.subscription = self.create_subscription(
            Odometry, "/ekf/odometry_earth", self.ekf_odometry_callback, 10
        )

        # ROS2 Publisher: /nav/odometry_earth
        self.publisher_ = self.create_publisher(Odometry, "/nav/odometry_earth", 10)

        # Initialize pyproj Transformer for ECEF -> UTM conversion
        self.ecef_to_utm_transformer = pyproj.Transformer.from_crs(
            "epsg:4978", "epsg:32633", always_xy=True
        )  # Change EPSG as required

    def ekf_odometry_callback(self, msg):
        """Callback function to process incoming odometry message, convert it, and publish the transformed message"""

        # Create a new Odometry message for the output
        converted_msg = Odometry()
        converted_msg.header = msg.header

        # ---- 1. Convert ECEF Position to UTM Coordinates ----
        ecef_x = msg.pose.pose.position.x
        ecef_y = msg.pose.pose.position.y
        ecef_z = msg.pose.pose.position.z

        # Convert ECEF to UTM using pyproj
        utm_x, utm_y, _ = self.ecef_to_utm_transformer.transform(ecef_x, ecef_y, ecef_z)

        # Set the converted UTM position in the new message
        converted_msg.pose.pose.position.x = utm_x  # UTM Easting
        converted_msg.pose.pose.position.y = utm_y  # UTM Northing
        converted_msg.pose.pose.position.z = 0.0  # Ignored in this context

        # ---- 2. Convert Quaternion to Heading (Radians) ----
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w

        # Convert quaternion to euler angles (roll, pitch, yaw)
        yaw = self.quaternion_to_yaw(qx, qy, qz, qw)

        # Store the heading in the x component of the orientation (0 is East)
        converted_msg.pose.pose.orientation.x = yaw

        # ---- 3. Copy Covariances (Unchanged) ----
        converted_msg.pose.covariance = msg.pose.covariance
        converted_msg.twist.covariance = msg.twist.covariance

        # ---- 4. Convert Linear Velocities ----
        converted_msg.twist.twist.linear.x = (
            msg.twist.twist.linear.x
        )  # m/s in x-direction
        converted_msg.twist.twist.linear.y = (
            msg.twist.twist.linear.y
        )  # m/s in y-direction
        converted_msg.twist.twist.linear.z = (
            msg.twist.twist.linear.z
        )  # Keep original Z velocity (??)

        # ---- 5. Convert Angular Velocities (rad/s for heading rate of change) ----
        converted_msg.twist.twist.angular.x = (
            msg.twist.twist.angular.z
        )  # rad/s for heading change rate
        converted_msg.twist.twist.angular.y = (
            msg.twist.twist.angular.y
        )  # Keep original Y angular velocity (??)
        converted_msg.twist.twist.angular.z = (
            msg.twist.twist.angular.z
        )  # Keep original Z angular velocity (??)

        # Publish the converted message
        self.publisher_.publish(converted_msg)

    def quaternion_to_yaw(self, qx, qy, qz, qw):
        """Convert a quaternion to yaw (heading) in radians."""
        # Extract yaw (z-axis rotation) from quaternion
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        return np.arctan2(siny_cosp, cosy_cosp)


def main(args=None):
    rclpy.init(args=args)

    # Create the BuggyStateConverter node and spin it
    state_converter = BuggyStateConverter()
    # while rclpy.ok(): <- is this needed?
    rclpy.spin(state_converter)

    # Shutdown when done
    state_converter.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
