import rclpy
from rclpy.node import Node

from std_msgs.msg import Bool

class Watchdog(Node):

    def __init__(self):
        """
        Constructor for Watchdog class.

        Creates a ROS node with a publisher that periodically sends a message
        indicating whether the node is still alive.
        
        """
        super().__init__('watchdog')
        self.heartbeat_publisher = self.create_publisher(Bool, 'self/debug/heartbeat', 1)
        timer_period = 0.01  # seconds (10 Hz)
        self.timer = self.create_timer(timer_period, self.loop)
        self.i = 0 # Loop Counter

    def loop(self):
        # Loop for the code that operates at 0.1 Hz
        msg = Bool()
        msg.data = True
        self.heartbeat_publisher.publish(msg) 


def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = Watchdog()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()