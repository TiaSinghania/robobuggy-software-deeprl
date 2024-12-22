import threading

import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32, Float64, Bool
from nav_msgs.msg import Odometry

class Controller(Node):

    def __init__(self):
        """
        Constructor for Controller class.

        Creates a ROS node with a publisher that periodically sends a message
        indicating whether the node is still alive.
        
        """
        super().__init__('watchdog')
            

        #Parameters
        self.declare_parameter("controller_name", "stanley")
        self.local_controller_name = self.get_parameter("controller_name")

        # Publishers
        self.init_check_publisher = self.create_subscription(Bool,
            "debug/init_safety_check", queue_size=1
        )
        self.steer_publisher = self.create_subscription.Publisher(
            Float64, "/buggy/steering", queue_size=1
        )
        self.heading_publisher = self.create_subscription.Publisher(
            Float32, "/auton/debug/heading", queue_size=1
        )
        self.distance_publisher = self.create_subscription.Publisher(
            Float64, "/auton/debug/distance", queue_size=1
        )

        # Subscribers
        self.odom_subscriber = self.create_subscription(Odometry, 'self/buggy/state', self.odom_listener, 1)
        self.traj_subscriber = self.create_subscription(Odometry, 'self/cur_traj', self.traj_listener, 1)

        self.lock = threading.Lock()
        self.ticks = 0
        #TODO: FIGURE OUT WHAT THESE ARE BEFORE MERGING
        self.self_odom_msg = None
        self.gps_odom_msg = None
        self.other_odom_msg = None
        self.use_gps_pos = False

        timer_period = 0.01  # seconds (100 Hz)
        self.timer = self.create_timer(timer_period, self.loop)
        self.i = 0 # Loop Counter

        #

    def loop(self):
        # Loop for the code that operates every 10ms
        msg = Bool()
        msg.data = True
        self.heartbeat_publisher.publish(msg)

    def odom_listener(self, msg : Odometry):
        '''
        This is the subscriber that updates the buggies state for navigation
        msg, should be a CLEAN state as defined in the wiki
        '''
        raise NotImplemented
    
    def traj_listener(self, msg): #TYPE UNKOWN AS OF NOW?? CUSTOM TYPE WHEN #TODO: FIGURE OUT BEFORE MERGE
        '''
        This is the subscriber that updates the buggies trajectory for navigation
        '''
        raise NotImplemented


def main(args=None):
    rclpy.init(args=args)

    watchdog = Controller()

    rclpy.spin(watchdog)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    watchdog.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()