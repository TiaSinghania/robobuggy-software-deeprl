from dataclasses import dataclass

"""
Dataclass representing a buggy, mainly should just be holding buggy attributes

State Vector:
n_utm - position northing (utm)
e_utm - position easting (utm)
speed - speed (m/s)
theta - heading (degrees)

Constants:
wheelbase (length from center of buggy to front wheel) (m)
angle_clip - the max/min value that each buggy can steer (degrees)

"""
# TODO: these are the same for now bc i didnt want to think that hard, make them more accurate to true state + observations
# both need more info added (everything in a BuggyState/ ROS Odometry that we track for Stanley)
@dataclass
class Buggy:
    n_utm: float
    e_utm: float
    speed: float
    theta: float
    wheelbase: float
    angle_clip: float

@dataclass
class BuggyObs:
    n_utm: float
    e_utm: float
    speed: float
    theta: float
    wheelbase: float
    angle_clip: float


