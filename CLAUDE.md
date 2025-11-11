# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Deep RL (10-703) Fall Project based on [robobuggy-software](https://github.com/CMU-Robotics-Club/robobuggy-software). The project implements a Gymnasium-compatible reinforcement learning environment for training autonomous buggy navigation on the CMU buggy course.

## Development Setup

### Dependencies
Install Python dependencies:
```bash
pip install -r python-requirements.txt
```

Key dependencies include:
- gymnasium (for RL environment)
- numpy, scipy (numerical computation)
- matplotlib (visualization)
- utm, pymap3d, pyproj (coordinate transformations)
- osqp (optimization)

### Python Version
Python 3.12+ (tested with 3.12.12)

## Architecture

### Core Components

#### 1. Gymnasium Environment (`scripts/simulator/environment.py`)
- **Class**: `BuggyCourseEnv(gym.Env)`
- Implements the main RL training environment following Gymnasium API
- Simulates two buggies racing simultaneously:
  - **SC buggy**: Policy-controlled (the agent being trained)
  - **NAND buggy**: Classical Stanley controller (baseline opponent)
- **Observation space** (7D): SC buggy state (easting, northing, theta, speed, delta) + NAND buggy position (easting, northing)
- **Action space**: Continuous steering percentage [-1, 1] scaled to ±π/9 radians
- **Dynamics**: Bicycle model with RK4 integration for physics simulation
- **Reward function**: Negative squared distance from target trajectory point (scripts/simulator/environment.py:143-154)
  - Currently has known issue: optimal strategy is beeline to goal (no curb constraints implemented yet)

#### 2. Buggy State Management (`scripts/util/buggy.py`)
- **Dataclass**: `Buggy`
- Represents buggy state vector: position (UTM coordinates), speed, heading, steering angle
- State vector is 4D: [e_utm, n_utm, speed, theta]
- Constants: wheelbase, angle_clip (max steering angle)
- Provides methods for state access in different formats (full state, self observation, other observation)

#### 3. Trajectory Management (`scripts/util/trajectory.py`)
- **Class**: `Trajectory`
- Loads reference trajectories from JSON files (lat/lon waypoints)
- Converts coordinates to UTM (easting/northing)
- Interpolates waypoints using CubicSpline or Akima1DInterpolator
- Key methods:
  - `get_closest_index_on_path()`: Find nearest trajectory point to buggy position
  - `get_heading_by_index()`: Get reference heading at trajectory point
  - `get_curvature_by_index()`: Compute path curvature for steering
  - `get_distance_from_index()`: Track progress along trajectory
- Trajectory files located in `scripts/util/buggycourse_*.json`
- JSON format: array of objects with `{lat, lon, key, active}` fields
- Create trajectories using: https://rdudhagra.github.io/eracer-portal/

#### 4. Stanley Controller (`scripts/controller/stanley_controller.py`)
- **Class**: `StanleyController`
- Classical path-tracking controller for NAND buggy (baseline)
- Implements Stanford's Stanley controller algorithm
- Control law combines:
  - Heading error: difference between buggy heading and path heading
  - Cross-track error: perpendicular distance from path
- Key parameters (tuned for this course):
  - `CROSS_TRACK_GAIN = 1.3`
  - `K_SOFT = 1.0` (m/s)
  - `K_D_YAW = 0.012` (rad/(rad/s))
- Uses front axle as reference point
- Paper reference: https://ai.stanford.edu/~gabeh/papers/hoffmann_stanley_control07.pdf

### Coordinate Systems
- **Input**: GPS coordinates (latitude, longitude) in trajectory JSON files
- **Internal**: UTM coordinates (easting, northing) in meters
- All physics and control calculations done in UTM for Cartesian simplicity
- Conversion handled by `utm` library in trajectory loading

### Buggy Parameters
- **NAND wheelbase**: 1.3 m
- **SC wheelbase**: 1.104 m
- **Max steering angle**: π/9 radians (~20 degrees)

### Initial Conditions (scripts/simulator/environment.py:36-38)
- **SC buggy**: (589761.40, 4477321.07, -1.91986) - UTM easting, northing, heading
- **NAND buggy**: (589751.46, 4477322.07, -1.91986)
- **SC speed**: 12 m/s
- **NAND speed**: 6 m/s
- **Target finish line**: UTM (589693.75, 4477191.05)

### Simulation Details
- Default simulation rate: 100 Hz (configurable in `BuggyCourseEnv.__init__`)
- Physics integration: 4th-order Runge-Kutta (RK4) for accurate dynamics
- Bicycle model dynamics implemented in `_dynamics()` method

## Current Development Status

Based on git history:
- Gymnasium environment structure is defined
- Stanley controller implemented for NAND buggy
- Physics simulation with bicycle model complete
- `scripts/runner.py` is currently empty (main training script not yet implemented)

## Known Issues and TODOs

1. **Reward function limitation** (scripts/simulator/environment.py:147): Currently no curb constraints, so optimal strategy is beeline to goal flag
2. **Environment wrappers** (scripts/simulator/environment.py:5-6): TODO to implement Gymnasium wrappers for environment variations
3. **Render method** (scripts/simulator/environment.py:181): Not yet implemented for visualization
4. **TrajectoryMsg import** (scripts/util/trajectory.py:4): Commented out ROS message dependency (pack/unpack methods may not work)
5. **Runner script**: Empty file needs training loop implementation

## Important Implementation Notes

- All angles in radians internally (except where noted in comments as degrees)
- UTM coordinates are in meters, zone 17T for CMU campus
- Steering angle (delta) is positive for left turn by bicycle model convention
- Trajectory interpolation uses time-step dt=0.01 for distance calculations
- Stanley controller searches trajectory indices within window [current-20, current+50] for efficiency
