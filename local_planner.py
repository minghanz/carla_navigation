#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" This module contains a local planner to perform low-level waypoint following based on PID controllers. """

from enum import Enum
from collections import deque
import random
import math
import numpy as np

import carla
from agents.navigation.controller import PurePersuitController   
from agents.tools.misc import distance_vehicle, draw_waypoints
from agents.tools.misc import get_speed
from agents.navigation.EnvironmentState import EnvironmentState
from agents.navigation.Cognition import CognitionState
from agents.navigation.Decision import Decision


class RoadOption(Enum):
    """
    RoadOption represents the possible topological configurations when moving from a segment of lane to other.
    """
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4
    CHANGELANELEFT = 5
    CHANGELANERIGHT = 6


def distance_between_two_loc(loc1,loc2):
    d = (loc1.x-loc2.x)*(loc1.x-loc2.x)+(loc1.y-loc2.y)*(loc1.y-loc2.y)
    d = np.sqrt(d)
    return d

def angle_between_ego_vehicle_direction_and_point(ego_vehicle_transform,point_location):

    vehicle_transform = ego_vehicle_transform
    v_begin = vehicle_transform.location
    v_end = v_begin + carla.Location(x=math.cos(math.radians(vehicle_transform.rotation.yaw)),
                                        y=math.sin(math.radians(vehicle_transform.rotation.yaw)))

    v_vec = np.array([v_end.x - v_begin.x, v_end.y - v_begin.y, 0.0])

    target_x = point_location.x
    target_y = point_location.y

    w_vec = np.array([target_x-
                        v_begin.x, target_y -
                        v_begin.y, 0.0])

    _dot = math.acos(np.clip(np.dot(w_vec, v_vec) /
                         (np.linalg.norm(w_vec) * np.linalg.norm(v_vec)), -1.0, 1.0))

    _cross = np.cross(v_vec, w_vec)
    if _cross[2] < 0:
        _dot *= -1.0

    return _dot


class LocalPlanner(object):
    """
    LocalPlanner implements the basic behavior of following a trajectory of waypoints that is generated on-the-fly.
    The low-level motion of the vehicle is computed by using two PID controllers, one is used for the lateral control
    and the other for the longitudinal control (cruise speed).

    When multiple paths are available (intersections) this local planner makes a random choice.
    """

    # minimum distance to target waypoint as a percentage (e.g. within 90% of
    # total distance)
    MIN_DISTANCE_PERCENTAGE = 0.9

    def __init__(self, vehicle, opt_dict=None):
        """
        :param vehicle: actor to apply to local planner logic onto
        :param opt_dict: dictionary of arguments with the following semantics:
            dt -- time difference between physics control in seconds. This is typically fixed from server side
                  using the arguments -benchmark -fps=F . In this case dt = 1/F

            target_speed -- desired cruise speed in Km/h

            sampling_radius -- search radius for next waypoints in seconds: e.g. 0.5 seconds ahead

            lateral_control_dict -- dictionary of arguments to setup the lateral PID controller
                                    {'K_P':, 'K_D':, 'K_I':, 'dt'}

            longitudinal_control_dict -- dictionary of arguments to setup the longitudinal PID controller
                                        {'K_P':, 'K_D':, 'K_I':, 'dt'}
        """
        self._vehicle = vehicle
        self._map = self._vehicle.get_world().get_map()

        self._dt = 1.0 / 20.0
        self._target_speed = 20.0/3.6  # m/s
        self._target_road_option = None
        self.target_waypoint = None
        self._vehicle_controller = PurePersuitController()
        self._global_plan = False
        # queue with tuples of (waypoint, RoadOption)
        self._waypoints_queue = deque(maxlen=20000)
        self._buffer_size = 200
        self._waypoint_buffer = deque(maxlen=self._buffer_size)    ## Trajectory
        self.EnvironmentInfo = EnvironmentState(vehicle)
        self.CognitionState = CognitionState()
        self.Decision = Decision()
    
        self.local_path = deque(maxlen=50)

    def __del__(self):
        if self._vehicle:
            self._vehicle.destroy()
        print("Destroying ego-vehicle!")

    def reset_vehicle(self):
        self._vehicle = None
        print("Resetting ego-vehicle!")


    def set_speed(self, speed):
        """
        Request new target speed.

        :param speed: new target speed in Km/h
        :return:
        """
        self._target_speed = speed


    def set_global_plan(self, current_plan):
        self._waypoints_queue.clear()
        for elem in current_plan:
            self._waypoints_queue.append(elem)
        self._target_road_option = RoadOption.LANEFOLLOW
        self._global_plan = True


    def _generate_local_path(self):

        vehicle_transform = self.EnvironmentInfo.ego_vehicle_transform
        while self._waypoint_buffer:
            (waypoint,_) = self._waypoint_buffer[0]
            angle = angle_between_ego_vehicle_direction_and_point(vehicle_transform,waypoint.transform.location)
            if abs(angle) > np.pi/2:
                self._waypoint_buffer.popleft()
            else:
                break

        while self._waypoints_queue and len(self._waypoint_buffer)<self._buffer_size:
            waypoint_tuple =  self._waypoints_queue.popleft()
            waypoint = waypoint_tuple[0]
            self._waypoint_buffer.append(waypoint_tuple)

        self.local_path.clear()
        for i in range(0,49):
            if len(self._waypoint_buffer)>i:
                waypoint = self._waypoint_buffer[i]
                self.local_path.append(waypoint)
            else:
                break


    def run_step(self, debug=True):
        """
        Execute one step of local planning which involves running the longitudinal and lateral PID controllers to
        follow the waypoints trajectory.

        :param debug: boolean flag to activate waypoints debugging
        :return:
        """

        if len(self._waypoints_queue) == 0 and len(self._waypoint_buffer) == 0:
            control = carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 1.0
            control.hand_brake = False
            control.manual_gear_shift = False

            return control

        # Environment Perception
        self.EnvironmentInfo.perception()
        
        # Get reference plan
        self._generate_local_path()

        # Cognition
        self.CognitionState.scenario_cognition(self._waypoint_buffer,self.EnvironmentInfo)

        # Decision
        target_speed, self.target_waypoint = self.Decision.generate_decision(self.local_path,self.EnvironmentInfo,self.CognitionState)
        self.set_speed(target_speed)

        # Control
        control = self._vehicle_controller.run_step(self._target_speed, self.target_waypoint, self.EnvironmentInfo)
        
        # Print States
        print("target speed:",self._target_speed*3.6,"current speed:",self.EnvironmentInfo.ego_vehicle_speed*3.6)

        if True:
            world = self._vehicle.get_world()
            begin = carla.Location(x=self.EnvironmentInfo.ego_vehicle_location.x, y=self.EnvironmentInfo.ego_vehicle_location.y, z=self.EnvironmentInfo.ego_vehicle_location.z+1)
            end = carla.Location(x=self.target_waypoint.x,y=self.target_waypoint.y,z=self.target_waypoint.z+0.5)
            world.debug.draw_arrow(begin, end, arrow_size=0.3, life_time=1.0)

        return control


def _retrieve_options(list_waypoints, current_waypoint):
    """
    Compute the type of connection between the current active waypoint and the multiple waypoints present in
    list_waypoints. The result is encoded as a list of RoadOption enums.

    :param list_waypoints: list with the possible target waypoints in case of multiple options
    :param current_waypoint: current active waypoint
    :return: list of RoadOption enums representing the type of connection from the active waypoint to each
             candidate in list_waypoints
    """
    options = []
    for next_waypoint in list_waypoints:
        # this is needed because something we are linking to
        # the beggining of an intersection, therefore the
        # variation in angle is small
        next_next_waypoint = next_waypoint.next(3.0)[0]
        link = _compute_connection(current_waypoint, next_next_waypoint)
        options.append(link)

    return options


def _compute_connection(current_waypoint, next_waypoint):
    """
    Compute the type of topological connection between an active waypoint (current_waypoint) and a target waypoint
    (next_waypoint).

    :param current_waypoint: active waypoint
    :param next_waypoint: target waypoint
    :return: the type of topological connection encoded as a RoadOption enum:
             RoadOption.STRAIGHT
             RoadOption.LEFT
             RoadOption.RIGHT
    """
    n = next_waypoint.transform.rotation.yaw
    n = n % 360.0

    c = current_waypoint.transform.rotation.yaw
    c = c % 360.0

    diff_angle = (n - c) % 180.0
    if diff_angle < 1.0:
        return RoadOption.STRAIGHT
    elif diff_angle > 90.0:
        return RoadOption.LEFT
    else:
        return RoadOption.RIGHT
