### get cognition and state space


import math

import numpy as np

import carla
from agents.tools.misc import get_speed
from agents.navigation.EnvironmentState import EnvironmentState,LaneState,Surrounding_vehicle

def distance_between_two_loc(loc1,loc2):
    d = (loc1.x-loc2.x)*(loc1.x-loc2.x)+(loc1.y-loc2.y)*(loc1.y-loc2.y)
    d = np.sqrt(d)
    return d


def location_on_the_path(local_path,location,sensitive_range):

    v_loc = location
    d_to_waypoints = []
    for (waypoint,_) in local_path:
        w_loc = waypoint.transform.location
        d = distance_between_two_loc(v_loc,w_loc)
        d_to_waypoints.append(d)

    d_to_waypoints.sort()
    if d_to_waypoints[0]+d_to_waypoints[1] < sensitive_range:
        return True

    return False


def location_on_the_path_decouple(local_path,location,sensitive_range):

    v_loc = location
    d_to_waypoints = []
    for waypoint in local_path:
        w_loc = waypoint.transform.location
        d = distance_between_two_loc(v_loc,w_loc)
        d_to_waypoints.append(d)

    d_to_waypoints.sort()
    if d_to_waypoints[0]+d_to_waypoints[1] < sensitive_range:
        return True

    return False




class CognitionState(object):
    """
    AV driving cognition state
    """
    def __init__(self):
        """
        rightest lane = 1
        """
        self.follow_path = True
        self.ego_y = None
        self.lane_list = None
        self.target_lane_id = None
        self.length_before_follow_lane = None


    def scenario_cognition(self,reference_path,EnvironmentInfo):
        self.follow_path = True
        self.lane_list = None
        self.ego_y = None


        is_multilane = EnvironmentInfo.is_multilane(EnvironmentInfo.ego_vehicle_location)
        if is_multilane:
            # self.follow_path = False
            
            self._generate_lane_list(EnvironmentInfo,reference_path)
            
            self._find_target_lane_id(reference_path,EnvironmentInfo)

            print("target:",self.target_lane_id,self.length_before_follow_lane,"ego:",self.ego_y)

            self._should_follow_lane(reference_path,EnvironmentInfo,is_multilane)

            

    def _generate_lane_list(self,EnvironmentInfo,reference_path):
        
        self.lane_list = []

        central_lane = EnvironmentInfo.get_lane(0,reference_path)
        central_lane.id = 0
        self.lane_list.append(central_lane)
        
        left_lane_num = 1
        left_lane = EnvironmentInfo.get_lane(left_lane_num,reference_path)
        while left_lane is not None:
            left_lane.id = left_lane_num
            self.lane_list.append(left_lane)
            left_lane_num += 1
            left_lane = EnvironmentInfo.get_lane(left_lane_num,reference_path)

        left_lane_num += -1

        right_lane_num = -1
        right_lane = EnvironmentInfo.get_lane(right_lane_num,reference_path)
        while right_lane is not None:
            right_lane.id = right_lane_num
            self.lane_list.append(right_lane)
            right_lane_num += -1
            right_lane = EnvironmentInfo.get_lane(right_lane_num,reference_path)

        right_lane_num += 1

        for lane in self.lane_list:
            lane.id = lane.id - right_lane_num + 1
        self.ego_y = 0 - right_lane_num + 1


    def _find_target_lane_id(self,reference_path,EnvironmentInfo):

        free_driving = True
        min_length = 180

        for lane in self.lane_list:
            print(lane.id,lane.length_before_interaction)
            if lane.length_before_interaction < min_length:
                min_length = lane.length_before_interaction

        if min_length < 180:
            free_driving = False
        
        if free_driving:
            self.target_lane_id = -1
            self.length_before_follow_lane = -1
            return
        
        self.length_before_follow_lane = min_length

        target_id = -2
        for lane in self.lane_list:
            last_waypoint = lane.central_point_list[-1]
            if location_on_the_path(reference_path,last_waypoint.transform.location,6):
                target_id = lane.id
                break

        self.target_lane_id = target_id
        self.length_before_follow_lane = min_length
                
    def _should_follow_lane(self,reference_path,EnvironmentInfo,is_multilane):

        if is_multilane:
            self.follow_path = False

        if self.length_before_follow_lane > 0 and self.length_before_follow_lane < 50 and self.ego_y is not self.target_lane_id:
            self.follow_path = True

        
    def _find_rear_vehicle_on_target_lane(self,lane,EnvironmentInfo):


        min_distance = 100
        rear_vehicle = None

        for target_vehicle in EnvironmentInfo.surrounding_vehicle_list:
            location_list = EnvironmentInfo.longitudinal_position_after_distance(target_vehicle,EnvironmentInfo.sensor_range+10)
            if len(location_list) < 1:
                continue
            loc_target_vehicle = target_vehicle.location
            for target_location in location_list:
                if location_on_the_path_decouple(lane.central_point_list,target_location,EnvironmentInfo.lane_step+3):
                    if distance_between_two_loc(loc_target_vehicle,EnvironmentInfo.ego_vehicle_location) < min_distance:
                        min_distance = distance_between_two_loc(loc_target_vehicle,EnvironmentInfo.ego_vehicle_location)
                        rear_vehicle = target_vehicle

        
        return rear_vehicle
            





