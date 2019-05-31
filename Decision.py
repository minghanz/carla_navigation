#Generated decisions

from collections import deque
import math

import numpy as np

import carla
from agents.tools.misc import get_speed
from agents.navigation.EnvironmentState import EnvironmentState, Surrounding_vehicle, Surrounding_pedestrian
from agents.navigation.Cognition import CognitionState


def distance_between_two_loc(loc1,loc2):
    d = (loc1.x-loc2.x)*(loc1.x-loc2.x)+(loc1.y-loc2.y)*(loc1.y-loc2.y)
    d = np.sqrt(d)
    return d

def location_on_the_path(local_path,location,sensitive_range):

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

def decouple_local_path(local_path):

    decouple_local_path = []
    for (waypoint,_) in local_path:
        decouple_local_path.append(waypoint)

    return decouple_local_path


def waypoint_on_local_path_after_d(local_path,distance):
    w_last_loc = local_path[0].transform.location
    target_distance = distance
    target_location = None
    distance_to_ego = 0
    for waypoint in local_path:
        w_loc = waypoint.transform.location
        distance_to_ego += distance_between_two_loc(w_loc,w_last_loc)
        if distance_to_ego > target_distance:
            w_diff_vec = np.array([w_loc.x - w_last_loc.x, w_loc.y - w_last_loc.y, 0.0])
            w_last_vec = np.array([w_last_loc.x,w_last_loc.y,0.0])
            w_target = w_last_vec + w_diff_vec * (1-(distance_to_ego-target_distance)/np.linalg.norm(w_diff_vec))
            target_location = carla.Location(x=w_target[0], y=w_target[1], z=0.0)
            break
        w_last_loc = w_loc

    if target_location is None:
        target_location = w_loc

    return target_location


class Decision(object):
    """
    AV Decision
    """
    def __init__(self):

        self.target_speed = 30.0/3.6  # m/s
        self.dt = 0.05
        self.decision_dt = 0.75

    def _IDM_desired_speed(self,EnvironmentInfo,front_vehicle):

        if front_vehicle is not None:
            dis_to_front = distance_between_two_loc(EnvironmentInfo.ego_vehicle_location,front_vehicle.location)
            front_speed = front_vehicle.speed
            # print(dis_to_front,front_speed*3.6)
        ego_v = EnvironmentInfo.ego_vehicle_speed
        speed_limit = EnvironmentInfo.speed_limit

        g0 = 7
        T = 1.2
        v = ego_v
        if v<5:
            a = 2.73 + (5-v)/5*2
        else:
            a = 2.73
        v0 = speed_limit/3.6 # m/s
        b = 1.65
        delta = 4

        if front_vehicle is None:
            dv = 0
            g = 50
            g1 = 0
        else:
            v_f = front_speed
            dv = v-v_f
            g = dis_to_front
            g1 = g0+T*v+v*dv/(2*np.sqrt(a*b))

        acc = a*(1-pow(v/v0,delta)-(g1/g)*((g1/g)))

        return v+acc*self.decision_dt


    def _Find_front_vehicle(self,local_path,EnvironmentInfo,gap_between_two_points):

        # find front vehicle on the planned path
        nearest_distance = 50
        front_vehicle = None
        for target_vehicle in EnvironmentInfo.surrounding_vehicle_list:
            d = distance_between_two_loc(target_vehicle.location, EnvironmentInfo.ego_vehicle_location)
            if location_on_the_path(local_path,target_vehicle.location,gap_between_two_points+2.6):
                if d < nearest_distance:
                    nearest_distance = d
                    front_vehicle = target_vehicle

        return front_vehicle


    def _pred_location_after_t(self,target_vehicle,time):
        """
        return a surrounding vehicle location after a given time.
        assumption: surrounding vehicle follows its current speed and direction.
        """

        t_loc = target_vehicle.location
        t_direction = target_vehicle.speed_direction
        pred_loc = carla.Location(x=t_loc.x + t_direction[0]*time*target_vehicle.speed, y=t_loc.y + t_direction[1]*time*target_vehicle.speed, z=0.0)
        # print(t_loc,t_direction,pred_loc)

        return pred_loc


    def _ego_vehicle_location_after_t_using_v(self,local_path,EnvironmentInfo,time,speed):
        """
        return the ego vehicle location following a given speed after a given time
        """

        # vehicle_transform = EnvironmentInfo.ego_vehicle_transform
        w_last_loc = EnvironmentInfo.ego_vehicle_location
        target_distance = time * speed
        target_location = None
        distance_to_ego = 0
        for waypoint in local_path:
            w_loc = waypoint.transform.location
            distance_to_ego += distance_between_two_loc(w_loc,w_last_loc)
            if distance_to_ego > target_distance:
                w_diff_vec = np.array([w_loc.x - w_last_loc.x, w_loc.y - w_last_loc.y, 0.0])
                w_last_vec = np.array([w_last_loc.x,w_last_loc.y,0.0])
                w_target = w_last_vec + w_diff_vec * (1-(distance_to_ego-target_distance)/np.linalg.norm(w_diff_vec))
                target_location = carla.Location(x=w_target[0], y=w_target[1], z=0.0)
                break
            w_last_loc = w_loc

        if target_location is None:
            target_location = w_loc

        return target_location


    def _reachable_set_analysis(self,local_path,EnvironmentInfo,IDM_speed,gap_between_two_points):
        """
        analysis avaliable velocity set.
        prediction assumption:
        ego_vehicle follows the constant velocity following local_path
        surrounding vehicle follows the constant speed along the direction
        """
        # return IDM_speed
        target_speed = IDM_speed
        safe_range = 6

        ### for pedestrian
        for pred_t in np.arange(0.0, 4.0, 0.5):
            for target_pedestrian in EnvironmentInfo.surrounding_pedestrian_list:
                if target_pedestrian.speed < 1/3.6:
                    continue
                pred_surrounding_loc = self._pred_location_after_t(target_pedestrian,pred_t)
                if not location_on_the_path(local_path,pred_surrounding_loc,gap_between_two_points+3):
                    continue
                else:
                    return 0

        #### for vehicle
        while target_speed > 0:
            safe_speed = True
            for pred_t in np.arange(0.0, 3.0, 0.5):
                pred_ego_loc = self._ego_vehicle_location_after_t_using_v(local_path,EnvironmentInfo,pred_t,target_speed)
                for target_vehicle in EnvironmentInfo.surrounding_vehicle_list:
                    if target_vehicle.speed < 5/3.6 and not location_on_the_path(local_path,target_vehicle.location,gap_between_two_points+3):
                        continue
                    # if not EnvironmentInfo.in_intersection and not EnvironmentInfo._vehicle_is_front(target_vehicle) and EnvironmentInfo.ego_vehicle_speed > 10/3.6:
                    #     continue
                    pred_surrounding_loc = self._pred_location_after_t(target_vehicle,pred_t)
                    if not location_on_the_path(local_path,pred_surrounding_loc,gap_between_two_points+3):
                        continue
                    if distance_between_two_loc(pred_ego_loc,pred_surrounding_loc) < safe_range:
                        safe_speed = False
                        break
                if not safe_speed:
                    break

            if safe_speed:
                return target_speed
            target_speed = target_speed-1

        return 0


    def _response_traffic_light(self,EnvironmentInfo,desired_speed):

        if EnvironmentInfo.front_traffic_light:
            return desired_speed

        if EnvironmentInfo.in_intersection:
            return desired_speed

        distance_to_stop = EnvironmentInfo.distance_to_traffic_light
        if distance_to_stop is None:
            return desired_speed
        else:
            if distance_to_stop < 10 + EnvironmentInfo.ego_vehicle_speed*EnvironmentInfo.ego_vehicle_speed/2/2:
                return 0

            return desired_speed

        return desired_speed

    def generate_decision(self,local_path,EnvironmentInfo,CognitionState):

        # front_vehicle = self._Find_front_vehicle(local_path,EnvironmentInfo)
        # IDM_speed = self._IDM_desired_speed(EnvironmentInfo,front_vehicle)
        # avoidance_speed = self._reachable_set_analysis(local_path,EnvironmentInfo,IDM_speed)
        # traffic_light_response_speed = self._response_traffic_light(local_path,EnvironmentInfo,avoidance_speed)

        local_path_after_decision, gap_between_two_points = self.generate_lateral_decision(local_path,EnvironmentInfo,CognitionState)

        target_speed = self.generate_longitudinal_decision(local_path_after_decision,EnvironmentInfo,CognitionState,gap_between_two_points)

        target_waypoint_location = self.generate_control_target_point(local_path_after_decision,EnvironmentInfo,CognitionState)

        return target_speed,target_waypoint_location

    def generate_longitudinal_decision(self,local_path,EnvironmentInfo,CognitionState,gap_between_two_points):

        front_vehicle = self._Find_front_vehicle(local_path,EnvironmentInfo,gap_between_two_points)
        IDM_speed = self._IDM_desired_speed(EnvironmentInfo,front_vehicle)
        avoidance_speed = self._reachable_set_analysis(local_path,EnvironmentInfo,IDM_speed,gap_between_two_points)
        traffic_light_response_speed = self._response_traffic_light(EnvironmentInfo,avoidance_speed)
        target_speed = traffic_light_response_speed

        return target_speed


    def generate_lateral_decision(self,local_path,EnvironmentInfo,CognitionState):

        if CognitionState.follow_path:
            local_path_after_decision = decouple_local_path(local_path)
            step = 1
        else:
            local_path_after_decision = []
            target_lane = self.target_lane_to_change(local_path,EnvironmentInfo,CognitionState)

            if target_lane is not None:
                local_path_after_decision = target_lane.central_point_list
                step = EnvironmentInfo.lane_step

            if target_lane is None or len(local_path_after_decision) < 4:
                local_path_after_decision = decouple_local_path(local_path)
                step = 1

        return local_path_after_decision,step


    def target_lane_to_change(self,local_path,EnvironmentInfo,CognitionState):
        target_lane_id = CognitionState.target_lane_id
        ego_y = CognitionState.ego_y

        ######
        # Lane Change Part
        ######
        ego_lane = CognitionState.get_lane_of_id(ego_y)

        if ego_lane is None:
            return None

        ## ego vehicle is driving on the target_lane
        if target_lane_id == ego_y:
            return ego_lane

        ## Current lane is avaliable for driving, ego vehicle can still drive on its lane
        if ego_lane.length_before_interaction > 150:
            return ego_lane

        ## LaneKeeping if ego_vehicle is far from exit
        if not target_lane_id > 0:
            return ego_lane

        if target_lane_id > ego_y and target_lane_id > 0:
            left_lane = CognitionState.get_lane_of_id(ego_y+1)
            if self.safe_to_change_lane(left_lane,EnvironmentInfo,CognitionState):
                return left_lane
            else:
                return ego_lane

        if target_lane_id < ego_y and target_lane_id > 0:
            right_lane = CognitionState.get_lane_of_id(ego_y-1)
            if self.safe_to_change_lane(right_lane,EnvironmentInfo,CognitionState):
                return right_lane
            else:
                return ego_lane

        return ego_lane

    def safe_to_change_lane(self,lane,EnvironmentInfo,CognitionState):

        return True

    def generate_control_target_point(self,local_path_after_decision,EnvironmentInfo,CognitionState):

        if EnvironmentInfo.ego_vehicle_speed > 10:
            control_target_dt = 0.25
        else:
            control_target_dt = 0.5
        control_target_distance = control_target_dt * EnvironmentInfo.ego_vehicle_speed  ## m
        if control_target_distance < 3:
            control_target_distance = 3

        target_waypoint_location = waypoint_on_local_path_after_d(local_path_after_decision,control_target_distance)
        return target_waypoint_location
