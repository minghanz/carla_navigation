### Deal_with_EnvironmentInfo

from collections import deque
import math

import numpy as np

import carla
from agents.tools.misc import get_speed
from agents.tools.clock import WorldClock

def location_on_reference_path(reference_path,location,sensitive_range):

    if len(reference_path) < 4:
        return False

    v_loc = location
    d_to_waypoints = []
    for (waypoint,_) in reference_path:
        w_loc = waypoint.transform.location
        d = distance_between_two_loc(v_loc,w_loc)
        d_to_waypoints.append(d)

    d_to_waypoints.sort()
    if d_to_waypoints[0]+d_to_waypoints[1] < sensitive_range:
        return True

    return False

def distance_between_two_loc(loc1,loc2):
    d = (loc1.x-loc2.x)*(loc1.x-loc2.x)+(loc1.y-loc2.y)*(loc1.y-loc2.y)
    d = np.sqrt(d)
    return d

def compute_magnitude_angle(target_location, current_location, orientation, traffic_orientation):
    
    target_vector = np.array([target_location.x - current_location.x, target_location.y - current_location.y])
    norm_target = np.linalg.norm(target_vector)

    forward_vector = np.array([math.cos(math.radians(orientation)), math.sin(math.radians(orientation))])
    traffic_vector = np.array([math.cos(math.radians(traffic_orientation)), math.sin(math.radians(traffic_orientation))])
    d_angle = math.degrees(math.acos(np.dot(forward_vector, traffic_vector)))

    return (norm_target, d_angle)

class LaneState(object):
    def __init__(self):
        self.id = None
        self.length_before_interaction = None
        self.central_point_list = None
        self.front_vehicle = None
        self.rear_vehicle = None
        # self.start_waypoint = None
        # self.end_waypoint = None

class Surrounding_pedestrian(object):
    def __init__(self):
        self.location = None
        self.speed = None
        self.speed_direction = None

class Surrounding_vehicle(object):
    def __init__(self):
        self.location = None
        self.speed = None
        self.speed_direction = None

class EnvironmentState(object):
    """
    AV driving state
    """
    def __init__(self,vehicle):


        ####cannot be used
        self.ego_vehicle = vehicle   
        self.world = self.ego_vehicle.get_world()
        self.map = self.world.get_map()

        ####
        self.ego_vehicle_transform = None
        self.ego_vehicle_location = None
        self.ego_vehicle_speed = None
        self.dt = 0.05
        self._clock = WorldClock(self.world)
        
        self.sensor_range = 30
        self.in_intersection = False
        
        self.speed_limit = 30
        self.surrounding_vehicle_list = None
        self.surrounding_pedestrian_list = None
        self.front_traffic_light = True
        self.distance_to_traffic_light = None

        self.lane_step = 5

        self.enable_perception = False

    
    def perception(self):
        """
        Interface to main loop
        """
        self.Ego_Perception()
        self.Surrounding_Perception()
        self.get_traffic_lights()

    """
    Ego and surrounding Vehicles
    """
    def Ego_Perception(self):   
        """
        Currently, getting from the server directly
        """
        self.ego_vehicle_location = self.ego_vehicle.get_location()
        self.ego_vehicle_speed = get_speed(self.ego_vehicle)/3.6   ###### m/s
        self.ego_vehicle_transform = self.ego_vehicle.get_transform()
        self.dt = 0.05
        self.speed_limit = self.ego_vehicle.get_speed_limit()

        ego_vehicle_waypoint = self.map.get_waypoint(self.ego_vehicle_location)
        if ego_vehicle_waypoint.is_junction:
            self.in_intersection = True
        else:
            self.in_intersection = False

        self.dt = self._clock.dt()

    def Surrounding_Perception(self):   

        self.surrounding_vehicle_list = []
        self.surrounding_pedestrian_list = []
        actor_list = self.world.get_actors()
        vehicle_list = actor_list.filter("*vehicle*")
        pedestrian_list = actor_list.filter("*pedestrian*")

        for target_pedestrian in pedestrian_list:
            ped_loc = target_pedestrian.get_location()
            d = distance_between_two_loc(ped_loc,self.ego_vehicle_location)

            if d > self.sensor_range:
                continue
            pedestrian_transform = target_pedestrian.get_transform()
            add_pedestrian = Surrounding_pedestrian()
            add_pedestrian.location = ped_loc
            add_pedestrian.speed = get_speed(target_pedestrian)/3.6
            
            p_begin = pedestrian_transform.location
            p_end = p_begin + carla.Location(x=math.cos(math.radians(pedestrian_transform.rotation.yaw)),
                                                y=math.sin(math.radians(pedestrian_transform.rotation.yaw)))

            p_vec = np.array([p_end.x - p_begin.x, p_end.y - p_begin.y, 0.0])
            p_vec = p_vec/np.linalg.norm(p_vec)
            add_pedestrian.speed_direction = p_vec

            self.surrounding_pedestrian_list.append(add_pedestrian)

        for target_vehicle in vehicle_list:
            # do not account for the ego vehicle
            if target_vehicle.id == self.ego_vehicle.id:
                continue

            t_loc = target_vehicle.get_location()
            d = distance_between_two_loc(t_loc,self.ego_vehicle_location)
            if d > self.sensor_range:
                continue
            vehicle_transform = target_vehicle.get_transform()
            add_vehicle = Surrounding_vehicle()
            add_vehicle.location = t_loc
            add_vehicle.speed = get_speed(target_vehicle)/3.6
            v_begin = vehicle_transform.location
            v_end = v_begin + carla.Location(x=math.cos(math.radians(vehicle_transform.rotation.yaw)),
                                                y=math.sin(math.radians(vehicle_transform.rotation.yaw)))

            v_vec = np.array([v_end.x - v_begin.x, v_end.y - v_begin.y, 0.0])
            v_vec = v_vec/np.linalg.norm(v_vec)
            add_vehicle.speed_direction = v_vec

            self.surrounding_vehicle_list.append(add_vehicle)

    def _vehicle_is_front(self,surrounding_vehicle):
        target_location = surrounding_vehicle.location
        orientation = self.ego_vehicle_transform.rotation.yaw
        current_location = self.ego_vehicle_location

        target_vector = np.array([target_location.x - current_location.x, target_location.y - current_location.y])
        norm_target = np.linalg.norm(target_vector)

        forward_vector = np.array([math.cos(math.radians(orientation)), math.sin(math.radians(orientation))])

        costheta = np.dot(forward_vector, target_vector/norm_target) 

        if abs(costheta) > 1:
            print("---------------",costheta,target_location,current_location,orientation)
            return True

        d_angle = math.degrees(math.acos(costheta))

        if d_angle<90:
            return True
        else:
            return False
    
    
    """
    Traffic Lights
    """

    def _distance_before_intersection(self):

        current_waypoint = self.map.get_waypoint(self.ego_vehicle_location)
        d = 2
        response_range = 30
        target_waypoint_list = current_waypoint.next(d)
        target_waypoint = target_waypoint_list[0]
        
        while not target_waypoint.is_junction and d < response_range:
            d = d+1
            target_waypoint_list = current_waypoint.next(d)
            target_waypoint = target_waypoint_list[0]
        if d < response_range:
            return d
        else:
            return None
        
    def _traffic_light_is_front(self,traffic_light):

        target_location = traffic_light.get_location()
        orientation = self.ego_vehicle_transform.rotation.yaw
        current_location = self.ego_vehicle_location

        target_vector = np.array([target_location.x - current_location.x, target_location.y - current_location.y])
        norm_target = np.linalg.norm(target_vector)

        forward_vector = np.array([math.cos(math.radians(orientation)), math.sin(math.radians(orientation))])
        d_angle = math.degrees(math.acos(np.dot(forward_vector, target_vector) / norm_target))

        if d_angle<90:
            return True
        else:
            return False

    def get_traffic_lights(self):     
        
        self.front_traffic_light = True
        self.distance_to_traffic_light = None

        actor_list = self.world.get_actors()
        lights_list = actor_list.filter("*traffic_light*")

        min_angle = 180.0
        sel_magnitude = 0.0
        sel_traffic_light = None
        for traffic_light in lights_list:
            if not self._traffic_light_is_front(traffic_light):
                continue

            loc = traffic_light.get_location()
            magnitude, angle = compute_magnitude_angle(loc,
                                                        self.ego_vehicle_location,
                                                        self.ego_vehicle_transform.rotation.yaw,
                                                        traffic_light.get_transform().rotation.yaw-90)
            if magnitude < 60.0 and angle < min(25.0, min_angle):
                sel_magnitude = magnitude
                sel_traffic_light = traffic_light
                min_angle = angle
        
        if sel_traffic_light is not None:
            print('=== Magnitude = {} | Angle = {} | ID = {}'.format(
                sel_magnitude, min_angle, sel_traffic_light.id))
            print(sel_traffic_light.state)

            if sel_traffic_light.state == carla.TrafficLightState.Red:
                self.front_traffic_light = False

            print("distance_before_interaction:",self._distance_before_intersection())
            self.distance_to_traffic_light = self._distance_before_intersection()


    """
    For Cognition
    """

    def longitudinal_position_after_distance(self,target_vehicle,distance):

        current_waypoint = self.map.get_waypoint(target_vehicle.location)
        if current_waypoint is None:
            return None
        
        target_waypoint_list = current_waypoint.next(distance)
        if len(target_waypoint_list) < 1:
            return None

        location_list = []
        for target_waypoint in target_waypoint_list:
            if target_waypoint is not None:
                location_list.append(target_waypoint.transform.location)

        return location_list

    def is_multilane(self,target_location):



        current_waypoint = self.map.get_waypoint(target_location)

        if self._have_left_lane(current_waypoint) or self._have_right_lane(current_waypoint):
            return True
        return False

    
    def get_lane(self,lane_id,reference_path):

        current_waypoint = self.map.get_waypoint(self.ego_vehicle_location)
        if lane_id == 0:
            return self._generate_lane(current_waypoint,reference_path)

        if lane_id < 0:
            for i in range(lane_id,0):
                if self._have_right_lane(current_waypoint):
                    current_waypoint = current_waypoint.get_right_lane()
                else:
                    return None
            return self._generate_lane(current_waypoint,reference_path)

        if lane_id > 0:
            for i in range(0,lane_id):
                if self._have_left_lane(current_waypoint):
                    current_waypoint = current_waypoint.get_left_lane()
                else:
                    return None
            return self._generate_lane(current_waypoint,reference_path)
            
    def _have_left_lane(self,waypoint):
        lane_change = waypoint.lane_change
        left_lane_marking = waypoint.left_lane_marking
        if lane_change is None or lane_change == carla.LaneChange.Right:
            return False
        
        if left_lane_marking.type is not carla.LaneMarkingType.Broken:
            return False

        if left_lane_marking.color is not carla.LaneMarkingColor.White:
            return False

        return True

    def _have_right_lane(self,waypoint):
        lane_change = waypoint.lane_change
        right_lane_marking = waypoint.right_lane_marking
        if lane_change is None or lane_change == carla.LaneChange.Left:
            return False
        
        if right_lane_marking.type is not carla.LaneMarkingType.Broken:
            return False

        if right_lane_marking.color is not carla.LaneMarkingColor.White:
            return False

        return True

    def _is_parallel(self,begin1,end1,begin2,end2):
        # vehicle_transform = ego_vehicle_transform
        v_begin = begin1.transform.location
        v_end = end1.transform.location

        v_vec = np.array([v_end.x - v_begin.x, v_end.y - v_begin.y, 0.0])

        w_begin = begin2.transform.location
        w_end = end2.transform.location

        w_vec = np.array([w_end.x - w_begin.x, w_end.y - w_begin.y, 0.0])

        _dot = math.acos(np.clip(np.dot(w_vec, v_vec) /
                            (np.linalg.norm(w_vec) * np.linalg.norm(v_vec)), -1.0, 1.0))

        _cross = np.cross(v_vec, w_vec)
        if _cross[2] < 0:
            _dot *= -1.0
        
        if abs(_dot) < np.pi/18:
            return True
        else:
            return False


    def _generate_lane(self,start_waypoint,reference_path):

        if start_waypoint is None:
            return None

        step = self.lane_step

        if len(reference_path) < step+5:
            return None

        lane = LaneState()
        central_point_list = []
        
        last_waypoint = start_waypoint

        print(reference_path[0])
        last_reference_waypoint = reference_path[0][0]
        search_id = step
        target_reference_waypoint = reference_path[search_id][0]
        length_before_interaction = step
        target_waypoint_list = start_waypoint.next(length_before_interaction)
        if len(target_waypoint_list) > 0:
            target_waypoint = target_waypoint_list[0]
        else:
            # lane.length_before_interaction = 1
            # lane.central_point_list = []
            # return lane
            return None

        central_point_list.append(target_waypoint)

        is_parallel = self._is_parallel(last_waypoint,target_waypoint,last_reference_waypoint,target_reference_waypoint)
        if distance_between_two_loc(target_reference_waypoint.transform.location,last_reference_waypoint.transform.location) > step + 2:
            is_parallel = True

        while length_before_interaction < 180 and is_parallel:
            search_id += step
            length_before_interaction += step
            last_waypoint = target_waypoint
            last_reference_waypoint = target_reference_waypoint
            if search_id > len(reference_path)-5:
                break
            target_waypoint = None
            target_reference_waypoint = reference_path[search_id][0]
            target_waypoint_list = last_waypoint.next(step)
            if target_waypoint_list is None:
                break

            if distance_between_two_loc(target_reference_waypoint.transform.location,last_reference_waypoint.transform.location) > step + 2:
                target_waypoint = target_waypoint_list[0]
                central_point_list.append(target_waypoint)
                is_parallel = True
                continue

            for next_waypoint in target_waypoint_list:
                if self._is_parallel(last_waypoint,next_waypoint,last_reference_waypoint,target_reference_waypoint) or location_on_reference_path(reference_path,next_waypoint.transform.location,6):
                    target_waypoint = next_waypoint
        
            if target_waypoint is None:
                break
            central_point_list.append(target_waypoint)
            is_parallel = self._is_parallel(last_waypoint,target_waypoint,last_reference_waypoint,target_reference_waypoint)

        lane.length_before_interaction = length_before_interaction
        lane.central_point_list = central_point_list

        return lane



