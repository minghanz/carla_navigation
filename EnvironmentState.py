### Deal_with_EnvironmentInfo

from collections import deque
import math

import numpy as np

import carla
from agents.tools.misc import get_speed
from agents.tools.clock import WorldClock



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
        self.length_before_interaction = None
        self.central_point_list = None

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
        d_angle = math.degrees(math.acos(np.dot(forward_vector, target_vector) / norm_target))

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

    def longitudinal_position_after_distance(self,distance):

        current_waypoint = self.map.get_waypoint(self.ego_vehicle_location)
        target_waypoint_list = current_waypoint.next(distance)
        target_waypoint = target_waypoint_list[0]
        return target_waypoint.transform.location

    def is_multilane(self,target_location):

        current_waypoint = self.map.get_waypoint(target_location)
        lane_change = current_waypoint.lane_change
        left_lane_marking = current_waypoint.left_lane_marking
        right_lane_marking = current_waypoint.right_lane_marking

        if lane_change is None:
            return False
        if left_lane_marking.type is not carla.LaneMarkingType.Broken and right_lane_marking.type is not carla.LaneMarkingType.Broken:
            return False

        return True

    
    def get_lane(self,lane_id):

        current_waypoint = self.map.get_waypoint(self.ego_vehicle_location)
        lane = LaneState()
        central_point_list = []
        length_before_interaction = 1

        if lane_id == 0:
            target_waypoint_list = current_waypoint.next(length_before_interaction)
            target_waypoint = target_waypoint_list[0]
            central_point_list.append(target_waypoint)
            target_waypoint_is_multilane = self.is_multilane(target_waypoint.transform.location)
            while target_waypoint_is_multilane and length_before_interaction < 100:
                length_before_interaction += 1
                target_waypoint_list = current_waypoint.next(length_before_interaction)
                target_waypoint = target_waypoint_list[0]
                central_point_list.append(target_waypoint)
                target_waypoint_is_multilane = self.is_multilane(target_waypoint.transform.location)

            lane.length_before_interaction = length_before_interaction
            lane.central_point_list = central_point_list
            return lane


        if lane_id < 0:
            left_waypoint = current_waypoint.get_left_lane()
            for i in range(lane_id,-1):
                left_waypoint = left_waypoint.get_left_lane()
            

        # left_waypoint = current_waypoint.get_left_lane()
        # left_waypoint = left_waypoint.get_left_lane()
        # left_waypoint_lane_change = left_waypoint.lane_change
        # print(left_waypoint_lane_change)
        # right_waypoint = current_waypoint.get_right_lane()
        # right_waypoint_location = right_waypoint.transform.location
        # right_lane_marking = current_waypoint.right_lane_marking.lane_change
        
        # print(right_lane_marking)

        # begin = self.ego_vehicle_location
        # end = right_waypoint_location
        # self.world.debug.draw_arrow(begin, end, arrow_size=0.3, life_time=-0.1)

        

