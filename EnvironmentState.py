### Deal_with_EnvironmentInfo

from collections import deque
import math

import numpy as np

import carla
from agents.tools.misc import get_speed



def distance_between_two_loc(loc1,loc2):
    d = (loc1.x-loc2.x)*(loc1.x-loc2.x)+(loc1.y-loc2.y)*(loc1.y-loc2.y)
    d = np.sqrt(d)
    return d



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
        
        self.sensor_range = 50
        
        self.speed_limit = 30
        self.surrounding_vehicle_list = None

    def Ego_Perception(self):   
        """
        Currently, getting from the server directly
        """
        self.ego_vehicle_location = self.ego_vehicle.get_location()
        self.ego_vehicle_speed = get_speed(self.ego_vehicle)/3.6   ###### m/s
        self.ego_vehicle_transform = self.ego_vehicle.get_transform()
        self.dt = 0.05
        self.speed_limit = self.ego_vehicle.get_speed_limit()

    def Surrounding_Perception(self):   

        self.surrounding_vehicle_list = []
        actor_list = self.world.get_actors()
        vehicle_list = actor_list.filter("*vehicle*")

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

    
    def scenerio_analysis(self):

        self.follow_path = True
        current_waypoint = self.map.get_waypoint(self.ego_vehicle_location)
        if current_waypoint.lane_change is not None:
            self.follow_path = False
        
        
        # current_waypoint = self.map.get_waypoint(self.ego_vehicle_location)
        # left_lane_waypoint = current_way_point.get_left_lane()

    def longitudinal_position_after_distance(self,distance):

        current_waypoint = self.map.get_waypoint(self.ego_vehicle_location)
        target_waypoint_list = current_waypoint.next(distance)
        target_waypoint = target_waypoint_list[0]
        for i in target_waypoint_list:
            print(i.transform.location)
        return target_waypoint.transform.location
        
                


