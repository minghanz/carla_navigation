### get cognition and state space


import math

import numpy as np

import carla
from agents.tools.misc import get_speed
from agents.navigation.EnvironmentState import EnvironmentState,LaneState


class CognitionState(object):
    """
    AV driving cognition state
    """
    def __init__(self):
        self.follow_path = True

    
    def scenario_cognition(self,EnvironmentInfo):
        self.follow_path = True
        is_multilane = EnvironmentInfo.is_multilane(EnvironmentInfo.ego_vehicle_location)
        if is_multilane:
            central_lane = EnvironmentInfo.get_lane(-3)
            # print(central_lane.length_before_interaction)



    
