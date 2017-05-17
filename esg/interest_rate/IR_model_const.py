## Python packages
import numpy as np
from .IR_model_util import IR_model_util

# ==============================================
#          Constant short rate model
# ==============================================
class IR_model_const(IR_model_util):

    def __init__(self):
        self.time_horizon = 0 # Time horizon for trajectory.
        #self.number_trajectories = 1 # By defaut, the number of trajectories is equal to one.

    def calibrate(self, asset_data):
        """Load base IR curve from the initial market_environment"""
        spot_rate = asset_data.get_list_market('EUR').spot_rate
        deflator = np.power(1./np.add(1,spot_rate), np.arange(1, len(spot_rate)+1))
        self.curves = np.zeros(shape = (self.time_horizon+1, len(spot_rate)))
        self.deflators = np.zeros(shape = (self.time_horizon+1, len(spot_rate)))
        for time_step in range(self.time_horizon+1):
            self.curves[time_step,:] = spot_rate
            self.deflators[time_step,:] = deflator

    def get_IR_curve(self):
        self.trajectory = np.ones(self.time_horizon+1)
        return self.curves

    def get_deflator(self):
        return self.deflators
