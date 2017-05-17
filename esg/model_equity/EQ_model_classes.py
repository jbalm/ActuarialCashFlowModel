## Python packages
from abc import ABCMeta, abstractmethod, abstractproperty

######Original abstract class
class EQ_model_base:
    """This class is the purely abstract class for all IR models we will implement in the future
    see https://pymotw.com/2/abc/"""

    __metaclass__=ABCMeta

    @abstractmethod
    def add_time_horizon(self, time_horizon):
        """This method implements the time horizon for EQ model"""
        return

    @abstractmethod
    def get_EQ_prices(self):
        """This method returns equity prices per time steps and trajectories"""
        return

    @abstractmethod
    def add_dividend_rate(self, dividend_rate):
        """This method implements the dividend rate per time steps and/or trajectories"""
        return

    @abstractmethod
    def calibrate(self, asset_data):
        """Calibrate the IR model onto the market environment"""
        return


    @abstractmethod
    def add_short_rate_trajectory(self, short_rate_trajectory):
        """This method implements the short rate trajectory"""
        return