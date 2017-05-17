## Program packages
from .IR_model_classes import IR_model
## Python packages
from math import exp
import numpy as np


class IR_model_util(IR_model):
    """
    Document the role of this class TO DO
    """
    def __init__(self):
        pass

    @staticmethod
    def swap_rate_and_annuity(maturity, tenor, valuation_date, zcb_price):
        return (zcb_price[valuation_date, maturity - 1] - zcb_price[valuation_date, tenor + maturity -1])/(sum(zcb_price[valuation_date, t] for t in range(maturity, maturity+tenor))), sum(zcb_price[valuation_date, t] for t in range(maturity, maturity+tenor))
    
    @staticmethod
    def DF(trajectory, begin, end):
        v = [trajectory[t] for t in range(begin, end)]
        return exp(-sum(v))
        
    @staticmethod
    def discount_factor_custom(trajectory, begin, end):
        v = [(trajectory[t] + trajectory[t+1])/2 for t in range(begin, end)]
        return exp(-sum( v))

    def discount_factors(self, trajectory):
        return [self.discount_factor_custom(trajectory, 0, t) for t in range(1, len(trajectory))]

    def test_martingale(self):
        """
        This method should be put somewhere else TO DO
        """
        for time_step in range(1, self.time_horizon):
            discount_factor = []
            epsilon = []
            i = 1
            while i <= self.N:
                self.get_IR_curve()
                discount_factor.append(self.discount_factor(begin = 0, end = time_step))
                i += 1
            epsilon.append(np.absolute(np.mean(discount_factor) - self.zcb_price[0,time_step]))
        return np.amax(epsilon)

    def get_IR_curve(self):
        """This method returns IR term structures per time steps and trajectories"""
        pass

    def get_deflator(self):
        """This method returns deflators per time steps and trajectories"""
        pass

    def calibrate(self, asset_data):
        """Calibrate the IR model onto the market environment
            This method returns the IR_curve and deflator"""
        pass
