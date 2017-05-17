## Program Packages
from ..generator_correlated_variables import generator_correlated_variables
## Python Packages
from .IR_model_util import IR_model_util
from .IR_model_functions import diff
from math import exp
import numpy as np
import pickle
# ========================================================================================================================================================================
#       One factor Hull & White short rate model
# The Hull-White Short Rate Model is defined as:
# dr(t) = (theta(t) - a * r(t)) * dt  + sigma * dW(t), where a and sigma are constants, and theta(t) is chosen in order to fit the input term structure of interest rates.
# ========================================================================================================================================================================

class Hull_White_one_factor(IR_model_util):
    """
        Definition:
        ===========
        The Hull-White Short Rate Model is defined as:
        dr(t) = (theta(t) - a * r(t)) * dt  + sigma * dW(t), where a and sigma are constants, and theta(t) is chosen in order to fit the input term structure of interest rates.

        Objective:
        ==========
        This class is meant to generate simulated paths based on the Hull and White interest rate model.

        Attributes:
        ===========
        Input

        1. time_horizon:
            Type: int
            Function: time horizon of the simulation for the overall system analysis
        2. fixed_seed:
            Type: int
            Function: to define the random state
        3. num_instrument
            Type: int
            Function: number of instruments
        4. corr_matrix:
            Type: 2 dimensional array
            Function: correlated geometric Brownnian motions matrix
        5. sigma:
            Type: float (positive only)
            Function: constant volatility parameter
        6. a:
            Type: float (positive only)
            Function: mean reversion factor
        7. r0:
            Type: float
            Function: initial interest rate value.
    """

    def __init__(self, a = None, sigma = None, market_name = None, r0 = None, alpha = None, beta = None, gamma = None):
        self.time_horizon = 0
        self.fixed_seed = None
        self.a = a
        self.sigma = sigma
        self.market_name = market_name
        if r0 is not None:
            self.r0 = r0
        else:
            self.r0 = 0
        self.corr_matrix = None
        # ============================================================================
        # Afin de faire des calculs avec la méthode de MC, on prend N = 100.000
        # ============================================================================
        #self.N = 50
        # ==================================================
        # Shock lebels
        # ==================================================
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
    def calibrate(self, asset_data):
        """
            Method: calibrate

            Function: Calibrate the IR model onto the market environment. This method returns the calibrated model parameters

            Parameter:
                1. asset_data:
                    Type: instance of Asset_data class
                    Function: see class Asset_data for more details.
        """
        market = asset_data.get_list_market(self.market_name)
        if self.a is None:
            self.a = market.speed_IR_rate
            
        if self.sigma is None:
            self.sigma = market.vol_IR_rate
            
        self.spot_rate = market.spot_rates
        # =====================================
        if self.alpha is not None:
            with open(r'data\pickle\adjusted_alpha_zcb_curves.pkl', 'rb') as input:
                zcb_curves = pickle.load(input)
                self.spot_rate = zcb_curves[self.alpha] 

        if self.beta is not None:
            with open(r'data\pickle\adjusted_beta_zcb_curves.pkl', 'rb') as input:
                zcb_curves = pickle.load(input)
                self.spot_rate = zcb_curves[self.beta]

        if self.gamma is not None:
            with open(r'data\pickle\adjusted_gamma_zcb_curves.pkl', 'rb') as input:
                zcb_curves = pickle.load(input)
                self.spot_rate = zcb_curves[self.gamma]

        # =====================================
        self.forward_rate = np.zeros(len(self.spot_rate))
        # ===============================================
        # By defaut, self.forward_rate[0] = 0
        # ===============================================
        for t in range(1, len(self.forward_rate)):
            if t == 1:
                self.forward_rate[t] = np.log(1+self.spot_rate[t-1])
            else:
                self.forward_rate[t] = np.log((1+self.spot_rate[t-1])**t/(1+self.spot_rate[t-2])**(t-1))
        derivf = diff(self.forward_rate, range(len(self.forward_rate)))
        self.theta = derivf + np.dot(self.a, self.forward_rate[:-1]) + [self.sigma**2/(2*self.a)*(1 - exp(-2*self.a*t)) 
                                                                        for t in range(len(self.forward_rate)-1)]

    def get_IR_curve(self):
        """
            Method: get_IR_curve

            Function: return Interest Rate term structures, zero-bond curve per time step and its trajectories over time horizon

            Parameters: None
        """
        dw = generator_correlated_variables(corr_matrix = self.corr_matrix, time_horizon = self.time_horizon, fixed_seed = self.fixed_seed)
        # ===================================
        # Generate IR trajectory
        # ===================================
        self.trajectory = [self.forward_rate[0]]
        for time_step in range(1, self.time_horizon+1):
            dr = (self.theta[time_step-1] - self.a * self.trajectory[-1]) + self.sigma * dw[0, time_step-1]
            self.trajectory.append(self.trajectory[-1] + dr)
        # ====================================
        # Generate IR zero coupon curve
        # ====================================
        self.zcb_price = np.ones(shape = (self.time_horizon+30, len(self.spot_rate)))
        self.curves = np.ones(shape = (self.time_horizon+30, len(self.spot_rate)))
        zcb0 = [self.DF(trajectory = self.forward_rate, begin = 1, end = t) for t in range(2, len(self.spot_rate)+1)]
        zcb0.append(zcb0[-1]) # Linear extrapolation zero coupon price curve
        self.zcb_price[0,:] = zcb0
        
        self.curves[0,:] = [round(np.power(zcb0[t-1], -1/t) -1,5) for t in range(1, len(self.spot_rate)+1)]
        
        forward_rate = - diff(np.log(zcb0), range(1,len(self.spot_rate)+1))

        for time_step in range(1,self.time_horizon+1):
            B = np.array([(1 - exp(-self.a * t))/self.a for t in range(1, len(self.forward_rate) + 1 - time_step)])
            k1 = B* forward_rate[time_step]*zcb0[time_step]
            k2 = self.sigma**2/(4*self.a**3)*(exp(2 * self.a * time_step)-1)
            k3 = np.array([np.power(exp(-self.a*t) - exp(-self.a*time_step) ,2) for t in range(time_step+1, len(self.forward_rate)+1)])
            BB = np.exp( k1 - k2 * k3 )
            PP = np.array(self.zcb_price[0, time_step : len(self.forward_rate)+1])/self.zcb_price[0,time_step]
            A = PP*BB
            #Buid curve
            zcb_price = A*np.exp(-B*self.trajectory[time_step])
            curve = - np.log(zcb_price) * np.array([1/t for t in range(1, len(zcb_price) + 1)])
            #Constant extrapolation of curve
            curve_full = np.concatenate( (curve, [curve[-1] for n in range(len(self.forward_rate)-len(zcb_price))]), axis=0)
            zcb_price_full = np.power(1./np.add(1, curve_full), np.arange(1,len(curve_full)+1))
            self.zcb_price[time_step,] = zcb_price_full
            self.curves[time_step,] = curve_full

        return self.curves, self.trajectory

    def get_deflator(self):
        """
            Method: get_deflator

            Function: return deflators per time step

            Parameter: None
        """
        self.deflators = np.power(1./np.add(1,self.curves), np.arange(1, len(self.forward_rate)+1))
        return self.deflators
