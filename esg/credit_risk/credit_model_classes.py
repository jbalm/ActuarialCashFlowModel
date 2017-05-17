# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 15:36:21 2016

@author: Jun JING
"""

## Progam packages
#from ...asset.Asset_data import Asset_data
from ..generator_correlated_variables import generator_correlated_variables
from ...core_math.function_optim import function_optim
from ...core_math.functions_credit import generator_matrix, exp_matrix

## Python packages
from abc import ABCMeta, abstractmethod
from scipy.linalg import inv, norm
from numpy import linalg as la
from scipy.optimize import minimize
import numpy as np

class credit_model_base:
    """
        This is purely abstract class for all credit models we will implement in the future
    """

    __metaclass__=ABCMeta

    @abstractmethod
    def add_time_horizon(self, time_horizon):
        """This method add time horizon """
        return

    @abstractmethod
    def get_spread(self):
        """This method get the market spread"""
        return

    @abstractmethod
    def get_RN_transition_matrix(self):
        """This method get the historical transition matrix"""
        return
        

    @abstractmethod
    def calibrate_spread(self):
        """Calibrate the credit model onto the market environment"""
        return


    @abstractmethod
    def generate_spreads_and_matrix(self):
        """generate the spreads"""
        return



        
        
class JLT(credit_model_base):
    """
        The JLT model is inplemented here
        
        Attributes:
        ==========
        
        Input:
        _______
        1. pi_0 : initial value of the risk premium
            Type : float
        
        2. mu : long term average parameter
            Type : float
            
        3. alpha : speed of adjustment parameter
            Type : float
        
        4. recovery_rate : recorevry rate when default
            Type : float
            
        5. sigma : volatility parameter
            Type : float
        
        6. market_name : market name
            Type : string
            
        Output:
        _______
        1. RN_migration_matrix : risk-neutral migration matrix
            Type : matrix 7x7
        
        2. spreads : credit spreads
            Type : vector of length 7
        
        Methods:
        _______
        1. add_time_horizon
        
        2. get_spread
        
        3. get_hist_transition_matrix
        
        4. calibrate_spread
        
        5. calibrate_price
        
        6. generate_spreads_and_matrix
        
        7. test_diffusion_pi
        
    """
    
    def __init__(self, pi_0= None, mu= None, alpha= None, sigma= None, recovery_rate= None, market_name= None):
        self.time_horizon=0
        self.recovery_rate = recovery_rate
        
        # initiate
        self.market_spread = None
        self.eigenval_hist_gen= None
        self.eigenvect_hist_gen= None
        
        self.historical_transition_matrix = None
        self.RN_migration_matrix=[]
        
        self.spreads=[]
        
        self.mu=mu
        self.alpha=alpha
        self.sigma=sigma

        self.corr_matrix= None
        self.fixed_seed = None
        self.num_instrument = 0
        
        self.pi_0=pi_0
        
        self.market_name=market_name
        # prend comme entrée IR_model, ou pas... On a défini également une méthode qui permet d'aller récupérer les taux zéro coupons d'un modèle IR
        # ici, on a peut-etre seulement besoin de tout initialiser à vide
        # l'intérêt de la définition est qu'il est prêt d'être utilisé, plus simple
        # or une méthode permet de modifier/ d'accéder à des attributes depuis extérieur.
        
    def getMatrixJLT(self,t,T):
        out = None
        d = self.eigenval_hist_gen
        if self.sigma !=0:
            v = np.sqrt(self.alpha**2 - 2*d*self.sigma**2)
            denominator = (v+self.alpha)*(np.exp(v*(T-t))-1)+2*v
            A = (2*self.alpha*self.mu)/(self.sigma**2)*np.log((2*v*np.exp(0.5*(self.alpha+v)*(T-t)))/denominator)
            B = - (2*d*(np.exp(v*(T-t))-1))/denominator
            value = np.exp(A - B*self.risk_premium[t])
            out = np.diag(value)
        else:
            temp = (self.risk_premium[t]+np.exp(-self.alpha*t))*(T-t) + 1/(self.alpha)*(np.exp(-self.alpha*T)-np.exp(-self.alpha*t))
            value = np.exp(d*temp)
            out = np.diag(value)
        return out
        
    def add_time_horizon(self,time_horizon):
        """
            Method : add_time_horizon
            
            Function : add the time horizon
            
            Parameter :
                1. time_horizon
                    Type : int
                    Function : correspond to the time horizon
        """
        self.time_horizon = time_horizon

        
    
    def get_spread(self,asset_data):
        """
            Method : get_spread
            
            Function : retrieve the spread from the pickle file
            
            Parameter : None
        """
        # read the market spread data ''of time 0''  
        market = asset_data.get_list_market(self.market_name)
        spread_list = market.spread_list
        col_index = market.col_index
        row_index = market.row_index
        self.market_spread = spread_list, col_index, row_index

    
    def get_hist_transition_matrix(self, asset_data):
        """
            Method : get_hist_transition_matrix
            
            Function : retrieve the historical transition matrix from the pickle file and then deduce the generator matrix, its eigenvectors and its eigenvalues.
            
            Parameter : None
        """
        
        market = asset_data.get_list_market(self.market_name)
        historical_transition_matrix = market.historical_transition_matrix
        
        self.historical_transition_matrix = historical_transition_matrix
        
        self.historical_generator_matrix = generator_matrix(self.historical_transition_matrix)
        
        w, v = la.eig(self.historical_generator_matrix)
        eigenval_hist_gen = w.real
        eigenvect_hist_gen = (v.T).real
        for l in range(len(eigenvect_hist_gen)):
            eigenvect_hist_gen[l] = eigenvect_hist_gen[l]/norm(eigenvect_hist_gen[l])
    
        eigenvect_hist_gen = eigenvect_hist_gen.T
        
        self.eigenval_hist_gen= eigenval_hist_gen      
        self.eigenvect_hist_gen= eigenvect_hist_gen
        
    def calibrate_spread(self, asset_data, AAA_AA):
        """
            Method : calibrate_spread
            
            Function : calibrate the model on the market data of spread
            
            Parameter :
                1. asset_data
                    Type : instance of Asset_data class
                    Function : see class Asset_data for more details.
                    
                2. AAA_AA
                    Type : boolean
                    Function : if it is true, then only spreads of AAA and AA ratings are used for the calibration
        """
        market = asset_data.get_list_market(self.market_name)
        if self.mu is None:
            self.mu = market.JLT_mu
        if self.sigma is None:
            self.sigma = market.JLT_sigma
        if self.alpha is None:
            self.alpha = market.JLT_alpha
        if self.pi_0 is None:
            self.pi_0 = market.JLT_pi
        if self.recovery_rate is None:
            self.recovery_rate = market.recovery_rate

        spread_list, col_index, row_index = self.market_spread
        
        def f(pi_0):
            return function_optim(pi_0, self.alpha, self.mu, self.sigma, self.recovery_rate,
                                  self.eigenvect_hist_gen, self.eigenval_hist_gen,
                                  row_index, col_index, spread_list,AAA_AA)        
        
        bds = [(0.001,None)]
        res = minimize(f,x0=2, bounds=bds )                    
        self.pi_0 = res.x[0]    
        return self.pi_0



    def calibrate_price(self, asset_data):
        """
            Method : calibrate_price
            
            Function : calibrate the model on the market data of bonds' price
            
            Parameter :
                1. asset_data
                    Type : instance of Asset_data class
                    Function : see class Asset_data for more details.
                    
        """        
        market = asset_data.get_list_market(self.market_name)
        if self.mu is None:
            self.mu = market.JLT_mu
        if self.sigma is None:
            self.sigma = market.JLT_sigma
        if self.alpha is None:
            self.alpha = market.JLT_alpha
        if self.pi_0 is None:
            self.pi_0 = market.JLT_pi
        if self.recovery_rate is None:
            self.recovery_rate = market.recovery_rate

        spread_list, col_index, row_index = self.market_spread
        
        def f(pi_0):
            return function_optim(pi_0, self.alpha, self.mu, self.sigma,
                                  self.recovery_rate, self.eigenvect_hist_gen, self.eigenval_hist_gen,
                                  row_index, col_index, spread_list)     
    
        res = minimize(f,x0=2)
        self.pi_0 = res.x[0]
        return self.pi_0
        

      
    def generate_spreads_and_matrix(self):
        """
            Method : generate_spreads_and_matrix
            
            Function : generate the spreads and risk-neutral transition matrix with parameters in the model
            
            Parameter : None
                    
        """        
        self.spreads=[]
        self.RN_migration_matrix=[]
        
        dw = generator_correlated_variables(corr_matrix = self.corr_matrix, time_horizon = self.time_horizon,  fixed_seed = self.fixed_seed)
        # ===================================
        # Generate CIR process
        # ===================================
        self.risk_premium=[self.pi_0]
        for time_step in range(1,self.time_horizon+1):
            dpi = self.alpha*(self.mu-self.risk_premium[-1]) + self.sigma*np.sqrt(self.risk_premium[-1])*dw[2,time_step-1]
            self.risk_premium.append(max(0,self.risk_premium[-1] + dpi))
            
        
        for t in range(self.time_horizon+1):
        #une boucle de bas de temps    
            RN_generator_matrix_t = np.dot(np.dot(self.eigenvect_hist_gen, np.diag(self.risk_premium[t]*self.eigenval_hist_gen)), inv(self.eigenvect_hist_gen))
            RN_migration_matrix_t = exp_matrix(RN_generator_matrix_t).astype('Float64')
            self.RN_migration_matrix.append(RN_migration_matrix_t)
        
        for t in range(self.time_horizon+1):
            spread_T = []
            for T in range(t+1,t+21):
                spread_t_T = []
                JLTmatrix = self.getMatrixJLT(t,T)
                I = np.identity(len(self.eigenval_hist_gen))
                RN_migration_matrix_t_T = I + np.dot(np.dot(self.eigenvect_hist_gen,(JLTmatrix-I)),inv(self.eigenvect_hist_gen))
                if all(1-(1-self.recovery_rate)*RN_migration_matrix_t_T.T[-1][:-1] > 0):
                    spread_t_T = -1/(T-t)* np.log(1-(1-self.recovery_rate)*RN_migration_matrix_t_T.T[-1][:-1])
                else:
                    raise ValueError('value in log not defined!')
                spread_T.append(spread_t_T)
            self.spreads.append(spread_T)
            
        # self.spreads est une liste qui possède trois dimensions : 1. bas de temps; 2. maturité; 3. notation
        return self.RN_migration_matrix, self.spreads 
        
