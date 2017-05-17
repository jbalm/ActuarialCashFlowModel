# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 14:14:23 2016

@author: FR011526
"""
#Program packages
from ..core_math import excel_toolbox as et

## Python packages
import numpy as np
import collections
from xlrd import open_workbook
import xlrd
import pickle


class Smith_Wilson(object):
    def __init__(self, file_name, sheet_name = 'Smith Wilson extrapolation', exec=True):
        self.UFR_maturity = 59
        # =========================================
        # Initialize Swap_Rate_List
        self.swap_rate_lists = collections.OrderedDict()
        # =========================================

        wb = open_workbook(file_name)
        sheet = wb.sheet_by_name(sheet_name)
        
        maturities = []
        coupon_yields = []
        
        row = 15
        while sheet.cell_type(row,1) != xlrd.XL_CELL_EMPTY:
            maturities.append(int(sheet.cell( row, 1).value))
            coupon_yields.append(sheet.cell( row, 2).value/100)
            row +=1
            
        self.swap_rate_lists['Maturities'] = maturities
        self.swap_rate_lists['Coupon_yields'] = coupon_yields

        if exec:
            with open(r'data\pickle\pca_swap.pkl', 'rb') as input:
                self.a1 = pickle.load(input)
                self.a2 = pickle.load(input)
                self.a3 = pickle.load(input)
            # ===========================================================================================
            # Get First Three Principal Components of the swap rates curve
            self.PC1 = np.dot(self.swap_rate_lists['Coupon_yields'], self.a1)
            self.PC2 = np.dot(self.swap_rate_lists['Coupon_yields'], self.a2)
            self.PC3 = np.dot(self.swap_rate_lists['Coupon_yields'], self.a3)
            # ===========================================================================================

        self.num_instrument = len(self.swap_rate_lists['Maturities'])
        self.llp = int(self.swap_rate_lists['Maturities'][-1])
        self.m = np.ones(len(self.swap_rate_lists['Maturities']))
    
    def get_time_step(self, s):
        self.s = s
        
    def get_UFR(self, UFR):
        self.UFR = UFR
        self.mu = [np.exp(-self.UFR * t) for t in range(1, self.llp+1)]
        
    def get_alpha(self, alpha):
        self.alpha = alpha

    def Wvector(self, time_step):
        W = [np.exp(-self.UFR * (time_step + u))*(self.alpha * min(time_step, u) - np.exp(-self.alpha * max(time_step, u)) * np.sinh(self.alpha * min(time_step, u))) for u in range(1,self.llp+1)]
        return W

    def Wmatrix(self):
        self.W = np.zeros(shape = (self.llp, self.llp))
        for i in range(1,self.llp+1):
            for j in range(1,self.llp+1):
                self.W[i-1][j-1] = np.exp(-self.UFR*(i+j))*(self.alpha * min(i,j) - np.exp(-self.alpha * max(i,j)) * np.sinh(self.alpha * min(i,j)))
        return self.W

    def Cmatrix(self, coupon_rate_list):
        C = np.zeros(shape = (self.num_instrument, self.llp))
        for i in range(1, self.num_instrument+1):
            for j in range(1, self.llp+1):
                if j < int(self.swap_rate_lists['Maturities'][i-1]):
                    C[i-1][j-1] = coupon_rate_list[i-1]
                elif j == int(self.swap_rate_lists['Maturities'][i-1]):
                    C[i-1][j-1] = 1 + coupon_rate_list[i-1]
        return C

    def get_eta_vector(self, coupon_rate_list):
        C = self.Cmatrix(coupon_rate_list)
        W = self.Wmatrix()
        CWC = np.dot(np.dot(C,W),np.transpose(C))
        CWC_inv = np.linalg.inv(CWC)
        Cmu = np.dot(C,self.mu)
        mCmu = np.subtract(self.m, Cmu)
        eta = np.dot(CWC_inv, mCmu)
        return eta, C

    def SM_extrapolation(self, end_time, stress_test = False, output_excel = True):
        eta, C = self.get_eta_vector(coupon_rate_list = self.swap_rate_lists['Coupon_yields'])
        zcb_price = np.zeros(end_time)
        for time_step in range(end_time):
            W = self.Wvector(time_step + 1)
            zcb_price[time_step] = np.exp(-self.UFR * (time_step + 1)) + np.dot(eta, np.dot(C,W))
        self.swap_rate_lists['Zero_Bond_Prices'] = zcb_price
        self.swap_rate_lists['Spot_Rates'] = - np.log(zcb_price) * np.array([1/t for t in range(1, len(zcb_price) + 1)])
        # Build the forward rates curve from the extrapolated spot rates
        N = len(self.swap_rate_lists['Spot_Rates'])
        self.swap_rate_lists['Forward_Rates'] = np.zeros(N)
        for t in range(N):
            if t == 0:
                self.swap_rate_lists['Forward_Rates'][t] = np.log(1+self.swap_rate_lists['Spot_Rates'][t])
            else:
                self.swap_rate_lists['Forward_Rates'][t] = np.log((1+self.swap_rate_lists['Spot_Rates'][t])**(t+1)/(1+self.swap_rate_lists['Spot_Rates'][t-1])**t)       
        if output_excel:
            et.dump_array(self.swap_rate_lists['Zero_Bond_Prices'], filename = r'data\Zero_Bond_Prices.csv')
            et.dump_array(self.swap_rate_lists['Spot_Rates'], filename = r'data\Spot_Rates.csv')
            et.dump_array(self.swap_rate_lists['Forward_Rates'], filename = r'data\Forward_Rates.csv')





#if __name__ == '__main__':
#    file_name = r'..\..\Feuille_de_calcul_ALM.xlsm'
#    sm = Smith_Wilson(file_name)
    # =========================================================
    # get time horizon
    #end_time = int(xw.Range('Smith Wilson extrapolation', 'C3').value)
    # =========================================================
    # get time step
    #s = int(xw.Range('Smith Wilson extrapolation', 'C9').value)
    #sm.get_time_step(s)
    # =========================================================
    # get UFR
    #UFR = xw.Range('Smith Wilson extrapolation', 'C5').value/100
    #sm.get_UFR(UFR)
    # =========================================================
    # get alpha
    #alpha = xw.Range('Smith Wilson extrapolation', 'C7').value
    #sm.get_alpha(alpha)
    #sm.SM_extrapolation(end_time = end_time)