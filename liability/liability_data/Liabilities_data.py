## Progam packages
from ..Model_Point import Model_Point
## Python packages
import datetime as dt
from xlrd import open_workbook
import xlrd
import numpy as np
import xlwings as xw


class Liabilities_data(object):
    """
        Objective:
        ==========
        This class is meant to build up the policyholders database

        Attributes:
        ===========

        1. model_points:
            Type: array
            Function: collection of the model points characterized by its id.

        Methods:
        ========

        1. update:
    """
    def __init__(self):
        self.model_points = []

    def update(self,path):
        """
            Method: update

            Function: updates data from an excel file named "data\Liability_Data.xls".

            Parameter:
                1. path:
                    Type: string
                    Function: a single directory or a file name (By default, path = 'data\Market_Environment.xls' and the excel file must be placed in the same folder as the main executed file)
        """
        wb2 = open_workbook(path)
        sheet = wb2.sheet_by_name("MP_test")
        number_of_rows = sheet.nrows        
        
        mdp = Model_Point()
        mdp.id = str(xw.sheets['MP_test'].range('B4').value)
        mdp.average_age = int(xw.sheets['MP_test'].range('B5').value)
        mdp.sexe = str(xw.sheets['MP_test'].range('B6').value)
        # ========================================================================================
        # Souscription Date
        # ========================================================================================
        assert sheet.cell(6,1).ctype == 3, 'Souscription Date must be datetime type'
        ms_date_number = sheet.cell(6,1).value
        year, month, day, hour, minute, second = xlrd.xldate_as_tuple(ms_date_number,wb2.datemode)
        mdp.subscription_date = dt.datetime(year, month, day)
        # ========================================================================================
        # Valuation Date
        # ========================================================================================
        assert sheet.cell(7,1).ctype == 3, 'Valuation Date must be datetime type'
        ms_date_number = sheet.cell(7,1).value
        year, month, day, hour, minute, second = xlrd.xldate_as_tuple(ms_date_number,wb2.datemode)
        mdp.valuation_date = dt.datetime(year, month, day)
        mdp.get_seniority()        
        # =======================================================================================
        mdp.premium = xw.sheets['MP_test'].range('B9').value
        mdp.actual_math_provision = xw.sheets['MP_test'].range('B10').value
        mdp.mathematical_provision.append(mdp.actual_math_provision)
        # ===============================================================
        # get TMG
        mdp.TMG_type = xw.sheets['MP_test'].range('B11').value
        mdp.TMG = mdp.TMG_type * np.ones(100)
        # ===============================================================
        mdp.rate_sensibility = xw.sheets['MP_test'].range('B12').value
        mdp.margin_rate = xw.sheets['MP_test'].range('B13').value
        mdp.number_contract = xw.sheets['MP_test'].range('B14').value
        # ===============================================================
        # get lapse rate
        mdp.lapse_type = xw.sheets['MP_test'].range('B15').value
        mdp.lapse_rate = mdp.lapse_type * np.ones(100)
        # ===============================================================
        
        mortality_rate = []
        for row in range(3, number_of_rows):
            mort_rate = (sheet.cell(row, 4).value)
            mortality_rate.append(mort_rate)
        mdp.mortality_rate = np.concatenate((mortality_rate, [mortality_rate[-1] for t in range(100)]), axis = 0)
        
        self.model_points.append(mdp)

    def affiche(self):
        for mdl_point in self.model_points:
            print(mdl_point)
    