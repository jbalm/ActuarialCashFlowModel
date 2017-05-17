## Progam packages
from ..Model_Point import Model_Point
## Python packages
import datetime as dt
from xlrd import open_workbook
import xlrd
import numpy as np

class Liabilities_data_m(object):
    def __init__(self):
        self.model_points = []

    def get_TMG_dict(self, path, sheet_name = 'TMG_table'):
        # Initialize TMG dictionary
        self.TMG_dict ={}
        wb = xlrd.open_workbook(path)
        sheet = wb.sheet_by_name(sheet_name)
        number_of_rows = sheet.nrows
        number_of_columns = sheet.ncols
        for row in range(1, number_of_rows):
            TMG_name = str(sheet.cell(row, 0).value)
            TMG_list = []
            for col in range(1, number_of_columns):
                TMG_list.append(sheet.cell(row,col).value)
            self.TMG_dict[TMG_name] = TMG_list

    def get_surrender_rate_dict(self, path, sheet_name = 'Lapse_table'):
        # Initialize surrender dictionary
        self.surrender_dict ={}
        wb = xlrd.open_workbook(path)
        sheet = wb.sheet_by_name(sheet_name)
        number_of_rows = sheet.nrows
        number_of_columns = sheet.ncols
        for row in range(1, number_of_rows):
            prod_name = str(sheet.cell(row, 0).value)
            surrender_list = []
            for col in range(1, number_of_columns):
                surrender_list.append(sheet.cell(row,col).value)
            self.surrender_dict[prod_name] = np.concatenate((surrender_list, [surrender_list[-1] for t in range(100)]), axis = 0)

    def get_mortality_rate_dict(self, path, sheet_name = 'Mortality_table'):
        # Initialize mortality rate dictionary
        self.mortality_dict = {}
        wb = xlrd.open_workbook(path)
        sheet = wb.sheet_by_name(sheet_name)
        number_of_rows = sheet.nrows
        number_of_cols = sheet.ncols
        for col in range(1, number_of_cols):
            morta_list = []
            sexe = str(sheet.cell(0,col).value)
            for row in range(1, number_of_rows):
                morta_list.append(sheet.cell(row,col).value)
            self.mortality_dict[sexe] = np.concatenate((morta_list,[morta_list[-1] for t in range(100)]), axis = 0)

    def update(self, path, sheet_name = 'Model_Points'):
        # Retrieve TMG dictionary and Surrender Rate dictionary
        self.get_TMG_dict(path)
        self.get_surrender_rate_dict(path)
        self.get_mortality_rate_dict(path)
        # Retrieve Model Points data
        wb = xlrd.open_workbook(path)
        sheet = wb.sheet_by_name(sheet_name)
        number_of_rows = sheet.nrows
        number_of_columns = sheet.ncols
        for row in range(1, number_of_rows):
            mp_data = []
            for col in range(number_of_columns):
                cell = sheet.cell(row, col)
                if cell.ctype == 3:
                    year, month, day, hour, minute, second = xlrd.xldate_as_tuple(cell.value,wb.datemode)
                    value = dt.datetime(year,month,day)
                else:
                    value = cell.value
                mp_data.append(value)
            mp = Model_Point()
            mp.update(*mp_data)
            mp.get_seniority()
            mp.mathematical_provision.append(mp.actual_math_provision)
            mp.TMG = self.TMG_dict[mp.TMG_type]
            mp.lapse_rate = self.surrender_dict[mp.lapse_type]
            mp.mortality_rate = self.mortality_dict[mp.sexe]
            self.model_points.append(mp)

    def affiche(self):
        for mdl_point in self.model_points:
            print(mdl_point)