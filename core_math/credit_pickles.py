## Python packages
from __future__ import division
from xlrd import open_workbook
import numpy as np
import pickle



def get_historical_transition_matrix(file_name):
    """
        Method : get_historical_transition_matrix
        Function : get the historical transition matrix from the excel file
        Parameter : 
            1. file_name 
                Type : string
                Function : the file name
    """
    wb = open_workbook(file_name)
    sheet = wb.sheet_by_name('Credit_Data')
    row_begin = 3
    row_end = 11
    col_begin = 1
    col_end = 9
    historical_transition_matrix = []
        
    for row in range(row_begin, row_end):
        hist_trans_matrix=[]
        for col in range(col_begin, col_end):
            hist_trans_matrix.append(sheet.cell(row, col).value)
        historical_transition_matrix.append(hist_trans_matrix)
       
    return historical_transition_matrix
        
def get_spread(file_name):
    """
        Method : get_spread
        Function : get the market data of spread from the excel file
        Parameter : 
            1. file_name 
                Type : string
                Function : the file name
    """
        
    wb = open_workbook(file_name)
    sheet = wb.sheet_by_name('Credit_Data')
    
    row_begin = 3
    row_end = 8    
    col_begin = 13  
    col_end = 17
    spread_list = []
        
    for row in range(row_begin, row_end):
        s_list=[]
        for col in range(col_begin, col_end):
            s_list.append(sheet.cell(row, col).value)
        spread_list.append(s_list)
        
    col_index =[sheet.cell(row_begin-1, col).value for col in range(col_begin, col_end)]
    row_index=[sheet.cell(row, col_begin-1).value for row in range(row_begin, row_end)]
    return spread_list, col_index, row_index 
    

def get_prices(file_name):
    wb = open_workbook(file_name)
    sheet = wb.sheet_by_name('Credit_Data')
    row_begin = 3
    col_begin = 22
    col_end = 26
    number_of_rows = sheet.nrows
    
    ps_list=[]
    for row in range(row_begin, number_of_rows):
        p_list=[]
        for col in range(col_begin, col_end):
            p_list.append(sheet.cell(row, col).value)
        ps_list.append(p_list)
        
    ps_vector = np.asarray(ps_list)
    
    prix_marche = np.zeros((7,30))
    coupon_marche = np.zeros((7,30))
    for row in range(len(ps_vector)):
        rating = int(ps_vector[row,0])
        TtM = int(ps_vector[row,1])
        prix_marche[rating, TtM-1] = ps_vector[row,2]/100
        coupon_marche[rating, TtM-1] = ps_vector[row,3]/100
    return prix_marche, coupon_marche

        
def saving():
    """
        Method : saving
        
        Function : save objects like historical transition matrix and spreads (read from excel) to pickle files
        
        Parameter : None
    """
    
    historical_transition_matrix = get_historical_transition_matrix()
    with open('data\pickle\historical_transition_matrix.pkl', 'wb') as output:
        pickle.dump(historical_transition_matrix, output, pickle.HIGHEST_PROTOCOL)
        
        
    spread_list, col_index, row_index = get_spread()
    with open('data\pickle\spread.pkl', 'wb') as output:
        pickle.dump(spread_list, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(col_index, output, pickle.HIGHEST_PROTOCOL)    
        pickle.dump(row_index, output, pickle.HIGHEST_PROTOCOL)    
        

    
    prix_marche, coupon_marche = get_prices()
    with open('data\pickle\bonds_prices.pkl','wb') as output:
        pickle.dump(prix_marche, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(coupon_marche,output, pickle.HIGHEST_PROTOCOL)
       
       