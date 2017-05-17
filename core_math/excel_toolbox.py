# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 16:58:38 2015

@author: FR007967


This script is meant to encapsulate a number of functions which allow to easily load and dump data within excel spreadsheets
"""

## Python packages
import xlrd as xl
import xlwt as xw
import numpy as np

def dump_array(array, filename = 'dump.csv'):
    """
    Function : dump array within a spreadsheet

    Parameters :
        1. array :
            Type : array
            Function : array to dump within an excel spreadsheet
        2. filename :
            Type : string
            Function : name of the file used to dump the array
    """
    np.savetxt(filename, array, delimiter=';', fmt='%1.5f')
    print('FILE {} SAVED'.format(filename))

def get_balance_sheet(file_name, list_data = None):
    pass

def output_excel(file_name, dictionary):
    # ==================================
    # Create a Workbook
    # ==================================
    book = xw.Workbook()
    sheet = book.add_sheet("sheet_1")

    len_max = 0
    for key in dictionary.keys():
        if len(key) > len_max:
            len_max = len(key)
        for subkey in dictionary[key].keys():
            if len(subkey) > len_max:
                len_max = len(key)

    style1 = xw.easyxf('font: bold 1, color red;')
    style2 = xw.easyxf('font: bold 1, color blue;')
    sheet.col(0).width = 256 * (2 + len_max)

    n = 0
    for key in dictionary.keys():
       sheet.write(n, 0, key, style1)
       n += 1
       for subkey in dictionary[key].keys():
           sheet.write(n, 0, subkey, style2)
           for m in range(1, len(dictionary[key][subkey])+1):
               sheet.write(n, m,  dictionary[key][subkey][m-1])
           n += 1

    book.save(file_name)


def WriteDicttoExcel(file_name, obj):
    book = xw.Workbook()
    sheet = book.add_sheet("Main")
    len_max = 0
    for key in obj.keys():
        if len(key) > len_max:
            len_max = len(key)
    sheet.col(0).width = 256 * (2 + len_max)
    row = 0
    for key in obj.keys():
        sheet.write(row, 0, key)
        for col in range(1, len(obj[key])+1):
            sheet.write(row, col, obj[key][col-1])
        row += 1
    book.save(file_name)

def Write2DListtoExcel(file_name, obj):
    book = xw.Workbook()
    sheet = book.add_sheet("Main")
    row = 0
    for l in obj:
        for col in range(len(l)):
            sheet.write(row, col, l[col])
        row += 1
    book.save(file_name)

#
#if __name__ == '__main__':
#    #a = np.array([1,2,3])
#    #dump_array(a)
#    filename1 = 'test_output_function.xls'
#    l = np.ones(52)
#    list_data = []
#    list_data.append(l)
#    # Output
#    filename2 = 'test_output_excel_function.xls'
#    sub_dict_1 = collections.OrderedDict()
#    sub_dict_1['subkey 1_1'] = l
#    sub_dict_2 = collections.OrderedDict()
#    sub_dict_2['subkey 2_1'] = l
#    sub_dict_2['subkey 2_2'] = l
#    sub_dict_3 = collections.OrderedDict()
#    sub_dict_3['subkey 3_1'] = l
#    sub_dict_3['subkey 3_2'] = l
#    sub_dict_3['subkey 3_1'] = l
#    dictionary = collections.OrderedDict()
#    dictionary['key_lv_1'] = sub_dict_1
#    dictionary['key_lv_2'] = sub_dict_2
#    dictionary['key_lv_3'] = sub_dict_3
#    output_excel(file_name = filename2, dictionary = dictionary)