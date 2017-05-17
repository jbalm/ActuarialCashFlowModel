# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 15:05:48 2015

@author: Quang Dien DUONG
"""

class Balance_Sheets(object):
    """
        DOCUMENTATION WILL BE COMPLETED LATER
    """
    def __init__(self):
        self.assets = {}
        self.liabilities = {}
        self.cash_flows_in = {}
        self.cash_flows_out = {}

    def update(self, key_lv1, key_lv2, value):
        if key_lv1 == 'assets':
            self.assets[key_lv2] = value
        elif key_lv1 == 'liabilities':
            self.liabilities[key_lv2] = value
        elif key_lv1 == 'cash_flows_in':
            self.cash_flows_in[key_lv2] = value
        elif key_lv1 == 'cash_flows_out':
            self.cash_flows_out[key_lv2] = value
        else:
            raise ValueError("Could not find ", key_lv1)

    def get_value(self, key_lv1, key_lv2):
        if key_lv1 == 'assets':
            resu = self.assets[key_lv2]
        elif key_lv1 == 'liabilities':
            resu = self.liabilities[key_lv2]
        elif key_lv1 == 'cash_flows_in':
            resu = self.cash_flows_in[key_lv2]
        elif key_lv1 == 'cash_flows_out':
            resu = self.cash_flows_out[key_lv2]
        else:
            raise ValueError("Could not find ", key_lv1)
        return resu

    def get_cash_flow(self):
        return sum(self.cash_flows_in[key] for key in self.cash_flows_in.keys()) - sum(self.cash_flows_out[key] for key in self.cash_flows_out.keys())

