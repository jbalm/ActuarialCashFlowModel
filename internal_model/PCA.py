# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 11:09:38 2015

@author: Quang Dien DUONG
"""

## Python packages
from numpy import linalg as la
from xlrd import open_workbook
import numpy as np
import pickle


# ===================================================================
#               PCA on the interest swap rate
# ===================================================================

def read_swap_rate(file_name):
    book = open_workbook(file_name)
    sheet = book.sheet_by_name('Swap Rates')
    number_of_rows = sheet.nrows
    #number_of_columns  = sheet.ncols
    swap_rates = []
    maturity_range = [int(sheet.cell(2, col).value) for col in range(2,8)]
    for row in range(3, number_of_rows):
        rate_list = []
        for col in range(2, 8):
            rate_list.append(sheet.cell(row,col).value/100)
        swap_rates.append(rate_list)
    return swap_rates, maturity_range

def pca_swap_rate(file_name):
    swap_matrix, maturity_range = read_swap_rate(file_name)
    cov_mat = np.cov(np.array(swap_matrix).T)
    w, v = la.eig(cov_mat)
    eigenvalues = w.real
    eigenvectors = (v.T).real
    norm_a1 = np.linalg.norm(eigenvectors[0])
    norm_a2 = np.linalg.norm(eigenvectors[1])
    norm_a3 = np.linalg.norm(eigenvectors[2])
    a1 = np.divide(eigenvectors[0], norm_a1)
    a2 = np.divide(-eigenvectors[1], norm_a2)
    a3 = np.divide(eigenvectors[2], norm_a3)
    alpha_range = []
    beta_range = []
    gamma_range = []
    for l in swap_matrix:
        alpha_range.append(np.dot(l, a1))
        beta_range.append(np.dot(l, a2))
        gamma_range.append(np.dot(l, a3))

    alpha_min = min(alpha_range)
    alpha_max = max(alpha_range)
    beta_min = min(beta_range)
    beta_max = max(beta_range)
    gamma_min = min(gamma_range)
    gamma_max = max(gamma_range)
    return eigenvalues, eigenvectors, a1, a2, a3, alpha_min, alpha_max, beta_min, beta_max, gamma_min, gamma_max, maturity_range

# ==================================================================
#               PCA on the interest spot rate
# ==================================================================

def read_spot_rate(file_name):
    book = open_workbook(file_name)
    sheet = book.sheet_by_name('Main')
    number_of_rows = sheet.nrows
    number_of_columns = sheet.ncols
    spot_rates = []
    for row in range(number_of_rows):
        rate_list = []
        for col in range(1, number_of_columns):
            rate_list.append(sheet.cell(row, col).value)
        spot_rates.append(rate_list)
    time_horizon = len(spot_rates[0])
    return range(1, time_horizon+1), spot_rates

def get_increment_forward_curve(file_name):
    maturities, a = read_spot_rate(file_name)
    dr = []
    for i in range(len(a)-1):
        dr.append(np.subtract(a[i], a[i+1]))
    mu = np.mean(dr, axis = 0)
    x = np.subtract(dr, mu)
    N = len(x)
    cov_mat = np.multiply(1/N, np.dot(x.T, x))
    eigenvalues, eigenvectors = la.eig(cov_mat)
    w = eigenvalues.real
    v = (eigenvectors.T).real
    return w, v, a

def pca_analysis(file_name):
    w, v, a = get_increment_forward_curve(file_name)
    norm_v0 = np.linalg.norm(v[0])
    norm_v1 = np.linalg.norm(v[1])
    norm_v2 = np.linalg.norm(v[2])
    # ============================
    a1 = np.absolute(np.divide(v[0], norm_v0))
    #PC1 = np.dot(a[0],a1)
    # ============================
    a2 = np.divide(v[1], norm_v1)
    #PC2 = np.dot(a[0],a2)
    # ============================
    a3 = np.divide(v[2], norm_v2)
    #PC3 = np.dot(a[0],a3)
    return a1, a2, a3

def saving():
    eigval, eigvec, a1, a2, a3, alpha_min, alpha_max, beta_min, beta_max, gamma_min, gamma_max, ignore = pca_swap_rate(file_name = 'data\Tool_extrapolation_risk_free_rate.xlsx')
    with open('data\pickle\pca_swap.pkl', 'wb') as output:
        pickle.dump(a1, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(a2, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(a3, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(alpha_min, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(alpha_max, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(beta_min, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(beta_max, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(gamma_min, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(gamma_max, output, pickle.HIGHEST_PROTOCOL)


if __name__=='__main__':
    #file_name = 'Continuously_Compounded_Spot_Rate.xls'
    # Test read_spot_rate method
    #maturity, spot_rate = read_spot_rate(file_name)
    #plt.plot(maturity, spot_rate[0])
    # Test pca_analysis method
    #a1, a2, a3 = pca_analysis(file_name)
    #with open('pca.pkl', 'wb') as output:
    #    pickle.dump(a1, output, pickle.HIGHEST_PROTOCOL)
    #    pickle.dump(a2, output, pickle.HIGHEST_PROTOCOL)
    #    pickle.dump(a3, output, pickle.HIGHEST_PROTOCOL)
    #plt.plot(maturity, a1)
    #plt.plot(maturity, a2)
    #plt.plot(maturity, a3)
    #plt.show()
    #et.dump_array(a1, filename = 'a1.csv')
    #et.dump_array(a2, filename = 'a2.csv')
    #et.dump_array(a3, filename = 'a3.csv')


    # ==================================================
    #           test read_swap_rate method
    # ==================================================
    sr, ignore = read_swap_rate(r'C:\Users\FR011526\Documents\ALM_credit(working)_modified\code\internal_model\pca_test.xlsx')
    #print(sr[0])
    #print(sr[-1])
    # =================================================
    #           test pca_swap_rate method
    # =================================================
    filename = r'C:\Users\FR011526\Documents\ALM_credit(working)_modified\code\internal_model\pca_test.xlsx'
    eigval, eigvec, a1, a2, a3, alpha_min, alpha_max, beta_min, beta_max, gamma_min, gamma_max, ignore = pca_swap_rate(file_name = filename)
    #print("Explained variance of the first principal component  = ", eigval[0]/(sum(eigval)))
    #print("Explained variance of the second principal component = ", eigval[1]/(sum(eigval)))
    #print("Explained variance of the third principal component  = ", eigval[2]/(sum(eigval)))
    #print("alpha_min = ", alpha_min)
    #print("alpha_max = ", alpha_max)
    #print("beta_min  = ", beta_min)
    #print("beta_max  = ", beta_max)
    #print("gamma_min = ", gamma_min)
    #print("gamma_max = ", gamma_max)
    #et.dump_array(a1, filename = 'First_vector_loading.csv')
    #et.dump_array(a2, filename = 'Second_vector_loading.csv')
    #et.dump_array(a3, filename = 'Third_vector_loading.csv')
    #saving()
    # =======================================================
    # Test pca.pkl
    # =======================================================
    #with open('pca_swap.pkl', 'rb') as input:
    #    a1 = pickle.load(input)
    #    a2 = pickle.load(input)
    #    a3 = pickle.load(input)
    #    alpha_min = pickle.load(input)
    #    alpha_max = pickle.load(input)
    #    beta_min = pickle.load(input)
    #    beta_max = pickle.load(input)
    #    gamma_min = pickle.load(input)
    #    gamma_max = pickle.load(input)

    #print("alpha_min = ", alpha_min)
    #print("alpha_max = ", alpha_max)
    #print("beta_min  = ", beta_min)
    #print("beta_max  = ", beta_max)
    #print("gamma_min = ", gamma_min)
    #print("gamma_max = ", gamma_max)