# Python Programs
from .Smith_Wilson import Smith_Wilson
#from ..core_math.excel_toolbox import excel_toolbox as et 
## Python packages
import numpy as np
import unittest
import xlwings as xw
from xlrd import open_workbook
from xlwt import Workbook

class TestSmithWilson(unittest.TestCase):
          
    def test_Cmatrix(self):
        # Collect data specific to test
        file_name = r'Feuille_de_calcul_ALM(Working).xlsm'
        sheet_name = 'Smith Wilson extrapolation'
        
        wb = xw.books(file_name)
        # Backup existing values
        backup_maturities = xw.sheets[sheet_name].range('B16:B50').value
        backup_coupon_yields = xw.sheets[sheet_name].range('C16:C50').value
        # Values required for testing
        maturities = np.asarray([1,2,3,5])
        coupon_yields = np.asarray([1.0,2.0,2.6,3.4])      
        ## Replace values for testing
        xw.sheets[sheet_name].range('B16:C50').value = None 
        xw.sheets[sheet_name].range('B16:B50').value = maturities.reshape(4,1)
        xw.sheets[sheet_name].range('C16:C50').value = coupon_yields.reshape(4,1)
        
        wb.save()
        
        sm = Smith_Wilson(file_name = file_name, exec = False)
        
        sm.UFR = 0.042
        sm.alpha = 0.1
        sm.num_instrument = 4
        
        ## Execute function
        C_output = sm.Cmatrix(sm.swap_rate_lists['Coupon_yields'])
        C_output = C_output[:,[0,1,2,3,4]]  

        ## Test
        C = np.matrix('1.01,0,0,0,0; 0.02,1.02,0,0,0; 0.026, 0.026, 1.026, 0, 0; 0.034, 0.034, 0.034, 0.034, 1.034') 
        identical_mat = np.array_equal( C_output.round(decimals=2) , C.round(decimals=2) )

        xw.sheets[sheet_name].range('B16:C50').value = None
        xw.sheets[sheet_name].range('B16:B50').value = np.asarray(backup_maturities).reshape(len(backup_maturities),1)
        xw.sheets[sheet_name].range('C16:C50').value = np.asarray(backup_coupon_yields).reshape(len(backup_coupon_yields),1)

        self.assertTrue(identical_mat)  

    def test_Wmatrix(self):
        # Collect data specific to test
        file_name = r'Feuille_de_calcul_ALM(Working).xlsm'
        sheet_name = 'Smith Wilson extrapolation'

        # Backup existing values
        backup_maturities = xw.sheets[sheet_name].range('B16:B50').value
        backup_coupon_yields = xw.sheets[sheet_name].range('C16:C50').value
        # Values required for testing
        maturities = np.asarray([1,2,3,5])
        coupon_yields = np.asarray([0.01,0.02,0.026,0.034])
        
        ## Replace values for testing
        xw.sheets[sheet_name].range('B16:C50').value = None
        
        xw.sheets[sheet_name].range('B16:B50').value = maturities.reshape(4,1)
        xw.sheets[sheet_name].range('C16:C50').value = coupon_yields.reshape(4,1)
        
        sm = Smith_Wilson(file_name = file_name, exec = False)
        sm.UFR = 0.042
        sm.alpha = 0.1
        
        ## Execute function
        W_output = sm.Wmatrix()
        maturities_tab = np.asarray([1,2,3,4,5])
        W_output = W_output[np.ix_(maturities_tab-1,maturities_tab-1)]
        
        xw.sheets[sheet_name].range('B16:C50').value = None
        xw.sheets[sheet_name].range('B16:B50').value = np.asarray(backup_maturities).reshape(len(backup_maturities),1)
        xw.sheets[sheet_name].range('C16:C50').value = np.asarray(backup_coupon_yields).reshape(len(backup_coupon_yields),1)

        ## Test
        W = np.matrix('0.009,0.016,0.022,0.027,0.031; 0.016, 0.030, 0.041, 0.051, 0.058; 0.022, 0.041, 0.058, 0.072, 0.083; 0.027, 0.051, 0.072, 0.090, 0.104; 0.031, 0.058, 0.083, 0.104, 0.122')
        W_rounded = W.round(decimals=2, out=None)
        W_output_rounded = W_output.round(decimals=2, out=None) 
        
        identical_mat = np.array_equal( W_output_rounded , W_rounded )
        self.assertTrue(identical_mat)
  
        
    def test_Eta_vector(self):
        # Collect data specific to test
        file_name = r'Feuille_de_calcul_ALM(Working).xlsm'
        sheet_name = 'Smith Wilson extrapolation'
        
        wb = xw.books(file_name)
        # Backup existing values
        backup_maturities = xw.sheets[sheet_name].range('B16:B50').value
        backup_coupon_yields = xw.sheets[sheet_name].range('C16:C50').value
        # Values required for testing
        maturities = np.asarray([1,2,3,5])
        coupon_yields = np.asarray([1.0,2.0,2.6,3.4])      
        ## Replace values for testing
        xw.sheets[sheet_name].range('B16:C50').value = None 
        xw.sheets[sheet_name].range('B16:B50').value = maturities.reshape(4,1)
        xw.sheets[sheet_name].range('C16:C50').value = coupon_yields.reshape(4,1)
        
        wb.save()
        
        sm = Smith_Wilson(file_name = file_name, exec = False)
        
        sm.UFR = 0.042
        sm.alpha = 0.1
        sm.num_instrument = 4
        
        ## Execute function
        # C matrix
        C_output = sm.Cmatrix(sm.swap_rate_lists['Coupon_yields'])
        C_output = C_output[:,[0,1,2,3,4]]  
        # W matrix
        W_output = sm.Wmatrix()
        maturities_tab = np.asarray([1,2,3,4,5])
        W_output = W_output[np.ix_(maturities_tab-1,maturities_tab-1)]
        # Get UFR => mu
        sm.get_UFR(0.042)
        # eta vector
        eta_vector_output , C = sm.get_eta_vector(sm.swap_rate_lists['Coupon_yields'])
        
        ## Test
        eta_vector =  np.matrix('57.79, -33.5, 11.40, -5.47')
        eta_vector_output_rounded = eta_vector_output.round(decimals=0)
        eta_vector_rounded = eta_vector.round(decimals=0)
        eta_vector_rounded = np.asarray(eta_vector_rounded)
        
        identical_mat = np.allclose(eta_vector_output_rounded , eta_vector_rounded )

        xw.sheets[sheet_name].range('B16:C50').value = None
        xw.sheets[sheet_name].range('B16:B50').value = np.asarray(backup_maturities).reshape(len(backup_maturities),1)
        xw.sheets[sheet_name].range('C16:C50').value = np.asarray(backup_coupon_yields).reshape(len(backup_coupon_yields),1)

        self.assertTrue(identical_mat) 
        
def Test_SmithWilson():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSmithWilson)
    a = unittest.TextTestRunner(verbosity=0).run(suite)
    xw.sheets['Testing'].range('K49').clear_contents()
    xw.sheets['Testing'].range('K49').value = str(a)
 
