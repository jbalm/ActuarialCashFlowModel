# Python Programs
from . import functions_credit
#from ..core_math.excel_toolbox import excel_toolbox as et 
## Python packages
import numpy as np
import unittest
import pickle 
import xlwings as xw

class TestFunctionCredit(unittest.TestCase):
          
    def test_fact(self):
        self.assertEqual(functions_credit.fact(1),1)
        self.assertEqual(functions_credit.fact(10),3628800)

    def test_diag_matrix(self):
        ### Input functions
        mat_A = np.matrix('2,-1,0; -1,0,2; -1,-2,4')
       # Execute function
        P,D,P_inv = functions_credit.diag_matrix(mat_A)
        res_A = np.dot(np.dot(P,D),P_inv)
        # Test
        identical_mat_A = np.array_equal(np.array(res_A.round(decimals=2)), np.array(mat_A))
        self.assertTrue(identical_mat_A)
            
    def test_generator_matrix(self):
        # Input data
        with open('data\pickle\historical_transition_matrix.pkl', 'rb') as input:
            historical_transition_matrix = pickle.load(input)
            
        ### Execute function
        generator= functions_credit.generator_matrix(historical_transition_matrix)
        
        ### test output
        generator_expected = np.matrix('-6.21514196e-02, 6.20507783e-02, 4.15425424e-06, 5.86161767e-06, 5.00878176e-06, 2.76352434e-05, 1.09027149e-04, 5.21000537e-05; 2.25176281e-02, -1.12883379e-01, 8.13383698e-02, 1.89613619e-03, 3.20815086e-03, 2.37667333e-03, 2.51094053e-04, 1.29428837e-03; 1.69557137e-02, 3.75325542e-02, -1.13742915e-01, 4.94417097e-02, 3.72963067e-03, 4.22646616e-03, 0.00000000e+00, 1.85620504e-03; 1.62612024e-02, 1.59140165e-02, 5.77992721e-02, -1.30890913e-01, 2.37662606e-02, 5.96790526e-03, 7.53905945e-03, 3.64225885e-03; 0.00000000e+00, 5.62895560e-03, 1.18148693e-02, 7.76577797e-02, -1.99508524e-01, 8.19527762e-02, 5.64595570e-03, 1.68123276e-02; 0.00000000e+00, 0.00000000e+00, 1.37694066e-02, 1.49271265e-02, 7.58396271e-02, -2.23577332e-01, 7.43893517e-02,  4.45360632e-02; 0.00000000e+00, 4.00406374e-05, 1.02522921e-02, 1.58101388e-02, 2.97984279e-02, 1.12971607e-01,-3.19110235e-01, 1.50360743e-01;0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,  0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00')
        
        identical = np.array_equal(np.array(generator.round(decimals=5)), np.array(generator_expected.round(decimals=5)))
        self.assertTrue(identical)
 	
def Test_FunctionsCredit():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestFunctionCredit)
    a = unittest.TextTestRunner(verbosity=0).run(suite)
    xw.sheets['Testing'].range('E2').clear_contents()
    xw.sheets['Testing'].range('E2').value = str(a)
 
