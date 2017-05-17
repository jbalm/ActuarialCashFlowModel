# Python Programs
from . import PCA
## Python packages
from sklearn.decomposition import PCA as PCA_skl
import numpy as np
import unittest
import xlwings as xw

class TestPCAFunctions(unittest.TestCase):
    
    def test_pca_swap_rate(self, file='code\internal_model\pca_test.xlsx'):
        ## Input
        swap_rates, ignore = PCA.read_swap_rate(file)
        swap_rates = np.asarray(swap_rates)
        pca = PCA_skl(n_components=6)
        pca.fit(swap_rates)
        explained_variance1 = np.asarray(pca.explained_variance_ratio_)
        
        # Test
        eigval, eigvec, a1, a2, a3, alpha_min, alpha_max, beta_min, beta_max, gamma_min, gamma_max, ignore = PCA.pca_swap_rate(file_name = file)
        explained_variance2 = eigval/np.sum(eigval)
        identical = np.array_equal(np.round(explained_variance1,3), np.round(explained_variance2,3))        
        self.assertTrue(identical)
        
def Test_PCAFunctions():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPCAFunctions)
    a = unittest.TextTestRunner(verbosity=0).run(suite)
    xw.sheets['Testing'].range('K3').clear_contents()
    xw.sheets['Testing'].range('K3').value = str(a)
    
