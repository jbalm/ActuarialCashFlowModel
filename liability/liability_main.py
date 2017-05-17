from .liability_data.Liabilities_data import Liabilities_data
from .liability_data.Liabilities_data_m import Liabilities_data_m
import pickle

Working_URL = r'Feuille_de_calcul_ALM(Working).xlsm'

def Liabilities_data_update():
    liabilities_data = Liabilities_data_m()
    liabilities_data.update(Working_URL)
    with open(r'data\pickle\Liabilities_data1.pkl', 'wb') as output:
        pickle.dump(liabilities_data, output, pickle.HIGHEST_PROTOCOL)
        
def Liabilities_data_test_update():
    liabilities_data = Liabilities_data()
    liabilities_data.update(Working_URL)
    with open(r'data\pickle\Liabilities_data0.pkl', 'wb') as output:
        pickle.dump(liabilities_data, output, pickle.HIGHEST_PROTOCOL)
