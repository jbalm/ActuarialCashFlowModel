# Python Programs
from ..esg.ESG_main import *
from .Assets import Assets
from .bond.Bonds_Portfolio import Bonds_Portfolio
## Python packages
import unittest
import pickle as pk
import numpy as np




class TestAssets(unittest.TestCase):
    def test_add_delete_EQ(self):
        ## Input
        temp_mk_name = xw.sheets['ESG'].range('D5').value 
        temp_time_horizon = xw.sheets['ESG'].range('D3').value 
        temp_number_trajectories = xw.sheets['ESG'].range('D1').value 
        temp_volatilité_IR = xw.sheets['Market_Environment'].range('C4').value 
        
        time_horizon = 50
        
        xw.sheets['ESG'].range('D5').value = "EUR"
        xw.sheets['ESG'].range('D3').value = time_horizon
        xw.sheets['ESG'].range('D1').value = 1
        xw.sheets['Market_Environment'].range('C4').value = 0
        
        ESG_calibrate()
        ESG_generate_scenarios(modif = False)
        ESG = None
        with open(r'data\pickle\ESG_updated.pkl', 'rb') as input:
            ESG = pickle.load(input)
        
        spot_rates = ESG.asset_data.get_list_market("EUR").spot_rates
        test_bonds_target_allocation = np.asarray([
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,6,7,8,9,10],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,7,8,9],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,7,8],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,7],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]        
            ])
            
        bonds_portfolio = Bonds_Portfolio(time_horizon = time_horizon,
                                          ESG_RN_scenarios_traj_i = ESG.scenarios[0],
                                          target_allocation= test_bonds_target_allocation,
                                          init_IR_curve= spot_rates)
        bonds_portfolio.initialize_allocation(amount=1000)
        bonds_portfolio.initialize_unit_bonds_book_value()
        bonds_portfolio.initialize_unit_bonds_market_value()
        
        Portfolio = Assets(ESG, bonds_portfolio)
        Portfolio.initialize_EQ_value(value = 100)
        
        amount = 10
        valuation_date = 10
        initial_number_EQ = Portfolio.number_EQ
        
        for time_step in range(1,time_horizon):
            Portfolio.get_asset_income_and_EQ_value(traj_i = 0, valuation_date = time_step)
            Portfolio.reset_asset_income_and_EQ_value(valuation_date = time_step)
            PMVL = Portfolio.EQ_market_value[time_step] - Portfolio.EQ_book_value[time_step]
            PVL = max(0, PMVL)
            MVL = max(0, -PMVL)
            PVL_hors_obligation = PVL
            Revenu = Portfolio.ESG_RN.EQ_model.dividend_rate * Portfolio.EQ_market_value[time_step - 1]
            income_dict = {'PMVL': PMVL, 'PVL': PVL, 'MVL': MVL, 'Revenu': Revenu, 'PVL_obligation_TF': 0, 'MVL_obligation_TF': 0, 'PMVR_hors_obligation': 0, 'PMVR_obligation_TF': 0, 'PVL_hors_obligation': PVL_hors_obligation}
            Portfolio.asset_income.append(income_dict)
            ## Execute
            if time_step == valuation_date:
                Portfolio.add_EQ(amount = amount, traj_i=0, valuation_date= valuation_date)
                Portfolio.delete_EQ(amount= amount, valuation_date= valuation_date)
        
        # Test Output
        self.assertEqual(round(Portfolio.number_EQ,3),initial_number_EQ)
        
        # Restore previous values
        xw.sheets['ESG'].range('D5').value = temp_mk_name
        xw.sheets['ESG'].range('D3').value = temp_time_horizon  
        xw.sheets['ESG'].range('D1').value  = temp_number_trajectories
        xw.sheets['Market_Environment'].range('C4').value = temp_volatilité_IR
        
    def test_add_delete_bonds(self):
        ## input
        temp_mk_name = xw.sheets['ESG'].range('D5').value 
        temp_time_horizon = xw.sheets['ESG'].range('D3').value 
        temp_number_trajectories = xw.sheets['ESG'].range('D1').value 
        temp_volatilité_IR = xw.sheets['Market_Environment'].range('C4').value 
        
        time_horizon = 50
        
        xw.sheets['ESG'].range('D5').value = "EUR"
        xw.sheets['ESG'].range('D3').value = time_horizon
        xw.sheets['ESG'].range('D1').value = 1
        xw.sheets['Market_Environment'].range('C4').value = 0
        
        ESG_calibrate()
        ESG_generate_scenarios(modif = False)
        ESG = None
        with open(r'data\pickle\ESG_updated.pkl', 'rb') as input:
            ESG = pickle.load(input)
        
        spot_rates = ESG.asset_data.get_list_market("EUR").spot_rates
        test_bonds_target_allocation = np.asarray([
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,6,7,8,9,10],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,7,8,9],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,7,8],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,7],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]        
            ])
            
        bonds_portfolio = Bonds_Portfolio(time_horizon = time_horizon,
                                          ESG_RN_scenarios_traj_i = ESG.scenarios[0],
                                          target_allocation= test_bonds_target_allocation,
                                          init_IR_curve= spot_rates)
        bonds_portfolio.initialize_allocation(amount=1000)
        bonds_portfolio.initialize_unit_bonds_book_value()
        bonds_portfolio.initialize_unit_bonds_market_value()
        
        valuation_date1 = 10
        valuation_date2 = 15
        amount = 100.
        
        for time_step in range(1, time_horizon):
            if len(bonds_portfolio.allocation_matrix) == time_step:
                bonds_portfolio.allocation_matrix.append(bonds_portfolio.allocation_matrix[-1])
        
                new_num=[]
                for k in range(7):
                    num_k = []
                    for t in range(time_step):
                        num_k.append(bonds_portfolio.num_of_bonds[k][t,:])
                    num_k.append(np.zeros(20))    
                    new_num.append(np.asarray(num_k))
                bonds_portfolio.num_of_bonds = new_num            
            
            num_of_bonds_temps =[]
            allocation_mat_temps =[]
            for k in range(7):
                num_of_bonds_k_temps = []
                allocation_mat_k_temps =[]
                for k2 in range(8):
                    num_of_bonds_k_temps.append(bonds_portfolio.num_of_bonds[k]*ESG.scenarios[0]['RN_migration_matrix'][time_step][k,k2])
                    # num_of_bonds_k_temps de dimension 8*une matrice(temps d'achat et maturité)
                    allocation_mat_k_temps.append(bonds_portfolio.allocation_matrix[time_step][k,:]*ESG.scenarios[0]['RN_migration_matrix'][time_step][k,k2])
                num_of_bonds_temps.append(num_of_bonds_k_temps)
                # num_of_bonds_temps de dimension 7*8*XXX
                allocation_mat_temps.append(allocation_mat_k_temps)
            
            # une somme sur k, selon k2, cela donne les nouveaux num_of_bonds et allocation_mat après migration
            num_of_bonds = sum(np.asarray(num_of_bonds_temps))
            allocation_mat = sum(np.asarray(allocation_mat_temps))
            # num_of_bonds de dim 8*XXX
            
            bonds_portfolio.num_of_bonds = num_of_bonds[:7]
            bonds_portfolio.allocation_matrix[time_step] = allocation_mat[:7]
            
            if time_step == valuation_date1:
                bef_MV = np.sum(np.asarray(bonds_portfolio.get_market_value_list(valuation_date1)))
                # Test
                bonds_portfolio.add_bonds(amount=amount, valuation_date=valuation_date1)
                bonds_portfolio.delete_bonds(amount=amount, valuation_date=valuation_date1, book_value=False)
                aft_MV = np.sum(np.asarray(bonds_portfolio.get_market_value_list(valuation_date1)))
                # Output
                self.assertEqual(bef_MV,aft_MV)
            elif time_step == valuation_date2:
                bef_BV = np.sum(np.asarray(bonds_portfolio.get_book_value_list(valuation_date2)))
                # Test
                bonds_portfolio.add_bonds(amount=amount, valuation_date=valuation_date2)
                bonds_portfolio.delete_bonds(amount=amount, valuation_date=valuation_date2, book_value=True)
                aft_BV = np.sum(np.asarray(bonds_portfolio.get_book_value_list(valuation_date2)))
                # Output
                self.assertEqual(bef_BV,aft_BV)
            
        # Restore previous values
        xw.sheets['ESG'].range('D5').value = temp_mk_name
        xw.sheets['ESG'].range('D3').value = temp_time_horizon  
        xw.sheets['ESG'].range('D1').value  = temp_number_trajectories
        xw.sheets['Market_Environment'].range('C4').value = temp_volatilité_IR
        
    def test_execute_unrealised_bonds(self):
        ## Input
        temp_mk_name = xw.sheets['ESG'].range('D5').value 
        temp_time_horizon = xw.sheets['ESG'].range('D3').value 
        temp_number_trajectories = xw.sheets['ESG'].range('D1').value 
        temp_volatilité_IR = xw.sheets['Market_Environment'].range('C4').value 
        
        time_horizon = 50
        
        xw.sheets['ESG'].range('D5').value = "EUR"
        xw.sheets['ESG'].range('D3').value = time_horizon
        xw.sheets['ESG'].range('D1').value = 1
        xw.sheets['Market_Environment'].range('C4').value = 0
        
        ESG_calibrate()
        ESG_generate_scenarios(modif = False)
        ESG = None
        with open(r'data\pickle\ESG_updated.pkl', 'rb') as input:
            ESG = pickle.load(input)
        
        spot_rates = ESG.asset_data.get_list_market("EUR").spot_rates
        test_bonds_target_allocation = np.asarray([
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,6,7,8,9,10],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,7,8,9],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,7,8],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,7],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]        
            ])
            
        bonds_portfolio = Bonds_Portfolio(time_horizon = time_horizon,
                                          ESG_RN_scenarios_traj_i = ESG.scenarios[0],
                                          target_allocation= test_bonds_target_allocation,
                                          init_IR_curve= spot_rates)
        bonds_portfolio.initialize_allocation(amount=1000)
        bonds_portfolio.initialize_unit_bonds_book_value()
        bonds_portfolio.initialize_unit_bonds_market_value()
        
        valuation_date = 10
        
        for time_step in range(1, time_horizon):
            if len(bonds_portfolio.allocation_matrix) == time_step:
                bonds_portfolio.allocation_matrix.append(bonds_portfolio.allocation_matrix[-1])
        
                new_num=[]
                for k in range(7):
                    num_k = []
                    for t in range(time_step):
                        num_k.append(bonds_portfolio.num_of_bonds[k][t,:])
                    num_k.append(np.zeros(20))    
                    new_num.append(np.asarray(num_k))
                bonds_portfolio.num_of_bonds = new_num            
            
            num_of_bonds_temps =[]
            allocation_mat_temps =[]
            for k in range(7):
                num_of_bonds_k_temps = []
                allocation_mat_k_temps =[]
                for k2 in range(8):
                    num_of_bonds_k_temps.append(bonds_portfolio.num_of_bonds[k]*ESG.scenarios[0]['RN_migration_matrix'][time_step][k,k2])
                    # num_of_bonds_k_temps de dimension 8*une matrice(temps d'achat et maturité)
                    allocation_mat_k_temps.append(bonds_portfolio.allocation_matrix[time_step][k,:]*ESG.scenarios[0]['RN_migration_matrix'][time_step][k,k2])
                num_of_bonds_temps.append(num_of_bonds_k_temps)
                # num_of_bonds_temps de dimension 7*8*XXX
                allocation_mat_temps.append(allocation_mat_k_temps)
            
            # une somme sur k, selon k2, cela donne les nouveaux num_of_bonds et allocation_mat après migration
            num_of_bonds = sum(np.asarray(num_of_bonds_temps))
            allocation_mat = sum(np.asarray(allocation_mat_temps))
            # num_of_bonds de dim 8*XXX
            
            bonds_portfolio.num_of_bonds = num_of_bonds[:7]
            bonds_portfolio.allocation_matrix[time_step] = allocation_mat[:7]
            
            if time_step == valuation_date:       
                bef_BV = np.sum(np.asarray(bonds_portfolio.get_book_value_list(valuation_date)))
                bef_MV = np.sum(np.asarray(bonds_portfolio.get_market_value_list(valuation_date)))
                 
                PMVL = bef_MV - bef_BV
                if PMVL > 0.:
                    amount = PMVL
                    # Test
                    bonds_portfolio.execute_unrealised_bonds_gain(amount= amount, valuation_date= valuation_date)
                    aft_BV = np.sum(np.asarray(bonds_portfolio.get_book_value_list(valuation_date)))
                    aft_MV = np.sum(np.asarray(bonds_portfolio.get_market_value_list(valuation_date)))
                    # Output
                    self.assertEqual(bef_BV,aft_BV)
                    self.assertEqual(aft_MV,aft_BV)
                else:
                    amount = np.absolute(PMVL)
                    # Test
                    bonds_portfolio.execute_unrealised_bonds_loss(amount= amount, valuation_date= valuation_date)
                    aft_BV = np.sum(np.asarray(bonds_portfolio.get_book_value_list(valuation_date)))
                    aft_MV = np.sum(np.asarray(bonds_portfolio.get_market_value_list(valuation_date)))
                    # Output
                    self.assertEqual(bef_BV,aft_BV)
                    self.assertEqual(aft_MV,aft_BV)
        
        
        # Restore previous values
        xw.sheets['ESG'].range('D5').value = temp_mk_name
        xw.sheets['ESG'].range('D3').value = temp_time_horizon  
        xw.sheets['ESG'].range('D1').value  = temp_number_trajectories
        xw.sheets['Market_Environment'].range('C4').value = temp_volatilité_IR
        
def Test_Assets():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAssets)
    a = unittest.TextTestRunner(verbosity=0).run(suite)
    xw.sheets['Testing'].range('K2').clear_contents()
    xw.sheets['Testing'].range('K2').value = str(a)
    