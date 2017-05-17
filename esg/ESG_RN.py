## Python packages
import numpy as np

def discount_factor_function_EQ(trajectory, begin, end):
    out = 1
    for t in range(begin, end):
        v = (trajectory[t] + trajectory[t+1])/2
        out *= 1/(1+v)
    return out

class ESG_RN(object):
    """
        This class is meant to
            1.Take models and market environment data as input
               2.Calibrate the models
               3.Generates multi-variate scenarios

           Attributes
           ==========

           1. asset_data:
               Type: instance of Asset_data class
               Function: corresponding market environment data

           2. IR_model:
               Type: instance of IR_model class
               Function: corresponding interest rate model

           3. EQ_model:
               Type: instance of EQ_model class
               Function: equity model like Black-Scholes-Merton geometric Brownnian motion model

           4. time_horizon:
               Type: int
            Function: time horizon of the simulation for the overall system analysis

            5. number_trajectories:
                Type: int
                Function: number of scenarios
    
            6. scenarios:
                Type: dictionary
                Function: return EQ_prices, EQ_incomes, EQ_return_rates (see EQ_model class for more details) over time horizon
                          IR_curves, IR_deflators (see IR_model class for more details) over time horizon
    
            7. market_name:
                Type: string
                Function: market environment's name

        Methods
        =======

        1. add_corr_matrix:

        2. add_num_instrument:

        3. add_fixed_seed:

        4. add_market_name:

        5. calibrate_models:

        6. get_scenario:

    """

    def __init__(self, asset_data, IRmodel, EQmodel, Creditmodel) :
        self.asset_data = asset_data
        self.IR_model = IRmodel
        self.EQ_model = EQmodel
        self.credit_model = Creditmodel
        
        #Define other containers
        self.scenarios = {}
        self.market_name = None
        
        # ==================================================================
        # These following attributs are meant to study the BEL sensitivities
        # ==================================================================
        self.alpha = None                    # Shock level on PC1
        self.beta = None                     # Shock level on PC2
        self.gamma = None                    # Shock level on PC3
        self.market_equity_shock = 0         # Shock level on EQ initial value
    
    def update_time_horizon(self, time_horizon):
        self.time_horizon = time_horizon
        # Assign time_horizon for IR_model, EQ_model and Credit_model
        self.IR_model.time_horizon = time_horizon
        self.EQ_model.time_horizon = time_horizon
        self.credit_model.time_horizon = time_horizon
        
    def get_seed(self):
        self.fixed_seed = [2016 + i for i in range(self.number_trajectories)]

    def add_corr_matrix(self, corr_matrix):
        """
            Method: add_corr_matrix

            Function: implement correlated geometric Brownnian motions matrix

            Parameters:
                1. corr_matrix:
                    Type: 3 dimensional array
                    Function: correlated geometric Brownnian motions matrix
        """
        self.IR_model.corr_matrix = corr_matrix
        self.EQ_model.corr_matrix = corr_matrix
        self.credit_model.corr_matrix = corr_matrix

    def update_seed(self, fixed_seed):
        """
            Method: update_seed

            Function: update seed which is used for random variable generation within the ESG

            Parameter:
                1. fixed_seed:
                    Type: int
                    Function: the random state characteristic.
        """
        self.IR_model.fixed_seed = fixed_seed
        self.EQ_model.fixed_seed = fixed_seed
        self.credit_model.fixed_seed = fixed_seed

    def add_market_name(self, market_name):
        """
            Method: add_market_name

            Function: pick out the corresponding market environment given the market_name.

            Parameter:
                1. market_name:
                    Type: string
                    Function: market environment's name
        """
        self.IR_model.market_name = market_name
        self.EQ_model.market_name = market_name
        self.credit_model.market_name = market_name
        
        self.market_name = market_name

    def calibrate_models(self):
        """
            Method: calibrate

            Function: Calibrate the IR model and the EQ_model onto the market environment. (see IR_model.calibrate and EQ_model.calibrate for more details)

            Parameter: None
        """
        # ========================
        # Calibrate IR model first
        # ========================
        self.IR_model.alpha = self.alpha
        self.IR_model.beta = self.beta
        self.IR_model.gamma = self.gamma
        self.IR_model.calibrate(self.asset_data)
        # ==================================
        # Then Calibrate the equity EQ_model
        # ==================================
        self.EQ_model.calibrate(self.asset_data)
        # ==================================
        #  Calibrate the credit_model
        # ==================================
        self.credit_model.get_spread(self.asset_data)
        self.credit_model.get_hist_transition_matrix(self.asset_data)
        self.credit_model.calibrate_spread(self.asset_data,AAA_AA=True)
        

    def get_scenario(self, traj_i):
        """
            Method: get_scenario

            Function: Once we calibrate the model parameters, we generate the simulated paths

            Parameters:
                1. market_name:
                    Type: string
                    Function: market environment's name
                2. traj_i:
                    Type: int
                    Function: the i-th economic scenario

        """
        #update seed associated to scenario :
        seed = self.fixed_seed[traj_i]
        self.update_seed(seed)
        # Retrieve the asset data
        initial_value = self.EQ_model.EQ_init_value
        dividend_rate = self.EQ_model.dividend_rate
        
        self.EQ_model.add_dividend_rate(dividend_rate)
        
        IR_curves, IR_trajectory = self.IR_model.get_IR_curve()
        IR_deflators = self.IR_model.get_deflator()
        
        self.EQ_model.add_dividend_rate(dividend_rate)      
        
        self.EQ_model.add_short_rate_trajectory(self.IR_model.trajectory)
        # Keep track of the number of scenarios generated
        EQ_prices, EQ_total_returns, EQ_perf_rates, EQ_return_rates = self.EQ_model.get_EQ_prices(initial_value, market_equity_shock = self.market_equity_shock)
        
        
        RN_migration_matrix, spreads = self.credit_model.generate_spreads_and_matrix()
        
        actuarial_spreads =[]
        for i in range(7):
            actuarial_spreads_i =  np.add(1, IR_curves[:self.time_horizon+1,:len(spreads[0])]) * np.add(np.exp(np.asarray(spreads)[:,:,i]), -1) 
            actuarial_spreads.append(actuarial_spreads_i)
    
        rating_based_deflators=[]
        for i in range(7):
            rating_based_deflators_i = np.power(1./np.add(1,np.add(IR_curves[:self.time_horizon+1,:len(spreads[0])], np.asarray(actuarial_spreads)[i,:,:])), np.arange(1, len(spreads[0])+1))
            rating_based_deflators.append(rating_based_deflators_i)
        # ========================================================================================================================================
        ti = {'EQ_prices':EQ_prices, 'EQ_total_returns':EQ_total_returns, 'EQ_return_rates':EQ_return_rates, 
              'IR_curves':IR_curves, 'Deflators':IR_deflators, 'Short_rates':IR_trajectory,
              'RN_migration_matrix':RN_migration_matrix, 'spreads':actuarial_spreads, 'spreads_spot':spreads,
              'rating_based_deflators':rating_based_deflators}
        
        self.scenarios[traj_i] = ti
        
    def test_martingale_IR(self):
        """
            Need documentation !!!!
        """
        # ==================================================
        # Test of the martingale hypothesis on Interest Rate
        # ==================================================
        var = []
        mean = []
        for time_step in range(1, self.time_horizon):
            discount_factor = []
            epsilon = []
            i = 1
            while i <= self.number_trajectories:
                IR_trajectory = self.scenarios[i-1]['Short_rates']
                discount_factor.append(self.IR_model.DF(trajectory = IR_trajectory, begin = 1, end = time_step+1))
                i += 1
            var.append(np.std(discount_factor))
            mean.append(np.mean(discount_factor))
            epsilon.append(np.absolute(np.mean(discount_factor)/self.IR_model.zcb_price[0, time_step - 1] - 1))
        return np.max(epsilon), var, mean
       
    def test_martingale_EQ(self):
        """
            Need documentation !!!
        """
        # ===========================================
        # Test of the martingale hypothesis on Equity
        # ===========================================
        var = []
        mean = []
        for time_step in range(1, self.time_horizon):
            discount_factor = []
            epsilon = []
            i = 1
            while i <= self.number_trajectories:
                IR_trajectory = self.scenarios[i-1]['Short_rates']
                DF = discount_factor_function_EQ(trajectory = IR_trajectory, begin = 0, end = time_step)
                EQ_price = self.scenarios[i-1]['EQ_total_returns'][time_step]
                discount_factor.append(DF * EQ_price)
                i += 1
            var.append(np.std(discount_factor))
            mean.append(np.mean(discount_factor))
            epsilon.append(np.absolute(np.mean(discount_factor)/self.EQ_model.EQ_init_value - 1))
        return np.max(epsilon), var, mean
           

    def test_martingale_credit(self, rating, maturity):
        """
            Method: test_martingale_credit

            Function: to test the martingale property of bond's price taking into account the coupon and the default probability

            Parameter:
                1. rating
                    Type: int
                    Function: the rating class of the bond
                2. maturity
                    Type: int
                    Function: the time to maturity of the bond        
        """
        
        vect_t_jK = inv(self.credit_model.eigenvect_hist_gen).transpose()[-1]
        
        proba_theo = []
        for t in range(1, maturity+1):
            if self.credit_model.sigma !=0:
                v = np.sqrt(self.credit_model.alpha**2-2*self.credit_model.eigenval_hist_gen[:-1]*self.credit_model.sigma**2)
                A = 2*self.credit_model.alpha*self.credit_model.mu/self.credit_model.sigma**2 * np.log(2* v *np.exp(1/2*(self.credit_model.alpha+v)* t) / ((v+self.credit_model.alpha)*(np.exp(v* t)-1)+2*v  ) )  
                B = (-2* self.credit_model.eigenval_hist_gen[:-1]* (np.exp(v * t)-1)) / ((v+self.credit_model.alpha)*(np.exp(v * t)-1)+2*v)
                exp_Q = np.exp(A - B*self.credit_model.pi_0)
            else:
                exp_Q = np.exp(self.credit_model.eigenval_hist_gen[:-1]*self.credit_model.mu*t)

            proba_to_default= np.dot(np.dot(self.credit_model.eigenvect_hist_gen[:-1,:-1],np.diag(exp_Q-1)),vect_t_jK[:-1])
            proba_theo.append(proba_to_default[rating])
                  
        proba_theo = np.asarray(proba_theo)
        proba_theo_diff=[proba_theo[0]]
        for t in range(1, maturity):
            proba_theo_diff.append(proba_theo[t]-proba_theo[t-1])
        
        proba_theo_diff = np.asarray(proba_theo_diff)
        
        
        defl = np.power(1./np.add(1,self.IR_model.spot_rate[:maturity]), np.arange(1, maturity+1))
        each_interval = self.IR_model.spot_rate[maturity-1] *  (1-proba_theo) + self.credit_model.recovery_rate * proba_theo_diff
        sum_interval = np.sum( defl * each_interval)
        
        price = sum_interval + np.power(1./np.add(1,self.IR_model.spot_rate[maturity-1]), maturity) * (1 - proba_theo[-1])
        
        
        val_list = []
        val_list_2 =[]
        diff_list =[]
        diff_list_2 = []
        
        for traj in range(self.number_trajectories):
            val2 = np.asarray(self.scenarios[traj]['rating_based_deflators'][rating])[0, maturity-1] + np.sum(np.asarray(self.scenarios[traj]['rating_based_deflators'][rating])[0,:maturity-1])* self.IR_model.spot_rate[maturity-1]
            val_list_2.append(val2)
            diff_list_2.append((np.sum(val_list_2)/len(val_list_2) - price))
            val = 0
            matrice =np.diag(np.ones(8))
            proba_default_1_year =[0]
            counter =0
            
            for time_step in range(1,maturity+1):
                matrice = np.dot(matrice,np.asarray(self.scenarios[traj]['RN_migration_matrix'][time_step-1])[:,:] )
                proba_default_new = matrice[rating,-1] - proba_default_1_year[-1]
                proba_default_1_year.append(matrice[rating,-1])
                
                CF_default =   proba_default_new * self.credit_model.recovery_rate # les nouveaux défauté
                if time_step != maturity:
                    counter += proba_default_new

                number_non_default = sum(matrice[rating,:-1])
                
                CF_non_default = number_non_default* self.IR_model.spot_rate[maturity-1]

                val_actualise = (CF_default + CF_non_default)* np.asarray(self.scenarios[traj]['Deflators'])[0,time_step-1]
                val += val_actualise
                if time_step == maturity:
                    val += np.asarray(self.scenarios[traj]['Deflators'])[0,time_step-1]*number_non_default
                    #counter += number_non_default
                #if traj ==1:
                    #print(val)
            val_list.append(val)
            diff_list.append((np.sum(val_list)/len(val_list) - price) )
        # main
        sigma = np.std(val_list)
        N = self.number_trajectories
        mu = np.mean(val_list)
        upper_bound = mu + 1.96 * sigma/(np.sqrt(N))
        lower_bound = mu - 1.96 * sigma/(np.sqrt(N))
        #return diff_list, diff_list_2, val_list, val_list_2, price
        return upper_bound, lower_bound, price
        
        

#if __name__ == '__main__':
#    Working_URL = r'..\..\Feuille_de_calcul_ALM(Working).xlsm'
#    
#    data = Asset_data()
#    data.update(Working_URL)
#    market = data.get_list_market('EUR')
#    #=========
#    # EQ_model
#    #=========
#    Equity = EQ_model.GBM_constant_volatility()
#    #=========
#    # IR_model
#    #=========
#    Interest_rate = IR_model.Hull_White_one_factor()  
#    #=========
#    # credit_model
#    #=========
#    credit = credit_model.JLT()
#    #====================================
#    # time_horizon and correlation matrix
#    #====================================
#    time_horizon = int(xw.sheets['ESG'].range('D3').value)
#    corr_matrix = market.corr_matrix
#    #====================
#    # number_trajectories
#    #====================
#    number_trajectories = int(xw.sheets['ESG'].range('D1').value)
#    # ==========================
#    ESG = ESG_RN(data,Interest_rate,Equity,credit)
#    ESG.number_trajectories = number_trajectories
#    ESG.update_time_horizon(time_horizon)
#    ESG.add_corr_matrix(corr_matrix)
#    ESG.add_market_name(market.name)
#    ESG.calibrate_models()
#    ESG.get_seed()
#    
#    for traj in range(number_trajectories):
#        ESG.get_scenario(traj_i=traj)
#        
#    plt.plot(range(time_horizon), ESG.scenarios[0]['Short_rates'][:time_horizon])
#    plt.show()
#    
#    plt.plot(range(time_horizon), ESG.scenarios[0]['EQ_prices'][:time_horizon])
#    plt.show()
#    
#    
    
    
    
    """plt.plot(range(Interest_rate.time_horizon), ESG.scenarios[0]['EQ_prices'], label = 'EQ_prices')
    plt.plot(range(Interest_rate.time_horizon), ESG.scenarios[0]['EQ_total_returns'], label = 'EQ_total_returns')
    
    plt.xlabel('Time (in years)')
    plt.ylabel('Equity Indice')
    plt.title('Geometric Brownian Motion model with constant volatility')
    plt.legend()
    plt.show()
    
    plt.plot(range(20), np.asarray(ESG.scenarios[0]['spreads'][0])[0,:], label = 'spreads AAA')
    plt.plot(range(20), np.asarray(ESG.scenarios[0]['spreads'][1])[0,:], label = 'spreads AA')
    plt.plot(range(20), np.asarray(ESG.scenarios[0]['spreads'][2])[0,:], label = 'spreads A')
    plt.plot(range(20), np.asarray(ESG.scenarios[0]['spreads'][3])[0,:], label = 'spreads BBB')
    plt.plot(range(20), np.asarray(ESG.scenarios[0]['spreads'][4])[0,:], label = 'spreads BB')
    plt.plot(range(20), np.asarray(ESG.scenarios[0]['spreads'][5])[0,:], label = 'spreads B')
    plt.plot(range(20), np.asarray(ESG.scenarios[0]['spreads'][6])[0,:], label = 'spreads CCC')
   
    
    plt.xlabel('Time (in years)')
    plt.ylabel('spreads')
    plt.title('spreads of different ratings at t=0')
    plt.legend()
    plt.show()
    
    
    plt.plot(range(20), np.asarray(ESG.scenarios[0]['spreads_spot'][0])[:,0], label = 'spreads AAA')
    plt.plot(range(20), np.asarray(ESG.scenarios[0]['spreads_spot'][0])[:,1], label = 'spreads AA')
    plt.plot(range(20), np.asarray(ESG.scenarios[0]['spreads_spot'][0])[:,2], label = 'spreads A')
    plt.plot(range(20), np.asarray(ESG.scenarios[0]['spreads_spot'][0])[:,3], label = 'spreads BBB')
    plt.plot(range(20), np.asarray(ESG.scenarios[0]['spreads_spot'][0])[:,4], label = 'spreads BB')
    plt.plot(range(20), np.asarray(ESG.scenarios[0]['spreads_spot'][0])[:,5], label = 'spreads B')
    plt.plot(range(20), np.asarray(ESG.scenarios[0]['spreads_spot'][0])[:,6], label = 'spreads CCC')
   
    
    plt.xlabel('Time (in years)')
    plt.ylabel('spreads')
    plt.title('spot spreads of different ratings at t=0')
    plt.legend()
    plt.show()
    
    
    plt.plot(range(20), ESG.scenarios[0]['Deflators'][0,:20], label = ' risk free ')
    plt.plot(range(20), np.asarray(ESG.scenarios[0]['rating_based_deflators'][0])[0,:], label = ' AAA')
    plt.plot(range(20), np.asarray(ESG.scenarios[0]['rating_based_deflators'][1])[0,:], label = ' AA')
    plt.plot(range(20), np.asarray(ESG.scenarios[0]['rating_based_deflators'][2])[0,:], label = ' A')
    plt.plot(range(20), np.asarray(ESG.scenarios[0]['rating_based_deflators'][3])[0,:], label = ' BBB')
    plt.plot(range(20), np.asarray(ESG.scenarios[0]['rating_based_deflators'][4])[0,:], label = ' BB')
    plt.plot(range(20), np.asarray(ESG.scenarios[0]['rating_based_deflators'][5])[0,:], label = ' B')
    plt.plot(range(20), np.asarray(ESG.scenarios[0]['rating_based_deflators'][6])[0,:], label = ' CCC')
    
    
    plt.xlabel('Time (in years)')
    plt.ylabel('rating based deflator')
    plt.title('rating based deflator at t=0')
    plt.legend()
    plt.show()"""


