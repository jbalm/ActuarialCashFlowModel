## Python packages
from scipy.linalg import inv
import numpy as np

def function_optim(pi_0, alpha,mu, sigma, recovery_rate, vect, val, maturity_list,rating_list, spread_list, AAA_AA):
    
    """
        Method : function_optim
        
        Function : compute the square of the difference between the market spread and the theoretical spread of the model
        
        Parameter : 
            1. pi_0
                Type : float
                Function : initial value of the risk premium
            
            2. alpha
                Type : float
                Function : the rate of adjustment parameter
                
            3. mu
                Type : float
                Function : the long-term average parameter
                                
            4. sigma
                Type : float
                Function : the volatility parameter
            
            5. recovery_rate
                Type : float
                Function : the recovery rate parameter
                
            6. vect
                Type : matrix
                Function : the eigenvectors of historical generator matrix
                
            7. val
                Type : vector
                Function : the eingenvalues of historical generator matrix
                
            8. maturity_list
                Type : list
                Function : the maturity list of the market data of credit spread
                
            9. rating_list
                Type : list
                Function : the rating list of the market data of credit spread
            
            10. spread_list
                Type : list
                Function : the market data of credit spread depending on rating and time to maturity
                
            
            11. AAA_AA
                Type : boolean
                Function : if true, then the model is only calibrated on the rating AAA and AA
                
    """
    
    vect_t_jK = inv(vect).transpose()[-1]
        
    theoretical_spread_list_pi=[]
    for t in maturity_list:
        if sigma != 0:
            v = np.sqrt(alpha**2-2*val[:-1]*sigma**2)
            A = 2*alpha*mu/sigma**2 * np.log(2* v *np.exp(1/2*(alpha+v)* t) / ((v+alpha)*(np.exp(v* t)-1)+2*v  ) )  
            B = (-2* val[:-1]* (np.exp(v * t)-1)) / ((v+alpha)*(np.exp(v * t)-1)+2*v)
            exp_Q = np.exp(A - B*pi_0)
        else:
            exp_Q = np.exp(val[:-1]*mu*t)    
                            
        proba_to_default= np.dot(np.dot(vect[:-1,:-1],np.diag(exp_Q-1)),vect_t_jK[:-1])
           
            # c'est un vecteur de dimension K-1, et il depend de time to maturity 
            
        theoretical_spread = -1/t * np.log(1-(1-recovery_rate) * proba_to_default)
        theoretical_spread_list_pi.append(theoretical_spread)
        # theoretical_spread_list de taille len(maturity)*(K-1)
        
    shortened_theo_spread_list_pi =[]
    for k in range(len(rating_list)):
        # uniquement quand les notations sont dans l'ordre
        shortened_theo_spread_list_pi.append(np.asarray(theoretical_spread_list_pi)[:,k])       
    
    
    if AAA_AA is True:
        diff_matrix_2=(np.asarray(shortened_theo_spread_list_pi).T[:,:2] - np.asarray(spread_list)[:,:2] )**2          
        norme2_diff_spread_pi=np.sum(diff_matrix_2)
    else:
        diff_matrix_2=(np.asarray(shortened_theo_spread_list_pi).T - np.asarray(spread_list) )**2
        norme2_diff_spread_pi=np.sum(diff_matrix_2)
    
    
    return norme2_diff_spread_pi




def function_optim_v_price(pi_0, alpha,mu, sigma, recovery_rate, vect, val, price_matrix,coupon_matrix ,IR0):
    
    """
        Method : function_optim_v_price
        
        Function : compute the theorectical price of bond whose price is observable on the market
        
        Parameter : 
            1. pi_0
                Type : float
                Function : initial value of the risk premium
            
            2. alpha
                Type : float
                Function : the rate of adjustment parameter
                
            3. mu
                Type : float
                Function : the long-term average parameter
                                
            4. sigma
                Type : float
                Function : the volatility parameter
            
            5. recovery_rate
                Type : float
                Function : the recovery rate parameter
                
            6. vect
                Type : matrix
                Function : the eigenvectors of historical generator matrix
                
            7. val
                Type : vector
                Function : the eingenvalues of historical generator matrix
                
            8. price_matrix
                Type : matrix
                Function : the market price of bonds
                
            9. coupon_matrix
                Type : matrix
                Function : the coupon rate of bonds
            
            10. IR_0
                Type : vector
                Function : IR curve at time t=0
            
                
    """    
    
    
    #allocation_init = allocation_init/sum(sum(allocation_init))
    
    vect_t_jK = inv(vect).transpose()[-1]
    proba_to_survive =[]
    for TtM in range(1, len(price_matrix[0,:])+1):
        v = np.sqrt(alpha**2-2*val[:-1]*sigma**2)
        A = 2*alpha*mu/sigma**2 * np.log(2* v *np.exp(1/2*(alpha+v)* TtM) / ((v+alpha)*(np.exp(v* TtM)-1)+2*v  ) )  
        B = (-2* val[:-1]* (np.exp(v * TtM)-1)) / ((v+alpha)*(np.exp(v * TtM)-1)+2*v)
        exp_Q = np.exp(A - B*pi_0)
        
                        
        proba_to_default_TtM= np.dot(np.dot(vect[:-1, :-1],np.diag(exp_Q-1)),vect_t_jK[:-1])
        # c'est un vecteur de dimension K-1, et il depend de time to maturity 
        proba_to_survive_TtM = np.ones(len(proba_to_default_TtM))- proba_to_default_TtM
        proba_to_survive.append(proba_to_survive_TtM)
    
    proba_to_survive = np.asarray(proba_to_survive)
    
    p_theorique =0
    for TtM in range(1, len(price_matrix[0,:])+1):
        for k in range(len(price_matrix)):
            p_theorique +=   sum(np.multiply(np.divide(coupon_matrix[k,TtM-1], np.power(1+IR0[0:TtM],np.arange(1,TtM+1))), proba_to_survive[:TtM,k])) + np.multiply(np.divide(1, np.power(1+IR0[TtM-1],TtM)), proba_to_survive[TtM-1,k])
            
            # normalement, on doit calibrer les prix des obligations qui paient le même taux de coupon que celles sur le marché
                
    return p_theorique


def function_spread(pi_0, alpha,mu, sigma, recovery_rate, vect, val):
    """
        Method : function_spread
        
        Function : compute the theorectical spread
        
        Parameter : 
            1. pi_0
                Type : float
                Function : initial value of the risk premium
            
            2. alpha
                Type : float
                Function : the rate of adjustment parameter
                
            3. mu
                Type : float
                Function : the long-term average parameter
                                
            4. sigma
                Type : float
                Function : the volatility parameter
            
            5. recovery_rate
                Type : float
                Function : the recovery rate parameter
                
            6. vect
                Type : matrix
                Function : the eigenvectors of historical generator matrix
                
            7. val
                Type : vector
                Function : the eingenvalues of historical generator matrix
                
    """
    vect_t_jK = inv(vect).transpose()[-1]
    
    theoretical_spread_list_pi=[]
    for t in range(1,21):
        v = np.sqrt(alpha**2-2*val[:-1]*sigma**2)
        A = 2*alpha*mu/sigma**2 * np.log(2* v *np.exp(1/2*(alpha+v)* t) / ((v+alpha)*(np.exp(v* t)-1)+2*v  ) )  
        B = (-2* val[:-1]* (np.exp(v * t)-1)) / ((v+alpha)*(np.exp(v * t)-1)+2*v)
        exp_Q = np.exp(A - B*pi_0)
        
                        
        proba_to_default= np.dot(np.dot(vect[:-1,:-1],np.diag(exp_Q-1)),vect_t_jK[:-1])
        
        # c'est un vecteur de dimension K-1, et il depend de time to maturity 
        
        theoretical_spread = -1/t * np.log(1-(1-recovery_rate) * proba_to_default)
        theoretical_spread_list_pi.append(theoretical_spread)
        
        
    return theoretical_spread_list_pi
    
    
#if __name__ == '__main__':
#
#        
#        
#    mu=5
#    alpha=0.1   
#    sigma=0.75
#    recovery_rate=0.35
#    
#    
#    with open('historical_transition_matrix.pkl', 'rb') as input:
#        historical_transition_matrix = pickle.load(input)
#        
#
#        
#    historical_generator_matrix = generator_matrix(historical_transition_matrix)
#        
#    w, v = la.eig(historical_generator_matrix)
#    eigenval_hist_gen = w.real
#    eigenvect_hist_gen = (v.T).real
#    for l in range(len(eigenvect_hist_gen)):
#        eigenvect_hist_gen[l] = eigenvect_hist_gen[l]/norm(eigenvect_hist_gen[l])
#    
#    eigenvect_hist_gen = eigenvect_hist_gen.T
#        
#    eigenval_hist_gen= eigenval_hist_gen      
#    eigenvect_hist_gen= eigenvect_hist_gen    
#        
#    
#    with open('spread.pkl', 'rb') as input:
#        spread_list = pickle.load(input)
#        col_index = pickle.load(input)
#        row_index = pickle.load(input)
#
#    
#    AAA_AA=True
#    def f(pi_0):
#        return function_optim(pi_0, alpha, mu, sigma, recovery_rate, eigenvect_hist_gen, eigenval_hist_gen, row_index, col_index, spread_list, AAA_AA)        



 
#    bds = [(0.001,None)]
#    res = minimize(f,x0=0.001, bounds=bds )
#    print('le pi 0 calibré est ', res.x)
#    print('Erreur:', f(pi_0=res.x[0]))
#    
#    
#    def f_penal(v):
#        return function_optim(v[0], v[1], v[2], v[3], recovery_rate, eigenvect_hist_gen, eigenval_hist_gen, row_index, col_index, spread_list, AAA_AA ) 
#        + 1000 *(v[1] - 0.1)**2 + 1000 * (v[2] - 5 )**2 + 100* (v[3] -0.75)**2
#    bdss = [(0.001,None), (0.01, 5), (1,20), (0.01,1)]
#    res_penal = minimize(f_penal, x0=[3, 0.1, 5, 0.75], bounds = bdss)
#    print('parametres...', res_penal.x)
#    
#    
#    
#    
#    with open('bonds_prices.pkl','rb') as input:
#        prix_marche = pickle.load(input) 
#        coupon_marche = pickle.load(input)
#    
#    
#    IR0 = np.asarray([ 0.0385451 ,  0.03759469,  0.03740671,  0.03754671,  0.03778315,
#        0.03804784,  0.03833469,  0.03864606,  0.03898394,  0.03935032,
#        0.03974095,  0.04013042,  0.04049706,  0.04082651,  0.04110895,
#        0.04133755,  0.04150739,  0.04161484,  0.04165719,  0.04163233,
#        0.04154272,  0.04140577,  0.04123881,  0.04105542,  0.04086619,
#        0.0406794 ,  0.04050149,  0.04033747,  0.04019122,  0.0400658 ,
#        0.03996266,  0.03987917,  0.03981225,  0.03975929,  0.03971813,
#        0.03968691,  0.03966409,  0.03964836,  0.0396386 ,  0.03963387,
#        0.03963337,  0.0396364 ,  0.03964239,  0.03965084,  0.03966133,
#        0.0396735 ,  0.03968703,  0.03970166,  0.03971717,  0.03973336,
#        0.03975007,  0.03976717,  0.03978453,  0.03980205,  0.03981965,
#        0.03983726,  0.03985481,  0.03987226,  0.03988957,  0.03990669,
#        0.0399236 ,  0.03994029,  0.03995672,  0.03997289,  0.03998878,
#        0.04000439,  0.0400197 ,  0.04003473,  0.04004946,  0.04006389,
#        0.04007802,  0.04009186,  0.04010542,  0.04011868,  0.04013166,
#        0.04014436,  0.04015679,  0.04016895,  0.04018084,  0.04019248,
#        0.04020387,  0.04021501,  0.04022591,  0.04023658,  0.04024702,
#        0.04025723,  0.04026723,  0.04027702,  0.0402866 ,  0.04029599,
#        0.04030518,  0.04031418,  0.04032299,  0.04033163,  0.04034009,
#        0.04034838,  0.04035651,  0.04036447,  0.04037228,  0.04037994,
#        0.04038745,  0.04039482,  0.04040205,  0.04040914,  0.04041609,
#        0.04042292,  0.04042963,  0.04043621,  0.04044267,  0.04044901,
#        0.04045525,  0.04046137,  0.04046738,  0.04047329,  0.0404791 ,
#        0.04048481,  0.04049042,  0.04049594,  0.04050136,  0.0405067 ,
#        0.04051195,  0.04051711,  0.04052219,  0.04052718,  0.0405321 ,
#        0.04053694,  0.0405417 ,  0.04054639,  0.04055101,  0.04055555,
#        0.04056003,  0.04056444,  0.04056878,  0.04057306,  0.04057727,
#        0.04058142,  0.04058552,  0.04058955,  0.04059352,  0.04059744,
#        0.0406013 ,  0.04060511,  0.04060887,  0.04061257,  0.04061622,
#        0.04061982,  0.04062337,  0.04062688,  0.04063034,  0.04063375])
#    
#    #pi_0 = 2
#    #a = function_optim_v_price(pi_0, alpha,mu, sigma, recovery_rate, eigenvect_hist_gen, eigenval_hist_gen, prix_marche,coupon_marche ,IR0)
#    
#    #print(a)
#    #print(sum(sum(prix_marche)))
#    
#    def g(pi_0):
#        return  np.absolute(function_optim_v_price(pi_0, alpha,mu, sigma, recovery_rate, eigenvect_hist_gen, eigenval_hist_gen, prix_marche,coupon_marche ,IR0) - sum(sum(prix_marche)))
#    





    
#    pi_solution = minimize(g,x0=0.1)
#    #pi_solution = fsolve(g, 5)
#    print('le pi 0 calibré par la méthode 2 : ', pi_solution.x)
#    print('Erreur lié: ', g(pi_0=pi_solution.x[0]) )
# 
#    pi_0 = res_penal.x[0]
#    alpha =res_penal.x[1]
#    mu = res_penal.x[2]
#    sigma = res_penal.x[3]
#    
#    #pi_0 = 5
#    spread = function_spread(pi_0, alpha,mu, sigma, recovery_rate, eigenvect_hist_gen, eigenval_hist_gen)
#    
