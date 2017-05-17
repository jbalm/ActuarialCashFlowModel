## Python packages
import numpy as np

class Bonds_Portfolio(object):
    """
        Objective :
            ========
            The Bonds_Portfolio class model all the bond related operations
        Attributes :
            ========
            
        
        Methods:
            ========
            get_num_of_bond
            initialize_allocation
            get_market_price_of_bonds (not called)
            get_book_value (not called)
            get_book_value_list 
            get_market_value (not called)
            get_market_value_list
            initialize_unit_bonds_book_value
            initialize_unit_bonds_market_value
            get_coupon_list
            valorize_bond_income
            add_bonds
            delete_bonds
            execute_unrealised_bonds_gain
            execute_unrealised_bonds_loss
            
    """
    def __init__(self, time_horizon , ESG_RN_scenarios_traj_i, target_allocation, init_IR_curve = None):
        """
        Input: 
        ======
                1. time_horizon
                    Type : int
                    Function : time horizon
                2. ESG_RN_scenarios_traj_i
                    Type : dictionary 
                    Function : the i-th trajectory of the risk-neutral ESG
                3. target_allocation
                    Type : array 2D [rating, Time to maturity]
                    Function : the target allocation matrix of bond portfolio
        
        Output: 
        =======
            Instancier un objet
        """
        

        self.time_horizon = time_horizon
        
        self.max_maturity=20

        self.rating_based_deflators= ESG_RN_scenarios_traj_i['rating_based_deflators']
        self.IR_curves = ESG_RN_scenarios_traj_i['IR_curves']
        self.spreads = ESG_RN_scenarios_traj_i['spreads']
        self.spreads_spot = ESG_RN_scenarios_traj_i['spreads_spot']
        
        self.book_yield = []
        
        if init_IR_curve is not None:
            self.IR_curves[0] = init_IR_curve
            for k in range(7):
                book_yield_k = np.add(self.IR_curves[:self.time_horizon+1,:len(self.spreads[0][0,:])], np.asarray(self.spreads)[k,:,:])
                self.book_yield.append(book_yield_k)
                
            actuarial_spreads =[]
            for i in range(7):
                actuarial_spreads_i =  np.add(1, self.IR_curves[:self.time_horizon+1,:len(self.spreads_spot[0])]) * np.add(np.exp(np.asarray(self.spreads_spot)[:,:,i]), -1) 
                actuarial_spreads.append(actuarial_spreads_i)
            self.init_rating_based_deflators=[]
            for i in range(7):
                rating_based_deflators_i = np.power(1./np.add(1,np.add(self.IR_curves[:self.time_horizon+1,:len(self.spreads_spot[0])], np.asarray(actuarial_spreads)[i,:,:])), np.arange(1, len(self.spreads_spot[0])+1))
                self.init_rating_based_deflators.append(rating_based_deflators_i)
        
        else:
            for k in range(7):
                book_yield_k = np.add(self.IR_curves[:self.time_horizon+1,:len(self.spreads[0][0,:])], np.asarray(self.spreads)[k,:,:])
                self.book_yield.append(book_yield_k)
                
        
        
        self.target_allocation = target_allocation
        # à définir
        
        #======================================================================
        # on initialise le portefeuille des obligations avec l'allocation cible
        # A t'on besoins de créer une méthode pour mettre cette partie à part
        
        self.allocation_matrix=[]
        
        self.num_of_bonds=[]
        
        
        # liste des book-values unitaires
        
        self.unit_bonds_book_value = []
        self.unit_bonds_market_value = []

    
    
    def get_num_of_bond(self, amount, purchase_date, rating, TtM):
        
        """
            Method: get_num_of_bond

            Function: compute the number of bond (not necessarily un integer) of un certain characteristic we can purchase with a certain amount of money

            Parameter:
                1. amount
                    Type: float (positive only)
                    Function: the cash used for the purchase
                2. purchase_date
                    Type: int
                    Function: purchase date (with reference to the time step)
                3. rating
                    Type: int
                    Function: the rating of the bond on question (rescaled in range(7))
                4. TtM
                    Type: int
                    Function: the time to maturity of the bond on question
            Output:
            ======
                Return: 
        
        """        
        
        coupon_MV_deflator_k_t_T = sum(np.asarray(self.init_rating_based_deflators[rating])[purchase_date,:TtM]) 
        
        number_bond= amount/(np.asarray(self.init_rating_based_deflators[rating])[purchase_date,TtM-1]+ coupon_MV_deflator_k_t_T * self.IR_curves[purchase_date, TtM-1])
        
        
        return number_bond
        

    def initialize_allocation(self, amount):
        """
            Method: initialize_allocation

            Function: initialize the allocation

            Parameter:
                1. amount
                    Type: float (positive only)
                    Function: the cash used for the allocation
                
        
        """         
        BV_0 =amount*self.target_allocation/sum(sum(self.target_allocation))
        
        
        num_k_add_dim = []
        allo_mat_k =[]
        for k in range(7):
            num_k_add_dim = []
            allo_mat_k_T = []
            for dim_supp in range(1):
                num_k_T = []
                for TtM in range(1,21):
                    num_k_T.append(self.get_num_of_bond(BV_0[k, TtM-1], 0, k, TtM))
                    allo_mat_k_T.append(self.get_num_of_bond(BV_0[k, TtM-1], 0, k, TtM))
                num_k_add_dim.append(num_k_T)
            self.num_of_bonds.append(np.asarray(num_k_add_dim))
            allo_mat_k.append(allo_mat_k_T)
        self.allocation_matrix.append(np.asarray(allo_mat_k))    
            
        
        #si on raisonne en proportionnel pour toutes les ventes et les achats, coupon equivalent devient une notion utile
        #coupon_rate = self.IR_curves[0,:range(20)]* np.ones((7,20))
        # repeat 7 times, dim 7*20
        #self.coupon_equivalent.append(coupon_rate)
                  
    
        
    def get_market_price_of_bonds(self, valuation_date):
        """
            Method: get_market_price_of_bonds

            Function: compute the market price of bonds

        
        """  
        price=[]
        for k in range(7):
            price_k = []
            for TtM in range(1,21):
                coupon_MV_deflator_k_t_T = sum(np.asarray(self.rating_based_deflators[k])[valuation_date,:TtM])
                price_k_T = np.asarray(self.rating_based_deflators[k])[valuation_date,TtM-1]+ coupon_MV_deflator_k_t_T * self.IR_curves[valuation_date, TtM-1]
                price_k.append(price_k_T)    
            price.append(price_k)
        
        price = np.asarray(price)
        
        return price


          
    def initialize_unit_bonds_book_value(self):
        """
            Method: initialize_unit_bonds_book_value

            Function: initialize the unit bonds book value depending on the ESG trajectory

            Parameter: Non
            
            Output: 
                1. unit_bonds_book_value
                    Type : list of dimension 7 x time_horizon (for valuation date)x maturity_max x time_horizon (for purchase date)
                    Function : the unit bonds book value
                
        
        """ 
        for k in range(7):
            BV_k = []
            for valuation_date in range(self.time_horizon):
            
                BV_k_s = []
                for TtM in range(1, 21):
                # TtM par rapport à valuation_date
                    BV_k_s_T = []
                    for t in range(self.time_horizon):
                        if t in range(valuation_date + TtM - 20, valuation_date+1):
                            # la borne de gauche est interprétée comme : tu peux pas "acheter" des obligations de maturité supérieur à 20 ans à la date d'achat
                            # la borne de droite : on tient compte des obligations acheté aujourd'hui
                            coupon_BV_deflator = sum(1./np.power(1+ np.asarray(self.book_yield[k])[t, (valuation_date-t):(TtM + valuation_date -t)], np.arange(1,TtM+1)))
                            BV_k_s_T.append(1./np.power(1+ np.asarray(self.book_yield[k])[t,TtM + valuation_date-t-1], TtM) + self.IR_curves[t,TtM + valuation_date -t-1] * coupon_BV_deflator)
                        else:
                            BV_k_s_T.append(0)
                    BV_k_s.append(BV_k_s_T)
                BV_k.append(BV_k_s)
            self.unit_bonds_book_value.append(BV_k)
        # c'est un objet de dim 7*20*20*20 (= rating * date d'évaluation * maturité * date d'achat)

    
    def initialize_unit_bonds_market_value(self):
        """
            Method: initialize_unit_bonds_market_value

            Function: initialize the unit bonds market value depending on the ESG trajectory

            Parameter: Non
            
            Output: 
                1. unit_bonds_market_value
                    Type : list of dimension 7 x time_horizon (for valuation date)x maturity_max x time_horizon (for purchase date)
                    Function : the unit bonds market value
                
        
        """ 

        for k in range(7):
            MV_k=[]
            for valuation_date in range(self.time_horizon):
                MV_k_s = []
                for TtM in range(1, 21):
                # TtM par rapport à valuation_date
                    MV_k_s_T = []
                    for t in range(self.time_horizon):
                        if t in range(valuation_date + TtM - 20, valuation_date+1):
                            # la borne de gauche est interprétée comme : tu peux pas "acheter" des obligations de maturité supérieur à 20 ans à la date d'achat
                            # la borne de droite : on tient compte des obligations acheté aujourd'hui
                            coupon_MV_deflator = sum(np.asarray(self.rating_based_deflators[k])[t,(valuation_date-t):(TtM + valuation_date -t)])
                            MV_k_s_T.append( np.asarray(self.rating_based_deflators[k])[valuation_date, TtM-1] + self.IR_curves[t,TtM + valuation_date -t-1] * coupon_MV_deflator)
                        else:
                            MV_k_s_T.append(0)
                    MV_k_s.append(MV_k_s_T)
                MV_k.append(MV_k_s)
            self.unit_bonds_market_value.append(MV_k)
        # c'est un objet de dim 7*20*20*20 (= rating * date d'évaluation * maturité * date d'achat)
              
                    
   

    def get_coupon_list(self, valuation_date):
        """
            Method: get_coupon_list

            Function: get the coupon list at a valuation date

            Parameter:
                1. valuation_date
                    Type: int
                    Function: the valuation date (with reference to the time step)
            
            Output:
                1. coupon_list
                    Type: int
                    Function: the coupons generated by the bond portfolio
        
        """
        coupon_list=[]
        for k in range(7):
            coupon_k=[]
            for TtM in range(1,21):
            # par rapport à valuation_date
                coupon_k_T=0
                for t in range(valuation_date):
                    if TtM + valuation_date - t <= 20 :
                        coupon_k_T += self.num_of_bonds[k][t,TtM-1]* self.IR_curves[t,TtM + valuation_date -1]
                        # faire attention que num_of_bonds est mis à jour dans la maturité
                coupon_k.append(coupon_k_T)
            coupon_list.append(coupon_k)
        return coupon_list


        
    def get_book_value_list(self, valuation_date):
        """
            Method: get_book_value_list

            Function: get book value list at a valuation date

            Parameter: 
                1. valuation_date
                    Type: int
                    Function: the valuation date (with reference to the time step)
            
            Output:
                1. BV_list
                    Type: list
                    Function: the book value list
        
        """
        BV_list = []           
        for k in range(7):
            BV_list_k_t = np.asarray(self.unit_bonds_book_value[k][valuation_date])[:,:valuation_date+1] * self.num_of_bonds[k][:valuation_date+1,:].T
            
            # produit élément par élément, donne un objet de dimension 20(maturité) * valuation_date(les différentes dates d'achat)
            BV_list_k = sum(BV_list_k_t.T)
            # une somme sur les différentes dates d'achat
            # ceci donne le BV équivalent
            BV_list.append(BV_list_k)
        return BV_list
                
    
    def get_market_value_list(self, valuation_date):
        """
            Method: get_market_value_list

            Function: get market value list at a valuation date

            Parameter:
                1. valuation_date
                    Type: int
                    Function: the valuation date (with reference to the time step)
            
            Output:
                1. MV_list
                    Type: list
                    Function: the market value list
        
        """
        MV_list = []
        for k in range(7):
            MV_list_k_t = np.asarray(self.unit_bonds_market_value[k][valuation_date])[:,:valuation_date+1] * self.num_of_bonds[k][:valuation_date+1,:].T
            
            # produit élément par élément, donne un objet de dimension 20(maturité) * valuation_date(les différentes dates d'achat)
            MV_list_k = sum(MV_list_k_t.T)
            # une somme sur les différentes dates d'achat
            # ceci donne le BV équivalent
            MV_list.append(MV_list_k)
        return MV_list
        
    
    
    
    def valorize_bonds_income(self, valuation_date):
        """
            Method: valorize_bonds_income

            Function: valorize the bond portfolio at a valuation date

            Parameter:
                1. valuation_date
                    Type: int
                    Function: the valuation date (with reference to the time step)
            

        """
        
        income = {}
        
        
        market_value_list = self.get_market_value_list(valuation_date)
        market_value = np.sum(np.asarray(market_value_list))
        book_value_list = self.get_book_value_list(valuation_date)
        book_value = np.sum(np.asarray(book_value_list))
        
        income['PMVL'] = market_value - book_value
        PMVL = income['PMVL']
        if income['PMVL'] >= 0:
            income['PVL'] = PMVL
            income['PVL_obligation_TF'] = PMVL
            income['MVL'] = 0.0
            income['MVL_obligation_TF'] = 0.0
        else:
            income['PVL'] = 0.0
            income['PVL_obligation_TF'] = 0.0
            income['MVL'] = -PMVL
            income['MVL_obligation_TF'] = -PMVL
        
        #coupons = self.get_coupon_list(valuation_date)
        #income['Revenu'] = sum(sum(np.asarray(coupons)))
        income['Revenu'] =0
            
        income['PMVR_hors_obligation'] = 0.
        income['PMVR_obligation_TF'] = 0.
        income['PVL_hors_obligation'] = 0.
        return income
                   
        
    def add_bonds(self, amount, valuation_date):
        """
            Method: add_bonds

            Function: to add bonds in the bond portfolio at a valuation date

            Parameter:
                1. valuation_date
                    Type: int
                    Function: the valuation date (with reference to the time step)
                
                2. amount
                    Type: float
                    Function: the amount of money to invest in the bonds
                    

        """
        rating=0
        TtM=10
        num_bond_10ans_AAA = self.get_num_of_bond( amount, valuation_date, rating, TtM)
        new_bond_vector = np.zeros(20)
        new_bond_vector[TtM-1] = num_bond_10ans_AAA
        
        if len(self.allocation_matrix) == valuation_date:
            for k in range(7):
                list_temps =[]
                for t in range(valuation_date):
                    list_temps.append(self.num_of_bonds[k][t,:])
                if k == rating:
                    list_temps.append(new_bond_vector)
                else:
                    list_temps.append(np.zeros(20))
                self.num_of_bonds[k] = np.asarray(list_temps)

            self.allocation_matrix.append(self.allocation_matrix[-1])
            self.allocation_matrix[-1][rating, TtM-1] += num_bond_10ans_AAA       
        
        elif len(self.allocation_matrix) == valuation_date+1:
            self.num_of_bonds[rating][valuation_date, TtM-1] += num_bond_10ans_AAA
            self.allocation_matrix[valuation_date][rating, TtM-1] += num_bond_10ans_AAA
        else: 
            raise ValueError('Dimension error in add_bonds')
        

    
    def delete_bonds(self, amount, valuation_date, book_value = True):
        """
            Method: delete_bonds

            Function: to delete (sell) bonds in the bond portfolio at a valuation date

            Parameter:
                1. valuation_date
                    Type: int
                    Function: the valuation date (with reference to the time step)
                
                2. amount
                    Type: float
                    Function: the amount of value to disinvest in the bonds
                
                3. book_value
                    Type: boolean
                    Function: if True, the amount corresponds to the book value that we want to delete; otherwise, it is in market value
                    

        """
        PMVR = 0
        
        BV_list = self.get_book_value_list(valuation_date)
        BV_matrix = np.asarray(BV_list)
        MV_list = self.get_market_value_list(valuation_date)
        MV_matrix = np.asarray(MV_list)
        
        PMVL_matrix = MV_matrix-BV_matrix
        
        if book_value:
            
            # on commence par les pires notés et les maturités courts
            for k in range(6,-1,-1):
                # la boucle à l'inverse
                for TtM in range(1,21):
                    if BV_matrix[k,TtM-1] > 0:
                        if BV_matrix[k,TtM-1] < amount :
                            self.num_of_bonds[k][:,TtM-1] = np.zeros(len(self.num_of_bonds[k][:,TtM-1]))
                            self.allocation_matrix[valuation_date][k, TtM-1]=0
                            PMVR += PMVL_matrix[k, TtM-1]
                            amount -= BV_matrix[k,TtM-1]
                    
                        else:
                            self.num_of_bonds[k][:,TtM-1] = (1-amount/BV_matrix[k,TtM-1])*self.num_of_bonds[k][:,TtM-1]
                            self.allocation_matrix[valuation_date][k,TtM-1] = (1-amount/BV_matrix[k,TtM-1])* self.allocation_matrix[valuation_date][k,TtM-1]
                            PMVR += PMVL_matrix[k,TtM-1] * amount/BV_matrix[k,TtM-1]
                            amount = 0
            
            
        else:
            
            # on commence par les pires notés et les maturités courts
            for k in range(6,-1,-1):
                # la boucle à l'inverse
                for TtM in range(1,21):
                    if MV_matrix[k,TtM-1] > 0:
                        if MV_matrix[k,TtM-1] < amount :
                            self.num_of_bonds[k][:,TtM-1] = np.zeros(len(self.num_of_bonds[k][:,TtM-1]))
                            self.allocation_matrix[valuation_date][k, TtM-1]=0
                            PMVR += PMVL_matrix[k, TtM-1]
                            amount -= MV_matrix[k,TtM-1]
                    
                        else:
                            self.num_of_bonds[k][:,TtM-1] = (1-amount/MV_matrix[k,TtM-1])*self.num_of_bonds[k][:,TtM-1]
                            self.allocation_matrix[valuation_date][k,TtM-1] = (1-amount/MV_matrix[k,TtM-1])* self.allocation_matrix[valuation_date][k,TtM-1]
                            PMVR += PMVL_matrix[k, TtM-1] * amount/MV_matrix[k,TtM-1]
                            amount = 0
                    
        if round(amount,5) !=0:
                raise ValueError('There is not enough bonds to delete!')        
        return PMVR            

    def execute_unrealised_bonds_gain(self, amount, valuation_date):
        """
            Method: execute_unrealised_bonds_gain

            Function: to execute the unrealised bonds gain of a certain amount in the bond portfolio at a valuation date

            Parameter:
                1. valuation_date
                    Type: int
                    Function: the valuation date (with reference to the time step)
                
                2. amount
                    Type: float
                    Function: the amount of unrealiesd bonds gain to execute
                
               
        """        

        MV_list = self.get_market_value_list(valuation_date)
        BV_list = self.get_book_value_list(valuation_date)
                
        MV = np.asarray(MV_list)
        BV = np.asarray(BV_list)
        
        PMVL_matrix = MV-BV
        
        # si on réalise les plus values, le nombre des obligations baisse, donc on commence la boucle par les "pire"s, comme ce qu'on a fait pour delete_bond
        for k in range(6,-1,-1):
            for TtM in range(1,21):
                
                if PMVL_matrix[k, TtM-1] >0:
                    MV_k_T_today = np.asarray(self.rating_based_deflators[k])[valuation_date,TtM-1] + self.IR_curves[valuation_date,TtM-1] * sum(np.asarray(self.rating_based_deflators[k])[valuation_date,:TtM])
                    if MV_k_T_today <=0:
                        raise ValueError("Le prix ne peut pas être nul ou négatif!")
            
                    if PMVL_matrix[k, TtM-1] < amount:
                        
                        self.num_of_bonds[k][:valuation_date, TtM-1] = np.zeros(len(self.num_of_bonds[k][:valuation_date, TtM-1])) 
                        # 1. on vend 
                        self.num_of_bonds[k][valuation_date,TtM-1] = BV[k, TtM-1] / MV_k_T_today 
                        # 2. on achète une quantité de cette type d'obligation sur le marché telle que BV ne change pas
                        self.allocation_matrix[valuation_date][k, TtM-1]= BV[k, TtM-1] / MV_k_T_today 
                        
                        amount -= PMVL_matrix[k, TtM-1]
                        
                    else:
                        self.num_of_bonds[k][:valuation_date, TtM-1] = (1- amount/PMVL_matrix[k, TtM-1]) * self.num_of_bonds[k][:valuation_date, TtM-1] 
                        self.num_of_bonds[k][valuation_date, TtM-1] += (amount/PMVL_matrix[k,TtM-1])*BV[k, TtM-1] / MV_k_T_today
                        self.allocation_matrix[valuation_date][k, TtM-1] = sum(self.num_of_bonds[k][:,TtM-1])
                        
                        amount = 0
                        
        

    def execute_unrealised_bonds_loss(self, amount, valuation_date):

        """
            Method: execute_unrealised_bonds_loss

            Function: to execute the unrealised bonds loss of a certain amount in the bond portfolio at a valuation date

            Parameter:
                1. valuation_date
                    Type: int
                    Function: the valuation date (with reference to the time step)
                
                2. amount
                    Type: float
                    Function: the amount of unrealiesd bonds loss to execute
                
               
        """        

        MV = np.asarray(self.get_market_value_list(valuation_date))
        BV = np.asarray(self.get_book_value_list(valuation_date))  
        
        Book_value_bf = np.sum(BV)
        PMVL_matrix = MV-BV
        # si on réalise les moins values, le nombre des obligations augmente, on fait l'inverse d'execute_unrealised_bonds_gain
        for k in range(7):
            for TtM in range(20,0,-1):               
                if PMVL_matrix[k, TtM-1] < 0:
                    MV_k_T_today = (np.asarray(self.rating_based_deflators[k])[valuation_date,TtM-1] 
                                    + self.IR_curves[valuation_date,TtM-1]*sum(np.asarray(self.rating_based_deflators[k])[valuation_date,:TtM]))

                    assert MV_k_T_today >0, "Le prix ne peut pas être nul ou négatif!"
                    if abs(PMVL_matrix[k, TtM-1]) < amount:                       
                        self.num_of_bonds[k][:valuation_date, TtM-1] = np.zeros(len(self.num_of_bonds[k][:valuation_date, TtM-1])) 
                        # 1. on vend 
                        self.num_of_bonds[k][valuation_date,TtM-1] = BV[k, TtM-1] / MV_k_T_today 
                        # 2. on achète une quantité de cette type d'obligation sur le marché telle que BV ne change pas
                        self.allocation_matrix[valuation_date][k, TtM-1]= BV[k, TtM-1] / MV_k_T_today 
                        amount -= abs(PMVL_matrix[k, TtM-1])
                        
                    else:
                        self.num_of_bonds[k][:valuation_date, TtM-1] = (1- amount/abs(PMVL_matrix[k, TtM-1])) * self.num_of_bonds[k][:valuation_date, TtM-1] 
                        # 1. on vend
                        self.num_of_bonds[k][valuation_date, TtM-1] += (amount/abs(PMVL_matrix[k, TtM-1]))*BV[k,TtM-1]/MV_k_T_today
                        # 2. on achète une quantité de cette type d'obligation sur le marché telle que BV ne change pas                        
                        self.allocation_matrix[valuation_date][k, TtM-1] = sum(self.num_of_bonds[k][:,TtM-1]) 
                        amount = 0
            BV_aft = np.asarray(self.get_book_value_list(valuation_date))                 
            Book_value_aft = np.sum(BV_aft)
            if round(Book_value_bf,0) != round(Book_value_aft,0):
                raise ValueError("Error in Bonds_Portofolio.execute_unrealised_bonds_loss function")
                        
        
        
#                
#
#if __name__ == '__main__':
#    Working_URL = r'C:\Users\FR011526\Documents\ALM_credit(working)\Feuille_de_calcul_ALM(Working).xlsm'
#    
#    data = Asset_dat0()
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
#    # ================================================================
#    # Each lines represents the rating,
#    # e.g. line 1 ==> AAA
#    #      line 2 ==> AA
#    # Each columns represents the maturity for 1 to 20
#    # ================================================================
#    target_allocation = np.asarray([[50,50,50,50,50,60,70,80,100,100,0,0,0,0,0,0,0,0,0,0],
#                                    [10,10,10,10,10,10,10,10,10,10,0,0,0,0,0,0,0,0,0,0],
#                                    [6,5,4,3,2,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#                                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#                                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#                                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#                                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
#
#    bonds_portfolio = Bonds_Portfolio(time_horizon = time_horizon, ESG_RN_scenarios_traj_i = ESG.scenarios[0], target_allocation= target_allocation)
#    bonds_portfolio.initialize_allocation(amount = 1000)
#    bonds_portfolio.initialize_unit_bonds_book_value()
#    bonds_portfolio.initialize_unit_bonds_market_value()
#    
#    print('=====================================================================')
#    print('=================   EXEMPLE OF BONDS_PORTFOLIO   ====================')
#    print('=====================================================================')
#    print('Year | Num of bonds |  Book Vakue | Market Value |  Coupons  |  PMVL | PMVR |')
#    print('-----------------------------------------------------------------------------')
#    pmvr = 0
#    mv_0 = np.sum(bonds_portfolio.get_market_value_list(0))
#    bv_0 = np.sum(bonds_portfolio.get_book_value_list(0))
#    pmvl = mv_0 - bv_0
#    print('{0:.2f}    | {1:.2f}  |  {2:.2f}  |   {3:.2f}   | {4:.2f}  | {5:.2f}  | {6:.2f} |'.format(0, np.sum(np.asarray(bonds_portfolio.num_of_bonds)), np.sum(bonds_portfolio.get_book_value_list(0)),np.sum(bonds_portfolio.get_market_value_list(0)), 
#              np.sum(np.asarray(bonds_portfolio.get_coupon_list(0))) , pmvl, pmvr))
#
#    for time_step in range(1,time_horizon):
#        pmvr =0
#        # on ajoute une dimension au num_of_bonds et allocation_matrix si on appelle cette méthode en début de la période
#        if len(bonds_portfolio.allocation_matrix) == time_step:
#            bonds_portfolio.allocation_matrix.append(bonds_portfolio.allocation_matrix[-1])
#            
#            new_num=[]
#            for k in range(7):
#                num_k = []
#                for t in range(time_step):
#                    num_k.append(bonds_portfolio.num_of_bonds[k][t,:])
#                num_k.append(np.zeros(20))    
#                new_num.append(np.asarray(num_k))
#            bonds_portfolio.num_of_bonds = new_num
#            
#        if len(bonds_portfolio.allocation_matrix) != time_step+1 or len(bonds_portfolio.num_of_bonds[0]) != time_step+1:
#            raise ValueError('Bonds_Portofolio object has a wrong dimension in number of bonds')
#                        
#       
#        # on modifie la TtM (Time to maturity) dans num_of_bonds et allocation_matrix
#        for k in range(7):
#            for TtM in range(2,21):
#                bonds_portfolio.num_of_bonds[k][:,TtM-2] = bonds_portfolio.num_of_bonds[k][:,TtM-1]
#                bonds_portfolio.allocation_matrix[time_step][k, TtM-2] = bonds_portfolio.allocation_matrix[time_step][k, TtM-1]
#            bonds_portfolio.num_of_bonds[k][:,19] = np.zeros(len(bonds_portfolio.num_of_bonds[k]))
#            bonds_portfolio.allocation_matrix[time_step][k,19] = 0
#        
#        income_list =   bonds_portfolio.valorize_bonds_income(valuation_date = time_step)
#        
#        coupon_list = bonds_portfolio.get_coupon_list(valuation_date = time_step)
#        
#        # ===============================================================================
#        # on vérifie que le book value augmente et diminue du montant passant en argument
#        # ===============================================================================
#        if time_step <time_horizon-39:
#            print('{0:.2f}    | {1:.2f}  |  {2:.2f}  |   {3:.2f}   | {4:.2f}  | {5:.2f}  | {6:.2f} |'.format(time_step,
#                  np.sum(np.asarray(bonds_portfolio.num_of_bonds)),
#                  np.sum(bonds_portfolio.get_book_value_list(time_step)),
#                  np.sum(bonds_portfolio.get_market_value_list(time_step)), 
#                  np.sum(np.asarray(bonds_portfolio.get_coupon_list(time_step))),
#                  income_list['PMVL'], pmvr))  
#                  
#        if time_step ==2:
#            bonds_portfolio.add_bonds(amount=200,valuation_date=time_step)
#            income_list =   bonds_portfolio.valorize_bonds_income(valuation_date = time_step)
#            print('Execute unrealised loss for an amount of 2.')
#            print('----------------------------------------------')  
#            print('{0:.2f}    | {1:.2f}  |  {2:.2f}  |   {3:.2f}   | {4:.2f}  | {5:.2f}  | {6:.2f} |'.format(time_step,
#                  np.sum(np.asarray(bonds_portfolio.num_of_bonds)),
#                  np.sum(bonds_portfolio.get_book_value_list(time_step)),
#                  np.sum(bonds_portfolio.get_market_value_list(time_step)), 
#                  np.sum(np.asarray(bonds_portfolio.get_coupon_list(time_step))),
#                  income_list['PMVL'], 0.0))
#        
#        amount = 2
#        if time_step == 5:
#            bonds_portfolio.execute_unrealised_bonds_loss(amount=amount,valuation_date=time_step)
#            income_list =   bonds_portfolio.valorize_bonds_income(valuation_date = time_step)
#            print('Execute unrealised loss for an amount of 2.')
#            print('----------------------------------------------')  
#            print('{0:.2f}    | {1:.2f}  |  {2:.2f}  |   {3:.2f}   | {4:.2f}  | {5:.2f}  | {6:.2f} |'.format(time_step,
#                  np.sum(np.asarray(bonds_portfolio.num_of_bonds)),
#                  np.sum(bonds_portfolio.get_book_value_list(time_step)),
#                  np.sum(bonds_portfolio.get_market_value_list(time_step)), 
#                  np.sum(np.asarray(bonds_portfolio.get_coupon_list(time_step))),
#                  income_list['PMVL'], -amount))
#