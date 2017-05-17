class Model_Point(object):
    """
        Definition: Model Point:
        ========================
        The development of a portfolio of policies is modeled by using model points. 
        Each model point corresponds to an individual policyholder account or
        to a pool of similar policyholder aaccounts which can be used to reduce the computational complexity,
        in particular in the case of very large insurance portfolios.

        Objective:
        ==========
        This class provides a general framework for simulating a model point. Each model point objet will be integrated into the portfolio of 
        policies named "Liability_data0.model_points" (see class Liability_data0.py)

        Attributes:
        ===========

        id: string
            Model Point's identity.

        average_age: int
            Model Point's average age

        subscription_date: datetime (dd/mm/yyyy)
            Subscription Date.

        valuation_date: datetime (dd/mm/yyyy)
            Valuation Date.

        day_convention: int
            Day count convention assumes there are 30 days in a month and 360 days in a year.

        seniority: int
            Seniority or the number of years between the valuation date and the subscription date

        premium: float
            periodic premium

        mathematical_provision: array
            The model point's mathematical provision

        lapse_value: float (positive)
            contratual lapse value

        rate_sensibility: float (positive)

        mortality_rate: list/array-like
            corresponding list of mortality rate at differents age

        lapse_rate: list/array-like
            corresponding list of lapse rate at differents seniority

        TMG : float
            Taux minimum garanti

        desired_rate: float
            Expected revalorisation rate.

        Method
        ======

        1. __str__:

    """
    def __init__(self):
        self.id = None
        self.average_age = None
        self.sexe = None
        self.subscription_date = None
        self.valuation_date = None
        self.day_convention = 365.
        self.premium = None
        self.actual_math_provision = None
        self.mathematical_provision = []
        self.profit_sharing_rate = [0]
        self.TMG_type = None
        self.rate_sensibility = None
        self.margin_rate = None
        self.number_contract = None
        self.lapse_type = None
        self.TMG = []
        self.mortality_rate = []
        self.lapse_rate = []
        # Output
        self.cash_flow_in = []
        self.cash_flow_out = []
    
    def update(self, id, average_age, sexe, subscription_date, valuation_date, premium, actual_math_provision, TMG_type, rate_sensibility, margin_rate, number_contract, lapse_type):
        self.id = id
        self.average_age = average_age
        self.sexe = sexe
        self.subscription_date = subscription_date
        self.valuation_date = valuation_date
        self.premium = premium
        self.actual_math_provision = actual_math_provision
        self.TMG_type = TMG_type
        self.rate_sensibility = rate_sensibility
        self.margin_rate = margin_rate
        self.number_contract = number_contract
        self.lapse_type = lapse_type
        
    def get_seniority(self):
        self.seniority = int((self.valuation_date - self.subscription_date).days/self.day_convention)

    def __str__(self):
        """
            Method: __str__

            Function: Print Model Point's characteristics

            Parameters: None
        """
        return("Liability Data: \n"
                "\n"
                "    Model Point ID: {0} \n"
                "    ================ \n"
                "    Average Age: {1} \n"
                "    Sexe:  {2} \n"
                "    Subscription Date:  {3} \n"
                "    Valuation Date:  {4} \n"
                "    Premium:  {5} \n"
                "    Actual mathematical provision: {6} \n"
                "    TMG_type:  {7} \n"
                "    Rate Sensibility: {8} \n"
                "    Margin rate: {9} \n"
                "    Number of Contract:  {10} \n"
                "    Lapse_type:  {11} \n"
                .format(self.id, self.average_age, self.sexe, self.subscription_date,
                        self.valuation_date,self.premium, self.actual_math_provision,
                        self.TMG_type, self.rate_sensibility, self.margin_rate, self.number_contract, self.lapse_type))


