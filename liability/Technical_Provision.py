class Technical_Provision(object):
    """
    Objective:
    ==========
    This class provides a general framework in order to estimate the technical provision at each time steps.

    Attributes
    ==========

    1. mathematical_provision:
        Type: array
        Function: corresponding list of mathematical provison values over time horizon

    2. liquidity_risk_provision:
        Type: array
        Function: corresponding list of liquidity risk provision values over time horizon

    3. profit_sharing_reserve:
        Type: array
        Function: corresponding list of profit sharing reserve values over time horizon

    4. financial_risk_provision:
        Type: array
        Function: corresponding list of financial risk provision values over time horizon

    Methods
    =======

    1. update :
        update parameters if they are given before or if we want to customize their initial value.
        By default, all the attributes are empty.

    We just call the update() method once right after the object is initialized.
    """
    def __init__(self):
        self.mathematical_provision = []
        self.liquidity_risk_provision = []
        self.profit_sharing_reserve = []
        self.financial_risk_provision = []
    def update(self, mathematical_provision = None, liquidity_risk_provision = None, profit_sharing_reserve = None, financial_risk_provision = None):
        """
            Method: update

            Function: update their initial values if they are given. By default, all these parameters are None

            Parameters:
                1. mathematical_provision:
                    Type: float (positive only)
                    Function: PM
                2. liquidity_risk_provision:
                    Type: float (positive only)
                    Function: PRE
                3. profit_sharing_reserve:
                    Type: float (positive only)
                    Function: PPE
                4. financial_risk_provision:
                    Type: float (positive only)
                    Function: PAF
        """
        if mathematical_provision is not None:
            self.mathematical_provision.append(mathematical_provision)
        if liquidity_risk_provision is not None:
            self.liquidity_risk_provision.append(liquidity_risk_provision)
        if profit_sharing_reserve is not None:
            self.profit_sharing_reserve.append(profit_sharing_reserve)
        if financial_risk_provision is not None:
            self.financial_risk_provision.append(financial_risk_provision)

