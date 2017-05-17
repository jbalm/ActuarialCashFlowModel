## Python packages
from abc import ABCMeta, abstractmethod, abstractproperty
    
class IR_model:
    """This class is the purely abstract class for all IR models we will implement in the future
    see https://pymotw.com/2/abc/"""

    __metaclass__=ABCMeta

    @abstractmethod
    def get_IR_curve(self):
        """This method returns IR term structures per time steps and trajectories"""
        return

    @abstractmethod
    def get_deflator(self):
        """This method returns deflators per time steps and trajectories"""
        return

    @abstractmethod
    def calibrate(self, asset_data):
        """Calibrate the IR model onto the market environment
            This method returns the IR_curve and deflator"""
        return





#if __name__ == '__main__':
#    Working_URL = r'C:\Users\FR011526\Documents\ALM_credit(working)\Feuille_de_calcul_ALM(Working).xlsm'
#    
#    data = Asset_data()
#    data.update(Working_URL)
#    market = data.get_list_market('EUR')
#    # ===========================
#    # test Hull_White_one_factor
#    # ===========================
#    time_horizon = int(xw.sheets['ESG'].range('D3').value)
#    corr_matrix = market.corr_matrix
#    
#    Interest_rate = Hull_White_one_factor(market_name = market.name)
#    Interest_rate.time_horizon = time_horizon
#    Interest_rate.corr_matrix = corr_matrix
#    # ==============================
#    # Declare num_instrument
#    # ==============================
#    Interest_rate.calibrate(asset_data = data)
#    #Interest_rate.get_IR_curve()
#    #Interest_rate.get_deflator()
#    #plt.plot(range(Interest_rate.time_horizon), Interest_rate.curves[0][:Interest_rate.time_horizon])
#    #plt.title("Hull & White zero coupon cuvre at t = 0")
#    num_traj = 1
#    fixed_seed = [2016 + i for i in range(num_traj)]
#    for traj_i in range(num_traj):
#        Interest_rate.fixed_seed = fixed_seed[traj_i]
#        Interest_rate.get_IR_curve()
#        Interest_rate.get_deflator()
#        plt.plot(range(Interest_rate.time_horizon), Interest_rate.spot_rate[:Interest_rate.time_horizon])
#
#    plt.xlabel('Maturity (in years)')
#    plt.ylabel('Rate')
#    plt.title('Evolution des taux courts at t = 0')
#    plt.show()