## Python packages
from math import log, sqrt, exp



def f(x,y,a):
    return exp(-a*(x-y))

def integrand(x, a, theta):
    return sum( 0.5 * (f(x,y+1,a) * theta[y+1] + f(x,y,a) * theta[y]) for y in range(x))

def double_integration(time_horizon, a, theta):
    return sum( 0.5 * (integrand(x+1, a, theta) + integrand(x, a, theta)) for x in range(time_horizon + 1))

def upper_bounds(sigma_s, sigma_r, time_horizon, a, theta, S0, K, r0, div_yield = 0, rho = 0):
    T = time_horizon
    d = div_yield
    dblintegra = double_integration(time_horizon = T, a = a, theta = theta)
    sigma_11 = T * sigma_s ** 2
    sigma_12 = (rho * sigma_s * sigma_r/a) * (T - (1 - exp(-a * T))/a)
    sigma_22 = (sigma_r**2)/(a**2) * (T - 2 * (1 - exp(-a*T))/a + 0.5 * (1 - exp(-2*a*T))/a)
    D = sigma_11 + 2*sigma_12 + sigma_22
    C = - log((S0 * exp(-d*T))/K) + 0.5 * (sigma_s**2) * T - r0 * (1 - exp(-a*T))/a - dblintegra
    d1 = (sigma_11 + sigma_12 - C)/(sqrt(D))
    d2 = d1 - sqrt(D)
    return d1, d2



#if __name__ == '__main__':
#    # Initialize and calibrate IR model
#    path = 'Market_Environment.xls'
#    data = Asset_data()
#    data.update(path)
#    Interest_rate = IR_model.Hull_White_one_factor(market_name = 'EUR')
#    Interest_rate.calibrate(data)
#    # Parameters
#    a = Interest_rate.a
#    time_horizon = 10
#    theta = Interest_rate.theta
#    S0 = 1.
#    K = 1.
#    r0 = 0.01
#    sigma_s = 0.2
#    sigma_r = 0.01
#    # Test double_integration function
#    resu = double_integration(time_horizon, a, theta)
#    print(resu)
#    # Test upper_bounds function
#    d1, d2 = upper_bounds(sigma_s = sigma_s, sigma_r = sigma_r, time_horizon = time_horizon, a = a, theta = theta, S0 = S0, K = K, r0 = r0)
#    print("d1 = ",d1)
#    print("d2 = ",d2)