## Python packages
import numpy as np
import math

def f(x, *data):
    """
        data = (PM_init, PM_in, PM_out, richesse_max)
    """
    PM_init, PM_in, PM_out, richesse_max = data
    return PM_init*(1+x)+(PM_in - PM_out)*np.sqrt(1+x) - (PM_init + PM_in - PM_out) - richesse_max

def solve_quadratic_equation(a, b, c):
    x = ((-1)* b - math.sqrt(b**2-4*a*c))/2*a
    y = ((-1)* b + math.sqrt(b**2-4*a*c))/2*a
    return x, y