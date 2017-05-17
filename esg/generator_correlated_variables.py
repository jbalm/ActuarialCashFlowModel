import numpy as np
from scipy.linalg import eigh, cholesky
from pylab import plot, show, axis, subplot, xlabel, ylabel, grid

#from pylab import plot, show, axis, subplot, xlabel, ylabel, grid

def generator_correlated_variables(corr_matrix, time_horizon, fixed_seed, method = 'cholesky'):
    np.random.seed(fixed_seed)
    ran = np.random.standard_normal((len(corr_matrix), time_horizon))
        
    if method == 'cholesky':
        c = cholesky(corr_matrix, lower = True)
    else:
        evals, evecs = eigh(corr_matrix)
        c = np.dot(evecs, np.diag(np.sqrt(evals)))
    dw = np.dot(c, ran)
    return dw

# ===========================
#       Test function
# ===========================
#if __name__ == '__main__':
#    # =============================
#    # Declare the correlated matrix
#    # =============================
#    #corr_matrix = np.array([
#    #   [ 3.40, -2.75, -2.00],
#    #   [ -2.75, 5.50, 1.50],
#    #   [ -2.00, 1.50, 1.25]
#    #   ])
#    corr_matrix = np.array([
#        [ 1., 0.2, 0.9],
#        [ 0.2, 1., 0.5],
#        [ 0.9, 0.5, 1.]
#        ])
#    # ==============================
#    # Declare time_horizon
#    # ==============================
#    time_horizon = 1000
#    # ==============================
#    # Declare fixed_seed
#    # ==============================
#    fixed_seed = 1000
#    # ==============================
#    # Declare num_instrument
#    # ==============================
#    dw = generator_correlated_variables(corr_matrix = corr_matrix, time_horizon = time_horizon, fixed_seed = fixed_seed)
#
#    #
#    # Plot les projections variées.
#    #
#    subplot(2,2,1)
#    plot(dw[0], dw[1], 'b.')
#    ylabel('dw[1]')
#    axis('equal')
#    grid(True)
#
#    subplot(2,2,3)
#    plot(dw[0], dw[2], 'b.')
#    xlabel('dw[0]')
#    ylabel('dw[2]')
#    axis('equal')
#    grid(True)
#
#    subplot(2,2,4)
#    plot(dw[1], dw[2], 'b.')
#    xlabel('dw[1]')
#    axis('equal')
#    grid(True)
#
#    show()
#
#
#
