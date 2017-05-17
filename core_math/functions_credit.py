# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 14:30:31 2016

@author: Jun JING
"""
## Python packages
import numpy as np


def fact(n):
    """fact(n): compute the factorial of n (int >= 0)"""
    if n<2:
        return 1
    else:
        return n*fact(n-1)

def exp_matrix(mat):
    """exp_matrix(): compute the exponential of a matrix"""
    prod=np.identity(len(mat))
    som=np.identity(len(mat))
    
    for k in range(50):
        prod=np.dot(prod,mat)
        som=som+prod/fact(k+1)
    
    return som

def diag_matrix(mat):
    """ diag_matrix() : Matrix Diagonalization """
    M=np.linalg.eig(mat)[1]
    D=np.diag(np.linalg.eig(mat)[0])
    M_inv=np.linalg.inv(M)
    
    return M,D,M_inv


def generator_matrix(mat):
    """generator_matrix() : compute the generator matrix from a transition matrix
    
    """
    prod=np.identity(len(mat))
    gen=np.identity(len(mat))-np.identity(len(mat))
    
    for k in range(50):
        prod=np.dot(prod,(mat-np.identity(len(mat))))

        gen=gen+prod/(k+1)*(-1)**(k+2)
    
    # vérifier les conditions pour être un bon generator matrix
    for k in range(len(gen)):
        for l in range(len(gen)):
            if l!=k and gen[k][l]<=0:
                gen[k][k]=gen[k][k]+gen[k][l]
                gen[k][l]=0
    
    
    return gen        
    
    
#
#if __name__ == '__main__':
#    
#    Id=np.array([
#    [1,0,0],
#    [0,1,0],
#    [0,0,1]
#    ])
#    
#    gamma=np.array([
#    [-0.1,0.1,0],
#    [0,0,0],
#    [0,0,0]
#    ])
#    
#    gamma2=np.array([
#    [0,0,0],
#    [1/11,-1/11,0],
#    [0,0,0]
#    ])
#    
#    gamma3=np.array([
#    [0,0,0],
#    [0,-0.1,0.1],
#    [0,0,0]
#    ])
#    
#    gamma_hat=np.array([
#    [-0.10084,0.10084,0],
#    [0.10909,-0.21818,0.10909],
#    [0,0,0]
#    ])
#    
#    P_hat= np.dot(np.dot(Id+gamma,Id+gamma2),Id+gamma3)
#    # cas non homogène
#    P_hat_hom=exp_matrix(gamma_hat)
#    # cas homogène
#    
#    # Transition matrix
#    P=np.array([
#    [0.9,0.08,0.017,0.003],
#    [0.05,0.85,0.09,0.01],
#    [0.01,0.09,0.8,0.1],
#    [0,0,0,1]
#    ])
#    
#    # recovery rate
#    phi=0.35
#    
#    A,B,C=diag_matrix(P)
#
#    
#    # zero coupon risky bond 
#    v=np.array([0.5,0.75,0.6])
#    # zero coupon risk free bond
#    B=np.array([0.8]*len(v))
#    
#    q_tilde_to_Default=(B-v)/(B*(1-phi))
#    
#    # q_tilde_to_Default=np.array([0.006,0.03,0.2])
#    
#    p_to_Default=P[range(len(P)-1),3] 
#    
#    # calcution of vector pi    
#    pi=q_tilde_to_Default/p_to_Default
#    
#    
#    
#    # now, we fill the matrix of risk neutral probabilities
#    Q_tilde=np.eye(len(P))
#    Q_tilde[range(len(P)-1),3]=q_tilde_to_Default
#    for i in range(len(P)):
#        for j in range(i+1,len(P)):
#            Q_tilde[i,j]=pi[i]*P[i,j]
#        Q_tilde[i,i]=1-np.sum(Q_tilde[i, range(i+1,len(P))])
#            
#    
#    with open('historical_transition_matrix.pkl', 'rb') as input:
#        historical_transition_matrix = pickle.load(input)
#        
#    generator= generator_matrix(historical_transition_matrix)
#    print("generator matrix is", generator)
#    et.dump_array(generator, filename = 'generator.csv')