from __future__ import division
import numpy as np
import random
import wave, struct, numpy as np, matplotlib.mlab as mlab, pylab as pl
import math
import matplotlib.pyplot as plt
import gc
from scipy.io import wavfile

# Objective functions section

def compute_obj(v,W,h,eps):
    vhw = (v + eps) / (np.dot(W,h) + eps)
    div = vhw - np.log(vhw) - 1
    return np.sum( div )

def compute_grad(v,W,h,eps):
    (F,K) = W.shape
    grad = np.dot( ( 1 / (np.dot(W,h + eps)) - (v + eps) / (np.square(np.dot(W,h))) + eps).T , W )
    return grad.T

# important! 
# Not only do we need h = h_t but also h_m = h_(t-1) and h_p = h_(t+1)
# lambda is the smoothness constant
def compute_smooth_obj(v,W,h,h_m,h_p,lamb,eps):
    
    h = h.reshape(h.shape[0],1)
    h_m = h_m.reshape(h_m.shape[0],1)
    h_p = h_p.reshape(h_p.shape[0],1)
    
    # compute regular objective
    # maybe doing this direct instead of the function call is faster:
    # whv = (np.dot(W,h) + eps)/(v + eps)
    # div = whv - np.log(whv) - 1 
    div = compute_obj(v,W,h,eps)
    
    # compute smoothness terms with epsilon
    s1 = (h + eps) / (h_m + eps)
    s2 = (h + eps) / (h_p + eps)
    sm = s1 - np.log(s1) - 1
    sm += s2 - np.log(s2) - 1
    # returning properly scaled smooth objective
    return div + lamb * np.sum( sm )
    
# input parameters as above
def compute_smooth_grad(v,W,h,h_m,h_p,lamb,eps):

    # the famous reshape trio
    h = h.reshape(h.shape[0],1)
    h_m = h_m.reshape(h_m.shape[0],1)
    h_p = h_p.reshape(h_p.shape[0],1)

    # calculates gradient of regular divergence
    div_grad = compute_grad(v,W,h,eps)

    # calculating gradient of smoothness term
    sm_grad = 1 / (h_m + eps) + 1 / (h_p + eps) - 2 / (h + eps)
    sm_grad = sm_grad.reshape(sm_grad.shape[0],1)
    
    # returning properly scaled gradient of smooth objective 
    return div_grad + lamb * sm_grad

# gradient descent function
def gradient_backtracking(v, W, h, max_iter, compute_grad, compute_obj, eps):
    
    v = v.reshape(v.shape[0],1)
    h = h.reshape(h.shape[0],1)

    beta = 0.2 #backstep factor between 0.1 and 0.8
    opt_prec = 1-1e-6 # optimization precision
    eta = 1e-1 #initial step size
    
    #obj = [None]*max_iter
    
    max_backstep = 20 # maximum number of backsteps
    t = 0 # backstepping counter
    k = 0 # gradient step counter 
    
    old_obj = compute_obj(v,W,h,eps)

    while( k < max_iter and t != max_backstep ):
        
        grad = compute_grad(v,W,h,eps)
        #obj[k] = compute_obj(v,W,h,eps)
        
        t = 0 # reset backstepping counter
        eta = 1/beta*eta # try to increase stepsize slightly again
        
        # make sure h-n*grad is positive
        while(any(h - eta * grad < 0)  and t < max_backstep ):
            t += 1
            eta = beta * eta
    
        new_obj = compute_obj(v,W,(h - eta*grad),eps)
        
        while( new_obj > opt_prec * old_obj and t < max_backstep):
            t += 1
            eta = beta * eta
            new_obj = abs(compute_obj(v,W,(h - eta*grad),eps))
                      
        h = h - eta * grad # update h according to gradient step
        k += 1 # update gradient step counter
        old_obj = new_obj
        
    h = h.reshape(h.shape[0],)
    return h

def online_nmf(spectrum, W, H,A, B, rho, beta, eta, eps):
           
    a = np.zeros(W.shape)
    b = np.zeros(W.shape)
    
    t = 1
    W_old = W + 1.5*eta
    k = W.shape[1]
    h = np.random.rand(W.shape[1],)
    n = spectrum.shape[1]
    cost = []
    cost.append(compute_obj(spectrum,W,H,eps))
    
    while np.linalg.norm(W - W_old, ord = "fro") > eta:
        
        t = t+1 
        
        ind = random.randint(0, n-1)
        v = spectrum[:,ind]
    
        h = gradient_backtracking(v, W, H[:,ind], 100, compute_grad, compute_obj, eps)
        
        H[:, ind] = h
       
        h = h.reshape(h.shape[0],1)
        v = v.reshape(v.shape[0],1)
        den = eps + np.dot(W, h)
        
        a += np.dot(((eps+v)/(den)**2), h.T) * np.square(W) 
        
        b += np.dot(1/den, h.T)
       
        if t % beta == 0:
            A = A + rho*a
            a = 0
            B = B + rho*b
            b = 0
            W_old = W
            W = np.sqrt(A/B)
            
            for i in range(k):
                s = np.sum(W[:,i])
                W[:,i] = W[:,i]/s
                A[:,i] = A[:,i]/s
                B[:,i] = B[:,i]*s
                #print(i)

            #print(np.linalg.norm(compute_obj(spectrum,W_old,H,eps))- compute_obj(spectrum,W,H,eps)) 
            gc.disable()
            cost.append(compute_obj(spectrum,W,H,eps))
            gc.enable()
            
        #cost.append(compute_obj(spectrum,W,H,eps))
        if t > 100*n:
            print(" W shape" , W.shape)
            break

        #print("W", np.linalg.norm(W[:,1]))
        #print("H", np.linalg.norm(H[1]))
        #print(compute_obj(spectrum,W,H.T,eps))
        
    print("t" , t)
    print(cost[-1])
    return W, H, cost