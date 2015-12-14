from __future__ import division
import numpy as np
import random


def get_closest(x, centers):
    distances = []
    
    #get the distance from the point(x) to every centroid
    #and get the index of centroid that yields 
    #the minimum distance 
    
    for mu in centers:
        distances.append(np.linalg.norm(x-mu))
    index = np.argmin(distances)
    
    return index, min(distances)

def k_means_objective(X, k, C, centers):
    
    sum_distance = 0
    for j in range(k):
        for x in X[C==j]:
            sum_distance += (np.linalg.norm(x-centers[j]))**2
                
    return sum_distance

def mini_batch(X, k, b, max_iter):
    v = [0]*k
    centers = [0]*k
    
    centers[0] = X[random.randint(0, len(X)-1)]

    distortion = []
    
    for r in range(k-1):
        x = random.random()
        num_dx = (get_closest(X[0],centers)[1])**2
        den_dx = 0
        for x_i in X:
            den_dx += (get_closest(x_i,centers)[1])**2
        index = 0
        while num_dx/den_dx < x:
            
            index += 1
            num_dx += (get_closest(X[index],centers)[1])**2

        centers[r+1] = X[index]
    centers = np.array(centers)


    for i in range(max_iter):
        M = np.array(random.sample(list(X),b))
        C=[0]*len(M)
        for i,x in enumerate(M): 
            index = get_closest(x, centers)[0]
            C[i] = index
      
        
        for i, x in enumerate(M):
            index = C[i]
            v[index]+=1
            n = 1/v[index]
    
            centers[index] = (1- n)*centers[index] + n*x
            
    return centers
