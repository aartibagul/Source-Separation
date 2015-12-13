from __future__ import division
import numpy as np
import random
import wave, struct, numpy as np, matplotlib.mlab as mlab, pylab as pl
import math
import matplotlib.pyplot as plt
import gc

filename = "CML_Recording_Both.wav"

# returns wave array of filename.wav
def get_wave(filename):
    w = wave.open(filename,"rb")
    waveParams = w.getparams()
    s = w.readframes(waveParams[3])
    w.close()
    waveArray = np.fromstring(s, np.int16)
    return waveArray

# power spectrogram is the absolute value SQUARED of the stft
def get_spectrogram(stft):
    return np.abs(stft)**2

# takes in wave file as input
# win_size is the length of the window in samples
# overlap is the amount of overlap between windows in samples
def my_stft(wave, win_size, overlap):
    # power sinbell analysis window
    win = sinebell(win_size, overlap)
    # make the frames
    frames, wave_pad = make_frames(wave, win, overlap)
    # fft, for each column
    stft = np.fft.fft(frames)
    # keep the spectrum associated with the positive frequencies (potentially times two the upper frequencies)
    if len(win)%2 == 0:
        stft = stft[:int(len(win)/2)+1]
    else:
        stft = stft[:int((len(win)+1)/2)]
    return stft, wave_pad
    
# create sinebell window of length win_size, with overlap, DONE
def sinebell(win_size, overlap):
    win = np.zeros(win_size)
    win[0:overlap] = np.sin( math.pi * (np.array([i for i in range(overlap)])) / (2*(overlap-1)) )
    win[overlap : win_size - overlap] = 1
    win[win_size - overlap:] = np.sin( math.pi * (win_size - np.array([i for i in range(win_size-overlap, win_size)]) - 1 ) / (2*(overlap - 1)) )
    return win

# computes sinebell window with smooth edges, this should be an improvement
def power_sinebell(win_size, overlap):
    win = np.zeros(win_size)
    win[0:overlap] = np.sin( math.pi * (np.array([i for i in range(overlap)])) / (2*(overlap-1)) )**2
    win[overlap : win_size - overlap] = 1
    win[win_size - overlap:] = np.sin( math.pi * (win_size - np.array([i for i in range(win_size-overlap, win_size)]) - 1 ) / (2*(overlap - 1)) )**2
    return win

# x is the input signal, win is the analysis window, overlap
# returns frame matrix and padded input signal x_pad
def make_frames(x, win, overlap):
    win_len = win.shape[0]
    x_len = len(x)
    
    # number of frames
    num_frames = int(np.ceil((x_len + overlap)/(win_len - overlap)))
    
    # initializing zero padded signal
    pad_len = int(overlap + num_frames * (win_len - overlap))
    x_pad = np.zeros(pad_len)
    x_pad[overlap: overlap + x_len] = x
    
    # index of beginning of each frame in x_pad
    frame_ind = np.array([i for i in range(num_frames)]) * (win_len - overlap)
    
    # initialize frames matrix
    frames = np.zeros((win_size, num_frames))
    for i in range(num_frames):
        frames[:,i] = (x_pad[frame_ind[i] : frame_ind[i] + win_size] * win)
    
    return frames, x_pad
    
def my_istft(stft, win_size, overlap):
    # power sinbell analysis window
    win = sinebell(win_size, overlap)
    (num_coeff, num_frames) = stft.shape
    
    # recover full stft by conjugate symmetry of the fourier expansion of real signals
    stft_full = np.zeros((win_size, num_frames), dtype = np.complex128)
    stft_full[:num_coeff,:] = stft
    
    # reasoning: stft[ num_coeff - 1] is the both the negative and positive Nyquist frequency if win_size is even
    # thus, we take the conjugate only of stft[ num_coeff - 2:0:-1 ]
    if win_size%2 == 0:
        stft_full[num_coeff:, :] = np.conj(stft[num_coeff-2:0:-1, :])
    else:
        stft_full[num_coeff:,:] = np.conj(stft[num_coeff-1:0:-1, :])
        
    # take inverse fft of recovered stft
    istft = np.fft.ifft(stft_full)
    # reconstruct padded signal by taking overlap into account
    x_pad = overlap_add(istft, win_size, overlap)
    return x_pad

def overlap_add(signal, win_size, overlap):
    
    win = sinebell(win_size, overlap)
    (temp, num_frames) = signal.shape

    if temp != win_size:
        print("Dimensions of ISTFT are wrong!")
    
    pad_len = overlap + num_frames * (win_size - overlap)
    x_pad = np.zeros(pad_len, dtype = np.complex128)
    
     # index of beginning of each frame in x_pad
    frame_ind = np.array([i for i in range(num_frames)]) * (win_size - overlap)
    
    # do we really need the window again here?
    x_pad[frame_ind[0]:frame_ind[0] + win_size] = signal[:,0] * win
    for i in range(1,num_frames):
        x_pad[frame_ind[i]:frame_ind[i] + win_size] = x_pad[frame_ind[i]:frame_ind[i] + win_size] + signal[:,i] * win
    return x_pad


# Objective functions section

# epsilon divergence
def compute_obj(v,W,h,eps):
    whv = (np.dot(W,h) + eps)/(v + eps)
    div = whv - np.log(whv) - 1 
    div = np.array(div)
    #print(div.shape)
    return np.sum( div )

# epsilon divergence gradient
def compute_grad(v,W,h,eps):
    grad = np.dot(W.T, (1/(v + eps) - 1/(np.dot(W,h) + eps)))
    return grad

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

win_size = 256
overlap = 128
# get wave array
waveArr = get_wave(filename)
# compute stft, padded input signal
stft, wave_pad = my_stft(waveArr, win_size, overlap)
# get power spectrogram of stft
spectrum = get_spectrogram(stft)

eps = 1e-12
(F,N) = spectrum.shape
K = 10
W = abs(np.random.randn(F,K) + np.ones((F,K)))
H = np.zeros((K, N))
#abs(np.random.randn(K, spectrum.shape[1]) + np.ones((K, spectrum.shape[1])))

A = np.zeros(W.shape)
B = np.zeros(W.shape)

r = 1
beta = 100
rho = r**(beta/spectrum.shape[1])

# for profiling:
#import cProfile
#cProfile.run('online_nmf(spectrum, W, H, A, B, rho, beta, 1e-6, eps)')

W, H, cost = online_nmf(spectrum, W, H, A, B, rho, beta, 1e-6, eps)

fig = plt.figure(1)
plt.plot([i for i in range(len(cost[5:]))], cost[5:])
plt.xlabel('iteration')
plt.ylabel('IS divergence')
fig.savefig("objective_function.png")

# according to Fevotte's Matlab code
V = np.dot(W,H)
Tpad = win_size + (N-1)*(win_size - overlap);

C = np.zeros((K,Tpad))

for i in range(K):
    ct = np.dot(W[:,i].reshape(F,1),H[i,:].reshape(1,N))/V * stft
    print(ct.shape)
    C[i,:] = np.real( my_istft(ct, win_size, overlap))

