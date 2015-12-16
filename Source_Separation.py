from __future__ import division
import numpy as np
import random
import wave, struct, numpy as np, matplotlib.mlab as mlab, pylab as pl
import math
import matplotlib.pyplot as plt
import gc
from scipy.io import wavfile
from mini_batch import mini_batch
import istft 
import nmf
import stft
import sys
#"CML_Recording_Both.wav"
#"data.wav"
filename = "data.wav"
w = wave.open(filename,"rb")

# returns bit array of .wav file, and framerate
def get_wave(filename):
    w = wave.open(filename,"rb")
    waveParams = w.getparams()
    s = w.readframes(waveParams[3])
    w.close()
    waveArray = np.fromstring(s, np.int16)
    return waveArray, waveParams[2]

# returns wave array of one track of stereo .wav file
# activate line 2 to convert from stereo file
def get_wave_stereo(filename):
    rate, data = wavfile.read(filename)
    #data = data[:,0]
    return data, rate

# power spectrogram is the absolute value SQUARED of the stft
def get_spectrogram(stft):
    stft[0,:] = stft[0,:]/2
    return 4*np.square(np.absolute(stft))


win_size = 256
overlap = 128
# get wave array
waveArr, framerate = get_wave_stereo(filename)
# normalize wave_arr (int16)
waveArr = waveArr.astype(np.float64)
#waveArr = waveArr/32768

# compute stft, padded input signal
stft, wave_pad = stft.my_stft(waveArr, win_size, overlap)
# get power spectrogram of stft
spectrum = get_spectrogram(stft)

eps = 0
(F,N) = spectrum.shape
K = 10
W = abs(np.random.randn(F,K)) + np.ones((F,K))
H = abs(np.random.randn(K,N)) + np.ones((K, N))
#abs(np.random.randn(K, spectrum.shape[1]) + np.ones((K, spectrum.shape[1])))

A = np.zeros(W.shape)
B = np.zeros(W.shape)

r = 1
beta = 1000
rho = r**(beta/spectrum.shape[1])

# for profiling:
#import cProfile
#cProfile.run('online_nmf(spectrum, W, H, A, B, rho, beta, 1e-6, eps)')

W, H, cost = nmf.online_nmf(spectrum, W, H, A, B, rho, beta, 1e-4, eps)

'''
Initialization with mini batch


centers = mini_batch(spectrum.T, K, 100, 100)
centers = np.array(centers)
W2 = centers.T

H2 = abs(np.random.randn(K,N)) + np.ones((K, N))
A2 = np.zeros(W.shape)
B2 = np.zeros(W.shape)

W2, H2, cost2 = online_nmf(spectrum, W2, H2, A2, B2, rho, beta, 1e-4, eps)


fig = plt.figure(1)
plt.plot([i for i in range(len(cost))], cost, label = "Without k-means")
plt.plot([i for i in range(len(cost2))], cost2, label = "With k-means")
plt.legend(loc="upper right")
plt.xlabel('iteration')
plt.ylabel('IS divergence')
fig.savefig("objective_function.png")

'''
fig = plt.figure(1)
plt.plot([i for i in range(len(cost))], cost, label = "Without k-means")
plt.legend(loc="upper right")
plt.xlabel('iteration')
plt.ylabel('IS divergence')
fig.savefig("objective_function.png")
# according to Fevotte's Matlab code (Wiener Filtering + ISTFT)
V = np.dot(W,H)


print('spec norm: ', np.linalg.norm(spectrum))
print('V norm: ', np.linalg.norm(V))
print(np.linalg.norm(V-spectrum))
print('V dtype ', V.dtype)
print('spec dtype ', spectrum.dtype)


Tpad = win_size + (N-1)*(win_size - overlap);
C = np.zeros((K,Tpad))

for i in range(K):
    
    ct = np.zeros(V.shape)
    ct = (np.dot(W[:,i].reshape(F,1),H[i,:].reshape(1,N))/V) * stft
    ct[np.isnan(ct)] = 0
    s1 = np.real( istft.my_istft(ct, win_size, overlap))

    

    # scale it back to int16
    #s = s * 32768
    #print("Before:", max(s1))
    #s1 = s1.astype(np.int16)
    #print("After:", max(s1))
    #for j in range(len(s1)):
        #s1[j] += random.randint(-10000,10000)
    C[i,:] = s1
    #s1 = s1.tolist()
    #print(s1.shape)
    noise_output = wave.open('out-{}.wav'.format(i), 'w')
    w_copy = list(w.getparams())
    w_copy[0] = 1
    noise_output.setparams(tuple(w_copy))
    value_str = s1.tostring()
    
    noise_output.writeframes(value_str)

    noise_output.close()

    #wavfile.write('out-{}.wav'.format(i), framerate, s1)

