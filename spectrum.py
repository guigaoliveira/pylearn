from pylab import *
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.io import wavfile  # get the api

freq, snd = wavfile.read('sounds/som-flauta-16-bits.wav')  # load the data
Fs = freq 
n = len(snd) # length of the signal
k = np.arange(n)
T = n/Fs
frq = k/T # two sides frequency range
frq = frq[0:int(n/2)] # one side frequency range
fft = np.fft.rfft(snd)
Y = fft/n # fft computing and normalization
Y = Y[0:int(n/2)]

inverse = np.fft.irfft(fft)

wavfile.write('sounds/inverse_flauta.wav', freq, inverse.real.astype('int16'))

plt.plot(frq, abs(Y), 'r')
plt.show()
