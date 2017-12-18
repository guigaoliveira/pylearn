''' from pylab import *
from scipy.io import wavfile
freq, data = wavfile.read('som-flauta-16-bits.wav')
x = np.array([1, 2, 3, 6])
data = data / (2.**15)
rms = np.mean(list(map(lambda x: x**2, data)))**.5
print(data, 1e-05)
data = list(filter(lambda x: !(i > 0.001 and x < 0.001), data))
timeArray = arange(0, len(data), 1)
timeArray = timeArray / freq
rmsArray = np.repeat(rms, len(data))
nrmsArray = np.repeat(-rms, len(data))
plt.plot(timeArray, data, color='k')
plt.plot(timeArray, rmsArray, color='blue')
plt.plot(timeArray, nrmsArray, color='blue')
ylabel('Amplitude')
xlabel('Tempo (s)')
show() '''
from pylab import *
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.io import wavfile  # get the api
freq, snd = wavfile.read('test.wav')  # load the data
snd = snd / (2.**15)  # this is 8-bit track, b is now normalized on [-1,1)
c = fft(snd)  # calculate fourier transform (complex numbers list)
d = int(len(c) /
        2) - 1  # you only need half of the fft list (real signal symmetry)
timeArray = arange(0, d, 1)
timeArray = timeArray / freq
#plt.plot(timeArray, imag(c[0:d]), 'g')
#plt.plot(timeArray, real(c[0:d]), 'b')
k = arange(d)
T = len(k) / freq 
print(k, T, k / T)
frqLabel = k / T
plt.plot(frqLabel, abs(c[0:d]), 'r')
plt.show()
''' from pylab import *
from scipy.io import wavfile

sampFreq, snd = wavfile.read('som_flauta-16-bits.wav')
snd = snd / (2.**15)
n = len(snd)
p = fft(snd)  # take the fourier transform
nUniquePts = int(ceil((n + 1) / 2.0))
p = p[0:nUniquePts]
p = abs(p)
p = p / float(n)
p = p**2  # square it to get the power

# multiply by two (see technical document for details)
# odd nfft excludes Nyquist point
print(p)
if n % 2 > 0:  # we've got odd number of points fft
    p[1:len(p)] = p[1:len(p)] * 2
else:
    p[1:len(p) - 1] = p[1:
                        len(p) - 1] * 2  # we've got even number of points fft

freqArray = arange(0, nUniquePts, 1.0) * (sampFreq / n)
plot(freqArray / 1000, 10 * log10(p), color='k')
xlabel('Frequency (kHz)')
ylabel('Power (dB)')
show() '''
''' timeArray = arange(0, snd.shape[0], 1)
timeArray = timeArray / sampFreq
plot(timeArray, snd, color='k')
ylabel('Amplitude')
xlabel('Time (ms)')
show() 
print(timeArray)
'''
''' from pydub import AudioSegment


def detect_leading_silence(sound, silence_threshold=-40.0, chunk_size=10):
    
    sound is a pydub.AudioSegment
    silence_threshold in dB
    chunk_size in ms

    iterate over chunks until you find the first one with sound
    
    trim_ms = 0  # ms
    while sound[trim_ms:trim_ms + chunk_size].dBFS < silence_threshold:
        trim_ms += chunk_size

    return trim_ms


sound = AudioSegment.from_file('som_flauta-16-bits.wav', format="wav")

start_trim = detect_leading_silence(sound)
end_trim = detect_leading_silence(sound.reverse())

duration = len(sound)
trimmed_sound = sound[start_trim:duration - end_trim]
trimmed_sound.export("test.wav", format="wav") '''
'''from pydub import AudioSegment

song = AudioSegment.from_wav("audio.wav")
new = song.low_pass_filter(5000)'''
''' from numpy import fft
import numpy as np
import matplotlib.pyplot as plt
n = 1000  # Number of data points
dx = 5.0  # Sampling period (in meters)
x = dx * np.arange(0, n)  # x coordinates
w1 = 100.0  # wavelength (meters)
w2 = 20.0  # wavelength (meters)
fx = np.sin(2 * np.pi * x / w1) + 2 * np.cos(2 * np.pi * x / w2)  # signal
Fk = fft.fft(fx) / n  # Fourier coefficients (divided by n)
nu = fft.fftfreq(n, dx)  # Natural frequencies
Fk = fft.fftshift(Fk)  # Shift zero freq to center
nu = fft.fftshift(nu)  # Shift zero freq to center
f, ax = plt.subplots(3, 1, sharex=True)
ax[0].plot(nu, np.real(Fk))  # Plot Cosine terms
ax[0].set_ylabel(r'$Re[F_k]$', size='x-large')
ax[1].plot(nu, np.imag(Fk))  # Plot Sine terms
ax[1].set_ylabel(r'$Im[F_k]$', size='x-large')
ax[2].plot(nu, np.absolute(Fk)**2)  # Plot spectral power
ax[2].set_ylabel(r'$\vert F_k \vert ^2$', size='x-large')
ax[2].set_xlabel(r'$\widetilde{\nu}$', size='x-large')
plt.show() '''