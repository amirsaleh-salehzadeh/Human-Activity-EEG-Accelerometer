
'''
All signals were sampled at 256 samples per second

'''


import numpy as np
import pylab as pl
import pyedf
import copy


sampling_rate = 256
Time = 30 * 256
band = [0.5, 4, 8, 13, 30, 60]


gheader, sheader, signals = pyedf.loadSignals('1.edf')

sig = signals[0][: Time]

fft_sig = np.fft.rfft(sig) / Time

fft_sig_abs = np.clip(np.abs(fft_sig), 1e-20, 1e100)

freqs = np.fft.rfftfreq(Time, d = 1./ 100)

sig_back = []

print freqs.shape[0]

for index in range(len(band) - 1):
    freq = float(band[index])
    freqNext = float(band[index + 1])
    temp = list(freqs)
    fft_sig_temp = copy.deepcopy(fft_sig)

    for each in range(freqs.shape[0]):
        if temp[each] > freqNext or temp[each] < freq:
            fft_sig_temp[each] = 0

    print fft_sig_temp
    sig_back.append(np.fft.irfft(fft_sig_temp * Time))
    print freq, freqNext



pl.subplot(811)
pl.plot(sig)
pl.subplot(812)
pl.plot(freqs, fft_sig)
pl.subplot(813)
pl.plot(sig_back[0])
pl.subplot(814)
pl.plot(sig_back[1])
pl.subplot(815)
pl.plot(sig_back[2])
pl.subplot(816)
pl.plot(sig_back[3])
pl.subplot(817)
pl.plot(sig_back[4])
pl.subplot(818)
pl.plot(sig_back[5])


pl.show()
















