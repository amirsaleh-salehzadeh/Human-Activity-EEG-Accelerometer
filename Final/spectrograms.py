import numpy as np  # fast vectors and matrices
import matplotlib.pyplot as plt  # plotting
from scipy import fft  
import csv  # fast fourier transform
import pandas as pd
from IPython.display import Audio

from intervaltree import Interval, IntervalTree

fs = 44100

# train_data = np.loadtxt(open('C:\RecordingFiles\Watching\data2.csv','rb'), dtype = 'float')
# train_data = pd.read_csv(filepath_or_buffer="C:\RecordingFiles\Watching\data2.csv", sep=',', usecols=[0, 2, 3],
#                               dtype='float', index_col="T", names=["T", "EEG1", "EEG2"],
#                               parse_dates=True)
# train_data = np.array(train_data).astype("float")
train_data = np.array(list(csv.reader(open("C:\RecordingFiles\Watching\data2.csv", "rb"), delimiter=","))).astype("float")
X, Y = train_data['1024']

window_size = 128  # 2048-sample fourier windows
stride = 32  # 512 samples between windows
wps = fs / float(512)  # ~86 windows/second
Xs = np.empty([int(10 * wps), 2048])

for i in range(Xs.shape[0]):
    Xs[i] = np.abs(fft(X[i * stride:i * stride + window_size]))
    
second = 3

fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, sharey=True)
fig.set_figwidth(20)
ax1.plot(Xs[int(second * wps)], color=(41 / 255., 104 / 255., 168 / 255.))
ax1.set_xlim([0, window_size])
ax1.set_ylabel('amplitude')
ax2.plot(Xs[int(second * wps), 0:window_size / 2], color=(41 / 255., 104 / 255., 168 / 255.))
ax2.set_xlim([0, window_size / 2])
ax3.plot(Xs[int(second * wps), 0:150], color=(41 / 255., 104 / 255., 168 / 255.))
ax3.set_xlim([0, 150])

fig = plt.figure()
fig.set_figwidth(20)
fig.set_figheight(2)
plt.plot(X[0:10 * fs], color=(41 / 255., 104 / 255., 168 / 255.))
fig.axes[0].set_xlabel('sample (44,100Hz)')
fig.axes[0].set_ylabel('amplitude')

fig = plt.figure(figsize=(20, 7))
plt.imshow(Xs.T[0:150], aspect='auto')
plt.gca().invert_yaxis()
fig.axes[0].set_xlabel('windows (~86Hz)')
fig.axes[0].set_ylabel('frequency')
