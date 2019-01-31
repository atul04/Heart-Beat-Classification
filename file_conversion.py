# @Author: Atul Sahay <atul>
# @Date:   2018-11-22T14:28:53+05:30
# @Email:  atulsahay01@gmail.com
# @Last modified by:   nilanjan
# @Last modified time: 2018-11-22T19:55:30+05:30

import os
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import numpy as np

# Note : First Create a folder name spectrogram under the  directory where the python
#        file is kept


def spectrogram(filename, dLen):

    sample_rate, samples = wavfile.read(filename)
    np.set_printoptions(threshold=np.inf)
    # print(samples)
    upper = np.mean(samples) + np.std(samples)
    lower = np.mean(samples) - np.std(samples)
    samples = np.clip(samples, lower, upper)

    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

    for i in range(len(frequencies)):
        if frequencies[i] > np.mean(frequencies):
            frequencies[i] = 0

    fig = plt.figure(figsize=(3.60, 3.60), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    plt.axis('off')
    plt.pcolormesh(times, frequencies, spectrogram)
    plt.imshow(spectrogram)

    plt.ylim(30, 0)
    path = "validation/"+filename[dLen+1:-4]+".png"
    print(path)
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    plt.savefig(path, dpi=100, bbox_inches=extent, pad_inches=0)
    plt.close('all')

# Note: This directory contains all the .wav files


directory = 'validation_raw'
dLen = len(directory)

for filename in os.listdir(directory):
    # print("filename"+str(filename))
    if filename.endswith(".wav"):
        # print("dir: "+str(directory))
        path = directory+"/"+filename
        # print("path "+str(path))
        spectrogram(path, dLen)
    else:
        pass
