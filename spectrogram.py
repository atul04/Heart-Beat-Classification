import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import numpy as np
BASE_DIR = 'heartbeat-sounds/set_b/'

sample_rate, samples = wavfile.read("training/training-b/b0065.wav")
np.set_printoptions(threshold=np.inf)
# print(samples)
# upper = np.mean(samples) + np.std(samples)
# lower = np.mean(samples) - np.std(samples)
# samples = np.clip(samples, lower, upper)
# def butter_lowpass(cutoff, fs, order=5):
#     nyq = 0.5 * fs
#     normal_cutoff = cutoff / nyq
#     b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
#     return b, a
#
# def butter_lowpass_filter(data, cutoff, fs, order=5):
#     b, a = butter_lowpass(cutoff, fs, order=order)
#     y = signal.lfilter(b, a, data)
#     return y


# Filter requirements.
# order = 6
# fs = 30.0       # sample rate, Hz
# cutoff = 3.667  # desired cutoff frequency of the filter, Hz

# Get the filter coefficients so we can check its frequency response.
# b, a = butter_lowpass(cutoff, fs, order)
# y = butter_lowpass_filter(samples, cutoff, fs, order)
frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate, nperseg=128)
print(np.mean(frequencies))
# print(spectrogram)
# for i in range(len(frequencies)):
#     if frequencies[i] > np.mean(frequencies):
#         spectrogram[i] = 0
#         frequencies[i] = 0
#     spectrogram[i] *= 10000
print(np.shape(frequencies))
print(times.shape)
print(spectrogram.shape)
# print(spectrogram)
index = np.where(frequencies > 250)
spectrogram = np.delete(spectrogram, index, 0)
frequencies = np.delete(frequencies, index)
spectrogram = spectrogram * 100
plt.pcolormesh(times, frequencies, spectrogram)
plt.imshow(spectrogram)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
# plt.ylim(40, 0)
plt.show()
