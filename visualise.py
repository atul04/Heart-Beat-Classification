import wave
import struct
import matplotlib.pyplot as plt

BASE_DIR = 'heartbeat-sounds/set_b/'
file = wave.open(BASE_DIR + "normal__168_1307970069434_A.wav")
frames = file.readframes(-1)

samples = struct.unpack('h' * file.getnframes(), frames)

print(samples[:10])

framerate = file.getframerate()
t = [float(i) / framerate for i in range(len(samples))]
print(t[:10])

plt.plot(t, samples)
plt.show()
