import IPython.display as ipd
import os

fulldatasetpath = "D:/Datasets/UrbanSound8K/"
os.chdir(fulldatasetpath)


import librosa 
from scipy.io import wavfile as wav
import numpy as np

filename = fulldatasetpath+'audio/fold9/106955-6-0-0.wav' 

librosa_audio, librosa_sample_rate = librosa.load(filename) 
scipy_sample_rate, scipy_audio = wav.read(filename) 

print('Original sample rate:', scipy_sample_rate) 
print('Librosa sample rate:', librosa_sample_rate)

print('Original audio file min~max range:', np.min(scipy_audio), 'to', np.max(scipy_audio))
print('Librosa audio file min~max range:', np.min(librosa_audio), 'to', np.max(librosa_audio))

import matplotlib.pyplot as plt

# Original audio with 2 channels 
#plt.figure(figsize=(12, 4))
#plt.plot(scipy_audio)

mfccs = librosa.feature.mfcc(y=librosa_audio, sr=librosa_sample_rate, n_mfcc=40)
print(mfccs.shape)
import librosa.display
plt.figure()
librosa.display.specshow(mfccs, sr=librosa_sample_rate, x_axis='time')
plt.colorbar()
#plt.show()

plt.savefig('D:/Pictures/istassessment/spectrogram/'+'gunshot_spectrogram'+'.png')
