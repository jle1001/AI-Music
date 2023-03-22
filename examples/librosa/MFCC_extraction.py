import librosa
import matplotlib.pyplot as plt
import numpy as np

y, sr = librosa.load('examples/librosa/audio/audio_admiralbob77_-_Choice_-_Drum-bass.ogg')
mfccs = librosa.feature.mfcc(y=y, sr=sr)

# Plot the MFCC
# plt.figure(figsize=(10, 4))
# librosa.display.specshow(mfccs, x_axis='time')
# plt.colorbar()
# plt.title('MFCC')
# plt.tight_layout()
# plt.show()

mfcc = np.array(mfccs)  # convert list of MFCCs to a numpy array
np.save('./examples/librosa/audio/audio_admiralbob77_-_Choice_-_Drum-bass_MFCC.npy', mfcc)