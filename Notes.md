# Data processing

Librosa - Used to compute spectograms from audio files
pretty_midi - reads MIDI files, and extracts notes, durations, and pitches

* We define a fixed time frame we want each spectogram to have
* All segments should be the same
* N_BINDS is number of frequency bins in the spectogram
* Spectograms will have shape: (N_BINS, FIXED_LENGTH) = (84, 128)

# Converting MIDI to Wav
* Use FluidSynth

# Spectogram (CQT)
* sr - sample rate (Hz)
* CQT - Constant-Q Transformation

# CNN Architecture
https://medium.com/@myringoleMLGOD/simple-convolutional-neural-network-cnn-for-dummies-in-pytorch-a-step-by-step-guide-6f4109f6df80
Conv2D Layer:
Filters - small windows to detect specific features in an image
Stride - Step size filter moves
Padding - extra pixels at the border
Feature map - output after applying filter

Pooling layer:
* Reduces spatial dimensions of the feature maps to retain the most prominent features
* used to handle overfitting

1. Max Pooling - takes max value of window
2. Average Pooling - takes average of window

## Questions

CQT representation as an array -> Each row is a semitone on the range, columns are time. Each cell is energy there