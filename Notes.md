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