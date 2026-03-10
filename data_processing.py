import librosa
import pretty_midi
import os
import numpy as np
import pandas as pd

FIXED_LENGTH = 128
N_BINS = 84

def midi_to_wav(midi_file, output_dir, soundfont_path):
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, os.path.basename(midi_file).replace('.mid', '.wav'))
    os.system(f'fluidsynth -ni -F "{output_file}" "{soundfont_path}" "{midi_file}"')

    if os.path.exists(output_file):
        print(f"Successfully created: {output_file}")
    else:
        print(f"FAILED to create: {output_file}")

    return output_file

def create_cqt(y, sr, hop_length=512):
    cqt = librosa.cqt(y, sr=sr, hop_length=hop_length, n_bins=N_BINS, bins_per_octave=12)
    cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
    return cqt_db

def pad_or_truncate(cqt, fixed_length=FIXED_LENGTH):
    if cqt.shape[1] >= fixed_length:
        return cqt[:, :fixed_length]
    pad_width = fixed_length - cqt.shape[1]
    return np.pad(cqt, ((0, 0), (0, pad_width)))

def main():
    midi_dir = "/Users/melvincheng/Documents/Spring 2026/CS4100/CS4100_Final_Project_Group20/Midi Files"
    output_dir = "/Users/melvincheng/Documents/Spring 2026/CS4100/CS4100_Final_Project_Group20/Wav_Files"
    soundfont_path = "/Users/melvincheng/Documents/Spring 2026/CS4100/CS4100_Final_Project_Group20/GeneralUser-GS/GeneralUser-GS.sf2"
    save_dir = "/Users/melvincheng/Documents/Spring 2026/CS4100/CS4100_Final_Project_Group20"

    all_features, all_notes, all_durations = [], [], []

    for midi_file in os.listdir(midi_dir):
        if not midi_file.endswith('.mid'):
            continue

        print(f"Processing {midi_file}...")
        midi_path = os.path.join(midi_dir, midi_file)

        # Convert MIDI to WAV
        wav_path = midi_to_wav(midi_path, output_dir, soundfont_path)

        # Load WAV
        y, sr = librosa.load(wav_path, sr=22050)

        # Extract labels from MIDI
        midi_data = pretty_midi.PrettyMIDI(midi_path)
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                start = int(note.start * sr)
                end = int(note.end * sr)
                segment = y[start:end]

                if len(segment) == 0:
                    continue

                cqt = create_cqt(segment, sr)
                cqt_fixed = pad_or_truncate(cqt)

                all_features.append(cqt_fixed)
                all_notes.append(pretty_midi.note_number_to_name(note.pitch))
                all_durations.append(round(note.end - note.start, 4))

    # Save dataset
    features = np.stack(all_features)  # shape: (num_notes, 84, 128)
    df = pd.DataFrame({'note': all_notes, 'duration': all_durations})

    np.save(os.path.join(save_dir, 'features.npy'), features)
    df.to_csv(os.path.join(save_dir, 'labels.csv'), index=False)

    print(f"Saved {len(df)} notes")
    print(f"Features shape: {features.shape}")
    print(df['note'].value_counts())

if __name__ == "__main__":
    main()