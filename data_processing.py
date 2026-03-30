import librosa
import pretty_midi
import os
import glob
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count

FIXED_LENGTH = 128
N_BINS = 84
MIN_SEGMENT_LENGTH = 1024

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

def seconds_to_beat_fraction(duration_seconds, bpm):
    """
    Convert duration in seconds to number of quarter notes.
    Quarter note = 1.0, Half note = 2.0, Whole note = 4.0, etc.
    Snaps to nearest 0.25 subdivision.
    """
    quarter_note_duration = 60 / bpm
    beat_fraction = duration_seconds / quarter_note_duration
    return round(beat_fraction * 4) / 4  # snap to nearest 0.25


def process_single_midi(args):
    midi_path, output_dir, soundfont_path = args
    features, notes, durations = [], [], []

    try:
        wav_path = midi_to_wav(midi_path, output_dir, soundfont_path)
        y, sr = librosa.load(wav_path, sr=22050)
        midi_data = pretty_midi.PrettyMIDI(midi_path)
        tempo_times, tempo_values = midi_data.get_tempo_changes()
        bpm = tempo_values[0] if len(tempo_values) > 0 else 120.0

        for instrument in midi_data.instruments:
            for note in instrument.notes:
                start = int(note.start * sr)
                end = int(note.end * sr)
                segment = y[start:end]

                if len(segment) < MIN_SEGMENT_LENGTH:
                    continue

                cqt = create_cqt(segment, sr)
                cqt_fixed = pad_or_truncate(cqt)
                features.append(cqt_fixed)
                notes.append(pretty_midi.note_number_to_name(note.pitch))
                durations.append(seconds_to_beat_fraction(note.end - note.start, bpm))
    except Exception as e:
        print(f"Error processing {midi_path}: {e}")

    return features, notes, durations

def main():
    midi_dir = "Midi Files"
    output_dir = "Wav_Files"
    soundfont_path = "GeneralUser-GS/GeneralUser-GS.sf2"

    midi_files = glob.glob(os.path.join(midi_dir, "**/*.mid"), recursive=True)
    print(f"Found {len(midi_files)} MIDI files")
    print(f"Using {cpu_count()} CPU cores")

    args = [(f, output_dir, soundfont_path) for f in midi_files]

    all_features, all_notes, all_durations = [], [], []

    with Pool(processes=cpu_count()) as pool:
        for i, (features, notes, durations) in enumerate(pool.imap(process_single_midi, args)):
            all_features.extend(features)
            all_notes.extend(notes)
            all_durations.extend(durations)
            if (i + 1) % 100 == 0:
                print(f"Processed {i+1}/{len(midi_files)} files...")

    features = np.stack(all_features)
    df = pd.DataFrame({'note': all_notes, 'duration': all_durations})

    np.save('features.npy', features)
    df.to_csv('labels.csv', index=False)

    print(f"Saved {len(df)} notes")
    print(f"Features shape: {features.shape}")
    print(df['note'].value_counts())

if __name__ == "__main__":
    main()