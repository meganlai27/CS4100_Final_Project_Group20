import os
import glob
import subprocess
import numpy as np
import pandas as pd
import librosa
import pretty_midi
import data_processing as dp
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import matplotlib.pyplot as plt

SAMPLE_RATE = 22050
soundfont_path = "GeneralUser-GS/GeneralUser-GS.sf2"

# helpers for threading
print_lock = threading.Lock()
def safe_print(*args):
    with print_lock:
        print(*args, flush=True)

# convert a midi files to wav

def midi_to_wav(midi_file, output_dir, soundfont_path):
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(
        output_dir,
        os.path.basename(midi_file).replace('.mid', '.wav')
    )

    # Skip if already converted
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        return output_file, True

    cmd = [
        'fluidsynth', '-ni',
        '-F', output_file,
        '-r', f'{SAMPLE_RATE}',
        soundfont_path,
        midi_file
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        success = os.path.exists(output_file) and os.path.getsize(output_file) > 0
        return output_file, success
    except Exception as e:
        safe_print(f"FluidSynth failed on {midi_file}: {e}")
        return output_file, False
    
# Extract the features and corresponding labels from the wav/midi pairs
def extract_features(midi_path, wav_path):
    features, labels = [], []
    try:
        y, sr = librosa.load(wav_path, sr=SAMPLE_RATE)
        if len(y) == 0:
            return features, labels

        midi_data = pretty_midi.PrettyMIDI(midi_path)
        window_size = 0.5
        hop_size = 0.25
        total_duration = len(y) / sr

        t = 0.0
        while t + window_size <= total_duration:
            start_sample = int(t * sr)
            end_sample = int((t + window_size) * sr)
            segment = y[start_sample:end_sample]

            # Note: We can get more features from the dataset 
            # in addition to active_notes when expanding in the future
            active_notes = []
            for instrument in midi_data.instruments:
                for note in instrument.notes:
                    if note.start < (t + window_size) and note.end > t:
                        active_notes.append(pretty_midi.note_number_to_name(note.pitch))

            if active_notes:
                cqt = dp.create_cqt(segment, sr)
                cqt_fixed = dp.pad_or_truncate(cqt)
                features.append(cqt_fixed)
                labels.append(active_notes)

            t += hop_size

    except Exception as e:
        safe_print(f"Error extracting features from {midi_path}: {e}")

    return features, labels


from tqdm import tqdm
# batch for more efficient processing
# Citation: Used generative AI Claude to debug threading to process files more quickly
def batch_convert_midis(midi_files, output_dir, soundfont_path, num_workers=4):
    safe_print(f"Converting {len(midi_files)} MIDI files to WAV...")
    wav_map = {}  # midi_path -> wav_path
    failed = 0

    if num_workers > 1:

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(midi_to_wav, f, output_dir, soundfont_path): f
                for f in midi_files
            }
            for i, fut in enumerate(as_completed(futures)):
                midi_path = futures[fut]
                wav_path, success = fut.result()
                if success:
                    wav_map[midi_path] = wav_path
                else:
                    failed += 1
                if (i + 1) % 100 == 0:
                    safe_print(f"Converted {i+1}/{len(midi_files)} ({failed} failed)...")

    else:
        print("running in serial")
        for i, midi_path in tqdm(enumerate(midi_files)):
            wav_path, success = midi_to_wav(midi_path, output_dir, soundfont_path)

            if success:
                    wav_map[midi_path] = wav_path
            else:
                failed += 1


    safe_print(f"Conversion done: {len(wav_map)} succeeded, {failed} failed")
    return wav_map

def batch_extract_features(wav_map, num_workers=4):
    safe_print(f"Extracting features from {len(wav_map)} files...")
    all_features, all_labels = [], []

    if num_workers > 1:
        results_lock = threading.Lock()

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(extract_features, midi, wav): midi
                for midi, wav in wav_map.items()
            }
            for i, fut in tqdm(enumerate(as_completed(futures))):
                try:
                    features, labels = fut.result()
                    with results_lock:
                        all_features.extend(features)
                        all_labels.extend(labels)
                except Exception as e:
                    safe_print(f"Failed: {e}")
                if (i + 1) % 100 == 0:
                    safe_print(f"Extracted {i+1}/{len(wav_map)} files...")
    else:
        print("running in serial")
        for midi, wav in tqdm(wav_map.items()):
            features, labels = extract_features(midi, wav)
            all_features.extend(features)
            all_labels.extend(labels)


    return all_features, all_labels


# pad features to the same length
def pad_features(feature, max_len):
    if len(feature) < max_len:
        return np.pad(feature, (0, max_len - len(feature)))
    return feature[:max_len]


# process all the data for data preparation
def process_data(midi_dir, output_dir, soundfont_path, limit=None, num_workers=4):
    midi_files = glob.glob(os.path.join(midi_dir, "**/*.mid"), recursive=True)
    if limit:
        midi_files = midi_files[:limit]
    safe_print(f"Found {len(midi_files)} MIDI files")

    # convert midi to wav (skips wav files that are already created)
    wav_map = batch_convert_midis(midi_files, output_dir, soundfont_path, num_workers)

    # Extract features from all wav/midi pairs
    all_features, all_labels = batch_extract_features(wav_map, num_workers)

    # Pad features to fixed length
    all_features = [pad_features(f, dp.FIXED_LENGTH) for f in all_features]

    return all_features, all_labels

    
def save_dataprocessing(all_features, all_labels, save_features, save_labels):
    features = np.stack(all_features)
    df = pd.DataFrame({'labels': all_labels})

    np.save(save_features, all_features)
    df.to_csv(save_labels, index=False)

    print(f"Saved {len(df)} notes")
    print(f"Features shape: {features.shape}")
    print(df['labels'].value_counts())

def main():
    OUTPUT_DIR = "wav_chords"
    MIDI_DIR = "midi_chords"
    features_path = 'features_multilabel_chords.npy'
    labels_path = 'labels_multilabel_chords.csv'

    num_workers = 1

    all_features, all_labels = process_data(midi_dir = MIDI_DIR, output_dir=OUTPUT_DIR, soundfont_path=soundfont_path, limit=5000, num_workers = num_workers)
    
    save_dataprocessing(all_features, all_labels, features_path, labels_path)

    return 

if __name__ == "__main__":
    all_labels = main()