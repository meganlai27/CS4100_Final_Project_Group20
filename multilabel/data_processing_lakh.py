import kagglehub
import librosa
import data_processing as dp
import librosa
import pretty_midi
import os
import glob
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count


'''
This file downloads and processes the data from lakh.

Lakh is a dataset containing the midi files of 176,581 unique songs.
The goal of using this dataset is to understand common patterns, rhythms, and combined notes.
'''

# Contants
output_dir="lakh_wav" # where to place wav files
soundfont_path = "GeneralUser-GS/GeneralUser-GS.sf2"

'''
This method processes a singular midi file from this dataset such that it will extract
all notes from all voices at a given time frame.

This method is passed into a thread to be executed.

1. Get the wav file from the midi file
2. For each time frame in a given midi file
    3. Generate a spectogram from the wav file
    4. Get the labels from the midi file for that time frame
'''
def process_single_midi(args):
    midi_path, output_dir, soundfont_path = args
    features, labels = [], []

    try:
        # print(f"Processing midi file: {midi_path}", flush=True)
        wav_path, val = dp.midi_to_wav(midi_path, output_dir, soundfont_path)
        y, sr = librosa.load(wav_path, sr=22050)
        midi_data = pretty_midi.PrettyMIDI(midi_path)

        # Define time window
        window_size = 0.5
        hop_size = 0.25  
        total_duration = len(y) / sr
        
        t = 0.0
        while t + window_size <= total_duration:
            # Slice the mixed audio at this time window
            start_sample = int(t * sr)
            end_sample = int((t + window_size) * sr)
            segment = y[start_sample:end_sample]

            # Get all notes across different voices
            active_notes = []
            for instrument in midi_data.instruments:
                for note in instrument.notes:
                    # note overlaps with this window
                    if note.start < (t + window_size) and note.end > t:
                        active_notes.append(pretty_midi.note_number_to_name(note.pitch))

            if len(active_notes) == 0:
                t += hop_size
                continue

            # Compute spectrogram for all notes
            cqt = dp.create_cqt(segment, sr)
            cqt_fixed = dp.pad_or_truncate(cqt)

            features.append(cqt_fixed)
            labels.append(active_notes) 
            
            t += hop_size
        print(f"Finished processing midi file: {midi_path}")

    except Exception as e:
        print(f"Error processing {midi_path}: {e}. Skipping...")

    return features, labels

'''
Process all midi files and extract their features and labels.
'''
def process_data(midi_dir, with_pool = True):
    midi_files = glob.glob(os.path.join(midi_dir, "**/*.mid"), recursive=True)
    print(f"Found {len(midi_files)} MIDI files")
    
    num_cpu = 2 # cpu_count() # cpu_count can be slow
    print(f"Using {cpu_count()} CPU cores")

    args = [(f, output_dir, soundfont_path) for f in midi_files[:100]]
    print(f"Processing {len(args)} midi files")

    all_features, all_labels = [], []

    def pad_features(feature, max_len):
        if len(feature) < max_len:
            # pad with zeros at the end
            return np.pad(feature, (0, max_len - len(feature)))
        else:
            # truncate if too long
            return feature[:max_len]


    if with_pool:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        print("Processing with pooling")


        with ThreadPoolExecutor(max_workers=num_cpu) as executor:
            futures = [executor.submit(process_single_midi, arg) for arg in args]
            for fut in as_completed(futures):
                features, labels = fut.result()
                all_features.extend(features)
                all_labels.extend(labels)
    else:
        for i, arg in enumerate(args):
            features, labels = process_single_midi(arg)
            all_features.extend(features)
            all_labels.extend(labels)
            if (i + 1) % 10 == 0:
                print(f"Processed {i+1}/{len(args)} files...")

    # Apply before stacking
    all_features = [pad_features(f, dp.FIXED_LENGTH) for f in all_features]

    return all_features, all_labels
    
'''
Save all the features and labels into the given path.
'''
def save_dataprocessing(all_features, all_labels, save_features, save_labels):
    features = np.stack(all_features)
    df = pd.DataFrame({'labels': all_labels})

    np.save(save_features, all_features)
    df.to_csv(save_labels, index=False)

    print(f"Saved {len(df)} notes")
    print(f"Features shape: {features.shape}")
    print(df['labels'].value_counts())


# Run all data processing
def main():
    # Download the dataset from kaggle
    path = kagglehub.dataset_download("imsparsh/lakh-midi-clean")
    print("Path to dataset files:", path)

    # Process all the midi files and get all the labels and features from wav files
    all_features, all_labels = process_data(midi_dir = path, with_pool=True)

    # save all the files
    features_file_path = 'features_multilabel.npy'
    labels_file_path = 'labels_multilabel.csv'
    save_dataprocessing(all_features, all_labels, features_file_path, labels_file_path)

    print(f'Saved the features and labels to the following files: \n{features_file_path}\n{labels_file_path}')


if __name__ == "__main__":
    main()