import librosa
import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
from music21 import stream, note, tempo, metadata

import data_processing as dp
from note_cnn import CNN
from rhythm_cnn import RhythmCNN

FILENAME = "Hilary Hahn - J.S. Bach Partita for Violin Solo No. 1 in B Minor, BWV 1002 - 4. Double (Presto) - Hilary Hahn (128k)"

# Process real audio from wav file
def create_cqt(y, sr, hop_length=512):
    cqt = librosa.cqt(y, sr=sr, hop_length=hop_length, n_bins=dp.N_BINS, bins_per_octave=12)
    cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
    cqt_db = np.maximum(cqt_db, -80)  
    return cqt_db

# Try number #1 with raw audio (no noise reduction)
def process_recording(audio_path, delta=0.07, wait=8, pre_max=3, post_max=3):
    y, sr = librosa.load(audio_path, sr=22050)

    onset_frames = librosa.onset.onset_detect(
        y=y, sr=sr,
        delta=delta,       
        wait=wait,         
        pre_max=pre_max,
        post_max=post_max,
        backtrack=True    
    )
    onset_samples = librosa.frames_to_samples(onset_frames)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    features = []
    valid_onset_times = []

    # For each note sampled, get the spectogram/features
    for i, start in enumerate(onset_samples):
        end = onset_samples[i + 1] if i + 1 < len(onset_samples) else len(y)
        segment = y[start:end]

        if len(segment) < dp.MIN_SEGMENT_LENGTH:
            continue  

        cqt = create_cqt(segment, sr)
        cqt_fixed = dp.pad_or_truncate(cqt)
        features.append(cqt_fixed)
        valid_onset_times.append(onset_times[i])

    features = np.stack(features) if features else np.empty((0, dp.N_BINS, dp.FIXED_LENGTH))
    return features, valid_onset_times

def get_preds(model, features, device = 'cpu'):
    model.eval()
    all_preds = []

    with torch.no_grad():
        # for features, labels in loader:
        features = np.array(features, dtype=np.float32)  # force correct type
    
        # Now convert to tensor and add channel dim
        tensor = torch.from_numpy(features).unsqueeze(1)
        out = model(tensor)
        
        all_preds.extend(out.argmax(1).cpu().numpy())

    return np.array(all_preds)

import numpy as np
def predict_notes(features: np.ndarray, note_model, rhythm_model):
    # Ensure it's a plain numpy array of the right dtype first
    features = np.array(features, dtype=np.float32)  
    
    # Now convert to tensor and add channel dim
    tensor = torch.from_numpy(features).unsqueeze(1)  
    
    # print(type(tensor), tensor.dtype, tensor.shape)

    note_preds = get_preds(note_model, features)

    rhythm_preds = get_preds(rhythm_model, features)

    return note_preds, rhythm_preds

def generate_musicxml(preds, le_note, rhythm_preds=None, le_duration=None, file_path="predicted_output.mid"):
    s = stream.Stream()
    s.append(metadata.Metadata())
    s.append(tempo.MetronomeMark(number=120))

    classes = le_note.classes_

    for i, idx in enumerate(preds):
        try:
            n = note.Note(str(classes[idx]))
            
            # Integrate rhythm predictions if both models were evaluated
            if rhythm_preds is not None and le_duration is not None:
                n.duration.quarterLength = float(le_duration.classes_[rhythm_preds[i]])
            else:
                n.duration.quarterLength = 1.0 
                
            s.append(n)
        except Exception:
            continue

    save_path = os.path.join('predictions_midi', file_path)
    s.write("midi", fp=save_path)
    s.write("musicxml", fp=save_path.replace(".mid", ".musicxml"))

def generate_sheet_music(wav_file, output_path):
    # Prepare models and get labels
    df = pd.read_csv('./labels.csv')

    le_note = LabelEncoder()
    y_note = le_note.fit_transform(df['note'].values)

    le_duration = LabelEncoder()
    y_duration = le_duration.fit_transform(df['duration'].values)
    if os.path.exists('rhythm_label_encoder_classes.npy'):
        le_duration.classes_ = np.load('rhythm_label_encoder_classes.npy', allow_pickle=True)

    note_model = CNN(num_notes=len(le_note.classes_))
    note_model.load_state_dict(torch.load('note_cnn.pth'))
    rhythm_model = RhythmCNN(num_classes=len(le_duration.classes_))
    rhythm_model.load_state_dict(torch.load('rhythm_classifier.pth'))

    features, onset_times = process_recording(wav_file)
    print(f"Extracted {len(features)} note segments")
    print(f"Features shape: {features.shape}")

    notes, rhythms = predict_notes(features, note_model, rhythm_model)

    if notes is not None:
        print("\nGenerating final sheet music...")
        if rhythms is not None:
            generate_musicxml(notes, le_note, rhythm_preds=rhythms, le_duration=le_duration, file_path = output_path)
        else:
            generate_musicxml(notes, le_note, output_path)

    print(f'Generate Sheet Music saved to {output_path}')

    return output_path


if __name__ == "__main__":
    filename = FILENAME
    generate_sheet_music(f"process_recording_files/{filename}.wav", f"pred_{filename}.mid")