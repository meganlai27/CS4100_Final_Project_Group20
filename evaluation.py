import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from music21 import stream, note, tempo, metadata

from model import CNN
from rhythm_cnn import RhythmCNN
from dataset import NoteDataset


def load_data():
    # Load dataset
    features = np.load('features.npy')
    df = pd.read_csv('labels.csv')

    # Encode note labels
    le_note = LabelEncoder()
    y_note = le_note.fit_transform(df['note'].values)

    y_duration, le_duration = None, None
    if 'duration' in df.columns:
        le_duration = LabelEncoder()
        y_duration = le_duration.fit_transform(df['duration'].values)
        if os.path.exists('rhythm_label_encoder_classes.npy'):
            le_duration.classes_ = np.load('rhythm_label_encoder_classes.npy', allow_pickle=True)

    return features, y_note, y_duration, le_note, le_duration


def get_test_loader(features, labels):
    dataset = NoteDataset(features, labels)
    
    # Train/val/test split matching dataset length
    total = len(dataset)
    test_size = int(total * 0.1)
    val_size = int(total * 0.1)
    train_size = total - test_size - val_size

    generator = torch.Generator().manual_seed(42)
    _, _, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    return DataLoader(test_dataset, batch_size=32, shuffle=False)


def evaluate_model(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            out = model(features)
            
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_preds), np.array(all_labels)


def print_top_confusions(y_true, y_pred, classes, top_n=3):
    cm = confusion_matrix(y_true, y_pred)
    np.fill_diagonal(cm, 0) # Ignore correct predictions
    
    confusions = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > 0:
                confusions.append((classes[i], classes[j], cm[i, j]))
                
    confusions.sort(key=lambda x: x[2], reverse=True)
    
    print(f"Top {top_n} Confusions:")
    for i, (true_val, pred_val, count) in enumerate(confusions[:top_n]):
        print(f"  {i+1}. True: {true_val} -> Predicted: {pred_val} (Count: {count})")


def save_confusion_matrix(y_true, y_pred, classes, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join('evaluation_results', filename))
    plt.close()


def generate_sheet_music(preds, le_note, rhythm_preds=None, le_duration=None, filename="predicted_output.mid"):
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

    save_path = os.path.join('evaluation_results', filename)
    s.write("midi", fp=save_path)
    s.write("musicxml", fp=save_path.replace(".mid", ".musicxml"))


def main():
    os.makedirs('evaluation_results', exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    features, y_note, y_duration, le_note, le_duration = load_data()

    note_exists = os.path.exists('note_classifier.pth')
    rhythm_exists = y_duration is not None and os.path.exists('rhythm_classifier.pth')

    # Handle scenario where neither model is trained
    if not note_exists and not rhythm_exists:
        print("Neither 'note_classifier.pth' nor 'rhythm_classifier.pth' was found.")
        return

    note_preds = None
    rhythm_preds = None

    # Note model evaluation
    if note_exists:
        print("\nEvaluating Note Classifier...")
        test_loader = get_test_loader(features, y_note)
        
        note_model = CNN(num_notes=len(le_note.classes_)).to(device)
        note_model.load_state_dict(torch.load('note_classifier.pth', map_location=device))
        
        note_preds, labels = evaluate_model(note_model, test_loader, device)
        
        acc = accuracy_score(labels, note_preds)
        f1 = f1_score(labels, note_preds, average="macro", zero_division=0)
        print(f"Note Accuracy: {acc:.4f} | Note F1 (Macro): {f1:.4f}")
        
        print_top_confusions(labels, note_preds, le_note.classes_)
        save_confusion_matrix(labels, note_preds, le_note.classes_, "Note Confusion Matrix", "note_cm.png")
    else:
        print("\nSkipping Note Classifier.")

    # Rhythm model evaluation
    if rhythm_exists:
        print("\nEvaluating Rhythm Classifier...")
        rhythm_loader = get_test_loader(features, y_duration)
        
        rhythm_model = RhythmCNN(num_classes=len(le_duration.classes_)).to(device)
        rhythm_model.load_state_dict(torch.load('rhythm_classifier.pth', map_location=device))
        
        rhythm_preds, r_labels = evaluate_model(rhythm_model, rhythm_loader, device)
        
        r_acc = accuracy_score(r_labels, rhythm_preds)
        r_f1 = f1_score(r_labels, rhythm_preds, average="macro", zero_division=0)
        print(f"Rhythm Accuracy: {r_acc:.4f} | Rhythm F1 (Macro): {r_f1:.4f}")
        
        print_top_confusions(r_labels, rhythm_preds, le_duration.classes_)
        save_confusion_matrix(r_labels, rhythm_preds, le_duration.classes_, "Rhythm Confusion Matrix", "rhythm_cm.png")
    else:
        print("\nSkipping Rhythm Classifier.")

    # Final Sheet Music Generation (combines both if available)
    if note_preds is not None:
        print("\nGenerating final sheet music...")
        if rhythm_preds is not None:
            generate_sheet_music(note_preds[:50], le_note, rhythm_preds=rhythm_preds[:50], le_duration=le_duration)
        else:
            generate_sheet_music(note_preds[:50], le_note)
            
    print("\nEvaluation complete. Results saved to 'evaluation_results' folder.")

if __name__ == "__main__":
    main()