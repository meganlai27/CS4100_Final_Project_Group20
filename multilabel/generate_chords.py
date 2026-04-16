from music21 import stream, note, chord, tempo, key, meter
import random
import os
from tqdm import tqdm

NUM_FILES = 500
OUTPUT_DIR = "midi_chords"

# Named chords: (chord name, intervals from root in semitones)
CHORD_TYPES = {
    "major":       [0, 4, 7],
    "minor":       [0, 3, 7],
    "diminished":  [0, 3, 6],
    "augmented":   [0, 4, 8],
    "major7":      [0, 4, 7, 11],
    "minor7":      [0, 3, 7, 10],
    "dominant7":   [0, 4, 7, 10],
    "sus2":        [0, 2, 7],
    "sus4":        [0, 5, 7],
}

DURATIONS = [0.5, 1.0, 1.5, 2.0, 4.0]
ROOT_NOTES = list(range(48, 72))  # C3 to B4 as roots

def build_chord(root_midi, chord_type_name):
    """Build a music21 chord from a root MIDI note and chord type name."""
    intervals = CHORD_TYPES[chord_type_name]
    pitches = [root_midi + i for i in intervals]
    c = chord.Chord(pitches)
    return c, chord_type_name  # return label too

def generate_chord_midi(output_path, num_chords=16):
    random.seed()

    s = stream.Stream()
    bpm = random.randint(60, 140)
    s.append(tempo.MetronomeMark(number=bpm))
    s.append(meter.TimeSignature('4/4'))

    labels = []  # track what chords were used

    for _ in range(num_chords):
        root = random.choice(ROOT_NOTES)
        chord_type = random.choice(list(CHORD_TYPES.keys()))
        duration = random.choice(DURATIONS)

        c, label = build_chord(root, chord_type)
        c.duration.quarterLength = duration
        s.append(c)
        labels.append(label)

    s.write('midi', fp=output_path)
    return labels

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_labels = {}

    for i in tqdm(range(NUM_FILES)):
        output_path = os.path.join(OUTPUT_DIR, f"chords_{i+1}.mid")
        labels = generate_chord_midi(output_path)
        all_labels[f"chords_{i+1}.mid"] = labels

    return all_labels

if __name__ == "__main__":
    all_labels = main()