from music21 import stream, note, tempo
import random
import os

NUM_FILES = 5000    
MIN_PITCH = 36             # C2
MAX_PITCH = 96             # C7
DURATIONS = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]  # sixteenth, eighth, quarter, dotted quarter, half, dotted half, whole
OUTPUT_DIR = "Midi Files"

def generate_random_midi(output_path):
    s = stream.Stream()
    bpm = random.randint(60, 180)
    s.append(tempo.MetronomeMark(number=bpm))
    notes_per_file = random.randint(10, 50)
    for _ in range(notes_per_file):
        pitch = random.randint(MIN_PITCH, MAX_PITCH)
        duration = random.choice(DURATIONS)
        n = note.Note(pitch)
        n.duration.quarterLength = duration
        s.append(n)

    s.write('midi', fp=output_path)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for i in range(NUM_FILES):
        output_path = os.path.join(OUTPUT_DIR, f"random_{i+1}.mid")
        generate_random_midi(output_path)

        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1}/{NUM_FILES} files...")

    print(f"Done! {NUM_FILES} MIDI files saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()