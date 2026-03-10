from music21 import stream, note, tempo
import random
import os

# Configuration
NUM_FILES = 5000
NOTES_PER_FILE = 10        # number of random notes per MIDI file
MIN_PITCH = 36             # C2
MAX_PITCH = 96             # C7
DURATIONS = [0.25, 0.5, 1.0, 1.5, 2.0]  # quarter, half, whole, etc.
BPM = 120
OUTPUT_DIR = "/Users/melvincheng/Documents/Spring 2026/CS4100/CS4100_Final_Project_Group20/Midi Files"

def generate_random_midi(output_path, notes_per_file=NOTES_PER_FILE):
    s = stream.Stream()
    s.append(tempo.MetronomeMark(number=BPM))

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