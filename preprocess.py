from music21 import converter, instrument, note, chord
import os
import pickle

notes = []

for file in os.listdir("dataset"):
    if file.endswith(".mid"):
        midi = converter.parse(f"dataset/{file}")

        parts = instrument.partitionByInstrument(midi)

        if parts:
            notes_to_parse = parts.parts[0].recurse()
        else:
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))

            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

with open("notes.pkl", "wb") as f:
    pickle.dump(notes, f)

print("Preprocessing completed")