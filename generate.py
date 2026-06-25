import pickle
import random

from music21 import note, stream
from tensorflow.keras.models import load_model
import numpy as np

model = load_model("music_model.h5")

with open("notes.pkl", "rb") as f:
    notes = pickle.load(f)

pitchnames = sorted(set(notes))

int_to_note = dict(
    (number, note)
    for number, note in enumerate(pitchnames)
)

sequence_length = 100

start = random.randint(
    0,
    len(notes)-sequence_length-1
)

pattern = notes[start:start+sequence_length]

prediction_output = []

for note_index in range(200):

    input_sequence = np.reshape(
        [pitchnames.index(n) for n in pattern],
        (1, sequence_length, 1)
    )

    input_sequence = input_sequence / float(len(pitchnames))

    prediction = model.predict(
        input_sequence,
        verbose=0
    )

    index = np.argmax(prediction)

    result = int_to_note[index]

    prediction_output.append(result)

    pattern.append(result)
    pattern = pattern[1:]

output_notes = []

offset = 0

for pattern in prediction_output:

    new_note = note.Note(pattern)

    new_note.offset = offset

    output_notes.append(new_note)

    offset += 0.5

midi_stream = stream.Stream(output_notes)

midi_stream.write(
    'midi',
    fp='generated/output.mid'
)

print("Music generated successfully")