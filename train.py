import pickle
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

sequence_length = 100

with open("notes.pkl", "rb") as f:
    notes = pickle.load(f)

pitchnames = sorted(set(notes))

note_to_int = dict((note, number)
                   for number, note in enumerate(pitchnames))

network_input = []
network_output = []

for i in range(len(notes)-sequence_length):
    seq_in = notes[i:i+sequence_length]
    seq_out = notes[i+sequence_length]

    network_input.append([note_to_int[c] for c in seq_in])
    network_output.append(note_to_int[seq_out])

n_patterns = len(network_input)

network_input = np.reshape(
    network_input,
    (n_patterns, sequence_length, 1)
)

network_input = network_input / float(len(pitchnames))

model = Sequential()

model.add(
    LSTM(
        256,
        input_shape=(network_input.shape[1],
                     network_input.shape[2]),
        return_sequences=True
    )
)

model.add(Dropout(0.3))

model.add(LSTM(256))

model.add(Dense(len(pitchnames), activation="softmax"))

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam"
)

model.fit(
    network_input,
    np.array(network_output),
    epochs=50,
    batch_size=64
)

model.save("music_model.h5")