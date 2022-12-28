import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
# Can create neural network
from keras.models import Sequential
# LSTM is long short term memory (reccurent layer, memory of model)
# Dense layers for hidden layers
# Activation for output layer
from keras.models import LSTM, Dense, Activation
# RSM optimizer (compile model)
from keras.models import RMSprep

# Directly load data into script
filepath = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

# rb = read binary mode
# lowercase makes better performace
text = open(filepath, 'rb').read().decode(encoding='utf-8').lower()

# Entire text will take too long, need to slice
text = text[300000:800000]

# Filter out unique characters and sort
characters = sorted(set(text))

# Convert into numerical format so we can
# pass numpy array into neural network

# c for character, i for index
# (i.e,) {'a': 1, 'f': 25}
char_to_index = dict((c, i) for i, c in enumerate(characters))

# Convert back
# (i.e,) {1: 'a', 25: 'f'}
index_to_char = dict((i, c) for i, c in enumerate(characters))

# Amount of characters used to predict next one
SEQ_LEN = 40
# Amount of characters to shift to start the next sequence
STEP_SIZE = 3

# Create entry list of sentences and next char
sentences = []
next_characters = []

for i in range(0, len(text) - SEQ_LEN, STEP_SIZE):

    # Need to include last value (i + SEQ_LEN)
    sentences.append(text[i: i + SEQ_LEN])
    next_characters.append(text[i: i + SEQ_LEN])
