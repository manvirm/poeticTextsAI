import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
# Can create neural network
from keras.models import Sequential
# LSTM is long short term memory (reccurent layer, memory of model)
# Dense layers for hidden layers
# Activation for output layer
from keras.layers import LSTM, Dense, Activation
# RSM optimizer (compile model)
from keras.optimizers import RMSprop

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
    next_characters.append(text[i + SEQ_LEN])

# Need numerical format of training data
# Numpy array full of zeros with length of amount of sentences
# times sequence len times amount of characters

# One dimension for all possible sentences
# Another dimension for positions of sentences
# Last dimension for all characters
x = np.zeros((len(sentences), SEQ_LEN, len(characters)), dtype=np.bool_)
y = np.zeros((len(sentences), len(characters)),dtype=np.bool_)

# Fill arrays x and y
# Go through every single character in sentence
# True (1) if character is in sentence
for i, sentence in enumerate(sentences):
    for t, character in enumerate(sentence):
        x[i, t, char_to_index[character]] = 1
    y[i, char_to_index[next_characters[i]]] = 1

'''
# Build Neural Network
model = Sequential()
# Store in memory
model.add(LSTM(128, input_shape = (SEQ_LEN, len(characters))))
model.add(Dense(len(characters)))
# Scales output so probabilities add to 1
model.add(Activation('softmax'))
# Loss function, learning rate 0.01
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.01))
# Fit model on training data, batch_size means how many examples we put in 
# network at once
model.fit(x, y, batch_size=256, epochs=4)
# No need to keep training over again
model.save('textgenerator.model')
'''
model = tf.keras.models.load_model('textgenerator.model')

# Take prediction of model and one character
# Gets softmax result and will choice thats conservative or safe
# Higher temp will be more conservative (more creative sentence)
def sample(preds, temperature = 1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.orgnax(probas)

def generate_text(length, temeprature):
    start_index = random.randint(0, len(text) - SEQ_LEN)
