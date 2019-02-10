# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 23:10:12 2019

@author: tanma
"""
# Import Filenames
import os,sys
filenames = []
for file in os.listdir():
    filenames.append(file)

# Importing Modules
from keras.models import Model
from keras.layers import Input, LSTM, GRU, Dense, Embedding
from keras.layers import Bidirectional, RepeatVector, Concatenate, Activation, Dot, Lambda
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import keras.backend as K
from gensim.models import Word2Vec
import numpy as np
import matplotlib.pyplot as plt
from PyPDF2 import PdfFileReader

if len(K.tensorflow_backend._get_available_gpus()) > 0:
  from keras.layers import CuDNNLSTM as LSTM
  from keras.layers import CuDNNGRU as GRU    


# Parse Input
input_text = []
input_texts = []
target_texts = []
target_texts_inputs = []
lines = []
for line in open('all_rel_data.txt'):
  lines.append(line.rstrip())

while '' in lines:
    lines.remove('')
    
for i in lines[:-1]:
    input_texts.append(i)

translation_texts = lines[1:]

for i in translation_texts:
    target_texts.append(i+' <eos>')
    target_texts_inputs.append('<sos> '+i)

# Config
BATCH_SIZE = 48
EPOCHS = 100  
LATENT_DIM = 256  
NUM_SAMPLES = 10000  
MAX_SEQUENCE_LENGTH = 100
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100

# Tokenize
tokenizer_inputs = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer_inputs.fit_on_texts(input_texts)
input_sequences = tokenizer_inputs.texts_to_sequences(input_texts)

tokenizer_outputs = Tokenizer(num_words=MAX_NUM_WORDS, filters='')
tokenizer_outputs.fit_on_texts(target_texts + target_texts_inputs) 
target_sequences = tokenizer_outputs.texts_to_sequences(target_texts)
target_sequences_inputs = tokenizer_outputs.texts_to_sequences(target_texts_inputs)

# Word to Index Dictionary
word2idx_inputs = tokenizer_inputs.word_index

max_len_input = max(len(s) for s in input_sequences)

word2idx_outputs = tokenizer_outputs.word_index

num_words_output = len(word2idx_outputs) + 1

max_len_target = max(len(s) for s in target_sequences)

# Pad Sequences
encoder_inputs = pad_sequences(input_sequences, maxlen=max_len_input)

decoder_inputs = pad_sequences(target_sequences_inputs, maxlen=max_len_target, padding='post')

decoder_targets = pad_sequences(target_sequences, maxlen=max_len_target, padding='post')

# Load Glove
word2vec = {}
with open(os.path.join('very_large_data/glove.6B.%sd.txt' % EMBEDDING_DIM),'rb') as f:
  for line in f:
    values = line.split()
    word = values[0]
    vec = np.asarray(values[1:], dtype='float32')
    word2vec[word] = vec
print('Found %s word vectors.' % len(word2vec))

# Fill in the Embedding Matrix
num_words = min(MAX_NUM_WORDS, len(word2idx_inputs) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word2idx_inputs.items():
  if i < MAX_NUM_WORDS:
    embedding_vector = word2vec.get(word)
    if embedding_vector is not None:
      embedding_matrix[i] = embedding_vector     

# Embedding Layer with Glove
embedding_layer = Embedding(
  num_words,
  EMBEDDING_DIM,
  weights=[embedding_matrix],
  input_length=max_len_input,
  trainable=True
)

# One hot the targets for Categorical Crossentropy
decoder_targets_one_hot = np.zeros(
  (
    len(input_texts),
    max_len_target,
    num_words_output
  ),
  dtype='float32'
)

for i, d in enumerate(decoder_targets):
  for t, word in enumerate(d):
    decoder_targets_one_hot[i, t, word] = 1

" ----------------------------- Training Model -------------------------------"
encoder_inputs_placeholder = Input(shape=(max_len_input,))
x = embedding_layer(encoder_inputs_placeholder)
encoder = LSTM(
  LATENT_DIM,
  return_state=True,
)
encoder_outputs, h, c = encoder(x)

encoder_states = [h, c]

decoder_inputs_placeholder = Input(shape=(max_len_target,))

decoder_embedding = Embedding(num_words_output, LATENT_DIM)
decoder_inputs_x = decoder_embedding(decoder_inputs_placeholder)

decoder_lstm = LSTM(
  LATENT_DIM,
  return_sequences=True,
  return_state=True,
)
decoder_outputs, _, _ = decoder_lstm(
  decoder_inputs_x,
  initial_state=encoder_states
)

decoder_dense = Dense(num_words_output, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs_placeholder, decoder_inputs_placeholder], decoder_outputs)

model.compile(
  optimizer='rmsprop',
  loss='categorical_crossentropy',
  metrics=['accuracy']
)
r = model.fit(
  [encoder_inputs, decoder_inputs], decoder_targets_one_hot,
  batch_size=BATCH_SIZE,
  epochs=EPOCHS
)

model.save('s2s.h5')

"-------------------------------- Prediction Model ---------------------------"
encoder_model = Model(encoder_inputs_placeholder, encoder_states)

decoder_state_input_h = Input(shape=(LATENT_DIM,))
decoder_state_input_c = Input(shape=(LATENT_DIM,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_inputs_single = Input(shape=(1,))
decoder_inputs_single_x = decoder_embedding(decoder_inputs_single)

decoder_outputs, h, c = decoder_lstm(
  decoder_inputs_single_x,
  initial_state=decoder_states_inputs
)

decoder_states = [h, c]

decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model(
  [decoder_inputs_single] + decoder_states_inputs, 
  [decoder_outputs] + decoder_states
)

# Inverse Word Indexes
idx2word_eng = {v:k for k, v in word2idx_inputs.items()}
idx2word_trans = {v:k for k, v in word2idx_outputs.items()}

# Function to get output
def decode_sequence(input_seq):
  states_value = encoder_model.predict(input_seq)

  target_seq = np.zeros((1, 1))

  target_seq[0, 0] = word2idx_outputs['<sos>']

  eos = word2idx_outputs['<eos>']

  output_sentence = []
  for _ in range(max_len_target):
    output_tokens, h, c = decoder_model.predict(
      [target_seq] + states_value
    )
    
    idx = np.argmax(output_tokens[0, 0, :])

    if eos == idx:
      break

    word = ''
    if idx > 0:
      word = idx2word_trans[idx]
      output_sentence.append(word)

    target_seq[0, 0] = idx

    states_value = [h, c]

  return ' '.join(output_sentence)

# Custom Input
def custom_input(string):
    input_seq = tokenizer_inputs.texts_to_sequences(string)
    encoder_in = pad_sequences(input_seq, maxlen=max_len_input)
    translation = decode_sequence(encoder_in)
    return translation

take_input = str(input())
print(custom_input(take_input))

    
