#!/usr/bin/env python
# coding: utf-8

# # Home 5: Build a seq2seq model for machine translation.
# 
# ### Name: [Tianyi Li]
# 
# ### Task: Translate English to [Dutch]

# ## 0. You will do the following:
# 
# 1. Read and run my code.
# 2. Complete the code in Section 1.1 and Section 4.2.
# 
#     * Translation **English** to **German** is not acceptable!!! Try another pair of languages.
#     
# 3. **Make improvements.** Directly modify the code in Section 3. Do at least one of the two. By doing both correctly, you will get up to 1 bonus score to the total.
# 
#     * Bi-LSTM instead of LSTM.
#         
#     * Attention. (You are allowed to use existing code.)
#     
# 4. Evaluate the translation using the BLEU score. 
# 
#     * Optional. Up to 1 bonus scores to the total.
#     
# 5. Convert the notebook to .HTML file. 
# 
#     * The HTML file must contain the code and the output after execution.
# 
# 6. Put the .HTML file in your Google Drive, Dropbox, or Github repo.  (If you submit the file to Google Drive or Dropbox, you must make the file "open-access". The delay caused by "deny of access" may result in late penalty.)
# 
# 7. Submit the link to the HTML file to Canvas.    
# 

# ### Hint: 
# 
# To implement ```Bi-LSTM```, you will need the following code to build the encoder. Do NOT use Bi-LSTM for the decoder.

# In[114]:


# from keras.layers import Bidirectional, Concatenate

# encoder_bilstm = Bidirectional(LSTM(latent_dim, return_state=True, 
#                                   dropout=0.5, name='encoder_lstm'))
# _, forward_h, forward_c, backward_h, backward_c = encoder_bilstm(encoder_inputs)

# state_h = Concatenate()([forward_h, backward_h])
# state_c = Concatenate()([forward_c, backward_c])


# ## 1. Data preparation
# 
# 1. Download data (e.g., "deu-eng.zip") from http://www.manythings.org/anki/
# 2. Unzip the .ZIP file.
# 3. Put the .TXT file (e.g., "deu.txt") in the directory "./Data/".

# ### 1.1. Load and clean text
# 

# In[1]:


import re
import string
from unicodedata import normalize
import numpy

# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, mode='rt', encoding='utf-8')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


# split a loaded document into sentences
def to_pairs(doc):
    lines = doc.strip().split('\n')
    pairs = [line.split('\t') for line in  lines]
    return pairs

def clean_data(lines):
    cleaned = list()
    # prepare regex for char filtering
    re_print = re.compile('[^%s]' % re.escape(string.printable))
    # prepare translation table for removing punctuation
    table = str.maketrans('', '', string.punctuation)
    for pair in lines:
        clean_pair = list()
        for line in pair:
            # normalize unicode characters
            line = normalize('NFD', line).encode('ascii', 'ignore')
            line = line.decode('UTF-8')
            # tokenize on white space
            line = line.split()
            # convert to lowercase
            line = [word.lower() for word in line]
            # remove punctuation from each token
            line = [word.translate(table) for word in line]
            # remove non-printable chars form each token
            line = [re_print.sub('', w) for w in line]
            # remove tokens with numbers in them
            line = [word for word in line if word.isalpha()]
            # store as string
            clean_pair.append(' '.join(line))
        cleaned.append(clean_pair)
    return numpy.array(cleaned)


# #### Fill the following blanks:

# In[2]:


# e.g., filename = 'Data/deu.txt'
filename = 'Data/nld.txt'

# e.g., n_train = 20000
# n_train = <how many sentences are you going to use for training?>
n_train = 50000


# In[3]:


# load dataset
doc = load_doc(filename)

# split into Language1-Language2 pairs
pairs = to_pairs(doc)

# clean sentences
clean_pairs = clean_data(pairs)[0:n_train, :]


# In[4]:


for i in range(3000, 3010):
    print('[' + clean_pairs[i, 0] + '] => [' + clean_pairs[i, 1] + ']')


# In[5]:


input_texts = clean_pairs[:, 0]
target_texts = ['\t' + text + '\n' for text in clean_pairs[:, 1]]

print('Length of input_texts:  ' + str(input_texts.shape))
print('Length of target_texts: ' + str(len(target_texts))) # Modified a tiny error in the source code


# In[6]:


max_encoder_seq_length = max(len(line) for line in input_texts)
max_decoder_seq_length = max(len(line) for line in target_texts)

print('max length of input  sentences: %d' % (max_encoder_seq_length))
print('max length of target sentences: %d' % (max_decoder_seq_length))


# In[7]:


### partition the dataset to training, validation, and test.###
from sklearn.model_selection import train_test_split

# Randomly partition the dataset
train_X,test_X,train_y,test_y = train_test_split(input_texts,target_texts,test_size=0.3,random_state=5)

input_texts = train_X
target_texts = train_y


# **Remark:** To this end, you have two lists of sentences: input_texts and target_texts

# ## 2. Text processing
# 
# ### 2.1. Convert texts to sequences
# 
# - Input: A list of $n$ sentences (with max length $t$).
# - It is represented by a $n\times t$ matrix after the tokenization and zero-padding.

# In[8]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# encode and pad sequences
def text2sequences(max_len, lines):
    tokenizer = Tokenizer(char_level=True, filters='')
    tokenizer.fit_on_texts(lines)
    seqs = tokenizer.texts_to_sequences(lines)
    seqs_pad = pad_sequences(seqs, maxlen=max_len, padding='post')
    return seqs_pad, tokenizer.word_index


encoder_input_seq, input_token_index = text2sequences(max_encoder_seq_length, 
                                                      input_texts)
decoder_input_seq, target_token_index = text2sequences(max_decoder_seq_length, 
                                                       target_texts)

print('shape of encoder_input_seq: ' + str(encoder_input_seq.shape))
print('shape of input_token_index: ' + str(len(input_token_index)))
print('shape of decoder_input_seq: ' + str(decoder_input_seq.shape))
print('shape of target_token_index: ' + str(len(target_token_index)))


# In[9]:


num_encoder_tokens = len(input_token_index) + 1
num_decoder_tokens = len(target_token_index) + 1

print('num_encoder_tokens: ' + str(num_encoder_tokens))
print('num_decoder_tokens: ' + str(num_decoder_tokens))


# **Remark:** To this end, the input language and target language texts are converted to 2 matrices. 
# 
# - Their number of rows are both n_train.
# - Their number of columns are respective max_encoder_seq_length and max_decoder_seq_length.

# The followings print a sentence and its representation as a sequence.

# In[10]:


target_texts[100]


# In[11]:


decoder_input_seq[100, :]


# ## 2.2. One-hot encode
# 
# - Input: A list of $n$ sentences (with max length $t$).
# - It is represented by a $n\times t$ matrix after the tokenization and zero-padding.
# - It is represented by a $n\times t \times v$ tensor ($t$ is the number of unique chars) after the one-hot encoding.

# In[12]:


from keras.utils import to_categorical

# one hot encode target sequence
def onehot_encode(sequences, max_len, vocab_size):
    n = len(sequences)
    data = numpy.zeros((n, max_len, vocab_size))
    for i in range(n):
        data[i, :, :] = to_categorical(sequences[i], num_classes=vocab_size)
    return data

encoder_input_data = onehot_encode(encoder_input_seq, max_encoder_seq_length, num_encoder_tokens)
decoder_input_data = onehot_encode(decoder_input_seq, max_decoder_seq_length, num_decoder_tokens)

decoder_target_seq = numpy.zeros(decoder_input_seq.shape)
decoder_target_seq[:, 0:-1] = decoder_input_seq[:, 1:]
decoder_target_data = onehot_encode(decoder_target_seq, 
                                    max_decoder_seq_length, 
                                    num_decoder_tokens)

print(encoder_input_data.shape)
print(decoder_input_data.shape)


# ## 3. Build the networks (for training)
# 
# - Build encoder, decoder, and connect the two modules to get "model". 
# 
# - Fit the model on the bilingual data to train the parameters in the encoder and decoder.

# ### 3.1. Encoder network
# 
# - Input:  one-hot encode of the input language
# 
# - Return: 
# 
#     -- output (all the hidden states   $h_1, \cdots , h_t$) are always discarded
#     
#     -- the final hidden state  $h_t$
#     
#     -- the final conveyor belt $c_t$

# In[14]:


from keras.layers import Input, LSTM, Bidirectional, Concatenate
from keras.models import Model
from attention import Attention

latent_dim = 256

# inputs of the encoder network
encoder_inputs = Input(shape=(None, num_encoder_tokens), 
                       name='encoder_inputs')

# set the LSTM layer
# encoder_lstm = LSTM(latent_dim, return_state=True, 
#                     dropout=0.5, name='encoder_lstm')
# _, state_h, state_c = encoder_lstm(encoder_inputs)

# set the Bi-LSTM layer
encoder_bilstm = Bidirectional(LSTM(latent_dim, return_state=True, 
                                  dropout=0.5, name='encoder_lstm'))
_, forward_h, forward_c, backward_h, backward_c = encoder_bilstm(encoder_inputs)
state_h = Concatenate()([forward_h, backward_h])
state_c = Concatenate()([forward_c, backward_c])

# set the Attention
# _ = Attention()(_)


# build the encoder network model
encoder_model = Model(inputs=encoder_inputs, 
                      outputs=[state_h, state_c],
                      name='encoder')


# Print a summary and save the encoder network structure to "./encoder.pdf"

# In[15]:


from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot, plot_model

SVG(model_to_dot(encoder_model, show_shapes=False).create(prog='dot', format='svg'))

plot_model(
    model=encoder_model, show_shapes=False,
    to_file='encoder.pdf'
)

encoder_model.summary()


# ### 3.2. Decoder network
# 
# - Inputs:  
# 
#     -- one-hot encode of the target language
#     
#     -- The initial hidden state $h_t$ 
#     
#     -- The initial conveyor belt $c_t$ 
# 
# - Return: 
# 
#     -- output (all the hidden states) $h_1, \cdots , h_t$
# 
#     -- the final hidden state  $h_t$ (discarded in the training and used in the prediction)
#     
#     -- the final conveyor belt $c_t$ (discarded in the training and used in the prediction)

# In[16]:


from keras.layers import Input, LSTM, Dense
from keras.models import Model

latent_dim = 512

# inputs of the decoder network
decoder_input_h = Input(shape=(latent_dim,), name='decoder_input_h')
decoder_input_c = Input(shape=(latent_dim,), name='decoder_input_c')
decoder_input_x = Input(shape=(None, num_decoder_tokens), name='decoder_input_x')

# set the LSTM layer
decoder_lstm = LSTM(latent_dim, return_sequences=True, 
                    return_state=True, dropout=0.5, name='decoder_lstm')
decoder_lstm_outputs, state_h, state_c = decoder_lstm(decoder_input_x, 
                                                      initial_state=[decoder_input_h, decoder_input_c])

# set the dense layer
decoder_dense = Dense(num_decoder_tokens, activation='softmax', name='decoder_dense')
decoder_outputs = decoder_dense(decoder_lstm_outputs)

# build the decoder network model
decoder_model = Model(inputs=[decoder_input_x, decoder_input_h, decoder_input_c],
                      outputs=[decoder_outputs, state_h, state_c],
                      name='decoder')


# Print a summary and save the encoder network structure to "./decoder.pdf"

# In[17]:


from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot, plot_model

SVG(model_to_dot(decoder_model, show_shapes=False).create(prog='dot', format='svg'))

plot_model(
    model=decoder_model, show_shapes=False,
    to_file='decoder.pdf'
)

decoder_model.summary()


# ### 3.3. Connect the encoder and decoder

# In[18]:


# input layers
encoder_input_x = Input(shape=(None, num_encoder_tokens), name='encoder_input_x')
decoder_input_x = Input(shape=(None, num_decoder_tokens), name='decoder_input_x')

# connect encoder to decoder
encoder_final_states = encoder_model([encoder_input_x])
decoder_lstm_output, _, _ = decoder_lstm(decoder_input_x, initial_state=encoder_final_states)
decoder_pred = decoder_dense(decoder_lstm_output)

model = Model(inputs=[encoder_input_x, decoder_input_x], 
              outputs=decoder_pred, 
              name='model_training')


# In[19]:


print(state_h)
print(decoder_input_h)


# In[20]:


from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot, plot_model

SVG(model_to_dot(model, show_shapes=False).create(prog='dot', format='svg'))

plot_model(
    model=model, show_shapes=False,
    to_file='model_training.pdf'
)

model.summary()


# ### 3.5. Fit the model on the bilingual dataset
# 
# - encoder_input_data: one-hot encode of the input language
# 
# - decoder_input_data: one-hot encode of the input language
# 
# - decoder_target_data: labels (left shift of decoder_input_data)
# 
# - tune the hyper-parameters
# 
# - stop when the validation loss stop decreasing.

# In[21]:


print('shape of encoder_input_data' + str(encoder_input_data.shape))
print('shape of decoder_input_data' + str(decoder_input_data.shape))
print('shape of decoder_target_data' + str(decoder_target_data.shape))


# In[23]:


model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

model.fit([encoder_input_data, decoder_input_data],  # training data
          decoder_target_data,                       # labels (left shift of the target sequences)
          batch_size=64, epochs=30, validation_split=0.2)

model.save('seq2seq.h5')


# ## 4. Make predictions
# 
# 
# ### 4.1. Translate English to Dutch
# 
# 1. Encoder read a sentence (source language) and output its final states, $h_t$ and $c_t$.
# 2. Take the [star] sign "\t" and the final state $h_t$ and $c_t$ as input and run the decoder.
# 3. Get the new states and predicted probability distribution.
# 4. sample a char from the predicted probability distribution
# 5. take the sampled char and the new states as input and repeat the process (stop if reach the [stop] sign "\n").

# In[24]:


# Reverse-lookup token index to decode sequences back to something readable.
reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())


# In[25]:


def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq)

    target_seq = numpy.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, target_token_index['\t']] = 1.

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # this line of code is greedy selection
        # try to use multinomial sampling instead (with temperature)
        sampled_token_index = numpy.argmax(output_tokens[0, -1, :])
        
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        target_seq = numpy.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        states_value = [h, c]

    return decoded_sentence


# In[40]:


for seq_index in range(2100, 2120):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('English:       ', input_texts[seq_index])
    print('Dutch (true): ', target_texts[seq_index][1:-1])
    print('Dutch (pred): ', decoded_sentence[0:-1])


# ### 4.2. Translate an English sentence to the target language
# 
# 1. Tokenization
# 2. One-hot encode
# 3. Translate

# In[38]:


input_sentence = ['I love you ']

def decodeToDutch(input_sentence):

    # generate new token
    tokenString = tokenizer.texts_to_sequences(input_sentence) # list of list
    tokenString_pad = pad_sequences(tokenString, maxlen=max_encoder_seq_length, padding='post') # Uniform length

    # input_x = <do one-hot encode...>
    encoder_input_data = onehot_encode(tokenString_pad, max_encoder_seq_length, num_encoder_tokens)

    # translated_sentence = <do translation...>
    decoded_sentence = decode_sequence(encoder_input_data)
    
    return decoded_sentence

# input_sequence = <do tokenization...>
# Reuse the previous code; Use all the previous training sets to set up the dict and tokens
lines = input_texts
tokenizer = Tokenizer(char_level=True, filters='')
tokenizer.fit_on_texts(lines) #1.token 2.build dict
seqs = tokenizer.texts_to_sequences(lines) # list of list
# decodeToDutch
decoded_sentence = decodeToDutch(input_sentence)

print('source sentence is: ' + input_sentence[0])
print('translated sentence is: ' + decoded_sentence)


# ## 5. Evaluate the translation using BLEU score
# 
# Reference: 
# - https://machinelearningmastery.com/calculate-bleu-score-for-text-python/
# - https://en.wikipedia.org/wiki/BLEU
# 
# 
# **Hint:** 
# 
# - Randomly partition the dataset to training, validation, and test. 
# 
# - Evaluate the BLEU score using the test set. Report the average.
# 
# - A reasonable BLEU score should be 0.1 ~ 0.5.

# In[39]:


from nltk.translate.bleu_score import sentence_bleu

# The code for Randomly partition the dataset is in 1.1.
# And test_X,test_y are the test sets calculated in 1.1

# Set the size of the data set to be used. Set a small value here to save time
numberOfTestSample = 100
# the average BLEU score
scoreAver = 0

# cal average bleu
for i in range(numberOfTestSample):
    
    text = [] # candidate, list
    text.append(test_X[i])
    candidate = decodeToDutch(text)
    candidate = candidate.replace('\n', '')
    candidate = candidate.split(' ')
    
    ref = [] # ref, list of list
    ref.append(test_y[i][1:-1].split(' '))

    score = sentence_bleu(ref, candidate, weights=(1, 0, 0, 0)) # here,using BLEU-1 by weights=(1, 0, 0, 0)
#     print(ref)
#     print(candidate)
#     print(score)
    scoreAver += score
    
scoreAver = scoreAver/(numberOfTestSample)
print('Using the test set, BLEU score(BLEU-1): %f'%(scoreAver))
    


# In[ ]:




