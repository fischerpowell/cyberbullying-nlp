import json
import pickle
from keras.api._v2.keras.preprocessing.text import Tokenizer
from keras.api._v2.keras.preprocessing.sequence import pad_sequences
import keras.api._v2.keras as keras
#import tensorflow.keras as keras
#from tensorflow.keras.layers import Dropout
import numpy as np
import tensorflow as tf
from tensorflow import keras
#from tensorflow.keras import layers 
#layers import Dropout
from matplotlib import pyplot as plt

#Constants
vocab_size = 10000
embedding_dim = 16
max_length = 100
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_size = 20000

sentences = []
labels = []

# file_list = ['aggression_parsed_dataset', 'attack_parsed_dataset', 'toxicity_parsed_dataset', 
#              'twitter_parsed_dataset', 'twitter_racism_parsed_dataset', 'twitter_sexism_parsed_dataset',
#              'youtube_parsed_dataset']

file_list = ['kaggle_parsed_dataset','twitter_racism_parsed_dataset','attack_parsed_dataset','aggression_parsed_dataset']

for doc in file_list:
    #Init Datastore for sentenes and labels
    with open(f"data/{doc}.json", "r") as f:
        datastore = json.load(f)


    # type_of_bullying = []

    for item in datastore:
        if item['oh_label'].isdigit():
            sentences.append(item['Text'])
            labels.append(int(item['oh_label']))
        # type_of_bullying.append(item['cyberbullying_type'])
	

#Slice data for testing

training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]

#Format data
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)

word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

#Arrayify

training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)

#Assemble tf model

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


#

model.summary()

#Epic epochs

num_epochs = 30
#history = model.fit(training_padded, training_labels, epochs=num_epochs, validation_data=(testing_padded, testing_labels), verbose=2)
history = model.fit(training_padded, training_labels, epochs=num_epochs, validation_split=0.2, verbose=2)
#history = model.fit(training_padded, training_labels, epochs=num_epochs, validation_data=(testing_padded, testing_labels), validation_split=0.0, verbose=2)


#Analysis time

sentence = ["Suck my kiss is a good song", "shut the front door", "shut up nerd", "Stop being ridiculous", "You killed your presentation", "kill yourself"]
sequences = tokenizer.texts_to_sequences(sentence)
padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
print(model.predict(padded))

#Saving the model
model.save('cyberbullying_model.keras')

#Saving the tokenizer
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()