from keras.api._v2.keras.preprocessing.text import Tokenizer
from keras.api._v2.keras.preprocessing.sequence import pad_sequences
import keras.api._v2.keras as keras
#import tensorflow.keras as keras
#from tensorflow.keras.layers import Dropout
from tensorflow import keras
import json
\

model = keras.models.load_model('cyberbullying_model.keras')






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




tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)


sentence = ["Suck my kiss is a good song", "suck a dick", "shut the front door", "shut up nerd", "Stop being ridiculous", "stop being a bitch", "You killed your presentation", "kill yourself", "faggot"]
sequences = tokenizer.texts_to_sequences(sentence)
print(sequences)
padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
print(model.predict(padded))