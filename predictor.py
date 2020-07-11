import pandas as pd
import numpy as np
import gensim
import sys
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from collections import Counter

n_gram = 2  # Number of considered words for the prediction
dimensions = 100  # Number of dimensions for the word vector model
epochs = 100  # Number of epochs for the neural network

f = open("file", "r+")
sentences = f.readlines()   # Sentences for training the prediction and creating the word vector model

words = []
model = None
#   Creating the word vector model (just once)
for sentence in sentences:
        for word in sentence:
            words.append(word)
        vocabulary = [[key for key, value in Counter(words).most_common()]]
        model = gensim.models.Word2Vec(min_count=1, workers=5, iter=10)
        model.build_vocab(vocabulary)
        model.train(sentences, total_examples=len(sentences), epochs=model.iter)
        model.save('trained_model')

#   Loading the trained model
model = gensim.models.Word2Vec.load("trained_model")

csv_data = {
    'current': [],
    'next': []
}
index = 0
print("Loading sentences...")
# Creation of the dataset for training the neural network
for line in sentences:
    index += 1
    line = 'y ' + line
    words = [elem if elem else None for elem in line.split()]
    for i in range(len(words) - n_gram):
        current = ''.join(words[i + j] + ' ' for j in range(n_gram))[:-1]
        next_word = words[i + n_gram]
        csv_data['current'].append(current)
        csv_data['next'].append(next_word)

# Loading into a Pandas datagram
print("Reading data from dictionary...")
df = pd.DataFrame(csv_data, columns=['current', 'next'])

print("Preparing data for model...")
current_inputs = []
next_inputs = []
for index, row in df.iterrows():
    sys.stdout.write("\rPreparing line %i..." % index)
    sys.stdout.flush()
    try:
        matrix = np.array([model.wv.get_vector(word) for word in row['current'].split()])
        current_vector = matrix.flatten()
        next_vector = model.wv.get_vector(row['next'])
        current_inputs.append(current_vector)
        next_inputs.append(next_vector)
    except KeyError as e:
        print(e)
        print("Omiting line %d" % index)
        continue

# Creating neural network model
nn = Sequential()
nn.add(Dense(210, input_dim=dimensions * n_gram))
nn.add(Dense(150, activation='relu'))
nn.add(Dense(dimensions, activation='relu'))
nn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

xs = np.array(current_inputs)
ys = np.array(next_inputs)
print("%d epochs..." % epochs)
# Training the neural network model
history = nn.fit(xs, ys, epochs=epochs, batch_size=10)
print(history)
model_json = nn.to_json()

# Saving neural network
with open('nn.json', "w") as json_file:
    json_file.write(model_json)

nn.save_weights('nn.h5')

# Predict function
def predict(sentence_for_prediction):
    cleaned_sentence = sentence_for_prediction.lower()
    words_in_sentence = [elem if elem else None for elem in cleaned_sentence.split()]
    if len(words_in_sentence) < n_gram:
        cleaned_sentence = 'y ' + cleaned_sentence
    words_in_sentence = [elem if elem else None for elem in cleaned_sentence.split()]
    last_words = words_in_sentence[-n_gram:]
    last_words_vector_list = []
    for word in last_words:
        last_words_vector_list.append(model.wv.get_vector(word))
    # Sentences must be a np array
    last_words_vector = np.array(last_words_vector_list).flatten()
    array_to_predict = np.array([last_words_vector])
    predicted_vector = nn.predict(array_to_predict, verbose=0)
    predicted_words = model.wv.similar_by_vector(predicted_vector[0], restrict_vocab=900)
    return predicted_words